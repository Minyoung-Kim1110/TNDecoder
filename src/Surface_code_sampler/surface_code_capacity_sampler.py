# Note that this relies on depolarizing channel 

from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim

from .stim_sampler import ( 
    StimSurfaceBatchSample,
    StimSurfaceSample,
    _dense_syndrome_arrays_from_checks_batch,
    _rounded_detector_coords,
    _split_check_types_from_coords,
)


@dataclass(frozen=True)
class CapacityCircuitMetadata:
    distance: int
    p: float
    memory_basis: str
    rounds: int
    noisy_round: int
    target_t: int
    num_before_round_depolarize1_occurrences: int


def _validate_capacity_args(
    distance: int,
    p: float,
    memory_basis: str,
    rounds: int,
    noisy_round: int,
    target_t: int,
) -> None:
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must satisfy 0 <= p <= 1.")
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")
    if rounds < 3:
        raise ValueError("rounds must be >= 3 so an interior clean slice exists.")
    if not (1 <= noisy_round <= rounds):
        raise ValueError("noisy_round must satisfy 1 <= noisy_round <= rounds.")
    if not (0 <= target_t < rounds):
        raise ValueError("target_t must satisfy 0 <= target_t < rounds.")
    if noisy_round != target_t + 1:
        raise ValueError(
            "For this single-round capacity sampler, use target_t = noisy_round - 1 "
            "so the decoded detector layer matches the unique noisy data round."
        )
    if noisy_round in (1, rounds):
        raise ValueError(
            "Use an interior noisy_round (typically 2 when rounds=3). "
            "This keeps the decoded slice away from initialization/final-readout boundaries."
        )


def _rewrite_flattened_before_round_data_depolarization_occurrences(
    circuit: stim.Circuit,
    *,
    p: float,
    keep_occurrence: int,
) -> tuple[stim.Circuit, int]:
    """
    Flatten the circuit and keep only one executed DEPOLARIZE1 occurrence nonzero.

    We first build the generated Stim circuit using a nonzero marker probability,
    so that these instructions are present even when the requested final p=0.
    """
    if keep_occurrence < 1:
        raise ValueError("keep_occurrence must be >= 1.")

    flat = circuit.flattened()
    out = stim.Circuit()
    occurrence = 0

    for op in flat:
        if op.name == "DEPOLARIZE1":
            occurrence += 1
            new_p = p if occurrence == keep_occurrence else 0.0
            out.append("DEPOLARIZE1", op.targets_copy(), [new_p])
        else:
            out.append(op)

    return out, occurrence


def make_surface_code_capacity_circuit(
    distance: int,
    p: float,
    memory_basis: str = "x",
    *,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: Optional[int] = None,
    return_metadata: bool = False,
):
    """
    Build a single-noisy-round unrotated surface-code memory circuit.

    The construction is:
      1. Generate Stim's unrotated memory circuit with only before-round data
         depolarization enabled, using a nonzero marker probability.
      2. Flatten the circuit so repeated rounds become explicit.
      3. Keep only one executed before-round DEPOLARIZE1 occurrence nonzero.
      4. Decode detector layer target_t = noisy_round - 1.
    """
    if target_t is None:
        target_t = noisy_round - 1
    _validate_capacity_args(distance, p, memory_basis, rounds, noisy_round, target_t)

    # Important: use a nonzero marker so the DEPOLARIZE1 instructions exist even when the desired final physical error rate is p=0.
    marker_p = 0.125

    base_circuit = stim.Circuit.generated(
        f"surface_code:unrotated_memory_{memory_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=marker_p,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )

    circuit, num_depolarize_occurrences = _rewrite_flattened_before_round_data_depolarization_occurrences(
        base_circuit,
        p=p,
        keep_occurrence=noisy_round,
    )
    if num_depolarize_occurrences != rounds:
        raise RuntimeError(
            "Expected exactly one executed before-round DEPOLARIZE1 occurrence per round "
            f"after flattening, but found {num_depolarize_occurrences} for rounds={rounds}."
        )

    if return_metadata:
        return circuit, CapacityCircuitMetadata(
            distance=distance,
            p=p,
            memory_basis=memory_basis,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            num_before_round_depolarize1_occurrences=num_depolarize_occurrences,
        )
    return circuit


def sample_surface_code_capacity_batch(
    distance: int,
    p: float,
    shots: int,
    memory_basis: str = "x",
    *,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: Optional[int] = None,
) -> StimSurfaceBatchSample:
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if target_t is None:
        target_t = noisy_round - 1

    circuit = make_surface_code_capacity_circuit(
        distance=distance,
        p=p,
        memory_basis=memory_basis,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
    )

    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    detector_bits = np.asarray(dets, dtype=np.uint8)
    observable_flips = np.asarray(obs, dtype=np.uint8)

    detector_coords = _rounded_detector_coords(circuit)
    x_checks, z_checks = _split_check_types_from_coords(
        detector_coords=detector_coords,
        memory_basis=memory_basis,
        target_t=target_t,
    )
    sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_checks_batch(
        detector_bits_batch=detector_bits,
        x_checks=x_checks,
        z_checks=z_checks,
    )

    return StimSurfaceBatchSample(
        circuit=circuit,
        detector_bits=detector_bits,
        observable_flips=observable_flips,
        sX=sX,
        sZ=sZ,
        active_X=active_X,
        active_Z=active_Z,
        detector_coords=detector_coords,
    )


def sample_surface_code_capacity(
    distance: int,
    p: float,
    memory_basis: str = "x",
    *,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: Optional[int] = None,
) -> StimSurfaceSample:
    batch = sample_surface_code_capacity_batch(
        distance=distance,
        p=p,
        shots=1,
        memory_basis=memory_basis,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
    )
    return batch.get_shot(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _count_nonzero_depolarize1_occurrences(circuit: stim.Circuit) -> int:
    flat = circuit.flattened()
    n = 0
    for op in flat:
        if op.name == "DEPOLARIZE1":
            args = op.gate_args_copy()
            if len(args) != 1:
                raise AssertionError("Expected DEPOLARIZE1 to carry exactly one argument.")
            if float(args[0]) != 0.0:
                n += 1
    return n


def test_capacity_circuit_has_one_nonzero_depolarize_occurrence() -> None:
    circuit, meta = make_surface_code_capacity_circuit(
        distance=3,
        p=0.02,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
        return_metadata=True,
    )
    assert meta.num_before_round_depolarize1_occurrences == 3
    assert _count_nonzero_depolarize1_occurrences(circuit) == 1


def test_capacity_sampler_zero_noise() -> None:
    batch = sample_surface_code_capacity_batch(
        distance=3,
        p=0.0,
        shots=16,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    assert np.all(batch.detector_bits == 0)
    assert np.all(batch.observable_flips == 0)
    assert np.all(batch.sX == 0)
    assert np.all(batch.sZ == 0)


def test_capacity_sampler_single_matches_batch() -> None:
    one = sample_surface_code_capacity(
        distance=3,
        p=0.01,
        memory_basis="z",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    batch = sample_surface_code_capacity_batch(
        distance=3,
        p=0.01,
        shots=1,
        memory_basis="z",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    assert one.sX.shape == batch.sX[0].shape
    assert one.sZ.shape == batch.sZ[0].shape
    assert one.observable_flips.shape == batch.observable_flips[0].shape
    assert one.active_X.shape == batch.active_X[0].shape
    assert one.active_Z.shape == batch.active_Z[0].shape


def test_capacity_sampler_has_both_check_types() -> None:
    batch = sample_surface_code_capacity_batch(
        distance=5,
        p=0.0,
        shots=1,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    assert batch.sX.shape[-2] > 0 and batch.sX.shape[-1] > 0
    assert batch.sZ.shape[-2] > 0 and batch.sZ.shape[-1] > 0
    assert batch.active_X.shape == batch.sX.shape
    assert batch.active_Z.shape == batch.sZ.shape


def run_surface_code_capacity_sampler_tests() -> None:
    test_capacity_circuit_has_one_nonzero_depolarize_occurrence()
    test_capacity_sampler_zero_noise()
    test_capacity_sampler_single_matches_batch()
    test_capacity_sampler_has_both_check_types()
    print("All surface_code_capacity_sampler tests passed.")


if __name__ == "__main__":
    run_surface_code_capacity_sampler_tests()
