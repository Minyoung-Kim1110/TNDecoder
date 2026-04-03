from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import stim
from .stim_sampler import (  # type: ignore
        StimSurfaceBatchSample,
        StimSurfaceSample,
        _dense_syndrome_arrays_from_checks_batch,
        _rounded_detector_coords,
        _split_check_types_from_coords,
    )


@dataclass(frozen=True)
class FullLogicalMetadata:
    distance: int
    p: float
    rounds: int
    noisy_round: int
    target_t: int
    shots: int
    num_data_qubits_in_noisy_round: int


@dataclass
class StimSurfaceBatchSampleFullLogical:
    """
    Paired batches generated from the same sampled data-Pauli errors.

    batch_x: X-memory experiment. Its one-bit observable is the logical Z bit.
    batch_z: Z-memory experiment. Its one-bit observable is the logical X bit.

    logical_bits has shape (shots, 2) with column order (z_log, x_log), so that
        residual logical operator = Z_L**z_log X_L**x_log.
    """

    batch_x: StimSurfaceBatchSample
    batch_z: StimSurfaceBatchSample
    logical_bits: np.ndarray
    metadata: FullLogicalMetadata

    @property
    def shots(self) -> int:
        return int(self.logical_bits.shape[0])

    @property
    def logical_labels(self) -> np.ndarray:
        out = np.empty(self.shots, dtype=object)
        for k, (z_log, x_log) in enumerate(np.asarray(self.logical_bits, dtype=np.uint8)):
            if z_log == 0 and x_log == 0:
                out[k] = "I"
            elif z_log == 1 and x_log == 0:
                out[k] = "Z"
            elif z_log == 0 and x_log == 1:
                out[k] = "X"
            else:
                out[k] = "Y"
        return out

    def as_basis_batch(self, memory_basis: str) -> StimSurfaceBatchSample:
        if memory_basis == "x":
            return self.batch_x
        if memory_basis == "z":
            return self.batch_z
        raise ValueError("memory_basis must be 'x' or 'z'.")

    def get_shot(self, shot: int) -> Tuple[StimSurfaceSample, StimSurfaceSample, np.ndarray]:
        return (
            self.batch_x.get_shot(shot),
            self.batch_z.get_shot(shot),
            self.logical_bits[shot].astype(np.uint8, copy=False),
        )


@dataclass
class StimSurfaceSampleFullLogical:
    batch: StimSurfaceBatchSampleFullLogical
    shot: int = 0

    @property
    def sample_x(self) -> StimSurfaceSample:
        return self.batch.batch_x.get_shot(self.shot)

    @property
    def sample_z(self) -> StimSurfaceSample:
        return self.batch.batch_z.get_shot(self.shot)

    @property
    def logical_bits(self) -> np.ndarray:
        return self.batch.logical_bits[self.shot].astype(np.uint8, copy=False)


# ---------------------------------------------------------------------------
# Circuit helpers
# ---------------------------------------------------------------------------


def _validate_args(
    distance: int,
    p: float,
    rounds: int,
    noisy_round: int,
    target_t: int,
    shots: int,
) -> None:
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3.")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must satisfy 0 <= p <= 1.")
    if rounds < 3:
        raise ValueError("rounds must be >= 3.")
    if not (1 <= noisy_round <= rounds):
        raise ValueError("noisy_round must satisfy 1 <= noisy_round <= rounds.")
    if noisy_round in (1, rounds):
        raise ValueError("Use an interior noisy_round, typically 2 when rounds=3.")
    if not (0 <= target_t < rounds):
        raise ValueError("target_t must satisfy 0 <= target_t < rounds.")
    if target_t != noisy_round - 1:
        raise ValueError(
            "Use target_t = noisy_round - 1 so the decoded slice matches the unique noisy round."
        )
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")



def _generated_marker_circuit(distance: int, rounds: int, memory_basis: str) -> stim.Circuit:
    marker_p = 0.125
    return stim.Circuit.generated(
        f"surface_code:unrotated_memory_{memory_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=marker_p,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )



def _targets_of_noisy_round(circuit: stim.Circuit, noisy_round: int):
    occurrence = 0
    for op in circuit.flattened():
        if op.name == "DEPOLARIZE1":
            occurrence += 1
            if occurrence == noisy_round:
                return op.targets_copy()
    raise RuntimeError(
        f"Could not find noisy_round={noisy_round} DEPOLARIZE1 occurrence in flattened circuit."
    )



def _sample_pauli_pattern(num_targets: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """
    Return integer codes per data qubit:
        0 -> I, 1 -> X, 2 -> Y, 3 -> Z
    under the depolarizing channel P(I)=1-p, P(X)=P(Y)=P(Z)=p/3.
    """
    if p == 0.0:
        return np.zeros(num_targets, dtype=np.uint8)
    probs = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0], dtype=float)
    return rng.choice(4, size=num_targets, p=probs).astype(np.uint8)



def _rewrite_marker_circuit_with_explicit_paulis(
    circuit: stim.Circuit,
    *,
    noisy_round: int,
    pauli_pattern: np.ndarray,
) -> stim.Circuit:
    """
    Flatten the marker circuit, remove all DEPOLARIZE1 instructions, and on the
    selected noisy_round replace the marker with explicit deterministic X/Y/Z
    gates implementing one sampled data-Pauli error pattern.
    """
    out = stim.Circuit()
    occurrence = 0
    pauli_pattern = np.asarray(pauli_pattern, dtype=np.uint8).reshape(-1)

    for op in circuit.flattened():
        if op.name != "DEPOLARIZE1":
            out.append(op)
            continue

        occurrence += 1
        if occurrence != noisy_round:
            continue

        targets = op.targets_copy()
        if len(targets) != int(pauli_pattern.size):
            raise RuntimeError(
                "Mismatch between sampled Pauli pattern length and noisy-round data-target count: "
                f"len(targets)={len(targets)}, len(pauli_pattern)={pauli_pattern.size}."
            )

        x_targets = [t for t, code in zip(targets, pauli_pattern) if int(code) == 1]
        y_targets = [t for t, code in zip(targets, pauli_pattern) if int(code) == 2]
        z_targets = [t for t, code in zip(targets, pauli_pattern) if int(code) == 3]

        if x_targets:
            out.append("X", x_targets)
        if y_targets:
            out.append("Y", y_targets)
        if z_targets:
            out.append("Z", z_targets)

    if occurrence == 0:
        raise RuntimeError("No DEPOLARIZE1 instructions were found after flattening.")
    if occurrence < noisy_round:
        raise RuntimeError(
            f"Requested noisy_round={noisy_round}, but only found {occurrence} occurrences."
        )

    return out



def _sample_detector_and_observable_bits(circuit: stim.Circuit):
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=1, separate_observables=True)
    detector_bits = np.asarray(dets, dtype=np.uint8)
    observable_bits = np.asarray(obs, dtype=np.uint8)
    return detector_bits.reshape(1, -1), observable_bits.reshape(1, -1)



def _dense_batch_from_circuit_sample(
    circuit: stim.Circuit,
    detector_bits: np.ndarray,
    observable_flips: np.ndarray,
    *,
    memory_basis: str,
    target_t: int,
) -> StimSurfaceBatchSample:
    detector_coords: Dict[int, Tuple[int, int, int]] = _rounded_detector_coords(circuit)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_surface_code_capacity_batch_full_logical(
    distance: int,
    p: float,
    shots: int,
    *,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    seed: Optional[int] = None,
) -> StimSurfaceBatchSampleFullLogical:
    """
    Generate a true code-capacity surface-code dataset with full two-bit logical truth labels.

    For each shot, one depolarizing Pauli is sampled independently on each data qubit in the
    chosen noisy_round. The same sampled data error is then injected into both
        - surface_code:unrotated_memory_x  (to read the logical Z bit), and
        - surface_code:unrotated_memory_z  (to read the logical X bit).

    The returned logical_bits array has columns ordered as (z_log, x_log).
    """
    _validate_args(distance, p, rounds, noisy_round, target_t, shots)
    rng = np.random.default_rng(seed)

    marker_x = _generated_marker_circuit(distance=distance, rounds=rounds, memory_basis="x")
    marker_z = _generated_marker_circuit(distance=distance, rounds=rounds, memory_basis="z")

    targets_x = _targets_of_noisy_round(marker_x, noisy_round=noisy_round)
    targets_z = _targets_of_noisy_round(marker_z, noisy_round=noisy_round)
    if len(targets_x) != len(targets_z):
        raise RuntimeError(
            "The X-memory and Z-memory marker circuits expose different numbers of noisy-round "
            f"data targets: len_x={len(targets_x)}, len_z={len(targets_z)}."
        )

    batch_x_list = []
    batch_z_list = []
    logical_bits_list = []

    for _ in range(shots):
        pauli_pattern = _sample_pauli_pattern(len(targets_x), p=p, rng=rng)

        circuit_x = _rewrite_marker_circuit_with_explicit_paulis(
            marker_x,
            noisy_round=noisy_round,
            pauli_pattern=pauli_pattern,
        )
        circuit_z = _rewrite_marker_circuit_with_explicit_paulis(
            marker_z,
            noisy_round=noisy_round,
            pauli_pattern=pauli_pattern,
        )

        det_x, obs_x = _sample_detector_and_observable_bits(circuit_x)
        det_z, obs_z = _sample_detector_and_observable_bits(circuit_z)

        bx = _dense_batch_from_circuit_sample(
            circuit_x,
            det_x,
            obs_x,
            memory_basis="x",
            target_t=target_t,
        )
        bz = _dense_batch_from_circuit_sample(
            circuit_z,
            det_z,
            obs_z,
            memory_basis="z",
            target_t=target_t,
        )

        batch_x_list.append(bx)
        batch_z_list.append(bz)

        z_log = int(obs_x[0, 0])
        x_log = int(obs_z[0, 0])
        logical_bits_list.append([z_log, x_log])

    def _stack_batches(items):
        first = items[0]
        return StimSurfaceBatchSample(
            circuit=first.circuit,
            detector_bits=np.concatenate([b.detector_bits for b in items], axis=0),
            observable_flips=np.concatenate([b.observable_flips for b in items], axis=0),
            sX=np.concatenate([b.sX for b in items], axis=0),
            sZ=np.concatenate([b.sZ for b in items], axis=0),
            active_X=np.concatenate([b.active_X for b in items], axis=0),
            active_Z=np.concatenate([b.active_Z for b in items], axis=0),
            detector_coords=first.detector_coords,
        )

    return StimSurfaceBatchSampleFullLogical(
        batch_x=_stack_batches(batch_x_list),
        batch_z=_stack_batches(batch_z_list),
        logical_bits=np.asarray(logical_bits_list, dtype=np.uint8),
        metadata=FullLogicalMetadata(
            distance=distance,
            p=p,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            shots=shots,
            num_data_qubits_in_noisy_round=len(targets_x),
        ),
    )



def sample_surface_code_capacity_full_logical(
    distance: int,
    p: float,
    *,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    seed: Optional[int] = None,
) -> StimSurfaceSampleFullLogical:
    batch = sample_surface_code_capacity_batch_full_logical(
        distance=distance,
        p=p,
        shots=1,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        seed=seed,
    )
    return StimSurfaceSampleFullLogical(batch=batch, shot=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _label_counts_from_bits(bits: np.ndarray) -> Dict[str, int]:
    bits = np.asarray(bits, dtype=np.uint8)
    out = {"I": 0, "X": 0, "Z": 0, "Y": 0}
    for z_log, x_log in bits:
        if z_log == 0 and x_log == 0:
            out["I"] += 1
        elif z_log == 1 and x_log == 0:
            out["Z"] += 1
        elif z_log == 0 and x_log == 1:
            out["X"] += 1
        else:
            out["Y"] += 1
    return out



def test_full_logical_zero_noise() -> None:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=3,
        p=0.0,
        shots=4,
        rounds=3,
        noisy_round=2,
        target_t=1,
        seed=1,
    )
    assert data.logical_bits.shape == (4, 2)
    assert np.all(data.logical_bits == 0)
    assert np.all(data.batch_x.detector_bits == 0)
    assert np.all(data.batch_z.detector_bits == 0)



def test_full_logical_shapes() -> None:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=3,
        p=0.05,
        shots=3,
        rounds=3,
        noisy_round=2,
        target_t=1,
        seed=2,
    )
    assert data.batch_x.shots == 3
    assert data.batch_z.shots == 3
    assert data.logical_bits.shape == (3, 2)
    assert data.batch_x.observable_flips.shape == (3, 1)
    assert data.batch_z.observable_flips.shape == (3, 1)



def test_full_logical_consistent_one_bit_labels() -> None:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=3,
        p=0.07,
        shots=5,
        rounds=3,
        noisy_round=2,
        target_t=1,
        seed=3,
    )
    assert np.array_equal(data.logical_bits[:, 0], data.batch_x.observable_flips[:, 0])
    assert np.array_equal(data.logical_bits[:, 1], data.batch_z.observable_flips[:, 0])



def run_surface_code_capacity_full_logical_tests() -> None:
    test_full_logical_zero_noise()
    test_full_logical_shapes()
    test_full_logical_consistent_one_bit_labels()
    print("All full-logical surface-code capacity sampler tests passed.")


if __name__ == "__main__":
    run_surface_code_capacity_full_logical_tests()
