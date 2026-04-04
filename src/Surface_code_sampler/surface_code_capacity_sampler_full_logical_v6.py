from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import stim

try:
    from .stim_sampler import (
        StimSurfaceBatchSample,
        StimSurfaceSample,
        _dense_syndrome_arrays_from_checks_batch,
        _rounded_detector_coords,
        _split_check_types_from_coords,
    )
except ImportError:
    from stim_sampler import (
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
    batch_x: StimSurfaceBatchSample
    batch_z: StimSurfaceBatchSample
    logical_bits: np.ndarray
    metadata: FullLogicalMetadata

    @property
    def shots(self) -> int:
        return int(self.logical_bits.shape[0])

    @property
    def batch(self) -> StimSurfaceBatchSample:
        """
        Backward-compatible merged batch for existing v5 comparison code.

        We use:
          - sX, active_X from the x-memory circuit
          - sZ, active_Z from the z-memory circuit

        because v6 samples the same physical Pauli pattern in both circuits and
        uses x-memory / z-memory only to recover the two logical truth bits.
        """
        return StimSurfaceBatchSample(
            circuit=self.batch_x.circuit,
            detector_bits=self.batch_x.detector_bits,
            observable_flips=self.logical_bits,
            sX=self.batch_x.sX,
            sZ=self.batch_z.sZ,
            active_X=self.batch_x.active_X,
            active_Z=self.batch_z.active_Z,
            detector_coords=self.batch_x.detector_coords,
        )

    def get_shot(self, shot: int) -> Tuple[StimSurfaceSample, StimSurfaceSample, np.ndarray]:
        return (
            self.batch_x.get_shot(shot),
            self.batch_z.get_shot(shot),
            self.logical_bits[shot].astype(np.uint8, copy=False),
        )

    def iter_shots(self) -> Iterator[Tuple[StimSurfaceSample, StimSurfaceSample, np.ndarray]]:
        for k in range(self.shots):
            yield self.get_shot(k)


def _validate_args(distance: int, p: float, rounds: int, noisy_round: int, target_t: int, shots: int) -> None:
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
    if target_t != noisy_round - 1:
        raise ValueError("Use target_t = noisy_round - 1 for the single-round capacity slice.")
    if shots <= 0:
        raise ValueError("shots must be positive.")


def _generated_marker_circuit(distance: int, rounds: int, memory_basis: str) -> stim.Circuit:
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")
    return stim.Circuit.generated(
        f"surface_code:unrotated_memory_{memory_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=0.125,  # marker only; rewritten away later
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )


def _qubit_coords_dict(circuit: stim.Circuit) -> Dict[int, Tuple[float, ...]]:
    out: Dict[int, Tuple[float, ...]] = {}
    for op in circuit.flattened():
        if op.name == "QUBIT_COORDS":
            coords = tuple(float(x) for x in op.gate_args_copy())
            for t in op.targets_copy():
                out[int(t.value)] = coords
    return out


def _targets_of_noisy_round(circuit: stim.Circuit, noisy_round: int) -> List[int]:
    occurrence = 0
    for op in circuit.flattened():
        if op.name == "DEPOLARIZE1":
            occurrence += 1
            if occurrence == noisy_round:
                return [int(t.value) for t in op.targets_copy()]
    raise RuntimeError(f"Could not find noisy_round={noisy_round} DEPOLARIZE1 occurrence in flattened circuit.")


def _coords_of_noisy_round(circuit: stim.Circuit, noisy_round: int) -> List[Tuple[float, ...]]:
    qcoords = _qubit_coords_dict(circuit)
    targets = _targets_of_noisy_round(circuit, noisy_round)
    try:
        return [qcoords[q] for q in targets]
    except KeyError as ex:
        raise RuntimeError(f"Missing QUBIT_COORDS for noisy-round data qubit {ex.args[0]}.") from ex


def _canonical_noisy_qubits(
    marker_x: stim.Circuit,
    marker_z: stim.Circuit,
    noisy_round: int,
) -> Tuple[List[Tuple[float, ...]], np.ndarray, np.ndarray]:
    coords_x = _coords_of_noisy_round(marker_x, noisy_round)
    coords_z = _coords_of_noisy_round(marker_z, noisy_round)

    rx = [tuple(round(v, 8) for v in c) for c in coords_x]
    rz = [tuple(round(v, 8) for v in c) for c in coords_z]
    if set(rx) != set(rz):
        raise RuntimeError("The noisy-round data qubit coordinate sets differ between memory_x and memory_z circuits.")

    canonical = sorted(set(rx))
    map_x = {c: i for i, c in enumerate(rx)}
    map_z = {c: i for i, c in enumerate(rz)}
    perm_x = np.array([map_x[c] for c in canonical], dtype=np.int64)
    perm_z = np.array([map_z[c] for c in canonical], dtype=np.int64)
    return canonical, perm_x, perm_z


def _sample_pauli_pattern(num_targets: int, p: float, rng: np.random.Generator) -> np.ndarray:
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
                "Mismatch between sampled pattern length and noisy-round target count: "
                f"{len(pauli_pattern)} vs {len(targets)}."
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

    if occurrence == 0 or occurrence < noisy_round:
        raise RuntimeError("Failed to locate the requested noisy round.")
    return out


def _dense_batch_from_circuit_sample(
    circuit: stim.Circuit,
    detector_bits: np.ndarray,
    observable_flips: np.ndarray,
    *,
    target_t: int,
    memory_basis: str,
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


def _memory_bit_name(memory_basis: str) -> str:
    return "z_log" if memory_basis == "x" else "x_log"


def sample_surface_code_capacity_batch_full_logical(
    distance: int,
    p: float,
    shots: int,
    *,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    seed: Optional[int] = None,
    debug: bool = False,
) -> StimSurfaceBatchSampleFullLogical:
    _validate_args(distance, p, rounds, noisy_round, target_t, shots)
    rng = np.random.default_rng(seed)

    marker_x = _generated_marker_circuit(distance=distance, rounds=rounds, memory_basis="x")
    marker_z = _generated_marker_circuit(distance=distance, rounds=rounds, memory_basis="z")

    canonical_coords, perm_x, perm_z = _canonical_noisy_qubits(marker_x, marker_z, noisy_round)
    num_targets = len(canonical_coords)

    det_x_list: List[np.ndarray] = []
    obs_x_list: List[np.ndarray] = []
    det_z_list: List[np.ndarray] = []
    obs_z_list: List[np.ndarray] = []
    logical_bits_list: List[np.ndarray] = []
    patterns: List[np.ndarray] = []

    last_circuit_x: Optional[stim.Circuit] = None
    last_circuit_z: Optional[stim.Circuit] = None

    for _ in range(shots):
        pauli_pattern_canonical = _sample_pauli_pattern(num_targets, p=p, rng=rng)
        patterns.append(pauli_pattern_canonical.copy())

        pattern_x = pauli_pattern_canonical[perm_x]
        pattern_z = pauli_pattern_canonical[perm_z]

        circuit_x = _rewrite_marker_circuit_with_explicit_paulis(
            marker_x,
            noisy_round=noisy_round,
            pauli_pattern=pattern_x,
        )
        sampler_x = circuit_x.compile_detector_sampler()
        det_x, obs_x = sampler_x.sample(shots=1, separate_observables=True)

        circuit_z = _rewrite_marker_circuit_with_explicit_paulis(
            marker_z,
            noisy_round=noisy_round,
            pauli_pattern=pattern_z,
        )
        sampler_z = circuit_z.compile_detector_sampler()
        det_z, obs_z = sampler_z.sample(shots=1, separate_observables=True)

        obs_x_arr = np.asarray(obs_x, dtype=np.uint8).reshape(1, -1)
        obs_z_arr = np.asarray(obs_z, dtype=np.uint8).reshape(1, -1)
        if obs_x_arr.shape[1] != 1 or obs_z_arr.shape[1] != 1:
            raise RuntimeError(
                "Expected one observable in each memory-basis circuit, got "
                f"memory_x={obs_x_arr.shape[1]} and memory_z={obs_z_arr.shape[1]}."
            )

        # Convention used elsewhere in the repo:
        #   column 0 = z_log (from x-memory), column 1 = x_log (from z-memory)
        logical_bits = np.concatenate([obs_x_arr, obs_z_arr], axis=1).reshape(2)

        det_x_list.append(np.asarray(det_x, dtype=np.uint8).reshape(1, -1))
        obs_x_list.append(obs_x_arr)
        det_z_list.append(np.asarray(det_z, dtype=np.uint8).reshape(1, -1))
        obs_z_list.append(obs_z_arr)
        logical_bits_list.append(logical_bits.astype(np.uint8, copy=False))
        last_circuit_x = circuit_x
        last_circuit_z = circuit_z

    detector_bits_x = np.concatenate(det_x_list, axis=0)
    observable_flips_x = np.concatenate(obs_x_list, axis=0)
    detector_bits_z = np.concatenate(det_z_list, axis=0)
    observable_flips_z = np.concatenate(obs_z_list, axis=0)
    logical_bits = np.asarray(logical_bits_list, dtype=np.uint8)

    assert last_circuit_x is not None and last_circuit_z is not None
    batch_x = _dense_batch_from_circuit_sample(
        last_circuit_x,
        detector_bits_x,
        observable_flips_x,
        target_t=target_t,
        memory_basis="x",
    )
    batch_z = _dense_batch_from_circuit_sample(
        last_circuit_z,
        detector_bits_z,
        observable_flips_z,
        target_t=target_t,
        memory_basis="z",
    )

    if debug:
        print("DEBUG full-logical sampler")
        print("  num canonical noisy-round data qubits:", num_targets)
        print("  logical bit convention: [z_log_from_x_memory, x_log_from_z_memory]")
        print("  first 5 true logical bits:")
        print(logical_bits[:5])
        print("  first 5 observable flips (memory_x -> z_log):")
        print(observable_flips_x[:5])
        print("  first 5 observable flips (memory_z -> x_log):")
        print(observable_flips_z[:5])
        print("  zero-syndrome fractions:")
        zx = np.mean((batch_x.sX.sum(axis=1) + batch_x.sZ.sum(axis=1)) == 0)
        zz = np.mean((batch_z.sX.sum(axis=1) + batch_z.sZ.sum(axis=1)) == 0)
        print("    memory_x:", float(zx))
        print("    memory_z:", float(zz))

    return StimSurfaceBatchSampleFullLogical(
        batch_x=batch_x,
        batch_z=batch_z,
        logical_bits=logical_bits,
        metadata=FullLogicalMetadata(
            distance=distance,
            p=p,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            shots=shots,
            num_data_qubits_in_noisy_round=num_targets,
        ),
    )


# ------------------------------------------------------------
# Lightweight self-tests
# ------------------------------------------------------------

def _test_zero_noise() -> None:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=3,
        p=0.0,
        shots=8,
        seed=123,
    )
    assert data.logical_bits.shape == (8, 2)
    assert data.batch_x.observable_flips.shape == (8, 1)
    assert data.batch_z.observable_flips.shape == (8, 1)
    assert np.all(data.logical_bits == 0)
    assert np.all(data.batch_x.detector_bits == 0)
    assert np.all(data.batch_z.detector_bits == 0)


def _test_true_bits_match_stim_observables() -> None:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=5,
        p=0.05,
        shots=50,
        seed=7,
    )
    assert np.array_equal(data.logical_bits[:, [0]], data.batch_x.observable_flips)
    assert np.array_equal(data.logical_bits[:, [1]], data.batch_z.observable_flips)


if __name__ == "__main__":
    _test_zero_noise()
    _test_true_bits_match_stim_observables()
    print("surface_code_capacity_sampler_full_logical_v6: all tests passed.")
