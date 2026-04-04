from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

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


Coord = Tuple[float, ...]
PerSiteProbInput = Union[
    None,
    Sequence[float],
    np.ndarray,
    Mapping[Coord, float],
    Mapping[Coord, Sequence[float]],
    Mapping[Coord, np.ndarray],
]


@dataclass(frozen=True)
class FullLogicalMetadata:
    distance: int
    rounds: int
    noisy_round: int
    target_t: int
    shots: int
    num_data_qubits_in_noisy_round: int
    repeat_round_count: int
    injected_repeat_iteration: int
    memory_x_bit_name: str
    memory_z_bit_name: str


@dataclass
class StimSurfaceBatchSampleFullLogical:
    batch_x: StimSurfaceBatchSample
    batch_z: StimSurfaceBatchSample
    logical_bits: np.ndarray
    metadata: FullLogicalMetadata
    canonical_coords: List[Coord]
    local_pauli_probs: np.ndarray

    @property
    def shots(self) -> int:
        return int(self.logical_bits.shape[0])

    @property
    def batch(self) -> StimSurfaceBatchSample:
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


def _validate_args(
    distance: int,
    rounds: int,
    noisy_round: int,
    target_t: int,
    shots: int,
) -> None:
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3.")
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


def _generated_noiseless_circuit(distance: int, rounds: int, memory_basis: str) -> stim.Circuit:
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")
    return stim.Circuit.generated(
        f"surface_code:unrotated_memory_{memory_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=0.0,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )


def _iter_top_level_ops(circuit: stim.Circuit):
    return list(circuit)


def _append_instruction_like(out: stim.Circuit, op) -> None:
    if isinstance(op, stim.CircuitInstruction):
        out.append(op.name, op.targets_copy(), op.gate_args_copy())
    elif isinstance(op, stim.CircuitRepeatBlock):
        out += op.body_copy() * int(op.repeat_count)
    else:
        raise TypeError(f"Unsupported Stim op type: {type(op)}")


def _extract_qubit_target_values(targets: Sequence[object]) -> List[int]:
    out: List[int] = []
    for t in targets:
        # GateTarget API compatibility across Stim versions.
        is_qubit = getattr(t, "is_qubit_target", None)
        if callable(is_qubit):
            if is_qubit():
                out.append(int(t.value))
            continue
        if isinstance(is_qubit, bool):
            if is_qubit:
                out.append(int(t.value))
            continue

        # Fallback: keep only non-negative integer-valued targets.
        try:
            v = int(t.value)
        except Exception:
            continue
        if v >= 0:
            out.append(v)
    return out


def _qubit_coords_dict(circuit: stim.Circuit) -> Dict[int, Coord]:
    out: Dict[int, Coord] = {}
    for op in circuit.flattened():
        if op.name == "QUBIT_COORDS":
            coords = tuple(float(x) for x in op.gate_args_copy())
            for q in _extract_qubit_target_values(op.targets_copy()):
                out[q] = coords
    return out


def _coord_to_qubit_map(circuit: stim.Circuit) -> Dict[Coord, int]:
    out: Dict[Coord, int] = {}
    for q, c in _qubit_coords_dict(circuit).items():
        out[tuple(round(v, 8) for v in c)] = q
    return out


def _find_primary_repeat_block(circuit: stim.Circuit) -> Tuple[List[object], stim.CircuitRepeatBlock, List[object], int]:
    top = _iter_top_level_ops(circuit)
    repeat_entries = [
        (k, op)
        for k, op in enumerate(top)
        if isinstance(op, stim.CircuitRepeatBlock)
    ]
    if not repeat_entries:
        raise RuntimeError("No top-level REPEAT block found in generated Stim circuit.")

    # Use the longest repeat block as the syndrome-extraction round block.
    chosen_idx, chosen = max(repeat_entries, key=lambda kv: int(kv[1].repeat_count))
    return top[:chosen_idx], chosen, top[chosen_idx + 1 :], chosen_idx


def _body_ancilla_and_data_qubits(body: stim.Circuit) -> Tuple[List[int], List[int]]:
    ancilla: set[int] = set()
    touched: set[int] = set()

    reset_like = {
        "R", "RX", "RY", "RZ",
    }
    measure_like = {
        "M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ",
    }
    ignore = {
        "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK",
    }

    for op in body.flattened():
        if op.name in ignore:
            continue
        q_targets = _extract_qubit_target_values(op.targets_copy())
        if not q_targets:
            continue
        touched.update(q_targets)
        if op.name in reset_like or op.name in measure_like:
            ancilla.update(q_targets)

    data = sorted(touched - ancilla)
    anc = sorted(ancilla)
    if not data:
        raise RuntimeError(
            "Could not infer any data qubits from the repeat-body structure. "
            "This likely means the generated Stim circuit layout changed."
        )
    return anc, data

def _prepend_explicit_fault_layer(
    body: stim.Circuit,
    *,
    qubit_ids: Sequence[int],
    pauli_pattern: np.ndarray,
) -> stim.Circuit:
    """
    Insert a deterministic fault pattern as Stim noise instructions,
    not as ordinary Clifford gates.

    pauli_pattern codes:
      0 -> I
      1 -> X fault
      2 -> Y fault
      3 -> Z fault
    """
    pauli_pattern = np.asarray(pauli_pattern, dtype=np.uint8).reshape(-1)
    if len(qubit_ids) != int(pauli_pattern.size):
        raise ValueError(
            f"Mismatch between qubit_ids ({len(qubit_ids)}) and pauli_pattern ({pauli_pattern.size})."
        )

    x_targets = [int(q) for q, code in zip(qubit_ids, pauli_pattern) if int(code) == 1]
    y_targets = [int(q) for q, code in zip(qubit_ids, pauli_pattern) if int(code) == 2]
    z_targets = [int(q) for q, code in zip(qubit_ids, pauli_pattern) if int(code) == 3]

    out = stim.Circuit()
    if x_targets:
        out.append("X_ERROR", x_targets, [1.0])
    if y_targets:
        out.append("Y_ERROR", y_targets, [1.0])
    if z_targets:
        out.append("Z_ERROR", z_targets, [1.0])
    out += body
    return out

# def _prepend_explicit_pauli_layer(
#     body: stim.Circuit,
#     *,
#     qubit_ids: Sequence[int],
#     pauli_pattern: np.ndarray,
# ) -> stim.Circuit:
#     pauli_pattern = np.asarray(pauli_pattern, dtype=np.uint8).reshape(-1)
#     if len(qubit_ids) != int(pauli_pattern.size):
#         raise ValueError(
#             f"Mismatch between qubit_ids ({len(qubit_ids)}) and pauli_pattern ({pauli_pattern.size})."
#         )

#     x_targets = [int(q) for q, code in zip(qubit_ids, pauli_pattern) if int(code) == 1]
#     y_targets = [int(q) for q, code in zip(qubit_ids, pauli_pattern) if int(code) == 2]
#     z_targets = [int(q) for q, code in zip(qubit_ids, pauli_pattern) if int(code) == 3]

#     out = stim.Circuit()
#     if x_targets:
#         out.append("X", x_targets)
#     if y_targets:
#         out.append("Y", y_targets)
#     if z_targets:
#         out.append("Z", z_targets)
#     out += body
#     return out


def _rebuild_circuit_with_round_insertion(
    circuit: stim.Circuit,
    *,
    repeat_iteration_1based: int,
    qubit_ids: Sequence[int],
    pauli_pattern: np.ndarray,
) -> Tuple[stim.Circuit, int]:
    prefix, repeat_block, suffix, _ = _find_primary_repeat_block(circuit)
    repeat_count = int(repeat_block.repeat_count)

    if not (1 <= repeat_iteration_1based <= repeat_count):
        raise ValueError(
            f"repeat_iteration_1based must be between 1 and repeat_count={repeat_count}. "
            f"Got {repeat_iteration_1based}."
        )

    body = repeat_block.body_copy()
    body_with_error = _prepend_explicit_fault_layer(
        body,
        qubit_ids=qubit_ids,
        pauli_pattern=pauli_pattern,
    )

    out = stim.Circuit()
    for op in prefix:
        _append_instruction_like(out, op)

    before = repeat_iteration_1based - 1
    after = repeat_count - repeat_iteration_1based
    if before > 0:
        out += body * before
    out += body_with_error
    if after > 0:
        out += body * after

    for op in suffix:
        _append_instruction_like(out, op)

    return out, repeat_count


def inspect_v9_structure(
    *,
    distance: int,
    rounds: int = 3,
    memory_basis: str = "x",
) -> Dict[str, object]:
    circuit = _generated_noiseless_circuit(distance, rounds, memory_basis)
    prefix, repeat_block, suffix, repeat_index = _find_primary_repeat_block(circuit)
    anc, data = _body_ancilla_and_data_qubits(repeat_block.body_copy())
    qcoords = _qubit_coords_dict(circuit)
    return {
        "memory_basis": memory_basis,
        "repeat_block_top_level_index": repeat_index,
        "repeat_count": int(repeat_block.repeat_count),
        "prefix_len": len(prefix),
        "suffix_len": len(suffix),
        "num_ancilla_qubits_in_round_body": len(anc),
        "num_data_qubits_in_round_body": len(data),
        "first_data_coords": [qcoords[q] for q in data[: min(10, len(data))]],
        "first_ancilla_coords": [qcoords[q] for q in anc[: min(10, len(anc))]],
    }



def _canonical_round_data_coords(
    circuit_x: stim.Circuit,
    circuit_z: stim.Circuit,
) -> Tuple[List[Coord], np.ndarray, np.ndarray]:
    _, repeat_x, _, _ = _find_primary_repeat_block(circuit_x)
    _, repeat_z, _, _ = _find_primary_repeat_block(circuit_z)
    _, data_x = _body_ancilla_and_data_qubits(repeat_x.body_copy())
    _, data_z = _body_ancilla_and_data_qubits(repeat_z.body_copy())

    qcoords_x = _qubit_coords_dict(circuit_x)
    qcoords_z = _qubit_coords_dict(circuit_z)
    coords_x = [tuple(round(v, 8) for v in qcoords_x[q]) for q in data_x]
    coords_z = [tuple(round(v, 8) for v in qcoords_z[q]) for q in data_z]

    if set(coords_x) != set(coords_z):
        raise RuntimeError(
            "Round-body data-qubit coordinate sets differ between memory_x and memory_z circuits."
        )

    canonical = sorted(set(coords_x))
    map_x = {c: i for i, c in enumerate(coords_x)}
    map_z = {c: i for i, c in enumerate(coords_z)}
    perm_x = np.array([map_x[c] for c in canonical], dtype=np.int64)
    perm_z = np.array([map_z[c] for c in canonical], dtype=np.int64)
    return canonical, perm_x, perm_z


def _normalize_local_pauli_probs(
    canonical_coords: Sequence[Coord],
    p: Optional[float],
    per_site_probs: PerSiteProbInput,
) -> np.ndarray:
    n = len(canonical_coords)
    probs = np.zeros((n, 4), dtype=float)

    if per_site_probs is None:
        if p is None:
            raise ValueError("Provide either scalar p or per_site_probs.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must satisfy 0 <= p <= 1.")
        probs[:, 0] = 1.0 - float(p)
        probs[:, 1:] = float(p) / 3.0
        return probs

    if isinstance(per_site_probs, Mapping):
        for i, coord in enumerate(canonical_coords):
            if coord not in per_site_probs:
                raise KeyError(f"Missing local probability entry for coordinate {coord}.")
            val = per_site_probs[coord]
            arr = np.asarray(val, dtype=float).reshape(-1)
            if arr.size == 1:
                pi = float(arr[0])
                if not (0.0 <= pi <= 1.0):
                    raise ValueError(f"Invalid depolarizing rate {pi} at {coord}.")
                probs[i] = np.array([1.0 - pi, pi / 3.0, pi / 3.0, pi / 3.0], dtype=float)
            elif arr.size == 4:
                probs[i] = arr
            else:
                raise ValueError(
                    "Mapping values in per_site_probs must be either a scalar p_i or a 4-vector "
                    f"[pI,pX,pY,pZ]. Got shape {arr.shape} at {coord}."
                )
    else:
        arr = np.asarray(per_site_probs, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"Expected {n} per-site rates, got {arr.shape[0]}.")
            for i, pi in enumerate(arr.tolist()):
                if not (0.0 <= float(pi) <= 1.0):
                    raise ValueError(f"Invalid depolarizing rate {pi} at site index {i}.")
                probs[i] = np.array([1.0 - float(pi), float(pi) / 3.0, float(pi) / 3.0, float(pi) / 3.0])
        elif arr.ndim == 2 and arr.shape == (n, 4):
            probs = arr.copy()
        else:
            raise ValueError(
                "per_site_probs array must have shape (n_sites,) for scalar p_i or (n_sites,4) for "
                f"[pI,pX,pY,pZ]. Got {arr.shape}."
            )

    if np.any(probs < -1e-15):
        raise ValueError("Local Pauli probabilities must be nonnegative.")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise ValueError("Each local Pauli probability row must sum to 1.")
    probs = np.clip(probs, 0.0, 1.0)
    return probs


def _sample_pauli_pattern_nonuniform(local_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(local_probs.shape[0])
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        out[i] = np.uint8(rng.choice(4, p=local_probs[i]))
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


def sample_surface_code_capacity_batch_full_logical_v9(
    distance: int,
    shots: int,
    *,
    p: Optional[float] = None,
    per_site_probs: PerSiteProbInput = None,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    seed: Optional[int] = None,
    debug: bool = False,
) -> StimSurfaceBatchSampleFullLogical:
    """
    Structural full-logical sampler for the unrotated surface-code memory circuit.

    Design:
      1. Build fully noiseless Stim generated circuits.
      2. Find the primary top-level REPEAT block corresponding to repeated syndrome rounds.
      3. Infer the round-body data qubits from the repeat-body structure itself:
            data qubits = touched qubits - (reset/measured each round ancillas)
      4. Inject explicit X/Y/Z on those data qubits in the chosen repeat iteration.

    This avoids all marker-DEPOLARIZE1 inference and is naturally compatible with
    site-dependent local Pauli channels.

    Important convention:
      - memory_x observable gives z_log
      - memory_z observable gives x_log
      - logical_bits = [z_log, x_log]

    The mapping from noisy_round to repeat iteration is taken to be
        repeat_iteration = noisy_round - 1
    which matches the standard Stim memory-circuit layout where the first round is in
    the prefix and the subsequent syndrome rounds live in the primary REPEAT block.
    """
    _validate_args(distance, rounds, noisy_round, target_t, shots)
    rng = np.random.default_rng(seed)

    circuit_x = _generated_noiseless_circuit(distance, rounds, "x")
    circuit_z = _generated_noiseless_circuit(distance, rounds, "z")

    canonical_coords, perm_x, perm_z = _canonical_round_data_coords(circuit_x, circuit_z)
    coord_to_q_x = _coord_to_qubit_map(circuit_x)
    coord_to_q_z = _coord_to_qubit_map(circuit_z)
    qubits_x = [coord_to_q_x[c] for c in canonical_coords]
    qubits_z = [coord_to_q_z[c] for c in canonical_coords]
    local_probs = _normalize_local_pauli_probs(canonical_coords, p, per_site_probs)

    repeat_iteration = noisy_round - 1

    # Validate the chosen repeat iteration against the actual generated circuits.
    _, repeat_block_x, _, _ = _find_primary_repeat_block(circuit_x)
    _, repeat_block_z, _, _ = _find_primary_repeat_block(circuit_z)
    if int(repeat_block_x.repeat_count) != int(repeat_block_z.repeat_count):
        raise RuntimeError(
            "Primary REPEAT counts differ between memory_x and memory_z circuits: "
            f"{int(repeat_block_x.repeat_count)} vs {int(repeat_block_z.repeat_count)}."
        )
    repeat_count = int(repeat_block_x.repeat_count)
    if not (1 <= repeat_iteration <= repeat_count):
        raise RuntimeError(
            f"Requested noisy_round={noisy_round} implies repeat_iteration={repeat_iteration}, "
            f"but the generated circuit has repeat_count={repeat_count}."
        )

    det_x_list: List[np.ndarray] = []
    obs_x_list: List[np.ndarray] = []
    det_z_list: List[np.ndarray] = []
    obs_z_list: List[np.ndarray] = []
    logical_bits_list: List[np.ndarray] = []

    last_circuit_x: Optional[stim.Circuit] = None
    last_circuit_z: Optional[stim.Circuit] = None

    for _ in range(shots):
        pauli_pattern_canonical = _sample_pauli_pattern_nonuniform(local_probs, rng)
        pattern_x = pauli_pattern_canonical[perm_x]
        pattern_z = pauli_pattern_canonical[perm_z]

        sampled_x, _ = _rebuild_circuit_with_round_insertion(
            circuit_x,
            repeat_iteration_1based=repeat_iteration,
            qubit_ids=qubits_x,
            pauli_pattern=pattern_x,
        )
        sampled_z, _ = _rebuild_circuit_with_round_insertion(
            circuit_z,
            repeat_iteration_1based=repeat_iteration,
            qubit_ids=qubits_z,
            pauli_pattern=pattern_z,
        )

        sampler_x = sampled_x.compile_detector_sampler()
        det_x, obs_x = sampler_x.sample(shots=1, separate_observables=True)
        sampler_z = sampled_z.compile_detector_sampler()
        det_z, obs_z = sampler_z.sample(shots=1, separate_observables=True)

        obs_x_arr = np.asarray(obs_x, dtype=np.uint8).reshape(1, -1)
        obs_z_arr = np.asarray(obs_z, dtype=np.uint8).reshape(1, -1)
        if obs_x_arr.shape[1] != 1 or obs_z_arr.shape[1] != 1:
            raise RuntimeError(
                "Expected one observable in each memory-basis circuit, got "
                f"memory_x={obs_x_arr.shape[1]} and memory_z={obs_z_arr.shape[1]}."
            )

        logical_bits = np.concatenate([obs_x_arr, obs_z_arr], axis=1).reshape(2)
        det_x_list.append(np.asarray(det_x, dtype=np.uint8).reshape(1, -1))
        obs_x_list.append(obs_x_arr)
        det_z_list.append(np.asarray(det_z, dtype=np.uint8).reshape(1, -1))
        obs_z_list.append(obs_z_arr)
        logical_bits_list.append(logical_bits.astype(np.uint8, copy=False))
        last_circuit_x = sampled_x
        last_circuit_z = sampled_z

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
        print("DEBUG full-logical sampler v9 (structural)")
        print(" repeat_count:", repeat_count)
        print(" injected repeat iteration:", repeat_iteration)
        print(" num inferred round-body data qubits:", len(canonical_coords))
        print(" logical bit convention: [z_log_from_x_memory, x_log_from_z_memory]")
        print(" first 5 true logical bits:")
        print(logical_bits[:5])
        print(" first 5 observable flips (memory_x -> z_log):")
        print(observable_flips_x[:5])
        print(" first 5 observable flips (memory_z -> x_log):")
        print(observable_flips_z[:5])
        print(" average local depolarizing rate:", float(1.0 - local_probs[:, 0].mean()))
        zx = np.mean((batch_x.sX.sum(axis=1) + batch_x.sZ.sum(axis=1)) == 0)
        zz = np.mean((batch_z.sX.sum(axis=1) + batch_z.sZ.sum(axis=1)) == 0)
        print(" zero-syndrome fractions:")
        print(" memory_x:", float(zx))
        print(" memory_z:", float(zz))

    return StimSurfaceBatchSampleFullLogical(
        batch_x=batch_x,
        batch_z=batch_z,
        logical_bits=logical_bits,
        metadata=FullLogicalMetadata(
            distance=distance,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            shots=shots,
            num_data_qubits_in_noisy_round=len(canonical_coords),
            repeat_round_count=repeat_count,
            injected_repeat_iteration=repeat_iteration,
            memory_x_bit_name="z_log",
            memory_z_bit_name="x_log",
        ),
        canonical_coords=list(canonical_coords),
        local_pauli_probs=local_probs,
    )

# Backward-compatible alias.
sample_surface_code_capacity_batch_full_logical = sample_surface_code_capacity_batch_full_logical_v9


# ----------------------------------------------------------------------
# Lightweight tests that do not require running a Stim simulation.
# ----------------------------------------------------------------------
def _test_prob_normalization_uniform() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0)]
    probs = _normalize_local_pauli_probs(coords, 0.12, None)
    assert probs.shape == (2, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[:, 1], 0.04)


def _test_prob_normalization_nonuniform() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0)]
    probs = _normalize_local_pauli_probs(coords, None, np.array([0.1, 0.2]))
    assert probs.shape == (2, 4)
    assert np.allclose(probs[0], [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3])


def _test_qubit_target_extraction_fallback() -> None:
    class DummyTarget:
        def __init__(self, value: int):
            self.value = value

    vals = _extract_qubit_target_values([DummyTarget(1), DummyTarget(-1), DummyTarget(7)])
    assert vals == [1, 7]


if __name__ == "__main__":
    _test_prob_normalization_uniform()
    _test_prob_normalization_nonuniform()
    _test_qubit_target_extraction_fallback()
    print("surface_code_capacity_sampler_full_logical_v9: lightweight tests passed.")
