import numpy as np
import stim

from typing import Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
import re


# ---------------------------------------------------------------------------
# Single-shot sample (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class StimSurfaceSample:
    circuit: stim.Circuit
    detector_bits: np.ndarray          # (num_detectors,)
    observable_flips: np.ndarray       # (num_obs,)
    sX: np.ndarray                     # (nrow, ncol)
    sZ: np.ndarray                     # (nrow, ncol)
    active_X: np.ndarray               # (nrow, ncol)
    active_Z: np.ndarray               # (nrow, ncol)
    detector_coords: Dict[int, Tuple[int, int, int]]


# ---------------------------------------------------------------------------
# Batch sample
# ---------------------------------------------------------------------------

@dataclass
class StimSurfaceBatchSample:
    circuit: stim.Circuit
    detector_bits: np.ndarray          # (shots, num_detectors)
    observable_flips: np.ndarray       # (shots, num_obs)
    sX: np.ndarray                     # (shots, nrow, ncol)
    sZ: np.ndarray                     # (shots, nrow, ncol)
    active_X: np.ndarray               # (shots, nrow, ncol)
    active_Z: np.ndarray               # (shots, nrow, ncol)
    detector_coords: Dict[int, Tuple[int, int, int]]

    @property
    def shots(self) -> int:
        return int(self.detector_bits.shape[0])

    @property
    def syndrome_shape(self) -> Tuple[int, int]:
        return tuple(self.sX.shape[1:])

    def get_shot(self, shot: int) -> StimSurfaceSample:
        return StimSurfaceSample(
            circuit=self.circuit,
            detector_bits=self.detector_bits[shot].astype(np.uint8, copy=False),
            observable_flips=self.observable_flips[shot].astype(np.uint8, copy=False),
            sX=self.sX[shot].astype(np.uint8, copy=False),
            sZ=self.sZ[shot].astype(np.uint8, copy=False),
            active_X=self.active_X[shot].astype(np.uint8, copy=False),
            active_Z=self.active_Z[shot].astype(np.uint8, copy=False),
            detector_coords=self.detector_coords,
        )

    def iter_shots(self) -> Iterator[StimSurfaceSample]:
        for k in range(self.shots):
            yield self.get_shot(k)


# ---------------------------------------------------------------------------
# Circuit factory and geometry helpers
# ---------------------------------------------------------------------------

def make_unrotated_sc_depolarizing_capacity_circuit(
    distance: int = 5,
    p: float = 1e-3,
    memory_basis: str = "x",
    rounds: int = 3,
) -> stim.Circuit:
    """
    Use Stim's built-in before_round_data_depolarization noise and keep all syndrome-extraction noise terms at zero.

    With rounds=3, detector coordinates have t=0,1,2 layers available.
    """
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")
    if rounds < 1:
        raise ValueError("rounds must be >= 1.")

    circuit = stim.Circuit.generated(
        f"surface_code:unrotated_memory_{memory_basis}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=p,
        before_measure_flip_probability=0.0,
        after_reset_flip_probability=0.0,
    )
    return circuit


def _rounded_detector_coords(circuit: stim.Circuit) -> Dict[int, Tuple[int, int, int]]:
    """
    Convert detector coordinates to integer triples (x, y, t).
    """
    raw = circuit.get_detector_coordinates()
    out = {}
    for det_id, coords in raw.items():
        if len(coords) != 3:
            raise ValueError(f"Detector {det_id} has non-3D coordinates: {coords}")
        x, y, t = (int(round(v)) for v in coords)
        out[int(det_id)] = (x, y, t)
    return out

from collections import defaultdict

def _infer_spatial_families_from_all_coords(
    detector_coords: Dict[int, Tuple[int, int, int]],
    memory_basis: str = "x",
):
    """
    Infer the two spatial stabilizer families from all detector coordinates,
    ignoring time.

    Current repo behavior used target_t slicing plus overlap with t=0.
    For noiseless repeated extraction, we instead classify spatial sites and
    then collapse each site's detector worldline over time.

    Returns:
        x_sites, z_sites as sets of spatial coordinates (x, y).
    """
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")

    spatial_sites = sorted({(x, y) for _, (x, y, _) in detector_coords.items()})

    # In Stim's unrotated surface code ALL check qubits have (x+y)%2==1,
    # so a (x+y) checkerboard cannot separate X-checks from Z-checks.
    # The correct split is by which individual coordinate is odd:
    #   x odd, y even  →  one check type
    #   x even, y odd  →  the other check type
    #
    # Verified empirically (debug_single_data_error_mapping.py):
    #   Z error fires detectors at x_odd, y_even positions
    #   → those are X-type checks for memory_x.
    xodd_yeven = set()
    xeven_yodd = set()
    for x, y in spatial_sites:
        if x % 2 == 1 and y % 2 == 0:
            xodd_yeven.add((x, y))
        elif x % 2 == 0 and y % 2 == 1:
            xeven_yodd.add((x, y))
        # positions with both-even or both-odd coords are data qubits, skip

    if memory_basis == "x":
        x_sites = xodd_yeven   # X-checks: x odd, y even
        z_sites = xeven_yodd   # Z-checks: x even, y odd
    else:
        z_sites = xodd_yeven
        x_sites = xeven_yodd

    return x_sites, z_sites


def _build_worldline_groups(
    detector_coords: Dict[int, Tuple[int, int, int]],
    x_sites,
    z_sites,
):
    """
    Group detector ids by spatial stabilizer site across all times.
    """
    x_groups = defaultdict(list)
    z_groups = defaultdict(list)

    for det_id, (x, y, t) in detector_coords.items():
        xy = (x, y)
        if xy in x_sites:
            x_groups[xy].append((t, det_id))
        elif xy in z_sites:
            z_groups[xy].append((t, det_id))

    # Sort by time for readability/debugging.
    for groups in (x_groups, z_groups):
        for xy in groups:
            groups[xy].sort()
            groups[xy] = [det_id for _, det_id in groups[xy]]

    return x_groups, z_groups


def _full_grid_axes(x_groups, z_groups):
    """
    Compute the full (2d-1)×(2d-1) coordinate axes even when only one check
    type has detectors (e.g. memory_x rounds=1 has only X-checks).

    Rule: if the maximum observed coordinate along an axis is odd, extend by 1
    to reach the next even value (which is the far boundary of the code).
    Example:  X-checks have x ∈ {1,3} (max=3, odd) → full x range [0,1,2,3,4]
              X-checks have y ∈ {0,2,4} (max=4, even) → full y range [0,1,2,3,4]
    """
    all_check_x = {x for x, _ in x_groups} | {x for x, _ in z_groups}
    all_check_y = {y for _, y in x_groups} | {y for _, y in z_groups}
    if not all_check_x or not all_check_y:
        raise ValueError("No check groups found; cannot build syndrome grid.")
    max_x = max(all_check_x)
    max_y = max(all_check_y)
    full_max_x = (max_x + 1) if (max_x % 2 == 1) else max_x
    full_max_y = (max_y + 1) if (max_y % 2 == 1) else max_y
    all_x = list(range(full_max_x + 1))
    all_y = list(range(full_max_y + 1))
    return all_x, all_y


def _route_by_physical_type(x, y):
    """
    Return which physical check type a Stim (x, y) coordinate belongs to.

    xodd_yeven → X-check (fires on Z errors via z-parity of bonds)
    xeven_yodd → Z-check (fires on X errors via x-parity of bonds)
    both-even or both-odd → data qubit (no check)
    """
    if x % 2 == 1 and y % 2 == 0:
        return "X"
    if x % 2 == 0 and y % 2 == 1:
        return "Z"
    return "data"


def _dense_syndrome_arrays_from_worldlines_single(detector_bits, x_groups, z_groups):
    all_x, all_y = _full_grid_axes(x_groups, z_groups)
    x_to_col = {x: j for j, x in enumerate(all_x)}
    y_to_row = {y: i for i, y in enumerate(all_y)}
    shape = (len(all_y), len(all_x))

    sX = np.zeros(shape, dtype=np.uint8)
    sZ = np.zeros(shape, dtype=np.uint8)
    active_X = np.zeros(shape, dtype=np.uint8)
    active_Z = np.zeros(shape, dtype=np.uint8)

    detector_bits = np.asarray(detector_bits, dtype=np.uint8)

    # Route each detector group to sX/active_X or sZ/active_Z based on the
    # *physical* check type (coordinate parity), not the group label.
    # This ensures memory_z Z-check detectors end up in sZ/active_Z, not sX/active_X.
    for groups in (x_groups, z_groups):
        for (x, y), det_ids in groups.items():
            i = y_to_row[y]
            j = x_to_col[x]
            val = np.bitwise_xor.reduce(detector_bits[det_ids]) if det_ids else 0
            ptype = _route_by_physical_type(x, y)
            if ptype == "X":
                sX[i, j] = val
                active_X[i, j] = 1
            elif ptype == "Z":
                sZ[i, j] = val
                active_Z[i, j] = 1

    return sX, sZ, active_X, active_Z


def _dense_syndrome_arrays_from_worldlines_batch(detector_bits_batch, x_groups, z_groups):
    detector_bits_batch = np.asarray(detector_bits_batch, dtype=np.uint8)
    if detector_bits_batch.ndim != 2:
        raise ValueError("detector_bits_batch must have shape (shots, num_detectors).")

    all_x, all_y = _full_grid_axes(x_groups, z_groups)
    x_to_col = {x: j for j, x in enumerate(all_x)}
    y_to_row = {y: i for i, y in enumerate(all_y)}

    shots = detector_bits_batch.shape[0]
    shape = (shots, len(all_y), len(all_x))

    sX = np.zeros(shape, dtype=np.uint8)
    sZ = np.zeros(shape, dtype=np.uint8)
    active_X = np.zeros(shape, dtype=np.uint8)
    active_Z = np.zeros(shape, dtype=np.uint8)

    # Route each detector group to sX/active_X or sZ/active_Z based on the
    # *physical* check type (coordinate parity), not the group label.
    for groups in (x_groups, z_groups):
        for (x, y), det_ids in groups.items():
            i = y_to_row[y]
            j = x_to_col[x]
            ptype = _route_by_physical_type(x, y)
            if ptype == "X":
                if det_ids:
                    sX[:, i, j] = np.bitwise_xor.reduce(
                        detector_bits_batch[:, det_ids], axis=1
                    )
                active_X[:, i, j] = 1
            elif ptype == "Z":
                if det_ids:
                    sZ[:, i, j] = np.bitwise_xor.reduce(
                        detector_bits_batch[:, det_ids], axis=1
                    )
                active_Z[:, i, j] = 1

    return sX, sZ, active_X, active_Z



# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_surface_code_depolarizing(
    distance: int,
    p: float,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
) -> StimSurfaceSample:
    """
    Backward-compatible single-shot sampler.
    """
    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=1,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )
    return batch.get_shot(0)


def sample_surface_code_depolarizing_batch(
    distance: int,
    p: float,
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
) -> StimSurfaceBatchSample:
    """
    End-to-end Stim sampler returning batched dense syndrome arrays + masks.
    """
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")

    circuit = make_unrotated_sc_depolarizing_capacity_circuit(
        distance=distance,
        p=p,
        memory_basis=memory_basis,
        rounds=rounds,
    )
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)

    detector_bits = np.asarray(dets, dtype=np.uint8)
    observable_flips = np.asarray(obs, dtype=np.uint8)

    detector_coords = _rounded_detector_coords(circuit)
    x_sites, z_sites = _infer_spatial_families_from_all_coords(
        detector_coords=detector_coords,
        memory_basis=memory_basis,
    )
    x_groups, z_groups = _build_worldline_groups(
        detector_coords=detector_coords,
        x_sites=x_sites,
        z_sites=z_sites,
    )
    sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_worldlines_batch(
        detector_bits_batch=detector_bits,
        x_groups=x_groups,
        z_groups=z_groups,
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
# Spatially inhomogeneous depolarizing-noise support
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Spin qubit — spatially inhomogeneous Z dephasing (Z_ERROR)
# ---------------------------------------------------------------------------

def _replace_depolarize1_with_z_error(
    *,
    circuit: stim.Circuit,
    pz_map: Dict[Tuple[int, int], float],
    p_fallback: float = 0.0,
    clip_eps: float = 1e-12,
) -> stim.Circuit:
    """Replace each DEPOLARIZE1(p) line with per-qubit Z_ERROR(pz_i) lines."""
    qcoords = _rounded_qubit_coords_xy(circuit)
    in_lines = str(circuit).splitlines()
    out_lines = []
    for line in in_lines:
        m = _DEPOLARIZE1_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        indent = m.group(1)
        for tok in m.group(3).split():
            if not tok.isdigit():
                continue
            q = int(tok)
            xy = qcoords.get(q)
            pz_i = float(np.clip(pz_map.get(xy, p_fallback) if xy is not None else p_fallback, 0.0, 1.0))
            if pz_i <= 0.0:
                continue
            if clip_eps > 0.0:
                pz_i = float(np.clip(pz_i, clip_eps, 1.0 - clip_eps))
            out_lines.append(f"{indent}Z_ERROR({pz_i:.12g}) {q}")
    return stim.Circuit("\n".join(out_lines) + "\n")


def make_unrotated_sc_spin_qubit_circuit(
    *,
    distance: int,
    pz_map: Dict[Tuple[int, int], float],
    memory_basis: str = "x",
    rounds: int = 3,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
) -> stim.Circuit:
    """Surface-code circuit with per-data-qubit Z dephasing (spin qubit model)."""
    reference_circuit = make_unrotated_sc_depolarizing_capacity_circuit(
        distance=distance,
        p=max(reference_p, 1e-12),
        memory_basis=memory_basis,
        rounds=rounds,
    )
    return _replace_depolarize1_with_z_error(
        circuit=reference_circuit,
        pz_map=pz_map,
        p_fallback=p_fallback,
        clip_eps=clip_eps,
    )


def generate_spin_qubit_pz_map(
    *,
    distance: int,
    p_mean: float,
    sigma_frac: float,
    memory_basis: str = "x",
    rounds: int = 3,
    seed: Optional[int] = None,
    reference_p: float = 1e-3,
) -> Dict[Tuple[int, int], float]:
    """
    Generate per-qubit Z dephasing rates for the spin qubit model.
    pz_i ~ Normal(p_mean, sigma_frac * p_mean), clipped to [0, 1].
    """
    return generate_local_p_map(
        distance=distance,
        p_mean=p_mean,
        sigma_frac=sigma_frac,
        memory_basis=memory_basis,
        rounds=rounds,
        seed=seed,
        reference_p=reference_p,
    )


def sample_surface_code_spin_qubit_batch(
    *,
    distance: int,
    pz_map: Dict[Tuple[int, int], float],
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
) -> "StimSurfaceBatchSample":
    """Spin qubit sampler: spatially varying Z dephasing only."""
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")
    circuit = make_unrotated_sc_spin_qubit_circuit(
        distance=distance,
        pz_map=pz_map,
        memory_basis=memory_basis,
        rounds=rounds,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    detector_bits = np.asarray(dets, dtype=np.uint8)
    observable_flips = np.asarray(obs, dtype=np.uint8)
    detector_coords = _rounded_detector_coords(circuit)
    x_sites, z_sites = _infer_spatial_families_from_all_coords(
        detector_coords=detector_coords, memory_basis=memory_basis,
    )
    x_groups, z_groups = _build_worldline_groups(
        detector_coords=detector_coords, x_sites=x_sites, z_sites=z_sites,
    )
    sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_worldlines_batch(
        detector_bits_batch=detector_bits, x_groups=x_groups, z_groups=z_groups,
    )
    return StimSurfaceBatchSample(
        circuit=circuit,
        detector_bits=detector_bits,
        observable_flips=observable_flips,
        sX=sX, sZ=sZ, active_X=active_X, active_Z=active_Z,
        detector_coords=detector_coords,
    )


# ---------------------------------------------------------------------------
# EO (exchange-only) qubit — spatially inhomogeneous PAULI_CHANNEL_1(px, 0, pz)
# ---------------------------------------------------------------------------

def _replace_depolarize1_with_pauli_channel(
    *,
    circuit: stim.Circuit,
    px_map: Dict[Tuple[int, int], float],
    pz_map: Dict[Tuple[int, int], float],
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    clip_eps: float = 1e-12,
) -> stim.Circuit:
    """Replace each DEPOLARIZE1(p) line with per-qubit PAULI_CHANNEL_1(px_i, 0, pz_i) lines."""
    qcoords = _rounded_qubit_coords_xy(circuit)
    in_lines = str(circuit).splitlines()
    out_lines = []
    for line in in_lines:
        m = _DEPOLARIZE1_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        indent = m.group(1)
        for tok in m.group(3).split():
            if not tok.isdigit():
                continue
            q = int(tok)
            xy = qcoords.get(q)
            px_i = float(np.clip(px_map.get(xy, p_fallback_x) if xy is not None else p_fallback_x, 0.0, 1.0))
            pz_i = float(np.clip(pz_map.get(xy, p_fallback_z) if xy is not None else p_fallback_z, 0.0, 1.0))
            total = px_i + pz_i
            if total > 1.0:
                px_i /= total
                pz_i /= total
            if px_i <= 0.0 and pz_i <= 0.0:
                continue
            if clip_eps > 0.0:
                px_i = float(np.clip(px_i, 0.0, 1.0 - clip_eps))
                pz_i = float(np.clip(pz_i, 0.0, 1.0 - clip_eps))
            out_lines.append(f"{indent}PAULI_CHANNEL_1({px_i:.12g}, 0, {pz_i:.12g}) {q}")
    return stim.Circuit("\n".join(out_lines) + "\n")


@dataclass
class EOQubitPMaps:
    """Per-qubit error rate maps for the EO qubit noise model."""
    pz_axis_map: Dict[Tuple[int, int], float]   # raw Z-axis fluctuation rates p_i^z
    pn_axis_map: Dict[Tuple[int, int], float]   # raw N-axis fluctuation rates p_i^n
    px_map: Dict[Tuple[int, int], float]         # effective px = (3/4) * p_i^n
    pz_map: Dict[Tuple[int, int], float]         # effective pz = p_i^z + (1/4) * p_i^n


def generate_eo_qubit_p_maps(
    *,
    distance: int,
    p_mean_z: float,
    sigma_frac_z: float,
    p_mean_n: float,
    sigma_frac_n: float,
    memory_basis: str = "x",
    rounds: int = 3,
    seed: Optional[int] = None,
    reference_p: float = 1e-3,
) -> EOQubitPMaps:
    """
    Generate per-qubit noise maps for the EO qubit model.

    The two controlled axes are Z and N = (sqrt(3)/2)X - (1/2)Z (120 deg from Z in XZ plane).
    Fluctuations along Z give Z errors; fluctuations along N decompose as:
        px_i = (3/4) * p_i^n
        pz_i = p_i^z + (1/4) * p_i^n

    Two independent Gaussian maps are drawn:
        p_i^z ~ Normal(p_mean_z, sigma_frac_z * p_mean_z)
        p_i^n ~ Normal(p_mean_n, sigma_frac_n * p_mean_n)
    both clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    seed_z = int(rng.integers(0, 2**31))
    seed_n = int(rng.integers(0, 2**31))

    pz_axis_map = generate_local_p_map(
        distance=distance,
        p_mean=p_mean_z,
        sigma_frac=sigma_frac_z,
        memory_basis=memory_basis,
        rounds=rounds,
        seed=seed_z,
        reference_p=reference_p,
    )
    pn_axis_map = generate_local_p_map(
        distance=distance,
        p_mean=p_mean_n,
        sigma_frac=sigma_frac_n,
        memory_basis=memory_basis,
        rounds=rounds,
        seed=seed_n,
        reference_p=reference_p,
    )
    px_map = {xy: 0.75 * pn for xy, pn in pn_axis_map.items()}
    pz_map = {xy: pz_axis_map.get(xy, 0.0) + 0.25 * pn for xy, pn in pn_axis_map.items()}

    return EOQubitPMaps(
        pz_axis_map=pz_axis_map,
        pn_axis_map=pn_axis_map,
        px_map=px_map,
        pz_map=pz_map,
    )


def make_unrotated_sc_eo_qubit_circuit(
    *,
    distance: int,
    px_map: Dict[Tuple[int, int], float],
    pz_map: Dict[Tuple[int, int], float],
    memory_basis: str = "x",
    rounds: int = 3,
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
) -> stim.Circuit:
    """Surface-code circuit with per-data-qubit PAULI_CHANNEL_1(px, 0, pz) (EO qubit model)."""
    reference_circuit = make_unrotated_sc_depolarizing_capacity_circuit(
        distance=distance,
        p=max(reference_p, 1e-12),
        memory_basis=memory_basis,
        rounds=rounds,
    )
    return _replace_depolarize1_with_pauli_channel(
        circuit=reference_circuit,
        px_map=px_map,
        pz_map=pz_map,
        p_fallback_x=p_fallback_x,
        p_fallback_z=p_fallback_z,
        clip_eps=clip_eps,
    )


def sample_surface_code_eo_qubit_batch(
    *,
    distance: int,
    px_map: Dict[Tuple[int, int], float],
    pz_map: Dict[Tuple[int, int], float],
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
) -> "StimSurfaceBatchSample":
    """EO qubit sampler: spatially varying PAULI_CHANNEL_1(px, 0, pz)."""
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")
    circuit = make_unrotated_sc_eo_qubit_circuit(
        distance=distance,
        px_map=px_map,
        pz_map=pz_map,
        memory_basis=memory_basis,
        rounds=rounds,
        p_fallback_x=p_fallback_x,
        p_fallback_z=p_fallback_z,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    detector_bits = np.asarray(dets, dtype=np.uint8)
    observable_flips = np.asarray(obs, dtype=np.uint8)
    detector_coords = _rounded_detector_coords(circuit)
    x_sites, z_sites = _infer_spatial_families_from_all_coords(
        detector_coords=detector_coords, memory_basis=memory_basis,
    )
    x_groups, z_groups = _build_worldline_groups(
        detector_coords=detector_coords, x_sites=x_sites, z_sites=z_sites,
    )
    sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_worldlines_batch(
        detector_bits_batch=detector_bits, x_groups=x_groups, z_groups=z_groups,
    )
    return StimSurfaceBatchSample(
        circuit=circuit,
        detector_bits=detector_bits,
        observable_flips=observable_flips,
        sX=sX, sZ=sZ, active_X=active_X, active_Z=active_Z,
        detector_coords=detector_coords,
    )

_DEPOLARIZE1_LINE_RE = re.compile(r"^(\s*)DEPOLARIZE1\(([^)]+)\)\s+(.*)$")


def _rounded_qubit_coords_xy(circuit: stim.Circuit) -> Dict[int, Tuple[int, int]]:
    """
    Extract integer (x, y) qubit coordinates from QUBIT_COORDS.
    """
    out: Dict[int, Tuple[int, int]] = {}
    for inst in circuit:
        if not isinstance(inst, stim.CircuitInstruction):
            continue
        if inst.name != "QUBIT_COORDS":
            continue
        args = [float(v) for v in inst.gate_args_copy()]
        if len(args) < 2:
            continue
        x = int(round(args[0]))
        y = int(round(args[1]))
        for target in inst.targets_copy():
            out[int(target.value)] = (x, y)
    return out


def _collect_depolarize1_data_qubits(circuit_text: str) -> set[int]:
    """
    Collect qubit ids targeted by DEPOLARIZE1 instructions.
    """
    data_qubits: set[int] = set()
    for line in circuit_text.splitlines():
        m = _DEPOLARIZE1_LINE_RE.match(line)
        if not m:
            continue
        target_tokens = m.group(3).split()
        for tok in target_tokens:
            if tok.isdigit():
                data_qubits.add(int(tok))
    return data_qubits


def generate_local_p_map(
    *,
    distance: int,
    p_mean: float,
    sigma_frac: float,
    memory_basis: str = "x",
    rounds: int = 3,
    seed: Optional[int] = None,
    reference_p: float = 1e-3,
) -> Dict[Tuple[int, int], float]:
    """
    Generate a spatial p-map keyed by data-qubit coordinates (x, y).

    Each data qubit gets:
        p_i ~ Normal(mean=p_mean, std=sigma_frac * p_mean), clipped to [0, 1].
    """
    if p_mean < 0.0 or p_mean > 1.0:
        raise ValueError("p_mean must satisfy 0 <= p_mean <= 1.")
    if sigma_frac < 0.0:
        raise ValueError("sigma_frac must be non-negative.")

    reference_circuit = make_unrotated_sc_depolarizing_capacity_circuit(
        distance=distance,
        p=max(reference_p, 1e-12),
        memory_basis=memory_basis,
        rounds=rounds,
    )
    qcoords = _rounded_qubit_coords_xy(reference_circuit)
    data_qubits = sorted(_collect_depolarize1_data_qubits(str(reference_circuit)))
    if not data_qubits:
        raise RuntimeError("Failed to identify data qubits from DEPOLARIZE1 targets.")

    rng = np.random.default_rng(seed)
    std = sigma_frac * p_mean
    samples = rng.normal(loc=p_mean, scale=std, size=len(data_qubits))
    samples = np.clip(samples, 0.0, 1.0)

    p_map: Dict[Tuple[int, int], float] = {}
    for q, p_i in zip(data_qubits, samples):
        if q not in qcoords:
            continue
        p_map[qcoords[q]] = float(p_i)

    if not p_map:
        raise RuntimeError("Constructed empty p_map; could not map data qubits to coordinates.")
    return p_map


def _replace_depolarize1_with_local_rates(
    *,
    circuit: stim.Circuit,
    p_map: Dict[Tuple[int, int], float],
    p_fallback: float = 0.0,
    clip_eps: float = 1e-12,
) -> stim.Circuit:
    """
    Replace each DEPOLARIZE1(p) line with per-qubit DEPOLARIZE1(p_i) lines.
    """
    if p_fallback < 0.0 or p_fallback > 1.0:
        raise ValueError("p_fallback must satisfy 0 <= p_fallback <= 1.")
    if clip_eps < 0.0 or clip_eps >= 0.5:
        raise ValueError("clip_eps must satisfy 0 <= clip_eps < 0.5.")

    qcoords = _rounded_qubit_coords_xy(circuit)
    in_lines = str(circuit).splitlines()
    out_lines = []

    for line in in_lines:
        m = _DEPOLARIZE1_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        indent = m.group(1)
        target_tokens = m.group(3).split()
        for tok in target_tokens:
            if not tok.isdigit():
                continue
            q = int(tok)
            xy = qcoords.get(q)
            p_i = p_map.get(xy, p_fallback) if xy is not None else p_fallback
            p_i = float(np.clip(p_i, 0.0, 1.0))
            if p_i <= 0.0:
                continue
            if clip_eps > 0.0:
                p_i = float(np.clip(p_i, clip_eps, 1.0 - clip_eps))
            out_lines.append(f"{indent}DEPOLARIZE1({p_i:.12g}) {q}")

    new_text = "\n".join(out_lines) + "\n"
    return stim.Circuit(new_text)


def make_unrotated_sc_local_depolarizing_capacity_circuit(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    memory_basis: str = "x",
    rounds: int = 3,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
) -> stim.Circuit:
    """
    Build an unrotated surface-code circuit with per-data-qubit depolarizing rates.
    """
    if not p_map:
        raise ValueError("p_map must be non-empty.")

    reference_circuit = make_unrotated_sc_depolarizing_capacity_circuit(
        distance=distance,
        p=max(reference_p, 1e-12),
        memory_basis=memory_basis,
        rounds=rounds,
    )
    return _replace_depolarize1_with_local_rates(
        circuit=reference_circuit,
        p_map=p_map,
        p_fallback=p_fallback,
        clip_eps=clip_eps,
    )


def sample_surface_code_local_depolarizing(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
) -> StimSurfaceSample:
    """
    Backward-compatible single-shot local-noise sampler.
    """
    batch = sample_surface_code_local_depolarizing_batch(
        distance=distance,
        p_map=p_map,
        shots=1,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )
    return batch.get_shot(0)


def sample_surface_code_local_depolarizing_batch(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
) -> StimSurfaceBatchSample:
    """
    End-to-end Stim sampler with spatially inhomogeneous depolarizing data noise.
    """
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")

    circuit = make_unrotated_sc_local_depolarizing_capacity_circuit(
        distance=distance,
        p_map=p_map,
        memory_basis=memory_basis,
        rounds=rounds,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)

    detector_bits = np.asarray(dets, dtype=np.uint8)
    observable_flips = np.asarray(obs, dtype=np.uint8)

    detector_coords = _rounded_detector_coords(circuit)
    x_sites, z_sites = _infer_spatial_families_from_all_coords(
        detector_coords=detector_coords,
        memory_basis=memory_basis,
    )
    x_groups, z_groups = _build_worldline_groups(
        detector_coords=detector_coords,
        x_sites=x_sites,
        z_sites=z_sites,
    )
    sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_worldlines_batch(
        detector_bits_batch=detector_bits,
        x_groups=x_groups,
        z_groups=z_groups,
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


