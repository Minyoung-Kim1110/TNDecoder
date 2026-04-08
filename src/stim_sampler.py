import numpy as np
import stim

from typing import Dict, Tuple, Iterator
from dataclasses import dataclass


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
    if rounds < 3:
        raise ValueError("Use rounds >= 3 so there is a clean interior syndrome layer.")

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


def _dense_syndrome_arrays_from_worldlines_single(detector_bits, x_groups, z_groups):
    all_x = sorted({x for x, _ in x_groups.keys()} | {x for x, _ in z_groups.keys()})
    all_y = sorted({y for _, y in x_groups.keys()} | {y for _, y in z_groups.keys()})
    x_to_col = {x: j for j, x in enumerate(all_x)}
    y_to_row = {y: i for i, y in enumerate(all_y)}
    shape = (len(all_y), len(all_x))

    sX = np.zeros(shape, dtype=np.uint8)
    sZ = np.zeros(shape, dtype=np.uint8)
    active_X = np.zeros(shape, dtype=np.uint8)
    active_Z = np.zeros(shape, dtype=np.uint8)

    detector_bits = np.asarray(detector_bits, dtype=np.uint8)

    for (x, y), det_ids in x_groups.items():
        i = y_to_row[y]
        j = x_to_col[x]
        sX[i, j] = np.bitwise_xor.reduce(detector_bits[det_ids]) if det_ids else 0
        active_X[i, j] = 1

    for (x, y), det_ids in z_groups.items():
        i = y_to_row[y]
        j = x_to_col[x]
        sZ[i, j] = np.bitwise_xor.reduce(detector_bits[det_ids]) if det_ids else 0
        active_Z[i, j] = 1

    return sX, sZ, active_X, active_Z


def _dense_syndrome_arrays_from_worldlines_batch(detector_bits_batch, x_groups, z_groups):
    detector_bits_batch = np.asarray(detector_bits_batch, dtype=np.uint8)
    if detector_bits_batch.ndim != 2:
        raise ValueError("detector_bits_batch must have shape (shots, num_detectors).")

    all_x = sorted({x for x, _ in x_groups.keys()} | {x for x, _ in z_groups.keys()})
    all_y = sorted({y for _, y in x_groups.keys()} | {y for _, y in z_groups.keys()})
    x_to_col = {x: j for j, x in enumerate(all_x)}
    y_to_row = {y: i for i, y in enumerate(all_y)}

    shots = detector_bits_batch.shape[0]
    shape = (shots, len(all_y), len(all_x))

    sX = np.zeros(shape, dtype=np.uint8)
    sZ = np.zeros(shape, dtype=np.uint8)
    active_X = np.zeros(shape, dtype=np.uint8)
    active_Z = np.zeros(shape, dtype=np.uint8)

    for (x, y), det_ids in x_groups.items():
        i = y_to_row[y]
        j = x_to_col[x]
        if det_ids:
            sX[:, i, j] = np.bitwise_xor.reduce(detector_bits_batch[:, det_ids], axis=1)
        active_X[:, i, j] = 1

    for (x, y), det_ids in z_groups.items():
        i = y_to_row[y]
        j = x_to_col[x]
        if det_ids:
            sZ[:, i, j] = np.bitwise_xor.reduce(detector_bits_batch[:, det_ids], axis=1)
        active_Z[:, i, j] = 1

    return sX, sZ, active_X, active_Z
# def _split_check_types_from_coords(
#     detector_coords: Dict[int, Tuple[int, int, int]],
#     memory_basis: str = "x",
#     target_t: int = 1,
# ):
#     """
#     Split detectors on a chosen time slice into X-check and Z-check groups.

#     Classification rule:
#     sites appearing already at t=0 correspond to one stabilizer type; the
#     complementary sites on the chosen later slice correspond to the other type.
#     """
#     if memory_basis not in ("x", "z"):
#         raise ValueError("memory_basis must be 'x' or 'z'.")

#     first_round_xy = {
#         (x, y) for _, (x, y, t) in detector_coords.items() if t == 0
#     }
#     target_round = [
#         (det_id, x, y)
#         for det_id, (x, y, t) in detector_coords.items()
#         if t == target_t
#     ]

#     a_type = []
#     b_type = []
#     for det_id, x, y in target_round:
#         if (x, y) in first_round_xy:
#             a_type.append((det_id, x, y))
#         else:
#             b_type.append((det_id, x, y))

#     if memory_basis == "x":
#         x_checks = a_type
#         z_checks = b_type
#     else:
#         z_checks = a_type
#         x_checks = b_type

#     return x_checks, z_checks


# def _dense_syndrome_arrays_from_checks_single(detector_bits, x_checks, z_checks):
#     """
#     Build dense rectangular syndrome arrays sX, sZ and activity masks active_X, active_Z
#     for one shot.
#     """
#     all_x = sorted({x for _, x, _ in x_checks} | {x for _, x, _ in z_checks})
#     all_y = sorted({y for _, _, y in x_checks} | {y for _, _, y in z_checks})
#     x_to_col = {x: j for j, x in enumerate(all_x)}
#     y_to_row = {y: i for i, y in enumerate(all_y)}
#     shape = (len(all_y), len(all_x))

#     sX = np.zeros(shape, dtype=np.uint8)
#     sZ = np.zeros(shape, dtype=np.uint8)
#     active_X = np.zeros(shape, dtype=np.uint8)
#     active_Z = np.zeros(shape, dtype=np.uint8)

#     for det_id, x, y in x_checks:
#         i = y_to_row[y]
#         j = x_to_col[x]
#         sX[i, j] = int(detector_bits[det_id])
#         active_X[i, j] = 1

#     for det_id, x, y in z_checks:
#         i = y_to_row[y]
#         j = x_to_col[x]
#         sZ[i, j] = int(detector_bits[det_id])
#         active_Z[i, j] = 1

#     return sX, sZ, active_X, active_Z


# def _dense_syndrome_arrays_from_checks_batch(detector_bits_batch, x_checks, z_checks):
#     """
#     Batched version:
#         detector_bits_batch: (shots, num_detectors)

#     Returns:
#         sX, sZ, active_X, active_Z with shapes
#         (shots, nrow, ncol), (shots, nrow, ncol), (shots, nrow, ncol), (shots, nrow, ncol)
#     """
#     detector_bits_batch = np.asarray(detector_bits_batch, dtype=np.uint8)
#     if detector_bits_batch.ndim != 2:
#         raise ValueError("detector_bits_batch must have shape (shots, num_detectors).")

#     all_x = sorted({x for _, x, _ in x_checks} | {x for _, x, _ in z_checks})
#     all_y = sorted({y for _, _, y in x_checks} | {y for _, _, y in z_checks})
#     x_to_col = {x: j for j, x in enumerate(all_x)}
#     y_to_row = {y: i for i, y in enumerate(all_y)}

#     shots = detector_bits_batch.shape[0]
#     shape = (shots, len(all_y), len(all_x))

#     sX = np.zeros(shape, dtype=np.uint8)
#     sZ = np.zeros(shape, dtype=np.uint8)
#     active_X = np.zeros(shape, dtype=np.uint8)
#     active_Z = np.zeros(shape, dtype=np.uint8)

#     for det_id, x, y in x_checks:
#         i = y_to_row[y]
#         j = x_to_col[x]
#         sX[:, i, j] = detector_bits_batch[:, det_id]
#         active_X[:, i, j] = 1

#     for det_id, x, y in z_checks:
#         i = y_to_row[y]
#         j = x_to_col[x]
#         sZ[:, i, j] = detector_bits_batch[:, det_id]
#         active_Z[:, i, j] = 1

#     return sX, sZ, active_X, active_Z


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
    # x_checks, z_checks = _split_check_types_from_coords(
    #     detector_coords=detector_coords,
    #     memory_basis=memory_basis,
    #     target_t=target_t,
    # )
    # sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_checks_batch(
    #     detector_bits_batch=detector_bits,
    #     x_checks=x_checks,
    #     z_checks=z_checks,
    # )

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


