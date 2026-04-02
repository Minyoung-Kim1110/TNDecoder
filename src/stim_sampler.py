import numpy as np
import stim


from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# Surface code sampler 
@dataclass
class StimSurfaceSample:
    circuit: stim.Circuit
    detector_bits: np.ndarray          # (num_detectors,)
    observable_flips: np.ndarray       # (num_obs,)
    sX: np.ndarray                     # dense rectangular embedding
    sZ: np.ndarray
    active_X: np.ndarray               # same shape as sX/sZ
    active_Z: np.ndarray
    detector_coords: Dict[int, Tuple[int, int, int]]
    
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

def _split_check_types_from_coords(
    detector_coords: Dict[int, Tuple[int, int, int]],
    memory_basis: str = "x",
    target_t: int = 1,
):
    """
    Split detectors on a chosen time slice into X-check and Z-check groups.

    Classification rule:
      sites appearing already at t=0 correspond to one stabilizer type;
      the complementary sites on the chosen later slice correspond to the other type.
    """
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")

    first_round_xy = {(x, y) for _, (x, y, t) in detector_coords.items() if t == 0}

    target_round = [(det_id, x, y) for det_id, (x, y, t) in detector_coords.items() if t == target_t]

    a_type = []
    b_type = []
    for det_id, x, y in target_round:
        if (x, y) in first_round_xy:
            a_type.append((det_id, x, y))
        else:
            b_type.append((det_id, x, y))

    if memory_basis == "x":
        x_checks = a_type
        z_checks = b_type
    else:
        z_checks = a_type
        x_checks = b_type

    return x_checks, z_checks

def _dense_syndrome_arrays_from_checks(detector_bits, x_checks, z_checks):
    """
    Build dense rectangular syndrome arrays sX, sZ and activity masks active_X,active_Z.
    """
    all_x = sorted({x for _, x, _ in x_checks} | {x for _, x, _ in z_checks})
    all_y = sorted({y for _, _, y in x_checks} | {y for _, _, y in z_checks})

    x_to_col = {x: j for j, x in enumerate(all_x)}
    y_to_row = {y: i for i, y in enumerate(all_y)}

    shape = (len(all_y), len(all_x))
    sX = np.zeros(shape, dtype=np.uint8)
    sZ = np.zeros(shape, dtype=np.uint8)
    active_X = np.zeros(shape, dtype=np.uint8)
    active_Z = np.zeros(shape, dtype=np.uint8)

    for det_id, x, y in x_checks:
        i = y_to_row[y]
        j = x_to_col[x]
        sX[i, j] = int(detector_bits[det_id])
        active_X[i, j] = 1

    for det_id, x, y in z_checks:
        i = y_to_row[y]
        j = x_to_col[x]
        sZ[i, j] = int(detector_bits[det_id])
        active_Z[i, j] = 1

    return sX, sZ, active_X, active_Z


def sample_surface_code_depolarizing(distance, p, memory_basis: str = "x", rounds: int = 3, target_t: int = 1,
) -> StimSurfaceSample:
    """
    End-to-end Stim sampler returning dense syndrome arrays + masks.
    """
    circuit = make_unrotated_sc_depolarizing_capacity_circuit(distance=distance, p=p, memory_basis=memory_basis, rounds=rounds)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=1, separate_observables=True)
    detector_bits = dets[0].astype(np.uint8)
    observable_flips = obs[0].astype(np.uint8)

    detector_coords = _rounded_detector_coords(circuit)
    x_checks, z_checks = _split_check_types_from_coords(
        detector_coords=detector_coords,
        memory_basis=memory_basis,
        target_t=target_t,
    )

    sX, sZ, active_X, active_Z = _dense_syndrome_arrays_from_checks(
        detector_bits=detector_bits,
        x_checks=x_checks,
        z_checks=z_checks,
    )

    return StimSurfaceSample(
        circuit=circuit,
        detector_bits=detector_bits,
        observable_flips=observable_flips,
        sX=sX,
        sZ=sZ,
        active_X=active_X,
        active_Z=active_Z,
        detector_coords=detector_coords,
    )
    
    
    
    