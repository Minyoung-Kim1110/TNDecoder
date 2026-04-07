# debug_worldline_collapse_check.py

import numpy as np
from src.stim_sampler import (
    sample_surface_code_depolarizing_batch,
    _rounded_detector_coords,
    make_unrotated_sc_depolarizing_capacity_circuit,
    _infer_spatial_families_from_all_coords,
    _build_worldline_groups,
    _dense_syndrome_arrays_from_worldlines_batch,
)

def debug_worldline_collapse(
    distance=5,
    p=0.01,
    shots=20,
    memory_basis="x",
    rounds=3,
):
    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=1,   # ignored logically after replacement, fine to keep for API compatibility
    )

    print(f"syndrome shape = {batch.syndrome_shape}")
    print(f"mean obs flip   = {float(np.mean(batch.observable_flips[:,0])):.6f}")
    print(f"mean sX weight  = {float(np.mean(np.sum(batch.sX, axis=(1,2)))):.6f}")
    print(f"mean sZ weight  = {float(np.mean(np.sum(batch.sZ, axis=(1,2)))):.6f}")

    # show nontrivial shots
    obs = batch.observable_flips[:,0].astype(np.uint8)
    idx = np.flatnonzero(obs)
    print("nontrivial shots:", idx.tolist())

    for k in idx[:5]:
        shot = batch.get_shot(int(k))
        print("-" * 80)
        print(f"shot {k}")
        print(f"  obs = {int(obs[k])}")
        print(f"  detector weight = {int(np.sum(shot.detector_bits))}")
        print(f"  sX weight = {int(np.sum(shot.sX))}")
        print(f"  sZ weight = {int(np.sum(shot.sZ))}")
        print(f"  sX nz = {np.argwhere(shot.sX != 0).tolist()}")
        print(f"  sZ nz = {np.argwhere(shot.sZ != 0).tolist()}")