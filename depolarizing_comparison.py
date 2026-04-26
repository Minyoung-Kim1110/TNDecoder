"""
Depolarizing local noise comparison: PEPS ML decoder vs MWPM.

Sweeps over (distance, p_mean) and for each:
  - Draws num_maps independent Gaussian p_maps (local depolarizing rates)
  - Decodes each map with PEPS and MWPM (both informed about local rates)
  - Also runs uniform baseline (both decoders at uniform p = p_mean)
  - Saves combined results dict to depolarizing_results.pkl

Results dict:
  results[d] = {
      'local_full_mean':    array,   # PEPS full logical p_L, mean over maps
      'local_full_std':     array,
      'local_x_mean':       array,   # X-logical channel mean over maps
      'local_z_mean':       array,   # Z-logical channel mean over maps
      'uniform_full':       array,   # PEPS uniform baseline
      'mwpm_local_full_mean': array, # MWPM full logical p_L, mean over maps
      'mwpm_local_full_std':  array,
      'mwpm_local_x_mean':    array,
      'mwpm_local_z_mean':    array,
      'mwpm_uniform_full':    array,
  }
"""

import sys
sys.path.insert(0, '.')

import pickle
import numpy as np
import multiprocessing as mp

from src.stim_PEPS_wrapper import (
    run_surface_code_peps_full_logical_local_from_normal,
    run_surface_code_peps_full_logical,
)
from src.mwpm_decoder import (
    run_surface_code_mwpm_full_logical_local_from_normal,
    run_surface_code_mwpm_full_logical,
)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

distances     = [3, 5, 7]
p_mean_values = np.geomspace(0.01, 0.20, 12)

shots      = 500
Nkeep      = 32
Nsweep     = 1
sigma_frac = 0.30
num_maps   = 3
seed_base  = 42
rounds     = 1
num_workers = mp.cpu_count() - 1
output_file = "depolarizing_results.pkl"

print("Depolarizing local noise comparison: PEPS ML vs MWPM")
print(f"  distances      = {distances}")
print(f"  p_mean_values  = {[float(f'{p:.4g}') for p in p_mean_values]}")
print(f"  shots          = {shots}")
print(f"  Nkeep          = {Nkeep}")
print(f"  sigma_frac     = {sigma_frac}")
print(f"  num_maps       = {num_maps}")
print(f"  num_workers    = {num_workers}")
print(f"  Total PEPS decodes: {len(distances) * len(p_mean_values) * num_maps * shots * 2}")
print()

# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def run_trial(args):
    d, j, p_mean, k = args
    seed = seed_base + 10_000 * d + 100 * j + k
    res_peps = run_surface_code_peps_full_logical_local_from_normal(
        distance=d,
        p_mean=float(p_mean),
        sigma_frac=sigma_frac,
        shots=shots,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
        rounds=rounds,
        seed=seed,
        verbose=False,
    )
    res_mwpm = run_surface_code_mwpm_full_logical_local_from_normal(
        distance=d,
        p_mean=float(p_mean),
        sigma_frac=sigma_frac,
        shots=shots,
        rounds=rounds,
        seed=seed,
    )
    return (
        d, j, k,
        res_peps.logical_error_rate, res_peps.p_L_X, res_peps.p_L_Z,
        res_mwpm.logical_error_rate, res_mwpm.p_L_X, res_mwpm.p_L_Z,
    )


def run_uniform(args):
    d, j, p_mean = args
    res_peps_uni = run_surface_code_peps_full_logical(
        distance=d, p=float(p_mean), shots=shots,
        Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
    )
    res_mwpm_uni = run_surface_code_mwpm_full_logical(
        distance=d, p=float(p_mean), shots=shots, rounds=rounds,
    )
    return (d, j, res_peps_uni.logical_error_rate, res_mwpm_uni.logical_error_rate)

# ---------------------------------------------------------------------------
# Build task lists
# ---------------------------------------------------------------------------

trial_tasks = [(d, j, p_mean, k)
               for d in distances
               for j, p_mean in enumerate(p_mean_values)
               for k in range(num_maps)]

uniform_tasks = [(d, j, p_mean)
                 for d in distances
                 for j, p_mean in enumerate(p_mean_values)]

# ---------------------------------------------------------------------------
# Run in parallel
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    with mp.Pool(num_workers) as pool:
        print("Running local-noise trials...")
        trial_results   = pool.map(run_trial,   trial_tasks)
        print("Running uniform-baseline trials...")
        uniform_results = pool.map(run_uniform, uniform_tasks)

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------

    peps_full_raw = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    peps_lx_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    peps_lz_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    mwpm_full_raw = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    mwpm_lx_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    mwpm_lz_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}

    for d, j, k, peps_full, peps_lx, peps_lz, mwpm_full, mwpm_lx, mwpm_lz in trial_results:
        peps_full_raw[d][j][k] = peps_full
        peps_lx_raw[d][j][k]   = peps_lx
        peps_lz_raw[d][j][k]   = peps_lz
        mwpm_full_raw[d][j][k] = mwpm_full
        mwpm_lx_raw[d][j][k]   = mwpm_lx
        mwpm_lz_raw[d][j][k]   = mwpm_lz

    peps_uniform_raw = {d: {} for d in distances}
    mwpm_uniform_raw = {d: {} for d in distances}
    for d, j, peps_rate, mwpm_rate in uniform_results:
        peps_uniform_raw[d][j] = peps_rate
        mwpm_uniform_raw[d][j] = mwpm_rate

    results = {}
    for d in distances:
        local_full_mean    = []
        local_full_std     = []
        local_x_mean       = []
        local_z_mean       = []
        uniform_full       = []
        mwpm_local_full_mean = []
        mwpm_local_full_std  = []
        mwpm_local_x_mean    = []
        mwpm_local_z_mean    = []
        mwpm_uniform_full    = []

        for j, p_mean in enumerate(p_mean_values):
            peps_arr = np.array([peps_full_raw[d][j][k] for k in range(num_maps)], dtype=float)
            mwpm_arr = np.array([mwpm_full_raw[d][j][k] for k in range(num_maps)], dtype=float)

            local_full_mean.append(peps_arr.mean())
            local_full_std.append(peps_arr.std(ddof=0))
            local_x_mean.append(np.mean([peps_lx_raw[d][j][k] for k in range(num_maps)]))
            local_z_mean.append(np.mean([peps_lz_raw[d][j][k] for k in range(num_maps)]))
            uniform_full.append(peps_uniform_raw[d][j])

            mwpm_local_full_mean.append(mwpm_arr.mean())
            mwpm_local_full_std.append(mwpm_arr.std(ddof=0))
            mwpm_local_x_mean.append(np.mean([mwpm_lx_raw[d][j][k] for k in range(num_maps)]))
            mwpm_local_z_mean.append(np.mean([mwpm_lz_raw[d][j][k] for k in range(num_maps)]))
            mwpm_uniform_full.append(mwpm_uniform_raw[d][j])

            print(
                f"  d={d}  p_mean={p_mean:.4f}  "
                f"peps_local={local_full_mean[-1]:.4f}+-{local_full_std[-1]:.4f}  "
                f"mwpm_local={mwpm_local_full_mean[-1]:.4f}+-{mwpm_local_full_std[-1]:.4f}  "
                f"peps_uni={uniform_full[-1]:.4f}  "
                f"mwpm_uni={mwpm_uniform_full[-1]:.4f}"
            )

        results[d] = {
            'local_full_mean':      np.asarray(local_full_mean),
            'local_full_std':       np.asarray(local_full_std),
            'local_x_mean':         np.asarray(local_x_mean),
            'local_z_mean':         np.asarray(local_z_mean),
            'uniform_full':         np.asarray(uniform_full),
            'mwpm_local_full_mean': np.asarray(mwpm_local_full_mean),
            'mwpm_local_full_std':  np.asarray(mwpm_local_full_std),
            'mwpm_local_x_mean':    np.asarray(mwpm_local_x_mean),
            'mwpm_local_z_mean':    np.asarray(mwpm_local_z_mean),
            'mwpm_uniform_full':    np.asarray(mwpm_uniform_full),
        }

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------

    meta = {
        'distances':     distances,
        'p_mean_values': p_mean_values,
        'shots':         shots,
        'Nkeep':         Nkeep,
        'sigma_frac':    sigma_frac,
        'num_maps':      num_maps,
        'seed_base':     seed_base,
        'rounds':        rounds,
        'noise_model':   'depolarizing_local',
    }

    with open(output_file, 'wb') as f:
        pickle.dump({'meta': meta, 'results': results}, f)

    print(f"\nSaved results to {output_file}")
