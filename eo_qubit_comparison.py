"""
EO (exchange-only) qubit comparison: PEPS ML decoder vs MWPM.

Noise model: quasistatic local PAULI_CHANNEL_1(px_i, 0, pz_i) where
  px_i = (3/4) * p_i^n,   pz_i = p_i^z + (1/4) * p_i^n
and p_i^z, p_i^n are drawn iid from Normal(p_mean_z/n, sigma_frac * p_mean_z/n).

The sweep uses p_mean_z = p_mean_n = p_mean (equal axis strengths).

Results dict:
  results[d] = {
      'peps_local_mean':    array,   # PEPS full logical p_L, mean over maps
      'peps_local_std':     array,
      'mwpm_local_mean':    array,   # MWPM full logical p_L, mean over maps
      'mwpm_local_std':     array,
      'peps_uniform':       array,   # PEPS uniform baseline
      'mwpm_uniform':       array,   # MWPM uniform baseline
      'peps_local_lx_mean': array,   # X logical (memory_z)
      'peps_local_lz_mean': array,   # Z logical (memory_x)
      'mwpm_local_lx_mean': array,
      'mwpm_local_lz_mean': array,
  }
"""

import sys
sys.path.insert(0, '.')

import pickle
import numpy as np
import multiprocessing as mp

from src.stim_PEPS_wrapper import (
    run_surface_code_peps_eo_qubit_from_normal,
    run_surface_code_peps_eo_qubit_uniform,
)
from src.mwpm_decoder import (
    run_surface_code_mwpm_eo_qubit_from_normal,
    run_surface_code_mwpm_eo_qubit_uniform,
)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

distances     = [3, 5]
p_mean_values = np.geomspace(0.01, 0.20, 10)

shots      = 500
Nkeep      = 32
Nsweep     = 1
sigma_frac = 0.30   # same for both Z and N axes
num_maps   = 3
seed_base  = 42
rounds     = 1
num_workers = mp.cpu_count() - 1
output_file = "eo_qubit_results.pkl"

# Equal axis strengths: p_mean_z = p_mean_n = p_mean
# Effective rates: px_mean = (3/4)*p_mean, pz_mean = (5/4)*p_mean

print("EO qubit comparison: PEPS vs MWPM")
print(f"  distances      = {distances}")
print(f"  p_mean_values  = {[float(f'{p:.4g}') for p in p_mean_values]}")
print(f"  shots          = {shots}")
print(f"  Nkeep          = {Nkeep}")
print(f"  sigma_frac     = {sigma_frac}  (both Z and N axes)")
print(f"  num_maps       = {num_maps}")
print(f"  num_workers    = {num_workers}")
print(f"  axis model     = p_mean_z = p_mean_n = p_mean")
print(f"  Total PEPS decodes: {len(distances) * len(p_mean_values) * num_maps * shots * 2}")
print()

# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def run_trial(args):
    d, j, p_mean, k = args
    p_mean_z = float(p_mean)
    p_mean_n = float(p_mean)
    seed = seed_base + 10_000 * d + 100 * j + k

    res_peps = run_surface_code_peps_eo_qubit_from_normal(
        distance=d,
        p_mean_z=p_mean_z, sigma_frac_z=sigma_frac,
        p_mean_n=p_mean_n, sigma_frac_n=sigma_frac,
        shots=shots, Nkeep=Nkeep, Nsweep=Nsweep,
        rounds=rounds, seed=seed, verbose=False,
    )
    res_mwpm = run_surface_code_mwpm_eo_qubit_from_normal(
        distance=d,
        p_mean_z=p_mean_z, sigma_frac_z=sigma_frac,
        p_mean_n=p_mean_n, sigma_frac_n=sigma_frac,
        shots=shots, rounds=rounds, seed=seed,
    )
    return (
        d, j, k,
        res_peps.logical_error_rate, res_mwpm.logical_error_rate,
        res_peps.p_L_X, res_peps.p_L_Z,
        res_mwpm.p_L_X, res_mwpm.p_L_Z,
    )


def run_uniform(args):
    d, j, p_mean = args
    p_mean_z = float(p_mean)
    p_mean_n = float(p_mean)
    res_peps_uni = run_surface_code_peps_eo_qubit_uniform(
        distance=d, p_mean_z=p_mean_z, p_mean_n=p_mean_n,
        shots=shots, Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
    )
    res_mwpm_uni = run_surface_code_mwpm_eo_qubit_uniform(
        distance=d, p_mean_z=p_mean_z, p_mean_n=p_mean_n,
        shots=shots, rounds=rounds,
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
    mwpm_full_raw = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    peps_lx_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    peps_lz_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    mwpm_lx_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    mwpm_lz_raw   = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}

    for d, j, k, peps_full, mwpm_full, peps_lx, peps_lz, mwpm_lx, mwpm_lz in trial_results:
        peps_full_raw[d][j][k] = peps_full
        mwpm_full_raw[d][j][k] = mwpm_full
        peps_lx_raw[d][j][k]   = peps_lx
        peps_lz_raw[d][j][k]   = peps_lz
        mwpm_lx_raw[d][j][k]   = mwpm_lx
        mwpm_lz_raw[d][j][k]   = mwpm_lz

    peps_uniform_raw = {d: {} for d in distances}
    mwpm_uniform_raw = {d: {} for d in distances}
    for d, j, peps_rate, mwpm_rate in uniform_results:
        peps_uniform_raw[d][j] = peps_rate
        mwpm_uniform_raw[d][j] = mwpm_rate

    results = {}
    for d in distances:
        peps_local_mean    = []
        peps_local_std     = []
        mwpm_local_mean    = []
        mwpm_local_std     = []
        peps_uniform       = []
        mwpm_uniform       = []
        peps_local_lx_mean = []
        peps_local_lz_mean = []
        mwpm_local_lx_mean = []
        mwpm_local_lz_mean = []

        for j, p_mean in enumerate(p_mean_values):
            peps_arr = np.array([peps_full_raw[d][j][k] for k in range(num_maps)], dtype=float)
            mwpm_arr = np.array([mwpm_full_raw[d][j][k] for k in range(num_maps)], dtype=float)

            peps_local_mean.append(peps_arr.mean())
            peps_local_std.append(peps_arr.std(ddof=0))
            mwpm_local_mean.append(mwpm_arr.mean())
            mwpm_local_std.append(mwpm_arr.std(ddof=0))
            peps_local_lx_mean.append(np.mean([peps_lx_raw[d][j][k] for k in range(num_maps)]))
            peps_local_lz_mean.append(np.mean([peps_lz_raw[d][j][k] for k in range(num_maps)]))
            mwpm_local_lx_mean.append(np.mean([mwpm_lx_raw[d][j][k] for k in range(num_maps)]))
            mwpm_local_lz_mean.append(np.mean([mwpm_lz_raw[d][j][k] for k in range(num_maps)]))
            peps_uniform.append(peps_uniform_raw[d][j])
            mwpm_uniform.append(mwpm_uniform_raw[d][j])

            print(
                f"  d={d}  p_mean={p_mean:.4f}  "
                f"peps_local={peps_local_mean[-1]:.4f}+-{peps_local_std[-1]:.4f}  "
                f"mwpm_local={mwpm_local_mean[-1]:.4f}+-{mwpm_local_std[-1]:.4f}  "
                f"peps_uni={peps_uniform[-1]:.4f}  "
                f"mwpm_uni={mwpm_uniform[-1]:.4f}"
            )

        results[d] = {
            'peps_local_mean':    np.asarray(peps_local_mean),
            'peps_local_std':     np.asarray(peps_local_std),
            'mwpm_local_mean':    np.asarray(mwpm_local_mean),
            'mwpm_local_std':     np.asarray(mwpm_local_std),
            'peps_uniform':       np.asarray(peps_uniform),
            'mwpm_uniform':       np.asarray(mwpm_uniform),
            'peps_local_lx_mean': np.asarray(peps_local_lx_mean),
            'peps_local_lz_mean': np.asarray(peps_local_lz_mean),
            'mwpm_local_lx_mean': np.asarray(mwpm_local_lx_mean),
            'mwpm_local_lz_mean': np.asarray(mwpm_local_lz_mean),
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
        'noise_model':   'eo_qubit_pauli_channel',
        'axis_model':    'p_mean_z = p_mean_n = p_mean',
    }

    with open(output_file, 'wb') as f:
        pickle.dump({'meta': meta, 'results': results}, f)

    print(f"\nSaved results to {output_file}")
