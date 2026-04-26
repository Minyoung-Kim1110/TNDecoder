"""
Spin qubit comparison: PEPS ML decoder vs MWPM under Z dephasing noise.

Sweeps over (distance, p_mean) and for each:
  - Draws num_maps independent Gaussian pz_maps (local noise)
  - Decodes each batch with PEPS and MWPM (both informed about local rates)
  - Also runs uniform baseline (all qubits at p_mean)
  - Saves results dict to spin_qubit_results.pkl

Results dict:
  results[d] = {
      'peps_local_mean': array,   # PEPS logical error rate, mean over maps
      'peps_local_std':  array,
      'mwpm_local_mean': array,   # MWPM logical error rate, mean over maps
      'mwpm_local_std':  array,
      'peps_uniform':    array,   # PEPS uniform baseline
      'mwpm_uniform':    array,   # MWPM uniform baseline
  }
"""

import sys
sys.path.insert(0, '.')

import pickle
import numpy as np
import multiprocessing as mp

from src.stim_PEPS_wrapper import (
    run_surface_code_peps_spin_qubit_from_normal,
    run_surface_code_peps_spin_qubit_uniform,
)
from src.mwpm_decoder import (
    run_surface_code_mwpm_spin_qubit_from_normal,
    run_surface_code_mwpm_spin_qubit_uniform,
)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

distances     = [3, 5]
p_mean_values = np.geomspace(0.01, 0.20, 12)

shots      = 500
Nkeep      = 32
Nsweep     = 1
sigma_frac = 0.30
num_maps   = 3
seed_base  = 42
rounds     = 1
num_workers = mp.cpu_count()-1
output_file = "spin_qubit_results.pkl"

print("Spin qubit (Z dephasing) comparison: PEPS vs MWPM")
print(f"  distances      = {distances}")
print(f"  p_mean_values  = {[float(f'{p:.4g}') for p in p_mean_values]}")
print(f"  shots          = {shots}")
print(f"  Nkeep          = {Nkeep}")
print(f"  sigma_frac     = {sigma_frac}")
print(f"  num_maps       = {num_maps}")
print(f"  num_workers    = {num_workers}")
print(f"  Total PEPS decodes: {len(distances) * len(p_mean_values) * num_maps * shots}")
print()

# ---------------------------------------------------------------------------
# Worker function — one (d, j, k) trial
# ---------------------------------------------------------------------------

def run_trial(args):
    d, j, p_mean, k = args
    seed = seed_base + 10_000 * d + 100 * j + k
    res_peps = run_surface_code_peps_spin_qubit_from_normal(
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
    res_mwpm = run_surface_code_mwpm_spin_qubit_from_normal(
        distance=d,
        p_mean=float(p_mean),
        sigma_frac=sigma_frac,
        shots=shots,
        rounds=rounds,
        seed=seed,
    )
    return (d, j, k, res_peps.logical_error_rate, res_mwpm.logical_error_rate)


def run_uniform(args):
    d, j, p_mean = args
    res_peps_uni = run_surface_code_peps_spin_qubit_uniform(
        distance=d, p=float(p_mean), shots=shots,
        Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
    )
    res_mwpm_uni = run_surface_code_mwpm_spin_qubit_uniform(
        distance=d, p=float(p_mean), shots=shots, rounds=rounds,
    )
    return (d, j, res_peps_uni.logical_error_rate, res_mwpm_uni.logical_error_rate)

# ---------------------------------------------------------------------------
# Build task lists
# ---------------------------------------------------------------------------

trial_tasks   = [(d, j, p_mean, k)
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

    # Collect per-trial results into arrays indexed by [d][j][k]
    peps_trials_raw = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    mwpm_trials_raw = {d: {j: {} for j in range(len(p_mean_values))} for d in distances}
    for d, j, k, peps_rate, mwpm_rate in trial_results:
        peps_trials_raw[d][j][k] = peps_rate
        mwpm_trials_raw[d][j][k] = mwpm_rate

    peps_uniform_raw = {d: {} for d in distances}
    mwpm_uniform_raw = {d: {} for d in distances}
    for d, j, peps_rate, mwpm_rate in uniform_results:
        peps_uniform_raw[d][j] = peps_rate
        mwpm_uniform_raw[d][j] = mwpm_rate

    results = {}
    for d in distances:
        peps_local_mean = []
        peps_local_std  = []
        mwpm_local_mean = []
        mwpm_local_std  = []
        peps_uniform    = []
        mwpm_uniform    = []

        for j, p_mean in enumerate(p_mean_values):
            peps_arr = np.array([peps_trials_raw[d][j][k] for k in range(num_maps)], dtype=float)
            mwpm_arr = np.array([mwpm_trials_raw[d][j][k] for k in range(num_maps)], dtype=float)

            peps_local_mean.append(peps_arr.mean())
            peps_local_std.append(peps_arr.std(ddof=0))
            mwpm_local_mean.append(mwpm_arr.mean())
            mwpm_local_std.append(mwpm_arr.std(ddof=0))
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
            'peps_local_mean': np.asarray(peps_local_mean),
            'peps_local_std':  np.asarray(peps_local_std),
            'mwpm_local_mean': np.asarray(mwpm_local_mean),
            'mwpm_local_std':  np.asarray(mwpm_local_std),
            'peps_uniform':    np.asarray(peps_uniform),
            'mwpm_uniform':    np.asarray(mwpm_uniform),
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
        'noise_model':   'spin_qubit_z_dephasing',
    }

    payload = {'meta': meta, 'results': results}

    with open(output_file, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\nSaved results to {output_file}")
