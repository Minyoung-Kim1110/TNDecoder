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
p_mean_values = np.geomspace(0.01, 0.20, 10)

shots      = 500
Nkeep      = 32
Nsweep     = 1
sigma_frac = 0.30
num_maps   = 3
seed_base  = 42
rounds     = 1
output_file = "spin_qubit_results.pkl"

print("Spin qubit (Z dephasing) comparison: PEPS vs MWPM")
print(f"  distances      = {distances}")
print(f"  p_mean_values  = {[float(f'{p:.4g}') for p in p_mean_values]}")
print(f"  shots          = {shots}")
print(f"  Nkeep          = {Nkeep}")
print(f"  sigma_frac     = {sigma_frac}")
print(f"  num_maps       = {num_maps}")
print(f"  Total PEPS decodes: {len(distances) * len(p_mean_values) * num_maps * shots}")
print()

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

results = {}

for d in distances:
    print(f"distance = {d}")
    peps_local_mean = []
    peps_local_std  = []
    mwpm_local_mean = []
    mwpm_local_std  = []
    peps_uniform    = []
    mwpm_uniform    = []

    for j, p_mean in enumerate(p_mean_values):

        peps_trials = []
        mwpm_trials = []

        for k in range(num_maps):
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
            peps_trials.append(res_peps.logical_error_rate)
            mwpm_trials.append(res_mwpm.logical_error_rate)

        peps_trials = np.asarray(peps_trials, dtype=float)
        mwpm_trials = np.asarray(mwpm_trials, dtype=float)

        peps_local_mean.append(peps_trials.mean())
        peps_local_std.append(peps_trials.std(ddof=0))
        mwpm_local_mean.append(mwpm_trials.mean())
        mwpm_local_std.append(mwpm_trials.std(ddof=0))

        # Uniform baseline
        res_peps_uni = run_surface_code_peps_spin_qubit_uniform(
            distance=d, p=float(p_mean), shots=shots,
            Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        )
        res_mwpm_uni = run_surface_code_mwpm_spin_qubit_uniform(
            distance=d, p=float(p_mean), shots=shots, rounds=rounds,
        )
        peps_uniform.append(res_peps_uni.logical_error_rate)
        mwpm_uniform.append(res_mwpm_uni.logical_error_rate)

        print(
            f"  p_mean={p_mean:.4f}  "
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
