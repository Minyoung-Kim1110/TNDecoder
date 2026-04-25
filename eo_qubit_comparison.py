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
print(f"  axis model     = p_mean_z = p_mean_n = p_mean")
print(f"  Total PEPS decodes: {len(distances) * len(p_mean_values) * num_maps * shots * 2}")
print()

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

results = {}

for d in distances:
    print(f"distance = {d}")
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
        p_mean_z = float(p_mean)
        p_mean_n = float(p_mean)

        peps_full_trials = []
        mwpm_full_trials = []
        peps_lx_trials   = []
        peps_lz_trials   = []
        mwpm_lx_trials   = []
        mwpm_lz_trials   = []

        for k in range(num_maps):
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
            peps_full_trials.append(res_peps.logical_error_rate)
            mwpm_full_trials.append(res_mwpm.logical_error_rate)
            peps_lx_trials.append(res_peps.p_L_X)
            peps_lz_trials.append(res_peps.p_L_Z)
            mwpm_lx_trials.append(res_mwpm.p_L_X)
            mwpm_lz_trials.append(res_mwpm.p_L_Z)

        peps_full_trials = np.asarray(peps_full_trials)
        mwpm_full_trials = np.asarray(mwpm_full_trials)

        peps_local_mean.append(peps_full_trials.mean())
        peps_local_std.append(peps_full_trials.std(ddof=0))
        mwpm_local_mean.append(mwpm_full_trials.mean())
        mwpm_local_std.append(mwpm_full_trials.std(ddof=0))
        peps_local_lx_mean.append(np.mean(peps_lx_trials))
        peps_local_lz_mean.append(np.mean(peps_lz_trials))
        mwpm_local_lx_mean.append(np.mean(mwpm_lx_trials))
        mwpm_local_lz_mean.append(np.mean(mwpm_lz_trials))

        # Uniform baseline (same axis rates, no spatial variation)
        res_peps_uni = run_surface_code_peps_eo_qubit_uniform(
            distance=d, p_mean_z=p_mean_z, p_mean_n=p_mean_n,
            shots=shots, Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        )
        res_mwpm_uni = run_surface_code_mwpm_eo_qubit_uniform(
            distance=d, p_mean_z=p_mean_z, p_mean_n=p_mean_n,
            shots=shots, rounds=rounds,
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
