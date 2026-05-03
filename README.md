# Noise-Adapted Tensor Network Maximum Likelihood Decoder

Benchmarks a noise-adapted maximum-likelihood (ML) decoder for the surface code against minimum-weight perfect matching (MWPM) under spatially inhomogeneous hardware noise.

The ML decoder is implemented via PEPS tensor-network contraction, which computes exact coset likelihoods using per-qubit error rates. MWPM is implemented via PyMatching and is also given access to the same local rates (informed MWPM), making the comparison fair.

## Noise models

| Model | Noise channel | Spatial variation |
|---|---|---|
| Depolarizing | `DEPOLARIZE1(p_i)` | `p_i ~ Normal(p_mean, sigma_frac * p_mean)` |
| Spin qubit | `Z_ERROR(pz_i)` | `pz_i ~ Normal(p_mean, sigma_frac * p_mean)` |
| EO qubit | `PAULI_CHANNEL_1(px_i, 0, pz_i)` | `pz_i, pn_i ~ Normal(p_mean, sigma_frac * p_mean)`, `px_i = (3/4)pn_i`, `pz_i = pz_i + (1/4)pn_i` |

All experiments use the unrotated surface code under the code-capacity noise model (noiseless syndrome extraction).

## Repository structure

```
├── depolarizing_comparison.py    # sweep script: depolarizing noise
├── spin_qubit_comparison.py      # sweep script: spin qubit Z dephasing
├── eo_qubit_comparison.py        # sweep script: EO qubit biaxial noise
│
├── Generate_figure.ipynb         # plots for all three noise models
│
├── Dockerfile                    # reproducible environment
├── test.py                       # sanity checks
│
└── src/
    ├── stim_PEPS_wrapper.py      # ML decoder entry points
    ├── mwpm_decoder.py           # MWPM decoder entry points
    ├── PEPS_Pauli_decoder.py     # coset likelihood computation via PEPS
    ├── PEPS.py                   # PEPS tensor network contraction
    ├── weights_PEPS.py           # noise-model weight tensors
    ├── stim_sampler.py           # Stim circuit generation and sampling
    ├── functions.py              # tensor contraction utilities
    └── mtimes_MPO.py             # MPO-MPO multiplication
```

## How to run

### 1. Set up the environment

**Using Docker (recommended):**
```bash
docker build -t tndecoder .
docker run -it --rm -v $(pwd):/workspace tndecoder bash
conda activate TNDecoder
```

**Using conda directly:**
```bash
conda create -n TNDecoder python=3.11
conda activate TNDecoder
pip install numpy==2.2.6 stim==1.15.0 PyMatching==2.3.1 matplotlib==3.10.8 scipy==1.15.3
```

### 2. Run a sweep

Each script sweeps over `(distance, p_mean)` pairs, runs both decoders in parallel, and saves results to a `.pkl` file:

```bash
python depolarizing_comparison.py   # → depolarizing_results.pkl
python spin_qubit_comparison.py     # → spin_qubit_results.pkl
python eo_qubit_comparison.py       # → eo_qubit_results.pkl
```

Key sweep parameters (edit at the top of each script):

| Parameter | Default | Description |
|---|---|---|
| `distances` | `[3, 5, 7]` | Surface code distances |
| `p_mean_values` | `geomspace(0.01, 0.20, 12)` | Mean physical error rates |
| `shots` | `500` | Samples per (distance, p_mean, noise map) |
| `sigma_frac` | `0.30` | Spatial noise spread: std = sigma_frac × p_mean |
| `num_maps` | `3` | Independent noise map realizations to average over |
| `Nkeep` | `32` | PEPS bond dimension |
| `num_workers` | `cpu_count - 1` | Parallel workers |

### 3. Plot results

Run all three sweep scripts first to generate the `.pkl` result files (not tracked in the repo), then open `Generate_figure.ipynb` and run all cells.

## Dependencies

| Package | Version |
|---|---|
| Python | 3.11 |
| numpy | 2.2.6 |
| stim | 1.15.0 |
| PyMatching | 2.3.1 |
| matplotlib | 3.10.8 |
| scipy | 1.15.3 |
