
## Project Goal

This project implements an **optimal Maximum Likelihood (ML) decoder using tensor networks (PEPS)** and compares its performance against a **Minimum Weight Perfect Matching (MWPM)** decoder under spatially varying noise.

The main research hypothesis:

**An ML decoder adapted to local noise should outperform MWPM when noise is spatially inhomogeneous.**

The focus is on rigorous comparison under controlled simulation conditions using the surface code.

---

## Scientific Objective

We want to demonstrate:

1. ML decoding gives optimal logical inference when the noise model is known.
2. MWPM is optimal primarily under uniform noise assumptions.
3. Under spatially varying depolarizing noise, ML should outperform MWPM.
4. Tensor network contraction (PEPS) provides a tractable ML decoding method.

Primary performance metric:

**Logical error rate (diamond norm logical channel error)**

Secondary metrics:

---

## Pipeline

### Step 1 — Syndrome generation

Generate syndromes using **Stim** with:

- Unrotated surface code
- Code capacity noise model
- Exact (noiseless) syndrome extraction
- Depolarizing channel

Pipeline:
Sample Pauli error
↓
Stim detector simulation
↓
Extract syndrome (sX, sZ)
↓
True logical sector labels


Stim is used only for **data generation**, not decoding.

---

### Step 2 — Decoder comparison (uniform noise)

Given the syndrome:

Compute logical error rate of:

### ML decoder (PEPS)

Computes coset likelihood:

\[
Z_L(s)=\sum_{E\in\text{coset}(L)} P(E)
\]

Select:

\[
L^* = \arg\max_L Z_L
\]

### MWPM decoder

Uses PyMatching:

- Minimum weight matching
- Assumes uniform error rates
- Note that pymatching usually uses 3D syndrome data (2D spatial + 1D time), but here, to compare performance with ML, we use 2D spatial data (at fixed time)

Comparison metrics:

- Diamond norm logical error

---

### Step 3 — Local noise adaptation

Generalize depolarizing model:

Instead of:

\[
p_i = p
\]

Use:

\[
p_i \neq p_j
\]

Implement:

- Site dependent depolarizing rates
- Spatial noise maps
- Inhomogeneous error weights

Repeat comparison.

ML should improve because:

ML uses:

\[
P(E)=\prod_i P_i(E_i)
\]

MWPM approximates:

\[
\text{weight} \sim \log(p)
\]

and does not fully capture spatial structure.

---

## Expected Scientific Result

We aim to demonstrate:

### Uniform noise:

ML ≈ MWPM

### Local noise:

ML > MWPM
(ML performs better, which means ML gives lower error rate)

Specifically:

under inhomogeneous noise.

---

## Code Architecture

### Syndrome generation
surface_code_sampler_full.py

Responsibilities:

- Generate Pauli errors
- Run Stim
- Extract syndrome
- Provide logical truth labels

### ML decoder (tensor network)
PEPS_Pauli_decoder.py

Responsibilities:

- Construct PEPS from noise model
- Contract tensor network
- Compute coset likelihoods
- Return ML logical sector

Core operations:
build_pauli_peps()
contract_finPEPS()
pauli_coset_likelihoods_peps()


---

### MWPM decoder
mwpm_decoder_2d.py

Responsibilities:

- Convert syndrome to graph
- Run PyMatching
- Return logical prediction

---

### Comparison drivers
compare_peps_mwpm_surface_code.py

Responsibilities:

- Run both decoders on same batch
- Compute logical error rates
- Produce performance tables

---

## Key Correctness Requirements

### Same input requirement

Both decoders must use:

- Same syndrome
- Same noise realization
- Same boundary conventions

---

