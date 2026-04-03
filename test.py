
import numpy as np
import stim
import string
from typing import List, Tuple, Dict, Optional
from src.functions import * 
from src.mtimes_MPO import * 
from src.PEPS import * 
from src.PEPS_Pauli_decoder import *
from src.weights_PEPS import * 
from src.stim_sampler import * 
from src.stim_PEPS_wrapper import * 
from src.mwpm_decoder import *

# Generate random open boundary grid for test 
def make_random_open_boundary_grid(
    nrow: int,
    ncol: int,
    Dh: int,
    Dv: int,
    seed: int = 0,
    complex_: bool = True,
) -> List[List[np.ndarray]]:
    """
    Create a grid T[m][n] of rank-4 reduced tensors with legs (L,U,D,R),
    satisfying open boundary conditions:
      - top boundary: U = 1
      - bottom boundary: D = 1
      - left boundary: L = 1
      - right boundary: R = 1
    and internal bonds:
      - horizontal bonds have dimension Dh
      - vertical bonds have dimension Dv
    """
    rng = np.random.default_rng(seed)

    # bond dimensions
    # Horizontal bond between (i,j) and (i,j+1): dim Dh
    # Vertical bond between (i,j) and (i+1,j): dim Dv
    T = [[None for _ in range(ncol)] for _ in range(nrow)]

    for i in range(nrow):
        for j in range(ncol):
            Dl = 1 if j == 0 else Dh
            Dr = 1 if j == ncol - 1 else Dh
            Du = 1 if i == 0 else Dv
            Dd = 1 if i == nrow - 1 else Dv

            shape = (Dl, Du, Dd, Dr)
            if complex_:
                X = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                X = X / np.sqrt(np.prod(shape))
            else:
                X = rng.normal(size=shape)
                X = X / np.sqrt(np.prod(shape))
            T[i][j] = X.astype(np.complex128)

    # sanity check bond matching
    for i in range(nrow):
        for j in range(ncol):
            if j < ncol - 1:
                assert T[i][j].shape[3] == T[i][j + 1].shape[0]
            if i < nrow - 1:
                assert T[i][j].shape[2] == T[i + 1][j].shape[1]
    return T

# Exact contraction (for small grid)
def exact_contract_grid_einsum(T: List[List[np.ndarray]]) -> complex:
    """
    Exact contraction of a small open-boundary grid of rank-4 tensors
    with legs ordered (L,U,D,R), using numpy.einsum.

    This is intended for TESTING on small lattices (e.g., up to ~4x4).
    """
    nrow = len(T)
    ncol = len(T[0])

    # einsum needs single-character labels; we keep grids small.
    labels = list(string.ascii_letters)  # 52 labels
    # We need labels for every unique bond index. Count:
    # horizontal internal bonds: nrow*(ncol-1)
    # vertical internal bonds: (nrow-1)*ncol
    # plus boundary legs (all dim 1) can be unique too, but we can also reuse.
    needed = nrow * (ncol - 1) + (nrow - 1) * ncol + 2 * nrow * ncol  # generous
    if needed > len(labels):
        raise ValueError(
            f"Grid too large for this simple einsum labeler (need {needed}, have {len(labels)})."
        )

    # Create unique labels for each bond:
    # horizontal bond label h[i,j] corresponds to bond between (i,j) right and (i,j+1) left
    # vertical bond label v[i,j] corresponds to bond between (i,j) down and (i+1,j) up
    h = [[None for _ in range(ncol - 1)] for _ in range(nrow)]
    v = [[None for _ in range(ncol)] for _ in range(nrow - 1)]

    k = 0
    for i in range(nrow):
        for j in range(ncol - 1):
            h[i][j] = labels[k]
            k += 1
    for i in range(nrow - 1):
        for j in range(ncol):
            v[i][j] = labels[k]
            k += 1

    # Boundary labels (can be unique; dims are 1 anyway)
    # L boundary at (i,0)
    Lb = [labels[k + i] for i in range(nrow)]
    k += nrow
    # R boundary at (i,ncol-1)
    Rb = [labels[k + i] for i in range(nrow)]
    k += nrow
    # U boundary at (0,j)
    Ub = [labels[k + j] for j in range(ncol)]
    k += ncol
    # D boundary at (nrow-1,j)
    Db = [labels[k + j] for j in range(ncol)]
    k += ncol

    subs = []
    ops = []

    for i in range(nrow):
        for j in range(ncol):
            # legs: (L,U,D,R)
            L = Lb[i] if j == 0 else h[i][j - 1]
            R = Rb[i] if j == ncol - 1 else h[i][j]
            U = Ub[j] if i == 0 else v[i - 1][j]
            D = Db[j] if i == nrow - 1 else v[i][j]

            subs.append(L + U + D + R)
            ops.append(T[i][j])

    # output is scalar: contract all indices
    eq = ",".join(subs) + "->"
    out = np.einsum(eq, *ops)
    return complex(out)

# Check if contract_finPEPS() is similar to exact contraction 
def run_finpeps_test():
    # -------------------------
    # Choose a SMALL test grid
    # -------------------------
    nrow, ncol = 3, 4
    Dh, Dv = 2, 2
    Nsweep = 2

    # Large enough Nkeep should reproduce exact contraction
    # For random tensors, safe upper bound is Dh^2 * Dv^2-ish, but keep it generous for testing.
    Nkeep = 64

    T = make_random_open_boundary_grid(nrow, ncol, Dh=Dh, Dv=Dv, seed=7, complex_=True)

    exact = exact_contract_grid_einsum(T)

    approx = contract_finPEPS(T, Nkeep=Nkeep, Nsweep=Nsweep)

    print("Abs diff          :", abs(exact - approx))
    print("Rel diff          :", abs(exact - approx) / (abs(exact) + 1e-30))

    # assert tight agreement
    assert np.allclose(approx, exact, rtol=1e-10, atol=1e-10), "Mismatch: finPEPS vs exact"




# ============================================================
# Exact reference for tiny instances
# ============================================================

def _x_syndrome_from_z_bits(h_z, v_z):
    """
    X-check syndrome on faces from z-bits.
    """
    nrow = v_z.shape[0]
    ncol = h_z.shape[1]
    sX = np.zeros((nrow, ncol), dtype=np.uint8)
    for r in range(nrow):
        for c in range(ncol):
            sX[r, c] = h_z[r, c] ^ h_z[r + 1, c] ^ v_z[r, c] ^ v_z[r, c + 1]
    return sX

def _z_syndrome_from_x_bits(h_x, v_x):
    """
    Z-check syndrome on faces from x-bits.
    """
    nrow = v_x.shape[0]
    ncol = h_x.shape[1]
    sZ = np.zeros((nrow, ncol), dtype=np.uint8)
    for r in range(nrow):
        for c in range(ncol):
            sZ[r, c] = h_x[r, c] ^ h_x[r + 1, c] ^ v_x[r, c] ^ v_x[r, c + 1]
    return sZ

def _logical_x_parity(v_x, cut_col):
    return int(np.bitwise_xor.reduce(v_x[:, cut_col]))

def _logical_z_parity(h_z, cut_row):
    return int(np.bitwise_xor.reduce(h_z[cut_row, :]))

# Exact contraction for Pauli coset likelihoods 
def pauli_coset_likelihoods_exact(sX, sZ, W_h, W_v, logical_x_cut_col, logical_z_cut_row):
    """
    Exact brute-force reference for tiny lattices only.
    """
    sX = np.asarray(sX, dtype=np.uint8)
    sZ = np.asarray(sZ, dtype=np.uint8)

    nrow, ncol = sX.shape
    W_h, W_v = validate_local_weight_tensor(W_h, W_v, nrow, ncol)

    out = {(0, 0): 0.0, (1, 0): 0.0, (0, 1): 0.0, (1, 1): 0.0}

    nh = (nrow + 1) * ncol
    nv = nrow * (ncol + 1)

    # Pauli index convention induced by W[x,z]:
    # idx 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # x bit = idx // 2, z bit = idx % 2
    for flat_h in np.ndindex(*(4,) * nh):
        h_idx = np.array(flat_h, dtype=np.uint8).reshape(nrow + 1, ncol)
        h_x = h_idx // 2
        h_z = h_idx % 2

        for flat_v in np.ndindex(*(4,) * nv):
            v_idx = np.array(flat_v, dtype=np.uint8).reshape(nrow, ncol + 1)
            v_x = v_idx // 2
            v_z = v_idx % 2

            if not np.array_equal(_x_syndrome_from_z_bits(h_z, v_z), sX):
                continue
            if not np.array_equal(_z_syndrome_from_x_bits(h_x, v_x), sZ):
                continue

            lx = _logical_x_parity(v_x, logical_x_cut_col)
            lz = _logical_z_parity(h_z, logical_z_cut_row)

            prob = 1.0
            for idx, flat in np.ndenumerate(h_idx):
                x = flat // 2
                z = flat % 2
                prob *= W_h[idx + (x, z)]
            for idx, flat in np.ndenumerate(v_idx):
                x = flat // 2
                z = flat % 2
                prob *= W_v[idx + (x, z)]

            out[(lx, lz)] += prob

    return out

# Compute logical coset likelihood using peps, exact and compare 
def test_zero_noise():
    """
    If every qubit is I with probability 1, only zero syndrome and zero coset survive.
    """
    nrow, ncol = 2, 2
    W_h, W_v = biased_pauli_weights(
        nrow, ncol,
        pI=1.0, pX=0.0, pY=0.0, pZ=0.0
    )
    sX = np.zeros((2, 2), dtype=np.uint8)
    sZ = np.zeros((2, 2), dtype=np.uint8)

    cosets = pauli_coset_likelihoods_peps(
        sX, sZ, W_h, W_v,
        logical_x_cut_col=1,   # for 1x1 there is no internal cut; this PEPS twist
        logical_z_cut_row=1,   # interface is for larger lattices
        Nkeep=32, Nsweep=0
    )
    # For 1x1 there are no nontrivial internal twists. So don't use this test
    # for cut-based parity separation. Keep it only for total-likelihood sanity.
    total = total_likelihood_from_cosets(cosets)
    assert np.isclose(total, 1.0, atol=1e-12), total
      
def test_peps_matches_exact_small():
    """
    Compare PEPS and exact on a tiny 2x2 face lattice.
    """
    nrow, ncol = 2, 2
    W_h, W_v = depolarizing_weights(nrow, ncol, p=0.12)

    sX = np.array([[0, 1],
                   [1, 0]], dtype=np.uint8)
    sZ = np.array([[1, 0],
                   [0, 1]], dtype=np.uint8)

    # Internal cuts exist for 2x2 faces:
    # vertical cut between columns 0 and 1 -> cut_col=1
    # horizontal cut between rows 0 and 1 -> cut_row=1
    cos_peps = pauli_coset_likelihoods_peps(
        sX, sZ, W_h, W_v,
        logical_x_cut_col=1,
        logical_z_cut_row=1,
        Nkeep=128, Nsweep=0
    )

    cos_exact = pauli_coset_likelihoods_exact(
        sX, sZ, W_h, W_v,
        logical_x_cut_col=1,
        logical_z_cut_row=1
    )

    for key in cos_exact:
        if not np.isclose(cos_peps[key], cos_exact[key], atol=1e-10, rtol=1e-10):
            raise AssertionError(
                f"Mismatch for coset {key}: PEPS={cos_peps[key]}, exact={cos_exact[key]}"
            )

def run_peps_decoder_tests():
    # zero_noise is only a weak total-probability check under this cut convention
    test_zero_noise()
    # and is not the main correctness test.
    test_peps_matches_exact_small()
    print("All PEPS Pauli decoder tests passed.")


# Test surface code stim sampler with connection to PEPS decoder 
def test_peps_tensor_shapes_are_consistent():
    sample = sample_surface_code_depolarizing(
        distance=5,
        p=0.0,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    nrow, ncol = sample.sX.shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p=0.0)

    T = build_pauli_peps(
        sX=sample.sX,
        sZ=sample.sZ,
        active_X=sample.active_X,
        active_Z=sample.active_Z,
        W_h=W_h,
        W_v=W_v,
    )

    assert len(T) == nrow
    assert all(len(row) == ncol for row in T)

    for r in range(nrow):
        for c in range(ncol):
            A = T[r][c]
            assert A.ndim == 4
            if r == 0:
                assert A.shape[1] == 1
            if r == nrow - 1:
                assert A.shape[2] == 1
            if c == 0:
                assert A.shape[0] == 1
            if c == ncol - 1:
                assert A.shape[3] == 1
            if r < nrow - 1:
                assert A.shape[2] == T[r + 1][c].shape[1], (
                    f"vertical mismatch at ({r},{c})"
                )
            if c < ncol - 1:
                assert A.shape[3] == T[r][c + 1].shape[0], (
                    f"horizontal mismatch at ({r},{c})"
                )
                
def test_no_noise_gives_zero_syndrome():
    sample = sample_surface_code_depolarizing(distance=5, p=0.0, memory_basis="x")
    assert np.all(sample.sX == 0), f"sX not zero:\n{sample.sX}"
    assert np.all(sample.sZ == 0), f"sZ not zero:\n{sample.sZ}"


def test_masks_are_binary_and_disjoint():
    sample = sample_surface_code_depolarizing(distance=5, p=1e-3, memory_basis="x")
    assert set(np.unique(sample.active_X)).issubset({0, 1})
    assert set(np.unique(sample.active_Z)).issubset({0, 1})
    assert np.all((sample.active_X & sample.active_Z) == 0), "X/Z masks overlap."


def test_shapes_match_decoder_api():
    sample = sample_surface_code_depolarizing(distance=5, p=1e-3, memory_basis="x")
    assert sample.sX.shape == sample.sZ.shape
    assert sample.active_X.shape == sample.sX.shape
    assert sample.active_Z.shape == sample.sX.shape


def test_masked_decoder_runs_noiseless():
    sample = sample_surface_code_depolarizing(distance=5, p=0.0, memory_basis="x")
    nrow, ncol = sample.sX.shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p=0.0)

    cosets = pauli_coset_likelihoods_peps(
        sX=sample.sX,
        sZ=sample.sZ,
        active_X=sample.active_X,
        active_Z=sample.active_Z,
        W_h=W_h,
        W_v=W_v,
        Nkeep=64,
        Nsweep=1,
    )

    for k, v in cosets.items():
        assert np.isfinite(v), f"Non-finite coset value for {k}: {v}"


def test_one_shot_end_to_end():
    out = sample_and_decode_surface_code_depolarizing(
        distance=5,
        p=1e-3,
        memory_basis="x",
        Nkeep=64,
        Nsweep=1,
    )
    assert "sample" in out
    assert "cosets" in out
    assert "ml_coset" in out
    assert len(out["cosets"]) == 4
    
def run_sampler_connection_test():
    test_no_noise_gives_zero_syndrome()
    test_masks_are_binary_and_disjoint()
    test_peps_tensor_shapes_are_consistent()
    test_shapes_match_decoder_api()
    test_masked_decoder_runs_noiseless()
    test_one_shot_end_to_end()
    
    print("All sampler with Stim tests passed.")
    
    
def test_batch_shapes() -> None:
    batch = sample_surface_code_depolarizing_batch(
        distance=5,
        p=1e-3,
        shots=8,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    assert batch.detector_bits.ndim == 2
    assert batch.observable_flips.ndim == 2
    assert batch.sX.ndim == 3
    assert batch.sZ.ndim == 3
    assert batch.active_X.shape == batch.sX.shape
    assert batch.active_Z.shape == batch.sZ.shape
    assert batch.shots == 8


def test_single_shot_wrapper_matches_batch() -> None:
    batch = sample_surface_code_depolarizing_batch(
        distance=5,
        p=1e-3,
        shots=1,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    shot = batch.get_shot(0)
    assert shot.detector_bits.ndim == 1
    assert shot.observable_flips.ndim == 1
    assert shot.sX.ndim == 2
    assert shot.sZ.ndim == 2


def run_sampler_batch_tests() -> None:
    test_batch_shapes()
    test_single_shot_wrapper_matches_batch()
    print("All stim_sampler batch tests passed.")
    
def test_mwpm_decodes_one_stim_surface_sample() -> None:
    sample = sample_surface_code_depolarizing(
        distance=5,
        p=1e-3,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_stim_surface_sample_with_mwpm(sample)

    assert out.predicted_observable_flips.shape == sample.observable_flips.shape
    assert out.predicted_observable_flips.dtype == np.uint8
    

def test_mwpm_decodes_stim_surface_batch(shots: int = 32) -> None:
    batch = sample_surface_code_depolarizing_batch(
        distance=5,
        p=1e-3,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_stim_surface_batch_with_mwpm(batch)

    assert out.detector_bits.shape[0] == shots
    assert out.actual_observable_flips.shape[0] == shots
    assert out.predicted_observable_flips.shape == out.actual_observable_flips.shape
    assert out.logical_failures.shape == (shots,)
    

def test_zero_noise_stim_surface_sample() -> None:
    sample = sample_surface_code_depolarizing(
        distance=5,
        p=0.0,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_stim_surface_sample_with_mwpm(sample)

    assert np.all(sample.observable_flips == 0), (
        f"Observable flips should be zero at p=0, got {sample.observable_flips}"
    )
    assert not out.logical_failure
   

def test_zero_noise_stim_surface_batch(shots: int = 64) -> None:
    batch = sample_surface_code_depolarizing_batch(
        distance=5,
        p=0.0,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_stim_surface_batch_with_mwpm(batch)

    assert out.logical_error_rate == 0.0
    

def test_same_sample_can_be_used_for_peps_and_mwpm() -> None:
    sample = sample_surface_code_depolarizing(
        distance=5,
        p=1e-3,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_stim_surface_sample_with_mwpm(sample)

    assert sample.sX.shape == sample.sZ.shape
    assert sample.active_X.shape == sample.sX.shape
    assert sample.active_Z.shape == sample.sX.shape
    assert sample.detector_bits.ndim == 1
    assert sample.observable_flips.ndim == 1
    assert out.predicted_observable_flips.shape == sample.observable_flips.shape
    

def test_same_batch_can_be_used_for_peps_and_mwpm(shots: int = 16) -> None:
    batch = sample_surface_code_depolarizing_batch(
        distance=5,
        p=1e-3,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_stim_surface_batch_with_mwpm(batch)

    assert batch.sX.ndim == 3
    assert batch.sZ.ndim == 3
    assert batch.active_X.shape == batch.sX.shape
    assert batch.active_Z.shape == batch.sZ.shape
    assert batch.detector_bits.ndim == 2
    assert batch.observable_flips.ndim == 2
    assert out.predicted_observable_flips.shape == batch.observable_flips.shape


def test_batch_mwpm_surface_code_same_circuit_factory(shots: int = 128) -> None:
    result = run_surface_code_mwpm_batch(
        distance=5,
        p=1e-3,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )

    assert result.detector_bits.shape[0] == shots
    assert result.actual_observable_flips.shape[0] == shots
    assert result.predicted_observable_flips.shape == result.actual_observable_flips.shape


def test_noise_trend_surface_code_same_sampler(
    shots: int = 2000,
    p_low: float = 5e-4,
    p_high: float = 2e-2,
) -> None:
    low = run_surface_code_mwpm_batch(
        distance=5,
        p=p_low,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    high = run_surface_code_mwpm_batch(
        distance=5,
        p=p_high,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )

    assert high.logical_error_rate >= low.logical_error_rate
    

def run_MWPM_decoder_tests() -> None:
    test_mwpm_decodes_one_stim_surface_sample()
    test_mwpm_decodes_stim_surface_batch()
    test_zero_noise_stim_surface_sample()
    test_zero_noise_stim_surface_batch()
    test_same_sample_can_be_used_for_peps_and_mwpm()
    test_same_batch_can_be_used_for_peps_and_mwpm()
    test_batch_mwpm_surface_code_same_circuit_factory()
    test_noise_trend_surface_code_same_sampler()
    print("All MWPM tests using StimSurfaceSample / StimSurfaceBatchSample passed.")


if __name__=="__main__":
    
    # print("Test1 : PEPS contraction ")
    # run_finpeps_test() 
    # print("Test2 : PEPS Pauli coset likelihood decoder")
    # run_peps_decoder_tests()
    # print("Test3 : Surface code - sample syndrome with Stim and convert to input API of PEPS decoder")
    # run_sampler_connection_test() 
    # print("Test4: Sampler for batch version")
    # run_sampler_batch_tests() 
    print("Test5: build MWPM with PyMatching and test decoder")
    run_MWPM_decoder_tests()
    
    
    