
import numpy as np
import string
from typing import List, Tuple, Dict
from src.functions import * 
from src.mtimes_MPO import * 
from src.PEPS import * 

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

    print("Exact contraction :", exact)
    print("MPO-MPS contraction:", approx)
    print("Abs diff          :", abs(exact - approx))
    print("Rel diff          :", abs(exact - approx) / (abs(exact) + 1e-30))

    # assert tight agreement
    assert np.allclose(approx, exact, rtol=1e-10, atol=1e-10), "Mismatch: finPEPS vs exact"


if __name__=="__main__":
    
    print("Test1 : PEPS contraction ")
    run_finpeps_test() 
    