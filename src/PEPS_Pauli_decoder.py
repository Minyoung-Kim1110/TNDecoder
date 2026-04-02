import numpy as np
from typing import Dict, Tuple, Optional
from .PEPS import contract_finPEPS


# Pauli utils 
PAULI_TO_XZ = {"I": (0, 0),
               "X": (1, 0),
               "Y": (1, 1),
               "Z": (0, 1),}

# Weights for each edges 
def pauli_probs_to_weight_matrix(pI, pX, pY, pZ):
    """
    Return W[x,z] where
        (x,z) = (0,0)->I, (1,0)->X, (1,1)->Y, (0,1)->Z

    So:
        W[0,0] = pI
        W[1,0] = pX
        W[1,1] = pY
        W[0,1] = pZ
    """
    W = np.zeros((2, 2), dtype=np.float64)
    W[0, 0] = pI
    W[1, 0] = pX
    W[1, 1] = pY
    W[0, 1] = pZ
    return W

def validate_local_weight_tensor(W_h, W_v, nrow, ncol):
    """
    W_h: shape (nrow+1, ncol, 2, 2)
         horizontal edge qubits
    W_v: shape (nrow, ncol+1, 2, 2)
         vertical edge qubits

    Each W[..., x, z] must sum to 1 over (x,z).
    """
    W_h = np.asarray(W_h, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)

    if W_h.shape != (nrow + 1, ncol, 2, 2):
        raise ValueError(f"W_h must have shape {(nrow+1, ncol, 2, 2)}, got {W_h.shape}")
    if W_v.shape != (nrow, ncol + 1, 2, 2):
        raise ValueError(f"W_v must have shape {(nrow, ncol+1, 2, 2)}, got {W_v.shape}")

    sums_h = W_h.sum(axis=(-2, -1))
    sums_v = W_v.sum(axis=(-2, -1))

    if not np.allclose(sums_h, 1.0, atol=1e-12):
        raise ValueError("Each horizontal qubit weight tensor must sum to 1.")
    if not np.allclose(sums_v, 1.0, atol=1e-12):
        raise ValueError("Each vertical qubit weight tensor must sum to 1.")

    return W_h, W_v

def biased_pauli_weights(nrow, ncol, pI, pX, pY, pZ):
    """
    Uniform but biased Pauli noise.
    """
    if not np.isclose(pI + pX + pY + pZ, 1.0, atol=1e-12):
        raise ValueError("Probabilities must sum to 1.")
    
    W = pauli_probs_to_weight_matrix(pI, pX, pY, pZ)
    W_h = np.broadcast_to(W, (nrow + 1, ncol, 2, 2)).copy()
    W_v = np.broadcast_to(W, (nrow, ncol + 1, 2, 2)).copy()
    return W_h, W_v

def depolarizing_weights(nrow, ncol, p):
    """
    Uniform depolarizing noise: p(I)=1-p, p(X)=p(Y)=p(Z)=p/3
    """
    return biased_pauli_weights(nrow, ncol, 1.0 - p, p / 3.0, p / 3.0, p / 3.0)

# Local tensor construction 

def _build_face_tensor(
    row: int,
    col: int, # location of plaquette 
    sX: np.ndarray,
    sZ: np.ndarray, # syndrome measurements 
    W_h: np.ndarray,
    W_v: np.ndarray, # weights to each edges 
    active_X: np.ndarray,
    active_Z: np.ndarray, # masks 
):
    r"""
    Build one local tensor on a rectangular face grid, but only enforce:
      - X-check syndrome if active_X[row,col] == 1
      - Z-check syndrome if active_Z[row,col] == 1

    Conventions inherited from your existing code:
      - horizontal edges: W_h[row, col] (top), W_h[row+1, col] (bottom)
      - vertical edges:   W_v[row, col] (left), W_v[row, col+1] (right)
      - bond state flattening: idx = 2*x + z
    """
    sx = int(sX[row, col])
    sz = int(sZ[row, col])
    ax = int(active_X[row, col])
    az = int(active_Z[row, col])

    nrow, ncol = sX.shape

    Dl = 1 if col == 0 else 4
    Du = 1 if row == 0 else 4
    Dd = 1 if row == nrow - 1 else 4
    Dr = 1 if col == ncol - 1 else 4

    T = np.zeros((Dl, Du, Dd, Dr), dtype=np.float64)

    Wu = W_h[row, col]
    Wd = W_h[row + 1, col]
    Wl = W_v[row, col]
    Wr = W_v[row, col + 1]

    def idx_to_xz(idx: int):
        return idx // 2, idx % 2

    all_states = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for l in range(Dl):
        left_states = all_states if Dl == 1 else [idx_to_xz(l)]
        for u in range(Du):
            up_states = all_states if Du == 1 else [idx_to_xz(u)]
            for d in range(Dd):
                down_states = all_states if Dd == 1 else [idx_to_xz(d)]
                for r in range(Dr):
                    right_states = all_states if Dr == 1 else [idx_to_xz(r)]

                    val = 0.0

                    for xl, zl in left_states:
                        for xu, zu in up_states:
                            for xd, zd in down_states:
                                for xr, zr in right_states:
                                    # Existing convention in your code:
                                    # X-check syndrome comes from Z-parity on incident edges
                                    # Z-check syndrome comes from X-parity on incident edges
                                    parity_for_Xcheck = zu ^ zd ^ zl ^ zr
                                    parity_for_Zcheck = xu ^ xd ^ xl ^ xr

                                    if ax and parity_for_Xcheck != sx:
                                        continue
                                    if az and parity_for_Zcheck != sz:
                                        continue

                                    weight = 1.0
                                    weight *= np.sqrt(Wl[xl, zl]) if Dl != 1 else Wl[xl, zl]
                                    weight *= np.sqrt(Wu[xu, zu]) if Du != 1 else Wu[xu, zu]
                                    weight *= np.sqrt(Wd[xd, zd]) if Dd != 1 else Wd[xd, zd]
                                    weight *= np.sqrt(Wr[xr, zr]) if Dr != 1 else Wr[xr, zr]

                                    val += weight

                    T[l, u, d, r] = val

    return T

def build_pauli_peps(
    sX: np.ndarray,
    sZ: np.ndarray,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X=None,
    active_Z=None,

):
    """
    Build PEPS for a rectangular embedding.
    """
    sX = np.asarray(sX, dtype=np.uint8)
    sZ = np.asarray(sZ, dtype=np.uint8)
    nrow, ncol = sX.shape
    
    if active_X is None:
        active_X = np.ones((nrow, ncol), dtype=np.uint8)
    else:
        active_X = np.asarray(active_X, dtype=np.uint8)

    if active_Z is None:
        active_Z = np.ones((nrow, ncol), dtype=np.uint8)
    else:
        active_Z = np.asarray(active_Z, dtype=np.uint8)
    
    if sX.shape != sZ.shape:
        raise ValueError("sX and sZ must have the same shape.")
    if active_X.shape != sX.shape or active_Z.shape != sX.shape:
        raise ValueError("active_X and active_Z must have the same shape as sX.")

    W_h, W_v = validate_local_weight_tensor(W_h, W_v, nrow, ncol)

    T = []
    for r in range(nrow):
        row_tensors = []
        for c in range(ncol):
            A = _build_face_tensor(
                row=r,
                col=c,
                sX=sX,
                sZ=sZ,
                active_X=active_X,
                active_Z=active_Z,
                W_h=W_h,
                W_v=W_v,
            )
            row_tensors.append(A)
        T.append(row_tensors)

    return T

# Logical coset twists 
def _twist_vertical_cut_x(T, cut_col):
    """
    Insert (-1)^x on the vertical cut, matching the convention in PEPS_Pauli_decoder.
    """
    nrow = len(T)
    ncol = len(T[0])
    if not (1 <= cut_col <= ncol - 1):
        raise ValueError(f"cut_col must satisfy 1 <= cut_col <= {ncol-1}")

    T_tw = [[np.array(A, copy=True) for A in row] for row in T]
    c_left = cut_col - 1
    for r in range(nrow):
        A = T_tw[r][c_left]
        A[..., 2] *= -1.0
        A[..., 3] *= -1.0
        T_tw[r][c_left] = A
    return T_tw


def _twist_horizontal_cut_z(T, cut_row):
    """
    Insert (-1)^z on the horizontal cut, matching the convention in PEPS_Pauli_decoder.
    """
    nrow = len(T)
    ncol = len(T[0])
    if not (1 <= cut_row <= nrow - 1):
        raise ValueError(f"cut_row must satisfy 1 <= cut_row <= {nrow-1}")

    T_tw = [[np.array(A, copy=True) for A in row] for row in T]
    r_up = cut_row - 1
    for c in range(ncol):
        A = T_tw[r_up][c]
        A[:, :, 1, :] *= -1.0
        A[:, :, 3, :] *= -1.0
        T_tw[r_up][c] = A
    return T_tw


def _contract_with_optional_twists(
    T,
    *,
    twist_x=False,
    cut_col=None,
    twist_z=False,
    cut_row=None,
    Nkeep=128,
    Nsweep=1,
):
    T_work = [[np.array(A, copy=True) for A in row] for row in T]
    if twist_x:
        T_work = _twist_vertical_cut_x(T_work, cut_col=cut_col)
    if twist_z:
        T_work = _twist_horizontal_cut_z(T_work, cut_row=cut_row)
    val = contract_finPEPS(T_work, Nkeep=Nkeep, Nsweep=Nsweep)
    return float(np.real_if_close(val))


def choose_default_logical_cuts(active_X: np.ndarray, active_Z: np.ndarray):
    """
    Choose simple middle cuts in the dense rectangular embedding.

    For now:
      - logical X cut: vertical cut near the middle active columns
      - logical Z cut: horizontal cut near the middle active rows
    """
    active_any = (active_X | active_Z).astype(bool)
    rows = np.where(active_any.any(axis=1))[0]
    cols = np.where(active_any.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No active stabilizer support found.")

    # Choose cuts between rows/cols, so convert active extent into an interior cut.
    cmin, cmax = int(cols[0]), int(cols[-1])
    rmin, rmax = int(rows[0]), int(rows[-1])

    cut_col = max(1, min((cmin + cmax + 1) // 2, active_any.shape[1] - 1))
    cut_row = max(1, min((rmin + rmax + 1) // 2, active_any.shape[0] - 1))

    return cut_col, cut_row

def pauli_coset_likelihoods_peps(
    sX: np.ndarray,
    sZ: np.ndarray,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X=None,
    active_Z=None,
    logical_x_cut_col: Optional[int] = None,
    logical_z_cut_row: Optional[int] = None,
    Nkeep: int = 128,
    Nsweep: int = 1,
):
    """
    Same output format as pauli_coset_likelihoods_peps, but for masked surface-code geometry.
    """
    T = build_pauli_peps(
        sX=sX,
        sZ=sZ,
        active_X=active_X,
        active_Z=active_Z,
        W_h=W_h,
        W_v=W_v,
    )

    if logical_x_cut_col is None or logical_z_cut_row is None:
        default_col, default_row = choose_default_logical_cuts(active_X, active_Z)
        if logical_x_cut_col is None:
            logical_x_cut_col = default_col
        if logical_z_cut_row is None:
            logical_z_cut_row = default_row

    plain = _contract_with_optional_twists(T, Nkeep=Nkeep, Nsweep=Nsweep)
    zx = _contract_with_optional_twists(
        T, twist_x=True, cut_col=logical_x_cut_col, Nkeep=Nkeep, Nsweep=Nsweep
    )
    zz = _contract_with_optional_twists(
        T, twist_z=True, cut_row=logical_z_cut_row, Nkeep=Nkeep, Nsweep=Nsweep
    )
    zxz = _contract_with_optional_twists(
        T,
        twist_x=True,
        cut_col=logical_x_cut_col,
        twist_z=True,
        cut_row=logical_z_cut_row,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
    )

    L00 = 0.25 * (plain + zx + zz + zxz)
    L10 = 0.25 * (plain - zx + zz - zxz)
    L01 = 0.25 * (plain + zx - zz - zxz)
    L11 = 0.25 * (plain - zx - zz + zxz)

    return {
        (0, 0): float(np.real_if_close(L00)),
        (1, 0): float(np.real_if_close(L10)),
        (0, 1): float(np.real_if_close(L01)),
        (1, 1): float(np.real_if_close(L11)),
    }


# Helper functions  
def total_likelihood_from_cosets(cosets):
    return sum(cosets.values())

def most_likely_coset(cosets):
    return max(cosets.items(), key=lambda kv: kv[1])


