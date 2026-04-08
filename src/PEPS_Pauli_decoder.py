import numpy as np
from typing import Dict, Tuple, Optional
from .PEPS import contract_finPEPS
from .weights_PEPS import * 


# Pauli utils 
PAULI_TO_XZ = {"I": (0, 0),
               "X": (1, 0),
               "Y": (1, 1),
               "Z": (0, 1),}


# Local tensor construction 

def _build_face_tensor(
    row: int,
    col: int,
    sX: np.ndarray,
    sZ: np.ndarray,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X: np.ndarray,
    active_Z: np.ndarray,
):
    r"""
    Build one local tensor for the dense 9×9 CSS-syndrome grid.

    The grid mixes two node types:

    DATA QUBIT NODE  (active_X == 0 and active_Z == 0)
        A Pauli error P on the qubit must be seen consistently by every
        adjacent check.  Implemented as a DELTA tensor:

            T[l, u, d, r] = W_data[x_P, z_P]
                            if all interior (non-boundary) bond indices
                            encode the same Pauli P,  else 0.

        Full weight W placed here; check tensors carry no weight.

    CHECK NODE  (active_X == 1 or active_Z == 1)
        Enforces the CSS parity constraint.  No weight (lives in data
        qubit tensors).  Boundary bonds (dim == 1) are pinned to the
        "no error" state (x=0, z=0) — there is no qubit at the code edge.

    Bond index convention:  idx = 2*x + z
        0 → I (x=0, z=0)
        1 → Z (x=0, z=1)
        2 → X (x=1, z=0)
        3 → Y (x=1, z=1)

    Syndrome parity convention:
        X-check syndrome sX  ←  Z-parity of incident bonds  (zu^zd^zl^zr)
        Z-check syndrome sZ  ←  X-parity of incident bonds  (xu^xd^xl^xr)
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

    def idx_to_xz(idx: int):
        return idx // 2, idx % 2

    if not ax and not az:
        # ----------------------------------------------------------------
        # DATA QUBIT NODE — delta tensor
        #
        # All interior bonds must carry the same Pauli error (x, z).
        # T[l, u, d, r] = W_data[x, z]  when all interior indices agree,
        #                = 0             otherwise.
        #
        # Boundary bonds (dim == 1) have index 0 and carry no information;
        # they are excluded from the delta condition.
        #
        # VACANCY positions (col-odd AND row-odd in the dense grid) are
        # not real qubits in the unrotated surface code.  The dense grid
        # includes them because both odd-x and odd-y coordinates arise
        # from distinct check types.  Treating vacancies as real qubits
        # would create spurious connections between checks and add fake
        # probability mass.  Fix: force identity (W[0,0]=1, rest=0) so
        # that every bond at a vacancy is pinned to index 0 (I), making
        # the vacancy transparent to the surrounding checks.
        #
        # Convention valid for unrotated surface code where
        # x_to_col and y_to_row are identity maps (all integers 0..2d-2
        # appear in check coordinates), so col = physical_x and
        # row = physical_y.
        # ----------------------------------------------------------------
        is_vacancy = (row % 2 == 1) and (col % 2 == 1)
        if is_vacancy:
            W_data = np.zeros((2, 2), dtype=np.float64)
            W_data[0, 0] = 1.0   # identity only — no qubit at this site
        else:
            W_data = W_h[row, col]

        for l in range(Dl):
            for u in range(Du):
                for d in range(Dd):
                    for r in range(Dr):
                        interior = []
                        if Dl > 1: interior.append(l)
                        if Du > 1: interior.append(u)
                        if Dd > 1: interior.append(d)
                        if Dr > 1: interior.append(r)

                        if not interior:
                            T[l, u, d, r] = 1.0
                            continue

                        # Delta condition
                        if len(set(interior)) > 1:
                            continue  # mismatched Paulis → 0

                        xq, zq = idx_to_xz(interior[0])
                        T[l, u, d, r] = W_data[xq, zq]

    else:
        # ----------------------------------------------------------------
        # ACTIVE CHECK NODE — syndrome constraint, no weight
        #
        # Boundary bonds (dim == 1) are pinned to (x=0, z=0):
        # there is no data qubit outside the code boundary.
        # Interior bonds enforce the CSS parity condition.
        # Weight is carried entirely by the data qubit tensors.
        # ----------------------------------------------------------------
        for l in range(Dl):
            for u in range(Du):
                for d in range(Dd):
                    for r in range(Dr):
                        xl, zl = (0, 0) if Dl == 1 else idx_to_xz(l)
                        xu, zu = (0, 0) if Du == 1 else idx_to_xz(u)
                        xd, zd = (0, 0) if Dd == 1 else idx_to_xz(d)
                        xr, zr = (0, 0) if Dr == 1 else idx_to_xz(r)

                        parity_for_Xcheck = zu ^ zd ^ zl ^ zr
                        parity_for_Zcheck = xu ^ xd ^ xl ^ xr

                        if ax and parity_for_Xcheck != sx:
                            continue
                        if az and parity_for_Zcheck != sz:
                            continue

                        T[l, u, d, r] = 1.0

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


