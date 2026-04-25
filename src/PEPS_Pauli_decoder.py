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
        # Check-type site with no active check → transparent tensor.
        #
        # Coordinate parity in the dense (2d-1)×(2d-1) grid:
        #   col%2==1, row%2==0  →  X-check site
        #   col%2==0, row%2==1  →  Z-check site
        #   otherwise           →  data qubit site
        #
        # When a check-type position has active=0 the syndrome was not
        # measured (e.g. Z-checks in memory_x rounds=1).  The correct ML
        # treatment is to marginalise over both possible syndromes, which
        # is achieved by a transparent tensor: T=1 everywhere.  This lets
        # the x- or z-parity of neighbouring bonds contract freely so the
        # unobserved check contributes no constraint.
        # ----------------------------------------------------------------
        is_x_check_site = (col % 2 == 1) and (row % 2 == 0)
        is_z_check_site = (col % 2 == 0) and (row % 2 == 1)
        if is_x_check_site or is_z_check_site:
            T[:] = 1.0
            return T

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
        # All non-check positions are real data qubits.
        #
        # The unrotated surface code has 2d²-2d+1 data qubits, NOT d².
        # In the dense (2d-1)×(2d-1) grid:
        #   - (row_even, col_even): data qubits at (x_even, y_even)  — 5×5=25 for d=5
        #   - (row_odd,  col_odd):  data qubits at (x_odd,  y_odd)   — 4×4=16 for d=5
        # Both families are real qubits with errors and weights.
        # ----------------------------------------------------------------
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
#
# Stim unrotated surface code (memory_x):
#   Logical X = VERTICAL column of X operators (fixed stim_x, varying stim_y)
#   Logical Z = HORIZONTAL row of Z operators (fixed stim_y, varying stim_x)
#
# To detect a VERTICAL logical X string we need a HORIZONTAL seam:
#   insert (-1)^x on DOWN bonds crossing the seam (row cut_row-1 → cut_row).
# To detect a HORIZONTAL logical Z string we need a VERTICAL seam:
#   insert (-1)^z on RIGHT bonds crossing the seam (col cut_col-1 → cut_col).
#
# Tensor axis order: (left, up, down, right)
# Bond index encoding: idx = 2*x_pauli + z_pauli  →  I=0, Z=1, X=2, Y=3

def _twist_horizontal_cut_x(T, cut_row):
    """Insert (-1)^x on DOWN bonds at the horizontal seam below row cut_row-1.

    Only twists data qubit nodes (skip X-check sites at odd cols in an even row).
    In the (2d-1)x(2d-1) PEPS grid, data qubits in row r_up are at columns
    where c%2 == r_up%2.  X-check sites (col%2==1, row%2==0) would otherwise
    propagate the sign to neighbouring diagonal data qubits, computing the
    character for the wrong logical representative.
    """
    nrow = len(T)
    ncol = len(T[0])
    if not (1 <= cut_row <= nrow - 1):
        raise ValueError(f"cut_row must satisfy 1 <= cut_row <= {nrow-1}")
    T_tw = [[np.array(A, copy=True) for A in row] for row in T]
    r_up = cut_row - 1
    for c in range(ncol):
        if c % 2 != r_up % 2:  # skip check sites
            continue
        A = T_tw[r_up][c]
        A[:, :, 2, :] *= -1.0  # DOWN bond X (x=1, z=0)
        A[:, :, 3, :] *= -1.0  # DOWN bond Y (x=1, z=1)
        T_tw[r_up][c] = A
    return T_tw


def _twist_vertical_cut_z(T, cut_col):
    """Insert (-1)^z on RIGHT bonds at the vertical seam right of col cut_col-1.

    Only twists data qubit nodes (skip Z-check sites at odd rows in an even col).
    In the (2d-1)x(2d-1) PEPS grid, data qubits in col c_left are at rows where
    r%2 == c_left%2.  Z-check sites (col%2==0, row%2==1) would otherwise propagate
    the sign to neighbouring diagonal data qubits, computing the character for the
    wrong logical representative.
    """
    nrow = len(T)
    ncol = len(T[0])
    if not (1 <= cut_col <= ncol - 1):
        raise ValueError(f"cut_col must satisfy 1 <= cut_col <= {ncol-1}")
    T_tw = [[np.array(A, copy=True) for A in row] for row in T]
    c_left = cut_col - 1
    for r in range(nrow):
        if r % 2 != c_left % 2:  # skip check sites
            continue
        A = T_tw[r][c_left]
        A[..., 1] *= -1.0  # RIGHT bond Z (x=0, z=1)
        A[..., 3] *= -1.0  # RIGHT bond Y (x=1, z=1)
        T_tw[r][c_left] = A
    return T_tw


def _contract_with_optional_twists(
    T,
    *,
    twist_x=False,
    cut_x=None,   # row index: horizontal seam for vertical X-string detection
    twist_z=False,
    cut_z=None,   # col index: vertical seam for horizontal Z-string detection
    Nkeep=128,
    Nsweep=1,
):
    T_work = [[np.array(A, copy=True) for A in row] for row in T]
    if twist_x:
        T_work = _twist_horizontal_cut_x(T_work, cut_row=cut_x)
    if twist_z:
        T_work = _twist_vertical_cut_z(T_work, cut_col=cut_z)
    val = contract_finPEPS(T_work, Nkeep=Nkeep, Nsweep=Nsweep)
    return float(np.real_if_close(val))


def choose_default_logical_cuts(active_X: np.ndarray, active_Z: np.ndarray):
    """
    Choose boundary-adjacent seam cuts for the dense rectangular PEPS.

    For the unrotated surface code (memory_x) in the (2d-1)×(2d-1) PEPS grid:
      - Logical Z = HORIZONTAL row of Z at row=0 (stim_x=0 column in stim coords).
        Detected by a VERTICAL seam (cut_col=1): Z/Y-parity of RIGHT bonds of col=0.
      - Logical X = VERTICAL column of X at col=0 (stim_y=0 row in stim coords).
        Detected by a HORIZONTAL seam (cut_row=1): X/Y-parity of DOWN bonds of row=0.

    For open-boundary surface codes the seam must be placed adjacent to the
    boundary where the logical operator is supported (cut=1), NOT at the middle.
    A middle seam (cut≈d) is only valid for torus topology.
    """
    nrow, ncol = active_X.shape
    cut_col = 1
    cut_row = 1
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

    # logical_x_cut_col is used as the row index for the horizontal x-seam
    # logical_z_cut_row is used as the col index for the vertical z-seam
    # (values coincide for square symmetric codes; naming preserved for API compat)
    plain = _contract_with_optional_twists(T, Nkeep=Nkeep, Nsweep=Nsweep)
    zx = _contract_with_optional_twists(
        T, twist_x=True, cut_x=logical_x_cut_col, Nkeep=Nkeep, Nsweep=Nsweep
    )
    zz = _contract_with_optional_twists(
        T, twist_z=True, cut_z=logical_z_cut_row, Nkeep=Nkeep, Nsweep=Nsweep
    )
    zxz = _contract_with_optional_twists(
        T,
        twist_x=True,
        cut_x=logical_x_cut_col,
        twist_z=True,
        cut_z=logical_z_cut_row,
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


