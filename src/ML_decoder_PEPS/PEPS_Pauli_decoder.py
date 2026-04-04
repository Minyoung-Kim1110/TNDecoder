import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional
from .PEPS import contract_finPEPS
from .weights_PEPS import * 
from ..Surface_code_sampler.stim_sampler import StimSurfaceBatchSample, StimSurfaceSample
from ..metric import * 

# Pauli utils 
PAULI_TO_XZ = {"I": (0, 0),
               "X": (1, 0),
               "Y": (1, 1),
               "Z": (0, 1),}

@dataclass
class PEPSBatchDecodeResult:
    coset_likelihoods: List[Any]
    ml_cosets: List[Any]
    predicted_observable_flips: np.ndarray   # (shots, num_obs)
    logical_failures: np.ndarray             # (shots,)
    logical_error_rate: float

    @property
    def logical_fidelity(self) -> float:
        return 1.0 - self.logical_error_rate
def _validate_edge_shift_masks(
    mask_h_x: Optional[np.ndarray],
    mask_h_z: Optional[np.ndarray],
    mask_v_x: Optional[np.ndarray],
    mask_v_z: Optional[np.ndarray],
    nrow: int,
    ncol: int,
):
    """
    Validate / default per-edge logical/reference shifts.

    Shapes:
      horizontal edges: (nrow+1, ncol)
      vertical edges:   (nrow, ncol+1)
    """
    if mask_h_x is None:
        mask_h_x = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    else:
        mask_h_x = np.asarray(mask_h_x, dtype=np.uint8)

    if mask_h_z is None:
        mask_h_z = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    else:
        mask_h_z = np.asarray(mask_h_z, dtype=np.uint8)

    if mask_v_x is None:
        mask_v_x = np.zeros((nrow, ncol + 1), dtype=np.uint8)
    else:
        mask_v_x = np.asarray(mask_v_x, dtype=np.uint8)

    if mask_v_z is None:
        mask_v_z = np.zeros((nrow, ncol + 1), dtype=np.uint8)
    else:
        mask_v_z = np.asarray(mask_v_z, dtype=np.uint8)

    if mask_h_x.shape != (nrow + 1, ncol):
        raise ValueError(f"mask_h_x must have shape {(nrow + 1, ncol)}")
    if mask_h_z.shape != (nrow + 1, ncol):
        raise ValueError(f"mask_h_z must have shape {(nrow + 1, ncol)}")
    if mask_v_x.shape != (nrow, ncol + 1):
        raise ValueError(f"mask_v_x must have shape {(nrow, ncol + 1)}")
    if mask_v_z.shape != (nrow, ncol + 1):
        raise ValueError(f"mask_v_z must have shape {(nrow, ncol + 1)}")

    return mask_h_x, mask_h_z, mask_v_x, mask_v_z


def _make_logical_shift_masks(
    nrow: int,
    ncol: int,
    *,
    logical_z: bool,
    logical_x: bool,
    logical_x_cut_col: int,
    logical_z_cut_row: int,
):
    r"""
    Build physical logical masks on edge variables.

    Convention:
      - residual logical X is implemented by toggling x-bits on all horizontal
        edges crossing a vertical cut.
      - residual logical Z is implemented by toggling z-bits on all vertical
        edges crossing a horizontal cut.

    These are cocycles of the rectangular embedding, so they preserve the
    local syndrome constraints while changing the logical class.
    """
    mask_h_x = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    mask_h_z = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    mask_v_x = np.zeros((nrow, ncol + 1), dtype=np.uint8)
    mask_v_z = np.zeros((nrow, ncol + 1), dtype=np.uint8)

    if logical_x:
        if not (1 <= logical_x_cut_col <= ncol - 1):
            raise ValueError(f"logical_x_cut_col must satisfy 1 <= cut <= {ncol-1}")
        c_left = logical_x_cut_col - 1
        mask_h_x[:, c_left] = 1

    if logical_z:
        if not (1 <= logical_z_cut_row <= nrow - 1):
            raise ValueError(f"logical_z_cut_row must satisfy 1 <= cut <= {nrow-1}")
        r_up = logical_z_cut_row - 1
        mask_v_z[r_up, :] = 1

    return mask_h_x, mask_h_z, mask_v_x, mask_v_z

#Local tensor construction
def _build_face_tensor(
    row: int,
    col: int,
    sX: np.ndarray,
    sZ: np.ndarray,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X: np.ndarray,
    active_Z: np.ndarray,
    mask_h_x: Optional[np.ndarray] = None,
    mask_h_z: Optional[np.ndarray] = None,
    mask_v_x: Optional[np.ndarray] = None,
    mask_v_z: Optional[np.ndarray] = None,
):
    r"""
    Build one local tensor on the rectangular face grid.

    Local edge-state convention:
      idx = 2*x + z
      x,z in {0,1}

    Syndrome convention:
      - X-check syndrome from Z-parity on incident edges
      - Z-check syndrome from X-parity on incident edges

    The optional masks implement a fixed physical Pauli shift on each edge:
      (x,z) -> (x xor mask_x, z xor mask_z)

    This is the key mechanism for constructing logical cosets physically.
    """
    sx = int(sX[row, col])
    sz = int(sZ[row, col])
    ax = int(active_X[row, col])
    az = int(active_Z[row, col])

    nrow, ncol = sX.shape

    mask_h_x, mask_h_z, mask_v_x, mask_v_z = _validate_edge_shift_masks(
        mask_h_x, mask_h_z, mask_v_x, mask_v_z, nrow, ncol
    )

    Dl = 1 if col == 0 else 4
    Du = 1 if row == 0 else 4
    Dd = 1 if row == nrow - 1 else 4
    Dr = 1 if col == ncol - 1 else 4

    T = np.zeros((Dl, Du, Dd, Dr), dtype=np.float64)

    Wu = W_h[row, col]
    Wd = W_h[row + 1, col]
    Wl = W_v[row, col]
    Wr = W_v[row, col + 1]

    # physical shifts on the four incident edges
    sh_u_x = int(mask_h_x[row, col])
    sh_u_z = int(mask_h_z[row, col])

    sh_d_x = int(mask_h_x[row + 1, col])
    sh_d_z = int(mask_h_z[row + 1, col])

    sh_l_x = int(mask_v_x[row, col])
    sh_l_z = int(mask_v_z[row, col])

    sh_r_x = int(mask_v_x[row, col + 1])
    sh_r_z = int(mask_v_z[row, col + 1])

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
                                    # local syndrome constraints depend on the raw edge variables,
                                    # not on the external logical shift masks
                                    parity_for_Xcheck = zu ^ zd ^ zl ^ zr
                                    parity_for_Zcheck = xu ^ xd ^ xl ^ xr

                                    if ax and parity_for_Xcheck != sx:
                                        continue
                                    if az and parity_for_Zcheck != sz:
                                        continue

                                    # evaluate physical weight in the shifted coset
                                    xl_eff = xl ^ sh_l_x
                                    zl_eff = zl ^ sh_l_z

                                    xu_eff = xu ^ sh_u_x
                                    zu_eff = zu ^ sh_u_z

                                    xd_eff = xd ^ sh_d_x
                                    zd_eff = zd ^ sh_d_z

                                    xr_eff = xr ^ sh_r_x
                                    zr_eff = zr ^ sh_r_z

                                    weight = 1.0
                                    weight *= np.sqrt(Wl[xl_eff, zl_eff]) if Dl != 1 else Wl[xl_eff, zl_eff]
                                    weight *= np.sqrt(Wu[xu_eff, zu_eff]) if Du != 1 else Wu[xu_eff, zu_eff]
                                    weight *= np.sqrt(Wd[xd_eff, zd_eff]) if Dd != 1 else Wd[xd_eff, zd_eff]
                                    weight *= np.sqrt(Wr[xr_eff, zr_eff]) if Dr != 1 else Wr[xr_eff, zr_eff]

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
    mask_h_x: Optional[np.ndarray] = None,
    mask_h_z: Optional[np.ndarray] = None,
    mask_v_x: Optional[np.ndarray] = None,
    mask_v_z: Optional[np.ndarray] = None,
):
    """
    Build PEPS for a rectangular embedding, optionally in a shifted physical coset.

    The mask_* arrays represent a fixed Pauli shift on each edge:
      horizontal edges:
        mask_h_x, mask_h_z with shape (nrow+1, ncol)
      vertical edges:
        mask_v_x, mask_v_z with shape (nrow, ncol+1)
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
    mask_h_x, mask_h_z, mask_v_x, mask_v_z = _validate_edge_shift_masks(
        mask_h_x, mask_h_z, mask_v_x, mask_v_z, nrow, ncol
    )

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
                mask_h_x=mask_h_x,
                mask_h_z=mask_h_z,
                mask_v_x=mask_v_x,
                mask_v_z=mask_v_z,
            )
            row_tensors.append(A)
        T.append(row_tensors)

    return T
# def _build_face_tensor(
#     row: int,
#     col: int, # location of plaquette 
#     sX: np.ndarray,
#     sZ: np.ndarray, # syndrome measurements 
#     W_h: np.ndarray,
#     W_v: np.ndarray, # weights to each edges 
#     active_X: np.ndarray,
#     active_Z: np.ndarray, # masks 
# ):
#     r"""
#     Build one local tensor on a rectangular face grid, but only enforce:
#       - X-check syndrome if active_X[row,col] == 1
#       - Z-check syndrome if active_Z[row,col] == 1

#     Conventions inherited from your existing code:
#       - horizontal edges: W_h[row, col] (top), W_h[row+1, col] (bottom)
#       - vertical edges:   W_v[row, col] (left), W_v[row, col+1] (right)
#       - bond state flattening: idx = 2*x + z
#     """
#     sx = int(sX[row, col])
#     sz = int(sZ[row, col])
#     ax = int(active_X[row, col])
#     az = int(active_Z[row, col])

#     nrow, ncol = sX.shape

#     Dl = 1 if col == 0 else 4
#     Du = 1 if row == 0 else 4
#     Dd = 1 if row == nrow - 1 else 4
#     Dr = 1 if col == ncol - 1 else 4

#     T = np.zeros((Dl, Du, Dd, Dr), dtype=np.float64)

#     Wu = W_h[row, col]
#     Wd = W_h[row + 1, col]
#     Wl = W_v[row, col]
#     Wr = W_v[row, col + 1]

#     def idx_to_xz(idx: int):
#         return idx // 2, idx % 2

#     all_states = [(0, 0), (0, 1), (1, 0), (1, 1)]

#     for l in range(Dl):
#         left_states = all_states if Dl == 1 else [idx_to_xz(l)]
#         for u in range(Du):
#             up_states = all_states if Du == 1 else [idx_to_xz(u)]
#             for d in range(Dd):
#                 down_states = all_states if Dd == 1 else [idx_to_xz(d)]
#                 for r in range(Dr):
#                     right_states = all_states if Dr == 1 else [idx_to_xz(r)]

#                     val = 0.0

#                     for xl, zl in left_states:
#                         for xu, zu in up_states:
#                             for xd, zd in down_states:
#                                 for xr, zr in right_states:
#                                     # Existing convention in your code:
#                                     # X-check syndrome comes from Z-parity on incident edges
#                                     # Z-check syndrome comes from X-parity on incident edges
#                                     parity_for_Xcheck = zu ^ zd ^ zl ^ zr
#                                     parity_for_Zcheck = xu ^ xd ^ xl ^ xr

#                                     if ax and parity_for_Xcheck != sx:
#                                         continue
#                                     if az and parity_for_Zcheck != sz:
#                                         continue

#                                     weight = 1.0
#                                     weight *= np.sqrt(Wl[xl, zl]) if Dl != 1 else Wl[xl, zl]
#                                     weight *= np.sqrt(Wu[xu, zu]) if Du != 1 else Wu[xu, zu]
#                                     weight *= np.sqrt(Wd[xd, zd]) if Dd != 1 else Wd[xd, zd]
#                                     weight *= np.sqrt(Wr[xr, zr]) if Dr != 1 else Wr[xr, zr]

#                                     val += weight

#                     T[l, u, d, r] = val

#     return T

# def build_pauli_peps(
#     sX: np.ndarray,
#     sZ: np.ndarray,
#     W_h: np.ndarray,
#     W_v: np.ndarray,
#     active_X=None,
#     active_Z=None,

# ):
#     """
#     Build PEPS for a rectangular embedding.
#     """
#     sX = np.asarray(sX, dtype=np.uint8)
#     sZ = np.asarray(sZ, dtype=np.uint8)
#     nrow, ncol = sX.shape
    
#     if active_X is None:
#         active_X = np.ones((nrow, ncol), dtype=np.uint8)
#     else:
#         active_X = np.asarray(active_X, dtype=np.uint8)

#     if active_Z is None:
#         active_Z = np.ones((nrow, ncol), dtype=np.uint8)
#     else:
#         active_Z = np.asarray(active_Z, dtype=np.uint8)
    
#     if sX.shape != sZ.shape:
#         raise ValueError("sX and sZ must have the same shape.")
#     if active_X.shape != sX.shape or active_Z.shape != sX.shape:
#         raise ValueError("active_X and active_Z must have the same shape as sX.")

#     W_h, W_v = validate_local_weight_tensor(W_h, W_v, nrow, ncol)

#     T = []
#     for r in range(nrow):
#         row_tensors = []
#         for c in range(ncol):
#             A = _build_face_tensor(
#                 row=r,
#                 col=c,
#                 sX=sX,
#                 sZ=sZ,
#                 active_X=active_X,
#                 active_Z=active_Z,
#                 W_h=W_h,
#                 W_v=W_v,
#             )
#             row_tensors.append(A)
#         T.append(row_tensors)

#     return T

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

def _zero_edge_masks(nrow: int, ncol: int):
    return (
        np.zeros((nrow + 1, ncol), dtype=np.uint8),  # h_x
        np.zeros((nrow + 1, ncol), dtype=np.uint8),  # h_z
        np.zeros((nrow, ncol + 1), dtype=np.uint8),  # v_x
        np.zeros((nrow, ncol + 1), dtype=np.uint8),  # v_z
    )


def _copy_edge_masks(masks):
    return tuple(np.array(x, copy=True, dtype=np.uint8) for x in masks)


def _xor_edge_masks(m1, m2):
    return tuple((np.asarray(a, dtype=np.uint8) ^ np.asarray(b, dtype=np.uint8)).astype(np.uint8)
                 for a, b in zip(m1, m2))


def _edge_mask_weight(masks) -> int:
    return int(sum(np.sum(x) for x in masks))


def _induced_syndrome_from_edge_masks(
    active_X: np.ndarray,
    active_Z: np.ndarray,
    mask_h_x: np.ndarray,
    mask_h_z: np.ndarray,
    mask_v_x: np.ndarray,
    mask_v_z: np.ndarray,
):
    """
    Compute the syndrome induced by a fixed edge-Pauli mask.

    Conventions:
      X-check syndrome comes from Z-parity on incident edges.
      Z-check syndrome comes from X-parity on incident edges.
    """
    active_X = np.asarray(active_X, dtype=np.uint8)
    active_Z = np.asarray(active_Z, dtype=np.uint8)
    nrow, ncol = active_X.shape

    sX = np.zeros((nrow, ncol), dtype=np.uint8)
    sZ = np.zeros((nrow, ncol), dtype=np.uint8)

    for r in range(nrow):
        for c in range(ncol):
            if active_X[r, c]:
                sX[r, c] = (
                    int(mask_h_z[r, c])
                    ^ int(mask_h_z[r + 1, c])
                    ^ int(mask_v_z[r, c])
                    ^ int(mask_v_z[r, c + 1])
                )
            if active_Z[r, c]:
                sZ[r, c] = (
                    int(mask_h_x[r, c])
                    ^ int(mask_h_x[r + 1, c])
                    ^ int(mask_v_x[r, c])
                    ^ int(mask_v_x[r, c + 1])
                )
    return sX, sZ


def _is_zero_syndrome_mask(active_X, active_Z, masks) -> bool:
    sX, sZ = _induced_syndrome_from_edge_masks(active_X, active_Z, *masks)
    return bool(np.sum(sX) == 0 and np.sum(sZ) == 0)


def _symplectic_overlap_masks(m1, m2) -> int:
    """
    Symplectic overlap mod 2 between two edge-Pauli masks.

    overlap = sum_edges (x1*z2 + z1*x2) mod 2
    """
    h_x1, h_z1, v_x1, v_z1 = m1
    h_x2, h_z2, v_x2, v_z2 = m2
    ov = (
        np.sum(h_x1 & h_z2)
        + np.sum(h_z1 & h_x2)
        + np.sum(v_x1 & v_z2)
        + np.sum(v_z1 & v_x2)
    ) % 2
    return int(ov)


def _boundary_line_candidates(active_X: np.ndarray, active_Z: np.ndarray):
    """
    Generate simple boundary/straight-line logical-mask candidates on the doubled lattice.

    We deliberately search over sparse every-other-edge lines, because the real
    code distance is d = (#faces+1)/2, whereas the doubled embedding has size 2d-1.

    Candidates include:
      - horizontal lines on rows r with parity-subsampled columns
      - vertical lines on columns c with parity-subsampled rows

    Each candidate is returned as edge masks (h_x, h_z, v_x, v_z).
    """
    active_X = np.asarray(active_X, dtype=np.uint8)
    active_Z = np.asarray(active_Z, dtype=np.uint8)
    nrow, ncol = active_X.shape

    cands_x = []
    cands_z = []

    # X-type candidates: masks with x-bits only
    # try horizontal sparse lines
    for r in range(nrow + 1):
        for parity in (0, 1):
            mhx, mhz, mvx, mvz = _zero_edge_masks(nrow, ncol)
            mhx[r, parity::2] = 1
            meta = {"kind": "X", "orientation": "h", "row": r, "parity": parity}
            cands_x.append((_copy_edge_masks((mhx, mhz, mvx, mvz)), meta))

    # also try vertical sparse lines
    for c in range(ncol + 1):
        for parity in (0, 1):
            mhx, mhz, mvx, mvz = _zero_edge_masks(nrow, ncol)
            mvx[parity::2, c] = 1
            meta = {"kind": "X", "orientation": "v", "col": c, "parity": parity}
            cands_x.append((_copy_edge_masks((mhx, mhz, mvx, mvz)), meta))

    # Z-type candidates: masks with z-bits only
    for r in range(nrow + 1):
        for parity in (0, 1):
            mhx, mhz, mvx, mvz = _zero_edge_masks(nrow, ncol)
            mhz[r, parity::2] = 1
            meta = {"kind": "Z", "orientation": "h", "row": r, "parity": parity}
            cands_z.append((_copy_edge_masks((mhx, mhz, mvx, mvz)), meta))

    for c in range(ncol + 1):
        for parity in (0, 1):
            mhx, mhz, mvx, mvz = _zero_edge_masks(nrow, ncol)
            mvz[parity::2, c] = 1
            meta = {"kind": "Z", "orientation": "v", "col": c, "parity": parity}
            cands_z.append((_copy_edge_masks((mhx, mhz, mvx, mvz)), meta))

    return cands_x, cands_z
def _auto_choose_logical_masks(active_X: np.ndarray, active_Z: np.ndarray):
    """
    Search for simple zero-syndrome logical representatives adapted to the
    actual masked surface-code embedding.

    Strategy:
      1. generate sparse straight-line candidates,
      2. keep only zero-syndrome candidates,
      3. choose an X-mask and Z-mask with odd symplectic overlap,
      4. among those, prefer larger support.

    Returns:
      logical_X_masks, logical_Z_masks, debug_info
    """
    cands_x, cands_z = _boundary_line_candidates(active_X, active_Z)

    zero_x = []
    zero_z = []

    for masks, meta in cands_x:
        if _is_zero_syndrome_mask(active_X, active_Z, masks):
            zero_x.append((masks, meta))

    for masks, meta in cands_z:
        if _is_zero_syndrome_mask(active_X, active_Z, masks):
            zero_z.append((masks, meta))

    if not zero_x:
        raise RuntimeError("No zero-syndrome X-type logical-mask candidates found.")
    if not zero_z:
        raise RuntimeError("No zero-syndrome Z-type logical-mask candidates found.")

    best = None
    best_score = None

    for mx, meta_x in zero_x:
        wx = _edge_mask_weight(mx)
        for mz, meta_z in zero_z:
            wz = _edge_mask_weight(mz)
            ov = _symplectic_overlap_masks(mx, mz)
            # Require odd overlap so they represent noncommuting logicals.
            if ov != 1:
                continue

            # Prefer shorter masks first, then larger support if tied? 
            # For distance-d logicals on the doubled lattice, true reps should be sparse.
            score = (wx + wz, -abs(wx - wz))
            if best is None or score < best_score:
                best = (mx, mz, meta_x, meta_z)
                best_score = score

    if best is None:
        # fallback: return largest-support zero-syndrome masks even if overlap test failed,
        # but mark it clearly in debug info
        mx, meta_x = max(zero_x, key=lambda t: _edge_mask_weight(t[0]))
        mz, meta_z = max(zero_z, key=lambda t: _edge_mask_weight(t[0]))
        debug_info = {
            "warning": "No odd-overlap pair found; using fallback zero-syndrome masks.",
            "meta_x": meta_x,
            "meta_z": meta_z,
            "weight_x": _edge_mask_weight(mx),
            "weight_z": _edge_mask_weight(mz),
            "overlap": _symplectic_overlap_masks(mx, mz),
            "num_zero_x": len(zero_x),
            "num_zero_z": len(zero_z),
        }
        return mx, mz, debug_info

    mx, mz, meta_x, meta_z = best
    debug_info = {
        "meta_x": meta_x,
        "meta_z": meta_z,
        "weight_x": _edge_mask_weight(mx),
        "weight_z": _edge_mask_weight(mz),
        "overlap": _symplectic_overlap_masks(mx, mz),
        "num_zero_x": len(zero_x),
        "num_zero_z": len(zero_z),
    }
    return mx, mz, debug_info


def pauli_coset_likelihoods_peps(
    sX: np.ndarray,
    sZ: np.ndarray,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X=None,
    active_Z=None,
    logical_x_cut_col: Optional[int] = None,   # unused now, kept for compatibility
    logical_z_cut_row: Optional[int] = None,   # unused now, kept for compatibility
    Nkeep: int = 128,
    Nsweep: int = 1,
    logical_X_masks=None,
    logical_Z_masks=None,
    return_debug_info: bool = False,
):
    r"""
    Compute physical logical-coset likelihoods using logical masks adapted to
    the actual masked surface-code embedding.

    Coset labels:
      (0,0) -> I
      (1,0) -> Z
      (0,1) -> X
      (1,1) -> Y
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

    if logical_X_masks is None or logical_Z_masks is None:
        auto_X, auto_Z, auto_info = _auto_choose_logical_masks(active_X, active_Z)
        if logical_X_masks is None:
            logical_X_masks = auto_X
        if logical_Z_masks is None:
            logical_Z_masks = auto_Z
        debug_info = auto_info
    else:
        debug_info = {
            "meta_x": "user-supplied",
            "meta_z": "user-supplied",
            "weight_x": _edge_mask_weight(logical_X_masks),
            "weight_z": _edge_mask_weight(logical_Z_masks),
            "overlap": _symplectic_overlap_masks(logical_X_masks, logical_Z_masks),
        }

    zero_masks = _zero_edge_masks(nrow, ncol)
    Y_masks = _xor_edge_masks(logical_X_masks, logical_Z_masks)

    def contract_coset(masks):
        mask_h_x, mask_h_z, mask_v_x, mask_v_z = masks
        T = build_pauli_peps(
            sX=sX,
            sZ=sZ,
            W_h=W_h,
            W_v=W_v,
            active_X=active_X,
            active_Z=active_Z,
            mask_h_x=mask_h_x,
            mask_h_z=mask_h_z,
            mask_v_x=mask_v_x,
            mask_v_z=mask_v_z,
        )
        val = contract_finPEPS(T, Nkeep=Nkeep, Nsweep=Nsweep)
        return float(np.real_if_close(val))

    L00 = contract_coset(zero_masks)         # I
    L10 = contract_coset(logical_Z_masks)    # Z
    L01 = contract_coset(logical_X_masks)    # X
    L11 = contract_coset(Y_masks)            # Y

    out = {
        (0, 0): L00,
        (1, 0): L10,
        (0, 1): L01,
        (1, 1): L11,
    }

    if return_debug_info:
        return out, debug_info
    return out
# def pauli_coset_likelihoods_peps(
#     sX: np.ndarray,
#     sZ: np.ndarray,
#     W_h: np.ndarray,
#     W_v: np.ndarray,
#     active_X=None,
#     active_Z=None,
#     logical_x_cut_col: Optional[int] = None,
#     logical_z_cut_row: Optional[int] = None,
#     Nkeep: int = 128,
#     Nsweep: int = 1,
# ):
#     r"""
#     Compute physical logical-coset likelihoods for the rectangular masked surface-code PEPS.

#     Coset labels:
#       (0,0) -> I
#       (1,0) -> Z
#       (0,1) -> X
#       (1,1) -> Y

#     Important:
#       We do NOT use virtual-bond sign twists here.
#       Instead, each logical sector is implemented as a fixed physical shift mask
#       on the edge variables before PEPS contraction.
#     """
#     sX = np.asarray(sX, dtype=np.uint8)
#     sZ = np.asarray(sZ, dtype=np.uint8)
#     nrow, ncol = sX.shape

#     if active_X is None:
#         active_X = np.ones((nrow, ncol), dtype=np.uint8)
#     else:
#         active_X = np.asarray(active_X, dtype=np.uint8)

#     if active_Z is None:
#         active_Z = np.ones((nrow, ncol), dtype=np.uint8)
#     else:
#         active_Z = np.asarray(active_Z, dtype=np.uint8)

#     if logical_x_cut_col is None or logical_z_cut_row is None:
#         default_col = max(1, min(ncol // 2, ncol - 1))
#         default_row = max(1, min(nrow // 2, nrow - 1))
#         if logical_x_cut_col is None:
#             logical_x_cut_col = default_col
#         if logical_z_cut_row is None:
#             logical_z_cut_row = default_row

#     def contract_coset(logical_z: bool, logical_x: bool) -> float:
#         mask_h_x, mask_h_z, mask_v_x, mask_v_z = _make_logical_shift_masks(
#             nrow=nrow,
#             ncol=ncol,
#             logical_z=logical_z,
#             logical_x=logical_x,
#             logical_x_cut_col=logical_x_cut_col,
#             logical_z_cut_row=logical_z_cut_row,
#         )

#         T = build_pauli_peps(
#             sX=sX,
#             sZ=sZ,
#             W_h=W_h,
#             W_v=W_v,
#             active_X=active_X,
#             active_Z=active_Z,
#             mask_h_x=mask_h_x,
#             mask_h_z=mask_h_z,
#             mask_v_x=mask_v_x,
#             mask_v_z=mask_v_z,
#         )
#         val = contract_finPEPS(T, Nkeep=Nkeep, Nsweep=Nsweep)
#         return float(np.real_if_close(val))

#     L00 = contract_coset(logical_z=False, logical_x=False)  # I
#     L10 = contract_coset(logical_z=True,  logical_x=False)  # Z
#     L01 = contract_coset(logical_z=False, logical_x=True)   # X
#     L11 = contract_coset(logical_z=True,  logical_x=True)   # Y

#     return {
#         (0, 0): L00,
#         (1, 0): L10,
#         (0, 1): L01,
#         (1, 1): L11,
#     }
# def pauli_coset_likelihoods_peps(
#     sX: np.ndarray,
#     sZ: np.ndarray,
#     W_h: np.ndarray,
#     W_v: np.ndarray,
#     active_X=None,
#     active_Z=None,
#     logical_x_cut_col: Optional[int] = None,
#     logical_z_cut_row: Optional[int] = None,
#     Nkeep: int = 128,
#     Nsweep: int = 1,
# ):
#     """
#     Same output format as pauli_coset_likelihoods_peps, but for masked surface-code geometry.
#     """
#     T = build_pauli_peps(
#         sX=sX,
#         sZ=sZ,
#         active_X=active_X,
#         active_Z=active_Z,
#         W_h=W_h,
#         W_v=W_v,
#     )
#     nrow, ncol = np.shape(sX)
#     if logical_x_cut_col is None or logical_z_cut_row is None:
#         default_col = ncol//2 
#         default_row = nrow //2 
#         # default_col, default_row = choose_default_logical_cuts(active_X, active_Z)
#         if logical_x_cut_col is None:
#             logical_x_cut_col = default_col
#         if logical_z_cut_row is None:
#             logical_z_cut_row = default_row

#     plain = _contract_with_optional_twists(T, Nkeep=Nkeep, Nsweep=Nsweep)
#     zx = _contract_with_optional_twists(
#         T, twist_x=True, cut_col=logical_x_cut_col, Nkeep=Nkeep, Nsweep=Nsweep
#     )
#     zz = _contract_with_optional_twists(
#         T, twist_z=True, cut_row=logical_z_cut_row, Nkeep=Nkeep, Nsweep=Nsweep
#     )
#     zxz = _contract_with_optional_twists(
#         T,
#         twist_x=True,
#         cut_col=logical_x_cut_col,
#         twist_z=True,
#         cut_row=logical_z_cut_row,
#         Nkeep=Nkeep,
#         Nsweep=Nsweep,
#     )

#     L00 = 0.25 * (plain + zx + zz + zxz)
#     L10 = 0.25 * (plain - zx + zz - zxz)
#     L01 = 0.25 * (plain + zx - zz - zxz)
#     L11 = 0.25 * (plain - zx - zz + zxz)

#     return {
#         (0, 0): float(np.real_if_close(L00)),
#         (1, 0): float(np.real_if_close(L10)),
#         (0, 1): float(np.real_if_close(L01)),
#         (1, 1): float(np.real_if_close(L11)),
#     }


# ---------------------------------------------------------------------------
# PEPS logical-coset handling
# ---------------------------------------------------------------------------

def _logical_bits_from_ml_coset(ml_coset):
    """
    Convert PEPS most_likely_coset(...) output into logical bits (z_log, x_log).

    Supported conventions:
      1. ((z_log, x_log), score)
      2. (z_log, x_log)
      3. integer 0,1,2,3
      4. string labels "I","Z","X","Y"
    """

    # Case 0: output like ((0,0), score)
    if (
        isinstance(ml_coset, (tuple, list))
        and len(ml_coset) == 2
        and isinstance(ml_coset[0], (tuple, list))
        and len(ml_coset[0]) == 2
    ):
        a, b = ml_coset[0]
        if a in (0, 1) and b in (0, 1):
            return int(a), int(b)

    # Case 1: tuple/list of two bits
    if isinstance(ml_coset, (tuple, list)) and len(ml_coset) == 2:
        a, b = ml_coset
        if a in (0, 1) and b in (0, 1):
            return int(a), int(b)

    # Case 2: integer label
    if isinstance(ml_coset, (int, np.integer)):
        lut = {
            0: (0, 0),
            1: (1, 0),
            2: (0, 1),
            3: (1, 1),
        }
        if int(ml_coset) in lut:
            return lut[int(ml_coset)]

    # Case 3: string label
    s = str(ml_coset).strip().upper()
    lut = {
        "I": (0, 0),
        "Z": (1, 0),
        "X": (0, 1),
        "Y": (1, 1),
        "II": (0, 0),
        "IZ": (1, 0),
        "IX": (0, 1),
        "IY": (1, 1),
        "L0": (0, 0),
        "LZ": (1, 0),
        "LX": (0, 1),
        "LY": (1, 1),
    }
    if s in lut:
        return lut[s]

    raise ValueError(
        f"Could not infer logical bits from most_likely_coset output: {ml_coset!r}"
    )

def predicted_observable_flip_from_ml_coset(
    ml_coset,
    memory_basis: str,
    num_obs: int = 1,
):
    """
    Convert PEPS ML logical coset to Stim observable flip prediction.

    Convention:
      ml_coset -> (z_log, x_log)
      residual logical operator = Z_L^z_log X_L^x_log
    """
    if num_obs != 1:
        raise NotImplementedError(
            "Currently assumes one logical observable."
        )

    z_log, x_log = _logical_bits_from_ml_coset(ml_coset)

    if memory_basis == "x":
        # measured logical X flips if residual contains Z
        bit = z_log
    elif memory_basis == "z":
        # measured logical Z flips if residual contains X
        bit = x_log
    else:
        raise ValueError("memory_basis must be 'x' or 'z'.")

    return np.array([bit], dtype=np.uint8)

# ---------------------------------------------------------------------------
# PEPS decoding on an exact batch
# ---------------------------------------------------------------------------

def peps_coset_likelihoods_for_batch(
    batch: StimSurfaceBatchSample,
    *,
    p: float,
    Nkeep: int = 128,
    Nsweep: int = 1,
    W_h: Optional[np.ndarray] = None,
    W_v: Optional[np.ndarray] = None,
) -> List[Any]:
    """
    Compute PEPS logical-coset likelihoods shot-by-shot for a batched Stim sample.
    """
    nrow, ncol = batch.sX[0].shape
    if W_h is None or W_v is None:
        W_h_default, W_v_default = depolarizing_weights(nrow, ncol,p=p)
        if W_h is None:
            W_h = W_h_default
        if W_v is None:
            W_v = W_v_default

    shots = batch.shots
    cosets: List[Any] = []
    for k in range(shots):
        cosets_k = pauli_coset_likelihoods_peps(
            sX=batch.sX[k],
            sZ=batch.sZ[k],
            active_X=batch.active_X[k],
            active_Z=batch.active_Z[k],
            W_h=W_h,
            W_v=W_v,
            Nkeep=Nkeep,
            Nsweep=Nsweep,
        )
        cosets.append(cosets_k)
    return cosets

def decode_batch_with_peps(
    batch: StimSurfaceBatchSample,
    *,
    p: float,
    memory_basis: str,
    Nkeep: int = 128,
    Nsweep: int = 1,
    W_h=None,
    W_v=None,
    debug_failures=False, 
):
    coset_likelihoods = peps_coset_likelihoods_for_batch(
        batch=batch,
        p=p,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
        W_h=W_h,
        W_v=W_v,
    )

    ml_cosets = [most_likely_coset(c) for c in coset_likelihoods]

    num_obs = int(batch.observable_flips.shape[1])

    predicted_obs = np.stack(
        [
            predicted_observable_flip_from_ml_coset(
                ml_coset=c,
                memory_basis=memory_basis,
                num_obs=num_obs,
            )
            for c in ml_cosets
        ],
        axis=0,
    ).astype(np.uint8)

    failures = logical_failures_from_predictions(
        actual_observable_flips=batch.observable_flips,
        predicted_observable_flips=predicted_obs,
    )

    # ==========================
    # DEBUG PRINTS
    # ==========================

    if debug_failures:

        for k in range(batch.shots):

            if failures[k]:

                print("\n==========================")
                print("PEPS FAILURE")
                print("shot =", k)

                print("actual observable =",
                      batch.observable_flips[k])

                print("predicted observable =",
                      predicted_obs[k])

                print("ml coset =",
                      ml_cosets[k])

                print("coset likelihoods =")

                cos = coset_likelihoods[k]

                # pretty print likelihoods
                if isinstance(cos, dict):

                    for key,val in cos.items():

                        print("   ", key, ":", val)

                else:

                    print(cos)

                # optional syndrome inspection
                print("sX:")
                print(batch.sX[k])

                print("sZ:")
                print(batch.sZ[k])
                
                print("detector_bits:")
                print(batch.detector_bits[k])

                print("observable_flips:")
                print(batch.observable_flips[k])
                print("nonzero detector coordinates:")
                for det_id in np.flatnonzero(batch.detector_bits[k]):
                    print(det_id, batch.detector_coords[det_id])
                print("==========================")

    return PEPSBatchDecodeResult(
        coset_likelihoods=coset_likelihoods,
        ml_cosets=ml_cosets,
        predicted_observable_flips=predicted_obs,
        logical_failures=failures,
        logical_error_rate=float(np.mean(failures)),
    )
# def decode_batch_with_peps(
#     batch: StimSurfaceBatchSample,
#     *,
#     p: float,
#     memory_basis: str,
#     Nkeep: int = 128,
#     Nsweep: int = 1,
#     W_h: Optional[np.ndarray] = None,
#     W_v: Optional[np.ndarray] = None,
# ) -> PEPSBatchDecodeResult:
#     """
#     Decode the exact same Stim batch with the PEPS ML decoder.
#     """
#     coset_likelihoods = peps_coset_likelihoods_for_batch(
#         batch=batch,
#         p=p,
#         Nkeep=Nkeep,
#         Nsweep=Nsweep,
#         W_h=W_h,
#         W_v=W_v,
#     )

#     ml_cosets = [most_likely_coset(c) for c in coset_likelihoods]
#     num_obs = int(batch.observable_flips.shape[1])
#     predicted_obs = np.stack(
#         [
#             predicted_observable_flip_from_ml_coset(
#                 ml_coset=c,
#                 memory_basis=memory_basis,
#                 num_obs=num_obs,
#             )
#             for c in ml_cosets
#         ],
#         axis=0,
#     ).astype(np.uint8)

#     failures = logical_failures_from_predictions(
#         actual_observable_flips=batch.observable_flips,
#         predicted_observable_flips=predicted_obs,
#     )

#     return PEPSBatchDecodeResult(
#         coset_likelihoods=coset_likelihoods,
#         ml_cosets=ml_cosets,
#         predicted_observable_flips=predicted_obs,
#         logical_failures=failures,
#         logical_error_rate=float(np.mean(failures)),
#     )

# Helper functions  
def total_likelihood_from_cosets(cosets):
    return sum(cosets.values())

def most_likely_coset(cosets):
    return max(cosets.items(), key=lambda kv: kv[1])


