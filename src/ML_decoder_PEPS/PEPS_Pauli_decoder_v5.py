import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from .PEPS import contract_finPEPS
from .weights_PEPS import depolarizing_weights, validate_local_weight_tensor
from .PEPS_Pauli_decoder import (
    PEPSBatchDecodeResult,
    _validate_edge_shift_masks,
    _make_logical_shift_masks,
    pauli_coset_likelihoods_peps
)
from ..metric import logical_failures_from_predictions


# ============================================================
# Conventions
# ============================================================
#
# Coset key = (z_log, x_log)
#
#   (0,0) -> I
#   (1,0) -> Z
#   (0,1) -> X
#   (1,1) -> Y
#
# Same convention as your sampler/debug pipeline:
#   logical_bits = [z_log, x_log]
#
# The important change from the old implementation is:
#
#   old:
#     sum over raw variables satisfying measured syndrome s,
#     evaluate weights in shifted coset
#
#   new:
#     choose one fixed reference mask R(s) with boundary R = s,
#     sum only over zero-syndrome cycle variables C,
#     evaluate weights on C xor R(s) xor L_ab
#
# This avoids the exact change-of-variables degeneracy.
# ============================================================


CosetKey = Tuple[int, int]
MaskTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# (mask_h_x, mask_h_z, mask_v_x, mask_v_z)


def choose_default_logical_cuts(nrow: int, ncol: int) -> Tuple[int, int]:
    """
    Choose canonical interior cuts.

    Returns
    -------
    logical_x_cut_col : int
    logical_z_cut_row : int
    """
    if nrow < 2 or ncol < 2:
        raise ValueError(f"Need nrow,ncol >= 2, got {(nrow, ncol)}")

    logical_x_cut_col = max(1, min(ncol - 1, ncol // 2))
    logical_z_cut_row = max(1, min(nrow - 1, nrow // 2))
    return logical_x_cut_col, logical_z_cut_row


def _zero_masks(nrow: int, ncol: int) -> MaskTuple:
    return (
        np.zeros((nrow + 1, ncol), dtype=np.uint8),  # h_x
        np.zeros((nrow + 1, ncol), dtype=np.uint8),  # h_z
        np.zeros((nrow, ncol + 1), dtype=np.uint8),  # v_x
        np.zeros((nrow, ncol + 1), dtype=np.uint8),  # v_z
    )


def _xor_masks(a: MaskTuple, b: MaskTuple) -> MaskTuple:
    return tuple((np.asarray(x, dtype=np.uint8) ^ np.asarray(y, dtype=np.uint8)).astype(np.uint8)
                 for x, y in zip(a, b))  # type: ignore


def _compose_masks(*masks: MaskTuple) -> MaskTuple:
    """
    XOR-compose any number of edge masks.
    """
    out = None
    for m in masks:
        if out is None:
            out = tuple(np.array(x, copy=True, dtype=np.uint8) for x in m)
        else:
            out = _xor_masks(out, m)
    if out is None:
        raise ValueError("Need at least one mask to compose.")
    return out


def _induced_syndrome_from_edge_masks(
    active_X: np.ndarray,
    active_Z: np.ndarray,
    mask_h_x: np.ndarray,
    mask_h_z: np.ndarray,
    mask_v_x: np.ndarray,
    mask_v_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute induced syndrome of a fixed edge-Pauli mask.

    Conventions:
      X-check syndrome <- Z-parity on incident edges
      Z-check syndrome <- X-parity on incident edges
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


def _mask_weight(mask: MaskTuple) -> int:
    return int(sum(np.asarray(x, dtype=np.uint8).sum() for x in mask))


def _build_reference_mask_from_syndrome(
    sX: np.ndarray,
    sZ: np.ndarray,
    active_X: np.ndarray,
    active_Z: np.ndarray,
) -> MaskTuple:
    """
    Construct one fixed reference error mask R(s) satisfying:
        induced_syndrome(R) = (sX, sZ)

    We solve the Z-part and X-part independently.

    For the Z-part (which produces sX):
      choose all horizontal z-edges = 0
      solve vertical z-edges row-by-row:
          v_z[r,0] = 0
          v_z[r,c+1] = sX[r,c] xor v_z[r,c]

    For the X-part (which produces sZ):
      choose all vertical x-edges = 0
      solve horizontal x-edges column-by-column:
          h_x[0,c] = 0
          h_x[r+1,c] = sZ[r,c] xor h_x[r,c]

    This gives one deterministic representative.
    """
    sX = np.asarray(sX, dtype=np.uint8)
    sZ = np.asarray(sZ, dtype=np.uint8)
    active_X = np.asarray(active_X, dtype=np.uint8)
    active_Z = np.asarray(active_Z, dtype=np.uint8)

    if sX.shape != sZ.shape or sX.shape != active_X.shape or sX.shape != active_Z.shape:
        raise ValueError("sX, sZ, active_X, active_Z must all have the same shape")

    nrow, ncol = sX.shape

    # X-part of reference (to reproduce sZ)
    ref_h_x = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    ref_v_x = np.zeros((nrow, ncol + 1), dtype=np.uint8)

    # Z-part of reference (to reproduce sX)
    ref_h_z = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    ref_v_z = np.zeros((nrow, ncol + 1), dtype=np.uint8)

    # Solve for Z-part from sX using only vertical z-edges
    for r in range(nrow):
        ref_v_z[r, 0] = 0
        for c in range(ncol):
            target = int(sX[r, c]) if active_X[r, c] else 0
            ref_v_z[r, c + 1] = ref_v_z[r, c] ^ target

    # Solve for X-part from sZ using only horizontal x-edges
    for c in range(ncol):
        ref_h_x[0, c] = 0
        for r in range(nrow):
            target = int(sZ[r, c]) if active_Z[r, c] else 0
            ref_h_x[r + 1, c] = ref_h_x[r, c] ^ target

    ref = (ref_h_x, ref_h_z, ref_v_x, ref_v_z)

    chk_sX, chk_sZ = _induced_syndrome_from_edge_masks(active_X, active_Z, *ref)
    if not np.array_equal(chk_sX, sX):
        raise RuntimeError("Reference mask construction failed for sX.")
    if not np.array_equal(chk_sZ, sZ):
        raise RuntimeError("Reference mask construction failed for sZ.")

    return ref


def _build_face_tensor_zero_cycles(
    row: int,
    col: int,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X: np.ndarray,
    active_Z: np.ndarray,
    total_mask_h_x: np.ndarray,
    total_mask_h_z: np.ndarray,
    total_mask_v_x: np.ndarray,
    total_mask_v_z: np.ndarray,
):
    r"""
    Local tensor for zero-syndrome cycle variables.

    Raw edge variables in the tensor represent a cycle C with:
      X-check parity = 0
      Z-check parity = 0

    The physical edge value used in the weight is:
      edge_eff = raw_cycle xor total_mask
    where total_mask = reference_mask xor logical_mask.

    Local edge-state convention:
      idx = 2*x + z
      x,z in {0,1}
    """
    ax = int(active_X[row, col])
    az = int(active_Z[row, col])

    nrow, ncol = active_X.shape

    Dl = 1 if col == 0 else 4
    Du = 1 if row == 0 else 4
    Dd = 1 if row == nrow - 1 else 4
    Dr = 1 if col == ncol - 1 else 4

    T = np.zeros((Dl, Du, Dd, Dr), dtype=np.float64)

    Wu = W_h[row, col]
    Wd = W_h[row + 1, col]
    Wl = W_v[row, col]
    Wr = W_v[row, col + 1]

    # total fixed shift on the four incident edges
    sh_u_x = int(total_mask_h_x[row, col])
    sh_u_z = int(total_mask_h_z[row, col])

    sh_d_x = int(total_mask_h_x[row + 1, col])
    sh_d_z = int(total_mask_h_z[row + 1, col])

    sh_l_x = int(total_mask_v_x[row, col])
    sh_l_z = int(total_mask_v_z[row, col])

    sh_r_x = int(total_mask_v_x[row, col + 1])
    sh_r_z = int(total_mask_v_z[row, col + 1])

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
                                    # zero-syndrome constraint on raw cycle variables
                                    parity_for_Xcheck = zu ^ zd ^ zl ^ zr
                                    parity_for_Zcheck = xu ^ xd ^ xl ^ xr

                                    if ax and parity_for_Xcheck != 0:
                                        continue
                                    if az and parity_for_Zcheck != 0:
                                        continue

                                    # evaluate physical weight on cycle xor reference xor logical
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


def build_pauli_peps_zero_cycles(
    *,
    sX: np.ndarray,
    sZ: np.ndarray,
    W_h: np.ndarray,
    W_v: np.ndarray,
    active_X: Optional[np.ndarray] = None,
    active_Z: Optional[np.ndarray] = None,
    ref_h_x: Optional[np.ndarray] = None,
    ref_h_z: Optional[np.ndarray] = None,
    ref_v_x: Optional[np.ndarray] = None,
    ref_v_z: Optional[np.ndarray] = None,
    log_h_x: Optional[np.ndarray] = None,
    log_h_z: Optional[np.ndarray] = None,
    log_v_x: Optional[np.ndarray] = None,
    log_v_z: Optional[np.ndarray] = None,
):
    """
    Build PEPS whose raw variables are zero-syndrome cycles and whose physical
    weights are evaluated in the coset:
        cycle xor reference(s) xor logical
    """
    sX = np.asarray(sX, dtype=np.uint8)
    sZ = np.asarray(sZ, dtype=np.uint8)
    if sX.shape != sZ.shape:
        raise ValueError("sX and sZ must have the same shape.")

    nrow, ncol = sX.shape
    W_h, W_v = validate_local_weight_tensor(W_h, W_v, nrow, ncol)

    if active_X is None:
        active_X = np.ones((nrow, ncol), dtype=np.uint8)
    else:
        active_X = np.asarray(active_X, dtype=np.uint8)

    if active_Z is None:
        active_Z = np.ones((nrow, ncol), dtype=np.uint8)
    else:
        active_Z = np.asarray(active_Z, dtype=np.uint8)

    if active_X.shape != (nrow, ncol) or active_Z.shape != (nrow, ncol):
        raise ValueError("active_X and active_Z must have the same shape as sX.")

    # reference mask from measured syndrome
    if ref_h_x is None or ref_h_z is None or ref_v_x is None or ref_v_z is None:
        ref_h_x, ref_h_z, ref_v_x, ref_v_z = _build_reference_mask_from_syndrome(
            sX=sX, sZ=sZ, active_X=active_X, active_Z=active_Z
        )
    else:
        ref_h_x, ref_h_z, ref_v_x, ref_v_z = _validate_edge_shift_masks(
            ref_h_x, ref_h_z, ref_v_x, ref_v_z, nrow, ncol
        )

    # logical mask
    log_h_x, log_h_z, log_v_x, log_v_z = _validate_edge_shift_masks(
        log_h_x, log_h_z, log_v_x, log_v_z, nrow, ncol
    )

    total_h_x = ref_h_x ^ log_h_x
    total_h_z = ref_h_z ^ log_h_z
    total_v_x = ref_v_x ^ log_v_x
    total_v_z = ref_v_z ^ log_v_z

    T = []
    for r in range(nrow):
        row_tensors = []
        for c in range(ncol):
            A = _build_face_tensor_zero_cycles(
                row=r,
                col=c,
                W_h=W_h,
                W_v=W_v,
                active_X=active_X,
                active_Z=active_Z,
                total_mask_h_x=total_h_x,
                total_mask_h_z=total_h_z,
                total_mask_v_x=total_v_x,
                total_mask_v_z=total_v_z,
            )
            row_tensors.append(A)
        T.append(row_tensors)

    return T


# def pauli_coset_likelihoods_peps_v5(
#     *,
#     sX: np.ndarray,
#     sZ: np.ndarray,
#     W_h: np.ndarray,
#     W_v: np.ndarray,
#     active_X: Optional[np.ndarray] = None,
#     active_Z: Optional[np.ndarray] = None,
#     logical_x_cut_col: Optional[int] = None,
#     logical_z_cut_row: Optional[int] = None,
#     Nkeep: int = 128,
#     Nsweep: int = 1,
#     return_debug: bool = False,
# ) -> Dict[CosetKey, float]:
#     r"""
#     Compute the four logical-coset likelihoods correctly as:

#         L_ab(s) = sum_{cycles C : dC = 0} W(C xor R(s) xor L_ab)

#     where R(s) is one fixed reference error with boundary equal to the measured
#     syndrome and L_ab is a fixed logical representative.
#     """
#     sX = np.asarray(sX, dtype=np.uint8)
#     sZ = np.asarray(sZ, dtype=np.uint8)
#     if sX.shape != sZ.shape:
#         raise ValueError("sX and sZ must have same shape")

#     nrow, ncol = sX.shape
#     W_h, W_v = validate_local_weight_tensor(W_h, W_v, nrow, ncol)

#     if active_X is None:
#         active_X = np.ones((nrow, ncol), dtype=np.uint8)
#     else:
#         active_X = np.asarray(active_X, dtype=np.uint8)

#     if active_Z is None:
#         active_Z = np.ones((nrow, ncol), dtype=np.uint8)
#     else:
#         active_Z = np.asarray(active_Z, dtype=np.uint8)

#     if active_X.shape != (nrow, ncol) or active_Z.shape != (nrow, ncol):
#         raise ValueError("active_X and active_Z must have shape (nrow, ncol)")

#     if logical_x_cut_col is None or logical_z_cut_row is None:
#         default_col, default_row = choose_default_logical_cuts(nrow, ncol)
#         if logical_x_cut_col is None:
#             logical_x_cut_col = default_col
#         if logical_z_cut_row is None:
#             logical_z_cut_row = default_row

#     # reference representative of the measured syndrome
#     ref_mask = _build_reference_mask_from_syndrome(
#         sX=sX,
#         sZ=sZ,
#         active_X=active_X,
#         active_Z=active_Z,
#     )

#     # canonical logical masks from the repo's native convention
#     log_I = _make_logical_shift_masks(
#         nrow=nrow,
#         ncol=ncol,
#         logical_z=False,
#         logical_x=False,
#         logical_x_cut_col=logical_x_cut_col,
#         logical_z_cut_row=logical_z_cut_row,
#     )
#     log_Z = _make_logical_shift_masks(
#         nrow=nrow,
#         ncol=ncol,
#         logical_z=True,
#         logical_x=False,
#         logical_x_cut_col=logical_x_cut_col,
#         logical_z_cut_row=logical_z_cut_row,
#     )
#     log_X = _make_logical_shift_masks(
#         nrow=nrow,
#         ncol=ncol,
#         logical_z=False,
#         logical_x=True,
#         logical_x_cut_col=logical_x_cut_col,
#         logical_z_cut_row=logical_z_cut_row,
#     )
#     log_Y = _make_logical_shift_masks(
#         nrow=nrow,
#         ncol=ncol,
#         logical_z=True,
#         logical_x=True,
#         logical_x_cut_col=logical_x_cut_col,
#         logical_z_cut_row=logical_z_cut_row,
#     )

#     logical_masks = {
#         (0, 0): log_I,
#         (1, 0): log_Z,
#         (0, 1): log_X,
#         (1, 1): log_Y,
#     }

#     cosets: Dict[CosetKey, float] = {}

#     for key, log_mask in logical_masks.items():
#         T = build_pauli_peps_zero_cycles(
#             sX=sX,
#             sZ=sZ,
#             W_h=W_h,
#             W_v=W_v,
#             active_X=active_X,
#             active_Z=active_Z,
#             ref_h_x=ref_mask[0],
#             ref_h_z=ref_mask[1],
#             ref_v_x=ref_mask[2],
#             ref_v_z=ref_mask[3],
#             log_h_x=log_mask[0],
#             log_h_z=log_mask[1],
#             log_v_x=log_mask[2],
#             log_v_z=log_mask[3],
#         )
#         val = contract_finPEPS(T, Nkeep=Nkeep, Nsweep=Nsweep)
#         cosets[key] = float(np.real_if_close(val))

#     vals = np.array([cosets[(0, 0)], cosets[(1, 0)], cosets[(0, 1)], cosets[(1, 1)]], dtype=float)
#     vmax = float(np.max(vals))
#     vmin = float(np.min(vals))
#     relspread = 0.0 if vmax == 0.0 else (vmax - vmin) / vmax

#     debug = {
#         "logical_x_cut_col": logical_x_cut_col,
#         "logical_z_cut_row": logical_z_cut_row,
#         "reference_weight": _mask_weight(ref_mask),
#         "reference_syndrome_ok": True,
#         "relspread": relspread,
#     }

#     if return_debug:
#         return cosets, debug
#     return cosets


def ml_sector_from_cosets(cosets: Dict[CosetKey, float]) -> Tuple[CosetKey, float]:
    sector, value = max(cosets.items(), key=lambda kv: kv[1])
    return sector, float(value)


def marginal_decision_from_cosets(
    cosets: Dict[CosetKey, float],
    *,
    memory_basis: str,
    tie_tol: float = 1e-14,
) -> int:
    """
    Correct one-bit Bayes decision from the four-sector posterior.

    X-memory  -> predict z_log
    Z-memory  -> predict x_log
    """
    p00 = float(cosets[(0, 0)])  # I
    p10 = float(cosets[(1, 0)])  # Z
    p01 = float(cosets[(0, 1)])  # X
    p11 = float(cosets[(1, 1)])  # Y

    basis = memory_basis.lower()
    if basis == "x":
        p0 = p00 + p01
        p1 = p10 + p11
    elif basis == "z":
        p0 = p00 + p10
        p1 = p01 + p11
    else:
        raise ValueError("memory_basis must be 'x' or 'z'")

    if abs(p1 - p0) <= tie_tol:
        return 0
    return int(p1 > p0)


def _get_default_weight_builder():
    return depolarizing_weights


def decode_batch_with_peps_v5(
    *,
    batch: Any,
    p: float,
    memory_basis: str,
    Nkeep: int = 128,
    Nsweep: int = 1,
    weight_builder=None,
    logical_x_cut_col: Optional[int] = None,
    logical_z_cut_row: Optional[int] = None,
    debug_shots: int = 0,
    debug_failures: bool = False,
    tie_tol: float = 1e-14,
    **kwargs,
) -> PEPSBatchDecodeResult:
    """
    Repo-compatible batch PEPS decoder using reference-error + zero-cycle ML.

    Expected batch fields:
        batch.sX               : (shots, nrow, ncol)
        batch.sZ               : (shots, nrow, ncol)
        batch.active_X         : (shots, nrow, ncol) or (nrow, ncol)
        batch.active_Z         : (shots, nrow, ncol) or (nrow, ncol)
        batch.observable_flips : (shots, 1)
    """
    if weight_builder is None:
        weight_builder = _get_default_weight_builder()

    sX_all = np.asarray(batch.sX, dtype=np.uint8)
    sZ_all = np.asarray(batch.sZ, dtype=np.uint8)

    if sX_all.shape != sZ_all.shape or sX_all.ndim != 3:
        raise ValueError("batch.sX and batch.sZ must have shape (shots, nrow, ncol)")

    shots, nrow, ncol = sX_all.shape

    obs_true = np.asarray(batch.observable_flips, dtype=np.uint8).reshape(shots, -1)
    if obs_true.shape[1] != 1:
        raise ValueError(f"Expected one logical observable, got shape {obs_true.shape}")

    active_X_all = getattr(batch, "active_X", None)
    active_Z_all = getattr(batch, "active_Z", None)

    if active_X_all is None:
        active_X_all = np.ones((shots, nrow, ncol), dtype=np.uint8)
    else:
        active_X_all = np.asarray(active_X_all, dtype=np.uint8)
        if active_X_all.shape == (nrow, ncol):
            active_X_all = np.broadcast_to(active_X_all, (shots, nrow, ncol)).copy()

    if active_Z_all is None:
        active_Z_all = np.ones((shots, nrow, ncol), dtype=np.uint8)
    else:
        active_Z_all = np.asarray(active_Z_all, dtype=np.uint8)
        if active_Z_all.shape == (nrow, ncol):
            active_Z_all = np.broadcast_to(active_Z_all, (shots, nrow, ncol)).copy()

    if active_X_all.shape != (shots, nrow, ncol):
        raise ValueError(f"Bad active_X shape: {active_X_all.shape}")
    if active_Z_all.shape != (shots, nrow, ncol):
        raise ValueError(f"Bad active_Z shape: {active_Z_all.shape}")

    W_h, W_v = weight_builder(nrow, ncol, p)

    if logical_x_cut_col is None or logical_z_cut_row is None:
        default_col, default_row = choose_default_logical_cuts(nrow, ncol)
        if logical_x_cut_col is None:
            logical_x_cut_col = default_col
        if logical_z_cut_row is None:
            logical_z_cut_row = default_row

    coset_likelihoods: List[Dict[CosetKey, float]] = []
    ml_cosets: List[CosetKey] = []
    predicted = np.zeros((shots, 1), dtype=np.uint8)

    for k in range(shots):
        cosets = pauli_coset_likelihoods_peps(
            sX=sX_all[k],
            sZ=sZ_all[k],
            W_h=W_h,
            W_v=W_v,
            active_X=active_X_all[k],
            active_Z=active_Z_all[k],
            logical_x_cut_col=logical_x_cut_col,
            logical_z_cut_row=logical_z_cut_row,
            Nkeep=Nkeep,
            Nsweep=Nsweep,
        )
        dbg = {"logical_x_cut_col": logical_x_cut_col, "logical_z_cut_row": logical_z_cut_row,
            "reference_weight": 0, "relspread": 0.0}

        coset_likelihoods.append(cosets)

        ml_sector, _ = ml_sector_from_cosets(cosets)
        ml_cosets.append(ml_sector)

        predicted[k, 0] = np.uint8(
            marginal_decision_from_cosets(
                cosets,
                memory_basis=memory_basis,
                tie_tol=tie_tol,
            )
        )

        should_print = (k < debug_shots)
        if debug_failures and predicted[k, 0] != obs_true[k, 0]:
            should_print = True

        if should_print:
            print("=" * 90)
            print(f"shot               = {k}")
            print(f"memory_basis       = {memory_basis}")
            print(f"true observable    = {int(obs_true[k, 0])}")
            print(f"pred observable    = {int(predicted[k, 0])}")
            print(f"ml sector          = {ml_sector}")
            print(f"cosets             = {cosets}")
            print(f"logical_x_cut_col  = {dbg['logical_x_cut_col']}")
            print(f"logical_z_cut_row  = {dbg['logical_z_cut_row']}")
            print(f"reference_weight   = {dbg['reference_weight']}")
            print(f"relspread          = {dbg['relspread']:.6e}")

    logical_failures = np.bitwise_xor(predicted[:, 0], obs_true[:, 0]).astype(np.uint8)
    logical_error_rate = float(np.mean(logical_failures))

    return PEPSBatchDecodeResult(
        coset_likelihoods=coset_likelihoods,
        ml_cosets=ml_cosets,
        predicted_observable_flips=predicted,
        logical_failures=logical_failures,
        logical_error_rate=logical_error_rate,
    )


# convenient alias if you want to import this file as the active PEPS decoder
decode_batch_with_peps = decode_batch_with_peps_v5