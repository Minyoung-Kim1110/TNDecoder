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


