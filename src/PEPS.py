import numpy as np
from typing import List
from .mtimes_MPO import mtimes_MPO
from .functions import contract, get_identity, canon_form  

# assumes you already have mtimes_MPO(B, A, Nkeep, Nsweep) from the previous translation


def contract_finPEPS(T: List[List[np.ndarray]], Nkeep: int, Nsweep: int):
    """
    Contract reduced tensors on a finite square lattice with open boundaries
    using MPO-MPS method (row-by-row absorption via mtimes_MPO).

    Input:
      T[m][n] : rank-4 reduced tensor at site (m,n), legs ordered as (left, up, down, right).
      Nkeep   : max horizontal bond dimension during truncations.
      Nsweep  : number of sweeps inside each mtimes_MPO multiplication.

    Output:
      res : complex scalar, full contraction result.

    Notes:
      We permute each T[m][n] from (L,U,D,R) to (D,U,L,R) to match mtimes_MPO's
      expected MPO convention (bottom, top, left, right).
    """

    nrow = len(T)
    if nrow == 0:
        raise ValueError("ERR: Empty tensor array T.")
    ncol = len(T[0])
    if any(len(row) != ncol for row in T):
        raise ValueError("ERR: T must be a rectangular (nrow x ncol) list-of-lists.")

    # -----------------------
    # sanity check (MATLAB)
    # -----------------------
    for it1 in range(nrow):
        for it2 in range(ncol):
            Tij = T[it1][it2]
            if Tij.ndim != 4:
                raise ValueError(f"ERR: T[{it1},{it2}] must be rank-4.")
            # Tij legs are (L,U,D,R) == axes (0,1,2,3)
            if it1 < nrow - 1:
                # down of current (axis 2) matches up of below (axis 1)
                if Tij.shape[2] != T[it1 + 1][it2].shape[1]:
                    raise ValueError(
                        f"ERR: down leg of T[{it1},{it2}] and up leg of T[{it1+1},{it2}] do not match."
                    )
            if it2 < ncol - 1:
                # right of current (axis 3) matches left of right neighbor (axis 0)
                if Tij.shape[3] != T[it1][it2 + 1].shape[0]:
                    raise ValueError(
                        f"ERR: right leg of T[{it1},{it2}] and left leg of T[{it1},{it2+1}] do not match."
                    )
            if it1 == 0 and Tij.shape[1] != 1:
                raise ValueError(f"ERR: up leg of T[{it1},{it2}] has non-singleton dimension.")
            if it1 == nrow - 1 and Tij.shape[2] != 1:
                raise ValueError(f"ERR: down leg of T[{it1},{it2}] has non-singleton dimension.")
            if it2 == 0 and Tij.shape[0] != 1:
                raise ValueError(f"ERR: left leg of T[{it1},{it2}] has non-singleton dimension.")
            if it2 == ncol - 1 and Tij.shape[3] != 1:
                raise ValueError(f"ERR: right leg of T[{it1},{it2}] has non-singleton dimension.")

    # -----------------------------------------
    # permute legs to use mtimes_MPO:
    # MATLAB: T{itN} = permute(T{itN}, [3 2 1 4])  i.e. (D,U,L,R)
    # -----------------------------------------
    Tp: List[List[np.ndarray]] = [[None for _ in range(ncol)] for _ in range(nrow)]
    for it1 in range(nrow):
        for it2 in range(ncol):
            Tp[it1][it2] = np.transpose(T[it1][it2], (2, 1, 0, 3))  # (D,U,L,R)

    # -----------------------------------------
    # stabilize norm by extracting it each row
    # -----------------------------------------
    logNorm = 0.0

    # first row as "MPS-like MPO"
    T2: List[np.ndarray] = [np.array(Tp[0][j], copy=True) for j in range(ncol)]

    # absorb rows 2..(nrow-1) excluding the last row (MATLAB: for it1=2:size(T,1)-1)
    for it1 in range(1, nrow - 1):
        # MATLAB: T2 = mtimes_MPO(T(it1,:), T2, Nkeep, Nsweep);
        B_row = [Tp[it1][j] for j in range(ncol)]
        T2 = mtimes_MPO(B_row, T2, Nkeep, Nsweep)

        # factor out norm by bringing into right-canonical form
        # Merge (bottom, top) legs to convert MPO -> MPS rank-3
        Aloc = [None] * ncol
        T2_mps = [None] * ncol
        for j in range(ncol):
            # identity in (bottom x top) space
            Aloc[j] = get_identity(T2[j], 0, T2[j], 1)  # :contentReference[oaicite:2]{index=2}
            # contract physical legs (0,1) with Aloc legs (0,1) -> (left,right,physMerged)
            T2_mps[j] = contract(T2[j], [0, 1], Aloc[j], [0, 1])

        # canon_form(..., id=0) makes the chain right-canonical and returns Schmidt values S on the left cut
        T2_mps, S, _dw = canon_form(T2_mps, 0, Nkeep, None)  # :contentReference[oaicite:3]{index=3}

        # In MATLAB they do logNorm += log(S) where S is effectively a scalar norm.
        # Here canon_form returns a vector of Schmidt coefficients. For a right-canonical MPS,
        # the global norm is ||S||_2.
        norm_row = float(np.linalg.norm(S))
        if norm_row == 0.0 or not np.isfinite(norm_row):
            raise FloatingPointError(f"ERR: extracted norm is {norm_row} at absorbed row {it1}.")
        logNorm += np.log(norm_row)

        # restore MPO rank-4: contract merged leg back with conj(Aloc)
        for j in range(ncol):
            tmp = contract(T2_mps[j], 2, np.conjugate(Aloc[j]), 2)  # (left,right,bottom,top)
            T2[j] = np.transpose(tmp, (2, 3, 0, 1))                  # (bottom,top,left,right)

        # also divide out the norm so tensors stay well-scaled
        # (MATLAB canonForm effectively does this separation; we do it explicitly.)
        T2[0] = T2[0] / norm_row

    # -----------------------------------------
    # final contraction with the last row
    # MATLAB:
    #   res = 1;
    #   for it2=1:L:
    #       Ttmp = contract(res,2,2,T2{it2},4,3);
    #       res  = contract(T{end,it2},4,[3 2 1],Ttmp,4,(1:3));
    #   end
    # -----------------------------------------
    res = np.array([[1.0 + 0.0j]])  # shape (1,1) "rank-2 scalar" for consistent contractions

    last_row = [Tp[-1][j] for j in range(ncol)]  # already (D,U,L,R)

    for j in range(ncol):
        # contract res (axis 1) with left leg (axis 2) of T2[j]
        Ttmp = contract(res, 1, T2[j], 2)  # -> remaining: (res_axis0, T2.bottom, T2.top, T2.right)

        # contract last_row[j] legs (left, up, down) == axes (2,1,0) with Ttmp axes (0,1,2)
        res = contract(last_row[j], [2, 1, 0], Ttmp, [0, 1, 2])  # -> remaining: (last_row.right, Ttmp.right)

    # after sweeping all columns, open boundaries imply both remaining legs are singleton -> scalar
    res = np.asarray(res).squeeze()

    # restore the separated norm
    res = res * np.exp(logNorm)
    return res