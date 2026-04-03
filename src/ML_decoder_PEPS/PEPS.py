import numpy as np
from typing import List
from .mtimes_MPO import mtimes_MPO
from .functions import contract, get_identity, canon_form  

def contract_finPEPS(T: List[List[np.ndarray]], Nkeep: int, Nsweep: int):
    """
    Parameters
    ----------
    T : list[list[np.ndarray]]
        T[m][n] is a rank-4 reduced tensor with legs ordered as
        (left, up, down, right).
    Nkeep : int
        Maximum bond dimension along the horizontal direction.
    Nsweep : int
        Number of sweeps used inside mtimes_MPO.

    Returns
    -------
    res : scalar
        Full contraction result.
    """

    # basic shape / rectangular checks
    nrow = len(T)
    if nrow == 0:
        raise ValueError("ERR: T is empty.")
    ncol = len(T[0])
    if any(len(row) != ncol for row in T):
        raise ValueError("ERR: T must be rectangular.")

    # make a working copy
    T = [[np.array(T[i][j], copy=True) for j in range(ncol)] for i in range(nrow)]

    # sanity check
    for it1 in range(nrow):
        for it2 in range(ncol):
            X = T[it1][it2]
            if X.ndim != 4:
                raise ValueError(f"ERR: T[{it1},{it2}] is not rank-4.")

            if (it1 < nrow - 1) and (T[it1][it2].shape[2] != T[it1 + 1][it2].shape[1]):
                raise ValueError(
                    f"ERR: The down leg of T[{it1},{it2}] and the up leg of T[{it1+1},{it2}] do not match."
                )
            elif (it2 < ncol - 1) and (T[it1][it2].shape[3] != T[it1][it2 + 1].shape[0]):
                raise ValueError(
                    f"ERR: The right leg of T[{it1},{it2}] and the left leg of T[{it1},{it2+1}] do not match."
                )
            elif (it1 == 0) and (T[it1][it2].shape[1] != 1):
                raise ValueError(f"ERR: The up leg of T[{it1},{it2}] has non-singleton dimension.")
            elif (it1 == nrow - 1) and (T[it1][it2].shape[2] != 1):
                raise ValueError(f"ERR: The down leg of T[{it1},{it2}] has non-singleton dimension.")
            elif (it2 == 0) and (T[it1][it2].shape[0] != 1):
                raise ValueError(f"ERR: The left leg of T[{it1},{it2}] has non-singleton dimension.")
            elif (it2 == ncol - 1) and (T[it1][it2].shape[3] != 1):
                raise ValueError(f"ERR: The right leg of T[{it1},{it2}] has non-singleton dimension.")

    for i in range(nrow):
        for j in range(ncol):
            T[i][j] = np.transpose(T[i][j], (2, 1, 0, 3))   # (D,U,L,R)
    logNorm = 0.0

    T2 = [np.array(T[0][j], copy=True) for j in range(ncol)]
    for it1 in range(1, nrow - 1):
        # MATLAB: T2 = mtimes_MPO(T(it1,:), T2, Nkeep, Nsweep);
        row_mpo = [T[it1][j] for j in range(ncol)]
        T2 = mtimes_MPO(row_mpo, T2, Nkeep, Nsweep)
        Aloc = [None] * ncol
        for it2 in range(ncol):
            Aloc[it2] = get_identity(T2[it2], 0, T2[it2], 1)
            T2[it2] = contract(T2[it2], [0, 1], Aloc[it2], [0, 1])
        T2, S, _ = canon_form(T2, 0, Nkeep, None)

        # bring back to rank-4
        for it2 in range(ncol):
            tmp = contract(T2[it2], 2, np.conjugate(Aloc[it2]), 2)
            T2[it2] = np.transpose(tmp, (2, 3, 0, 1))   # (D,U,L,R)
        S_scalar = np.asarray(S).squeeze()
        if np.size(S_scalar) != 1:
            raise ValueError(f"ERR: canon_form(..., id=0) returned non-scalar S with shape {np.shape(S)}")
        logNorm += np.log(float(S_scalar))
    res = np.array([[1.0]], dtype=np.result_type(*[x.dtype for row in T for x in row]))

    for it2 in range(ncol):
        # T2[it2] is (D,U,L,R); contract res axis 1 with T2 left leg axis 2
        Ttmp = contract(res, 1, T2[it2], 2)
        # Ttmp legs become: (res_left, D, U, R)

        # T[last,it2] is also (D,U,L,R)
        res = contract(T[-1][it2], [2, 1, 0], Ttmp, [0, 1, 2])

    # restore separated norm
    res = np.asarray(res).squeeze() * np.exp(logNorm)

    return res