import numpy as np
from typing import List
from .functions import contract, get_identity, canon_form, svd_tr


def _as_rank3_boundary(x):
    """
    In tensor code, treat that as a rank-3 tensor of all dummy dims = 1.
    """
    if isinstance(x, (int, float, complex, np.number)):
        return np.array([x], dtype=np.complex128).reshape(1, 1, 1)
    x = np.asarray(x)
    if x.ndim == 0:
        return np.array([x.item()], dtype=np.complex128).reshape(1, 1, 1)
    return x


def _contract_BA(T: np.ndarray, B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """

    Input conventions:
      A, B: rank-4 with legs (bottom, top, left, right)  == axes (0,1,2,3)
      T: rank-3 environment tensor (dummy dims allowed)

    Output:
      rank-5 tensor with legs ordered as:
        (T[0])-(B[0])-(B[3])-(A[1])-(A[3])
      i.e. (1st of T)-(1st of B)-(4th of B)-(2nd of A)-(4th of A)
    """
    inter = contract(T, 2, A, 2)  # remaining order: (T0,T1,A0,A1,A3)
    out = contract(inter, [1, 2], B, [2, 1])  # default remaining: (inter0,inter3,inter4,B0,B3)
    out = np.transpose(out, (0, 3, 4, 1, 2))
    return out


def _contract_CBA(T: np.ndarray, C: np.ndarray, B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """

    IMPORTANT:
      The contraction corresponds to contracting:
        C[left,bottom,top] with T[T0,B0,A_top]
      leaving (C_right, B_right, A_right).
    """
    T5 = _contract_BA(T, B, A)  # (T0, B0, B3, A1, A3)
    out = contract(np.conjugate(C), [2, 0, 1], T5, [0, 1, 3])
    return out


def _contract_CBA2(
    Tl: np.ndarray, Bl: np.ndarray, Al: np.ndarray,
    Tr: np.ndarray, Br: np.ndarray, Ar: np.ndarray
) -> np.ndarray:
    """

    Output:
      rank-6 tensor with legs ordered as:
        (Tl[0])-(Bl.bottom)-(Al.top)-(Br.bottom)-(Ar.top)-(Tr[0])
      i.e. (1st of Tl)-(1st of Bl)-(2nd of Al)-(1st of Br)-(2nd of Ar)-(1st of Tr)
    """
    Tl5 = _contract_BA(Tl, Bl, Al)  # (Tl0, Bl0, Bl3, Al1, Al3)

    Br = np.transpose(Br, (0, 1, 3, 2))
    Ar = np.transpose(Ar, (0, 1, 3, 2))
    Tr5 = _contract_BA(Tr, Br, Ar)  # (Tr0, Brp0, Brp3, Arp1, Arp3)
    out = contract(Tl5, [2, 4], Tr5, [2, 4])
    out = np.transpose(out, (0, 1, 2, 4, 5, 3))
    return out


def mtimes_MPO(B: List[np.ndarray], A: List[np.ndarray], Nkeep: int, Nsweep: int) -> List[np.ndarray]:
    """
    Variational multiplication of two MPOs

    MPO tensor convention :
      each A[n], B[n], C[n] is rank-4 with legs (bottom, top, left, right) == axes (0,1,2,3).

    Args:
      B, A   : list length N, each element rank-4 tensor (b,t,l,r)
      Nkeep  : maximum bond dimension for result MPO C
      Nsweep : number of round-trip sweeps (L->R then R->L), starting L->R and ending R->L

    Returns:
      C : MPO for product A (top) times B (bottom), optimized variationally.
    """

    N = len(A)

    # --- sanity checks ---
    if N != len(B):
        raise ValueError("ERR: Length of two input MPOs do not match.")

    for itN in range(N):
        An = A[itN]
        Bn = B[itN]
        if itN == 0 and not (An.shape[2] == 1 and Bn.shape[2] == 1):
            raise ValueError("ERR: The leftmost leg of an MPO should be dummy.")
        if itN == N - 1 and not (An.shape[3] == 1 and Bn.shape[3] == 1):
            raise ValueError("ERR: The rightmost leg of an MPO should be dummy.")
        if itN < N - 1 and An.shape[3] != A[itN + 1].shape[2]:
            raise ValueError(f"ERR: A[{itN}].right != A[{itN+1}].left in dimension.")
        if itN < N - 1 and Bn.shape[3] != B[itN + 1].shape[2]:
            raise ValueError(f"ERR: B[{itN}].right != B[{itN+1}].left in dimension.")
        if An.shape[0] != Bn.shape[1]:
            raise ValueError(f"ERR: A[{itN}].bottom != B[{itN}].top in dimension.")

    # --- Initialize C with A ---
    C = [np.array(x, copy=True) for x in A]

    # --- Bring C into right-canonical form (via MPS canonicalization after merging physical legs) ---
    Aloc: List[np.ndarray] = [None] * N  # isometries for merging bottom & top legs
    Cmps: List[np.ndarray] = [None] * N  # rank-3 MPS tensors (left,right,physMerged)

    for itN in range(N):
        Aloc[itN] = get_identity(C[itN], 0, C[itN], 1)  # shape (Db, Dt, Db*Dt)

        # Contract (bottom, top) of C with first two legs of Aloc, leaving (left,right,physMerged)
        Cmps[itN] = contract(C[itN], [0, 1], Aloc[itN], [0, 1])  # (l,r,pMerged)

    C, _, _ = canon_form(Cmps, 0, Nkeep, None)
    
    # Bring back to rank-4 MPO tensors (bottom, top, left, right)
    for itN in range(N):
        tmp = contract(C[itN], 2, np.conjugate(Aloc[itN]), 2)  # (l,r,b,t)
        C[itN] = np.transpose(tmp, (2, 3, 0, 1))                  # (b,t,l,r)
    
    # --- Build environments ABC (rank-3), with boundary dummies ---
    ABC: List[np.ndarray] = [None] * (N + 2)
    ABC[0] = _as_rank3_boundary(1)
    ABC[-1] = _as_rank3_boundary(1)

    # first sweep is left-to-right, so initialize by contracting from the right
    for itN in range(N - 1, -1, -1):
        ABC[itN + 1] = _contract_CBA(
            ABC[itN + 2],
            np.transpose(C[itN], (0, 1, 3, 2)),  
            np.transpose(B[itN], (0, 1, 3, 2)),
            np.transpose(A[itN], (0, 1, 3, 2)),
        )

    # --- Sweeps ---
    for _ in range(Nsweep):

        for itN in range(0, N - 2):
            T6 = _contract_CBA2(ABC[itN], B[itN], A[itN], ABC[itN + 3], B[itN + 1], A[itN + 1])
            U, S, Vd, _ = svd_tr(T6, 6, [0, 1, 2], Nkeep, None)

            # U legs: (Tl0, Bl.bottom, Al.top, bond) -> permute to (bottom, top, left, right)
            C[itN] = np.transpose(U, (1, 2, 0, 3))

            tmp = contract(np.diag(S), 1, Vd, 0)      # (bond, Br.bottom, Ar.top, Tr0)
            C[itN + 1] = np.transpose(tmp, (1, 2, 0, 3))

            ABC[itN + 1] = _contract_CBA(ABC[itN], C[itN], B[itN], A[itN])
        # right-to-left sweep:
        for itN in range(N - 2, -1, -1):
            T6 = _contract_CBA2(ABC[itN], B[itN], A[itN], ABC[itN + 3], B[itN + 1], A[itN + 1])
            U, S, Vd, _ = svd_tr(T6, 6, [0, 1, 2], Nkeep, None)

            # Vd legs: (bond, Br.bottom, Ar.top, Tr0) -> (bottom, top, left, right)
            C[itN + 1] = np.transpose(Vd, (1, 2, 0, 3))

            tmp = contract(U, 3, np.diag(S), 0)       # (Tl0, Bl.bottom, Al.top, bond)
            C[itN] = np.transpose(tmp, (1, 2, 0, 3))

            ABC[itN + 2] = _contract_CBA(
                ABC[itN + 3],
                np.transpose(C[itN + 1], (0, 1, 3, 2)),
                np.transpose(B[itN + 1], (0, 1, 3, 2)),
                np.transpose(A[itN + 1], (0, 1, 3, 2)),
            )
    return C