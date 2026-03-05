import numpy as np
import scipy.linalg
from src.functions import *

def dmrg_gs_1site(M, Hs, Nkeep, Nsweep, Krylov=5, tol=1e-8):
    """
    Single-site Density Matrix Renormalization Group (DMRG) for 1D systems.
    
    Parameters:
        M : list of np.ndarray
            MPS (Matrix Product State), list of rank-3 tensors.
        Hs : list of np.ndarray
            MPO (Matrix Product Operator) for the Hamiltonian.
        Nkeep : int
            Maximum bond dimension for MPS.
        Nsweep : int
            Number of sweeps (each sweep consists of left and right passes).
        Krylov : int, optional
            Maximum Krylov subspace size in the Lanczos method. Default: 5.
        tol : float, optional
            Tolerance for convergence in Lanczos. Default: 1e-8.
    
    Returns:
        M : list of np.ndarray
            Updated MPS in left-canonical form.
        E0 : float
            Ground state energy.
        Eiter : np.ndarray
            Energy values at each iteration (N x 2Nsweep).
        Sv : list
            Singular values at each bond.
    """

    # Number of sites in the chain
    N = len(M)
    if N < 2 or N != len(Hs):
        raise ValueError("ERR: Chain length mismatch between MPS and MPO.")

    # Initialize energy history storage
    Eiter = np.zeros((N, 2 * Nsweep))
    Sv = [None] * (N + 1) #TODO

    # Ensure M is in left-canonical form
    M = canon_form(M, N-1, None, 0)[0]

    # Initialize left and right Hamiltonian contractions
    Hlr = [None] * (N + 2)
    Hlr[0] = np.array([1]).reshape(1,1,1)  # Left boundary
    Hlr[-1] = np.array([1]).reshape(1,1,1)  # Right boundary

    # Compute initial left-to-right contraction of Hamiltonian
    for itN in range(N):
        Hlr[itN + 1] = update_left(Hlr[itN], 3, M[itN], Hs[itN], 4, M[itN])

    for itS in range(Nsweep):
        # Right -> Left Sweep
        for itN in range(N - 1, -1, -1):
            M[itN], Eiter[N - 1 - itN, 2 * itS] = eigs_1site_gs(Hlr[itN], Hs[itN], Hlr[itN + 2], M[itN], Krylov, tol)

            # SVD, do not reduce bond dimension
            U, Sv[itN], M[itN], _ = svd_tr(M[itN], 3, [0], Nkeep, 0)

            # Update MPS tensor for next iteration
            if itN > 0:
                M[itN - 1] = contract(M[itN - 1],  1, U @ np.diag(Sv[itN]),  0, permute_order=[0, 2, 1])
            else:
                M[itN] = contract(U,  1, M[itN],  0)

            # Update left Hamiltonian contraction
            Hlr[itN + 1] = update_left(Hlr[itN + 2], 3, np.transpose(M[itN], (1, 0, 2)), 
                                       np.transpose(Hs[itN], (0, 1, 3, 2)), 4, np.transpose(M[itN], (1, 0, 2)))

        # Left -> Right Sweep
        for itN in range(N):
            M[itN], Eiter[itN, 2 * itS + 1] = eigs_1site_gs(Hlr[itN], Hs[itN], Hlr[itN + 2], M[itN], Krylov, tol)

            # SVD, do not reduce bond dimension
            M[itN], Sv[itN + 1], Vd, _ = svd_tr(M[itN], 3, [0, 2], Nkeep, 0)
            M[itN] = np.transpose(M[itN], (0, 2, 1))

            # Update MPS tensor for next iteration
            if itN < N - 1:
                M[itN + 1] = contract(np.diag(Sv[itN + 1]) @ Vd,  1, M[itN + 1],  0)
            else:
                M[itN] = contract(M[itN],  1, Vd,  0, permute_order=[0, 2, 1])

            # Update right Hamiltonian contraction
            Hlr[itN + 1] = update_left(Hlr[itN], 3, M[itN], Hs[itN], 4, M[itN])

    E0 = Eiter[N - 1, 2 * itS]  # Final ground state energy
    return M, E0, Eiter, Sv

def eigs_1site_gs(Hleft, Hcen, Hright, Aold, nKrylov, tol):
    """
    Lanczos-based eigenvalue solver for single-site DMRG.

    Parameters:
        Hleft, Hcen, Hright : np.ndarray
            Hamiltonian tensors for left, center (MPO), and right blocks.
        Aold : np.ndarray
            Current MPS tensor.
        nKrylov : int
            Krylov subspace size.
        tol : float
            Convergence tolerance.

    Returns:
        Anew : np.ndarray
            Updated MPS tensor.
        Enew : float
            Ground state energy.
    """

    # Initialize Krylov basis
    shape = Aold.shape
    As = np.zeros((shape[0], shape[1], shape[2], nKrylov), dtype=Aold.dtype)
    As[..., 0] = Aold / np.linalg.norm(Aold)

    alphas = np.zeros(nKrylov)
    betas = np.zeros(nKrylov - 1)
    cnt = 0

    for itn in range(nKrylov):
        Amul = contract(Hleft, 1, As[..., itn],  0)
        Amul = contract(Amul,  [1, 3], Hcen,  [2, 1])
        Amul = contract(Amul,  [1, 3], Hright,  [1, 2], permute_order=[0, 2, 1])

        alphas[itn] = np.real(contract(As[..., itn],  [0, 1, 2], Amul,  [0, 1, 2]))

        cnt += 1

        if itn < nKrylov - 1:
            for _ in range(2):
                T = contract(As[..., :itn+1],  [0, 1, 2], Amul,  [0, 1, 2])
                T = contract(As[..., :itn+1],  3, T,  0)
                Amul -= T

            Anorm = np.linalg.norm(Amul)
            if Anorm < tol:
                break

            As[..., itn + 1] = Amul / Anorm
            betas[itn] = Anorm

    # Solve tridiagonal eigenproblem
    Hkrylov = np.diag(betas[:cnt-1], -1) + np.diag(alphas[:cnt]) + np.diag(betas[:cnt-1], 1)
    eigvals, eigvecs = scipy.linalg.eigh(Hkrylov)
    Anew = contract(As[..., :cnt],  3, eigvecs[:, 0],  0)

    # Compute ground state energy
    Amul = contract(Hleft,  1, Anew,  0)
    Amul = contract(Amul,  [1, 3], Hcen,  [2, 1])
    Amul = contract(Amul,  [1, 3], Hright,  [1, 2], permute_order=[0, 2, 1])
    Enew = np.real(contract(Anew,  [0, 1, 2], Amul,  [0, 1, 2]))

    return Anew, Enew

def dmrg_gs_2site(M, Hs, Nkeep, Nsweep, Krylov=5, tol=1e-8):
    """
    Two-site Density Matrix Renormalization Group (DMRG) for 1D systems.
    
    Parameters:
        M : list of np.ndarray
            MPS (Matrix Product State), list of rank-3 tensors.
        Hs : list of np.ndarray
            MPO (Matrix Product Operator) for the Hamiltonian.
        Nkeep : int
            Maximum bond dimension for MPS.
        Nsweep : int
            Number of sweeps (each sweep consists of left and right passes).
        Krylov : int, optional
            Maximum Krylov subspace size in the Lanczos method. Default: 5.
        tol : float, optional
            Tolerance for convergence in Lanczos. Default: 1e-8.
    
    Returns:
        M : list of np.ndarray
            Updated MPS in left-canonical form.
        E0 : float
            Ground state energy.
        Eiter : np.ndarray
            Energy values at each iteration ((N-1) x 2Nsweep).
        Sv : list
            Singular values at each bond.
    """

    # Number of sites in the chain
    N = len(M)
    if N < 2 or N != len(Hs):
        raise ValueError("ERR: Chain length mismatch between MPS and MPO.")

    # Initialize energy history storage
    Eiter = np.zeros((N-1, 2 * Nsweep))
    Sv = [None] * (N + 1)

    # Ensure M is in left-canonical form
    M = canon_form(M, N-1, None, 0)[0]

    # Initialize left and right Hamiltonian contractions
    Hlr = [None] * (N + 2)
    Hlr[0] = np.array([1]).reshape(1,1,1) # Left boundary
    Hlr[-1] = np.array([1]).reshape(1,1,1)  # Right boundary

    # Compute initial left-to-right contraction of Hamiltonian
    for itN in range(N):
        Hlr[itN + 1] = update_left(Hlr[itN], 3, M[itN], Hs[itN], 4, M[itN])

    for itS in range(Nsweep):
        # Right -> Left Sweep
        for itN in range(N - 1, 0, -1):
            Aold = contract(M[itN-1],  1, M[itN],  0, permute_order=[0, 2, 1, 3])
            Anew, Eiter[N - itN - 1, 2 * itS] = eigs_2site_gs(Hlr[itN-1], Hs[itN-1], Hs[itN], Hlr[itN+2], Aold, Krylov, tol)

            # SVD and truncate small singular values
            M[itN-1], Sv[itN], M[itN], _ = svd_tr(Anew, 4, [0, 2], Nkeep, tol)
            M[itN-1] = np.transpose(M[itN-1], (0, 2, 1))
            

            M[itN-1] = contract(M[itN-1], 1, np.diag(Sv[itN]), 0, [0, 2,1])
                
            # print('after', itN, M_opt[itN-1].shape, M_opt[itN].shape)
            
            # Update left Hamiltonian contraction
            Hlr[itN+1] = update_left(Hlr[itN+2], 3, np.transpose(M[itN], (1, 0, 2)), 
                                     np.transpose(Hs[itN], (0, 1, 3, 2)), 4, np.transpose(M[itN], (1, 0, 2)))
            # print(f'updated shape for H_rl{itN+1}', Hlr[itN+1].shape)
        # Left -> Right Sweep
        for itN in range(N - 1):
            Aold = contract(M[itN],  1, M[itN+1], 0, permute_order=[0, 2, 1, 3])
            Anew, Eiter[itN, 2 * itS + 1] = eigs_2site_gs(Hlr[itN], Hs[itN], Hs[itN+1], Hlr[itN+3], Aold, Krylov, tol)

            # SVD and truncate small singular values
            M[itN], Sv[itN+1], M[itN+1], _ = svd_tr(Anew, 4, [0, 2], Nkeep, 1e-8)
            M[itN] = np.transpose(M[itN], (0, 2, 1))

            # Update next tensor
            M[itN+1] = contract(np.diag(Sv[itN+1]),  1, M[itN+1],  0)

            # Update right Hamiltonian contraction
            Hlr[itN+1] = update_left(Hlr[itN], 3, M[itN], Hs[itN], 4, M[itN])

    E0 = Eiter[-1, -1]  # Final ground state energy
    return M, E0, Eiter, Sv

def eigs_2site_gs(Hleft, Hcen1, Hcen2, Hright, Aold, nKrylov, tol):
    """
    Compute the ground state of an MPS tensor acting on two neighboring sites using the Lanczos method.

    Parameters:
        Hleft  : np.ndarray (Left part of effective Hamiltonian, rank-3)
        Hcen1  : np.ndarray (First MPO tensor, rank-4)
        Hcen2  : np.ndarray (Second MPO tensor, rank-4)
        Hright : np.ndarray (Right part of effective Hamiltonian, rank-3)
        Aold   : np.ndarray (Current MPS tensor, rank-4)
        nKrylov: int (Max Krylov subspace size)
        tol    : float (Convergence tolerance for Lanczos method)

    Returns:
        Anew : np.ndarray (Updated MPS tensor)
        Enew : float (Ground state energy)
    """

    shape = Aold.shape
    As = np.zeros((*shape, nKrylov), dtype=Aold.dtype)
    
    # Normalize the first Krylov vector
    norm_Aold = np.sqrt(abs(contract(Aold.conj(), (0, 1, 2, 3), Aold, (0, 1, 2, 3))))
    As[..., 0] = Aold / norm_Aold

    alphas = np.zeros(nKrylov)
    betas = np.zeros(nKrylov - 1)
    cnt = 0

    for itn in range(nKrylov):
        # "Matrix-vector" multiplication
        # print(itn, Hleft.shape, As[..., itn].shape)
        Amul = contract(Hleft, [1], As[..., itn], [0])
        Amul = contract(Amul, [1, 3], Hcen1, [2, 1])
        Amul = contract(Amul, [2, 4], Hcen2, [1, 2])
        Amul = contract(Amul, [1, 4], Hright, [1, 2], permute_order=[0, 3, 1, 2])

        # Compute diagonal element (energy contribution)
        alphas[itn] = np.real(contract(As[..., itn].conj(), [0, 1, 2, 3], Amul, [0, 1, 2, 3]))

        cnt += 1
        if itn < nKrylov - 1:
            # Orthogonalization to get the next Krylov vector
            for _ in range(2):  # Do twice to reduce numerical noise
                T = contract(As[..., :itn+1].conj(), [0, 1, 2, 3], Amul, [0, 1, 2, 3])
                T = contract(As[..., :itn+1], 4, T, 0)
                Amul -= T

            # Compute norm
            Anorm = np.sqrt(abs(contract(Amul.conj(), (0, 1, 2, 3), Amul, (0, 1, 2, 3))))

            if Anorm < tol:  # Stop if norm is too small
                break

            As[..., itn + 1] = Amul / Anorm
            betas[itn] = Anorm

    # Build and diagonalize the Krylov Hamiltonian
    Hkrylov = np.diag(betas[:cnt-1], -1) + np.diag(alphas[:cnt]) + np.diag(betas[:cnt-1], 1)
    eigvals, eigvecs = scipy.linalg.eigh(Hkrylov)
    Anew = contract(As[..., :cnt], 4, eigvecs[:, 0], 0)

    # Compute expectation value <Anew | H | Anew>
    Amul = contract(Hleft, 1, Anew, 0)
    Amul = contract(Amul, [1, 3], Hcen1, [2, 1])
    Amul = contract(Amul, [2, 4], Hcen2, [1, 2])
    Amul = contract(Amul, [1, 4], Hright, [1, 2], permute_order=[0, 3, 1, 2])

    Enew = np.real(contract(Anew.conj(), (0, 1, 2, 3), Amul, (0, 1, 2, 3)))

    return Anew, Enew

def dmrg_1ES_1site(M, M0, Hs, Nkeep, Nsweep, Krylov=5, tol=1e-8):
    """
    Single-site DMRG for the first excited state with orthogonality to the ground state.

    Args:
        M: List[np.ndarray] - Initial guess for excited MPS.
        M0: List[np.ndarray] - Ground state MPS to stay orthogonal to.
        Hs: List[np.ndarray] - MPO for Hamiltonian (rank-4 tensors).
        Nkeep: int - Max bond dimension.
        Nsweep: int - Number of full DMRG sweeps.
        Krylov: int - Krylov subspace dimension (default 5).
        tol: float - Lanczos convergence threshold (default 1e-8).

    Returns:
        M: List[np.ndarray] - Excited state MPS (left-canonical form).
        E1: float - Energy of the excited state.
        Eiter: np.ndarray - Energy values per iteration.
        Sv: List[np.ndarray] - Singular values across bonds.
    """
    N = len(M)
    Eiter = np.zeros((N, 2 * Nsweep))
    Sv = [None] * (N + 1)

    # Step 1: Bring M to left-canonical form
    M, _, _ = canon_form(M, N - 1, None, 0)

    # Step 2: Initialize contractions
    Hlr = [None] * (N + 2)
    Olr = [None] * (N + 2)
    Hlr[0] = np.array([1]).reshape(1, 1, 1)
    Hlr[-1] = np.array([1]).reshape(1, 1, 1)
    Olr[0] = np.array([1]).reshape(1, 1)
    Olr[-1] = np.array([1]).reshape(1, 1)

    for itN in range(N):
        Hlr[itN + 1] = update_left(Hlr[itN], 3, M[itN], Hs[itN], 4, M[itN])
        Olr[itN + 1] = update_left(Olr[itN], 2, M[itN], None, None, M0[itN])

    for itS in range(Nsweep):
        # Right-to-left sweep
        for itN in reversed(range(N)):
            Aorth = contract(Olr[itN], 1, M0[itN], 0)
            Aorth = contract(Aorth, 1, Olr[itN + 2], 1, [0, 2, 1])
            M[itN], Eiter[N - 1 - itN, 2 * itS] = eigs_1site_1ES(
                Hlr[itN], Hs[itN], Hlr[itN + 2], M[itN], Aorth, Krylov, tol)

            U, Sv[itN], M[itN], _ = svd_tr(M[itN], 3, [0], Nkeep, 0)
            if itN > 0:
                M[itN - 1] = contract(M[itN - 1], 1, U @ np.diag(Sv[itN]), 0, [0, 2, 1])
            else:
                M[itN] = contract(U, 1, M[itN], 0)

            Hlr[itN + 1] = update_left(Hlr[itN + 2], 3, M[itN].transpose(1, 0, 2),
                                       Hs[itN].transpose(0, 1, 3, 2), 4, M[itN].transpose(1, 0, 2))
            Olr[itN + 1] = update_left(Olr[itN + 2], 2, M[itN].transpose(1, 0, 2),
                                       None, None, M0[itN].transpose(1, 0, 2))

        # Left-to-right sweep
        for itN in range(N):
            Aorth = contract(Olr[itN], 1, M0[itN], 0)
            Aorth = contract(Aorth, 1, Olr[itN + 2], 1, [0, 2, 1])
            M[itN], Eiter[itN, 2 * itS + 1] = eigs_1site_1ES(
                Hlr[itN], Hs[itN], Hlr[itN + 2], M[itN], Aorth, Krylov, tol)

            M[itN], Sv[itN + 1], Vd, _ = svd_tr(M[itN], 3, [0, 2], Nkeep, 0)
            M[itN] = M[itN].transpose(0, 2, 1)

            if itN < N - 1:
                M[itN + 1] = contract(np.diag(Sv[itN + 1]) @ Vd, 1, M[itN + 1], 0)
            else:
                M[itN] = contract(M[itN], 1, Vd, 0, [0, 2, 1])

            Hlr[itN + 1] = update_left(Hlr[itN], 3, M[itN], Hs[itN], 4, M[itN])
            Olr[itN + 1] = update_left(Olr[itN], 2, M[itN], None, None, M0[itN])

    E1 = Eiter[N - 1, 2 * itS]
    return M, E1, Eiter, Sv

def eigs_1site_1ES(Hleft, Hcen, Hright, Aold, Aorth, nKrylov, tol):
    """
    Lanczos-based eigenvalue solver for 1-site DMRG with orthogonality constraint.

    Args:
        Hleft, Hcen, Hright : np.ndarray
            Effective Hamiltonian tensors (left, center MPO, and right).
        Aold : np.ndarray
            Current MPS tensor (rank-3).
        Aorth : np.ndarray
            Tensor to be orthogonal to (ground state component).
        nKrylov : int
            Max Krylov subspace size.
        tol : float
            Convergence threshold.

    Returns:
        Anew : np.ndarray
            Updated excited state tensor (rank-3).
        Enew : float
            Excited state energy.
    """

    shape = Aold.shape
    As = np.zeros(shape + (nKrylov + 1,), dtype=Aold.dtype)

    # Normalize Aorth
    Aorth_norm = np.sqrt(abs(contract(np.conj(Aorth), (0,1,2), Aorth, (0,1,2))))
    Aorth = Aorth / Aorth_norm

    # Orthogonalize Aold against Aorth
    coeff = contract(np.conj(Aorth), (0,1,2), Aold, (0,1,2))
    Aold = Aold - coeff * Aorth
    coeff = contract(np.conj(Aorth), (0,1,2), Aold, (0,1,2))
    Aold = Aold - coeff * Aorth
    Aold = Aold / np.sqrt(abs(contract(np.conj(Aold), (0,1,2), Aold, (0,1,2))))

    As[..., 0] = Aorth
    As[..., 1] = Aold

    alphas = np.zeros(nKrylov)
    betas = np.zeros(nKrylov - 1)
    cnt = 0

    for itn in range(nKrylov):
        Amul = contract(Hleft, 1, As[..., itn + 1], 0)
        Amul = contract(Amul, [1, 3], Hcen, [2, 1])
        Amul = contract(Amul, [1, 3], Hright, [1, 2], permute_order=[0, 2, 1])

        alphas[itn] = np.real(contract(np.conj(As[..., itn + 1]), (0, 1, 2), Amul, (0, 1, 2)))
        cnt += 1

        if itn < nKrylov - 1:
            for _ in range(2):
                T = contract(np.conj(As[..., :itn + 2]), (0, 1, 2), Amul, (0, 1, 2))
                T = contract(As[..., :itn + 2], 3, T, 0)
                Amul -= T

            Anorm = np.sqrt(abs(contract(np.conj(Amul), (0, 1, 2), Amul, (0, 1, 2))))
            if Anorm < tol:
                break

            As[..., itn + 2] = Amul / Anorm
            betas[itn] = Anorm

    # Construct tridiagonal matrix
    Hkrylov = np.diag(betas[:cnt - 1], -1) + np.diag(alphas[:cnt]) + np.diag(betas[:cnt - 1], 1)

    eigvals, eigvecs = scipy.linalg.eigh(Hkrylov)
    Anew = contract(As[..., 1:cnt + 1], 3, eigvecs[:, 0], 0)

    # Final energy
    Amul = contract(Hleft, 1, Anew, 0)
    Amul = contract(Amul, [1, 3], Hcen, [2, 1])
    Amul = contract(Amul, [1, 3], Hright, [1, 2], permute_order=[0, 2, 1])
    Enew = np.real(contract(np.conj(Anew), (0, 1, 2), Amul, (0, 1, 2)))

    return Anew, Enew


def iTEBD_GS_Vidal(Lambda, Gamma, H, Nkeep, taus):
    """
    iTEBD ground-state search (infinite MPS, two-site unit cell) by imaginary
    time evolution (Vidal's method).

    Args:
        Lambda : list of 2 np.ndarray, each shape (χ,)
        Gamma  : list of 2 np.ndarray, each shape (χ_prev, χ_next, d)
        H      : np.ndarray, shape (d,d,d,d) two-site Hamiltonian
        Nkeep  : int, maximum bond dimension
        taus   : array-like of imag. time steps

    Returns:
        Lambda, Gamma : updated tensors
        Eiter         : np.ndarray, shape (len(taus), 2, 2)
                        Eiter[m,n,k] = energy on bond k (0=odd,1=even)
                          after updating bonds of type n (0=odd,1=even)
                          in outer iteration m.
    """
    # --- prep and checks ---
    Lambda = list(Lambda)
    Gamma  = list(Gamma)
    Nstep = len(taus)
    d = H.shape[0]
    Skeep = 1e-8

    if len(Lambda) != 2 or len(Gamma) != 2:
        raise ValueError("Need exactly two-site unit cell (len=2).")
    if H.ndim != 4 or any(sz != d for sz in H.shape):
        raise ValueError("H must be rank-4 with each leg size = d.")

    for it in (0, 1):
        if Lambda[it].ndim != 1:
            raise ValueError(f"Lambda[{it}] must be a vector.")
        if Lambda[it].size != Gamma[it].shape[1]:
            raise ValueError("Dimension mismatch between Lambda and Gamma.")
        if Lambda[1-it].size != Gamma[it].shape[0]:
            raise ValueError("Dimension mismatch between Lambda and Gamma.")
        if Gamma[it].shape[2] != d:
            raise ValueError("Physical leg of Gamma must have size d.")

    # --- storage for measured energies ---
    Eiter = np.zeros((Nstep, 2, 2))

    # --- diagonalize H for exp(-τ H) ---
    # reshape H_{ij,kl} = H(i,k,j,l)
    Hmat = H.transpose(0,2,1,3).reshape(d*d, d*d)
    evals, evecs = scipy.linalg.eigh((Hmat + Hmat.T.conj())/2)
    # iterate
    for m, τ in enumerate(taus):
        # build the two-site gate
        expH = (evecs @ np.diag(np.exp(-τ*evals)) @ evecs.T).reshape(d, d, d, d)

        for parity in (0, 1):  # 0 = odd bonds, 1 = even bonds
            other = 1 - parity

            # ---- build 4-leg ket T ----
            T = contract(np.diag(Lambda[other]), 1, Gamma[parity], 0)
            T = contract(T, 1, np.diag(Lambda[parity]), 0)
            T = contract(T, 2, Gamma[other], 0)
            T = contract(T, 2, np.diag(Lambda[other]), 0)

            # ---- apply gate ----
            # gate legs (2,3) act on T legs (1,2)
            eHT = contract(expH, [2,3], T, [1,2], permute_order=[2,0,1,3])

            # ---- SVD + truncation ----
            # T_rank = 4, idU = [0,1] (the two bond legs)
            U, S, Vt, _ = svd_tr(eHT, 4, (0,1), Nkeep, Skeep)
            # reorder to (left, right, phys)
            U  = U.transpose(0,2,1)
            Vt = Vt.transpose(0,2,1)

            # renormalize singulars
            S = S/np.linalg.norm(S)
            Lambda[parity] = S

            # ---- update Gammas ----
            Gamma[parity] = contract(np.diag(1.0/Lambda[other]), 1, U, 0)
            Gamma[other]  = contract(Vt, 1, np.diag(1.0/Lambda[other]), 0, permute_order=[0,2,1])

            # ---- measure energies ----
            # build three pieces TA, TB, TC depending on parity
            if parity == 0:
                TA = U
                TB = contract(np.diag(Lambda[0]), 1, Vt, 0)
                TC = contract(Gamma[0], 1, np.diag(Lambda[0]), 0, permute_order=[0,2,1])
            else:
                TA = contract(np.diag(Lambda[1]), 1, Gamma[0], 0)
                TB = contract(U, 1, np.diag(Lambda[1]), 0, permute_order=[0,2,1])
                TC = Vt

            # odd-bond energy Ho
            X = contract(TA, 1, TB, 0, permute_order=[0,1,3,2])
            Ho = contract(H, [1,3], X, [1,2], permute_order=[2,0,1,3])
            Ho = contract(np.conj(X), (0,1,2), Ho, (0,1,2))
            Ho = contract(Ho, 1, TC, 0)
            Ho = contract(np.conj(TC), (0,1,2), Ho, (0,1,2))

            # even-bond energy He
            Y = contract(TB, 1, TC, 0, permute_order=[0,1,3,2])
            He = contract(H, [1,3], Y, [1,2], permute_order=[2,0,1,3])
            He = contract(np.conj(Y), (1,2,3), He, (1,2,3))
            He = contract(TA, 1, He, 1)
            He = contract(np.conj(TA), (0,2,1), He, (0,1,2))

            # normalization denominator
            ovl = update_left(None, None, TA, None, None, TA)
            ovl = update_left(ovl, 2, TB, None, None, TB)
            ovl = contract(ovl, 1, TC, 0)
            ovl = contract(np.conj(TC), (0,1,2), ovl, (0,1,2))

            Eiter[m, parity, 0] = Ho / ovl
            Eiter[m, parity, 1] = He / ovl

    return Lambda, Gamma, Eiter

def iTEBD_ES_Vidal(L_gs, G_gs, L_ex, G_ex, H, Nkeep, taus):
    """
    iTEBD imaginary-time evolution targeting the first excited state by
    orthogonalizing against a fixed ground-state MPS (L_gs, G_gs).

    Args:
        L_gs, G_gs : list of two arrays each
            Ground-state Λ and Γ tensors for odd/even bonds/sites.
        L_ex, G_ex : list of two arrays each
            Initial guess for excited-state Λ and Γ.
        H          : ndarray, shape (d,d,d,d)
            Two-site Hamiltonian tensor.
        Nkeep      : int
            Maximum bond dimension.
        taus       : array-like
            Sequence of imaginary time steps.

    Returns:
        L_ex, G_ex : updated excited-state MPS tensors
        Eiter_ex   : ndarray, shape (len(taus),2,2)
            Eiter_ex[m,n,k] is the energy on bond k (0=odd,1=even)
            after updating bonds of type n (0=odd,1=even) at time step m.
    """
    d = H.shape[0]
    Skeep = 1e-8
    Nstep = len(taus)
    Eiter_ex = np.zeros((Nstep, 2, 2))

    # pre-diagonalize H for exp(-τ H)
    Hmat = H.transpose(0,2,1,3).reshape(d*d, d*d)
    evals, evecs = scipy.linalg.eigh((Hmat + Hmat.T.conj())/2)

    for m, τ in enumerate(taus):
        # two-site gate
        expH = (evecs @ np.diag(np.exp(-τ * evals)) @ evecs.T).reshape(d, d, d, d)

        for parity in (0, 1):
            other = 1 - parity

            # build 2-site ket for GS and EX
            T_gs = contract(np.diag(L_gs[other]), 1, G_gs[parity], 0)
            T_gs = contract(T_gs, 2, np.diag(L_gs[parity]), 0)
            T_gs = contract(T_gs, 2, G_gs[other], 0)
            T_gs = contract(T_gs, 3, np.diag(L_gs[other]), 0)

            T_ex = contract(np.diag(L_ex[other]), 1, G_ex[parity], 0)
            T_ex = contract(T_ex, 2, np.diag(L_ex[parity]), 0)
            T_ex = contract(T_ex, 2, G_ex[other], 0)
            T_ex = contract(T_ex, 3, np.diag(L_ex[other]), 0)

            # apply gate
            eHT_gs = contract(expH, [2,3], T_gs, [2,3], permute_order=[2,0,1,3])
            eHT_ex = contract(expH, [2,3], T_ex, [2,3], permute_order=[2,0,1,3])

            # project out ground-state component
            vec_gs = eHT_gs.reshape(-1)
            vec_ex = eHT_ex.reshape(-1)
            # Should repeat here
            alpha_ = np.vdot(vec_gs, vec_ex)
            beta_ = np.vdot(vec_gs, vec_gs)
            proj = vec_ex - (alpha_/beta_) * vec_gs
            eHT_proj = proj.reshape(eHT_ex.shape)

            # SVD + truncate
            U, S, Vt = svd_tr(eHT_proj, 4, [0,1], Nkeep, Skeep)
            U  = U.transpose(0,2,1)
            Vt = Vt.transpose(0,2,1)
            S  = S / np.linalg.norm(S)
            L_ex[parity] = S

            # update excited Γ tensors
            G_ex[parity] = contract(np.diag(1.0/L_ex[other]), 1, U, 0)
            G_ex[other]  = contract(Vt, 2, np.diag(1.0/L_ex[other]), 1,
                                    permute_order=[0,2,1])

            # energy measurement on odd/even bonds
            # build TA, TB, TC for excited state
            if parity == 0:
                TA = U
                TB = contract(np.diag(L_ex[0]), 1, Vt, 0)
                TC = contract(G_ex[0], 2, np.diag(L_ex[0]), 1, permute_order=[0,2,1])
            else:
                TA = contract(np.diag(L_ex[1]), 1, G_ex[0], 0)
                TB = contract(U, 2, np.diag(L_ex[1]), 1, permute_order=[0,2,1])
                TC = Vt

            # odd-bond energy
            X = contract(TA, 2, TB, 2, permute_order=[0,1,3,2])
            Ho = contract(H, [1,3], X, [1,2], permute_order=[2,0,1,3])
            Ho = contract(np.conj(X), (0,1,2), Ho, (0,1,2))
            Ho = contract(Ho, 1, TC, 0)
            Ho = contract(np.conj(TC), (0,1,2), Ho, (0,1,2))

            # even-bond energy
            Y = contract(TB, 2, TC, 2, permute_order=[0,1,3,2])
            He = contract(H, [1,3], Y, [1,2], permute_order=[2,0,1,3])
            He = contract(np.conj(Y), (1,2,3), He, (1,2,3))
            He = contract(TA, 2, He, 1)
            He = contract(np.conj(TA), (0,2,1), He, (0,1,2))

            # normalization denominator
            ovl = update_left(None, None, TA, None, None, TA)
            ovl = update_left(ovl, 2, TB, None, None, TB)
            ovl = contract(ovl, 2, TC, 0)
            ovl = contract(np.conj(TC), (0,1,2), ovl, (0,1,2))

            Eiter_ex[m, parity, 0] = Ho / ovl
            Eiter_ex[m, parity, 1] = He / ovl

    return L_ex, G_ex, Eiter_ex
