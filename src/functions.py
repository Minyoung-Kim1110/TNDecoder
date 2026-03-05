import numpy as np 
import numpy.linalg as lin 
from typing import List, Tuple

MATLAB_style=False
order_type = 'F' if MATLAB_style else 'C' # call the reshape function with order_type 



def check_equality_tensor(A: np.array, B : np.array, tol = 10 ** ( -15) ): 
    """check equality of tensors A and B 
        Return True if A == B upto tolerance(tol)

    Args:
        A (np.array): a tensor 
        B (np.array): a tensor 
        tol (float, optional): criteria for checking equality of double precision variables. Defaults to 10**( -15).

    Returns:
        (Boolean): Whether two tensors are equal or not 

    """
    if A.shape != B.shape:
        return False 
    if np.sum(np.abs(A-B)>tol)>1:
        return False 
    else:
        return True 

def disptime(message: str):
    """Simple print utility for time-stamped messages."""
    print(f"[{np.datetime_as_string(np.datetime64('now'), unit='s')}] {message}")

def tic2():
    """Start timer."""
    return np.datetime64('now')

def toc2(tobj: np.datetime64, options: str = ''):
    """End timer and print duration."""
    duration = np.datetime64('now') - tobj
    print(f"Execution time: {duration}")

def entropy(s: np.array)->float: 
    """from singular values, compute entropy 

    Args:
        s (np.array): singluar values 

    Returns:
        entropy (float): Shannon entropy with log_2
    
    """
    s = s*s 
    return - np.dot(s, np.log2(s))

def get_MPS_QR(tensor:np.array)->List[np.array]:
    """Decompose a high rank tensor to the matrix product states(MPS) by QR factorization 
    
    Args:
        tensor (np.array): a high rank tensor

    Returns:
        MPS (List[np.array]): Matrix product state of a given high rank tensor 
    
    """
    tensor_dim = list(tensor.shape)
    MPS = []
    R = tensor 
    szl = 1 # bond dimension of left leg of MPS[i]
    for i in range(len(tensor_dim)-1):
        R = R.reshape((szl * tensor_dim[i], np.prod(tensor_dim[i+1:])), order=order_type)
        Q, R = lin.qr(R, mode = 'reduced')
        Q = np.transpose(Q.reshape((szl, tensor_dim[i], -1), order=order_type), (0, 2, 1))
        MPS.append(Q)
        (_, szl, _)  = Q.shape 
        R = R.reshape((tensor_dim[i+1:].insert(0, szl)), order=order_type)
    MPS.append(np.transpose(R[:, np.newaxis], (0, 2, 1)))
    return MPS

def get_MPS_SVD(tensor:np.array, tol = 10**(-16))->Tuple[List[np.array], List[float]]:
    """Decompose a high rank tensor to the matrix product states(MPS) by SVD decomposition  
    
    Args:
        tensor (np.array): a high rank tensor
        tol (float) : tolerance for floating point noise

    Returns:
        MPS, entropys (Tuple[List[np.array], List[float]]): 
        MPS: Matrix product state of a given high rank tensor
        entropys: entropy of each bipartition
    
    """
    tensor_dim = list(tensor.shape) 
    MPS = []
    entropys = []
    szl = 1 
    A = tensor
    for i in range(len(tensor_dim)-1):
        A = A.reshape((szl*tensor_dim[i], np.prod(tensor_dim[i+1:])), order=order_type)
        U, S, Vh = lin.svd(A, full_matrices=False)
        S_filtered = S>tol
        U , S, Vh= U[:, S_filtered], S[S_filtered], Vh[S_filtered, :]
        entropys.append(entropy(S))
        U = np.transpose(U.reshape((szl, tensor_dim[i], -1), order=order_type), (0, 2, 1))
        MPS.append(U)
        (_, szl, _) = U.shape
        A = (np.diag(S)@Vh).reshape((tensor_dim[i+1:].insert(0, szl)), order=order_type)
    MPS.append(A[:, np.newaxis, :])
    return (MPS, entropys)

def MPS_to_tensor(MPS:List[np.array]):
    """Reconstruct the high rank tensor from Matrix product states 

    Args:
        MPS (List[np.array]): matrix product states

    Returns:
        A (np.array):a high rank tensor

    """
    A = MPS[0]
    for i in range(1, len(MPS)):
        # rank = len(A.shape)
        A = contract(A, np.transpose(MPS[i], (0, 2, 1)), [len(A.shape)-1], [0])
    
    return A.squeeze()


def get_identity(B, idB, *args):
    """
    Obtain the identity tensor in the space of the idB-th leg of B.
    
    Usage 1:
        A = get_identity(B, idB)
        - Returns the identity tensor in the space of the idB-th axis of B.
    
    Usage 2:
        A = get_identity(B, idB, C, idC)
        - Returns the identity tensor in the direct product space of the
          Hilbert space of the idB-th axis of B and the idC-th axis of C.
    
    Optional:
        If additional permutation indices `p` are provided, the identity
        tensor is permuted accordingly.
    
    Parameters:
        B : np.ndarray
            A tensor.
        idB : int
            Index (axis) of B.
        C : np.ndarray (optional)
            Another tensor.
        idC : int (optional)
            Index (axis) of C.
        p : list (optional)
            Permutation indices for the identity tensor.
    
    Returns:
        A : np.ndarray
            Identity tensor with appropriate shape and permutation.
    """
    
    # Default values
    C = None
    idC = None
    p = None
    
    # Handle additional arguments
    if len(args) >= 2:
        C, idC = args[:2]  # Extract C and idC
        args = args[2:]  # Remove processed arguments
    
    # Check if a permutation is given
    if len(args) > 0:
        p = args[0]

    # Get the size of the idB-th axis of B
    DB = B.shape[idB]
    
    if C is not None:
        # Get the size of the idC-th axis of C
        DC = C.shape[idC]
        # Create an identity matrix and reshape it to the appropriate tensor
        A = np.eye(DB * DC).reshape(DB, DC, DB * DC)
    else:
        # Create an identity matrix
        A = np.eye(DB)
    
    # Apply permutation if provided
    if p is not None:
        if len(p) < A.ndim:
            raise ValueError("ERR: The number of elements in permutation 'p' is smaller than the rank of 'A'.")
        A = np.transpose(A, axes=p)
    
    return A 

def contract(A, idA, B, idB, permute_order=None):
    """
    Performs tensor contraction using NumPy's tensordot
    
    Parameters:
        A, B : np.ndarray
            Tensors to be contracted.
        rank1, rank2 : int
            Ranks (dimensions) of the tensors.
        idA, idB : int or list
            Indices of tensor contraction.
        permute_order : list (optional)
            If provided, the result will be permuted to this order.
    
    Returns:
        result : np.ndarray
            The contracted tensor.
    """
    result = np.tensordot(A, B, axes=(idA, idB))
    if permute_order is not None:
        result = np.transpose(result, axes=permute_order)
    return result

def update_left(Cleft, rankC, B, X, rankX, A):
    """
    Contract the operator Cleft that acts on the left part of an MPS 
    with tensors B, X, and A acting on a given site.
                     1     2
           /--------->- A ->--            /---->-- 2
           |            | 3               |
         2 ^            ^                 |
           |            | 2               |      
         Cleft          X         =>    Cleft 
           |            | 1               |
         1 ^            ^                 |
           |            | 3               |
           \---------<- B'-<--            \----<-- 1
                     2     1
        * When Cleft is rank-3 and X is rank-2:
                     1     2
           /--------->- A ->--            /---->-- 2
           |            | 3               |
         2 ^            ^                 |
           |    3       | 2               |      
         Cleft---       X         =>    Cleft ---- 3
           |            | 1               |
         1 ^            ^                 |
           |            | 3               |
           \---------<- B'-<--            \----<-- 1
                     2     1
        * When Cleft is rank-2 and X is rank-3:
                     1     2
           /--------->- A ->--            /---->-- 2
           |            | 3               |
         2 ^            ^                 |
           |          2 |   3             |      
         Cleft          X ----    =>    Cleft ---- 3
           |          1 |                 |
         1 ^            ^                 |
           |            | 3               |
           \---------<- B'-<--            \----<-- 1
                     2     1
        * When both Cleft and X are rank-3:
                     1     2
           /--------->- A ->--            /---->-- 2
           |            | 3               |
         2 ^            ^                 |
           |   3     3  | 2               |      
         Cleft--------- X         =>    Cleft
           |            | 1               |
         1 ^            ^                 |
           |            | 3               |
           \---------<- B'-<--            \----<-- 1
                     2     1
        * When Cleft is rank-3 and X is rank-4:
                     1     2
           /--------->- A ->--            /---->-- 2
           |            | 3               |
         2 ^            ^                 |
           |   3    3   | 2               |      
         Cleft--------- X ---- 4   =>   Cleft ---- 3
           |            | 1               |
         1 ^            ^                 |
           |            | 3               |
           \---------<- B'-<--            \----<-- 1
    Parameters:
        Cleft : np.ndarray or None
            Rank-2 or 3 tensor from the left part of the system.
        rankC : int or None
            Rank of Cleft.
        B, A : np.ndarray
            Ket tensors with leg order: left-right-bottom.
        X : np.ndarray or None
            Local operator (rank 2, 3, or 4).
        rankX : int or None
            Rank of X.
    
    Returns:
        Cleft : np.ndarray
            The contracted tensor.
    """
    
    # Handle empty cases (equivalent to identity tensors)
    if Cleft is None:
        rankC = 2  # Default to rank-2 identity
        Cleft = np.array([1]).reshape(1,1)
        CleftNone = True 
    else: 
        CleftNone= False 
        
    if X is None:
        rankX = 2  # Default to rank-2 identity
        X = np.array([1]).reshape(1,1)
        XNone = True
    else: 
        XNone = False
    # Sanity check: Ensure valid ranks
    valid_ranks = [(2, 2), (3, 2), (2, 3), (3, 3), (3, 4)]
    if not ((rankC, rankX) in valid_ranks or Cleft is None or X is None):
        raise ValueError("ERR: Invalid ranks of C and X.")

    # Take Hermitian conjugate of B (complex conjugate but no permute)
    B = np.conjugate(B)
    # Case: X is not empty
    if XNone is False: 
        T = contract(X, 1, A,  2)  # Contract X (bottom-top) with A (physical leg)
        if CleftNone is False:
            if rankC > 2 and rankX > 2:
                if rankX == 4:  # (rankC, rankX) = (3,4)
                    T = contract(Cleft,  [rankC - 1, 1], T,  [1, rankX - 1])
                    Cleft = contract(B,  [0, 2], T,  [0, 1], [0, 2, 1])
                else:  # (rankC, rankX) = (3,3)
                    T = contract(Cleft,  [rankC - 1, 1], T, [1, rankX - 1])
                    Cleft = contract(B,  [0, 2], T,  [0, 1])
            else:  # Cases: (2,2), (2,3), (3,2)
                T = contract(Cleft,  1, T,  rankX - 1)
                Cleft = contract(B, [0, 2], T, [0, rankC - 1], [0] + list(range(2, rankC)) + [1])
        
        elif rankX == 4 and X.shape[0] == 1:  # No Cleft, rankX = 4
            Cleft = contract(B,  [0, 2], T,  [rankX - 1, 1], [0, 3, 2, 1])
        else:  # No Cleft, rankX = 2,3
            Cleft = contract(B,  [0, 2], T,  [rankX - 1, 0], [0] + list(range(2, rankX)) + [1])
    
    # Case: No X, but Cleft exists
    elif CleftNone is False:
        T = contract(Cleft,  1, A,  0)
        Cleft = contract(B,  [0, 2], T,  [0, rankC], [0] + list(range(2, rankC)) + [1])
    
    # Case: No Cleft, No X
    else:
        Cleft = contract(B,  [0, 2], A,  [0, 2])

    return Cleft
 
def svd_tr(T, rankT, idU, Nkeep=None, Skeep=None):
    """
    Singular Value Decomposition (SVD) of a tensor such that T = U * diag(S) * Vd.
    
    Truncates based on the number of singular values (Nkeep) or their magnitude (Skeep).
    
    Parameters:
        T : np.ndarray
            Input tensor.
        rankT : int
            Rank of tensor T.
        idU : list or np.ndarray
            Indices of T to be associated with U.
        Nkeep : int or None, optional
            Max number of singular values to keep. Default is None (no truncation by count).
        Skeep : float or None, optional
            Minimum singular value to keep. Default is `10 * np.finfo(S.max()).eps`.
    
    Returns:
        U : np.ndarray
            Left singular vectors.
        S : np.ndarray
            Singular values.
        Vd : np.ndarray
            Right singular vectors.
        dw : float
            Discarded weight (sum of squared truncated singular values).
    """
    
    # Sanity check: rankT should not be smaller than the actual tensor rank
    if rankT < T.ndim:
        raise ValueError("ERR: Input 'rankT' is smaller than the actual rank of 'T'.")
    
    # Ensure idU indices are valid
    if not all(idx in range(0, rankT) for idx in idU):
        raise ValueError("ERR: Invalid index for tensor U (out of bound, non-integer).")
    
    # Get tensor dimensions
    Tsz = np.ones(rankT, dtype=int)
    Tsz[:T.ndim] = np.array(T.shape)  # Fill with tensor shape
    
    # Determine indices for Vd
    idV = [i for i in range(0, rankT ) if i not in idU]

    # Reshape tensor into matrix form
    permuted_T = np.transpose(T, axes=[i  for i in idU] + [i  for i in idV])
    T2 = permuted_T.reshape((np.prod(Tsz[np.array(idU) ]), np.prod(Tsz[np.array(idV) ])))
    # Perform SVD
    U2, S2, V2 = np.linalg.svd(T2, full_matrices=False)
    # Default truncation parameters
    if Nkeep is None:
        Nkeep = float('inf')  # Keep as float for comparison, not integer

    Nkeep = min(Nkeep, len(S2))  # Ensure Nkeep is at most the number of singular values
    Nkeep = int(Nkeep)  # Convert to a valid integer for slicing

    if Skeep is None and S2.size > 0:
        Skeep = 10 * np.finfo(S2.max()).eps  # Set Skeep threshold based on machine epsilon
    
    # Determine singular values to keep
    oks = (S2 >= Skeep)
    oks[min(len(S2), int(Nkeep)):] = False  # Apply Nkeep limit
    # S2 = np.diag(S2)  # Convert singular values to diagonal matrix
    
    # Compute discarded weight (sum of squared truncated singular values)
    dw = np.sum(S2[~oks] ** 2)
    
    # Truncate singular values and vectors
    S = S2[oks]
    U = U2[:, oks]
    U = U.reshape(list(Tsz[np.array(idU)]) + [len(S)])
    Vd = V2[oks, :].reshape([len(S)] + list(Tsz[np.array(idV)]))

    return U, S, Vd, dw

def canon_form(M, id, Nkeep=None, Skeep=None):
    """
    Convert an MPS (Matrix Product State) into canonical form.

    Depending on the bond index `id`, the function brings the left part of MPS into
    left-canonical form and the right part into right-canonical form.

    Parameters:
        M : list of np.ndarray
            List of rank-3 tensors representing an MPS.
        id : int
            Index of the bond between M[id] and M[id+1] for splitting into canonical form.
        Nkeep : int, optional
            Max number of singular values to keep at each SVD. Default is None (no truncation by count).
        Skeep : float, optional
            Minimum singular value to keep at each SVD. Default is `10 * np.finfo(S.max()).eps`.

    Returns:
        M : list of np.ndarray
            Updated MPS in left-, right-, or bond-canonical form.
        S : np.ndarray
            Singular values at the bond between M[id] and M[id+1].
        dw : np.ndarray
            Discarded weight (sum of squared truncated singular values) for each bond.
    """

    # Input validation
    if not isinstance(id, int) or id < 0 or id > len(M)-1:
        raise ValueError("ERR: The 2nd input 'id' must be an integer in range [0, len(M)).")
    
    if M[0].shape[0] != 1:
        raise ValueError("ERR: The first dimension (left leg) of M[0] should be 1.")
    
    if M[-1].shape[1] != 1:
        raise ValueError("ERR: The second dimension (right leg) of M[-1] should be 1.")

    # Initialize discarded weight vector
    dw = np.zeros(len(M))

    # Convert left part of MPS into left-canonical form
    for it in range(id):
        # SVD decomposition
        M[it], S, Vd, dw[it] = svd_tr(M[it], 3, [0, 2], Nkeep, Skeep)
        M[it] = np.transpose(M[it], axes=[0, 2, 1])

        # Contract S and Vd with the next tensor M[it+1]
        M[it + 1] = contract(np.diag(S) @ Vd,  1, M[it + 1],  0)

    # Convert right part of MPS into right-canonical form
    for it in range(len(M) - 1, id, -1):
    
        # SVD decomposition
        U, S, M[it], dw[it - 1] = svd_tr(M[it], 3, [0], Nkeep, Skeep)
        # Contract U and S with the previous tensor M[it-1]
        
        M[it - 1] = contract(M[it - 1],  1, U @ np.diag(S), 0,  permute_order=[0, 2, 1])

    # Purely right-canonical form (id == 0)
    if id == 0:
        U, S, M[0], _ = svd_tr(M[0], 3, [0], None, 0)  # No truncation
        M[0] = contract(U,  1, M[0],  0)

    # Purely left-canonical form (id == len(M)-1)
    elif id == len(M)-1:
        M[-1], S, Vd, _ = svd_tr(M[-1], 3, [0, 2], None, 0)  # No truncation
        M[-1] = contract(M[-1], 2, Vd,  0, permute_order=[0, 2, 1])

    # Bond-canonical form
    else:
        T = contract(M[id],  1, M[id+1],  0)
        M[id], S, M[id+1], dw[id] = svd_tr(T, 4, [0, 1], Nkeep, Skeep)
        M[id] = np.transpose(M[id], axes=[0, 2, 1])

    return M, S, dw






   
    # Placeholder for utility functions that exist in the original MATLAB environment

def get_local_space(op_type: str, s = None) :
    """
    Generates local operators as tensors based on the specified space.

    Args:
        op_type (str): Type of space: 'Spin', 'Fermion', or 'FermionS'.
        s (float, optional): Spin value for 'Spin' case (e.g., 0.5, 1, 1.5).

    Returns:
        List[np.ndarray]: List of generated local operators (S, I, F, Z, etc.).
    """
    
    # --- Input Parsing and Validation ---
    
    # Check if op_type is valid
    valid_types = {'Spin', 'Fermion', 'FermionS'}
    if op_type not in valid_types:
        raise ValueError(f"ERR: Input #1 should be one of {valid_types}.")

    is_fermion = op_type in {'Fermion', 'FermionS'}
    is_spin_op = op_type in {'Spin', 'FermionS'}
    
    # Default Identity matrix size
    if op_type == 'Spin':
        if s is None:
            raise ValueError("ERR: For 'Spin', input 's' is required.")
        # Check if s is positive (half-)integer
        s_rounded = np.round(2 * s) / 2.0
        if np.abs(2 * s - 2 * s_rounded) > np.finfo(float).eps * 10 or s <= 0:
            raise ValueError("ERR: Input 's' for 'Spin' should be positive (half-)integer.")
        s = s_rounded
        I = np.eye(int(2 * s + 1))
    elif op_type == 'Fermion':
        I = np.eye(2)
    elif op_type == 'FermionS':
        I = np.eye(4)
    
    # --- Operator Generation ---
    
    S = np.array([])
    F = np.array([])
    Z = np.array([])
    
    if is_fermion:
        if op_type == 'FermionS': # spinful fermion
            # Basis: |vac>, c'_up|vac>, c'_down|vac>, c'_down c'_up|vac> (0, 1, 2, 3)
            F = np.zeros((4, 4, 2), dtype=np.complex128)
            
            # F(:,:,0) [index 0]: spin-up annihilation f_up
            F[0, 1, 0] = 1.0 
            F[2, 3, 0] = -1.0 # -1 sign due to anticommutation f_up c'_down |up down> = -c'_down f_up |up down>
            
            # F(:,:,1) [index 1]: spin-down annihilation f_down
            F[0, 2, 1] = 1.0
            F[1, 3, 1] = 1.0 # f_down c'_up |up down> = c'_up f_down |up down>
            
            # Z: Jordan-Wigner string operator (-1)^(N_up + N_down)
            Z = np.diag([1.0, -1.0, -1.0, 1.0])
            
            # S: Spin operators (S+/sqrt(2), Sz, S-/sqrt(2))
            S = np.zeros((4, 4, 3), dtype=np.complex128)
            
            # Correct Spin Operators for the two-orbital (4-site) space:
            # S+ = f_up^\dagger f_down + f_down^\dagger f_up
            
            S[1, 2, 0] = 1.0 / np.sqrt(2.0)
            S[:, :, 2] = S[:, :, 0].conj().T
            S[1, 1, 1] = +0.5
            S[2, 2, 1] = -0.5
            
        else: # spinless fermion
            # Basis: |vac>, |occ> (0, 1)
            F = np.zeros((2, 2, 1), dtype=np.float64)
            F[0, 1, 0] = 1.0 # Annihilation f
            Z = np.diag([1.0, -1.0])
            
    elif op_type == 'Spin': # pure spin (no fermion operators)
        # Basis: +s, +s-1, ..., -s
        
        # Sp: Spin raising S+
        m_values = np.arange(s - 1, -s - 1, -1) # s-1, s-2, ..., -s
        Sp_diag = np.sqrt((s - m_values) * (s + m_values + 1))
        Sp = np.diag(Sp_diag, k=1)
        
        # Sz: Spin-z S_z
        Sz = np.diag(np.arange(s, -s - 1, -1))
        
        # S: cat(3, Sp/sqrt(2), Sz, Sp'/sqrt(2))
        S_plus_scaled = Sp / np.sqrt(2.0)
        S_minus_scaled = Sp.T / np.sqrt(2.0) # Sp' in MATLAB is transpose (conjugate not needed for real matrix)
        
        S = np.stack([S_plus_scaled, Sz, S_minus_scaled], axis=2)
        
    # --- Assign Output ---
    if op_type == 'FermionS':
        return [F, Z, S, I]
    elif op_type == 'Fermion':
        return [F, Z, I]
    elif op_type == 'Spin':
        return [S, I]



if __name__ == '__main__':
    dim = [2,3,2,3,4]
    T = (np.arange(np.prod(dim))+1).reshape(dim, order=order_type).astype(np.float64) # using 64 bit float 
    T = T/lin.norm(T.flatten())
    print(T.shape)

    
    def check_integrity_QR(T):     
        MPS = get_MPS_QR(T)
        T_reconstructed = MPS_to_tensor(MPS)
        return check_equality_tensor(T, T_reconstructed)

    def check_integrity_SVD(T): 
        (MPS, entropys )= get_MPS_SVD(T, tol = 2**(-40))
        T_reconstructed = MPS_to_tensor(MPS)
        for entropy in entropys: 
            print(f"{entropy:.5f}")
        return check_equality_tensor(T, T_reconstructed)
    
    if check_integrity_QR(T):
        print(f'integrity of to MPS using QR Succeed!')
    else: 
        print(f'integrity of to MPS using QR Failed!')
    
    if check_integrity_SVD(T):
        print(f'integrity of to MPS using SVD Succeed!')
    else: 
        print(f'integrity of to MPS using SVD Failed!')
    
