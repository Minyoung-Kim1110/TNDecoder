import numpy as np
from .PEPS import contract_finPEPS


# Pauli utils 
PAULI_TO_XZ = {"I": (0, 0),
               "X": (1, 0),
               "Y": (1, 1),
               "Z": (0, 1),}

# Weights for each edges 
def pauli_probs_to_weight_matrix(pI, pX, pY, pZ):
    """
    Return W[x,z] where
        (x,z) = (0,0)->I, (1,0)->X, (1,1)->Y, (0,1)->Z

    So:
        W[0,0] = pI
        W[1,0] = pX
        W[1,1] = pY
        W[0,1] = pZ
    """
    W = np.zeros((2, 2), dtype=np.float64)
    W[0, 0] = pI
    W[1, 0] = pX
    W[1, 1] = pY
    W[0, 1] = pZ
    return W

def validate_local_weight_tensor(W_h, W_v, nrow, ncol):
    """
    W_h: shape (nrow+1, ncol, 2, 2)
         horizontal edge qubits
    W_v: shape (nrow, ncol+1, 2, 2)
         vertical edge qubits

    Each W[..., x, z] must sum to 1 over (x,z).
    """
    W_h = np.asarray(W_h, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)

    if W_h.shape != (nrow + 1, ncol, 2, 2):
        raise ValueError(f"W_h must have shape {(nrow+1, ncol, 2, 2)}, got {W_h.shape}")
    if W_v.shape != (nrow, ncol + 1, 2, 2):
        raise ValueError(f"W_v must have shape {(nrow, ncol+1, 2, 2)}, got {W_v.shape}")

    sums_h = W_h.sum(axis=(-2, -1))
    sums_v = W_v.sum(axis=(-2, -1))

    if not np.allclose(sums_h, 1.0, atol=1e-12):
        raise ValueError("Each horizontal qubit weight tensor must sum to 1.")
    if not np.allclose(sums_v, 1.0, atol=1e-12):
        raise ValueError("Each vertical qubit weight tensor must sum to 1.")

    return W_h, W_v

def biased_pauli_weights(nrow, ncol, pI, pX, pY, pZ):
    """
    Uniform but biased Pauli noise.
    """
    if not np.isclose(pI + pX + pY + pZ, 1.0, atol=1e-12):
        raise ValueError("Probabilities must sum to 1.")
    
    W = pauli_probs_to_weight_matrix(pI, pX, pY, pZ)
    W_h = np.broadcast_to(W, (nrow + 1, ncol, 2, 2)).copy()
    W_v = np.broadcast_to(W, (nrow, ncol + 1, 2, 2)).copy()
    return W_h, W_v

def depolarizing_weights(nrow, ncol, p):
    """
    Uniform depolarizing noise: p(I)=1-p, p(X)=p(Y)=p(Z)=p/3
    """
    return biased_pauli_weights(nrow, ncol, 1.0 - p, p / 3.0, p / 3.0, p / 3.0)

# Local tensor construction 
def _build_face_tensor(row, col, # location of plaquette
                      sX, sZ, # syndrome measurements 
                      W_h, W_v):
    r"""
    Build one local PEPS tensor for plaquette (row, col) with open boundaries.
    Leg order: T[left, up, down, right]
    Bond state flattening:
        idx = 2*x + z
        0 -> (0,0)
        1 -> (0,1)
        2 -> (1,0)
        3 -> (1,1)

    Boundary convention:
        - if the corresponding edge is shared with a neighboring plaquette,
          the PEPS leg has dimension 4 and contributes sqrt(W)
        - if the corresponding edge is on the outer boundary of the lattice,
          the PEPS leg has dimension 1 and contributes full W
    """
    
    sx = int(sX[row, col])
    sz = int(sZ[row, col])
    nrow, ncol = sX.shape
    
    # Open-boundary bond dimensions
    Dl = 1 if col == 0 else 4
    Du = 1 if row == 0 else 4
    Dd = 1 if row == nrow - 1 else 4
    Dr = 1 if col == ncol - 1 else 4

    T = np.zeros((Dl, Du, Dd, Dr), dtype=np.float64)

    # Local incident qubit-weight matrices W[x,z]
    Wu = W_h[row, col]       # top horizontal qubit
    Wd = W_h[row + 1, col]   # bottom horizontal qubit
    Wl = W_v[row, col]       # left vertical qubit
    Wr = W_v[row, col + 1]   # right vertical qubit

    def idx_to_xz(idx):
        return idx // 2, idx % 2

    for l in range(Dl):
        # boundary left edge: sum over local x,z internally
        left_states = [(0, 0), (0, 1), (1, 0), (1, 1)] if Dl==1 else [idx_to_xz(l)]
        
        for u in range(Du):
            up_states = [(0, 0), (0, 1), (1, 0), (1, 1)] if Du==1 else [idx_to_xz(u)]
            
            for d in range(Dd):
                down_states =  [(0, 0), (0, 1), (1, 0), (1, 1)] if Dd ==1 else  [idx_to_xz(d)]

                for r in range(Dr):
                    right_states =  [(0, 0), (0, 1), (1, 0), (1, 1)] if Dr == 1 else [idx_to_xz(r)]
                    
                    val = 0.0

                    for xl, zl in left_states:
                        for xu, zu in up_states:
                            for xd, zd in down_states:
                                for xr, zr in right_states:

                                    parity_z = zu ^ zd ^ zl ^ zr
                                    parity_x = xu ^ xd ^ xl ^ xr

                                    # check if this matches with syndrome measurement 
                                    if parity_z != sx or parity_x != sz:
                                        continue

                                    weight = 1.0
                                    weights = weights * np.sqrt(Wl[xl, zl]) if Dl !=1 else weights *Wl[xl, zl]
                                    weights = weights * np.sqrt(Wu[xl, zl]) if Du !=1 else weights *Wu[xl, zl]
                                    weights = weights * np.sqrt(Wd[xl, zl]) if Dd !=1 else weights *Wd[xl, zl]
                                    weights = weights * np.sqrt(Wr[xl, zl]) if Dr !=1 else weights *Wr[xl, zl]
                                    val += weight

                    T[l, u, d, r] = val

    return T

def build_pauli_syndrome_peps(sX, sZ, W_h, W_v):
    r"""
    Build a PEPS T[row][col] such that
        contract_finPEPS(T) = P(sX, sZ),
    where
        P(sX, sZ)= Σ_{all Pauli assignments}[Π_qubits p_q(P_q)] 1[X-syndrome matches sX] 1[Z-syndrome matches sZ]

    Input shapes:
        sX, sZ : (nrow, ncol)
        W_h    : (nrow+1, ncol, 2, 2)
        W_v    : (nrow, ncol+1, 2, 2)
    """
    sX = np.asarray(sX, dtype=np.uint8)
    sZ = np.asarray(sZ, dtype=np.uint8)

    if sX.shape != sZ.shape:
        raise ValueError("sX and sZ must have the same shape.")

    nrow, ncol = sX.shape
    # check if it is validate
    W_h, W_v = validate_local_weight_tensor(W_h, W_v, nrow, ncol)

    T = []
    for r in range(nrow):
        row_tensors = []
        for c in range(ncol):
            row_tensors.append(_build_face_tensor(r, c, sX, sZ, W_h, W_v))
        T.append(row_tensors)
    return T

# Logical coset twists 
def _twist_vertical_cut_x(T, cut_col):
    r"""
    Twist a vertical cut that measures logical X parity.

    Logical X parity is read from x-bits crossing a vertical cut: \ell_X = XOR of x on vertical-edge string.

    In the local face tensor, the vertical cut between face columns
    cut_col-1 and cut_col crosses the RIGHT leg of faces at column cut_col-1.

    To insert (-1)^{x} on crossed edges, multiply components with x=1 by -1.
    The bond state flattening is idx = 2*x + z, so x=1 corresponds to idx 2,3.
    """
    nrow = len(T)
    ncol = len(T[0])

    if not (1 <= cut_col <= ncol - 1):
        raise ValueError(f"cut_col must satisfy 1 <= cut_col <= {ncol-1}")

    T_tw = [[np.array(A, copy=True) for A in row] for row in T]
    c_left = cut_col - 1

    for r in range(nrow):
        A = T_tw[r][c_left]
        # right leg is axis 3
        A[..., 2] *= -1.0
        A[..., 3] *= -1.0
        T_tw[r][c_left] = A

    return T_tw


def _twist_horizontal_cut_z(T, cut_row):
    r"""
    Twist a horizontal cut that measures logical Z parity.

    Logical Z parity is read from z-bits crossing a horizontal cut: \ell_Z = XOR of z on horizontal-edge string.

    The horizontal cut between face rows cut_row-1 and cut_row crosses the DOWN leg of faces at row cut_row-1.

    To insert (-1)^z on crossed edges, multiply components with z=1 by -1.
    Since idx = 2*x + z, z=1 corresponds to idx 1,3.
    """
    nrow = len(T)
    ncol = len(T[0])

    if not (1 <= cut_row <= nrow - 1):
        raise ValueError(f"cut_row must satisfy 1 <= cut_row <= {nrow-1}")

    T_tw = [[np.array(A, copy=True) for A in row] for row in T]
    r_up = cut_row - 1

    for c in range(ncol):
        A = T_tw[r_up][c]
        # down leg is axis 2
        A[:, :, 1, :] *= -1.0
        A[:, :, 3, :] *= -1.0
        T_tw[r_up][c] = A

    return T_tw


def _contract_with_optional_twists(T, twist_x=False, cut_col=None, twist_z=False, cut_row=None,
                                   Nkeep=128, Nsweep=1):
    """
    Contract plain/twisted variants of the PEPS.
    """
    T_work = [[np.array(A, copy=True) for A in row] for row in T]

    if twist_x:
        T_work = _twist_vertical_cut_x(T_work, cut_col=cut_col)
    if twist_z:
        T_work = _twist_horizontal_cut_z(T_work, cut_row=cut_row)

    val = contract_finPEPS(T_work, Nkeep=Nkeep, Nsweep=Nsweep)
    return float(np.real_if_close(val))


def pauli_coset_likelihoods_peps(sX, sZ, W_h, W_v, logical_x_cut_col, logical_z_cut_row, Nkeep=128, Nsweep=1):
    r"""
    Compute the four logical-coset likelihoods by plain/twisted contractions.

    Let:
        plain      = Z(0,0)
        x_twisted  = Z_x
        z_twisted  = Z_z
        xz_twisted = Z_xz

    Then the projectors give

        L00 = 1/4 (plain + Z_x + Z_z + Z_xz)
        L10 = 1/4 (plain - Z_x + Z_z - Z_xz)
        L01 = 1/4 (plain + Z_x - Z_z - Z_xz)
        L11 = 1/4 (plain - Z_x - Z_z + Z_xz)

    where first bit = logical X parity lx, second bit = logical Z parity lz.
    """
    T = build_pauli_syndrome_peps(sX, sZ, W_h, W_v)

    plain = _contract_with_optional_twists(T, Nkeep=Nkeep, Nsweep=Nsweep)
    zx = _contract_with_optional_twists(T, twist_x=True, cut_col=logical_x_cut_col, Nkeep=Nkeep, Nsweep=Nsweep)
    zz = _contract_with_optional_twists(T, twist_z=True, cut_row=logical_z_cut_row,Nkeep=Nkeep, Nsweep=Nsweep)
    zxz = _contract_with_optional_twists(T, twist_x=True, cut_col=logical_x_cut_col,twist_z=True, cut_row=logical_z_cut_row,Nkeep=Nkeep, Nsweep=Nsweep)

    L00 = 0.25 * (plain + zx + zz + zxz)
    L10 = 0.25 * (plain - zx + zz - zxz)
    L01 = 0.25 * (plain + zx - zz - zxz)
    L11 = 0.25 * (plain - zx - zz + zxz)

    out = {
        (0, 0): float(np.real_if_close(L00)),
        (1, 0): float(np.real_if_close(L10)),
        (0, 1): float(np.real_if_close(L01)),
        (1, 1): float(np.real_if_close(L11)),
    }
    
    return out

def total_likelihood_from_cosets(cosets):
    return sum(cosets.values())

def most_likely_coset(cosets):
    return max(cosets.items(), key=lambda kv: kv[1])


