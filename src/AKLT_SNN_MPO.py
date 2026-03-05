# implement function that generates code for 
# AKLT model with second nearest hopping (Heisenberg spin-1 chain with quadratic term, next nearest hopping term )
# $H = \sum_{l=0}^{L-1} [(S_l \cdot S_{l+1}) + \beta(S_l \cdot S_{l+1})^2 ] + \sum_{l=0}^{L-2} \gamma (S_l \cdot S_{l+2})$
import numpy as np 

# Define system parameters
d = 3  # Local Hilbert space dimension (spin-1 system)

# Spin operators: S^+ (raising), S^z, S^- (lowering)
Spin_op = np.zeros((3,3,3))
Spin_op[:, :, 0] += np.array([[0, 1, 0], [0,0, 1], [0,0,0]])  # S^+
Spin_op[:, :, 1] += np.array([[1, 0, 0],[0,0,0], [0, 0,-1]])  # S^z
Spin_op[:, :, 2] += np.array([[0, 0,0],[1,0,0], [0,1, 0]])  # S^-
# Identity matrix
I = np.eye(d)


def generate_AKLT_SNN_MPO_MPS(beta, gamma, L, Nkeep):
    """Generate Hamiltonian in MPO, initial state in MPS with beta and gamma 
    Hamiltonian H = \sum_{l=0}^{L-1} [(S_l \cdot S_{l+1}) + \beta(S_l \cdot S_{l+1})^2 ] + \sum_{l=0}^{L-2} \gamma (S_l \cdot S_{l+2})

    Args:
        beta (float): quadratic coupling strength 
        gamma (float): next nearest neighbor coupling strength 
        L (int): number of sites 
        Nkeep (int): bond dimension to keep 

    Returns:
        Hs (List[np.array]): Hamiltonian in MPO 
        M (List[np.array]) : initial state in MPS 
    """
    # Construct the bulk tensor Hloc (local MPO tensor)
    Hloc = np.zeros((d,d, 2+d+d**2+2, 2+d+d**2+2), dtype=np.complex128)

    Hloc[:,:,0, 0] = I  # Identity operator
    Hloc[:,:,-1, -1] = I  # Identity operator for last index

    for i in range(3):
        Hloc[:,:,i+1, 0] = Spin_op[:, :, i]  # Spin raising, z, lowering 
        Hloc[:,:,-1, i+1] = Spin_op[:, :, i].T  # Spin raising, z, lowering 


    for i in range(3):
        for j in range(3):
            Hloc[:, :, 1+3+i*3+j, 0] = (Spin_op[:, :, i] @ Spin_op[:, :, j])
            Hloc[:, :, -1, 1+3+i*3+j] = beta *(Spin_op[:, :, i].T@ Spin_op[:, :, j].T)

    for i in range(3): 
        Hloc[:, :, -4+i, i+1] = I 
        Hloc[:, :, -1, -4+i] = gamma * Spin_op[:, :, i].T
            
    # MPO for the full chain
    Hs = [Hloc.copy() for _ in range(L)]
    Hs[0] = Hs[0][:, :, -1:, :]  # Choose the last index of the left leg
    Hs[-1] = Hs[-1][:, :, :, :1]  # Choose the first index of the right leg


    # Generate a initial MPS
    M = [np.random.rand(1 if i == 0 else Nkeep, Nkeep if i < L - 1 else 1, d) for i in range(L)]
    
    return Hs, M