import numpy as np 

# Define system parameters
d = 3  # Local Hilbert space dimension (spin-1 system)
beta = 1/3

# Spin operators: S^+ (raising), S^z, S^- (lowering)
Spin_op = np.zeros((3,3,3))
Spin_op[:, :, 0] += np.array([[0, 1, 0], [0,0, 1], [0,0,0]])  # S^+
Spin_op[:, :, 1] += np.array([[1, 0, 0],[0,0,0], [0, 0,-1]])  # S^z
Spin_op[:, :, 2] += np.array([[0, 0,0],[1,0,0], [0,1, 0]])  # S^-
I = np.eye(d)

# Construct the bulk tensor Hloc (local MPO tensor)
Hloc = np.zeros((d,d, 14, 14), dtype=np.complex128)

Hloc[:,:,0, 0] = I  # Identity operator
Hloc[:,:,-1, -1] = I  # Identity operator for last index

for i in range(3):
    Hloc[:,:,i+1, 0] = Spin_op[:, :, i]  # Spin raising, z, lowering 
    Hloc[:,:,-1, i+1] = Spin_op[:, :, i].T  # Spin raising, z, lowering 
    
for i in range(3):
    for j in range(3):
        Hloc[:, :, 1+3+i*3+j, 0] = (Spin_op[:, :, i] @ Spin_op[:, :, j])
        Hloc[:, :, -1, 1+3+i*3+j] = beta *(Spin_op[:, :, i].T@ Spin_op[:, :, j].T)


def generate_AKLT_MPO_MPS(L, Nkeep):        
    # MPO for the full chain
    Hs = [Hloc.copy() for _ in range(L)]
    Hs[0] = Hs[0][:, :, -1:, :]  # Choose the last index of the left leg
    Hs[-1] = Hs[-1][:, :, :, :1]  # Choose the first index of the right leg
    # Generate a initial MPS
    M = [np.random.rand(1 if i == 0 else Nkeep, Nkeep if i < L - 1 else 1, d) for i in range(L)]
    return Hs, M
