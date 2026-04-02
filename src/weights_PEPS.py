import numpy as np

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
