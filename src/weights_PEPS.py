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


def z_only_depolarizing_weights(nrow, ncol, p):
    """
    Marginalized Z-only weights for memory_x code-capacity comparison.

    When only X-check syndrome is available (memory_x), we marginalise over
    the unobserved X component of each Pauli error:
        P_eff(z=0) = P(I) + P(X) = (1-p) + p/3 = 1 - 2p/3
        P_eff(z=1) = P(Z) + P(Y) =   p/3  + p/3 =   2p/3
    Setting P(x) = 0 in the PEPS weight tensor gives the correct
    marginalised probability for the Z-component only.
    """
    return biased_pauli_weights(nrow, ncol, 1.0 - 2 * p / 3, 0.0, 0.0, 2 * p / 3)


def local_depolarizing_weights(nrow, ncol, p_map, p_fallback=0.0):
    """
    Site-dependent depolarizing weights for PEPS local-noise decoding.

    p_map: dict {(stim_x, stim_y): p_i} — per-data-qubit error rates.
    PEPS grid convention: row = stim_y, col = stim_x.
    Check-site positions (not in p_map) use p_fallback (typically 0).
    """
    pI_fb = 1.0 - p_fallback
    pXYZ_fb = p_fallback / 3.0
    W_fb = pauli_probs_to_weight_matrix(pI_fb, pXYZ_fb, pXYZ_fb, pXYZ_fb)

    W_h = np.broadcast_to(W_fb, (nrow + 1, ncol, 2, 2)).copy()
    W_v = np.broadcast_to(W_fb, (nrow, ncol + 1, 2, 2)).copy()

    for (stim_x, stim_y), p_i in p_map.items():
        row, col = int(stim_y), int(stim_x)
        if 0 <= row < nrow and 0 <= col < ncol:
            p_i = float(np.clip(p_i, 0.0, 1.0))
            pXYZ = p_i / 3.0
            W_h[row, col] = pauli_probs_to_weight_matrix(1.0 - p_i, pXYZ, pXYZ, pXYZ)

    return W_h, W_v


# ---------------------------------------------------------------------------
# Spin qubit weights — pure Z dephasing
# ---------------------------------------------------------------------------

def spin_qubit_weights(nrow, ncol, p):
    """Uniform Z-only dephasing: pI=1-p, pX=0, pY=0, pZ=p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1.")
    return biased_pauli_weights(nrow, ncol, 1.0 - p, 0.0, 0.0, p)


def local_spin_qubit_weights(nrow, ncol, pz_map, p_fallback=0.0):
    """
    Site-dependent Z dephasing weights for PEPS spin qubit decoding.
    pz_map: dict {(stim_x, stim_y): pz_i}
    """
    W_fb = pauli_probs_to_weight_matrix(1.0 - p_fallback, 0.0, 0.0, p_fallback)
    W_h = np.broadcast_to(W_fb, (nrow + 1, ncol, 2, 2)).copy()
    W_v = np.broadcast_to(W_fb, (nrow, ncol + 1, 2, 2)).copy()
    for (stim_x, stim_y), pz_i in pz_map.items():
        row, col = int(stim_y), int(stim_x)
        if 0 <= row < nrow and 0 <= col < ncol:
            pz_i = float(np.clip(pz_i, 0.0, 1.0))
            W_h[row, col] = pauli_probs_to_weight_matrix(1.0 - pz_i, 0.0, 0.0, pz_i)
    return W_h, W_v


# ---------------------------------------------------------------------------
# EO qubit weights — biased PAULI_CHANNEL_1(px, 0, pz)
# ---------------------------------------------------------------------------

def eo_qubit_weights(nrow, ncol, px, pz):
    """Uniform biased Pauli noise: pI=1-px-pz, pX=px, pY=0, pZ=pz."""
    if px < 0.0 or pz < 0.0 or px + pz > 1.0:
        raise ValueError("px, pz must satisfy px >= 0, pz >= 0, px+pz <= 1.")
    return biased_pauli_weights(nrow, ncol, 1.0 - px - pz, px, 0.0, pz)


def local_eo_qubit_weights(nrow, ncol, px_map, pz_map, p_fallback_x=0.0, p_fallback_z=0.0):
    """
    Site-dependent EO qubit weights for PEPS decoding.
    px_map: dict {(stim_x, stim_y): px_i}  (effective X rate = (3/4)*p_i^n)
    pz_map: dict {(stim_x, stim_y): pz_i}  (effective Z rate = p_i^z + (1/4)*p_i^n)
    """
    px_fb = float(np.clip(p_fallback_x, 0.0, 1.0))
    pz_fb = float(np.clip(p_fallback_z, 0.0, 1.0))
    W_fb = pauli_probs_to_weight_matrix(1.0 - px_fb - pz_fb, px_fb, 0.0, pz_fb)
    W_h = np.broadcast_to(W_fb, (nrow + 1, ncol, 2, 2)).copy()
    W_v = np.broadcast_to(W_fb, (nrow, ncol + 1, 2, 2)).copy()
    all_xy = set(px_map) | set(pz_map)
    for xy in all_xy:
        stim_x, stim_y = xy
        row, col = int(stim_y), int(stim_x)
        if 0 <= row < nrow and 0 <= col < ncol:
            px_i = float(np.clip(px_map.get(xy, p_fallback_x), 0.0, 1.0))
            pz_i = float(np.clip(pz_map.get(xy, p_fallback_z), 0.0, 1.0))
            total = px_i + pz_i
            if total > 1.0:
                px_i /= total
                pz_i /= total
            W_h[row, col] = pauli_probs_to_weight_matrix(1.0 - px_i - pz_i, px_i, 0.0, pz_i)
    return W_h, W_v


def x_only_depolarizing_weights(nrow, ncol, p):
    """
    Marginalized X-only weights for memory_z code-capacity comparison.

    When only Z-check syndrome is available (memory_z), we marginalise over
    the unobserved Z component of each Pauli error:
        P_eff(x=0) = P(I) + P(Z) = (1-p) + p/3 = 1 - 2p/3
        P_eff(x=1) = P(X) + P(Y) =   p/3  + p/3 =   2p/3
    Setting P(z) = 0 in the PEPS weight tensor gives the correct
    marginalised probability for the X-component only.
    """
    return biased_pauli_weights(nrow, ncol, 1.0 - 2 * p / 3, 2 * p / 3, 0.0, 0.0)
