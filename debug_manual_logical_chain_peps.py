import numpy as np

from src.stim_sampler import sample_surface_code_depolarizing_batch
from src.PEPS_Pauli_decoder import (
    pauli_coset_likelihoods_peps,
    most_likely_coset,
    choose_default_logical_cuts,
)


# =============================================================================
# Geometry helpers
# =============================================================================

def get_zero_noise_geometry(distance=5, memory_basis="x", rounds=3, target_t=1):
    """
    Use the existing Stim sampler only to get the dense PEPS geometry:
      active_X, active_Z, nrow, ncol
    """
    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=0.0,
        shots=1,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )
    active_X = np.asarray(batch.active_X[0], dtype=np.uint8)
    active_Z = np.asarray(batch.active_Z[0], dtype=np.uint8)
    nrow, ncol = active_X.shape
    cut_col, cut_row = choose_default_logical_cuts(active_X, active_Z)
    return active_X, active_Z, nrow, ncol, cut_col, cut_row


# =============================================================================
# Bond representation
# =============================================================================
#
# We represent a Pauli configuration by four binary arrays:
#
#   X_h, Z_h : horizontal-edge Pauli components, shape (nrow+1, ncol)
#   X_v, Z_v : vertical-edge   Pauli components, shape (nrow,   ncol+1)
#
# Decoder convention inferred from _build_face_tensor:
#   left  bond = vertical edge V[r, c]
#   right bond = vertical edge V[r, c+1]
#   up    bond = horizontal edge H[r, c]
#   down  bond = horizontal edge H[r+1, c]
#
# X-check syndrome = parity of Z components on incident edges
# Z-check syndrome = parity of X components on incident edges


def zero_bonds(nrow, ncol):
    X_h = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    Z_h = np.zeros((nrow + 1, ncol), dtype=np.uint8)
    X_v = np.zeros((nrow, ncol + 1), dtype=np.uint8)
    Z_v = np.zeros((nrow, ncol + 1), dtype=np.uint8)
    return X_h, Z_h, X_v, Z_v


def syndrome_from_bonds(X_h, Z_h, X_v, Z_v, active_X, active_Z):
    """
    Compute the PEPS syndromes implied by a manually specified bond Pauli pattern.
    """
    nrow, ncol = active_X.shape
    sX = np.zeros((nrow, ncol), dtype=np.uint8)
    sZ = np.zeros((nrow, ncol), dtype=np.uint8)

    for r in range(nrow):
        for c in range(ncol):
            # Incident edges:
            zl = Z_v[r, c]
            zr = Z_v[r, c + 1]
            zu = Z_h[r, c]
            zd = Z_h[r + 1, c]

            xl = X_v[r, c]
            xr = X_v[r, c + 1]
            xu = X_h[r, c]
            xd = X_h[r + 1, c]

            if active_X[r, c]:
                sX[r, c] = zl ^ zr ^ zu ^ zd
            if active_Z[r, c]:
                sZ[r, c] = xl ^ xr ^ xu ^ xd

    return sX, sZ


def cut_parities_from_bonds(X_h, Z_h, X_v, Z_v, cut_col, cut_row):
    """
    Directly compute the PEPS cut parities that the twists are supposed to probe.

    Vertical cut at cut_col:
        uses X parity on vertical edges V[:, cut_col]

    Horizontal cut at cut_row:
        uses Z parity on horizontal edges H[cut_row, :]

    These are the same bond families probed by the current twist implementation.
    """
    lx = int(np.bitwise_xor.reduce(X_v[:, cut_col])) if X_v.shape[0] > 0 else 0
    lz = int(np.bitwise_xor.reduce(Z_h[cut_row, :])) if Z_h.shape[1] > 0 else 0
    return lx, lz


def make_delta_weights(X_h, Z_h, X_v, Z_v, eps=1e-15):
    """
    Create custom weights that overwhelmingly favor exactly the specified bond Pauli
    configuration. This lets us ask:

        "If this exact chain dominates the posterior, which coset does PEPS assign it to?"

    Weight matrices are of shape (..., 2, 2), indexed by (x, z).
    """
    W_h = np.full(X_h.shape + (2, 2), eps, dtype=np.float64)
    W_v = np.full(X_v.shape + (2, 2), eps, dtype=np.float64)

    for r in range(X_h.shape[0]):
        for c in range(X_h.shape[1]):
            W_h[r, c, X_h[r, c], Z_h[r, c]] = 1.0

    for r in range(X_v.shape[0]):
        for c in range(X_v.shape[1]):
            W_v[r, c, X_v[r, c], Z_v[r, c]] = 1.0

    return W_h, W_v


# =============================================================================
# Candidate chains
# =============================================================================

def make_h_row_chain(nrow, ncol, pauli, row):
    """
    Put a chain across an entire horizontal-edge row H[row, :].

    pauli in {"X", "Z", "Y"}
    """
    X_h, Z_h, X_v, Z_v = zero_bonds(nrow, ncol)
    if pauli in ("X", "Y"):
        X_h[row, :] = 1
    if pauli in ("Z", "Y"):
        Z_h[row, :] = 1
    return X_h, Z_h, X_v, Z_v


def make_v_col_chain(nrow, ncol, pauli, col):
    """
    Put a chain across an entire vertical-edge column V[:, col].

    pauli in {"X", "Z", "Y"}
    """
    X_h, Z_h, X_v, Z_v = zero_bonds(nrow, ncol)
    if pauli in ("X", "Y"):
        X_v[:, col] = 1
    if pauli in ("Z", "Y"):
        Z_v[:, col] = 1
    return X_h, Z_h, X_v, Z_v


# =============================================================================
# PEPS evaluation
# =============================================================================

def evaluate_chain_with_peps(
    X_h, Z_h, X_v, Z_v,
    active_X, active_Z,
    cut_col, cut_row,
    eps=1e-15,
    Nkeep=64,
    Nsweep=1,
):
    sX, sZ = syndrome_from_bonds(X_h, Z_h, X_v, Z_v, active_X, active_Z)
    lx_cut, lz_cut = cut_parities_from_bonds(X_h, Z_h, X_v, Z_v, cut_col, cut_row)
    W_h, W_v = make_delta_weights(X_h, Z_h, X_v, Z_v, eps=eps)

    cosets = pauli_coset_likelihoods_peps(
        sX=sX,
        sZ=sZ,
        active_X=active_X,
        active_Z=active_Z,
        W_h=W_h,
        W_v=W_v,
        logical_x_cut_col=cut_col,
        logical_z_cut_row=cut_row,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
    )
    ml_coset, ml_val = most_likely_coset(cosets)

    return {
        "sX_weight": int(np.sum(sX)),
        "sZ_weight": int(np.sum(sZ)),
        "lx_cut": lx_cut,
        "lz_cut": lz_cut,
        "cosets": cosets,
        "ml_coset": ml_coset,
        "ml_val": ml_val,
    }


# =============================================================================
# Main diagnostic
# =============================================================================

def run_manual_chain_test(
    distance=5,
    memory_basis="x",
    rounds=3,
    target_t=1,
    eps=1e-15,
    Nkeep=64,
    Nsweep=1,
):
    active_X, active_Z, nrow, ncol, cut_col, cut_row = get_zero_noise_geometry(
        distance=distance,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )

    print("=" * 100)
    print("MANUAL LOGICAL-CHAIN PEPS DEBUG")
    print("=" * 100)
    print(f"distance={distance}, memory_basis={memory_basis}, rounds={rounds}, target_t={target_t}")
    print(f"grid shape           = ({nrow}, {ncol})")
    print(f"default cuts         = cut_col={cut_col}, cut_row={cut_row}")
    print(f"active_X count       = {int(np.sum(active_X))}")
    print(f"active_Z count       = {int(np.sum(active_Z))}")
    print()

    tests = []

    # Try all full horizontal-edge rows and full vertical-edge columns
    # for X and Z chains separately.
    for row in range(nrow + 1):
        tests.append(("H-row", "X", row, make_h_row_chain(nrow, ncol, "X", row)))
        tests.append(("H-row", "Z", row, make_h_row_chain(nrow, ncol, "Z", row)))

    for col in range(ncol + 1):
        tests.append(("V-col", "X", col, make_v_col_chain(nrow, ncol, "X", col)))
        tests.append(("V-col", "Z", col, make_v_col_chain(nrow, ncol, "Z", col)))

    rows = []
    for orient, pauli, idx, bonds in tests:
        X_h, Z_h, X_v, Z_v = bonds
        out = evaluate_chain_with_peps(
            X_h, Z_h, X_v, Z_v,
            active_X, active_Z,
            cut_col=cut_col,
            cut_row=cut_row,
            eps=eps,
            Nkeep=Nkeep,
            Nsweep=Nsweep,
        )
        rows.append((orient, pauli, idx, out))

    # Print the most relevant cases first:
    # zero-syndrome + nontrivial cut parity are the likely logical candidates.
    print("-" * 100)
    print("Zero-syndrome candidates with nontrivial cut parity")
    print("-" * 100)
    found_any = False
    for orient, pauli, idx, out in rows:
        zero_syn = (out["sX_weight"] == 0 and out["sZ_weight"] == 0)
        nontrivial = (out["lx_cut"] != 0 or out["lz_cut"] != 0)
        if zero_syn and nontrivial:
            found_any = True
            c = out["cosets"]
            print(
                f"{orient:6s}  {pauli}  idx={idx:2d}  "
                f"cut_parity=(lx={out['lx_cut']}, lz={out['lz_cut']})  "
                f"ML={out['ml_coset']}  "
                f"L00={c[(0,0)]:.6e}  "
                f"L10={c[(1,0)]:.6e}  "
                f"L01={c[(0,1)]:.6e}  "
                f"L11={c[(1,1)]:.6e}"
            )
    if not found_any:
        print("No zero-syndrome, nontrivial-cut candidates found with these simple full-row/full-column chains.")
    print()

    print("-" * 100)
    print("All tested chains")
    print("-" * 100)
    for orient, pauli, idx, out in rows:
        c = out["cosets"]
        print(
            f"{orient:6s}  {pauli}  idx={idx:2d}  "
            f"sX_w={out['sX_weight']:2d}  sZ_w={out['sZ_weight']:2d}  "
            f"cut_parity=(lx={out['lx_cut']}, lz={out['lz_cut']})  "
            f"ML={out['ml_coset']}  "
            f"L00={c[(0,0)]:.3e}  L10={c[(1,0)]:.3e}  "
            f"L01={c[(0,1)]:.3e}  L11={c[(1,1)]:.3e}"
        )

    print()
    print("=" * 100)
    print("How to read this")
    print("=" * 100)
    print(
        "1) A zero-syndrome chain with nontrivial cut parity is a candidate logical operator.\n"
        "2) If the delta-weighted PEPS assigns ML coset equal to the chain's cut parity, then PEPS logical labeling is self-consistent.\n"
        "3) If a chain has nontrivial cut parity but PEPS still returns ML=(0,0), that points to a logical-sector / cut / geometry bug.\n"
        "4) If the horizontal family works but the vertical family does not, that isolates the broken logical direction."
    )


if __name__ == "__main__":
    run_manual_chain_test(
        distance=5,
        memory_basis="x",
        rounds=3,
        target_t=1,
        eps=1e-15,
        Nkeep=64,
        Nsweep=1,
    )