import numpy as np

from src.stim_sampler import sample_surface_code_depolarizing_batch
from src.PEPS_Pauli_decoder import (
    pauli_coset_likelihoods_peps,
    most_likely_coset,
    choose_default_logical_cuts,
)
from src.weights_PEPS import depolarizing_weights


def logical_idx_expected_from_basis(memory_basis: str) -> int:
    """
    Repo convention:
      memory_basis='x' -> observable tracks logical Z => index 1
      memory_basis='z' -> observable tracks logical X => index 0
    """
    if memory_basis == "x":
        return 1
    if memory_basis == "z":
        return 0
    raise ValueError("memory_basis must be 'x' or 'z'")


def decode_one_shot_peps(
    sX,
    sZ,
    active_X,
    active_Z,
    p,
    Nkeep=64,
    Nsweep=1,
):
    nrow, ncol = sX.shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p)

    cut_col, cut_row = choose_default_logical_cuts(active_X, active_Z)

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
        "ml_coset": tuple(int(x) for x in ml_coset),
        "ml_val": float(ml_val),
        "cosets": cosets,
        "cut_col": int(cut_col),
        "cut_row": int(cut_row),
    }


def summarize_counts(bit_pairs):
    """
    bit_pairs: array shape (shots, 2) of 0/1 bits
    returns dict with counts for (0,0),(1,0),(0,1),(1,1)
    """
    counts = {(0, 0): 0, (1, 0): 0, (0, 1): 0, (1, 1): 0}
    for row in bit_pairs:
        key = (int(row[0]), int(row[1]))
        counts[key] += 1
    return counts


def debug_peps_vs_stim_logical_bits(
    distance=5,
    p=0.01,
    shots=100,
    memory_basis="x",
    rounds=3,
    target_t=1,
    Nkeep=64,
    Nsweep=1,
    max_bad_shots=12,
):
    print("=" * 100)
    print("PEPS vs STIM LOGICAL-BITS DEBUG")
    print("=" * 100)
    print(
        f"distance={distance}, p={p}, shots={shots}, memory_basis={memory_basis}, "
        f"rounds={rounds}, target_t={target_t}, Nkeep={Nkeep}, Nsweep={Nsweep}"
    )

    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )

    if not hasattr(batch, "logical_bits"):
        raise RuntimeError(
            "This batch object does not contain logical_bits. "
            "Use a sampler version that stores full logical sector labels."
        )

    stim_bits = np.asarray(batch.logical_bits, dtype=np.uint8)
    stim_obs = np.asarray(batch.observable_flips[:, 0], dtype=np.uint8)

    active_X = np.asarray(batch.active_X[0], dtype=np.uint8)
    active_Z = np.asarray(batch.active_Z[0], dtype=np.uint8)

    logical_idx = logical_idx_expected_from_basis(memory_basis)

    print("\nBatch summary")
    print(f"  sX shape                : {batch.sX.shape}")
    print(f"  sZ shape                : {batch.sZ.shape}")
    print(f"  active_X count          : {int(np.sum(active_X))}")
    print(f"  active_Z count          : {int(np.sum(active_Z))}")
    print(f"  raw observable rate     : {float(np.mean(stim_obs)):.6f}")
    print(f"  Stim logical bit rates  : {np.mean(stim_bits, axis=0)}")
    print(f"  Stim logical counts     : {summarize_counts(stim_bits)}")

    pred_bits = np.zeros((shots, 2), dtype=np.uint8)
    pred_obs = np.zeros(shots, dtype=np.uint8)
    all_outputs = []

    for i in range(shots):
        out = decode_one_shot_peps(
            sX=np.asarray(batch.sX[i], dtype=np.uint8),
            sZ=np.asarray(batch.sZ[i], dtype=np.uint8),
            active_X=active_X,
            active_Z=active_Z,
            p=p,
            Nkeep=Nkeep,
            Nsweep=Nsweep,
        )
        pred_bits[i, 0] = out["ml_coset"][0]
        pred_bits[i, 1] = out["ml_coset"][1]
        pred_obs[i] = out["ml_coset"][logical_idx]
        all_outputs.append(out)

    # ------------------------------------------------------------------
    # summary metrics
    # ------------------------------------------------------------------
    bit0_acc = float(np.mean(pred_bits[:, 0] == stim_bits[:, 0]))
    bit1_acc = float(np.mean(pred_bits[:, 1] == stim_bits[:, 1]))
    full_acc = float(np.mean(np.all(pred_bits == stim_bits, axis=1)))
    obs_acc = float(np.mean(pred_obs == stim_obs))
    obs_fail = float(np.mean(pred_obs ^ stim_obs))

    print("\nPEPS prediction summary")
    print(f"  predicted logical bit rates : {np.mean(pred_bits, axis=0)}")
    print(f"  predicted logical counts    : {summarize_counts(pred_bits)}")
    print(f"  bit0 accuracy               : {bit0_acc:.6f}")
    print(f"  bit1 accuracy               : {bit1_acc:.6f}")
    print(f"  full coset accuracy         : {full_acc:.6f}")
    print(f"  observable accuracy         : {obs_acc:.6f}")
    print(f"  observable fail rate        : {obs_fail:.6f}")

    # ------------------------------------------------------------------
    # contingency table
    # ------------------------------------------------------------------
    true_keys = [(0, 0), (1, 0), (0, 1), (1, 1)]
    pred_keys = [(0, 0), (1, 0), (0, 1), (1, 1)]
    table = {(t, p): 0 for t in true_keys for p in pred_keys}

    for t, p_ in zip(stim_bits, pred_bits):
        tkey = (int(t[0]), int(t[1]))
        pkey = (int(p_[0]), int(p_[1]))
        table[(tkey, pkey)] += 1

    print("\nContingency table: Stim logical_bits -> PEPS predicted coset")
    header = "true\\pred | (0,0) (1,0) (0,1) (1,1)"
    print(header)
    print("-" * len(header))
    for tkey in true_keys:
        row = [table[(tkey, pkey)] for pkey in pred_keys]
        print(f"{tkey!s:9s} | {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d}")

    # ------------------------------------------------------------------
    # disagreement dump
    # ------------------------------------------------------------------
    print("\nDetailed disagreement examples")
    bad_indices = [i for i in range(shots) if not np.array_equal(pred_bits[i], stim_bits[i])]

    if not bad_indices:
        print("  No disagreements found.")
    else:
        for k, i in enumerate(bad_indices[:max_bad_shots]):
            out = all_outputs[i]
            cosets = out["cosets"]
            print("-" * 100)
            print(f"shot {i}")
            print(f"  Stim logical_bits   : {tuple(int(x) for x in stim_bits[i])}")
            print(f"  PEPS predicted      : {out['ml_coset']}")
            print(f"  Stim observable     : {int(stim_obs[i])}")
            print(f"  PEPS observable     : {int(pred_obs[i])}")
            print(f"  sX weight           : {int(np.sum(batch.sX[i]))}")
            print(f"  sZ weight           : {int(np.sum(batch.sZ[i]))}")
            print(f"  cut_col, cut_row    : ({out['cut_col']}, {out['cut_row']})")
            print(
                "  cosets              : "
                f"L00={cosets[(0,0)]:.6e}, "
                f"L10={cosets[(1,0)]:.6e}, "
                f"L01={cosets[(0,1)]:.6e}, "
                f"L11={cosets[(1,1)]:.6e}"
            )

    return {
        "stim_bits": stim_bits,
        "pred_bits": pred_bits,
        "stim_obs": stim_obs,
        "pred_obs": pred_obs,
        "outputs": all_outputs,
    }


if __name__ == "__main__":
    print("\n" + "#" * 100)
    print("# MEMORY X")
    print("#" * 100)
    out_x = debug_peps_vs_stim_logical_bits(
        distance=5,
        p=0.01,
        shots=100,
        memory_basis="x",
        rounds=3,
        target_t=1,
        Nkeep=64,
        Nsweep=1,
        max_bad_shots=12,
    )

    print("\n" + "#" * 100)
    print("# MEMORY Z")
    print("#" * 100)
    out_z = debug_peps_vs_stim_logical_bits(
        distance=5,
        p=0.01,
        shots=100,
        memory_basis="z",
        rounds=3,
        target_t=1,
        Nkeep=64,
        Nsweep=1,
        max_bad_shots=12,
    )