import numpy as np

from src.stim_sampler import sample_surface_code_depolarizing_batch
from src.PEPS_Pauli_decoder import (
    pauli_coset_likelihoods_peps,
    most_likely_coset,
    choose_default_logical_cuts,
)
from src.weights_PEPS import depolarizing_weights


def logical_idx_expected_from_basis(memory_basis: str) -> int:
    if memory_basis == "x":
        return 1   # observable tracks logical Z
    if memory_basis == "z":
        return 0   # observable tracks logical X
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


def debug_peps_vs_stim_observable_only(
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
    print("PEPS vs STIM OBSERVABLE-ONLY DEBUG")
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

    stim_obs = np.asarray(batch.observable_flips[:, 0], dtype=np.uint8)
    logical_idx = logical_idx_expected_from_basis(memory_basis)

    active_X = np.asarray(batch.active_X[0], dtype=np.uint8)
    active_Z = np.asarray(batch.active_Z[0], dtype=np.uint8)

    print("\nBatch summary")
    print(f"  sX shape            : {batch.sX.shape}")
    print(f"  sZ shape            : {batch.sZ.shape}")
    print(f"  active_X count      : {int(np.sum(active_X))}")
    print(f"  active_Z count      : {int(np.sum(active_Z))}")
    print(f"  raw observable rate : {float(np.mean(stim_obs)):.6f}")

    pred_bits = np.zeros((shots, 2), dtype=np.uint8)
    pred_obs = np.zeros(shots, dtype=np.uint8)
    outputs = []

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
        outputs.append(out)

    obs_acc = float(np.mean(pred_obs == stim_obs))
    obs_fail = float(np.mean(pred_obs ^ stim_obs))

    print("\nPrediction summary")
    print(f"  predicted observable rate : {float(np.mean(pred_obs)):.6f}")
    print(f"  predicted bit-1 rate bit0 : {float(np.mean(pred_bits[:,0])):.6f}")
    print(f"  predicted bit-1 rate bit1 : {float(np.mean(pred_bits[:,1])):.6f}")
    print(f"  observable accuracy       : {obs_acc:.6f}")
    print(f"  observable fail rate      : {obs_fail:.6f}")

    print("\nObservable contingency table")
    n00 = int(np.sum((stim_obs == 0) & (pred_obs == 0)))
    n01 = int(np.sum((stim_obs == 0) & (pred_obs == 1)))
    n10 = int(np.sum((stim_obs == 1) & (pred_obs == 0)))
    n11 = int(np.sum((stim_obs == 1) & (pred_obs == 1)))
    print("true\\pred |   0     1")
    print("----------------------")
    print(f"0         | {n00:4d} {n01:5d}")
    print(f"1         | {n10:4d} {n11:5d}")

    print("\nDetailed disagreement examples")
    bad = [i for i in range(shots) if pred_obs[i] != stim_obs[i]]
    if not bad:
        print("  No disagreements found.")
    else:
        for i in bad[:max_bad_shots]:
            out = outputs[i]
            c = out["cosets"]
            print("-" * 100)
            print(f"shot {i}")
            print(f"  Stim observable   : {int(stim_obs[i])}")
            print(f"  PEPS pred obs     : {int(pred_obs[i])}")
            print(f"  PEPS pred coset   : {out['ml_coset']}")
            print(f"  sX weight         : {int(np.sum(batch.sX[i]))}")
            print(f"  sZ weight         : {int(np.sum(batch.sZ[i]))}")
            print(f"  cut_col, cut_row  : ({out['cut_col']}, {out['cut_row']})")
            print(
                "  cosets            : "
                f"L00={c[(0,0)]:.6e}, "
                f"L10={c[(1,0)]:.6e}, "
                f"L01={c[(0,1)]:.6e}, "
                f"L11={c[(1,1)]:.6e}"
            )

    return {
        "stim_obs": stim_obs,
        "pred_obs": pred_obs,
        "pred_bits": pred_bits,
        "outputs": outputs,
    }


if __name__ == "__main__":
    print("\n" + "#" * 100)
    print("# MEMORY X")
    print("#" * 100)
    out_x = debug_peps_vs_stim_observable_only(
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
    out_z = debug_peps_vs_stim_observable_only(
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