import numpy as np

from src.stim_sampler import sample_surface_code_depolarizing_batch
from src.mwpm_decoder import decode_stim_surface_batch_with_mwpm
from src.weights_PEPS import depolarizing_weights
from src.PEPS_Pauli_decoder import (
    pauli_coset_likelihoods_peps,
    most_likely_coset,
    choose_default_logical_cuts,
)


def _syndrome_weight(arr: np.ndarray) -> int:
    return int(np.sum(arr))


def _logical_idx_expected_from_basis(memory_basis: str) -> int:
    """
    Current repo convention:
      memory_basis='x' -> Stim observable is Z-logical -> PEPS tuple index 1
      memory_basis='z' -> Stim observable is X-logical -> PEPS tuple index 0
    """
    if memory_basis == "x":
        return 1
    if memory_basis == "z":
        return 0
    raise ValueError("memory_basis must be 'x' or 'z'.")


def _run_peps_for_one_shot(
    shot,
    W_h,
    W_v,
    cut_col: int,
    cut_row: int,
    Nkeep: int,
    Nsweep: int,
):
    cosets = pauli_coset_likelihoods_peps(
        sX=shot.sX,
        sZ=shot.sZ,
        active_X=shot.active_X,
        active_Z=shot.active_Z,
        W_h=W_h,
        W_v=W_v,
        logical_x_cut_col=cut_col,
        logical_z_cut_row=cut_row,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
    )
    ml_coset, ml_val = most_likely_coset(cosets)
    return cosets, ml_coset, ml_val


def _all_cut_pairs(nrow: int, ncol: int):
    # interior cuts only
    for cut_col in range(1, ncol):
        for cut_row in range(1, nrow):
            yield cut_col, cut_row


def diagnose_peps_vs_mwpm_same_batch(
    *,
    distance: int = 5,
    p: float = 0.01,
    shots: int = 100,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
    Nkeep: int = 64,
    Nsweep: int = 1,
    max_print_shots: int = 10,
):
    """
    Main diagnosis routine.

    What it checks:
    1. Same Stim batch goes to both MWPM and PEPS.
    2. Sweep all PEPS cut choices and compare:
       - PEPS bit 0 vs Stim observable
       - PEPS bit 1 vs Stim observable
       - PEPS expected bit vs Stim observable
       - PEPS expected bit vs MWPM predicted observable
    3. Per-shot instability under cut changes:
       if PEPS prediction changes a lot when cuts move, bug is likely in logical-sector/cut convention.
    """

    print("=" * 100)
    print("PEPS vs MWPM SAME-BATCH DIAGNOSTIC")
    print("=" * 100)
    print(
        f"distance={distance}, p={p}, shots={shots}, "
        f"memory_basis={memory_basis}, rounds={rounds}, target_t={target_t}, "
        f"Nkeep={Nkeep}, Nsweep={Nsweep}"
    )

    # ------------------------------------------------------------------
    # 1) Generate one exact shared batch
    # ------------------------------------------------------------------
    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )

    nrow, ncol = batch.sX.shape[1:]
    expected_idx = _logical_idx_expected_from_basis(memory_basis)
    default_cut_col, default_cut_row = choose_default_logical_cuts(
        batch.active_X[0], batch.active_Z[0]
    )

    print("\nBatch metadata")
    print(f"  syndrome shape            : {(nrow, ncol)}")
    print(f"  expected PEPS logical idx : {expected_idx}")
    print(f"  default cut (col,row)     : ({default_cut_col}, {default_cut_row})")
    print(f"  observable_flips shape    : {batch.observable_flips.shape}")
    print(f"  detector_bits shape       : {batch.detector_bits.shape}")

    # Raw physical logical flip fraction from Stim
    stim_flip_rate = float(np.mean(batch.observable_flips[:, 0]))
    print(f"  raw Stim observable flip rate : {stim_flip_rate:.6f}")

    # ------------------------------------------------------------------
    # 2) MWPM on the exact same batch
    # ------------------------------------------------------------------
    mwpm = decode_stim_surface_batch_with_mwpm(batch)
    mwpm_pred = mwpm.predicted_observable_flips[:, 0].astype(np.uint8)
    stim_obs = batch.observable_flips[:, 0].astype(np.uint8)

    mwpm_fail = np.bitwise_xor(mwpm_pred, stim_obs)
    print("\nMWPM reference on same batch")
    print(f"  MWPM predicted flip rate      : {np.mean(mwpm_pred):.6f}")
    print(f"  MWPM residual logical rate    : {np.mean(mwpm_fail):.6f}")

    # ------------------------------------------------------------------
    # 3) PEPS sweep over all cut choices
    # ------------------------------------------------------------------
    W_h, W_v = depolarizing_weights(nrow, ncol, p)

    cut_rows = []
    per_shot_predictions = {}  # (cut_col,cut_row) -> array(shape=(shots,2))

    for cut_col, cut_row in _all_cut_pairs(nrow, ncol):
        pred_bits = np.zeros((shots, 2), dtype=np.uint8)

        for i, shot in enumerate(batch.iter_shots()):
            _, ml_coset, _ = _run_peps_for_one_shot(
                shot=shot,
                W_h=W_h,
                W_v=W_v,
                cut_col=cut_col,
                cut_row=cut_row,
                Nkeep=Nkeep,
                Nsweep=Nsweep,
            )
            pred_bits[i, 0] = ml_coset[0]
            pred_bits[i, 1] = ml_coset[1]

        per_shot_predictions[(cut_col, cut_row)] = pred_bits

        acc_bit0_vs_stim = np.mean(pred_bits[:, 0] == stim_obs)
        acc_bit1_vs_stim = np.mean(pred_bits[:, 1] == stim_obs)
        acc_expected_vs_stim = np.mean(pred_bits[:, expected_idx] == stim_obs)

        acc_bit0_vs_mwpm = np.mean(pred_bits[:, 0] == mwpm_pred)
        acc_bit1_vs_mwpm = np.mean(pred_bits[:, 1] == mwpm_pred)
        acc_expected_vs_mwpm = np.mean(pred_bits[:, expected_idx] == mwpm_pred)

        peps_fail_expected = np.mean(np.bitwise_xor(pred_bits[:, expected_idx], stim_obs))

        cut_rows.append(
            {
                "cut_col": cut_col,
                "cut_row": cut_row,
                "acc_bit0_vs_stim": acc_bit0_vs_stim,
                "acc_bit1_vs_stim": acc_bit1_vs_stim,
                "acc_expected_vs_stim": acc_expected_vs_stim,
                "acc_bit0_vs_mwpm": acc_bit0_vs_mwpm,
                "acc_bit1_vs_mwpm": acc_bit1_vs_mwpm,
                "acc_expected_vs_mwpm": acc_expected_vs_mwpm,
                "peps_residual_rate_expected": peps_fail_expected,
                "is_default_cut": (cut_col == default_cut_col and cut_row == default_cut_row),
            }
        )

    # df_cuts = pd.DataFrame(cut_rows).sort_values(
    #     by=["acc_expected_vs_stim", "acc_expected_vs_mwpm"],
    #     ascending=False,
    # )
    cut_rows_sorted = sorted(
        cut_rows,
        key=lambda r: (
            -r["acc_expected_vs_stim"],
            -r["acc_expected_vs_mwpm"],
        ),
    )

    df_cuts = cut_rows_sorted   # keep same variable name if used later
    print("\nTop cut choices")
    nprint = min(10, len(df_cuts))

    for r in df_cuts[:nprint]:
        print(
            f"cut=({r['cut_col']},{r['cut_row']})  "
            f"acc_exp_stim={r['acc_expected_vs_stim']:.4f}  "
            f"acc_exp_mwpm={r['acc_expected_vs_mwpm']:.4f}  "
            f"residual={r['peps_residual_rate_expected']:.4f}  "
            f"default={r['is_default_cut']}"
        )
    # print(df_cuts.head(min(10, len(df_cuts))).to_string(index=False))

    # ------------------------------------------------------------------
    # 4) Diagnose whether bit-index mapping itself is wrong
    # ------------------------------------------------------------------
    # best_bit0 = df_cuts["acc_bit0_vs_stim"].max()
    # best_bit1 = df_cuts["acc_bit1_vs_stim"].max()
    best_bit0 = max(r["acc_bit0_vs_stim"] for r in df_cuts)
    best_bit1 = max(r["acc_bit1_vs_stim"] for r in df_cuts)

    print("\nLogical-index mapping diagnosis")
    print(f"  best possible bit-0 agreement with Stim : {best_bit0:.6f}")
    print(f"  best possible bit-1 agreement with Stim : {best_bit1:.6f}")

    if abs(best_bit0 - best_bit1) > 0.05:
        better = 0 if best_bit0 > best_bit1 else 1
        print(f"  ==> PEPS tuple index {better} is much more consistent with Stim observable.")
    else:
        print("  ==> Both PEPS tuple indices behave similarly; mapping ambiguity remains.")

    # ------------------------------------------------------------------
    # 5) Default cut quality relative to best cut
    # ------------------------------------------------------------------
    # default_row = df_cuts[df_cuts["is_default_cut"]].iloc[0]
    default_row = next(r for r in df_cuts if r["is_default_cut"])
    best_row = df_cuts[0]
    # best_row = df_cuts.iloc[0]

    print("\nDefault-cut diagnosis")
    print(
        f"  default cut residual rate (expected idx) : "
        f"{default_row['peps_residual_rate_expected']:.6f}"
    )
    print(
        f"  best cut residual rate (expected idx)    : "
        f"{best_row['peps_residual_rate_expected']:.6f}"
    )
    print(
        f"  best cut                                : "
        f"({int(best_row['cut_col'])}, {int(best_row['cut_row'])})"
    )

    if default_row["peps_residual_rate_expected"] - best_row["peps_residual_rate_expected"] > 0.05:
        print("  ==> Strong evidence that the default logical cut choice is wrong or suboptimal.")
    else:
        print("  ==> Default cut is not dramatically worse than the best cut.")

    # ------------------------------------------------------------------
    # 6) Per-shot stability under cut changes
    # ------------------------------------------------------------------
    # If the PEPS logical prediction changes a lot when moving the cuts,
    # the logical-sector / boundary convention is not robust.
    shot_rows = []
    cut_keys = list(per_shot_predictions.keys())

    for i in range(shots):
        pred_expected_across_cuts = np.array(
            [per_shot_predictions[k][i, expected_idx] for k in cut_keys],
            dtype=np.uint8,
        )
        pred_bit0_across_cuts = np.array(
            [per_shot_predictions[k][i, 0] for k in cut_keys],
            dtype=np.uint8,
        )
        pred_bit1_across_cuts = np.array(
            [per_shot_predictions[k][i, 1] for k in cut_keys],
            dtype=np.uint8,
        )

        unique_expected = np.unique(pred_expected_across_cuts)
        unique_bit0 = np.unique(pred_bit0_across_cuts)
        unique_bit1 = np.unique(pred_bit1_across_cuts)

        shot_rows.append(
            {
                "shot": i,
                "stim_obs": int(stim_obs[i]),
                "mwpm_pred": int(mwpm_pred[i]),
                "mwpm_residual": int(mwpm_fail[i]),
                "sX_weight": _syndrome_weight(batch.sX[i]),
                "sZ_weight": _syndrome_weight(batch.sZ[i]),
                "n_unique_expected_over_cuts": len(unique_expected),
                "n_unique_bit0_over_cuts": len(unique_bit0),
                "n_unique_bit1_over_cuts": len(unique_bit1),
            }
        )

    # df_shots = pd.DataFrame(shot_rows).sort_values(
    #     by=["n_unique_expected_over_cuts", "sX_weight", "sZ_weight"],
    #     ascending=[False, False, False],
    # )
    
    shot_rows_sorted = sorted(
        shot_rows,
        key=lambda r: (
            -r["n_unique_expected_over_cuts"],
            -r["sX_weight"],
            -r["sZ_weight"],
        ),
    )

    df_shots = shot_rows_sorted   # keep same variable name if used later

    print("\nMost cut-sensitive shots")
    nprint = min(max_print_shots, shots)

    for r in df_shots[:nprint]:
        print(
            f"shot={r['shot']:3d}  "
            f"stim={r['stim_obs']}  "
            f"mwpm={r['mwpm_pred']}  "
            f"mwpm_fail={r['mwpm_residual']}  "
            f"sX_w={r['sX_weight']:2d}  "
            f"sZ_w={r['sZ_weight']:2d}  "
            f"unique_expected={r['n_unique_expected_over_cuts']}"
        )
    # print(df_shots.head(min(max_print_shots, shots)).to_string(index=False))
    unstable_fraction = np.mean([
        r["n_unique_expected_over_cuts"] > 1
        for r in df_shots
    ])
    # unstable_fraction = np.mean(df_shots["n_unique_expected_over_cuts"].to_numpy() > 1)
    print(f"\nFraction of shots whose expected PEPS bit changes with cuts: {unstable_fraction:.6f}")

    if unstable_fraction > 0.1:
        print("  ==> Strong evidence of a logical-sector/boundary-condition problem.")
    else:
        print("  ==> PEPS prediction is mostly stable under cut changes.")

    # ------------------------------------------------------------------
    # 7) One-shot deep dump for suspicious examples
    # ------------------------------------------------------------------
    # suspicious = df_shots[df_shots["n_unique_expected_over_cuts"] > 1]["shot"].tolist()
    suspicious = [
    r["shot"]
        for r in df_shots
        if r["n_unique_expected_over_cuts"] > 1
    ]
    suspicious = suspicious[:max_print_shots]

    if suspicious:
        print("\nDetailed suspicious-shot dump")
        for i in suspicious:
            print("-" * 80)
            print(
                f"shot={i}, stim_obs={int(stim_obs[i])}, mwpm_pred={int(mwpm_pred[i])}, "
                f"mwpm_residual={int(mwpm_fail[i])}, "
                f"sX_weight={_syndrome_weight(batch.sX[i])}, sZ_weight={_syndrome_weight(batch.sZ[i])}"
            )

            shot = batch.get_shot(i)
            for cut_col, cut_row in cut_keys:
                cosets, ml_coset, ml_val = _run_peps_for_one_shot(
                    shot=shot,
                    W_h=W_h,
                    W_v=W_v,
                    cut_col=cut_col,
                    cut_row=cut_row,
                    Nkeep=Nkeep,
                    Nsweep=Nsweep,
                )
                print(
                    f"  cut=({cut_col},{cut_row})  ml={ml_coset}  "
                    f"L00={cosets[(0,0)]:.6e}  L10={cosets[(1,0)]:.6e}  "
                    f"L01={cosets[(0,1)]:.6e}  L11={cosets[(1,1)]:.6e}"
                )

    return {
        "batch": batch,
        "mwpm": mwpm,
        "cuts_table": df_cuts,
        "shots_table": df_shots,
        "default_cut": (default_cut_col, default_cut_row),
    }


if __name__ == "__main__":
    # Try both bases separately.
    # memory_basis='x': Stim observable should correspond to PEPS tuple index 1
    out_x = diagnose_peps_vs_mwpm_same_batch(
        distance=5,
        p=0.01,
        shots=100,
        memory_basis="x",
        rounds=3,
        target_t=1,
        Nkeep=64,
        Nsweep=1,
        max_print_shots=6,
    )

    print("\n" + "=" * 100 + "\n")

    # memory_basis='z': Stim observable should correspond to PEPS tuple index 0
    out_z = diagnose_peps_vs_mwpm_same_batch(
        distance=5,
        p=0.01,
        shots=100,
        memory_basis="z",
        rounds=3,
        target_t=1,
        Nkeep=64,
        Nsweep=1,
        max_print_shots=6,
    )