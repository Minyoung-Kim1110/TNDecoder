import numpy as np

from src.stim_sampler import sample_surface_code_depolarizing_batch
from src.PEPS_Pauli_decoder import (
    pauli_coset_likelihoods_peps,
    most_likely_coset,
    choose_default_logical_cuts,
)
from src.weights_PEPS import depolarizing_weights


# =============================================================================
# Helpers
# =============================================================================

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


# def transform_grid(arr: np.ndarray, mode: str) -> np.ndarray:
#     """
#     Apply a spatial transform to a 2D grid.
#     """
#     if mode == "identity":
#         return np.array(arr, copy=True)
#     elif mode == "transpose":
#         return np.array(arr.T, copy=True)
#     elif mode == "flipud":
#         return np.array(np.flipud(arr), copy=True)
#     elif mode == "fliplr":
#         return np.array(np.fliplr(arr), copy=True)
#     elif mode == "transpose_flipud":
#         return np.array(np.flipud(arr.T), copy=True)
#     elif mode == "transpose_fliplr":
#         return np.array(np.fliplr(arr.T), copy=True)
#     else:
#         raise ValueError(f"Unknown transform mode: {mode}")


# def apply_variant(sX, sZ, active_X, active_Z, variant_name: str):
#     """
#     Variant names:
#       identity
#       swap_types
#       transpose
#       transpose_swap_types
#       flipud
#       fliplr
#       transpose_flipud
#       transpose_fliplr
#       flipud_swap_types
#       fliplr_swap_types
#     """
#     swap = variant_name.endswith("_swap_types")
#     base = variant_name.replace("_swap_types", "")

#     sX_t = transform_grid(sX, base)
#     sZ_t = transform_grid(sZ, base)
#     aX_t = transform_grid(active_X, base)
#     aZ_t = transform_grid(active_Z, base)

#     if swap:
#         sX_t, sZ_t = sZ_t, sX_t
#         aX_t, aZ_t = aZ_t, aX_t

#     return sX_t, sZ_t, aX_t, aZ_t


def decode_one_shot_with_variant(
    sX,
    sZ,
    active_X,
    active_Z,
    p,
    variant_name,
    Nkeep=64,
    Nsweep=1,
):
    sX_t, sZ_t, aX_t, aZ_t = apply_variant(sX, sZ, active_X, active_Z, variant_name)

    nrow, ncol = sX_t.shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p)

    cut_col, cut_row = choose_default_logical_cuts(aX_t, aZ_t)

    cosets = pauli_coset_likelihoods_peps(
        sX=sX_t,
        sZ=sZ_t,
        active_X=aX_t,
        active_Z=aZ_t,
        W_h=W_h,
        W_v=W_v,
        logical_x_cut_col=cut_col,
        logical_z_cut_row=cut_row,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
    )
    ml_coset, ml_val = most_likely_coset(cosets)

    return {
        "ml_coset": ml_coset,
        "ml_val": ml_val,
        "cosets": cosets,
        "cut_col": cut_col,
        "cut_row": cut_row,
        "sX_weight": int(np.sum(sX_t)),
        "sZ_weight": int(np.sum(sZ_t)),
    }


# =============================================================================
# Main experiment
# =============================================================================

def investigate_mapping_bug(
    distance=5,
    p=0.01,
    shots=50,
    memory_basis="x",
    rounds=3,
    target_t=1,
    Nkeep=64,
    Nsweep=1,
    print_bad_shots=8,
):
    print("=" * 100)
    print("STIM -> PEPS MAPPING INVESTIGATION")
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

    logical_idx = logical_idx_expected_from_basis(memory_basis)
    stim_obs = np.asarray(batch.observable_flips[:, 0], dtype=np.uint8)

    if hasattr(batch, "logical_bits"):
        stim_bits = np.asarray(batch.logical_bits, dtype=np.uint8)
    else:
        stim_bits = None

    active_X = np.asarray(batch.active_X[0], dtype=np.uint8)
    active_Z = np.asarray(batch.active_Z[0], dtype=np.uint8)

    print("\nBatch summary")
    print(f"  syndrome shape            : {batch.sX.shape[1:]}")
    print(f"  active_X count            : {int(np.sum(active_X))}")
    print(f"  active_Z count            : {int(np.sum(active_Z))}")
    print(f"  raw Stim observable rate  : {float(np.mean(stim_obs)):.6f}")
    if stim_bits is not None:
        print(f"  logical_bits shape        : {stim_bits.shape}")
        print(f"  logical bit-1 fractions   : {np.mean(stim_bits, axis=0)}")

    variants = [
        "identity",
        "swap_types",
        "transpose",
        "transpose_swap_types",
        "flipud",
        "flipud_swap_types",
        "fliplr",
        "fliplr_swap_types",
        "transpose_flipud",
        "transpose_fliplr",
    ]

    results = []
    per_variant_predictions = {}

    for variant in variants:
        pred_obs = np.zeros(shots, dtype=np.uint8)
        pred_bits = np.zeros((shots, 2), dtype=np.uint8)

        for i in range(shots):
            out = decode_one_shot_with_variant(
                sX=np.asarray(batch.sX[i], dtype=np.uint8),
                sZ=np.asarray(batch.sZ[i], dtype=np.uint8),
                active_X=active_X,
                active_Z=active_Z,
                p=p,
                variant_name=variant,
                Nkeep=Nkeep,
                Nsweep=Nsweep,
            )
            pred_bits[i, 0] = out["ml_coset"][0]
            pred_bits[i, 1] = out["ml_coset"][1]
            pred_obs[i] = out["ml_coset"][logical_idx]

        obs_acc = float(np.mean(pred_obs == stim_obs))
        obs_fail = float(np.mean(pred_obs ^ stim_obs))

        if stim_bits is not None:
            full_coset_acc = float(np.mean(np.all(pred_bits == stim_bits, axis=1)))
            bit0_acc = float(np.mean(pred_bits[:, 0] == stim_bits[:, 0]))
            bit1_acc = float(np.mean(pred_bits[:, 1] == stim_bits[:, 1]))
        else:
            full_coset_acc = float("nan")
            bit0_acc = float("nan")
            bit1_acc = float("nan")

        results.append({
            "variant": variant,
            "obs_acc": obs_acc,
            "obs_fail": obs_fail,
            "full_coset_acc": full_coset_acc,
            "bit0_acc": bit0_acc,
            "bit1_acc": bit1_acc,
            "pred_obs_rate": float(np.mean(pred_obs)),
            "pred_bit0_rate": float(np.mean(pred_bits[:, 0])),
            "pred_bit1_rate": float(np.mean(pred_bits[:, 1])),
        })

        per_variant_predictions[variant] = {
            "pred_obs": pred_obs,
            "pred_bits": pred_bits,
        }

    # Sort by observable accuracy, then full coset accuracy
    results = sorted(
        results,
        key=lambda r: (-r["obs_acc"], -r["full_coset_acc"]),
    )

    print("\nVariant summary")
    for r in results:
        print(
            f"{r['variant']:22s}  "
            f"obs_acc={r['obs_acc']:.4f}  "
            f"obs_fail={r['obs_fail']:.4f}  "
            f"full_coset_acc={r['full_coset_acc']:.4f}  "
            f"bit0_acc={r['bit0_acc']:.4f}  "
            f"bit1_acc={r['bit1_acc']:.4f}  "
            f"pred_obs_rate={r['pred_obs_rate']:.4f}"
        )

    best = results[0]
    print("\nBest variant")
    print(best)

    # -------------------------------------------------------------------------
    # Detailed disagreement dump for identity vs best
    # -------------------------------------------------------------------------
    identity_pred = per_variant_predictions["identity"]["pred_bits"]
    best_pred = per_variant_predictions[best["variant"]]["pred_bits"]
    identity_obs = per_variant_predictions["identity"]["pred_obs"]
    best_obs = per_variant_predictions[best["variant"]]["pred_obs"]

    print("\nDisagreement examples: identity vs best variant")
    nshown = 0
    for i in range(shots):
        changed = not np.array_equal(identity_pred[i], best_pred[i])
        if not changed:
            continue

        print("-" * 100)
        print(f"shot {i}")
        if stim_bits is not None:
            print(f"  Stim logical bits      : {tuple(int(x) for x in stim_bits[i])}")
        print(f"  Stim observable        : {int(stim_obs[i])}")
        print(f"  identity pred bits     : {tuple(int(x) for x in identity_pred[i])}")
        print(f"  identity pred obs      : {int(identity_obs[i])}")
        print(f"  best[{best['variant']}] bits : {tuple(int(x) for x in best_pred[i])}")
        print(f"  best[{best['variant']}] obs  : {int(best_obs[i])}")
        print(f"  raw sX weight          : {int(np.sum(batch.sX[i]))}")
        print(f"  raw sZ weight          : {int(np.sum(batch.sZ[i]))}")
        nshown += 1
        if nshown >= print_bad_shots:
            break

    if nshown == 0:
        print("  No shot changed between identity and best variant.")

    return {
        "results": results,
        "best_variant": best,
        "predictions": per_variant_predictions,
        "stim_obs": stim_obs,
        "stim_bits": stim_bits,
    }

def transform_grid(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply a spatial transform to a 2D grid.
    """
    if mode in ("identity", ""):
        return np.array(arr, copy=True)
    elif mode == "transpose":
        return np.array(arr.T, copy=True)
    elif mode == "flipud":
        return np.array(np.flipud(arr), copy=True)
    elif mode == "fliplr":
        return np.array(np.fliplr(arr), copy=True)
    elif mode == "transpose_flipud":
        return np.array(np.flipud(arr.T), copy=True)
    elif mode == "transpose_fliplr":
        return np.array(np.fliplr(arr.T), copy=True)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")


def apply_variant(sX, sZ, active_X, active_Z, variant_name: str):
    """
    Supported variants:
      identity
      swap_types
      transpose
      transpose_swap_types
      flipud
      fliplr
      transpose_flipud
      transpose_fliplr
      flipud_swap_types
      fliplr_swap_types
    """
    if variant_name == "swap_types":
        base = "identity"
        swap = True
    elif variant_name.endswith("_swap_types"):
        base = variant_name[: -len("_swap_types")]
        swap = True
    else:
        base = variant_name
        swap = False

    sX_t = transform_grid(sX, base)
    sZ_t = transform_grid(sZ, base)
    aX_t = transform_grid(active_X, base)
    aZ_t = transform_grid(active_Z, base)

    if swap:
        sX_t, sZ_t = sZ_t, sX_t
        aX_t, aZ_t = aZ_t, aX_t

    return sX_t, sZ_t, aX_t, aZ_t

if __name__ == "__main__":
    print("\n" + "#" * 100)
    print("# MEMORY X")
    print("#" * 100)
    out_x = investigate_mapping_bug(
        distance=5,
        p=0.01,
        shots=50,
        memory_basis="x",
        rounds=3,
        target_t=1,
        Nkeep=64,
        Nsweep=1,
        print_bad_shots=8,
    )

    print("\n" + "#" * 100)
    print("# MEMORY Z")
    print("#" * 100)
    out_z = investigate_mapping_bug(
        distance=5,
        p=0.01,
        shots=50,
        memory_basis="z",
        rounds=3,
        target_t=1,
        Nkeep=64,
        Nsweep=1,
        print_bad_shots=8,
    )