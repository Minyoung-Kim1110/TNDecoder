from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, Tuple, List, Optional

import numpy as np

# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------

from src.Surface_code_sampler.surface_code_capacity_sampler_full_logical_v9 import (
    sample_surface_code_capacity_batch_full_logical_v9,
)

from src.ML_decoder_PEPS.PEPS_Pauli_decoder import (
    pauli_coset_likelihoods_peps,
    choose_default_logical_cuts,
    decode_batch_with_peps,
    most_likely_coset,
)

from src.ML_decoder_PEPS.weights_PEPS import depolarizing_weights


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

COSSET_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]
COSSET_NAME = {
    (0, 0): "I",
    (1, 0): "Z",
    (0, 1): "X",
    (1, 1): "Y",
}


def bits_to_sector(bits: np.ndarray | Tuple[int, int]) -> str:
    z, x = int(bits[0]), int(bits[1])
    return COSSET_NAME[(z, x)]


def clone_batch_with_one_observable(batch, observable_flips_1d_or_2d):
    obs = np.asarray(observable_flips_1d_or_2d, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    return replace(batch, observable_flips=obs)


def format_cosets(cosets: Dict[Tuple[int, int], float]) -> str:
    parts = []
    for key in COSSET_ORDER:
        parts.append(f"{COSSET_NAME[key]}:{cosets[key]: .8e}")
    return "  ".join(parts)


def normalize_cosets(cosets: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    total = float(sum(cosets.values()))
    if total <= 0.0:
        return {k: np.nan for k in cosets}
    return {k: float(v / total) for k, v in cosets.items()}


def full_argmax_bits(cosets: Dict[Tuple[int, int], float]) -> Tuple[int, int]:
    ml = most_likely_coset(cosets)  # returns ((z,x), score)
    return int(ml[0][0]), int(ml[0][1])


def marginal_decision_for_x_memory(cosets: Dict[Tuple[int, int], float]) -> int:
    # X-memory cares about whether residual logical Z is present.
    # z=0: I or X   ; z=1: Z or Y
    p_z0 = float(cosets[(0, 0)] + cosets[(0, 1)])
    p_z1 = float(cosets[(1, 0)] + cosets[(1, 1)])
    return int(p_z1 > p_z0)


def marginal_decision_for_z_memory(cosets: Dict[Tuple[int, int], float]) -> int:
    # Z-memory cares about whether residual logical X is present.
    # x=0: I or Z   ; x=1: X or Y
    p_x0 = float(cosets[(0, 0)] + cosets[(1, 0)])
    p_x1 = float(cosets[(0, 1)] + cosets[(1, 1)])
    return int(p_x1 > p_x0)


def diagnostic_flags(cosets: Dict[Tuple[int, int], float], atol: float = 1e-14) -> Dict[str, bool]:
    vals = np.array([float(cosets[k]) for k in COSSET_ORDER], dtype=float)
    return {
        "all_finite": bool(np.all(np.isfinite(vals))),
        "all_nonnegative": bool(np.all(vals >= -atol)),
        "all_equal": bool(np.allclose(vals, vals[0], atol=atol, rtol=0.0)),
        "only_I_nonzero": bool(
            abs(vals[0]) > atol and np.all(np.abs(vals[1:]) <= atol)
        ),
        "sum_positive": bool(np.sum(vals) > atol),
    }


def choose_shot_indices(
    true_bits: np.ndarray,
    max_shots: int,
    require_nontrivial_first: bool = True,
) -> List[int]:
    n = int(true_bits.shape[0])
    all_idx = list(range(n))

    if not require_nontrivial_first:
        return all_idx[:max_shots]

    nontrivial = [k for k in all_idx if np.any(true_bits[k] != 0)]
    trivial = [k for k in all_idx if np.all(true_bits[k] == 0)]
    chosen = nontrivial[:max_shots]
    if len(chosen) < max_shots:
        chosen += trivial[: max_shots - len(chosen)]
    return chosen


# ---------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------

def audit_joint_cosets_surface_code(
    *,
    distance: int = 5,
    p: float = 0.01,
    shots: int = 128,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    seed: int = 1234,
    max_print_shots: int = 12,
    force_central_cuts: bool = False,
    cut_col: Optional[int] = None,
    cut_row: Optional[int] = None,
    require_nontrivial_first: bool = True,
) -> None:
    data = sample_surface_code_capacity_batch_full_logical_v9(
        distance=distance,
        p=p,
        shots=shots,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        seed=seed,
    )

    true_bits = np.asarray(data.logical_bits, dtype=np.uint8)
    joint_batch = data.batch

    # Basis-separated wrappers for comparison
    batch_x = clone_batch_with_one_observable(data.batch_x, data.batch_x.observable_flips)
    batch_z = clone_batch_with_one_observable(data.batch_z, data.batch_z.observable_flips)

    out_x_peps = decode_batch_with_peps(
        batch=batch_x,
        p=p,
        memory_basis="x",
        Nkeep=peps_nkeep,
        Nsweep=peps_nsweep,
        debug_failures=False,
    )
    out_z_peps = decode_batch_with_peps(
        batch=batch_z,
        p=p,
        memory_basis="z",
        Nkeep=peps_nkeep,
        Nsweep=peps_nsweep,
        debug_failures=False,
    )
    

    wrapper_pred_z_from_xmem = np.asarray(out_x_peps.predicted_observable_flips, dtype=np.uint8).reshape(-1)
    wrapper_pred_x_from_zmem = np.asarray(out_z_peps.predicted_observable_flips, dtype=np.uint8).reshape(-1)

    nrow, ncol = joint_batch.sX[0].shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p=p)

    print("=" * 100)
    print("AUDIT JOINT COSETS SURFACE CODE")
    print(
        {
            "distance": distance,
            "p": p,
            "shots": shots,
            "rounds": rounds,
            "noisy_round": noisy_round,
            "target_t": target_t,
            "peps_nkeep": peps_nkeep,
            "peps_nsweep": peps_nsweep,
            "seed": seed,
            "syndrome_shape": tuple(joint_batch.sX[0].shape),
            "force_central_cuts": force_central_cuts,
            "user_cut_col": cut_col,
            "user_cut_row": cut_row,
        }
    )
    print("=" * 100)

    # Sampler consistency checks
    stim_zlog = np.asarray(data.batch_x.observable_flips, dtype=np.uint8).reshape(-1)
    stim_xlog = np.asarray(data.batch_z.observable_flips, dtype=np.uint8).reshape(-1)

    z_mismatch = np.where(stim_zlog != true_bits[:, 0])[0]
    x_mismatch = np.where(stim_xlog != true_bits[:, 1])[0]

    print("\nSampler truth consistency:")
    print("  x-memory observable vs true_bits[:,0] mismatches =", len(z_mismatch))
    print("  z-memory observable vs true_bits[:,1] mismatches =", len(x_mismatch))

    shot_indices = choose_shot_indices(
        true_bits=true_bits,
        max_shots=max_print_shots,
        require_nontrivial_first=require_nontrivial_first,
    )

    full_success = 0
    xmem_success = 0
    zmem_success = 0
    wrapper_xmem_agree = 0
    wrapper_zmem_agree = 0

    for k in shot_indices:
        sX = np.asarray(joint_batch.sX[k], dtype=np.uint8)
        sZ = np.asarray(joint_batch.sZ[k], dtype=np.uint8)
        active_X = np.asarray(joint_batch.active_X[k], dtype=np.uint8)
        active_Z = np.asarray(joint_batch.active_Z[k], dtype=np.uint8)

        default_cut_col, default_cut_row = choose_default_logical_cuts(active_X, active_Z)

        if cut_col is not None:
            use_cut_col = int(cut_col)
        elif force_central_cuts:
            use_cut_col = max(1, min(ncol // 2, ncol - 1))
        else:
            use_cut_col = int(default_cut_col)

        if cut_row is not None:
            use_cut_row = int(cut_row)
        elif force_central_cuts:
            use_cut_row = max(1, min(nrow // 2, nrow - 1))
        else:
            use_cut_row = int(default_cut_row)

        cosets = pauli_coset_likelihoods_peps(
            sX=sX,
            sZ=sZ,
            active_X=active_X,
            active_Z=active_Z,
            W_h=W_h,
            W_v=W_v,
            logical_x_cut_col=use_cut_col,
            logical_z_cut_row=use_cut_row,
            Nkeep=peps_nkeep,
            Nsweep=peps_nsweep,
        )

        cosets_norm = normalize_cosets(cosets)
        

        flags = diagnostic_flags(cosets)

        pred_full = np.array(full_argmax_bits(cosets), dtype=np.uint8)
        pred_z_for_xmem = marginal_decision_for_x_memory(cosets)
        pred_x_for_zmem = marginal_decision_for_z_memory(cosets)
        pred_marginal_bits = np.array([pred_z_for_xmem, pred_x_for_zmem], dtype=np.uint8)

        true_full = true_bits[k].astype(np.uint8)
        true_z_for_xmem = int(true_full[0])
        true_x_for_zmem = int(true_full[1])

        wrapper_z = int(wrapper_pred_z_from_xmem[k])
        wrapper_x = int(wrapper_pred_x_from_zmem[k])

        full_ok = bool(np.array_equal(pred_full, true_full))
        xmem_ok = bool(pred_z_for_xmem == true_z_for_xmem)
        zmem_ok = bool(pred_x_for_zmem == true_x_for_zmem)

        wrapper_xmem_same = bool(wrapper_z == pred_z_for_xmem)
        wrapper_zmem_same = bool(wrapper_x == pred_x_for_zmem)

        full_success += int(full_ok)
        xmem_success += int(xmem_ok)
        zmem_success += int(zmem_ok)
        wrapper_xmem_agree += int(wrapper_xmem_same)
        wrapper_zmem_agree += int(wrapper_zmem_same)

        print("\n" + "-" * 100)
        print(f"shot {k}")
        print("-" * 100)
        print(
            f"true bits      = {true_full.tolist()}  "
            f"(sector {bits_to_sector(true_full)})"
        )
        print(
            f"stim obs       = x-memory->z_log={int(stim_zlog[k])}, "
            f"z-memory->x_log={int(stim_xlog[k])}"
        )
        print(
            f"wrapper bits   = [z_from_xmem, x_from_zmem] = "
            f"[{wrapper_z}, {wrapper_x}]"
        )

        print(
            f"sX weight={int(np.sum(sX))}, sZ weight={int(np.sum(sZ))}, "
            f"active_X={int(np.sum(active_X))}, active_Z={int(np.sum(active_Z))}"
        )
        print(
            f"default cuts   = (col={default_cut_col}, row={default_cut_row})"
        )
        print(
            f"used cuts      = (col={use_cut_col}, row={use_cut_row})"
        )

        print("\nraw cosets:")
        print(" ", format_cosets(cosets))
        print("normalized:")
        print(" ", format_cosets(cosets_norm))

        print("\ndiagnostics:")
        print(" ", flags)

        p_z0 = float(cosets[(0, 0)] + cosets[(0, 1)])
        p_z1 = float(cosets[(1, 0)] + cosets[(1, 1)])
        p_x0 = float(cosets[(0, 0)] + cosets[(1, 0)])
        p_x1 = float(cosets[(0, 1)] + cosets[(1, 1)])

        print("\nfull-coset decision:")
        print(
            f"  pred_full    = {pred_full.tolist()} "
            f"(sector {bits_to_sector(pred_full)})   ok={full_ok}"
        )

        print("\nmarginal decisions from joint cosets:")
        print(
            f"  X-memory: P(z=0)={p_z0:.8e}, P(z=1)={p_z1:.8e}  "
            f"pred z_log={pred_z_for_xmem}  true={true_z_for_xmem}  ok={xmem_ok}"
        )
        print(
            f"  Z-memory: P(x=0)={p_x0:.8e}, P(x=1)={p_x1:.8e}  "
            f"pred x_log={pred_x_for_zmem}  true={true_x_for_zmem}  ok={zmem_ok}"
        )

        print("\ncomparison to basis-separated wrapper:")
        print(
            f"  wrapper x-memory bit = {wrapper_z}   "
            f"joint-marginal bit = {pred_z_for_xmem}   "
            f"same={wrapper_xmem_same}"
        )
        print(
            f"  wrapper z-memory bit = {wrapper_x}   "
            f"joint-marginal bit = {pred_x_for_zmem}   "
            f"same={wrapper_zmem_same}"
        )

    n = len(shot_indices)
    if n > 0:
        print("\n" + "=" * 100)
        print("AUDIT SUMMARY OVER PRINTED SHOTS")
        print("=" * 100)
        print(f"printed shots                  = {n}")
        print(f"full-coset success             = {full_success}/{n} = {full_success / n:.6f}")
        print(f"X-memory marginal success      = {xmem_success}/{n} = {xmem_success / n:.6f}")
        print(f"Z-memory marginal success      = {zmem_success}/{n} = {zmem_success / n:.6f}")
        print(
            f"wrapper/joint agreement (Xmem) = {wrapper_xmem_agree}/{n} = "
            f"{wrapper_xmem_agree / n:.6f}"
        )
        print(
            f"wrapper/joint agreement (Zmem) = {wrapper_zmem_agree}/{n} = "
            f"{wrapper_zmem_agree / n:.6f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit joint PEPS logical cosets shot-by-shot for the unrotated surface code."
    )
    p.add_argument("--distance", type=int, default=5)
    p.add_argument("--p", type=float, default=0.01)
    p.add_argument("--shots", type=int, default=128)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--noisy-round", type=int, default=2)
    p.add_argument("--target-t", type=int, default=1)
    p.add_argument("--peps-nkeep", type=int, default=128)
    p.add_argument("--peps-nsweep", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-print-shots", type=int, default=12)
    p.add_argument("--force-central-cuts", action="store_true")
    p.add_argument("--cut-col", type=int, default=None)
    p.add_argument("--cut-row", type=int, default=None)
    p.add_argument("--include-trivial-first", action="store_true")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    audit_joint_cosets_surface_code(
        distance=args.distance,
        p=args.p,
        shots=args.shots,
        rounds=args.rounds,
        noisy_round=args.noisy_round,
        target_t=args.target_t,
        peps_nkeep=args.peps_nkeep,
        peps_nsweep=args.peps_nsweep,
        seed=args.seed,
        max_print_shots=args.max_print_shots,
        force_central_cuts=args.force_central_cuts,
        cut_col=args.cut_col,
        cut_row=args.cut_row,
        require_nontrivial_first=not args.include_trivial_first,
    )