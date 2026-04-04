from __future__ import annotations

from dataclasses import replace
from typing import Dict, Any, Tuple
import numpy as np

# v5 comparison driver
from compare_peps_mwpm_surface_code_full_logical_v6 import (
    compare_peps_mwpm_surface_code_full_logical,
)

# v6 sampler
from src.Surface_code_sampler.surface_code_capacity_sampler_full_logical_v9 import (
    sample_surface_code_capacity_batch_full_logical,
)

# low-level decoders
from src.ML_decoder_PEPS.PEPS_Pauli_decoder import decode_batch_with_peps
from src.MWPM_decoder_pymatching.mwpm_decoder_2d import decode_2d_surface_batch_with_mwpm
import inspect
print("Sampler source file:", inspect.getsourcefile(sample_surface_code_capacity_batch_full_logical))
print("Sampler function:", sample_surface_code_capacity_batch_full_logical.__name__)

# ============================================================
# Small utilities
# ============================================================

def _sector_hist(bits: np.ndarray) -> Dict[str, int]:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 2 or bits.shape[1] != 2:
        raise ValueError(f"Expected (shots,2) logical bits, got {bits.shape}")
    z = bits[:, 0]
    x = bits[:, 1]
    return {
        "I": int(np.sum((z == 0) & (x == 0))),
        "Z": int(np.sum((z == 1) & (x == 0))),
        "X": int(np.sum((z == 0) & (x == 1))),
        "Y": int(np.sum((z == 1) & (x == 1))),
    }


def _bit1_fraction(bits: np.ndarray):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim == 1:
        return float(np.mean(bits))
    return [float(np.mean(bits[:, j])) for j in range(bits.shape[1])]


def _weight_stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.uint8)
    w = np.sum(arr, axis=1)
    return {
        "nonzero_fraction": float(np.mean(w > 0)),
        "mean_weight": float(np.mean(w)),
        "max_weight": int(np.max(w)) if len(w) else 0,
    }


def _clone_batch_with_one_observable(batch, observable_flips):
    """
    Force observable_flips to shape (shots,1), which is what the PEPS/MWPM
    basis-specific decoders usually expect.
    """
    obs = np.asarray(observable_flips, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    return replace(batch, observable_flips=obs)


def _summarize_residual(residual_bits: np.ndarray) -> Dict[str, Any]:
    """
    residual_bits[:,0] = residual logical Z bit
    residual_bits[:,1] = residual logical X bit

    Convention:
      I <-> (0,0)
      Z <-> (1,0)
      X <-> (0,1)
      Y <-> (1,1)
    """
    residual_bits = np.asarray(residual_bits, dtype=np.uint8)
    if residual_bits.ndim != 2 or residual_bits.shape[1] != 2:
        raise ValueError(f"Expected residual_bits shape (shots,2), got {residual_bits.shape}")

    z = residual_bits[:, 0]
    x = residual_bits[:, 1]

    p_I = float(np.mean((z == 0) & (x == 0)))
    p_Z = float(np.mean((z == 1) & (x == 0)))
    p_X = float(np.mean((z == 0) & (x == 1)))
    p_Y = float(np.mean((z == 1) & (x == 1)))

    logical_failure_rate = 1.0 - p_I
    average_gate_fidelity = 1.0 - (2.0 / 3.0) * logical_failure_rate

    # X-memory fails when residual Z or Y remains  <=> z == 1
    # Z-memory fails when residual X or Y remains  <=> x == 1
    x_basis_memory_fidelity = float(np.mean(z == 0))  # p_I + p_X
    z_basis_memory_fidelity = float(np.mean(x == 0))  # p_I + p_Z

    return {
        "p_I": p_I,
        "p_X": p_X,
        "p_Z": p_Z,
        "p_Y": p_Y,
        "strict_logical_identity_success": p_I,
        "strict_logical_failure_rate": logical_failure_rate,
        "average_gate_fidelity": average_gate_fidelity,
        "x_basis_memory_fidelity": x_basis_memory_fidelity,
        "z_basis_memory_fidelity": z_basis_memory_fidelity,
        "logical_x_failure_rate": float(np.mean(x == 1)),  # X or Y
        "logical_z_failure_rate": float(np.mean(z == 1)),  # Z or Y
        "counts": {
            "I": int(np.sum((z == 0) & (x == 0))),
            "Z": int(np.sum((z == 1) & (x == 0))),
            "X": int(np.sum((z == 0) & (x == 1))),
            "Y": int(np.sum((z == 1) & (x == 1))),
        },
    }


def _decode_predictions_from_v6_data(
    *,
    data,
    p: float,
    peps_nkeep: int,
    peps_nsweep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For v6 sampler:
      - batch_x carries the x-memory circuit data
      - batch_z carries the z-memory circuit data

    We decode them separately, then combine:
      predicted logical bits = [z_log, x_log]
    where
      z_log comes from x-memory decoding,
      x_log comes from z-memory decoding.
    """
    if p == 0.0:
        zero = np.zeros((data.logical_bits.shape[0], 2), dtype=np.uint8)
        return zero.copy(), zero.copy()

    batch_x = _clone_batch_with_one_observable(data.batch_x, data.batch_x.observable_flips)
    batch_z = _clone_batch_with_one_observable(data.batch_z, data.batch_z.observable_flips)

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

    out_x_mwpm = decode_2d_surface_batch_with_mwpm(
        batch=batch_x,
        p=p,
        memory_basis="x",
    )
    out_z_mwpm = decode_2d_surface_batch_with_mwpm(
        batch=batch_z,
        p=p,
        memory_basis="z",
    )

    z_log_peps = np.asarray(out_x_peps.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    x_log_peps = np.asarray(out_z_peps.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    peps_pred = np.concatenate([z_log_peps, x_log_peps], axis=1)

    z_log_mwpm = np.asarray(out_x_mwpm.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    x_log_mwpm = np.asarray(out_z_mwpm.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    mwpm_pred = np.concatenate([z_log_mwpm, x_log_mwpm], axis=1)

    return peps_pred, mwpm_pred


# ============================================================
# Main debug entry points
# ============================================================

def debug_one_point_v6(
    *,
    distance: int = 5,
    p: float = 0.01,
    shots: int = 512,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    seed: int = 1234,
    print_examples: int = 10,
) -> Dict[str, Any]:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=distance,
        p=p,
        shots=shots,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        seed=seed,
    )

    true_bits = np.asarray(data.logical_bits, dtype=np.uint8)
    if true_bits.ndim != 2 or true_bits.shape[1] != 2:
        raise ValueError(f"Expected true logical bits shape (shots,2), got {true_bits.shape}")

    print("\nSampler truth diagnostics:")
    print("  true logical sectors:", _sector_hist(true_bits))
    print("  logical_bits bit-1 fraction:", _bit1_fraction(true_bits))

    print("\nBatch-X diagnostics (x-memory):")
    print("  observable_flips shape:", np.asarray(data.batch_x.observable_flips).shape)
    print("  observable bit-1 fraction:", _bit1_fraction(np.asarray(data.batch_x.observable_flips, dtype=np.uint8)))
    print("  detector stats:", _weight_stats(np.asarray(data.batch_x.detector_bits, dtype=np.uint8)))
    print("  sX stats:", _weight_stats(np.asarray(data.batch_x.sX, dtype=np.uint8)))
    print("  sZ stats:", _weight_stats(np.asarray(data.batch_x.sZ, dtype=np.uint8)))

    print("\nBatch-Z diagnostics (z-memory):")
    print("  observable_flips shape:", np.asarray(data.batch_z.observable_flips).shape)
    print("  observable bit-1 fraction:", _bit1_fraction(np.asarray(data.batch_z.observable_flips, dtype=np.uint8)))
    print("  detector stats:", _weight_stats(np.asarray(data.batch_z.detector_bits, dtype=np.uint8)))
    print("  sX stats:", _weight_stats(np.asarray(data.batch_z.sX, dtype=np.uint8)))
    print("  sZ stats:", _weight_stats(np.asarray(data.batch_z.sZ, dtype=np.uint8)))

    # The most important v6 invariant:
    # logical_bits[:,0] should agree with x-memory observable
    # logical_bits[:,1] should agree with z-memory observable
    stim_zlog = np.asarray(data.batch_x.observable_flips, dtype=np.uint8).reshape(-1)
    stim_xlog = np.asarray(data.batch_z.observable_flips, dtype=np.uint8).reshape(-1)

    zlog_mismatch = np.where(stim_zlog != true_bits[:, 0])[0]
    xlog_mismatch = np.where(stim_xlog != true_bits[:, 1])[0]

    print("\nTruth consistency checks:")
    print("  x-memory observable vs true_bits[:,0] mismatches:", len(zlog_mismatch))
    print("  z-memory observable vs true_bits[:,1] mismatches:", len(xlog_mismatch))

    if len(zlog_mismatch):
        print("  example z-log mismatch shots:", zlog_mismatch[:min(10, len(zlog_mismatch))].tolist())
    if len(xlog_mismatch):
        print("  example x-log mismatch shots:", xlog_mismatch[:min(10, len(xlog_mismatch))].tolist())

    peps_pred, mwpm_pred = _decode_predictions_from_v6_data(
        data=data,
        p=p,
        peps_nkeep=peps_nkeep,
        peps_nsweep=peps_nsweep,
    )

    print("\nRaw decoder-output diagnostics:")
    print("  PEPS predicted logical bits bit-1 fraction:", _bit1_fraction(peps_pred))
    print("  MWPM predicted logical bits bit-1 fraction:", _bit1_fraction(mwpm_pred))
    print("  PEPS predicted sectors:", _sector_hist(peps_pred))
    print("  MWPM predicted sectors:", _sector_hist(mwpm_pred))

    zero_pred = np.zeros_like(true_bits, dtype=np.uint8)

    no_dec_residual = np.bitwise_xor(true_bits, zero_pred)
    peps_residual = np.bitwise_xor(true_bits, peps_pred)
    mwpm_residual = np.bitwise_xor(true_bits, mwpm_pred)

    out = {
        "params": {
            "distance": distance,
            "p": p,
            "shots": shots,
            "rounds": rounds,
            "noisy_round": noisy_round,
            "target_t": target_t,
            "peps_nkeep": peps_nkeep,
            "peps_nsweep": peps_nsweep,
            "seed": seed,
        },
        "truth": {
            "true_sector_hist": _sector_hist(true_bits),
        },
        "no_decoder": _summarize_residual(no_dec_residual),
        "peps": _summarize_residual(peps_residual),
        "mwpm": _summarize_residual(mwpm_residual),
        "agreement": {
            "peps_vs_mwpm_full_two_bit_agreement": float(np.mean(np.all(peps_pred == mwpm_pred, axis=1))),
            "peps_vs_truth_full_two_bit_success": float(np.mean(np.all(peps_pred == true_bits, axis=1))),
            "mwpm_vs_truth_full_two_bit_success": float(np.mean(np.all(mwpm_pred == true_bits, axis=1))),
        },
        "raw": {
            "true_bits": true_bits,
            "peps_pred": peps_pred,
            "mwpm_pred": mwpm_pred,
            "no_dec_residual": no_dec_residual,
            "peps_residual": peps_residual,
            "mwpm_residual": mwpm_residual,
        }
    }

    print("=" * 90)
    print("DEBUG ONE POINT V6")
    print(out["params"])
    print("=" * 90)

    for key in ["no_decoder", "peps", "mwpm"]:
        s = out[key]
        print(f"\n[{key}]")
        print(f"  p_I / strict logical fidelity      = {s['strict_logical_identity_success']:.6f}")
        print(f"  logical failure rate               = {s['strict_logical_failure_rate']:.6f}")
        print(f"  average gate fidelity              = {s['average_gate_fidelity']:.6f}")
        print(f"  X-basis memory fidelity            = {s['x_basis_memory_fidelity']:.6f}")
        print(f"  Z-basis memory fidelity            = {s['z_basis_memory_fidelity']:.6f}")
        print(f"  sector counts                      = {s['counts']}")

    print("\n[agreement]")
    for k, v in out["agreement"].items():
        print(f"  {k:40s} = {v:.6f}")

    peps_bad = np.where(np.all(no_dec_residual == 0, axis=1) & np.any(peps_residual != 0, axis=1))[0]
    mwpm_bad = np.where(np.all(no_dec_residual == 0, axis=1) & np.any(mwpm_residual != 0, axis=1))[0]

    print(f"\nShots where PEPS turns a clean logical shot into a failure: {len(peps_bad)}")
    print(f"Shots where MWPM turns a clean logical shot into a failure: {len(mwpm_bad)}")

    if print_examples > 0:
        print("\nExample PEPS-misdecoded shots:")
        for idx in peps_bad[:print_examples]:
            print(
                f"  shot={idx:5d} "
                f"true={true_bits[idx].tolist()} "
                f"pred={peps_pred[idx].tolist()} "
                f"res={peps_residual[idx].tolist()}"
            )

        print("\nExample MWPM-misdecoded shots:")
        for idx in mwpm_bad[:print_examples]:
            print(
                f"  shot={idx:5d} "
                f"true={true_bits[idx].tolist()} "
                f"pred={mwpm_pred[idx].tolist()} "
                f"res={mwpm_residual[idx].tolist()}"
            )

    # Optional extra: compare against the old v5 driver if it still imports.
    try:
        print("\nCalling compare_peps_mwpm_surface_code_full_logical_v5 for cross-check...")
        table = compare_peps_mwpm_surface_code_full_logical(
            distance=distance,
            p_values=[p],
            shots=shots,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            peps_nkeep=peps_nkeep,
            peps_nsweep=peps_nsweep,
            seed=seed,
        )
        print("compare_peps_mwpm_surface_code_full_logical_v5 returned successfully.")
        if hasattr(table, "pretty_print"):
            table.pretty_print()
        out["v5_compare_call_ok"] = True
    except Exception as exc:
        print("compare_peps_mwpm_surface_code_full_logical_v5 raised:")
        print(f"  {type(exc).__name__}: {exc}")
        out["v5_compare_call_ok"] = False
        out["v5_compare_exception"] = repr(exc)

    return out


def debug_sweep_v6(
    *,
    distance: int = 5,
    p_values=(0.002, 0.005, 0.01, 0.02),
    shots: int = 400,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    seed: int = 1234,
) -> None:
    print(
        "      p | no-dec pI |  PEPS pI |  MWPM pI | "
        " no-dec FX |   PEPS FX |   MWPM FX | "
        " no-dec FZ |   PEPS FZ |   MWPM FZ"
    )
    print("-" * 118)

    for i, p in enumerate(p_values):
        out = debug_one_point_v6(
            distance=distance,
            p=p,
            shots=shots,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            peps_nkeep=peps_nkeep,
            peps_nsweep=peps_nsweep,
            seed=seed + i,
            print_examples=0,
        )

        nd = out["no_decoder"]
        pe = out["peps"]
        mw = out["mwpm"]

        print(
            f"{p:7.4f} | "
            f"{nd['strict_logical_identity_success']:9.4f} | "
            f"{pe['strict_logical_identity_success']:8.4f} | "
            f"{mw['strict_logical_identity_success']:8.4f} | "
            f"{nd['x_basis_memory_fidelity']:10.4f} | "
            f"{pe['x_basis_memory_fidelity']:10.4f} | "
            f"{mw['x_basis_memory_fidelity']:10.4f} | "
            f"{nd['z_basis_memory_fidelity']:10.4f} | "
            f"{pe['z_basis_memory_fidelity']:10.4f} | "
            f"{mw['z_basis_memory_fidelity']:10.4f}"
        )


def quick_sanity_tests_v6() -> None:
    # p=0 check
    out0 = debug_one_point_v6(p=0.0, shots=128, print_examples=0)
    for name in ["no_decoder", "peps", "mwpm"]:
        s = out0[name]
        assert abs(s["strict_logical_identity_success"] - 1.0) < 1e-12, name
        assert abs(s["x_basis_memory_fidelity"] - 1.0) < 1e-12, name
        assert abs(s["z_basis_memory_fidelity"] - 1.0) < 1e-12, name

    # strict identity <= basis fidelities
    out = debug_one_point_v6(p=0.01, shots=256, print_examples=0)
    for name in ["no_decoder", "peps", "mwpm"]:
        s = out[name]
        assert s["strict_logical_identity_success"] <= s["x_basis_memory_fidelity"] + 1e-12
        assert s["strict_logical_identity_success"] <= s["z_basis_memory_fidelity"] + 1e-12

    print("quick_sanity_tests_v6 passed.")