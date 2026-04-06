
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.Surface_code_sampler.surface_code_capacity_sampler_full_logical_v9 import (
    StimSurfaceBatchSampleFullLogical,
    sample_surface_code_capacity_batch_full_logical,
)
from src.ML_decoder_PEPS.PEPS_Pauli_decoder_v5 import decode_batch_with_peps_v5
decode_batch_with_peps=decode_batch_with_peps_v5
from src.MWPM_decoder_pymatching.mwpm_decoder_2d import decode_2d_surface_batch_with_mwpm


@dataclass(frozen=True)
class DecoderFullLogicalPoint:
    p: float
    shots: int

    peps_pI: float
    peps_pX: float
    peps_pZ: float
    peps_pY: float
    peps_fail: float
    peps_favg: float
    peps_fx: float
    peps_fz: float

    mwpm_pI: float
    mwpm_pX: float
    mwpm_pZ: float
    mwpm_pY: float
    mwpm_fail: float
    mwpm_favg: float
    mwpm_fx: float
    mwpm_fz: float

    no_decoder_pI: float
    no_decoder_pX: float
    no_decoder_pZ: float
    no_decoder_pY: float
    no_decoder_fail: float
    no_decoder_favg: float
    no_decoder_fx: float
    no_decoder_fz: float

    peps_vs_mwpm_agreement_rate: float
    true_nontrivial_fraction: float


@dataclass
class DecoderFullLogicalTable:
    points: List[DecoderFullLogicalPoint]

    def pretty_print(self) -> None:
        print(
            "     p | no-dec pI |  PEPS pI |  MWPM pI | "
            "no-dec FX |  PEPS FX |  MWPM FX | "
            "no-dec FZ |  PEPS FZ |  MWPM FZ | Agree"
        )
        print("-" * 128)
        for pt in self.points:
            print(
                f"{pt.p:7.4f} | "
                f"{pt.no_decoder_pI:9.4f} | "
                f"{pt.peps_pI:8.4f} | "
                f"{pt.mwpm_pI:8.4f} | "
                f"{pt.no_decoder_fx:9.4f} | "
                f"{pt.peps_fx:9.4f} | "
                f"{pt.mwpm_fx:9.4f} | "
                f"{pt.no_decoder_fz:9.4f} | "
                f"{pt.peps_fz:9.4f} | "
                f"{pt.mwpm_fz:9.4f} | "
                f"{pt.peps_vs_mwpm_agreement_rate:5.4f}"
            )


def _clone_batch_with_one_observable(batch, observable_flips):
    obs = np.asarray(observable_flips, dtype=np.uint8)
    if obs.ndim == 1:
        obs = obs.reshape(-1, 1)
    return replace(batch, observable_flips=obs)


def _summarize_residual(residual_bits: np.ndarray) -> Dict[str, Any]:
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

    # X-memory fails when residual Z or Y remains <=> z == 1
    # Z-memory fails when residual X or Y remains <=> x == 1
    x_basis_memory_fidelity = float(np.mean(z == 0))
    z_basis_memory_fidelity = float(np.mean(x == 0))

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
    }


def _decode_predictions_from_v7_data(
    *,
    data: StimSurfaceBatchSampleFullLogical,
    p: float,
    peps_nkeep: int,
    peps_nsweep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For the full-logical sampler:

    - batch_x carries the x-memory circuit data
    - batch_z carries the z-memory circuit data

    We decode them separately, then combine predicted logical bits as
        [z_log, x_log]
    where z_log comes from x-memory decoding and x_log comes from z-memory
    decoding.
    """
    shots = data.logical_bits.shape[0]
    if p == 0.0:
        zero = np.zeros((shots, 2), dtype=np.uint8)
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
        debug_shots=5,
    )
    out_z_peps = decode_batch_with_peps(
        batch=batch_z,
        p=p,
        memory_basis="z",
        Nkeep=peps_nkeep,
        Nsweep=peps_nsweep,
        debug_failures=False,
        debug_shots=5,
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


def compare_peps_mwpm_surface_code_full_logical(
    *,
    distance: int,
    p_values: Iterable[float],
    shots: int,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    seed: Optional[int] = None,
) -> DecoderFullLogicalTable:
    points: List[DecoderFullLogicalPoint] = []
    seed_seq = np.random.SeedSequence(seed)

    for p in list(p_values):
        child_seed = int(seed_seq.spawn(1)[0].generate_state(1)[0]) if seed is not None else None

        data = sample_surface_code_capacity_batch_full_logical(
            distance=distance,
            p=p,
            shots=shots,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            seed=child_seed,
        )

        true_bits = np.asarray(data.logical_bits, dtype=np.uint8)

        peps_pred, mwpm_pred = _decode_predictions_from_v7_data(
            data=data,
            p=p,
            peps_nkeep=peps_nkeep,
            peps_nsweep=peps_nsweep,
        )

        no_decoder_residual = true_bits.copy()
        peps_residual = np.bitwise_xor(peps_pred, true_bits)
        mwpm_residual = np.bitwise_xor(mwpm_pred, true_bits)

        nd = _summarize_residual(no_decoder_residual)
        pe = _summarize_residual(peps_residual)
        mw = _summarize_residual(mwpm_residual)

        points.append(
            DecoderFullLogicalPoint(
                p=float(p),
                shots=int(shots),
                peps_pI=pe["p_I"],
                peps_pX=pe["p_X"],
                peps_pZ=pe["p_Z"],
                peps_pY=pe["p_Y"],
                peps_fail=pe["strict_logical_failure_rate"],
                peps_favg=pe["average_gate_fidelity"],
                peps_fx=pe["x_basis_memory_fidelity"],
                peps_fz=pe["z_basis_memory_fidelity"],
                mwpm_pI=mw["p_I"],
                mwpm_pX=mw["p_X"],
                mwpm_pZ=mw["p_Z"],
                mwpm_pY=mw["p_Y"],
                mwpm_fail=mw["strict_logical_failure_rate"],
                mwpm_favg=mw["average_gate_fidelity"],
                mwpm_fx=mw["x_basis_memory_fidelity"],
                mwpm_fz=mw["z_basis_memory_fidelity"],
                no_decoder_pI=nd["p_I"],
                no_decoder_pX=nd["p_X"],
                no_decoder_pZ=nd["p_Z"],
                no_decoder_pY=nd["p_Y"],
                no_decoder_fail=nd["strict_logical_failure_rate"],
                no_decoder_favg=nd["average_gate_fidelity"],
                no_decoder_fx=nd["x_basis_memory_fidelity"],
                no_decoder_fz=nd["z_basis_memory_fidelity"],
                peps_vs_mwpm_agreement_rate=float(np.mean(np.all(peps_pred == mwpm_pred, axis=1))),
                true_nontrivial_fraction=float(np.mean(np.any(true_bits != 0, axis=1))),
            )
        )

    return DecoderFullLogicalTable(points=points)


if __name__ == "__main__":
    table = compare_peps_mwpm_surface_code_full_logical(
        distance=5,
        p_values=[0.002, 0.005, 0.01, 0.02],
        shots=200,
        rounds=3,
        noisy_round=2,
        target_t=1,
        peps_nkeep=128,
        peps_nsweep=1,
        seed=1234,
    )
    table.pretty_print()
