from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from src.ML_decoder_PEPS.PEPS_Pauli_decoder import decode_batch_with_peps  # type: ignore
from src.MWPM_decoder_pymatching.mwpm_decoder_2d import decode_2d_surface_batch_with_mwpm
from src.Surface_code_sampler.stim_sampler import StimSurfaceBatchSample
from .Surface_code_sampler.surface_code_sampler_full import (
    StimSurfaceBatchSampleFullLogical,
    sample_surface_code_capacity_batch_full_logical,
)


@dataclass
class LogicalPauliRates:
    p_I: float
    p_X: float
    p_Z: float
    p_Y: float

    @property
    def logical_failure_rate(self) -> float:
        return 1.0 - self.p_I

    @property
    def logical_fidelity(self) -> float:
        return self.p_I

    @property
    def average_gate_fidelity(self) -> float:
        return (1.0 + 2.0 * self.p_I) / 3.0

    @property
    def logical_x_failure_rate(self) -> float:
        return self.p_X + self.p_Y

    @property
    def logical_z_failure_rate(self) -> float:
        return self.p_Z + self.p_Y

    @property
    def x_basis_memory_fidelity(self) -> float:
        return self.p_I + self.p_X

    @property
    def z_basis_memory_fidelity(self) -> float:
        return self.p_I + self.p_Z


@dataclass
class DecoderFullLogicalComparisonPoint:
    p: float
    shots: int
    peps_rates: LogicalPauliRates
    mwpm_rates: LogicalPauliRates
    peps_predicted_logical_bits: np.ndarray
    mwpm_predicted_logical_bits: np.ndarray
    true_logical_bits: np.ndarray
    peps_decoder_agreement_rate_between_bases: float
    mwpm_decoder_agreement_rate_between_bases: float
    peps_vs_mwpm_agreement_rate: float


@dataclass
class DecoderFullLogicalComparisonTable:
    distance: int
    rounds: int
    noisy_round: int
    target_t: int
    shots: int
    points: List[DecoderFullLogicalComparisonPoint]

    def as_dicts(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for pt in self.points:
            out.append(
                {
                    "p": pt.p,
                    "shots": pt.shots,
                    "peps_p_I": pt.peps_rates.p_I,
                    "peps_p_X": pt.peps_rates.p_X,
                    "peps_p_Z": pt.peps_rates.p_Z,
                    "peps_p_Y": pt.peps_rates.p_Y,
                    "peps_logical_failure_rate": pt.peps_rates.logical_failure_rate,
                    "peps_logical_fidelity": pt.peps_rates.logical_fidelity,
                    "peps_average_gate_fidelity": pt.peps_rates.average_gate_fidelity,
                    "peps_logical_x_failure_rate": pt.peps_rates.logical_x_failure_rate,
                    "peps_logical_z_failure_rate": pt.peps_rates.logical_z_failure_rate,
                    "mwpm_p_I": pt.mwpm_rates.p_I,
                    "mwpm_p_X": pt.mwpm_rates.p_X,
                    "mwpm_p_Z": pt.mwpm_rates.p_Z,
                    "mwpm_p_Y": pt.mwpm_rates.p_Y,
                    "mwpm_logical_failure_rate": pt.mwpm_rates.logical_failure_rate,
                    "mwpm_logical_fidelity": pt.mwpm_rates.logical_fidelity,
                    "mwpm_average_gate_fidelity": pt.mwpm_rates.average_gate_fidelity,
                    "mwpm_logical_x_failure_rate": pt.mwpm_rates.logical_x_failure_rate,
                    "mwpm_logical_z_failure_rate": pt.mwpm_rates.logical_z_failure_rate,
                    "peps_vs_mwpm_agreement_rate": pt.peps_vs_mwpm_agreement_rate,
                }
            )
        return out

    def pretty_print(self) -> None:
        header = (
            "      p | PEPS pI | MWPM pI | PEPS fail | MWPM fail | PEPS Favg | MWPM Favg"
        )
        print(header)
        print("-" * len(header))
        for pt in self.points:
            print(
                f"{pt.p:8.5g} | {pt.peps_rates.p_I:7.4f} | {pt.mwpm_rates.p_I:7.4f} | "
                f"{pt.peps_rates.logical_failure_rate:9.4f} | {pt.mwpm_rates.logical_failure_rate:9.4f} | "
                f"{pt.peps_rates.average_gate_fidelity:10.6f} | {pt.mwpm_rates.average_gate_fidelity:10.6f}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_probability_grid(p_values: Sequence[float]) -> List[float]:
    if len(p_values) == 0:
        raise ValueError("p_values must contain at least one depolarizing error rate.")
    out = [float(p) for p in p_values]
    for p in out:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Each p must satisfy 0 <= p <= 1, got {p}.")
    return out


def _clone_batch_with_one_observable(
    batch: StimSurfaceBatchSample,
    observable_flips: np.ndarray,
) -> StimSurfaceBatchSample:
    obs = np.asarray(observable_flips, dtype=np.uint8)
    if obs.ndim != 2 or obs.shape[1] != 1:
        raise ValueError("observable_flips must have shape (shots, 1).")
    if obs.shape[0] != batch.shots:
        raise ValueError("observable_flips must have the same number of shots as batch.")
    return StimSurfaceBatchSample(
        circuit=batch.circuit,
        detector_bits=batch.detector_bits,
        observable_flips=obs,
        sX=batch.sX,
        sZ=batch.sZ,
        active_X=batch.active_X,
        active_Z=batch.active_Z,
        detector_coords=batch.detector_coords,
    )


def _logical_rates_from_residual_bits(residual_bits: np.ndarray) -> LogicalPauliRates:
    residual_bits = np.asarray(residual_bits, dtype=np.uint8)
    if residual_bits.ndim != 2 or residual_bits.shape[1] != 2:
        raise ValueError("residual_bits must have shape (shots, 2) with columns (z_log, x_log).")
    shots = residual_bits.shape[0]
    if shots == 0:
        return LogicalPauliRates(0.0, 0.0, 0.0, 0.0)

    z = residual_bits[:, 0]
    x = residual_bits[:, 1]
    p_I = float(np.mean((z == 0) & (x == 0)))
    p_Z = float(np.mean((z == 1) & (x == 0)))
    p_X = float(np.mean((z == 0) & (x == 1)))
    p_Y = float(np.mean((z == 1) & (x == 1)))
    return LogicalPauliRates(p_I=p_I, p_X=p_X, p_Z=p_Z, p_Y=p_Y)



def _predict_peps_logical_bits(
    data: StimSurfaceBatchSampleFullLogical,
    *,
    p: float,
    peps_nkeep: int,
    peps_nsweep: int,
    peps_debug_failures: bool = False,
) -> np.ndarray:
    # X-basis memory returns the logical Z bit.
    batch_x = _clone_batch_with_one_observable(
        data.batch_x,
        data.batch_x.observable_flips,
    )
    out_x = decode_batch_with_peps(
        batch=batch_x,
        p=p,
        memory_basis="x",
        Nkeep=peps_nkeep,
        Nsweep=peps_nsweep,
        debug_failures=peps_debug_failures,
    )

    # Z-basis memory returns the logical X bit.
    batch_z = _clone_batch_with_one_observable(
        data.batch_z,
        data.batch_z.observable_flips,
    )
    out_z = decode_batch_with_peps(
        batch=batch_z,
        p=p,
        memory_basis="z",
        Nkeep=peps_nkeep,
        Nsweep=peps_nsweep,
        debug_failures=peps_debug_failures,
    )

    z_log = np.asarray(out_x.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    x_log = np.asarray(out_z.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    return np.concatenate([z_log, x_log], axis=1)



def _predict_mwpm_logical_bits(
    data: StimSurfaceBatchSampleFullLogical,
    *,
    p: float,
) -> np.ndarray:
    out_x = decode_2d_surface_batch_with_mwpm(
        batch=_clone_batch_with_one_observable(data.batch_x, data.batch_x.observable_flips),
        p=p,
        memory_basis="x",
    )
    out_z = decode_2d_surface_batch_with_mwpm(
        batch=_clone_batch_with_one_observable(data.batch_z, data.batch_z.observable_flips),
        p=p,
        memory_basis="z",
    )
    z_log = np.asarray(out_x.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    x_log = np.asarray(out_z.predicted_observable_flips, dtype=np.uint8).reshape(-1, 1)
    return np.concatenate([z_log, x_log], axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_peps_mwpm_surface_code_full_logical(
    *,
    distance: int,
    p_values: Sequence[float],
    shots: int,
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    peps_debug_failures: bool = False,
    seed: int | None = None,
) -> DecoderFullLogicalComparisonTable:
    p_grid = _validate_probability_grid(p_values)
    points: List[DecoderFullLogicalComparisonPoint] = []

    # Keep a reproducible but p-dependent RNG stream.
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(len(p_grid))

    for p, child_seed in zip(p_grid, child_seeds):
        data = sample_surface_code_capacity_batch_full_logical(
            distance=distance,
            p=p,
            shots=shots,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
            seed=int(child_seed.generate_state(1)[0]),
        )
        true_bits = np.asarray(data.logical_bits, dtype=np.uint8)

        peps_pred = _predict_peps_logical_bits(
            data,
            p=p,
            peps_nkeep=peps_nkeep,
            peps_nsweep=peps_nsweep,
            peps_debug_failures=peps_debug_failures,
        )
        mwpm_pred = _predict_mwpm_logical_bits(data, p=p)

        peps_residual = np.bitwise_xor(true_bits, peps_pred)
        mwpm_residual = np.bitwise_xor(true_bits, mwpm_pred)

        points.append(
            DecoderFullLogicalComparisonPoint(
                p=float(p),
                shots=shots,
                peps_rates=_logical_rates_from_residual_bits(peps_residual),
                mwpm_rates=_logical_rates_from_residual_bits(mwpm_residual),
                peps_predicted_logical_bits=peps_pred,
                mwpm_predicted_logical_bits=mwpm_pred,
                true_logical_bits=true_bits,
                peps_decoder_agreement_rate_between_bases=1.0,  # placeholder; two-bit reconstruction already combined.
                mwpm_decoder_agreement_rate_between_bases=1.0,
                peps_vs_mwpm_agreement_rate=float(np.mean(np.all(peps_pred == mwpm_pred, axis=1))),
            )
        )

    return DecoderFullLogicalComparisonTable(
        distance=distance,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        shots=shots,
        points=points,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_logical_rates_from_residual_bits() -> None:
    residual = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    rates = _logical_rates_from_residual_bits(residual)
    assert abs(rates.p_I - 0.25) < 1e-12
    assert abs(rates.p_Z - 0.25) < 1e-12
    assert abs(rates.p_X - 0.25) < 1e-12
    assert abs(rates.p_Y - 0.25) < 1e-12
    assert abs(rates.logical_failure_rate - 0.75) < 1e-12
    assert abs(rates.average_gate_fidelity - 0.5) < 1e-12



def test_clone_batch_with_one_observable_shape() -> None:
    data = sample_surface_code_capacity_batch_full_logical(
        distance=3,
        p=0.0,
        shots=2,
        rounds=3,
        noisy_round=2,
        target_t=1,
        seed=11,
    )
    cloned = _clone_batch_with_one_observable(data.batch_x, data.batch_x.observable_flips)
    assert cloned.observable_flips.shape == (2, 1)
    assert cloned.sX.shape[0] == 2



def run_compare_peps_mwpm_surface_code_full_logical_tests() -> None:
    test_logical_rates_from_residual_bits()
    test_clone_batch_with_one_observable_shape()
    print("Lightweight full-logical comparison tests passed.")


if __name__ == "__main__":
    run_compare_peps_mwpm_surface_code_full_logical_tests()
