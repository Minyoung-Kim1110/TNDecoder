from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .metric import *
from .ML_decoder_PEPS.PEPS_Pauli_decoder import decode_batch_with_peps  # type: ignore
from .MWPM_decoder_pymatching.mwpm_decoder_2d import decode_2d_surface_batch_with_mwpm
from .Surface_code_sampler.stim_sampler import StimSurfaceBatchSample
from .Surface_code_sampler.surface_code_capacity_sampler import (
    sample_surface_code_capacity_batch,
)



# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DecoderBasisComparisonPoint:
    p: float
    shots: int
    memory_basis: str
    peps_failure_rate: float
    peps_memory_fidelity: float
    mwpm_failure_rate: float
    mwpm_memory_fidelity: float
    peps_num_failures: int
    mwpm_num_failures: int
    peps_predicted_observable_flips: np.ndarray
    mwpm_predicted_observable_flips: np.ndarray
    decoder_agreement_rate: float

    @property
    def delta_failure_rate(self) -> float:
        return self.peps_failure_rate - self.mwpm_failure_rate

    @property
    def delta_memory_fidelity(self) -> float:
        return self.peps_memory_fidelity - self.mwpm_memory_fidelity


@dataclass
class DecoderBasisComparisonTable:
    distance: int
    rounds: int
    noisy_round: int
    target_t: int
    shots: int
    memory_basis: str
    points: List[DecoderBasisComparisonPoint]

    def as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "p": pt.p,
                "shots": pt.shots,
                "memory_basis": pt.memory_basis,
                "peps_failure_rate": pt.peps_failure_rate,
                "peps_memory_fidelity": pt.peps_memory_fidelity,
                "mwpm_failure_rate": pt.mwpm_failure_rate,
                "mwpm_memory_fidelity": pt.mwpm_memory_fidelity,
                "peps_num_failures": pt.peps_num_failures,
                "mwpm_num_failures": pt.mwpm_num_failures,
                "decoder_agreement_rate": pt.decoder_agreement_rate,
                "delta_failure_rate": pt.delta_failure_rate,
                "delta_memory_fidelity": pt.delta_memory_fidelity,
            }
            for pt in self.points
        ]

    def pretty_print(self) -> None:
        header = (
            "      p | shots | basis |  PEPS fail |  MWPM fail | "
            " PEPS fid | MWPM fid | agree"
        )
        print(header)
        print("-" * len(header))
        for pt in self.points:
            print(
                f"{pt.p:8.5g} | {pt.shots:5d} | {pt.memory_basis:>5s} | "
                f"{pt.peps_failure_rate:10.6f} | {pt.mwpm_failure_rate:10.6f} | "
                f"{pt.peps_memory_fidelity:9.6f} | {pt.mwpm_memory_fidelity:9.6f} | "
                f"{pt.decoder_agreement_rate:5.3f}"
            )


@dataclass
class DecoderCombinedComparisonPoint:
    p: float
    shots: int

    # Basis-conditioned memory quantities.
    peps_x_basis_memory_failure_rate: float
    peps_x_basis_memory_fidelity: float
    peps_z_basis_memory_failure_rate: float
    peps_z_basis_memory_fidelity: float
    mwpm_x_basis_memory_failure_rate: float
    mwpm_x_basis_memory_fidelity: float
    mwpm_z_basis_memory_failure_rate: float
    mwpm_z_basis_memory_fidelity: float

    # Logical-type failure rates inferred from basis-conditioned experiments.
    # memory_basis='z' probes logical X failure.
    peps_logical_x_failure_rate: float
    mwpm_logical_x_failure_rate: float
    # memory_basis='x' probes logical Z failure.
    peps_logical_z_failure_rate: float
    mwpm_logical_z_failure_rate: float

    # Aggregate summary from the two basis-conditioned memory fidelities.
    peps_average_memory_fidelity: float
    mwpm_average_memory_fidelity: float

    # Decoder agreement rates within each basis.
    x_basis_decoder_agreement_rate: float
    z_basis_decoder_agreement_rate: float

    @property
    def peps_average_failure_rate(self) -> float:
        return 1.0 - self.peps_average_memory_fidelity

    @property
    def mwpm_average_failure_rate(self) -> float:
        return 1.0 - self.mwpm_average_memory_fidelity


@dataclass
class DecoderCombinedComparisonTable:
    distance: int
    rounds: int
    noisy_round: int
    target_t: int
    shots: int
    x_basis_table: DecoderBasisComparisonTable
    z_basis_table: DecoderBasisComparisonTable
    points: List[DecoderCombinedComparisonPoint]

    def as_dicts(self) -> List[Dict[str, float]]:
        return [
            {
                "p": pt.p,
                "shots": pt.shots,
                "peps_logical_x_failure_rate": pt.peps_logical_x_failure_rate,
                "mwpm_logical_x_failure_rate": pt.mwpm_logical_x_failure_rate,
                "peps_logical_z_failure_rate": pt.peps_logical_z_failure_rate,
                "mwpm_logical_z_failure_rate": pt.mwpm_logical_z_failure_rate,
                "peps_x_basis_memory_fidelity": pt.peps_x_basis_memory_fidelity,
                "mwpm_x_basis_memory_fidelity": pt.mwpm_x_basis_memory_fidelity,
                "peps_z_basis_memory_fidelity": pt.peps_z_basis_memory_fidelity,
                "mwpm_z_basis_memory_fidelity": pt.mwpm_z_basis_memory_fidelity,
                "peps_average_memory_fidelity": pt.peps_average_memory_fidelity,
                "mwpm_average_memory_fidelity": pt.mwpm_average_memory_fidelity,
                "x_basis_decoder_agreement_rate": pt.x_basis_decoder_agreement_rate,
                "z_basis_decoder_agreement_rate": pt.z_basis_decoder_agreement_rate,
            }
            for pt in self.points
        ]

    def pretty_print(self) -> None:
        header = (
            "      p | PEPS LX fail | MWPM LX fail | PEPS LZ fail | MWPM LZ fail | "
            "PEPS avg fid | MWPM avg fid"
        )
        print(header)
        print("-" * len(header))
        for pt in self.points:
            print(
                f"{pt.p:8.5g} | {pt.peps_logical_x_failure_rate:12.6f} | "
                f"{pt.mwpm_logical_x_failure_rate:12.6f} | "
                f"{pt.peps_logical_z_failure_rate:12.6f} | "
                f"{pt.mwpm_logical_z_failure_rate:12.6f} | "
                f"{pt.peps_average_memory_fidelity:12.6f} | "
                f"{pt.mwpm_average_memory_fidelity:12.6f}"
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


def _ensure_batch(batch: StimSurfaceBatchSample, shots: int) -> None:
    if batch.shots != shots:
        raise ValueError(f"Expected batch.shots={shots}, got {batch.shots}.")
    if batch.observable_flips.ndim != 2:
        raise ValueError(
            "Expected batch.observable_flips to have shape (shots, num_obs)."
        )


def _compare_on_existing_batch(
    batch: StimSurfaceBatchSample,
    *,
    p: float,
    memory_basis: str,
    peps_nkeep: int,
    peps_nsweep: int,
    peps_debug_failures: bool = False,
) -> DecoderBasisComparisonPoint:
    peps_out = decode_batch_with_peps(
        batch=batch,
        p=p,
        memory_basis=memory_basis,
        Nkeep=peps_nkeep,
        Nsweep=peps_nsweep,
        debug_failures=peps_debug_failures,
    )
    mwpm_out = decode_2d_surface_batch_with_mwpm(
        batch=batch,
        p=p,
        memory_basis=memory_basis,
    )

    peps_pred = np.asarray(peps_out.predicted_observable_flips, dtype=np.uint8)
    mwpm_pred = np.asarray(mwpm_out.predicted_observable_flips, dtype=np.uint8)
    actual = np.asarray(batch.observable_flips, dtype=np.uint8)

    return DecoderBasisComparisonPoint(
        p=float(p),
        shots=int(batch.shots),
        memory_basis=str(memory_basis),
        peps_failure_rate=float(peps_out.logical_error_rate),
        peps_memory_fidelity=float(
            logical_fidelity_from_predictions(actual, peps_pred)
        ),
        mwpm_failure_rate=float(mwpm_out.logical_error_rate),
        mwpm_memory_fidelity=float(
            logical_fidelity_from_predictions(actual, mwpm_pred)
        ),
        peps_num_failures=int(np.sum(peps_out.logical_failures)),
        mwpm_num_failures=int(np.sum(mwpm_out.logical_failures)),
        peps_predicted_observable_flips=peps_pred,
        mwpm_predicted_observable_flips=mwpm_pred,
        decoder_agreement_rate=float(np.mean(np.all(peps_pred == mwpm_pred, axis=1))),
    )


def _validate_capacity_alignment(rounds: int, noisy_round: int, target_t: int) -> None:
    if rounds < 3:
        raise ValueError("rounds must be >= 3.")
    if not (1 <= noisy_round <= rounds):
        raise ValueError("noisy_round must satisfy 1 <= noisy_round <= rounds.")
    if target_t != noisy_round - 1:
        raise ValueError(
            "For the single-round capacity sampler, use target_t = noisy_round - 1 "
            "so the decoded detector slice matches the unique noisy data round."
        )


# ---------------------------------------------------------------------------
# Public comparison API
# ---------------------------------------------------------------------------

def compare_peps_mwpm_surface_code_basis(
    *,
    distance: int,
    p_values: Sequence[float],
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    noisy_round: int = 2,
    target_t: int = 1,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    peps_debug_failures: bool = False,
) -> DecoderBasisComparisonTable:
    """
    Compare PEPS ML and 2D MWPM on the same surface-code capacity batches for
    one chosen memory basis.

    Interpretation:
      - memory_basis='x' probes logical Z-type failure / X-basis memory fidelity.
      - memory_basis='z' probes logical X-type failure / Z-basis memory fidelity.
    """
    if distance < 2:
        raise ValueError("distance must be at least 2.")
    if shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")

    _validate_capacity_alignment(rounds, noisy_round, target_t)
    p_grid = _validate_probability_grid(p_values)

    points: List[DecoderBasisComparisonPoint] = []
    for p in p_grid:
        batch = sample_surface_code_capacity_batch(
            distance=distance,
            p=p,
            shots=shots,
            memory_basis=memory_basis,
            rounds=rounds,
            noisy_round=noisy_round,
            target_t=target_t,
        )
        _ensure_batch(batch, shots)
        points.append(
            _compare_on_existing_batch(
                batch=batch,
                p=p,
                memory_basis=memory_basis,
                peps_nkeep=peps_nkeep,
                peps_nsweep=peps_nsweep,
                peps_debug_failures=peps_debug_failures,
            )
        )

    return DecoderBasisComparisonTable(
        distance=distance,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        shots=shots,
        memory_basis=memory_basis,
        points=points,
    )


def compare_peps_mwpm_surface_code(
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
) -> DecoderCombinedComparisonTable:
    """
    Run both memory bases and return separate logical-X / logical-Z failure curves.

    Important:
    This current workflow still exposes one logical observable per basis, not the
    full two-bit logical Pauli class (I, X, Z, Y). Therefore:
      - logical X failure rate is inferred from memory_basis='z',
      - logical Z failure rate is inferred from memory_basis='x',
      - average_memory_fidelity is the mean of the X-basis and Z-basis memory
        fidelities, not the full logical average gate fidelity of the residual
        logical Pauli channel.
    """
    x_basis_table = compare_peps_mwpm_surface_code_basis(
        distance=distance,
        p_values=p_values,
        shots=shots,
        memory_basis="x",
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        peps_nkeep=peps_nkeep,
        peps_nsweep=peps_nsweep,
        peps_debug_failures=peps_debug_failures,
    )
    z_basis_table = compare_peps_mwpm_surface_code_basis(
        distance=distance,
        p_values=p_values,
        shots=shots,
        memory_basis="z",
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        peps_nkeep=peps_nkeep,
        peps_nsweep=peps_nsweep,
        peps_debug_failures=peps_debug_failures,
    )

    if len(x_basis_table.points) != len(z_basis_table.points):
        raise RuntimeError("Expected x/z basis tables to have the same length.")

    points: List[DecoderCombinedComparisonPoint] = []
    for x_pt, z_pt in zip(x_basis_table.points, z_basis_table.points):
        if not np.isclose(x_pt.p, z_pt.p):
            raise RuntimeError("Expected matching p values between x/z basis runs.")

        points.append(
            DecoderCombinedComparisonPoint(
                p=float(x_pt.p),
                shots=int(x_pt.shots),
                peps_x_basis_memory_failure_rate=float(x_pt.peps_failure_rate),
                peps_x_basis_memory_fidelity=float(x_pt.peps_memory_fidelity),
                peps_z_basis_memory_failure_rate=float(z_pt.peps_failure_rate),
                peps_z_basis_memory_fidelity=float(z_pt.peps_memory_fidelity),
                mwpm_x_basis_memory_failure_rate=float(x_pt.mwpm_failure_rate),
                mwpm_x_basis_memory_fidelity=float(x_pt.mwpm_memory_fidelity),
                mwpm_z_basis_memory_failure_rate=float(z_pt.mwpm_failure_rate),
                mwpm_z_basis_memory_fidelity=float(z_pt.mwpm_memory_fidelity),
                peps_logical_x_failure_rate=float(z_pt.peps_failure_rate),
                mwpm_logical_x_failure_rate=float(z_pt.mwpm_failure_rate),
                peps_logical_z_failure_rate=float(x_pt.peps_failure_rate),
                mwpm_logical_z_failure_rate=float(x_pt.mwpm_failure_rate),
                peps_average_memory_fidelity=float(
                    0.5 * (x_pt.peps_memory_fidelity + z_pt.peps_memory_fidelity)
                ),
                mwpm_average_memory_fidelity=float(
                    0.5 * (x_pt.mwpm_memory_fidelity + z_pt.mwpm_memory_fidelity)
                ),
                x_basis_decoder_agreement_rate=float(x_pt.decoder_agreement_rate),
                z_basis_decoder_agreement_rate=float(z_pt.decoder_agreement_rate),
            )
        )

    return DecoderCombinedComparisonTable(
        distance=distance,
        rounds=rounds,
        noisy_round=noisy_round,
        target_t=target_t,
        shots=shots,
        x_basis_table=x_basis_table,
        z_basis_table=z_basis_table,
        points=points,
    )


def compare_peps_mwpm_on_one_batch(
    *,
    batch: StimSurfaceBatchSample,
    p: float,
    memory_basis: str,
    peps_nkeep: int = 128,
    peps_nsweep: int = 1,
    peps_debug_failures: bool = False,
) -> DecoderBasisComparisonPoint:
    return _compare_on_existing_batch(
        batch=batch,
        p=p,
        memory_basis=memory_basis,
        peps_nkeep=peps_nkeep,
        peps_nsweep=peps_nsweep,
        peps_debug_failures=peps_debug_failures,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compare_peps_mwpm_surface_code_basis_shapes() -> None:
    table = compare_peps_mwpm_surface_code_basis(
        distance=3,
        p_values=[1e-4, 5e-4],
        shots=4,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
        peps_nkeep=32,
        peps_nsweep=0,
    )
    assert len(table.points) == 2
    for pt in table.points:
        assert pt.peps_predicted_observable_flips.shape == (4, 1)
        assert pt.mwpm_predicted_observable_flips.shape == (4, 1)
        assert 0.0 <= pt.peps_failure_rate <= 1.0
        assert 0.0 <= pt.mwpm_failure_rate <= 1.0
        assert np.isclose(pt.peps_memory_fidelity, 1.0 - pt.peps_failure_rate, atol=1e-12)
        assert np.isclose(pt.mwpm_memory_fidelity, 1.0 - pt.mwpm_failure_rate, atol=1e-12)
    print("test_compare_peps_mwpm_surface_code_basis_shapes passed.")


def test_compare_peps_mwpm_on_existing_batch() -> None:
    batch = sample_surface_code_capacity_batch(
        distance=3,
        p=1e-3,
        shots=5,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
    )
    result = compare_peps_mwpm_on_one_batch(
        batch=batch,
        p=1e-3,
        memory_basis="x",
        peps_nkeep=32,
        peps_nsweep=0,
    )
    assert result.shots == batch.shots
    assert result.peps_predicted_observable_flips.shape == batch.observable_flips.shape
    assert result.mwpm_predicted_observable_flips.shape == batch.observable_flips.shape
    print("test_compare_peps_mwpm_on_existing_batch passed.")


def test_compare_peps_mwpm_surface_code_combined() -> None:
    table = compare_peps_mwpm_surface_code(
        distance=3,
        p_values=[1e-4],
        shots=3,
        rounds=3,
        noisy_round=2,
        target_t=1,
        peps_nkeep=32,
        peps_nsweep=0,
    )
    assert len(table.points) == 1
    pt = table.points[0]
    assert 0.0 <= pt.peps_logical_x_failure_rate <= 1.0
    assert 0.0 <= pt.peps_logical_z_failure_rate <= 1.0
    assert 0.0 <= pt.mwpm_logical_x_failure_rate <= 1.0
    assert 0.0 <= pt.mwpm_logical_z_failure_rate <= 1.0
    assert 0.0 <= pt.peps_average_memory_fidelity <= 1.0
    assert 0.0 <= pt.mwpm_average_memory_fidelity <= 1.0
    print("test_compare_peps_mwpm_surface_code_combined passed.")


def run_compare_peps_mwpm_surface_code_tests() -> None:
    test_compare_peps_mwpm_surface_code_basis_shapes()
    test_compare_peps_mwpm_on_existing_batch()
    test_compare_peps_mwpm_surface_code_combined()
    print("All compare_peps_mwpm_surface_code_updated tests passed.")


if __name__ == "__main__":
    run_compare_peps_mwpm_surface_code_tests()

    print("\nX-basis table (probes logical Z-type failure):")
    x_table = compare_peps_mwpm_surface_code_basis(
        distance=3,
        p_values=[1e-4, 5e-4, 1e-3],
        shots=20,
        memory_basis="x",
        rounds=3,
        noisy_round=2,
        target_t=1,
        peps_nkeep=64,
        peps_nsweep=1,
    )
    x_table.pretty_print()

    print("\nCombined x/z-basis summary:")
    combined = compare_peps_mwpm_surface_code(
        distance=3,
        p_values=[1e-4, 5e-4, 1e-3],
        shots=20,
        rounds=3,
        noisy_round=2,
        target_t=1,
        peps_nkeep=64,
        peps_nsweep=1,
    )
    combined.pretty_print()
