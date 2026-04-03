from dataclasses import dataclass
from typing import Optional

import numpy as np
import stim
import pymatching

from .stim_sampler import (
    StimSurfaceSample,
    StimSurfaceBatchSample,
    make_unrotated_sc_depolarizing_capacity_circuit,
    sample_surface_code_depolarizing,
    sample_surface_code_depolarizing_batch,
)



@dataclass
class SurfaceCodeMWPMShotResult:
    """Result of decoding one StimSurfaceSample with MWPM."""

    sample: StimSurfaceSample
    predicted_observable_flips: np.ndarray

    @property
    def residual_observable_flips(self) -> np.ndarray:
        return np.bitwise_xor(
            self.sample.observable_flips.astype(np.uint8),
            self.predicted_observable_flips.astype(np.uint8),
        )

    @property
    def logical_failure(self) -> bool:
        return bool(np.any(self.residual_observable_flips != 0))


@dataclass
class SurfaceCodeMWPMBatchResult:
    """Result of decoding a StimSurfaceBatchSample or a freshly sampled batch."""

    circuit: stim.Circuit
    detector_bits: np.ndarray                  # (shots, num_detectors)
    actual_observable_flips: np.ndarray       # (shots, num_obs)
    predicted_observable_flips: np.ndarray    # (shots, num_obs)
    logical_failures: np.ndarray              # (shots,)

    @property
    def num_shots(self) -> int:
        return int(self.detector_bits.shape[0])

    @property
    def num_failures(self) -> int:
        return int(np.sum(self.logical_failures))

    @property
    def logical_error_rate(self) -> float:
        if self.num_shots == 0:
            return 0.0
        return self.num_failures / self.num_shots

    @property
    def logical_success_rate(self) -> float:
        return 1.0 - self.logical_error_rate

# Build matching

def build_matching_from_circuit(
    circuit: stim.Circuit,
    *,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> pymatching.Matching:
    """Build PyMatching decoder from a Stim circuit."""
    dem = circuit.detector_error_model(
        decompose_errors=decompose_errors,
        **dem_kwargs,
    )
    return pymatching.Matching.from_detector_error_model(
        dem,
        enable_correlations=enable_correlations,
    )


def build_matching_from_stim_surface_sample(
    sample: StimSurfaceSample,
    *,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> pymatching.Matching:
    """Build PyMatching decoder from the exact circuit stored in a StimSurfaceSample."""
    return build_matching_from_circuit(
        sample.circuit,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )


def build_matching_from_stim_surface_batch(
    batch: StimSurfaceBatchSample,
    *,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> pymatching.Matching:
    """Build PyMatching decoder from the exact circuit stored in a StimSurfaceBatchSample."""
    return build_matching_from_circuit(
        batch.circuit,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )


def build_matching_from_surface_code_params(
    *,
    distance: int,
    p: float,
    memory_basis: str = "x",
    rounds: int = 3,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> pymatching.Matching:
    """Build PyMatching decoder from the same circuit factory"""
    circuit = make_unrotated_sc_depolarizing_capacity_circuit(
        distance=distance,
        p=p,
        memory_basis=memory_basis,
        rounds=rounds,
    )
    return build_matching_from_circuit(
        circuit,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )

# Decode from syndrome 
def decode_detector_bits_with_matching(
    detector_bits: np.ndarray,
    matching: pymatching.Matching,
    *,
    return_weights: bool = False,
    enable_correlations: bool = False,
):
    """Decode one or many detector syndromes.

    Input:
        detector_bits:
            (num_detectors,) or (shots, num_detectors)

    Output:
        predicted observable flips:
            (1, num_obs) for one shot
            (shots, num_obs) for a batch
    """
    detector_bits = np.asarray(detector_bits, dtype=np.uint8)
    was_1d = detector_bits.ndim == 1

    if was_1d:
        detector_bits = detector_bits[None, :]

    out = matching.decode_batch(
        detector_bits,
        return_weights=return_weights,
        enable_correlations=enable_correlations,
    )

    if return_weights:
        predictions, weights = out
        predictions = np.asarray(predictions, dtype=np.uint8)
        weights = np.asarray(weights, dtype=float)
        return predictions, weights

    return np.asarray(out, dtype=np.uint8)

def logical_failure_from_observable_flips(
    actual_observable_flips: np.ndarray,
    predicted_observable_flips: np.ndarray,
) -> np.ndarray:
    """Shot-wise logical failure event from residual observable flips."""
    actual = np.asarray(actual_observable_flips, dtype=np.uint8)
    predicted = np.asarray(predicted_observable_flips, dtype=np.uint8)

    if actual.ndim == 1:
        actual = actual[None, :]
    if predicted.ndim == 1:
        predicted = predicted[None, :]

    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs predicted {predicted.shape}"
        )

    residual = np.bitwise_xor(actual, predicted)
    return np.any(residual != 0, axis=1).astype(np.uint8)


def decode_stim_surface_sample_with_mwpm(
    sample: StimSurfaceSample,
    *,
    matching: Optional[pymatching.Matching] = None,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMShotResult:
    """Decode the exact same single-shot sample used by the PEPS decoder."""
    if matching is None:
        matching = build_matching_from_stim_surface_sample(
            sample,
            decompose_errors=decompose_errors,
            enable_correlations=enable_correlations,
            **dem_kwargs,
        )

    predicted = decode_detector_bits_with_matching(
        sample.detector_bits,
        matching,
        enable_correlations=enable_correlations,
    )[0]

    return SurfaceCodeMWPMShotResult(
        sample=sample,
        predicted_observable_flips=predicted.astype(np.uint8),
    )


def decode_stim_surface_batch_with_mwpm(
    batch: StimSurfaceBatchSample,
    *,
    matching: Optional[pymatching.Matching] = None,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    return_weights: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMBatchResult:
    """Decode the exact same batched samples used by the PEPS decoder."""
    if matching is None:
        matching = build_matching_from_stim_surface_batch(
            batch,
            decompose_errors=decompose_errors,
            enable_correlations=enable_correlations,
            **dem_kwargs,
        )

    out = decode_detector_bits_with_matching(
        batch.detector_bits,
        matching,
        return_weights=return_weights,
        enable_correlations=enable_correlations,
    )
    if return_weights:
        predicted_obs, _weights = out
    else:
        predicted_obs = out

    failures = logical_failure_from_observable_flips(
        batch.observable_flips,
        predicted_obs,
    )

    return SurfaceCodeMWPMBatchResult(
        circuit=batch.circuit,
        detector_bits=batch.detector_bits.astype(np.uint8, copy=False),
        actual_observable_flips=batch.observable_flips.astype(np.uint8, copy=False),
        predicted_observable_flips=np.asarray(predicted_obs, dtype=np.uint8),
        logical_failures=failures,
    )


def run_surface_code_mwpm_batch(
    *,
    distance: int,
    p: float,
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    return_weights: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMBatchResult:
    """Generate a fresh batch using src.stim_sampler and decode it with MWPM."""
    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )
    return decode_stim_surface_batch_with_mwpm(
        batch,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        return_weights=return_weights,
        **dem_kwargs,
    )


