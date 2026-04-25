from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import stim
import pymatching

from .stim_sampler import (
    StimSurfaceSample,
    StimSurfaceBatchSample,
    make_unrotated_sc_depolarizing_capacity_circuit,
    sample_surface_code_depolarizing,
    sample_surface_code_depolarizing_batch,
    generate_local_p_map,
    sample_surface_code_local_depolarizing_batch,
    generate_spin_qubit_pz_map,
    sample_surface_code_spin_qubit_batch,
    EOQubitPMaps,
    generate_eo_qubit_p_maps,
    sample_surface_code_eo_qubit_batch,
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


@dataclass
class SurfaceCodeMWPMFullLogicalResult:
    """
    Full logical error rate combining both X and Z logical observables.

    Under code-capacity Pauli noise the X and Z error channels are independent:
      - memory_x experiment measures Z-type logical failures  (p_L_Z)
      - memory_z experiment measures X-type logical failures  (p_L_X)

    The combined full logical failure probability is:
        p_L_full = 1 - (1 - p_L_X) * (1 - p_L_Z)
    """

    result_x_basis: SurfaceCodeMWPMBatchResult   # memory_x → lz prediction
    result_z_basis: SurfaceCodeMWPMBatchResult   # memory_z → lx prediction

    @property
    def p_L_Z(self) -> float:
        """Z-type logical error rate (from memory_x experiment)."""
        return self.result_x_basis.logical_error_rate

    @property
    def p_L_X(self) -> float:
        """X-type logical error rate (from memory_z experiment)."""
        return self.result_z_basis.logical_error_rate

    @property
    def logical_error_rate(self) -> float:
        """Full logical error rate (either X or Z logical error occurs)."""
        return 1.0 - (1.0 - self.p_L_X) * (1.0 - self.p_L_Z)


def run_surface_code_mwpm_full_logical(
    *,
    distance: int,
    p: float,
    shots: int,
    rounds: int = 3,
    target_t: int = 1,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMFullLogicalResult:
    """
    Compute full logical error rate by running MWPM in both memory bases.

    memory_x → Z-type logical error rate (p_L_Z)
    memory_z → X-type logical error rate (p_L_X)
    Full logical failure = failure in either channel.

    Under depolarizing noise, X and Z channels are decoupled, so running
    the two bases independently gives the correct combined rate.
    """
    result_x = run_surface_code_mwpm_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis="x",
        rounds=rounds,
        target_t=target_t,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )
    result_z = run_surface_code_mwpm_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis="z",
        rounds=rounds,
        target_t=target_t,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )
    return SurfaceCodeMWPMFullLogicalResult(
        result_x_basis=result_x,
        result_z_basis=result_z,
    )


@dataclass
class SurfaceCodeMWPMLocalFullLogicalResult:
    """
    Full logical MWPM result under a shared spatially inhomogeneous p-map.
    """

    result_x_basis: SurfaceCodeMWPMBatchResult
    result_z_basis: SurfaceCodeMWPMBatchResult
    p_map: Dict[Tuple[int, int], float]

    @property
    def p_L_Z(self) -> float:
        return self.result_x_basis.logical_error_rate

    @property
    def p_L_X(self) -> float:
        return self.result_z_basis.logical_error_rate

    @property
    def logical_error_rate(self) -> float:
        return 1.0 - (1.0 - self.p_L_X) * (1.0 - self.p_L_Z)


def run_surface_code_mwpm_batch_local(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    return_weights: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMBatchResult:
    """
    Generate a fresh local-noise batch and decode it with DEM-based MWPM.
    """
    batch = sample_surface_code_local_depolarizing_batch(
        distance=distance,
        p_map=p_map,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    return decode_stim_surface_batch_with_mwpm(
        batch,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        return_weights=return_weights,
        **dem_kwargs,
    )


def run_surface_code_mwpm_full_logical_local(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    shots: int,
    rounds: int = 3,
    target_t: int = 1,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMLocalFullLogicalResult:
    """
    Full logical MWPM using the same physical p-map in memory_x and memory_z.
    """
    result_x = run_surface_code_mwpm_batch_local(
        distance=distance,
        p_map=p_map,
        shots=shots,
        memory_basis="x",
        rounds=rounds,
        target_t=target_t,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )
    result_z = run_surface_code_mwpm_batch_local(
        distance=distance,
        p_map=p_map,
        shots=shots,
        memory_basis="z",
        rounds=rounds,
        target_t=target_t,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )
    return SurfaceCodeMWPMLocalFullLogicalResult(
        result_x_basis=result_x,
        result_z_basis=result_z,
        p_map=p_map,
    )


# ---------------------------------------------------------------------------
# Spin qubit — Z dephasing MWPM runners
# ---------------------------------------------------------------------------

@dataclass
class SurfaceCodeMWPMSpinQubitResult:
    """
    MWPM result for spin qubit (Z dephasing only).
    Only memory_x is meaningful; Z errors cause Z-logical failures only.
    """
    result_x_basis: SurfaceCodeMWPMBatchResult
    pz_map: Dict[Tuple[int, int], float]

    @property
    def logical_error_rate(self) -> float:
        return self.result_x_basis.logical_error_rate


def run_surface_code_mwpm_spin_qubit(
    *,
    distance: int,
    pz_map: Dict[Tuple[int, int], float],
    shots: int,
    rounds: int = 3,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMSpinQubitResult:
    """MWPM for spin qubit noise (Z dephasing). Runs memory_x only."""
    batch = sample_surface_code_spin_qubit_batch(
        distance=distance,
        pz_map=pz_map,
        shots=shots,
        memory_basis="x",
        rounds=rounds,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    result_x = decode_stim_surface_batch_with_mwpm(
        batch,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )
    return SurfaceCodeMWPMSpinQubitResult(result_x_basis=result_x, pz_map=pz_map)


def run_surface_code_mwpm_spin_qubit_from_normal(
    *,
    distance: int,
    p_mean: float,
    sigma_frac: float,
    shots: int,
    rounds: int = 3,
    seed: Optional[int] = None,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMSpinQubitResult:
    """Draw a Gaussian pz_map and run MWPM for spin qubit noise."""
    pz_map = generate_spin_qubit_pz_map(
        distance=distance,
        p_mean=p_mean,
        sigma_frac=sigma_frac,
        memory_basis="x",
        rounds=rounds,
        seed=seed,
        reference_p=reference_p,
    )
    return run_surface_code_mwpm_spin_qubit(
        distance=distance,
        pz_map=pz_map,
        shots=shots,
        rounds=rounds,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )


def run_surface_code_mwpm_spin_qubit_uniform(
    *,
    distance: int,
    p: float,
    shots: int,
    rounds: int = 3,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMSpinQubitResult:
    """Uniform Z dephasing baseline: all qubits at rate p."""
    pz_map = generate_spin_qubit_pz_map(
        distance=distance, p_mean=p, sigma_frac=0.0,
        memory_basis="x", rounds=rounds, reference_p=reference_p,
    )
    return run_surface_code_mwpm_spin_qubit(
        distance=distance, pz_map=pz_map, shots=shots, rounds=rounds,
        p_fallback=0.0, reference_p=reference_p, clip_eps=clip_eps,
        decompose_errors=decompose_errors, enable_correlations=enable_correlations,
        **dem_kwargs,
    )


# ---------------------------------------------------------------------------
# EO qubit — PAULI_CHANNEL_1(px, 0, pz) MWPM runners
# ---------------------------------------------------------------------------

@dataclass
class SurfaceCodeMWPMEOQubitResult:
    """
    Full logical MWPM result for EO qubit (biased X+Z noise).
    memory_x → Z-logical failures; memory_z → X-logical failures.
    """
    result_x_basis: SurfaceCodeMWPMBatchResult
    result_z_basis: SurfaceCodeMWPMBatchResult
    p_maps: EOQubitPMaps

    @property
    def p_L_Z(self) -> float:
        return self.result_x_basis.logical_error_rate

    @property
    def p_L_X(self) -> float:
        return self.result_z_basis.logical_error_rate

    @property
    def logical_error_rate(self) -> float:
        return 1.0 - (1.0 - self.p_L_X) * (1.0 - self.p_L_Z)


def run_surface_code_mwpm_eo_qubit(
    *,
    distance: int,
    p_maps: EOQubitPMaps,
    shots: int,
    rounds: int = 3,
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    approximate_disjoint_errors: bool = True,
    **dem_kwargs,
) -> SurfaceCodeMWPMEOQubitResult:
    """MWPM for EO qubit noise. Runs memory_x and memory_z with shared p_maps."""
    dem_kwargs.setdefault('approximate_disjoint_errors', approximate_disjoint_errors)
    batch_x = sample_surface_code_eo_qubit_batch(
        distance=distance,
        px_map=p_maps.px_map,
        pz_map=p_maps.pz_map,
        shots=shots,
        memory_basis="x",
        rounds=rounds,
        p_fallback_x=p_fallback_x,
        p_fallback_z=p_fallback_z,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    batch_z = sample_surface_code_eo_qubit_batch(
        distance=distance,
        px_map=p_maps.px_map,
        pz_map=p_maps.pz_map,
        shots=shots,
        memory_basis="z",
        rounds=rounds,
        p_fallback_x=p_fallback_x,
        p_fallback_z=p_fallback_z,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )
    result_x = decode_stim_surface_batch_with_mwpm(
        batch_x, decompose_errors=decompose_errors,
        enable_correlations=enable_correlations, **dem_kwargs,
    )
    result_z = decode_stim_surface_batch_with_mwpm(
        batch_z, decompose_errors=decompose_errors,
        enable_correlations=enable_correlations, **dem_kwargs,
    )
    return SurfaceCodeMWPMEOQubitResult(
        result_x_basis=result_x, result_z_basis=result_z, p_maps=p_maps,
    )


def run_surface_code_mwpm_eo_qubit_from_normal(
    *,
    distance: int,
    p_mean_z: float,
    sigma_frac_z: float,
    p_mean_n: float,
    sigma_frac_n: float,
    shots: int,
    rounds: int = 3,
    seed: Optional[int] = None,
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMEOQubitResult:
    """Draw independent Gaussian p_maps and run MWPM for EO qubit noise."""
    p_maps = generate_eo_qubit_p_maps(
        distance=distance,
        p_mean_z=p_mean_z,
        sigma_frac_z=sigma_frac_z,
        p_mean_n=p_mean_n,
        sigma_frac_n=sigma_frac_n,
        memory_basis="x",
        rounds=rounds,
        seed=seed,
        reference_p=reference_p,
    )
    return run_surface_code_mwpm_eo_qubit(
        distance=distance,
        p_maps=p_maps,
        shots=shots,
        rounds=rounds,
        p_fallback_x=p_fallback_x,
        p_fallback_z=p_fallback_z,
        reference_p=reference_p,
        clip_eps=clip_eps,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )


def run_surface_code_mwpm_eo_qubit_uniform(
    *,
    distance: int,
    p_mean_z: float,
    p_mean_n: float,
    shots: int,
    rounds: int = 3,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMEOQubitResult:
    """Uniform EO qubit baseline: all qubits at axis rates (p_mean_z, p_mean_n)."""
    p_maps = generate_eo_qubit_p_maps(
        distance=distance,
        p_mean_z=p_mean_z, sigma_frac_z=0.0,
        p_mean_n=p_mean_n, sigma_frac_n=0.0,
        memory_basis="x", rounds=rounds, reference_p=reference_p,
    )
    return run_surface_code_mwpm_eo_qubit(
        distance=distance, p_maps=p_maps, shots=shots, rounds=rounds,
        reference_p=reference_p, clip_eps=clip_eps,
        decompose_errors=decompose_errors, enable_correlations=enable_correlations,
        **dem_kwargs,
    )


def run_surface_code_mwpm_full_logical_local_from_normal(
    *,
    distance: int,
    p_mean: float,
    sigma_frac: float,
    shots: int,
    rounds: int = 3,
    target_t: int = 1,
    seed: Optional[int] = None,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    decompose_errors: bool = True,
    enable_correlations: bool = False,
    **dem_kwargs,
) -> SurfaceCodeMWPMLocalFullLogicalResult:
    """
    Draw one shared Gaussian p-map and run full logical MWPM.
    """
    p_map = generate_local_p_map(
        distance=distance,
        p_mean=p_mean,
        sigma_frac=sigma_frac,
        memory_basis="x",
        rounds=rounds,
        seed=seed,
        reference_p=reference_p,
    )
    return run_surface_code_mwpm_full_logical_local(
        distance=distance,
        p_map=p_map,
        shots=shots,
        rounds=rounds,
        target_t=target_t,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
        decompose_errors=decompose_errors,
        enable_correlations=enable_correlations,
        **dem_kwargs,
    )


