from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .stim_sampler import (
    sample_surface_code_depolarizing,
    sample_surface_code_depolarizing_batch,
    sample_surface_code_local_depolarizing_batch,
    generate_local_p_map,
    sample_surface_code_spin_qubit_batch,
    generate_spin_qubit_pz_map,
    sample_surface_code_eo_qubit_batch,
    EOQubitPMaps,
    generate_eo_qubit_p_maps,
)
from .weights_PEPS import (
    depolarizing_weights,
    local_depolarizing_weights,
    spin_qubit_weights,
    local_spin_qubit_weights,
    eo_qubit_weights,
    local_eo_qubit_weights,
)
from .PEPS_Pauli_decoder import pauli_coset_likelihoods_peps, most_likely_coset


def sample_and_decode_surface_code_depolarizing(
    distance: int = 5,
    p: float = 1e-3,
    memory_basis: str = "x",
    Nkeep: int = 128,
    Nsweep: int = 1,
):
    """
    Sample one syndrome from Stim and decode it with the masked PEPS ML decoder.
    """
    sample = sample_surface_code_depolarizing(
        distance=distance,
        p=p,
        memory_basis=memory_basis,
    )

    nrow, ncol = sample.sX.shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p)

    cosets = pauli_coset_likelihoods_peps(
        sX=sample.sX,
        sZ=sample.sZ,
        active_X=sample.active_X,
        active_Z=sample.active_Z,
        W_h=W_h,
        W_v=W_v,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
    )

    return {
        "sample": sample,
        "cosets": cosets,
        "ml_coset": most_likely_coset(cosets),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

@dataclass
class PEPSBatchResult:
    """Result of decoding a batch of shots with the PEPS ML decoder."""

    actual_observable_flips: np.ndarray     # (shots, 1)
    predicted_observable_flips: np.ndarray  # (shots, 1)  — relevant logical component
    logical_failures: np.ndarray            # (shots,)

    @property
    def num_shots(self) -> int:
        return int(self.logical_failures.shape[0])

    @property
    def num_failures(self) -> int:
        return int(np.sum(self.logical_failures))

    @property
    def logical_error_rate(self) -> float:
        if self.num_shots == 0:
            return 0.0
        return self.num_failures / self.num_shots


def run_surface_code_peps_batch(
    *,
    distance: int,
    p: float,
    shots: int,
    memory_basis: str = "x",
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 3,
    target_t: int = 1,
    verbose: bool = False,
) -> PEPSBatchResult:
    """
    Sample a batch of syndromes from Stim and decode each shot with the PEPS ML decoder.

    Logical observable mapping:
      memory_x  →  Stim observable = Z-type logical flip  →  coset index 1 (lz)
      memory_z  →  Stim observable = X-type logical flip  →  coset index 0 (lx)

    W_h and W_v are built once from p and reused for all shots.
    """
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")

    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )

    nrow, ncol = batch.syndrome_shape
    W_h, W_v = depolarizing_weights(nrow, ncol, p)

    # memory_x: Stim observable tracks lz → coset tuple index 1
    # memory_z: Stim observable tracks lx → coset tuple index 0
    logical_idx = 1 if memory_basis == "x" else 0

    predicted_obs = np.zeros((shots, 1), dtype=np.uint8)

    for i, shot in enumerate(batch.iter_shots()):
        if verbose and i % max(1, shots // 10) == 0:
            print(f'    shot {i}/{shots}')

        cosets = pauli_coset_likelihoods_peps(
            sX=shot.sX,
            sZ=shot.sZ,
            active_X=shot.active_X,
            active_Z=shot.active_Z,
            W_h=W_h,
            W_v=W_v,
            Nkeep=Nkeep,
            Nsweep=Nsweep,
        )
        ml_coset, _ = most_likely_coset(cosets)
        predicted_obs[i, 0] = ml_coset[logical_idx]

    actual_obs = batch.observable_flips.astype(np.uint8)  # (shots, 1)
    failures = np.any(predicted_obs != actual_obs, axis=1).astype(np.uint8)

    return PEPSBatchResult(
        actual_observable_flips=actual_obs,
        predicted_observable_flips=predicted_obs,
        logical_failures=failures,
    )


@dataclass
class PEPSFullLogicalResult:
    """
    Full logical error rate combining both X and Z logical observables.

    memory_x experiment → Z-type logical error rate (p_L_Z)
    memory_z experiment → X-type logical error rate (p_L_X)

    Under code-capacity depolarizing noise the two channels are independent:
        p_L_full = 1 - (1 - p_L_X) * (1 - p_L_Z)
    """

    result_x_basis: PEPSBatchResult   # memory_x → lz prediction
    result_z_basis: PEPSBatchResult   # memory_z → lx prediction

    @property
    def p_L_Z(self) -> float:
        return self.result_x_basis.logical_error_rate

    @property
    def p_L_X(self) -> float:
        return self.result_z_basis.logical_error_rate

    @property
    def logical_error_rate(self) -> float:
        return 1.0 - (1.0 - self.p_L_X) * (1.0 - self.p_L_Z)


def run_surface_code_peps_full_logical(
    *,
    distance: int,
    p: float,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    # rounds: int = 3,
    target_t: int = 1,
    verbose: bool = False,
) -> PEPSFullLogicalResult:
    """
    Compute full logical error rate by running the PEPS ML decoder in both memory bases.

    Mirrors run_surface_code_mwpm_full_logical for direct comparison.
    """
    if verbose:
        print('  memory_x (lz):')
    result_x = run_surface_code_peps_batch(
        distance=distance, p=p, shots=shots,
        memory_basis="x", Nkeep=Nkeep, Nsweep=Nsweep,
        rounds=rounds, target_t=target_t, verbose=verbose,
    )
    if verbose:
        print('  memory_z (lx):')
    result_z = run_surface_code_peps_batch(
        distance=distance, p=p, shots=shots,
        memory_basis="z", Nkeep=Nkeep, Nsweep=Nsweep,
        rounds=rounds, target_t=target_t, verbose=verbose,
    )
    return PEPSFullLogicalResult(
        result_x_basis=result_x,
        result_z_basis=result_z,
    )


# ---------------------------------------------------------------------------
# Local-noise batch runner
# ---------------------------------------------------------------------------

def run_surface_code_peps_batch_local(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    shots: int,
    memory_basis: str = "x",
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSBatchResult:
    """
    Sample a batch with spatially inhomogeneous p_map and decode with PEPS ML.

    W_h is built site-by-site from p_map so the PEPS uses the correct local
    error weights for each data qubit.
    """
    if memory_basis not in ("x", "z"):
        raise ValueError("memory_basis must be 'x' or 'z'.")

    batch = sample_surface_code_local_depolarizing_batch(
        distance=distance,
        p_map=p_map,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
    )

    nrow, ncol = batch.syndrome_shape
    W_h, W_v = local_depolarizing_weights(nrow, ncol, p_map, p_fallback=p_fallback)

    logical_idx = 1 if memory_basis == "x" else 0
    predicted_obs = np.zeros((shots, 1), dtype=np.uint8)

    for i, shot in enumerate(batch.iter_shots()):
        if verbose and i % max(1, shots // 10) == 0:
            print(f'    shot {i}/{shots}')

        cosets = pauli_coset_likelihoods_peps(
            sX=shot.sX,
            sZ=shot.sZ,
            active_X=shot.active_X,
            active_Z=shot.active_Z,
            W_h=W_h,
            W_v=W_v,
            Nkeep=Nkeep,
            Nsweep=Nsweep,
        )
        ml_coset, _ = most_likely_coset(cosets)
        predicted_obs[i, 0] = ml_coset[logical_idx]

    actual_obs = batch.observable_flips.astype(np.uint8)
    failures = np.any(predicted_obs != actual_obs, axis=1).astype(np.uint8)

    return PEPSBatchResult(
        actual_observable_flips=actual_obs,
        predicted_observable_flips=predicted_obs,
        logical_failures=failures,
    )


@dataclass
class PEPSLocalFullLogicalResult:
    """Full logical error rate under a shared spatially inhomogeneous p_map."""

    result_x_basis: PEPSBatchResult
    result_z_basis: PEPSBatchResult
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


def run_surface_code_peps_full_logical_local(
    *,
    distance: int,
    p_map: Dict[Tuple[int, int], float],
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSLocalFullLogicalResult:
    """Run PEPS full logical decoding under a fixed spatially inhomogeneous p_map."""
    if verbose:
        print('  memory_x (lz):')
    result_x = run_surface_code_peps_batch_local(
        distance=distance, p_map=p_map, shots=shots,
        memory_basis="x", Nkeep=Nkeep, Nsweep=Nsweep,
        rounds=rounds, p_fallback=p_fallback,
        reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )
    if verbose:
        print('  memory_z (lx):')
    result_z = run_surface_code_peps_batch_local(
        distance=distance, p_map=p_map, shots=shots,
        memory_basis="z", Nkeep=Nkeep, Nsweep=Nsweep,
        rounds=rounds, p_fallback=p_fallback,
        reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )
    return PEPSLocalFullLogicalResult(
        result_x_basis=result_x,
        result_z_basis=result_z,
        p_map=p_map,
    )


# ---------------------------------------------------------------------------
# Spin qubit — Z dephasing PEPS runners
# ---------------------------------------------------------------------------

def run_surface_code_peps_spin_qubit(
    *,
    distance: int,
    pz_map: Dict[Tuple[int, int], float],
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSBatchResult:
    """
    Spin qubit PEPS decoder: Z dephasing only, memory_x basis.
    Samples with Z_ERROR(pz_i) and decodes with local_spin_qubit_weights.
    """
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
    nrow, ncol = batch.syndrome_shape
    W_h, W_v = local_spin_qubit_weights(nrow, ncol, pz_map, p_fallback=p_fallback)

    predicted_obs = np.zeros((shots, 1), dtype=np.uint8)
    for i, shot in enumerate(batch.iter_shots()):
        if verbose and i % max(1, shots // 10) == 0:
            print(f'    shot {i}/{shots}')
        cosets = pauli_coset_likelihoods_peps(
            sX=shot.sX, sZ=shot.sZ,
            active_X=shot.active_X, active_Z=shot.active_Z,
            W_h=W_h, W_v=W_v, Nkeep=Nkeep, Nsweep=Nsweep,
        )
        ml_coset, _ = most_likely_coset(cosets)
        predicted_obs[i, 0] = ml_coset[1]  # memory_x -> Z-type logical

    actual_obs = batch.observable_flips.astype(np.uint8)
    failures = np.any(predicted_obs != actual_obs, axis=1).astype(np.uint8)
    return PEPSBatchResult(
        actual_observable_flips=actual_obs,
        predicted_observable_flips=predicted_obs,
        logical_failures=failures,
    )


def run_surface_code_peps_spin_qubit_from_normal(
    *,
    distance: int,
    p_mean: float,
    sigma_frac: float,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    seed: Optional[int] = None,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSBatchResult:
    """Draw a Gaussian pz_map and run spin qubit PEPS decoder."""
    pz_map = generate_spin_qubit_pz_map(
        distance=distance, p_mean=p_mean, sigma_frac=sigma_frac,
        memory_basis="x", rounds=rounds, seed=seed, reference_p=reference_p,
    )
    return run_surface_code_peps_spin_qubit(
        distance=distance, pz_map=pz_map, shots=shots,
        Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        p_fallback=p_fallback, reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )


def run_surface_code_peps_spin_qubit_uniform(
    *,
    distance: int,
    p: float,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    verbose: bool = False,
) -> PEPSBatchResult:
    """Uniform Z dephasing baseline: all qubits at rate p."""
    pz_map = generate_spin_qubit_pz_map(
        distance=distance, p_mean=p, sigma_frac=0.0,
        memory_basis="x", rounds=rounds,
    )
    return run_surface_code_peps_spin_qubit(
        distance=distance, pz_map=pz_map, shots=shots,
        Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds, verbose=verbose,
    )


# ---------------------------------------------------------------------------
# EO qubit — PAULI_CHANNEL_1(px, 0, pz) PEPS runners
# ---------------------------------------------------------------------------

@dataclass
class PEPSEOQubitResult:
    """Full logical PEPS result for EO qubit (biased X+Z noise)."""
    result_x_basis: PEPSBatchResult   # memory_x -> Z logical
    result_z_basis: PEPSBatchResult   # memory_z -> X logical
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


def _run_peps_eo_qubit_basis(
    *,
    distance: int,
    px_map: Dict[Tuple[int, int], float],
    pz_map: Dict[Tuple[int, int], float],
    shots: int,
    memory_basis: str,
    Nkeep: int,
    Nsweep: int,
    rounds: int,
    p_fallback_x: float,
    p_fallback_z: float,
    reference_p: float,
    clip_eps: float,
    verbose: bool,
) -> PEPSBatchResult:
    batch = sample_surface_code_eo_qubit_batch(
        distance=distance, px_map=px_map, pz_map=pz_map,
        shots=shots, memory_basis=memory_basis, rounds=rounds,
        p_fallback_x=p_fallback_x, p_fallback_z=p_fallback_z,
        reference_p=reference_p, clip_eps=clip_eps,
    )
    nrow, ncol = batch.syndrome_shape
    W_h, W_v = local_eo_qubit_weights(
        nrow, ncol, px_map, pz_map,
        p_fallback_x=p_fallback_x, p_fallback_z=p_fallback_z,
    )
    logical_idx = 1 if memory_basis == "x" else 0
    predicted_obs = np.zeros((shots, 1), dtype=np.uint8)
    for i, shot in enumerate(batch.iter_shots()):
        if verbose and i % max(1, shots // 10) == 0:
            print(f'    shot {i}/{shots}')
        cosets = pauli_coset_likelihoods_peps(
            sX=shot.sX, sZ=shot.sZ,
            active_X=shot.active_X, active_Z=shot.active_Z,
            W_h=W_h, W_v=W_v, Nkeep=Nkeep, Nsweep=Nsweep,
        )
        ml_coset, _ = most_likely_coset(cosets)
        predicted_obs[i, 0] = ml_coset[logical_idx]
    actual_obs = batch.observable_flips.astype(np.uint8)
    failures = np.any(predicted_obs != actual_obs, axis=1).astype(np.uint8)
    return PEPSBatchResult(
        actual_observable_flips=actual_obs,
        predicted_observable_flips=predicted_obs,
        logical_failures=failures,
    )


def run_surface_code_peps_eo_qubit(
    *,
    distance: int,
    p_maps: EOQubitPMaps,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSEOQubitResult:
    """EO qubit PEPS decoder: PAULI_CHANNEL_1(px, 0, pz), runs memory_x and memory_z."""
    if verbose:
        print('  memory_x (lz):')
    result_x = _run_peps_eo_qubit_basis(
        distance=distance, px_map=p_maps.px_map, pz_map=p_maps.pz_map,
        shots=shots, memory_basis="x", Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        p_fallback_x=p_fallback_x, p_fallback_z=p_fallback_z,
        reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )
    if verbose:
        print('  memory_z (lx):')
    result_z = _run_peps_eo_qubit_basis(
        distance=distance, px_map=p_maps.px_map, pz_map=p_maps.pz_map,
        shots=shots, memory_basis="z", Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        p_fallback_x=p_fallback_x, p_fallback_z=p_fallback_z,
        reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )
    return PEPSEOQubitResult(result_x_basis=result_x, result_z_basis=result_z, p_maps=p_maps)


def run_surface_code_peps_eo_qubit_from_normal(
    *,
    distance: int,
    p_mean_z: float,
    sigma_frac_z: float,
    p_mean_n: float,
    sigma_frac_n: float,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    seed: Optional[int] = None,
    p_fallback_x: float = 0.0,
    p_fallback_z: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSEOQubitResult:
    """Draw independent Gaussian p_maps and run EO qubit PEPS decoder."""
    p_maps = generate_eo_qubit_p_maps(
        distance=distance, p_mean_z=p_mean_z, sigma_frac_z=sigma_frac_z,
        p_mean_n=p_mean_n, sigma_frac_n=sigma_frac_n,
        memory_basis="x", rounds=rounds, seed=seed, reference_p=reference_p,
    )
    return run_surface_code_peps_eo_qubit(
        distance=distance, p_maps=p_maps, shots=shots,
        Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        p_fallback_x=p_fallback_x, p_fallback_z=p_fallback_z,
        reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )


def run_surface_code_peps_eo_qubit_uniform(
    *,
    distance: int,
    p_mean_z: float,
    p_mean_n: float,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSEOQubitResult:
    """Uniform EO qubit baseline: all qubits at axis rates (p_mean_z, p_mean_n)."""
    return run_surface_code_peps_eo_qubit_from_normal(
        distance=distance,
        p_mean_z=p_mean_z, sigma_frac_z=0.0,
        p_mean_n=p_mean_n, sigma_frac_n=0.0,
        shots=shots, Nkeep=Nkeep, Nsweep=Nsweep, rounds=rounds,
        reference_p=reference_p, clip_eps=clip_eps, verbose=verbose,
    )


def run_surface_code_peps_full_logical_local_from_normal(
    *,
    distance: int,
    p_mean: float,
    sigma_frac: float,
    shots: int,
    Nkeep: int = 32,
    Nsweep: int = 1,
    rounds: int = 1,
    seed: Optional[int] = None,
    p_fallback: float = 0.0,
    reference_p: float = 1e-3,
    clip_eps: float = 1e-12,
    verbose: bool = False,
) -> PEPSLocalFullLogicalResult:
    """Draw one shared Gaussian p_map and run full logical PEPS decoding."""
    p_map = generate_local_p_map(
        distance=distance,
        p_mean=p_mean,
        sigma_frac=sigma_frac,
        memory_basis="x",
        rounds=rounds,
        seed=seed,
        reference_p=reference_p,
    )
    return run_surface_code_peps_full_logical_local(
        distance=distance,
        p_map=p_map,
        shots=shots,
        Nkeep=Nkeep,
        Nsweep=Nsweep,
        rounds=rounds,
        p_fallback=p_fallback,
        reference_p=reference_p,
        clip_eps=clip_eps,
        verbose=verbose,
    )
