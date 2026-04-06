from dataclasses import dataclass

import numpy as np

from .stim_sampler import (
    sample_surface_code_depolarizing,
    sample_surface_code_depolarizing_batch,
)
from .weights_PEPS import depolarizing_weights
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
    rounds: int = 3,
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
