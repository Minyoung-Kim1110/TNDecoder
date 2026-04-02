from .stim_sampler import sample_surface_code_depolarizing
from .weights_PEPS import *
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

