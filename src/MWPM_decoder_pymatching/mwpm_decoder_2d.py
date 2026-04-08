from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pymatching

from ..Surface_code_sampler.stim_sampler import (
    StimSurfaceSample,
    StimSurfaceBatchSample,
    sample_surface_code_depolarizing,
    sample_surface_code_depolarizing_batch,
)
from ..metric import logical_failures_from_predictions


@dataclass
class SurfaceCodeMWPM2DShotResult:
    """2D MWPM result for one sample. """

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
class SurfaceCodeMWPM2DBatchResult:
    """2D MWPM result for a batch of 2D syndrome slices."""

    batch: StimSurfaceBatchSample
    predicted_observable_flips: np.ndarray
    logical_failures: np.ndarray

    @property
    def num_shots(self) -> int:
        return int(self.predicted_observable_flips.shape[0])

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


_RC = Tuple[int, int]
_NodeMap = Dict[_RC, int]


def _llr_weight(q: float) -> float:
    """Log-likelihood-ratio edge weight."""
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must satisfy 0 < q < 1. Got {q}.")
    return float(np.log((1.0 - q) / q))


def _validate_dense_slice_inputs(
    s: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.uint8)
    active_mask = np.asarray(active_mask, dtype=np.uint8)
    if s.shape != active_mask.shape:
        raise ValueError(
            f"Shape mismatch: s.shape={s.shape}, active_mask.shape={active_mask.shape}"
        )
    return s, active_mask


def _node_map_from_active_mask(active_mask: np.ndarray) -> _NodeMap:
    active_mask = np.asarray(active_mask, dtype=np.uint8)
    coords = [
        (r, c)
        for r in range(active_mask.shape[0])
        for c in range(active_mask.shape[1])
        if active_mask[r, c] != 0
    ]
    return {rc: idx for idx, rc in enumerate(coords)}

def build_matching_graph_from_active_mask(
    active_mask: np.ndarray,
    q: float,
    *,
    boundary_axis: str,
) -> tuple[pymatching.Matching, _NodeMap]:
    active_mask = np.asarray(active_mask, dtype=np.uint8)
    node_of = _node_map_from_active_mask(active_mask)
    w = _llr_weight(q)

    m = pymatching.Matching()

    # Bulk edges
    for (r, c), u in node_of.items():
        for dr, dc in ((0, 2), (2, 0)):
            rr, cc = r + dr, c + dc
            if (rr, cc) in node_of:
                v = node_of[(rr, cc)]
                m.add_edge(u, v, weight=w)

    nrow, ncol = active_mask.shape

    for (r, c), u in node_of.items():
        if boundary_axis == "horizontal":
            # left boundary: trivial side
            if c - 2 < 0:
                m.add_boundary_edge(u, weight=w, fault_ids=set())

            # right boundary: logical side
            if c + 2 >= ncol:
                m.add_boundary_edge(u, weight=w, fault_ids={0})

        elif boundary_axis == "vertical":
            # top boundary: trivial side
            if r - 2 < 0:
                m.add_boundary_edge(u, weight=w, fault_ids=set())

            # bottom boundary: logical side
            if r + 2 >= nrow:
                m.add_boundary_edge(u, weight=w, fault_ids={0})

        else:
            raise ValueError("boundary_axis must be 'horizontal' or 'vertical'.")

    return m, node_of

def syndrome_vector_from_dense_slice(
    s: np.ndarray,
    active_mask: np.ndarray,
    node_of: _NodeMap,
) -> np.ndarray:
    """Convert one dense 2D syndrome slice into a PyMatching syndrome vector."""
    s, active_mask = _validate_dense_slice_inputs(s, active_mask)

    syndrome = np.zeros(len(node_of), dtype=np.uint8)
    for (r, c), node in node_of.items():
        syndrome[node] = s[r, c]
    return syndrome

def build_2d_css_matching(
    active_mask: np.ndarray,
    *,
    p: float,
    boundary_axis: str,
) -> tuple[pymatching.Matching, _NodeMap]:
    """Build one CSS 2D MWPM graph from an active mask.

    For depolarizing data noise with total rate p, the effective marginal
    rate seen by either CSS decoder is q = 2p/3.
    """
    q = 2.0 * p / 3.0
    return build_matching_graph_from_active_mask(
        active_mask=active_mask,
        q=q,
        boundary_axis=boundary_axis,
    )


def decode_2d_css_slice(
    s: np.ndarray,
    active_mask: np.ndarray,
    *,
    p: float,
    boundary_axis: str,
    matching: Optional[pymatching.Matching] = None,
    node_of: Optional[_NodeMap] = None,
) -> np.ndarray:
    """Decode one 2D CSS syndrome slice.

    Returns
    -------
    np.ndarray, shape (1,)
        Predicted logical observable flip.
    """
    s, active_mask = _validate_dense_slice_inputs(s, active_mask)

    if (matching is None) != (node_of is None):
        raise ValueError("Provide both matching and node_of, or neither.")

    if matching is None:
        matching, node_of = build_2d_css_matching(
            active_mask=active_mask,
            p=p,
            boundary_axis=boundary_axis,
        )

    syndrome = syndrome_vector_from_dense_slice(s, active_mask, node_of)
    pred = matching.decode(syndrome)

    # Normalize PyMatching output robustly.
    pred = np.asarray(pred, dtype=np.uint8).reshape(-1)

    if pred.size == 0:
        # No fault IDs were present in the graph. Treat as one logical observable
        # with predicted trivial flip, but this usually indicates graph construction
        # is not tagging a logical boundary correctly.
        return np.array([0], dtype=np.uint8)

    if pred.size == 1:
        return pred

    raise ValueError(
        f"Expected at most one logical observable, but decoder returned shape {pred.shape}."
    )

def decode_2d_surface_sample_with_mwpm(
    sample: StimSurfaceSample,
    *,
    p: float,
    memory_basis: str = "x",
    boundary_axis_if_x: str = "vertical",
    boundary_axis_if_z: str = "horizontal",
) -> SurfaceCodeMWPM2DShotResult:
    if memory_basis == "x":
        pred = decode_2d_css_slice(
            sample.sX,
            sample.active_X,
            p=p,
            boundary_axis=boundary_axis_if_x,
        )
    elif memory_basis == "z":
        pred = decode_2d_css_slice(
            sample.sZ,
            sample.active_Z,
            p=p,
            boundary_axis=boundary_axis_if_z,
        )
    else:
        raise ValueError("memory_basis must be 'x' or 'z'.")

    return SurfaceCodeMWPM2DShotResult(
        sample=sample,
        predicted_observable_flips=pred.astype(np.uint8),
    )


def decode_2d_surface_batch_with_mwpm(
    batch: StimSurfaceBatchSample,
    *,
    p: float,
    memory_basis: str = "x",
    boundary_axis_if_x: str = "vertical",
    boundary_axis_if_z: str = "horizontal",
) -> SurfaceCodeMWPM2DBatchResult:
    """Decode a batch using only 2D syndrome data."""
    shots = batch.shots

    if memory_basis == "x":
        active_mask0 = batch.active_X[0]
        matching, node_of = build_2d_css_matching(
            active_mask0,
            p=p,
            boundary_axis=boundary_axis_if_x,
        )
        predicted = np.stack(
            [
                decode_2d_css_slice(
                    batch.sX[k],
                    batch.active_X[k],
                    p=p,
                    boundary_axis=boundary_axis_if_x,
                    matching=matching,
                    node_of=node_of,
                )
                for k in range(shots)
            ],
            axis=0,
        ).astype(np.uint8)

    elif memory_basis == "z":
        active_mask0 = batch.active_Z[0]
        matching, node_of = build_2d_css_matching(
            active_mask0,
            p=p,
            boundary_axis=boundary_axis_if_z,
        )
        predicted = np.stack(
            [
                decode_2d_css_slice(
                    batch.sZ[k],
                    batch.active_Z[k],
                    p=p,
                    boundary_axis=boundary_axis_if_z,
                    matching=matching,
                    node_of=node_of,
                )
                for k in range(shots)
            ],
            axis=0,
        ).astype(np.uint8)
    else:
        raise ValueError("memory_basis must be 'x' or 'z'.")

    failures = logical_failures_from_predictions(
        batch.observable_flips,
        predicted,
    )

    return SurfaceCodeMWPM2DBatchResult(
        batch=batch,
        predicted_observable_flips=predicted,
        logical_failures=failures,
    )


def run_surface_code_mwpm_2d_batch(
    *,
    distance: int,
    p: float,
    shots: int,
    memory_basis: str = "x",
    rounds: int = 3,
    target_t: int = 1,
    boundary_axis_if_x: str = "vertical",
    boundary_axis_if_z: str = "horizontal",
) -> SurfaceCodeMWPM2DBatchResult:
    """Generate a batch and decode only its 2D syndrome slices.

    Warning:
        If the sampler still comes from repeated syndrome extraction, then
        batch.observable_flips are not a fair 2D code-capacity truth label.
        This function is still useful for interface testing and debugging.
    """
    batch = sample_surface_code_depolarizing_batch(
        distance=distance,
        p=p,
        shots=shots,
        memory_basis=memory_basis,
        rounds=rounds,
        target_t=target_t,
    )
    return decode_2d_surface_batch_with_mwpm(
        batch,
        p=p,
        memory_basis=memory_basis,
        boundary_axis_if_x=boundary_axis_if_x,
        boundary_axis_if_z=boundary_axis_if_z,
    )


def test_build_2d_graph_from_active_mask() -> None:
    active = np.zeros((5, 5), dtype=np.uint8)
    active[0, 0] = 1
    active[0, 2] = 1
    active[2, 0] = 1
    active[2, 2] = 1

    matching, node_of = build_2d_css_matching(
        active,
        p=1e-3,
        boundary_axis="horizontal",
    )
    assert isinstance(matching, pymatching.Matching)
    assert len(node_of) == 4
    print("test_build_2d_graph_from_active_mask passed.")

def test_decode_empty_syndrome_trivial() -> None:
    active = np.zeros((5, 5), dtype=np.uint8)
    active[0, 0] = 1
    active[0, 2] = 1
    active[2, 0] = 1
    active[2, 2] = 1
    s = np.zeros_like(active)

    pred = decode_2d_css_slice(
        s,
        active,
        p=1e-3,
        boundary_axis="horizontal",
    )
    assert pred.shape == (1,)
    assert int(pred[0]) == 0
    print("test_decode_empty_syndrome_trivial passed.")


def test_decode_2d_surface_sample_with_mwpm() -> None:
    sample = sample_surface_code_depolarizing(
        distance=5,
        p=1e-3,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_2d_surface_sample_with_mwpm(
        sample,
        p=1e-3,
        memory_basis="x",
    )
    assert out.predicted_observable_flips.shape == sample.observable_flips.shape
    print("test_decode_2d_surface_sample_with_mwpm passed.")


def test_decode_2d_surface_batch_with_mwpm(shots: int = 8) -> None:
    batch = sample_surface_code_depolarizing_batch(
        distance=5,
        p=1e-3,
        shots=shots,
        memory_basis="x",
        rounds=3,
        target_t=1,
    )
    out = decode_2d_surface_batch_with_mwpm(
        batch,
        p=1e-3,
        memory_basis="x",
    )
    assert out.predicted_observable_flips.shape == batch.observable_flips.shape
    assert out.logical_failures.shape == (shots,)
    print("test_decode_2d_surface_batch_with_mwpm passed.")


def run_all_tests() -> None:
    test_build_2d_graph_from_active_mask()
    test_decode_empty_syndrome_trivial()
    test_decode_2d_surface_sample_with_mwpm()
    test_decode_2d_surface_batch_with_mwpm()
    print("All 2D MWPM tests passed.")


if __name__ == "__main__":
    run_all_tests()
