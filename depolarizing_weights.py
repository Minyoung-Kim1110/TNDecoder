
"""Site-dependent depolarizing-channel utilities.

This file separates three related objects that are often conflated:

1. Full depolarizing probabilities for TN / PEPS local tensors:
       [p_I, p_X, p_Y, p_Z] = [1-p, p/3, p/3, p/3]

2. Effective marginal error probabilities for CSS matching decoders:
       q_X = P(X or Y) = 2p/3  (relevant for Z-check syndromes)
       q_Z = P(Z or Y) = 2p/3  (relevant for X-check syndromes)

3. Matching weights:
       w = log((1-q)/q)

For non-uniform noise, each site can have its own p_i.
"""

from typing import Literal, Sequence, Tuple

import numpy as np



def clip_probabilities(p: np.ndarray | Sequence[float], eps: float = 1e-15) -> np.ndarray:
    """Clip probabilities away from 0 and 1 to keep log-weights finite."""
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def sample_gaussian_site_error_rates(
    mean_p: float,
    delta_p: float,
    size: int | Tuple[int, ...],
    *,
    rng: np.random.Generator | None = None,
    clip_to_unit_interval: bool = True,
) -> np.ndarray:
    """Sample site-wise depolarizing rates p_i ~ N(mean_p, delta_p^2)."""
    if rng is None:
        rng = np.random.default_rng()
    p = rng.normal(loc=mean_p, scale=delta_p, size=size)
    if clip_to_unit_interval:
        p = np.clip(p, 0.0, 1.0)
    return np.asarray(p, dtype=float)


def depolarizing_site_probabilities(p: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return [..., 4] array containing [pI, pX, pY, pZ] at each site."""
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Each depolarizing rate must lie in [0, 1].")
    probs = np.stack([1.0 - p, p / 3.0, p / 3.0, p / 3.0], axis=-1)
    return probs


def log_likelihood_ratio_weight(q: np.ndarray | Sequence[float], *, eps: float = 1e-15) -> np.ndarray:
    """Return elementwise log((1-q)/q)."""
    q = clip_probabilities(q, eps=eps)
    return np.log((1.0 - q) / q)


def css_marginal_error_probability(
    p: np.ndarray | Sequence[float],
    *,
    sector: Literal["X", "Z"] = "X",
) -> np.ndarray:
    """Effective site-wise error probability seen by a CSS matching decoder.

    Under a depolarizing channel with total probability p:
        q_X = P(X or Y) = 2p/3
        q_Z = P(Z or Y) = 2p/3

    The same formula applies numerically to both sectors for symmetric
    depolarizing noise, but the sector label makes the intended usage explicit.
    """
    if sector not in {"X", "Z"}:
        raise ValueError("sector must be 'X' or 'Z'.")
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Each depolarizing rate must lie in [0, 1].")
    return (2.0 / 3.0) * p


def css_matching_weights_from_depolarizing(
    p: np.ndarray | Sequence[float],
    *,
    sector: Literal["X", "Z"] = "X",
    eps: float = 1e-15,
) -> np.ndarray:
    """Compute site-wise MWPM weights from site-wise depolarizing rates.

    This is the natural quantity if you want to compare your TN decoder against a
    CSS-type MWPM baseline under a site-dependent depolarizing model.
    """
    q = css_marginal_error_probability(p, sector=sector)
    return log_likelihood_ratio_weight(q, eps=eps)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import unittest


class TestDepolarizingWeights(unittest.TestCase):
    def test_site_probabilities_sum_to_one(self) -> None:
        p = np.array([0.0, 0.03, 0.3])
        probs = depolarizing_site_probabilities(p)
        self.assertTrue(np.allclose(np.sum(probs, axis=-1), 1.0))

    def test_css_marginal_probability(self) -> None:
        p = np.array([0.03, 0.06])
        qx = css_marginal_error_probability(p, sector="X")
        qz = css_marginal_error_probability(p, sector="Z")
        self.assertTrue(np.allclose(qx, 2 * p / 3))
        self.assertTrue(np.allclose(qz, 2 * p / 3))

    def test_matching_weights_are_finite(self) -> None:
        p = np.array([1e-6, 0.01, 0.1])
        w = css_matching_weights_from_depolarizing(p)
        self.assertTrue(np.all(np.isfinite(w)))

    def test_gaussian_sampling_shape(self) -> None:
        rng = np.random.default_rng(123)
        p = sample_gaussian_site_error_rates(0.02, 0.005, size=(3, 4), rng=rng)
        self.assertEqual(p.shape, (3, 4))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
