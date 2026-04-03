import numpy as np 



def logical_failures_from_predictions(
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
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")

    residual = np.bitwise_xor(actual, predicted)
    return np.any(residual != 0, axis=1).astype(np.uint8)

def logical_fidelity_from_predictions(
    actual_observable_flips: np.ndarray,
    predicted_observable_flips: np.ndarray,
) -> float:
    failures = logical_failures_from_predictions(
        actual_observable_flips=actual_observable_flips,
        predicted_observable_flips=predicted_observable_flips,
    )
    return float(1.0 - np.mean(failures))


