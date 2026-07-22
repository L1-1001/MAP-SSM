"""Forecasting metrics reported in the manuscript."""

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _paired_arrays(prediction, target)
    return float(np.mean(np.square(prediction - target)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = _paired_arrays(prediction, target)
    return float(np.mean(np.abs(prediction - target)))


def _paired_arrays(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}."
        )
    if prediction.size == 0:
        raise ValueError("Metrics require at least one value.")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require finite predictions and targets.")
    return prediction, target
