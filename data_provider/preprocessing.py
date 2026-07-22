"""Preprocessing primitives that avoid train/validation/test leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Feature-wise standardization fitted on training observations only."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        values = _as_2d_float(values)
        self.mean_ = np.nanmean(values, axis=0)
        self.scale_ = np.nanstd(values, axis=0)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        if not np.isfinite(self.mean_).all():
            raise ValueError("Each feature needs at least one observed training value.")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = _as_2d_float(values)
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        values = np.asarray(values, dtype=np.float32)
        return values * self.scale_ + self.mean_

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before use.")


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D [time, feature] array, got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Cannot process an empty time-series array.")
    return array
