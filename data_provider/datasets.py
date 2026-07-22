"""CSV loading and chronological splitting for public benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    feature_names: tuple[str, ...]
    scaler: StandardScaler


def load_csv_dataset(
    path: str | Path,
    split_ratios: Sequence[float],
    timestamp_column: str | None = None,
    feature_columns: Sequence[str] | None = None,
    standardize: bool = True,
) -> DatasetBundle:
    """Load, chronologically split, and optionally standardize a CSV dataset.

    The scaler is fitted only on the training partition. Rows are never shuffled.
    Missing values are preserved as NaN; their observation mask must be carried
    separately by downstream code.
    """

    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if timestamp_column is not None:
        if timestamp_column not in frame.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' is missing.")
        timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
        if not timestamps.is_monotonic_increasing:
            raise ValueError("Rows must be in chronological timestamp order.")

    if feature_columns is None:
        excluded = {timestamp_column} if timestamp_column is not None else set()
        selected = [column for column in frame.columns if column not in excluded]
    else:
        selected = list(feature_columns)

    if not selected:
        raise ValueError("No feature columns were selected.")
    missing_columns = sorted(set(selected) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Feature columns are missing: {missing_columns}")

    numeric = frame[selected].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float32)
    train_end, validation_end = chronological_boundaries(len(values), split_ratios)

    train = values[:train_end]
    validation = values[train_end:validation_end]
    test = values[validation_end:]

    scaler = StandardScaler()
    if standardize:
        scaler.fit(train)
        train = scaler.transform(train)
        validation = scaler.transform(validation)
        test = scaler.transform(test)
    else:
        scaler.mean_ = np.zeros(values.shape[1], dtype=np.float32)
        scaler.scale_ = np.ones(values.shape[1], dtype=np.float32)

    return DatasetBundle(
        train=train,
        validation=validation,
        test=test,
        feature_names=tuple(selected),
        scaler=scaler,
    )


def chronological_boundaries(
    length: int, split_ratios: Sequence[float]
) -> tuple[int, int]:
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios <= 0):
        raise ValueError("split_ratios must contain three positive values.")
    ratios = ratios / ratios.sum()
    train_end = int(length * ratios[0])
    validation_end = train_end + int(length * ratios[1])
    if train_end == 0 or validation_end <= train_end or validation_end >= length:
        raise ValueError("Dataset is too short for the requested split ratios.")
    return train_end, validation_end
