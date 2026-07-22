"""Sliding-window datasets for multivariate forecasting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WindowedTimeSeriesDataset(Dataset):
    """Return historical input, observation mask, and unmasked future target."""

    def __init__(
        self,
        values: np.ndarray,
        input_length: int,
        prediction_length: int,
        stride: int = 1,
        mask_factory: Callable[[int, int], np.ndarray] | None = None,
        missing_placeholder: float = 0.0,
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("values must have shape [time, feature].")
        if input_length <= 0 or prediction_length <= 0 or stride <= 0:
            raise ValueError("Window lengths and stride must be positive.")
        available = array.shape[0] - input_length - prediction_length
        if available < 0:
            raise ValueError("Time series is shorter than one complete window.")

        self.values = array
        self.input_length = input_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.mask_factory = mask_factory
        self.missing_placeholder = np.float32(missing_placeholder)
        self._length = available // stride + 1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        start = index * self.stride
        middle = start + self.input_length
        end = middle + self.prediction_length

        history = self.values[start:middle].copy()
        target = self.values[middle:end].copy()
        observed = np.isfinite(history)

        if self.mask_factory is not None:
            synthetic_mask = np.asarray(
                self.mask_factory(index, history.shape[1]), dtype=bool
            )
            if synthetic_mask.shape != history.shape:
                raise ValueError(
                    "mask_factory returned shape "
                    f"{synthetic_mask.shape}, expected {history.shape}."
                )
            observed &= synthetic_mask

        corrupted = np.where(observed, history, self.missing_placeholder)
        return {
            "history": torch.from_numpy(corrupted.astype(np.float32)),
            "observed_mask": torch.from_numpy(observed.astype(np.float32)),
            "target": torch.from_numpy(target.astype(np.float32)),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
