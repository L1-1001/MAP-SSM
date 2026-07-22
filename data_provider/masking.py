"""Missingness protocols reported in the MAP-SSM manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DEFAULT_TRAINING_BLOCK_MAX = {96: 48, 720: 250}


@dataclass(frozen=True)
class MaskSample:
    observed: np.ndarray
    block_start: int | None = None
    block_length: int = 0
    point_missing_rate: float = 0.0


def training_mask(
    length: int,
    n_features: int,
    rng: np.random.Generator,
    block_min: int = 10,
    block_max: int | None = None,
) -> MaskSample:
    """Dynamic mixed block-and-point mask used for missing-aware training."""

    _validate_shape(length, n_features)
    if block_max is None:
        try:
            block_max = DEFAULT_TRAINING_BLOCK_MAX[length]
        except KeyError as error:
            raise ValueError(
                "block_max is required when length is not 96 or 720."
            ) from error
    if not 1 <= block_min <= block_max <= length:
        raise ValueError("Require 1 <= block_min <= block_max <= length.")

    block_length = int(rng.integers(block_min, block_max + 1))
    block_start = int(rng.integers(0, length - block_length + 1))
    point_rate = float(rng.uniform(0.1, 0.3))

    observed = np.ones((length, n_features), dtype=bool)
    block_end = block_start + block_length
    observed[block_start:block_end, :] = False
    outside_block = observed.copy()
    point_observed = rng.random((length, n_features)) >= point_rate
    observed[outside_block] = point_observed[outside_block]
    return MaskSample(observed, block_start, block_length, point_rate)


def validation_mask(
    length: int,
    n_features: int,
    rng: np.random.Generator,
    block_length: int = 50,
    point_missing_rate: float = 0.15,
) -> MaskSample:
    """Fixed-per-seed validation protocol with a final shared block."""

    _validate_shape(length, n_features)
    _validate_rate(point_missing_rate)
    if not 1 <= block_length <= length:
        raise ValueError("block_length must be within the historical window.")

    observed = np.ones((length, n_features), dtype=bool)
    block_start = length - block_length
    observed[block_start:, :] = False
    point_observed = rng.random((block_start, n_features)) >= point_missing_rate
    observed[:block_start, :] = point_observed
    return MaskSample(observed, block_start, block_length, point_missing_rate)


def test_block_mask(length: int, n_features: int, gap_length: int) -> MaskSample:
    """Mask the final ``gap_length`` historical positions for every variable."""

    _validate_shape(length, n_features)
    if not 0 <= gap_length <= length:
        raise ValueError("gap_length must be between zero and length.")
    observed = np.ones((length, n_features), dtype=bool)
    block_start = length - gap_length
    if gap_length:
        observed[block_start:, :] = False
    return MaskSample(observed, block_start, gap_length, 0.0)


def point_test_mask(
    length: int,
    n_features: int,
    missing_rate: float,
    rng: np.random.Generator,
) -> MaskSample:
    _validate_shape(length, n_features)
    _validate_rate(missing_rate)
    observed = rng.random((length, n_features)) >= missing_rate
    return MaskSample(observed, point_missing_rate=missing_rate)


def mixed_test_mask(
    length: int,
    n_features: int,
    gap_length: int,
    point_missing_rate: float,
    rng: np.random.Generator,
) -> MaskSample:
    """Apply a final block, then point masking only to remaining entries."""

    _validate_rate(point_missing_rate)
    block = test_block_mask(length, n_features, gap_length)
    observed = block.observed.copy()
    remaining_length = length - gap_length
    if remaining_length:
        point_observed = (
            rng.random((remaining_length, n_features)) >= point_missing_rate
        )
        observed[:remaining_length, :] = point_observed
    return MaskSample(
        observed,
        block.block_start,
        block.block_length,
        point_missing_rate,
    )


def generate_mask_bank(
    mode: Literal["training", "validation"],
    n_windows: int,
    length: int,
    n_features: int,
    seed: int,
) -> tuple[MaskSample, ...]:
    """Generate one reusable bank so methods receive matched corrupted inputs."""

    if n_windows <= 0:
        raise ValueError("n_windows must be positive.")
    rng = np.random.default_rng(seed)
    if mode == "training":
        generator = training_mask
    elif mode == "validation":
        generator = validation_mask
    else:
        raise ValueError("mode must be 'training' or 'validation'.")
    return tuple(generator(length, n_features, rng) for _ in range(n_windows))


def _validate_shape(length: int, n_features: int) -> None:
    if length <= 0 or n_features <= 0:
        raise ValueError("length and n_features must be positive.")


def _validate_rate(rate: float) -> None:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("missing rate must lie in [0, 1].")
