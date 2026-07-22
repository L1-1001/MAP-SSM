"""Run deterministic checks of the released missingness protocol.

This utility uses synthetic arrays only. It does not load or evaluate MAP-SSM,
reproduce paper tables, or make claims about forecasting performance.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider import WindowedTimeSeriesDataset
from data_provider.masking import (
    mixed_test_mask,
    test_block_mask,
    training_mask,
    validation_mask,
)


def mask_digest(observed: np.ndarray) -> str:
    """Return a short integrity digest for a Boolean observation mask."""

    packed = np.packbits(np.asarray(observed, dtype=np.uint8), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify released MAP-SSM protocol utilities on synthetic data."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-length", type=int, choices=(96, 720), default=720)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--features", type=int, default=7)
    parser.add_argument("--gap-length", type=int, default=200)
    args = parser.parse_args()

    if args.seed < 0:
        raise SystemExit("--seed must be non-negative.")
    if args.prediction_length <= 0 or args.features <= 0:
        raise SystemExit("--prediction-length and --features must be positive.")
    if not 0 <= args.gap_length <= args.input_length:
        raise SystemExit("--gap-length must lie within the historical window.")

    train = training_mask(
        args.input_length,
        args.features,
        np.random.default_rng(args.seed),
    )
    validation = validation_mask(
        args.input_length,
        args.features,
        np.random.default_rng(args.seed),
    )
    block = test_block_mask(
        args.input_length,
        args.features,
        args.gap_length,
    )
    mixed = mixed_test_mask(
        args.input_length,
        args.features,
        args.gap_length,
        0.3,
        np.random.default_rng(args.seed),
    )

    block_slice = slice(train.block_start, train.block_start + train.block_length)
    assert not train.observed[block_slice].any()
    assert not validation.observed[-50:].any()
    if args.gap_length:
        assert not block.observed[-args.gap_length :].any()
        assert not mixed.observed[-args.gap_length :].any()

    values = np.zeros(
        (args.input_length + args.prediction_length, args.features),
        dtype=np.float32,
    )
    dataset = WindowedTimeSeriesDataset(
        values,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
    )
    sample = dataset[0]
    observed_zeros_retained = bool(sample["observed_mask"].bool().all())
    targets_unmasked = bool(sample["target"].shape[0] == args.prediction_length)
    assert observed_zeros_retained
    assert targets_unmasked

    print("scope: released data and missingness interfaces only")
    print(
        "training_mask: "
        f"block_start={train.block_start} "
        f"block_length={train.block_length} "
        f"point_rate={train.point_missing_rate:.6f} "
        f"digest={mask_digest(train.observed)}"
    )
    print(
        "validation_mask: "
        f"final_block=50 digest={mask_digest(validation.observed)}"
    )
    print(
        "block_test_mask: "
        f"final_block={args.gap_length} digest={mask_digest(block.observed)}"
    )
    print(
        "mixed_test_mask: "
        f"final_block={args.gap_length} point_rate=0.300000 "
        f"digest={mask_digest(mixed.observed)}"
    )
    print(f"observed_zero_retained: {observed_zeros_retained}")
    print(f"forecast_target_unmasked: {targets_unmasked}")


if __name__ == "__main__":
    main()
