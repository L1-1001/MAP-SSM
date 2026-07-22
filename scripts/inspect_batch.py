"""Inspect public data-loader shapes without loading a forecasting model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider import WindowedTimeSeriesDataset, build_dataloader, load_csv_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--timestamp-column")
    parser.add_argument("--input-length", type=int, default=96)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    args = parser.parse_args()

    bundle = load_csv_dataset(
        args.csv,
        split_ratios=args.split_ratios,
        timestamp_column=args.timestamp_column,
    )
    dataset = WindowedTimeSeriesDataset(
        bundle.train,
        input_length=args.input_length,
        prediction_length=args.prediction_length,
    )
    batch = next(iter(build_dataloader(dataset, args.batch_size)))
    for key, tensor in batch.items():
        print(f"{key}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
