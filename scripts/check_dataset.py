"""Validate local CSV structure without uploading or modifying the data."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--timestamp-column")
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Dataset not found: {args.csv}")
    frame = pd.read_csv(args.csv)
    if frame.empty:
        raise SystemExit("Dataset contains no rows.")

    if args.timestamp_column:
        if args.timestamp_column not in frame.columns:
            raise SystemExit(f"Missing timestamp column: {args.timestamp_column}")
        timestamps = pd.to_datetime(frame[args.timestamp_column], errors="raise")
        if not timestamps.is_monotonic_increasing:
            raise SystemExit("Timestamps are not monotonically increasing.")

    feature_frame = frame.drop(columns=[args.timestamp_column], errors="ignore")
    numeric = feature_frame.apply(pd.to_numeric, errors="raise")
    print(f"path: {args.csv}")
    print(f"sha256: {sha256(args.csv)}")
    print(f"rows: {len(frame)}")
    print(f"features: {numeric.shape[1]}")
    print(f"missing_values: {int(numeric.isna().sum().sum())}")


if __name__ == "__main__":
    main()
