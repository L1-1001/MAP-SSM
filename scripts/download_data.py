"""Show authoritative data sources; download only a user-supplied direct URL."""

import argparse
import shutil
import urllib.request
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", help="Explicit direct URL authorized by the user")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(f"dataset: {metadata['name']}")
    print(f"source_page: {metadata['source_page']}")
    print("Review the source license and terms before downloading.")

    if args.url is None:
        return
    if args.output is None:
        raise SystemExit("--output is required when --url is supplied.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url) as response, args.output.open("wb") as out:
        shutil.copyfileobj(response, out)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
