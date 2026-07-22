# MAP-SSM

Official partial pre-release for **Missing-Aware Prototype-Guided State Space
Model for Multivariate Time Series Forecasting**.

This repository currently contains the public data interface and experimental
protocol utilities used by the project. It does **not** contain the MAP-SSM
model, the training pipeline, checkpoints, or unpublished implementation
details. The complete experimental source code, configuration files,
preprocessing and mask-generation scripts, and seed records will be released
upon acceptance, as stated in the manuscript.

## Included

- chronological dataset splitting and train-only standardization;
- sliding-window PyTorch datasets and data loaders;
- block, point, and mixed missingness generation;
- deterministic seed handling for Python, NumPy, PyTorch, and data loaders;
- MSE and MAE metrics;
- public experimental-protocol and dataset metadata;
- data validation and batch-inspection utilities;
- unit tests for the released utilities.

No benchmark data are stored in this repository. Obtain each dataset from its
original source and review its license before use.

## Installation

The manuscript reports Python 3.9 and PyTorch 2.0.1. Create an isolated
environment and install the released utility dependencies:

```bash
python -m pip install -r requirements.txt
```

CUDA 12.1, cuDNN 8.9.0, and NVIDIA driver 580.105.08 describe the reported
runtime environment. The CUDA version bundled with a PyTorch wheel may differ
from the system toolkit version; install the PyTorch build appropriate for your
platform.

## Data layout

Place local CSV files under `data/`, which is excluded from version control:

```text
data/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  ETTm2.csv
  weather.csv
  electricity.csv
  traffic.csv
  exchange_rate.csv
  solar.csv
```

Dataset metadata are in `configs/data/`. File names can be overridden on the
command line; the repository never downloads or redistributes data silently.

## Quick checks

Validate a local CSV:

```bash
python scripts/check_dataset.py --csv data/ETTh1.csv --timestamp-column date
```

Inspect one forecasting batch without invoking a model:

```bash
python scripts/inspect_batch.py --csv data/ETTh1.csv --timestamp-column date
```

Run the released tests:

```bash
pytest -q
```

## Scope

The utilities in this pre-release document the public experimental interface;
they are not a complete reproduction package. In particular, there is no
training command in the current release. Please do not infer model performance
from the utility tests.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the paper once a
public bibliographic record is available.

## License

The released utility code is available under the MIT License. Dataset licenses
and third-party implementations remain governed by their original terms.
