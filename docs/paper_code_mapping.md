# Paper-to-Code Map

This page maps claims about the experimental interface to the components in the
current public pre-release. It does not claim that the paper results can be
reproduced without the unreleased model and training code.

| Paper component | Public artifact | Current verification |
| --- | --- | --- |
| Chronological splits and train-only standardization | `data_provider/datasets.py`, `data_provider/preprocessing.py`, `configs/data/` | Unit tests and dataset inspection utility |
| Sliding forecasting windows and unmasked targets | `data_provider/dataloader.py` | `tests/test_dataloader.py` |
| Block, point, mixed, training, and validation masks | `data_provider/masking.py`, `configs/protocol.yaml` | `tests/test_masks.py` and `scripts/verify_protocol.py` |
| Matched random-seed initialization | `utils/seed.py`, `configs/protocol.yaml` | Deterministic public utility |
| MSE and MAE definitions | `utils/metrics.py` | `tests/test_metrics.py` |
| Reported MAP-SSM hyperparameters | `configs/model.yaml`, `configs/protocol.yaml` | Static configuration cross-check |
| Dataset identities and source pages | `configs/data/`, `docs/datasets.md` | Metadata only; benchmark data are not redistributed |

## Not Included in This Pre-release

The following components are not present and should not be inferred from the
released utilities:

- the MAP-SSM model implementation and forecasting head;
- training, validation, early-stopping, and checkpoint orchestration;
- baseline integrations and imputation pipelines;
- experiment launchers and result-generation scripts;
- trained weights, logs, and paper-table reproduction commands.

The complete experimental release is planned upon acceptance. Until then, the
repository supports inspection of the public data and missingness protocol, not
independent reproduction of the reported forecasting results.
