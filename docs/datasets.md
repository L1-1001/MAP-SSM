# Datasets

The manuscript evaluates nine public multivariate forecasting benchmarks:
ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, Exchange, and
Solar. Dataset-specific metadata and source pages are listed in
`configs/data/`.

No raw or processed benchmark data are included in this repository. Users are
responsible for obtaining data from the original provider and complying with
its license and terms.

## Chronological splits

- ETT datasets: 60% training, 20% validation, 20% test.
- All remaining benchmarks: 70% training, 10% validation, 20% test.

Rows must remain in chronological order. Standardization statistics are fitted
only on the training partition and then applied to validation and test data.

## Missing values and observed zeros

Missingness is represented by a separate binary mask: `1` means observed and
`0` means unavailable. Numerical values must never be used to infer whether an
entry is missing. In particular, physical nighttime zeros in Solar are observed
values unless a synthetic test mask explicitly removes them.
