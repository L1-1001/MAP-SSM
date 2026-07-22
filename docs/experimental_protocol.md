# Public Experimental Protocol

This document summarizes the released data and missingness interface. It does
not describe unpublished model implementation details.

## Forecasting windows

- Standard fully observed experiments use input length 96.
- Long-sequence robustness experiments use input length 720.
- Prediction horizons are 96, 192, 336, and 720.
- Controlled block tests traverse chronological test windows with stride 1.
- Forecasting targets are never masked.

## Training masks

For every training window and epoch, draw an integer block length uniformly
from 10 through 48 for input length 96, or from 10 through 250 for input length
720. Draw a feasible block start uniformly and mask the block across all
variables. Draw a point-missing rate uniformly from 0.1 through 0.3 and apply
independent point masking only outside the block. The training mask bank is
resampled at the start of every epoch.

## Validation masks

Mask the final 50 historical positions across all variables, then independently
mask 15% of scalar entries outside that block. Generate one validation mask bank
per seed and hold it fixed across epochs.

## Test masks

For a requested block length, mask the final positions of the historical input
across every variable. Mixed tests apply independent point masking only to the
remaining observed entries. Identical test windows and masks are reused across
compared methods.

## Randomness

Repeated accuracy experiments use matched seeds 1, 2, and 3. Each seed controls
Python, NumPy, PyTorch, data-loader shuffling, initialization, and mask-bank
generation. Profiling seeds 2024, 2025, and 2026 are separate from accuracy
comparisons.
