# Reproducibility Notes

## Reported environment

- Python 3.9
- PyTorch 2.0.1
- CUDA toolkit/runtime reported by the experiment host: 12.1
- cuDNN 8.9.0
- NVIDIA driver 580.105.08
- NVIDIA RTX 4090D, 24 GB
- Intel Xeon Platinum 8481C, 16 allocated CPU cores
- 80 GB RAM

The CUDA toolkit reported by a host and the CUDA version used to build a
PyTorch wheel are distinct. Record both `nvcc --version` and
`torch.version.cuda` when reconstructing an environment.

## Released checks

`pytest -q` verifies the behavior of public loading, scaling, metric, and mask
utilities. These tests do not train MAP-SSM and do not reproduce paper tables.
The command below additionally exercises the released missingness interface on
deterministic synthetic data:

```bash
python scripts/verify_protocol.py --seed 1 --input-length 720 \
  --features 7 --gap-length 200
```

The printed mask digests are integrity checks for the released utility behavior;
they are not experimental results.

## Release boundary

This pre-release intentionally excludes model code, training orchestration,
checkpoints, baseline integrations, and result-generation scripts. The complete
experimental release is planned upon acceptance. The current public boundary is
listed explicitly in `paper_code_mapping.md`.
