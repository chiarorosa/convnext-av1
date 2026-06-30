# ConvNeXt-AV1

Research-grade implementation of a ConvNeXt-based AV1 partition decision pipeline.

This repository is experimental. The goal is to make every result auditable: data extraction from libaom, binary conversion, training, evaluation, export, and codec-facing experiments should be reproducible from versioned configs and manifests.

## Scope

- Baseline codec: libaom.
- Initial mode: All-Intra, 8-bit, YUV 4:2:0.
- Input: Y-plane blocks padded to `1x64x64` plus normalized `qindex`.
- Output: partition logits for 64x64, 32x32, and 16x16 decision levels.
- Heavy data, raw videos, binary logs, checkpoints, and benchmark outputs are intentionally not tracked by Git.

## Quick Start

```bash
python -m pip install -e ".[dev]"
pytest
```

Convert a libaom partition log:

```bash
convnext-av1 convert av1_partition_data.bin outputs/dataset.npz --manifest outputs/dataset.json
```

Inspect converted samples:

```bash
convnext-av1 inspect outputs/dataset.npz outputs/inspect --max-samples 16
```

Train from a config:

```bash
convnext-av1 train configs/smoke.yaml
```

Evaluate a checkpoint:

```bash
convnext-av1 eval configs/smoke.yaml --checkpoint outputs/checkpoints/best.pt
```

## Reproducibility Rules

Every reported experiment should include:

- Git commit SHA.
- Full YAML config.
- Dataset manifest hash and sample counts.
- libaom commit/build flags.
- Hardware and dependency versions.
- Exact command used to produce the result.

Preliminary results must be labeled as preliminary until the dataset, split, baseline, and benchmark scripts are independently reproducible.
