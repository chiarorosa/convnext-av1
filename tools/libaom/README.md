# libaom Instrumentation

This directory documents the codec-side logging contract. The production patch should keep all changes behind `LOG_PARTITION_DATA`.

Expected behavior:

- `LOG_PARTITION_DATA=OFF`: libaom behavior is unchanged.
- `LOG_PARTITION_DATA=ON`: libaom writes fixed-size partition samples after RDO has selected a partition mode.
- The output path must be configurable through an environment variable or encoder option.
- The binary record must match `docs/data-format.md`.

Suggested build flow:

```bash
cmake -DLOG_PARTITION_DATA=ON ..
make -j
```

Suggested encode flow:

```bash
AV1_PARTITION_LOG=av1_partition_data.bin ./aomenc \
  --cpu-used=0 --end-usage=q --cq-level=64 \
  --ivf -o output.ivf input.yuv -w 3840 -h 2160
```

The exact insertion point must be verified against the chosen libaom commit. Prefer the function that observes the final RDO partition decision for the block so that labels represent the baseline decision actually used by the encoder.
