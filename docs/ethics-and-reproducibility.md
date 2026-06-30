# Ethics and Reproducibility

This repository is public from the beginning to make methods auditable, but public code is not the same as validated science.

Rules for reporting:

- Mark all incomplete experiments as preliminary.
- Do not publish claims based on synthetic smoke tests.
- Do not commit raw videos, binary logs, converted datasets, checkpoints, or benchmark artifacts.
- Every table must reference a config, dataset manifest, git commit, libaom commit, hardware description, and exact command.
- Splits must be made by sequence, not by frame, to reduce temporal leakage.
- The unmodified libaom baseline must remain reproducible for every codec-level comparison.

Acceptable target thresholds for the initial research hypothesis:

- BD-Rate <= +1.0%.
- Delta PSNR >= -0.1 dB.
- Encoding time reduction >= 20-30%.
- Partition-mode accuracy >= 85%.

These thresholds are not proof of generality. Results must also be reported by sequence class, QP, and partition level.
