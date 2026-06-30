# Methodology

The project treats the neural model as a candidate-pruning component for AV1 partition decisions, not as an unbounded replacement for the codec.

1. Instrument libaom in All-Intra mode and log the RDO-selected partition mode with the Y block, qindex, geometry, and level.
2. Convert the binary log to a validated dataset with a manifest and visual inspection samples.
3. Train ConvNeXt-AV1 on sequence-level splits.
4. Evaluate mode accuracy and RD-sensitive loss offline.
5. Export the model and run an offline simulation of top-k candidate pruning.
6. Integrate experimentally with libaom only after the offline pipeline is reproducible.
7. Compare codec-level results against unmodified libaom using BD-Rate, delta PSNR, and encoding time.

Required ablations:

- ConvNeXt-AV1 with adapters and QP conditioning.
- Without adapters.
- Without QP conditioning.
- Cross-entropy loss vs RD-weighted loss.
- Cross-QP evaluation.
