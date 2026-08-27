# Reproducibility

The reusable benchmark machinery lives under `src/lpfn/benchmarking/`; paper and
research entry points live in `benchmarks/`.

The benchmark protocol supports:
- paired train/validation/test splits by seed;
- hard trainable-parameter caps;
- validation-only model selection;
- test evaluation only after selection;
- per-candidate histories and checkpoints;
- resumable runs;
- manifests and code hashes;
- long-form and aggregated CSV reporting.

Generated research outputs are intentionally excluded from the package and Git
repository by default. Archive large result bundles separately and record the
software version/commit used to produce them.
