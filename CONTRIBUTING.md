# Contributing to LPFN

LPFN is a research library. Contributions should preserve the separation between
scalar control models and Lie-group execution.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

## Design rules

1. A `ControlModel` maps `[batch, input_dim]` to real controls `[batch, K, r]`.
2. An `ExecutionEngine` consumes controls and generators; it must not encode a
   particular scalar approximation family.
3. Factor ordering must follow the convention documented in `docs/CONVENTIONS.md`.
4. New execution engines should be checked against `TorchMatrixEngine` on small
   matrix problems before being used in research benchmarks.
5. New mathematical shortcuts should receive numerical tests against the general
   reference path.

## Pull requests

Please include tests for new behavior and keep research outputs/checkpoints out of
the repository. Benchmark scripts belong in `benchmarks/`; reusable library code
belongs in `src/lpfn/`.
