# LPFN

**Lie-Product Function Networks for geometry-preserving matrix Lie-group learning.**

LPFN is a PyTorch research library for learning functions whose outputs must
remain in a compact matrix Lie group. The library separates two concerns:

1. **scalar controls** `x -> theta(x)`, represented by interchangeable
   `ControlModel` classes; and
2. **Lie-group execution**, represented by an `ExecutionEngine` that composes
   ordered exponentials of fixed generators.

For controls `theta` with shape `[batch, K, r]`, LPFN implements the convention

```text
U(x) = Q_M ... Q_2 Q_1 U_0,
Q_ell = exp(theta_ell B_ell),
```

where factors are flattened in increasing `(k, a)` order, so `Q_1` acts first.
The matrix output stays on the target group by construction.

> **Status:** `0.1.0` is a public research release. The core semantics are tested,
> but the API may still evolve before `1.0`.

## Install

```bash
pip install liepfn
```

Development install:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Minimal example

```python
import torch
from lpfn import (
    ChebyshevControls,
    LieProductNetwork,
    PauliGeneratorSet,
    PauliMatrixEngine,
)

generators = PauliGeneratorSet(1)
controls = ChebyshevControls(
    input_dim=1,
    depth=2,
    num_generators=generators.num_generators,
    degree=3,
)
model = LieProductNetwork(
    generators=generators,
    controls=controls,
    engine=PauliMatrixEngine(),
)

x = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(-1, 1)
U = model(x)  # [32, 2, 2]
```

Changing `ChebyshevControls` to a spline, Fourier, MLP, or other control model
does not change the Lie geometry. Changing the matrix execution engine does not
change the scalar model.

## Included control models

- `DirectControls` â€” exact/reference controls.
- `ChebyshevControls` â€” total-degree multivariate Chebyshev expansions.
- `FourierControls` â€” real multivariate trigonometric bases with mixed modes.
- `SplineControls` â€” tensor-product open-uniform B-splines.
- `MLPControls` â€” generic neural controls.
- `KANControls` â€” experimental KAN-style B-spline edge controls.

## Execution engines

- `TorchMatrixEngine` â€” differentiable reference engine based on matrix
  exponentials.
- `PauliMatrixEngine` â€” optimized Pauli path using
  `cos(theta) I - i sin(theta) P`.

## What is tested

The test suite covers:

- generator validation and Pauli algebra;
- noncommutative factor ordering and base-point semantics;
- exact group/unitarity preservation;
- cross-engine numerical agreement;
- analytical, autograd, and finite-difference derivative checks;
- scalar-control shape/parameter contracts;
- losses and metrics;
- target/data utilities;
- end-to-end training;
- reproducible benchmark and validation-selection logic.

## Documentation

The source repository includes:

- `docs/QUICKSTART.md` â€” quick start;
- `docs/PUBLIC_API.md` â€” supported research API;
- `docs/CONVENTIONS.md` â€” mathematical conventions and ordering;
- `docs/REPRODUCIBILITY.md` â€” benchmark/reproducibility protocol;
- `docs/KAN_CONTROLS.md` â€” experimental KAN controls;
- `CONTRIBUTING.md` â€” development guidance;
- `CHANGELOG.md` â€” release history.

## Research benchmarks

Research scripts are kept in `benchmarks/`, outside the installed package.
Generated checkpoints and result tables are intentionally excluded from the
release repository. This prevents benchmark artifacts from becoming part of the
runtime dependency or wheel.

## Citation

If you use LPFN in research, please cite the software and the associated
*Lie-Product Function Networks on Compact Matrix Lie Groups* manuscript. A
machine-readable software citation is provided in `CITATION.cff`.

## License

LPFN is released under the MIT License. See `LICENSE`.

