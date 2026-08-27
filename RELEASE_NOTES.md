# LPFN 0.1.0 — First public research release

LPFN 0.1.0 is the first public release of the Lie-Product Function Networks
research library.

Highlights:
- geometry-preserving Lie-product model for compact matrix Lie groups;
- PyTorch reference matrix engine and optimized Pauli engine;
- interchangeable Direct, Chebyshev, Fourier, B-spline, MLP, and experimental
  KAN control models;
- exact analytical control Jacobian and numerical validation tools;
- training, losses, metrics, targets, and reproducible benchmark utilities;
- 80 automated tests covering geometry, ordering, gradients, engines, controls,
  training, and benchmark-selection semantics;
- GitHub Actions CI for Python 3.10–3.13;
- PyPI/TestPyPI Trusted Publishing workflows.

This is a pre-1.0 research API. Backwards-incompatible changes may occur while
circuit execution engines, multi-qubit targets, and subgroup utilities mature.
