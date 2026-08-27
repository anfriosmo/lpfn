# Changelog

All notable public changes to LPFN are documented here.

## 0.1.0 — 2026-08-23

First public research release.

### Geometry and execution
- `GeneratorSet` and `PauliGeneratorSet` with validation and metadata.
- `ExecutionEngine` abstraction.
- `TorchMatrixEngine` as the general differentiable matrix reference engine.
- `PauliMatrixEngine` using the exact closed form for Pauli exponentials.
- `LieProductNetwork` as the public composition of controls, generators, and execution.
- Exact analytical control Jacobian for implementation validation.

### Scalar control models
- `DirectControls`.
- `ChebyshevControls`.
- `FourierControls`.
- `SplineControls`.
- `MLPControls`.
- Experimental `KANControls`.

### Training and evaluation
- Frobenius and phase-insensitive losses.
- Operator error, phase-insensitive fidelity, and unitarity defect.
- Reproducible dataset splits, target families, training utilities, and evaluation.
- Research benchmark infrastructure with parameter-cap selection and resumable runs.

### Validation
- Noncommutative factor ordering tests.
- Cross-engine numerical equivalence tests.
- Analytical/autograd/finite-difference derivative checks.
- End-to-end training tests.
