# Public API

LPFN 0.1.x treats the names exported from `lpfn.__all__` as its public research
API. The package is pre-1.0, so backwards-incompatible changes remain possible.

## Geometry
- `GeneratorSet`
- `PauliGeneratorSet`
- `LieProductNetwork`

## Execution
- `ExecutionEngine`
- `TorchMatrixEngine`
- `PauliMatrixEngine`

## Scalar controls
- `ControlModel`
- `DirectControls`
- `ChebyshevControls`
- `FourierControls`
- `SplineControls`
- `MLPControls`
- `KANControls` (experimental)

## Targets and data
- `TargetFamily`
- `BoxDomain`
- `XRotationTarget`
- `XZProductTarget`
- `NoncommutingHamiltonianTarget`
- `DatasetSplit`
- `make_uniform_split`

## Training, losses, and metrics
- `Trainer`
- `TrainingResult`
- `evaluate_unitary_model`
- `frobenius_loss`
- `phase_insensitive_loss`
- `operator_error`
- `phase_insensitive_fidelity`
- `unitarity_defect`

## Validation utility
- `analytic_control_jacobian`
