"""Lie-Product Function Networks (LPFN).

The package deliberately separates scalar control models from geometric
execution engines. The matrix engine is the reference semantics against
which future circuit engines should be validated.
"""

from ._version import __version__
from .controls import ChebyshevControls, ControlModel, DirectControls, FourierControls, KANControls, MLPControls, SplineControls
from .data import DatasetSplit, make_uniform_split
from .differential import analytic_control_jacobian
from .engines import ExecutionEngine, PauliMatrixEngine, TorchMatrixEngine
from .generators import GeneratorSet, PauliGeneratorSet
from .losses import frobenius_loss, phase_insensitive_loss
from .metrics import operator_error, phase_insensitive_fidelity, unitarity_defect
from .models import LieProductNetwork
from .targets import BoxDomain, NoncommutingHamiltonianTarget, TargetFamily, XRotationTarget, XZProductTarget
from .training import Trainer, TrainingResult, evaluate_unitary_model

__all__ = [
    "BoxDomain",
    "ChebyshevControls",
    "ControlModel",
    "DatasetSplit",
    "DirectControls",
    "ExecutionEngine",
    "FourierControls",
    "GeneratorSet",
    "KANControls",
    "LieProductNetwork",
    "MLPControls",
    "SplineControls",
    "NoncommutingHamiltonianTarget",
    "PauliGeneratorSet",
    "PauliMatrixEngine",
    "TargetFamily",
    "TorchMatrixEngine",
    "Trainer",
    "TrainingResult",
    "XRotationTarget",
    "XZProductTarget",
    "analytic_control_jacobian",
    "evaluate_unitary_model",
    "frobenius_loss",
    "make_uniform_split",
    "operator_error",
    "phase_insensitive_fidelity",
    "phase_insensitive_loss",
    "unitarity_defect",
]

