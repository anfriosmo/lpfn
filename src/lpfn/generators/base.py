from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping, Sequence

import torch

GeneratorConvention = Literal["hermitian", "skew_hermitian"]


class GeneratorSet:
    """Validated collection of fixed matrix generators.

    Parameters
    ----------
    matrices:
        Tensor of shape ``[r, N, N]``.
    labels:
        Human-readable labels, one per generator.
    convention:
        ``"hermitian"`` means matrices are H_a and the engine uses
        B_a = -i H_a. ``"skew_hermitian"`` means matrices are B_a directly.
    normalization:
        Descriptive metadata only in the reference implementation.
    metadata:
        Optional serializable or research metadata.

    Notes
    -----
    This class validates the geometric condition required to guarantee that
    every exponential factor lies in U(N).  It intentionally does not yet
    test whether the matrices form a real-linear basis of a particular Lie
    algebra; that is a later structural validation problem.
    """

    _VALID_CONVENTIONS = {"hermitian", "skew_hermitian"}

    def __init__(
        self,
        matrices: torch.Tensor,
        *,
        labels: Sequence[str] | None = None,
        convention: GeneratorConvention,
        normalization: str = "user",
        metadata: Mapping[str, Any] | None = None,
        atol: float = 1e-10,
        rtol: float = 1e-8,
    ) -> None:
        if convention not in self._VALID_CONVENTIONS:
            raise ValueError(
                "convention must be explicitly 'hermitian' or 'skew_hermitian'"
            )
        if not isinstance(matrices, torch.Tensor):
            raise TypeError("matrices must be a torch.Tensor")
        if matrices.ndim != 3:
            raise ValueError("matrices must have shape [r, N, N]")
        if matrices.shape[0] < 1:
            raise ValueError("at least one generator is required")
        if matrices.shape[-1] != matrices.shape[-2]:
            raise ValueError("every generator must be square")

        r = int(matrices.shape[0])
        if labels is None:
            labels = tuple(f"G{a}" for a in range(r))
        if len(labels) != r:
            raise ValueError("labels must contain exactly one label per generator")
        if len(set(labels)) != len(labels):
            raise ValueError("generator labels must be unique")

        self.matrices = matrices
        self.labels = tuple(str(label) for label in labels)
        self.convention: GeneratorConvention = convention
        self.normalization = str(normalization)
        self.metadata = deepcopy(dict(metadata or {}))
        self.atol = float(atol)
        self.rtol = float(rtol)
        self._validate_geometry()

    @classmethod
    def from_matrices(
        cls,
        generators: Sequence[torch.Tensor] | torch.Tensor,
        *,
        labels: Sequence[str] | None = None,
        convention: GeneratorConvention,
        normalization: str = "user",
        metadata: Mapping[str, Any] | None = None,
        atol: float = 1e-10,
        rtol: float = 1e-8,
    ) -> "GeneratorSet":
        if isinstance(generators, torch.Tensor):
            matrices = generators
            if matrices.ndim == 2:
                matrices = matrices.unsqueeze(0)
        else:
            if len(generators) == 0:
                raise ValueError("at least one generator is required")
            shapes = [tuple(g.shape) for g in generators]
            if any(len(shape) != 2 for shape in shapes):
                raise ValueError("each generator must be a matrix")
            if len(set(shapes)) != 1:
                raise ValueError("all generators must have the same matrix shape")
            matrices = torch.stack(tuple(generators), dim=0)

        return cls(
            matrices,
            labels=labels,
            convention=convention,
            normalization=normalization,
            metadata=metadata,
            atol=atol,
            rtol=rtol,
        )

    @property
    def num_generators(self) -> int:
        return int(self.matrices.shape[0])

    @property
    def matrix_dimension(self) -> int:
        return int(self.matrices.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.matrices.device

    def _validate_geometry(self) -> None:
        adjoint = self.matrices.mH
        if self.convention == "hermitian":
            valid = torch.allclose(
                self.matrices, adjoint, atol=self.atol, rtol=self.rtol
            )
            if not valid:
                raise ValueError(
                    "hermitian convention requires H_a^† = H_a for every generator"
                )
        else:
            valid = torch.allclose(
                self.matrices, -adjoint, atol=self.atol, rtol=self.rtol
            )
            if not valid:
                raise ValueError(
                    "skew_hermitian convention requires B_a^† = -B_a for every generator"
                )

    def as_skew_hermitian(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the B_a used in exp(theta B_a)."""
        matrices = self.matrices
        target_device = torch.device(device) if device is not None else matrices.device

        if dtype is None:
            if matrices.dtype in (torch.float64, torch.complex128):
                dtype = torch.complex128
            else:
                dtype = torch.complex64
        if dtype not in (torch.complex64, torch.complex128):
            raise TypeError("execution generators must use a complex dtype")

        matrices = matrices.to(device=target_device, dtype=dtype)
        if self.convention == "hermitian":
            matrices = (-1j) * matrices
        return matrices

    def operator_norms(self) -> torch.Tensor:
        return torch.linalg.matrix_norm(self.matrices, ord=2, dim=(-2, -1))

    def state_dict(self) -> dict[str, Any]:
        """Lightweight serialization used by tests and experiment configs."""
        return {
            "matrices": self.matrices.detach().clone(),
            "labels": list(self.labels),
            "convention": self.convention,
            "normalization": self.normalization,
            "metadata": deepcopy(self.metadata),
            "atol": self.atol,
            "rtol": self.rtol,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "GeneratorSet":
        return cls(
            state["matrices"],
            labels=state["labels"],
            convention=state["convention"],
            normalization=state.get("normalization", "user"),
            metadata=state.get("metadata", {}),
            atol=state.get("atol", 1e-10),
            rtol=state.get("rtol", 1e-8),
        )
