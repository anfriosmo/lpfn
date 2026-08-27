from __future__ import annotations

import math

import torch

from .base import BoxDomain, TargetFamily


X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


def _validate_x(x: torch.Tensor, input_dim: int) -> None:
    if x.ndim != 2 or int(x.shape[1]) != input_dim:
        raise ValueError(f"x must have shape [batch,{input_dim}]")
    if x.is_complex() or not x.dtype.is_floating_point:
        raise TypeError("target inputs must be real floating tensors")


class XRotationTarget(TargetFamily):
    """U(x) = exp(-i x X), x in [-1,1]."""

    def __init__(self) -> None:
        super().__init__(name="x_rotation", domain=BoxDomain((-1.0,), (1.0,)))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _validate_x(x, 1)
        Xd = X.to(device=x.device)
        angle = x[:, 0].to(torch.complex128)[:, None, None]
        return torch.linalg.matrix_exp(-1j * angle * Xd)


class XZProductTarget(TargetFamily):
    """U(x,y) = exp(-i x X) exp(-i y Z), (x,y) in [-1,1]^2."""

    def __init__(self) -> None:
        super().__init__(name="xz_product", domain=BoxDomain((-1.0, -1.0), (1.0, 1.0)))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _validate_x(x, 2)
        Xd = X.to(device=x.device)
        Zd = Z.to(device=x.device)
        ax = x[:, 0].to(torch.complex128)[:, None, None]
        ay = x[:, 1].to(torch.complex128)[:, None, None]
        qx = torch.linalg.matrix_exp(-1j * ax * Xd)
        qz = torch.linalg.matrix_exp(-1j * ay * Zd)
        return qx @ qz


class NoncommutingHamiltonianTarget(TargetFamily):
    """The nonlinear one-qubit family proposed in the LPFN paper.

    U(x,y) = exp[-i(sin(2pi x)X + xyY + cos(2pi y)Z)].
    """

    def __init__(self) -> None:
        super().__init__(
            name="noncommuting_hamiltonian",
            domain=BoxDomain((-1.0, -1.0), (1.0, 1.0)),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _validate_x(x, 2)
        Xd = X.to(device=x.device)
        Yd = Y.to(device=x.device)
        Zd = Z.to(device=x.device)
        sx = torch.sin(2.0 * math.pi * x[:, 0]).to(torch.complex128)[:, None, None]
        xy = (x[:, 0] * x[:, 1]).to(torch.complex128)[:, None, None]
        cy = torch.cos(2.0 * math.pi * x[:, 1]).to(torch.complex128)[:, None, None]
        H = sx * Xd + xy * Yd + cy * Zd
        return torch.linalg.matrix_exp(-1j * H)
