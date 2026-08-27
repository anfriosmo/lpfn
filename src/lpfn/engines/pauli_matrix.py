from __future__ import annotations

import torch

from lpfn.generators import GeneratorSet

from ._matrix_common import complex_dtype_for, prepare_base_point, validate_theta
from .base import ExecutionEngine


class PauliMatrixEngine(ExecutionEngine):
    """Closed-form matrix engine for Hermitian involutory generators.

    For every generator P with P^† = P and P^2 = I, the factor is evaluated as

        exp(-i theta P) = cos(theta) I - i sin(theta) P.

    Pauli strings are the intended use case. The implementation validates the
    involution property rather than trusting labels or metadata, so a custom
    Hermitian involutory generator set can also be used safely.
    """

    def __init__(
        self,
        *,
        validate_base_point: bool = True,
        validate_generators: bool = True,
    ) -> None:
        self.validate_base_point = bool(validate_base_point)
        self.validate_generators = bool(validate_generators)

    def execute(
        self,
        theta: torch.Tensor,
        generators: GeneratorSet,
        *,
        base_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_theta(theta, generators)
        if generators.convention != "hermitian":
            raise ValueError(
                "PauliMatrixEngine requires Hermitian generators P so that "
                "exp(-i theta P) can use the closed form"
            )

        batch_size, depth, _ = theta.shape
        n = generators.matrix_dimension
        dtype = complex_dtype_for(theta)
        device = theta.device
        P = generators.matrices.to(device=device, dtype=dtype)
        eye = torch.eye(n, dtype=dtype, device=device)

        if self.validate_generators:
            products = P @ P
            expected = eye.unsqueeze(0).expand(generators.num_generators, -1, -1)
            atol = 1e-10 if dtype == torch.complex128 else 1e-6
            if not torch.allclose(products, expected, atol=atol, rtol=1e-6):
                raise ValueError("PauliMatrixEngine requires P_a^2 = I for every generator")

        U = prepare_base_point(
            base_point,
            batch_size=int(batch_size),
            n=n,
            dtype=dtype,
            device=device,
            validate=self.validate_base_point,
        )
        eye_batch = eye.unsqueeze(0).expand(int(batch_size), -1, -1)

        for k in range(int(depth)):
            for a in range(generators.num_generators):
                angle = theta[:, k, a].to(dtype)[:, None, None]
                Q = torch.cos(angle) * eye_batch - 1j * torch.sin(angle) * P[a]
                U = Q @ U
        return U
