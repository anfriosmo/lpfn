from __future__ import annotations

import torch

from lpfn.generators import GeneratorSet

from ._matrix_common import complex_dtype_for, prepare_base_point, validate_theta
from .base import ExecutionEngine


class TorchMatrixEngine(ExecutionEngine):
    """Reference matrix implementation of an LPFN Lie product.

    Given factors flattened in increasing ``(k, a)`` order as
    Q_1, ..., Q_M, this engine returns

        U = Q_M ... Q_2 Q_1 U_0.

    Thus Q_1 acts first. This is the ordering convention fixed by the
    implementation plan and is tested with non-commuting generators.
    """

    def __init__(self, *, validate_base_point: bool = True) -> None:
        self.validate_base_point = bool(validate_base_point)

    def execute(
        self,
        theta: torch.Tensor,
        generators: GeneratorSet,
        *,
        base_point: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_theta(theta, generators)

        batch_size, depth, _ = theta.shape
        n = generators.matrix_dimension
        complex_dtype = complex_dtype_for(theta)
        device = theta.device
        B = generators.as_skew_hermitian(dtype=complex_dtype, device=device)
        U = prepare_base_point(
            base_point,
            batch_size=int(batch_size),
            n=n,
            dtype=complex_dtype,
            device=device,
            validate=self.validate_base_point,
        )

        # Reference implementation: favor transparent mathematical semantics.
        for k in range(int(depth)):
            for a in range(generators.num_generators):
                exponent = theta[:, k, a].to(complex_dtype)[:, None, None] * B[a]
                Q = torch.linalg.matrix_exp(exponent)
                U = Q @ U
        return U
