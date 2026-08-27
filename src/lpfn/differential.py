from __future__ import annotations

import torch

from lpfn.engines._matrix_common import (
    complex_dtype_for,
    prepare_base_point,
    validate_theta,
)
from lpfn.generators import GeneratorSet


def analytic_control_jacobian(
    theta: torch.Tensor,
    generators: GeneratorSet,
    *,
    base_point: torch.Tensor | None = None,
    validate_base_point: bool = True,
) -> torch.Tensor:
    """Exact derivative of the matrix LPFN output with respect to controls.

    Returns a tensor of shape ``[batch, K, r, N, N]`` whose ``(k,a)`` entry is

        dU/dtheta_{ka}
        = Q_M ... Q_{ell+1} B_a Q_ell ... Q_1 U_0,

    with ``ell`` the flattened increasing ``(k,a)`` factor index.

    This is a reference-calculus routine: it intentionally prioritizes direct
    correspondence with the mathematical formula over memory optimization.
    """

    validate_theta(theta, generators)
    batch_size, depth, r = theta.shape
    n = generators.matrix_dimension
    dtype = complex_dtype_for(theta)
    device = theta.device
    B = generators.as_skew_hermitian(dtype=dtype, device=device)
    U0 = prepare_base_point(
        base_point,
        batch_size=int(batch_size),
        n=n,
        dtype=dtype,
        device=device,
        validate=validate_base_point,
    )

    factors: list[torch.Tensor] = []
    right_states: list[torch.Tensor] = []
    state = U0
    for k in range(int(depth)):
        for a in range(int(r)):
            exponent = theta[:, k, a].to(dtype)[:, None, None] * B[a]
            Q = torch.linalg.matrix_exp(exponent)
            factors.append(Q)
            state = Q @ state
            right_states.append(state)

    m = len(factors)
    identity = torch.eye(n, dtype=dtype, device=device)
    suffix_after: list[torch.Tensor] = [
        identity.unsqueeze(0).expand(int(batch_size), -1, -1).clone()
        for _ in range(m)
    ]
    suffix = identity.unsqueeze(0).expand(int(batch_size), -1, -1).clone()
    for ell in range(m - 1, -1, -1):
        suffix_after[ell] = suffix
        suffix = suffix @ factors[ell]

    derivatives: list[torch.Tensor] = []
    ell = 0
    for _k in range(int(depth)):
        for a in range(int(r)):
            derivatives.append(suffix_after[ell] @ B[a] @ right_states[ell])
            ell += 1

    stacked = torch.stack(derivatives, dim=1)
    return stacked.reshape(int(batch_size), int(depth), int(r), n, n)
