import torch

from lpfn import analytic_control_jacobian
from lpfn.engines import TorchMatrixEngine
from lpfn.generators import GeneratorSet


def test_analytic_control_jacobian_shape(paulis):
    _, X, Y, _ = paulis
    gs = GeneratorSet.from_matrices([X, Y], labels=["X", "Y"], convention="hermitian")
    theta = torch.zeros(4, 3, 2, dtype=torch.float64)
    jac = analytic_control_jacobian(theta, gs)
    assert jac.shape == (4, 3, 2, 2, 2)


def test_exact_differential_matches_autograd_directional_derivative(paulis):
    _, X, Y, Z = paulis
    gs = GeneratorSet.from_matrices([X, Y, Z], labels=["X", "Y", "Z"], convention="hermitian")
    theta = torch.tensor(
        [[
            [0.19, -0.31, 0.08],
            [-0.12, 0.27, 0.33],
        ]],
        dtype=torch.float64,
        requires_grad=True,
    )
    direction = torch.tensor(
        [[
            [0.7, -0.2, 0.5],
            [0.1, 0.4, -0.6],
        ]],
        dtype=torch.float64,
    )
    engine = TorchMatrixEngine()

    def f(t):
        return engine.execute(t, gs)

    _, autograd_jvp = torch.autograd.functional.jvp(
        f, (theta,), (direction,), create_graph=False, strict=True
    )
    exact_jac = analytic_control_jacobian(theta.detach(), gs)
    exact_jvp = (
        exact_jac * direction[..., None, None].to(exact_jac.dtype)
    ).sum(dim=(1, 2))

    assert torch.allclose(exact_jvp, autograd_jvp, atol=2e-10, rtol=2e-10)


def test_exact_differential_matches_central_finite_difference(paulis):
    _, X, _, Z = paulis
    gs = GeneratorSet.from_matrices([X, Z], labels=["X", "Z"], convention="hermitian")
    theta = torch.tensor([[[0.41, -0.18], [0.07, 0.29]]], dtype=torch.float64)
    direction = torch.tensor([[[0.3, -0.7], [0.4, 0.2]]], dtype=torch.float64)
    jac = analytic_control_jacobian(theta, gs)
    exact_jvp = (jac * direction[..., None, None].to(jac.dtype)).sum(dim=(1, 2))

    engine = TorchMatrixEngine()
    eps = 1e-6
    plus = engine.execute(theta + eps * direction, gs)
    minus = engine.execute(theta - eps * direction, gs)
    finite_difference = (plus - minus) / (2 * eps)

    assert torch.allclose(exact_jvp, finite_difference, atol=3e-9, rtol=3e-9)
