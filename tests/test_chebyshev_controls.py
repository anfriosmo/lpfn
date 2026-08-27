import pytest
import torch

from lpfn import ChebyshevControls


def test_total_degree_basis_size():
    controls = ChebyshevControls(
        input_dim=2, depth=1, num_generators=1, degree=3
    )
    # Number of monomials/multi-indices alpha in 2 variables with |alpha| <= 3.
    assert controls.num_basis_functions == 10
    assert controls.trainable_parameter_count() == 10


def test_univariate_basis_matches_known_chebyshev_values():
    controls = ChebyshevControls(
        input_dim=1, depth=1, num_generators=1, degree=4
    )
    x = torch.tensor([[-0.4], [0.2], [0.9]], dtype=torch.float64)
    phi = controls.basis(x)
    expected = torch.stack(
        [
            torch.ones_like(x[:, 0]),
            x[:, 0],
            2 * x[:, 0] ** 2 - 1,
            4 * x[:, 0] ** 3 - 3 * x[:, 0],
            8 * x[:, 0] ** 4 - 8 * x[:, 0] ** 2 + 1,
        ],
        dim=1,
    )
    assert torch.allclose(phi, expected, atol=1e-14, rtol=1e-14)


def test_multivariate_basis_contains_expected_products():
    controls = ChebyshevControls(
        input_dim=2, depth=1, num_generators=1, degree=2
    )
    x = torch.tensor([[0.3, -0.6]], dtype=torch.float64)
    phi = controls.basis(x)[0]
    index_to_col = {
        tuple(alpha.tolist()): i for i, alpha in enumerate(controls.multi_indices)
    }
    # alpha=(1,1) gives T1(x1)T1(x2)=x1*x2.
    assert torch.allclose(phi[index_to_col[(1, 1)]], x[0, 0] * x[0, 1])
    # alpha=(0,2) gives T2(x2).
    expected_t2 = 2 * x[0, 1] ** 2 - 1
    assert torch.allclose(phi[index_to_col[(0, 2)]], expected_t2)


def test_coefficients_can_represent_theta_equal_x_exactly():
    controls = ChebyshevControls(
        input_dim=1, depth=2, num_generators=3, degree=1
    )
    with torch.no_grad():
        controls.coefficients.zero_()
        index_to_col = {
            tuple(alpha.tolist()): i for i, alpha in enumerate(controls.multi_indices)
        }
        controls.coefficients[1, 2, index_to_col[(1,)]] = 1.0

    x = torch.linspace(-1, 1, 11, dtype=torch.float64).reshape(-1, 1)
    theta = controls(x)
    assert torch.allclose(theta[:, 1, 2], x[:, 0], atol=1e-14, rtol=1e-14)
    mask = theta.clone()
    mask[:, 1, 2] = 0
    assert torch.count_nonzero(mask) == 0


def test_domain_validation_rejects_out_of_box_input():
    controls = ChebyshevControls(
        input_dim=1, depth=1, num_generators=1, degree=2
    )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        controls(torch.tensor([[1.2]], dtype=torch.float64))


def test_chebyshev_coefficients_receive_gradients():
    controls = ChebyshevControls(
        input_dim=2, depth=2, num_generators=2, degree=2, init_scale=0.05
    )
    x = torch.tensor([[0.1, -0.2], [0.5, 0.7]], dtype=torch.float64)
    theta = controls(x)
    loss = theta.square().sum()
    loss.backward()
    assert controls.coefficients.grad is not None
    assert torch.isfinite(controls.coefficients.grad).all()
