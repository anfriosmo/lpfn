from __future__ import annotations

import torch

from lpfn import KANControls
from lpfn.benchmarking.factory import admissible_control_specs, build_control_from_spec, kan_parameter_count


def test_single_layer_kan_shape_and_parameter_count() -> None:
    model = KANControls(
        input_dim=2, depth=2, num_generators=3,
        hidden_widths=(), num_basis_per_edge=4, degree=3,
        dtype=torch.float64,
    )
    x = torch.tensor([[-1.0, 0.5], [0.25, 1.0]], dtype=torch.float64)
    y = model(x)
    assert y.shape == (2, 2, 3)
    assert model.trainable_parameter_count() == 2 * 6 * 4


def test_two_layer_kan_parameter_formula_matches_module() -> None:
    model = KANControls(
        input_dim=2, depth=3, num_generators=3,
        hidden_widths=(2,), num_basis_per_edge=2, degree=1,
        dtype=torch.float64,
    )
    expected = kan_parameter_count(
        input_dim=2, output_dim=9, hidden_width=2, num_basis_per_edge=2
    )
    assert expected == 44
    assert model.trainable_parameter_count() == expected


def test_hidden_kan_nodes_stay_in_spline_domain() -> None:
    torch.manual_seed(5)
    model = KANControls(
        input_dim=2, depth=1, num_generators=3,
        hidden_widths=(4,), num_basis_per_edge=4, degree=3,
        init_scale=4.0, dtype=torch.float64,
    )
    x = torch.linspace(-1.0, 1.0, 41, dtype=torch.float64)
    grid = torch.cartesian_prod(x, x)
    _, edges = model.forward_with_edges(grid)
    hidden = edges[0].mean(dim=1)
    assert float(hidden.max().detach()) <= 1.0 + 1e-12
    assert float(hidden.min().detach()) >= -1.0 - 1e-12


def test_edge_history_is_inspectable_and_differentiable() -> None:
    model = KANControls(
        input_dim=2, depth=1, num_generators=3,
        hidden_widths=(2,), num_basis_per_edge=2, degree=1,
        init_scale=0.02, dtype=torch.float64,
    )
    x = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
    theta, edges = model.forward_with_edges(x)
    assert len(edges) == 2
    assert edges[0].shape == (2, 2, 2)
    assert edges[1].shape == (2, 2, 3)
    theta.square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_admissible_kan_specs_respect_parameter_cap_and_rebuild() -> None:
    specs = admissible_control_specs(
        "kan", input_dim=2, depth=2, num_generators=3, parameter_cap=60,
        kan_hidden_widths=(0, 1, 2, 4), kan_basis_sizes=(1, 2, 4),
        kan_degrees=(0, 1, 3),
    )
    assert specs
    assert all(spec.parameter_count <= 60 for spec in specs)
    assert any(spec.architecture["hidden_widths"] == [] for spec in specs)
    assert any(spec.architecture["hidden_widths"] for spec in specs)
    for spec in specs:
        model = build_control_from_spec(
            spec, input_dim=2, depth=2, num_generators=3, seed=11
        )
        assert model.trainable_parameter_count() == spec.parameter_count


def test_linear_direct_kan_can_encode_identity_control_exactly() -> None:
    model = KANControls(
        input_dim=1, depth=1, num_generators=3,
        hidden_widths=(), num_basis_per_edge=2, degree=1,
        init_scale=0.0, dtype=torch.float64,
    )
    with torch.no_grad():
        # Open linear B-spline basis on [-1,1]: coefficients [-1,+1]
        # reproduce the identity function x exactly.
        model.coefficients[0].zero_()
        model.coefficients[0][0, 0, :] = torch.tensor([-1.0, 1.0])
    x = torch.tensor([[-1.0], [-0.25], [0.0], [0.4], [1.0]], dtype=torch.float64)
    theta = model(x)
    assert torch.allclose(theta[:, 0, 0], x[:, 0], atol=1e-12, rtol=0.0)
    assert torch.allclose(theta[:, 0, 1:], torch.zeros_like(theta[:, 0, 1:]), atol=1e-12, rtol=0.0)
