import pytest
import torch

from lpfn import MLPControls


def test_mlp_controls_shape_and_parameter_count():
    controls = MLPControls(
        input_dim=2,
        depth=3,
        num_generators=4,
        hidden_widths=(5, 7),
        dtype=torch.float64,
    )
    x = torch.randn(6, 2, dtype=torch.float64)
    theta = controls(x)
    assert theta.shape == (6, 3, 4)
    expected = (2 * 5 + 5) + (5 * 7 + 7) + (7 * 12 + 12)
    assert controls.trainable_parameter_count() == expected


def test_linear_mlp_can_represent_theta_equal_input_exactly():
    controls = MLPControls(
        input_dim=1,
        depth=1,
        num_generators=1,
        hidden_widths=(),
        dtype=torch.float64,
    )
    linear = controls.network[0]
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.zero_()
    x = torch.linspace(-1, 1, 9, dtype=torch.float64).reshape(-1, 1)
    assert torch.allclose(controls(x)[:, 0, 0], x[:, 0], atol=1e-14, rtol=1e-14)


def test_mlp_controls_receive_gradients():
    controls = MLPControls(
        input_dim=2, depth=2, num_generators=2, hidden_widths=(8,)
    )
    x = torch.randn(5, 2, dtype=torch.float64)
    loss = controls(x).square().sum()
    loss.backward()
    grads = [p.grad for p in controls.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_mlp_rejects_dtype_mismatch():
    controls = MLPControls(
        input_dim=1, depth=1, num_generators=1, hidden_widths=(4,), dtype=torch.float64
    )
    with pytest.raises(TypeError, match="dtype"):
        controls(torch.zeros(2, 1, dtype=torch.float32))
