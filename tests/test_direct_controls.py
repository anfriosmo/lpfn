import torch

from lpfn.controls import DirectControls


def test_direct_controls_fixed_tensor_shape_and_batch_one():
    controls = DirectControls(torch.zeros(2, 3, dtype=torch.float64), depth=2, num_generators=3)
    x = torch.tensor([[0.25]], dtype=torch.float64)
    theta = controls(x)
    assert theta.shape == (1, 2, 3)


def test_direct_controls_broadcast_over_batch_deterministically():
    raw = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    controls = DirectControls(raw, depth=2, num_generators=3)
    x = torch.randn(5, 4, dtype=torch.float64)
    first = controls(x)
    second = controls(x)
    assert first.shape == (5, 2, 3)
    assert torch.equal(first, second)
    assert torch.equal(first[0], raw)
    assert torch.equal(first[-1], raw)


def test_direct_function_preserves_autograd_through_x():
    def fn(x):
        return x[:, :1].reshape(-1, 1, 1)

    controls = DirectControls(fn, depth=1, num_generators=1)
    x = torch.tensor([[0.2], [0.7]], dtype=torch.float64, requires_grad=True)
    theta = controls(x)
    loss = (theta**2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, 2 * x.detach())


def test_direct_controls_reject_integer_theta():
    controls = DirectControls(torch.zeros(1, 1, dtype=torch.int64), depth=1, num_generators=1)
    x = torch.zeros(1, 1, dtype=torch.float64)
    try:
        controls(x)
    except TypeError as exc:
        assert "floating dtype" in str(exc)
    else:
        raise AssertionError("integer-valued controls must be rejected")
