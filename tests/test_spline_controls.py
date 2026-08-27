import pytest
import torch

from lpfn import SplineControls


def test_spline_basis_partition_of_unity_including_endpoints():
    model = SplineControls(
        input_dim=1, depth=1, num_generators=1,
        num_basis_per_dim=7, degree=3,
    )
    x = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64).reshape(-1, 1)
    B = model.basis(x)
    assert B.shape == (101, 7)
    assert torch.min(B).item() >= -1e-14
    assert torch.allclose(B.sum(dim=1), torch.ones(101, dtype=torch.float64), atol=2e-12)
    assert torch.allclose(B[0], torch.tensor([1.,0.,0.,0.,0.,0.,0.], dtype=torch.float64))
    assert torch.allclose(B[-1], torch.tensor([0.,0.,0.,0.,0.,0.,1.], dtype=torch.float64))


def test_tensor_spline_shape_and_parameter_count():
    model = SplineControls(
        input_dim=2, depth=2, num_generators=3,
        num_basis_per_dim=5, degree=3,
    )
    x = torch.tensor([[-1.0, 1.0], [0.1, -0.2]], dtype=torch.float64)
    phi = model.basis(x)
    assert phi.shape == (2, 25)
    assert torch.allclose(phi.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-12)
    assert model(x).shape == (2, 2, 3)
    assert model.trainable_parameter_count() == 2 * 3 * 25


def test_spline_validates_basis_size_and_domain():
    with pytest.raises(ValueError):
        SplineControls(input_dim=1, depth=1, num_generators=1, num_basis_per_dim=3, degree=3)
    model = SplineControls(input_dim=1, depth=1, num_generators=1, num_basis_per_dim=4, degree=3)
    with pytest.raises(ValueError):
        model(torch.tensor([[1.01]], dtype=torch.float64))
