import math

import pytest
import torch

from lpfn import FourierControls


def test_fourier_controls_shape_parameter_count_and_domain():
    model = FourierControls(input_dim=2, depth=2, num_generators=3, max_frequency=2)
    x = torch.tensor([[-1.0, 0.0], [0.25, 1.0]], dtype=torch.float64)
    theta = model(x)
    assert theta.shape == (2, 2, 3)
    assert model.trainable_parameter_count() == 2 * 3 * model.num_basis_functions
    with pytest.raises(ValueError):
        model(torch.tensor([[1.1, 0.0]], dtype=torch.float64))


def test_fourier_basis_contains_mixed_modes_and_partition_constant():
    model = FourierControls(input_dim=2, depth=1, num_generators=1, max_frequency=2)
    freqs = {tuple(int(v) for v in row.tolist()) for row in model.frequencies}
    assert (1, 1) in freqs or (1, -1) in freqs
    x = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
    phi = model.basis(x)
    assert phi[0, 0].item() == 1.0
    assert phi.shape[1] == model.num_basis_functions


def test_fourier_can_exactly_encode_sin_2pi_x():
    model = FourierControls(input_dim=1, depth=1, num_generators=1, max_frequency=2)
    with torch.no_grad():
        model.coefficients.zero_()
        freqs = [int(v) for v in model.frequencies[:, 0].tolist()]
        j = freqs.index(2)
        # Layout is [constant, all cos, all sin].
        model.coefficients[0, 0, 1 + model.num_frequency_vectors + j] = 1.0
    x = torch.linspace(-1, 1, 51, dtype=torch.float64).reshape(-1, 1)
    got = model(x)[:, 0, 0]
    expected = torch.sin(2 * math.pi * x[:, 0])
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)
