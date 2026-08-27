import torch
import pytest


@pytest.fixture
def paulis():
    I = torch.eye(2, dtype=torch.complex128)
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    return I, X, Y, Z
