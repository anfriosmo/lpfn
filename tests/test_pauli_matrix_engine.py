import pytest
import torch

from lpfn.engines import PauliMatrixEngine, TorchMatrixEngine
from lpfn.generators import GeneratorSet, PauliGeneratorSet


def test_pauli_engine_matches_reference_for_random_one_qubit_controls():
    gs = PauliGeneratorSet(1, include_identity=False)
    torch.manual_seed(11)
    theta = torch.randn(17, 5, gs.num_generators, dtype=torch.float64)
    reference = TorchMatrixEngine().execute(theta, gs)
    closed_form = PauliMatrixEngine().execute(theta, gs)
    assert torch.allclose(closed_form, reference, atol=3e-11, rtol=3e-11)


def test_pauli_engine_matches_reference_for_two_qubits():
    gs = PauliGeneratorSet(2, include_identity=False)
    torch.manual_seed(19)
    theta = 0.15 * torch.randn(3, 2, gs.num_generators, dtype=torch.float64)
    reference = TorchMatrixEngine().execute(theta, gs)
    closed_form = PauliMatrixEngine().execute(theta, gs)
    assert torch.allclose(closed_form, reference, atol=5e-11, rtol=5e-11)


def test_pauli_engine_preserves_ordering(paulis):
    _, X, _, Z = paulis
    gs = GeneratorSet.from_matrices([Z, X], labels=["Z", "X"], convention="hermitian")
    theta = torch.tensor([[[0.57, -0.31]]], dtype=torch.float64)
    out = PauliMatrixEngine().execute(theta, gs)[0]
    qz = torch.cos(theta[0, 0, 0]) * torch.eye(2, dtype=torch.complex128) - 1j * torch.sin(theta[0, 0, 0]) * Z
    qx = torch.cos(theta[0, 0, 1]) * torch.eye(2, dtype=torch.complex128) - 1j * torch.sin(theta[0, 0, 1]) * X
    assert torch.allclose(out, qx @ qz, atol=1e-12, rtol=1e-12)


def test_pauli_engine_rejects_non_involutory_generator():
    H = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64)).to(torch.complex128)
    gs = GeneratorSet.from_matrices([H], convention="hermitian")
    theta = torch.zeros(1, 1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"P_a\^2 = I"):
        PauliMatrixEngine().execute(theta, gs)


def test_pauli_engine_gradients_match_reference_engine():
    gs = PauliGeneratorSet(1, include_identity=False)
    theta_a = torch.tensor([[[0.21, -0.44, 0.17]]], dtype=torch.float64, requires_grad=True)
    theta_b = theta_a.detach().clone().requires_grad_(True)

    out_a = TorchMatrixEngine().execute(theta_a, gs)
    out_b = PauliMatrixEngine().execute(theta_b, gs)
    loss_a = (out_a.real.square() + 0.23 * out_a.imag.square()).sum()
    loss_b = (out_b.real.square() + 0.23 * out_b.imag.square()).sum()
    grad_a = torch.autograd.grad(loss_a, theta_a)[0]
    grad_b = torch.autograd.grad(loss_b, theta_b)[0]

    assert torch.allclose(grad_b, grad_a, atol=2e-10, rtol=2e-10)
