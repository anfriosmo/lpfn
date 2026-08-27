import pytest
import torch

from lpfn import DirectControls, LieProductNetwork, PauliGeneratorSet
from lpfn.engines import PauliMatrixEngine, TorchMatrixEngine
from lpfn.generators import GeneratorSet


def test_model_composes_control_model_and_engine(paulis):
    _, X, _, _ = paulis
    gs = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")

    def controls_fn(x):
        return x[:, :1].reshape(-1, 1, 1)

    controls = DirectControls(controls_fn, depth=1, num_generators=1)
    model = LieProductNetwork(generators=gs, controls=controls)
    x = torch.linspace(-0.9, 0.9, 7, dtype=torch.float64).reshape(-1, 1)
    out = model(x)
    exact = torch.linalg.matrix_exp((-1j * x[:, 0].to(torch.complex128))[:, None, None] * X)
    assert torch.allclose(out, exact, atol=1e-12, rtol=1e-12)


def test_model_engine_can_be_swapped_without_changing_controls():
    gs = PauliGeneratorSet(1, include_identity=False)

    def controls_fn(x):
        batch = x.shape[0]
        theta = torch.zeros(batch, 2, 3, dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = x[:, 0]
        theta[:, 1, 2] = x[:, 1]
        return theta

    controls = DirectControls(controls_fn, depth=2, num_generators=3)
    x = torch.tensor([[0.2, -0.3], [0.7, 0.1]], dtype=torch.float64)
    torch_model = LieProductNetwork(
        generators=gs, controls=controls, engine=TorchMatrixEngine()
    )
    pauli_model = LieProductNetwork(
        generators=gs, controls=controls, engine=PauliMatrixEngine()
    )
    assert torch.allclose(torch_model(x), pauli_model(x), atol=2e-11, rtol=2e-11)


def test_model_rejects_control_generator_dimension_mismatch():
    gs = PauliGeneratorSet(1, include_identity=False)
    controls = DirectControls(
        torch.zeros(1, 2, dtype=torch.float64), depth=1, num_generators=2
    )
    with pytest.raises(ValueError, match="num_generators"):
        LieProductNetwork(generators=gs, controls=controls)


def test_model_preserves_gradient_from_output_back_to_input(paulis):
    _, X, _, _ = paulis
    gs = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")

    def controls_fn(x):
        return (1.7 * x[:, :1]).reshape(-1, 1, 1)

    model = LieProductNetwork(
        generators=gs,
        controls=DirectControls(controls_fn, depth=1, num_generators=1),
    )
    x = torch.tensor([[0.23]], dtype=torch.float64, requires_grad=True)
    U = model(x)
    loss = U[:, 0, 0].real.sum()
    loss.backward()
    expected = -1.7 * torch.sin(1.7 * x.detach())
    assert x.grad is not None
    assert torch.allclose(x.grad, expected, atol=2e-11, rtol=2e-11)
