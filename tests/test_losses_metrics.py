import torch

from lpfn.losses import frobenius_loss, phase_insensitive_loss
from lpfn.metrics import operator_error, phase_insensitive_fidelity, unitarity_defect


def test_losses_vanish_on_identical_unitaries(paulis):
    I, _, _, _ = paulis
    U = I.unsqueeze(0).repeat(3, 1, 1)
    assert frobenius_loss(U, U) == 0
    assert torch.allclose(phase_insensitive_loss(U, U), torch.tensor(0.0, dtype=torch.float64))


def test_phase_loss_ignores_global_phase(paulis):
    I, _, _, _ = paulis
    phase = torch.tensor(0.73, dtype=torch.float64)
    target = I.unsqueeze(0)
    pred = (torch.exp(1j * phase).to(torch.complex128) * I).unsqueeze(0)
    assert phase_insensitive_loss(pred, target) < 1e-14
    assert frobenius_loss(pred, target) > 0.1


def test_metrics_have_expected_values(paulis):
    I, X, _, _ = paulis
    identity = I.unsqueeze(0)
    x = X.unsqueeze(0)
    assert unitarity_defect(identity).max() < 1e-14
    assert unitarity_defect(x).max() < 1e-14
    assert torch.allclose(operator_error(identity, identity), torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(phase_insensitive_fidelity(identity, identity), torch.ones(1, dtype=torch.float64))
