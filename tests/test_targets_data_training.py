import torch

from lpfn import (
    ChebyshevControls,
    GeneratorSet,
    LieProductNetwork,
    PauliMatrixEngine,
    Trainer,
    XRotationTarget,
    make_uniform_split,
)
from lpfn.training import evaluate_unitary_model


def test_uniform_split_is_reproducible():
    target = XRotationTarget()
    a = make_uniform_split(target, n_train=10, n_val=5, n_test=7, seed=123)
    b = make_uniform_split(target, n_train=10, n_val=5, n_test=7, seed=123)
    c = make_uniform_split(target, n_train=10, n_val=5, n_test=7, seed=124)
    assert torch.equal(a.x_train, b.x_train)
    assert torch.equal(a.y_test, b.y_test)
    assert not torch.equal(a.x_train, c.x_train)


def test_target_outputs_are_unitary():
    target = XRotationTarget()
    split = make_uniform_split(target, n_train=8, n_val=4, n_test=6, seed=2)
    U = split.y_train
    eye = torch.eye(2, dtype=torch.complex128).expand(U.shape[0], -1, -1)
    assert torch.max((U.mH @ U - eye).abs()) < 1e-13


def test_trainer_fits_reference_x_rotation(paulis):
    _, X, _, _ = paulis
    target = XRotationTarget()
    split = make_uniform_split(target, n_train=32, n_val=16, n_test=32, seed=9)
    gs = GeneratorSet.from_matrices([X], labels=["X"], convention="hermitian")
    model = LieProductNetwork(
        generators=gs,
        controls=ChebyshevControls(
            input_dim=1, depth=1, num_generators=1, degree=3, init_scale=0.02
        ),
        engine=PauliMatrixEngine(),
    )
    result = Trainer(epochs=220, learning_rate=0.08, seed=9).fit(
        model,
        x_train=split.x_train,
        y_train=split.y_train,
        x_val=split.x_val,
        y_val=split.y_val,
    )
    metrics = evaluate_unitary_model(model, split.x_test, split.y_test)
    assert result.best_epoch >= 0
    assert result.best_val_loss < 1e-8
    assert metrics["frobenius_loss"] < 1e-8
    assert metrics["max_unitarity_defect"] < 1e-12


def test_all_one_qubit_reference_targets_are_unitary():
    from lpfn import NoncommutingHamiltonianTarget, XZProductTarget

    targets = [XRotationTarget(), XZProductTarget(), NoncommutingHamiltonianTarget()]
    for i, target in enumerate(targets):
        split = make_uniform_split(target, n_train=7, n_val=3, n_test=5, seed=30 + i)
        U = split.y_test
        eye = torch.eye(2, dtype=torch.complex128).expand(U.shape[0], -1, -1)
        assert torch.max((U.mH @ U - eye).abs()) < 2e-13
