import pytest
import torch

from lpfn.generators import GeneratorSet, PauliGeneratorSet


def test_hermitian_generators_are_accepted(paulis):
    _, X, Y, Z = paulis
    gs = GeneratorSet.from_matrices(
        [X, Y, Z], labels=["X", "Y", "Z"], convention="hermitian"
    )
    assert gs.num_generators == 3
    assert gs.matrix_dimension == 2


def test_invalid_hermitian_generator_is_rejected():
    bad = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128)
    with pytest.raises(ValueError, match="H_a"):
        GeneratorSet.from_matrices([bad], convention="hermitian")


def test_skew_hermitian_generators_are_accepted(paulis):
    _, X, _, _ = paulis
    B = -1j * X
    gs = GeneratorSet.from_matrices([B], labels=["-iX"], convention="skew_hermitian")
    assert torch.allclose(gs.as_skew_hermitian(), B.unsqueeze(0))


def test_mismatched_shapes_are_rejected():
    A = torch.eye(2, dtype=torch.complex128)
    B = torch.eye(3, dtype=torch.complex128)
    with pytest.raises(ValueError, match="same matrix shape"):
        GeneratorSet.from_matrices([A, B], convention="hermitian")


def test_pauli_strings_are_hermitian_and_square_to_identity():
    gs = PauliGeneratorSet(2, include_identity=False)
    eye = torch.eye(4, dtype=torch.complex128)
    assert gs.num_generators == 15
    assert torch.allclose(gs.matrices, gs.matrices.mH)
    for P in gs.matrices:
        assert torch.allclose(P @ P, eye)


def test_generator_serialization_preserves_label_matrix_correspondence(paulis):
    _, X, Y, _ = paulis
    original = GeneratorSet.from_matrices(
        [X, Y], labels=["X", "Y"], convention="hermitian", metadata={"tag": "test"}
    )
    rebuilt = GeneratorSet.from_state_dict(original.state_dict())
    assert rebuilt.labels == original.labels
    assert rebuilt.convention == original.convention
    assert rebuilt.metadata == original.metadata
    assert torch.equal(rebuilt.matrices, original.matrices)
