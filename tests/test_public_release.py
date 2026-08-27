import lpfn


def test_public_release_version_and_core_exports():
    assert lpfn.__version__ == "0.1.0"
    for name in (
        "LieProductNetwork",
        "GeneratorSet",
        "PauliGeneratorSet",
        "ControlModel",
        "ChebyshevControls",
        "FourierControls",
        "SplineControls",
        "MLPControls",
        "TorchMatrixEngine",
        "PauliMatrixEngine",
    ):
        assert hasattr(lpfn, name), name
        assert name in lpfn.__all__, name
