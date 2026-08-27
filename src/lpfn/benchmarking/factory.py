from __future__ import annotations

from math import comb

import torch

from lpfn.controls import (
    ChebyshevControls,
    ControlModel,
    FourierControls,
    KANControls,
    MLPControls,
    SplineControls,
)
from lpfn.controls.fourier import _canonical_frequency_vectors

from .config import ControlSpec


def chebyshev_parameter_count(*, input_dim: int, depth: int, num_generators: int, degree: int) -> int:
    basis_size = comb(input_dim + degree, degree)
    return depth * num_generators * basis_size


def fourier_basis_size(*, input_dim: int, max_frequency: int) -> int:
    return 1 + 2 * len(_canonical_frequency_vectors(input_dim, max_frequency))


def fourier_parameter_count(*, input_dim: int, depth: int, num_generators: int, max_frequency: int) -> int:
    return depth * num_generators * fourier_basis_size(
        input_dim=input_dim, max_frequency=max_frequency
    )


def spline_parameter_count(
    *, input_dim: int, depth: int, num_generators: int, num_basis_per_dim: int
) -> int:
    return depth * num_generators * (int(num_basis_per_dim) ** int(input_dim))


def kan_parameter_count(
    *, input_dim: int, output_dim: int, hidden_width: int, num_basis_per_edge: int
) -> int:
    if hidden_width < 0:
        raise ValueError("hidden_width must be nonnegative")
    if hidden_width == 0:
        edge_count = input_dim * output_dim
    else:
        edge_count = input_dim * hidden_width + hidden_width * output_dim
    return int(num_basis_per_edge) * edge_count


def one_hidden_mlp_parameter_count(*, input_dim: int, output_dim: int, width: int) -> int:
    return input_dim * width + width + width * output_dim + output_dim


def _nearest(candidates: list[tuple[int, int]], budget: int) -> tuple[int, int]:
    return min(candidates, key=lambda pair: (abs(pair[1] - budget), pair[0]))


def build_control_for_budget(
    kind: str,
    *,
    input_dim: int,
    depth: int,
    num_generators: int,
    budget: int,
    seed: int,
    chebyshev_init_scale: float = 0.02,
    fourier_init_scale: float = 0.02,
    spline_init_scale: float = 0.02,
    spline_degree: int = 3,
    mlp_activation: str = "tanh",
    dtype: torch.dtype = torch.float64,
) -> tuple[ControlModel, ControlSpec]:
    """Legacy quota builder kept for early benchmark compatibility."""
    if budget < 1:
        raise ValueError("budget must be positive")
    torch.manual_seed(int(seed))

    if kind == "chebyshev":
        degree, count = _nearest([
            (d, chebyshev_parameter_count(input_dim=input_dim, depth=depth, num_generators=num_generators, degree=d))
            for d in range(31)
        ], budget)
        model = ChebyshevControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            degree=degree, init_scale=chebyshev_init_scale, dtype=dtype,
        )
        arch = {"degree": degree, "basis": "total_degree_chebyshev", "num_basis_functions": model.num_basis_functions}
    elif kind == "fourier":
        freq, count = _nearest([
            (m, fourier_parameter_count(input_dim=input_dim, depth=depth, num_generators=num_generators, max_frequency=m))
            for m in range(21)
        ], budget)
        model = FourierControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            max_frequency=freq, init_scale=fourier_init_scale, dtype=dtype,
        )
        arch = {"max_frequency": freq, "basis": "l1_multivariate_fourier", "num_basis_functions": model.num_basis_functions}
    elif kind == "spline":
        start = spline_degree + 1
        n_basis, count = _nearest([
            (n, spline_parameter_count(input_dim=input_dim, depth=depth, num_generators=num_generators, num_basis_per_dim=n))
            for n in range(start, 65)
        ], budget)
        model = SplineControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            num_basis_per_dim=n_basis, degree=spline_degree,
            init_scale=spline_init_scale, dtype=dtype,
        )
        arch = {"num_basis_per_dim": n_basis, "degree": spline_degree, "basis": "open_uniform_tensor_bspline", "num_basis_functions": model.num_basis_functions}
    elif kind == "mlp":
        output_dim = depth * num_generators
        width, count = _nearest([
            (w, one_hidden_mlp_parameter_count(input_dim=input_dim, output_dim=output_dim, width=w))
            for w in range(1, 2049)
        ], budget)
        model = MLPControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            hidden_widths=(width,), activation=mlp_activation, dtype=dtype,
        )
        arch = {"hidden_widths": [width], "activation": mlp_activation}
    else:
        raise ValueError(
            f"legacy quota builder does not support {kind!r}; use admissible_control_specs"
        )

    spec = ControlSpec(kind=kind, parameter_budget=budget, parameter_count=count, architecture=arch)
    actual = model.trainable_parameter_count()
    if actual != count:
        raise RuntimeError(f"internal parameter-count mismatch for {kind}: formula={count}, actual={actual}")
    return model, spec


def admissible_control_specs(
    kind: str,
    *,
    input_dim: int,
    depth: int,
    num_generators: int,
    parameter_cap: int,
    chebyshev_degrees: tuple[int, ...] | list[int] = tuple(range(13)),
    mlp_widths: tuple[int, ...] | list[int] = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
    fourier_max_frequencies: tuple[int, ...] | list[int] = tuple(range(9)),
    spline_basis_sizes: tuple[int, ...] | list[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32),
    spline_degrees: tuple[int, ...] | list[int] = (0, 1, 2, 3),
    kan_hidden_widths: tuple[int, ...] | list[int] = (0, 1, 2, 4, 8),
    kan_basis_sizes: tuple[int, ...] | list[int] = (1, 2, 4, 8),
    kan_degrees: tuple[int, ...] | list[int] = (0, 1, 3),
    mlp_activation: str = "tanh",
) -> list[ControlSpec]:
    """Enumerate architecture candidates whose trainable parameter count <= cap."""
    if parameter_cap < 1:
        raise ValueError("parameter_cap must be positive")

    specs: list[ControlSpec] = []
    if kind == "chebyshev":
        for degree in sorted(set(int(d) for d in chebyshev_degrees)):
            count = chebyshev_parameter_count(
                input_dim=input_dim, depth=depth, num_generators=num_generators, degree=degree
            )
            if count <= parameter_cap:
                specs.append(ControlSpec(
                    kind=kind, parameter_budget=parameter_cap, parameter_count=count,
                    architecture={"degree": degree, "basis": "total_degree_chebyshev", "num_basis_functions": comb(input_dim + degree, degree)},
                ))
    elif kind == "fourier":
        for max_frequency in sorted(set(int(m) for m in fourier_max_frequencies)):
            if max_frequency < 0:
                raise ValueError("fourier_max_frequencies must be nonnegative")
            basis_size = fourier_basis_size(input_dim=input_dim, max_frequency=max_frequency)
            count = depth * num_generators * basis_size
            if count <= parameter_cap:
                specs.append(ControlSpec(
                    kind=kind, parameter_budget=parameter_cap, parameter_count=count,
                    architecture={"max_frequency": max_frequency, "basis": "l1_multivariate_fourier", "num_basis_functions": basis_size},
                ))
    elif kind == "spline":
        for degree in sorted(set(int(p) for p in spline_degrees)):
            if degree < 0:
                raise ValueError("spline_degrees must be nonnegative")
            for n_basis in sorted(set(int(n) for n in spline_basis_sizes)):
                if n_basis < degree + 1:
                    continue
                count = spline_parameter_count(
                    input_dim=input_dim, depth=depth, num_generators=num_generators,
                    num_basis_per_dim=n_basis,
                )
                if count <= parameter_cap:
                    specs.append(ControlSpec(
                        kind=kind, parameter_budget=parameter_cap, parameter_count=count,
                        architecture={"num_basis_per_dim": n_basis, "degree": degree, "basis": "open_uniform_tensor_bspline", "num_basis_functions": n_basis ** input_dim},
                    ))
    elif kind == "kan":
        output_dim = depth * num_generators
        for degree in sorted(set(int(p) for p in kan_degrees)):
            if degree < 0:
                raise ValueError("kan_degrees must be nonnegative")
            for n_basis in sorted(set(int(n) for n in kan_basis_sizes)):
                if n_basis < degree + 1:
                    continue
                for hidden_width in sorted(set(int(w) for w in kan_hidden_widths)):
                    if hidden_width < 0:
                        raise ValueError("kan_hidden_widths must be nonnegative")
                    count = kan_parameter_count(
                        input_dim=input_dim, output_dim=output_dim,
                        hidden_width=hidden_width, num_basis_per_edge=n_basis,
                    )
                    if count <= parameter_cap:
                        specs.append(ControlSpec(
                            kind=kind, parameter_budget=parameter_cap, parameter_count=count,
                            architecture={
                                "hidden_widths": [] if hidden_width == 0 else [hidden_width],
                                "num_basis_per_edge": n_basis,
                                "degree": degree,
                                "basis": "edge_bspline_kan",
                                "bounded_hidden_layers": hidden_width > 0,
                            },
                        ))
    elif kind == "mlp":
        output_dim = depth * num_generators
        for width in sorted(set(int(w) for w in mlp_widths)):
            count = one_hidden_mlp_parameter_count(input_dim=input_dim, output_dim=output_dim, width=width)
            if count <= parameter_cap:
                specs.append(ControlSpec(
                    kind=kind, parameter_budget=parameter_cap, parameter_count=count,
                    architecture={"hidden_widths": [width], "activation": mlp_activation},
                ))
    else:
        raise ValueError(f"unknown control model {kind!r}")

    def complexity_key(spec: ControlSpec) -> int:
        if spec.kind == "chebyshev":
            return int(spec.architecture["degree"])
        if spec.kind == "fourier":
            return int(spec.architecture["max_frequency"])
        if spec.kind == "spline":
            return 1000 * int(spec.architecture["num_basis_per_dim"]) + int(spec.architecture["degree"])
        if spec.kind == "kan":
            hidden = spec.architecture["hidden_widths"]
            width = int(hidden[0]) if hidden else 0
            return (
                1_000_000 * int(bool(hidden))
                + 10_000 * width
                + 100 * int(spec.architecture["num_basis_per_edge"])
                + int(spec.architecture["degree"])
            )
        return int(spec.architecture["hidden_widths"][0])

    # Deduplicate equal recorded architectures (useful when grids contain repeats).
    unique: dict[tuple[object, ...], ControlSpec] = {}
    for spec in specs:
        arch_key = tuple(sorted((k, str(v)) for k, v in spec.architecture.items()))
        unique[(spec.kind, spec.parameter_count, arch_key)] = spec
    specs = list(unique.values())
    specs.sort(key=lambda spec: (spec.parameter_count, complexity_key(spec)))
    if not specs:
        raise ValueError(f"no admissible {kind} candidate under parameter cap {parameter_cap}")
    return specs


def build_control_from_spec(
    spec: ControlSpec,
    *,
    input_dim: int,
    depth: int,
    num_generators: int,
    seed: int,
    chebyshev_init_scale: float = 0.02,
    fourier_init_scale: float = 0.02,
    spline_init_scale: float = 0.02,
    kan_init_scale: float = 0.02,
    mlp_activation: str = "tanh",
    dtype: torch.dtype = torch.float64,
) -> ControlModel:
    """Construct exactly the architecture recorded in a :class:`ControlSpec`."""
    torch.manual_seed(int(seed))
    if spec.kind == "chebyshev":
        model = ChebyshevControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            degree=int(spec.architecture["degree"]), init_scale=chebyshev_init_scale, dtype=dtype,
        )
    elif spec.kind == "fourier":
        model = FourierControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            max_frequency=int(spec.architecture["max_frequency"]), init_scale=fourier_init_scale, dtype=dtype,
        )
    elif spec.kind == "spline":
        model = SplineControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            num_basis_per_dim=int(spec.architecture["num_basis_per_dim"]),
            degree=int(spec.architecture["degree"]), init_scale=spline_init_scale, dtype=dtype,
        )
    elif spec.kind == "kan":
        model = KANControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            hidden_widths=tuple(int(w) for w in spec.architecture["hidden_widths"]),
            num_basis_per_edge=int(spec.architecture["num_basis_per_edge"]),
            degree=int(spec.architecture["degree"]), init_scale=kan_init_scale, dtype=dtype,
        )
    elif spec.kind == "mlp":
        model = MLPControls(
            input_dim=input_dim, depth=depth, num_generators=num_generators,
            hidden_widths=tuple(int(w) for w in spec.architecture["hidden_widths"]),
            activation=str(spec.architecture.get("activation", mlp_activation)), dtype=dtype,
        )
    else:
        raise ValueError(f"unknown control model {spec.kind!r}")

    actual = model.trainable_parameter_count()
    if actual != spec.parameter_count:
        raise RuntimeError(f"internal parameter-count mismatch for {spec.kind}: spec={spec.parameter_count}, actual={actual}")
    if actual > spec.parameter_budget:
        raise RuntimeError(f"constructed model exceeds cap: parameters={actual}, cap={spec.parameter_budget}")
    return model
