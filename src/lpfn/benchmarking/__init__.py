from .config import BenchmarkConfig, ControlSpec
from .factory import (
    admissible_control_specs,
    build_control_for_budget,
    build_control_from_spec,
    chebyshev_parameter_count,
    fourier_basis_size,
    fourier_parameter_count,
    spline_parameter_count,
    one_hidden_mlp_parameter_count,
)
from .manifest import build_manifest, source_tree_sha256, write_manifest
from .runner import paired_differences, postprocess_runs, run_benchmark, summarize_long
from .selection_config import SelectionBenchmarkConfig
from .selection_runner import (
    paired_selection_differences,
    run_selection_benchmark,
    selection_profiles,
    summarize_selections,
)

__all__ = [
    "BenchmarkConfig",
    "ControlSpec",
    "SelectionBenchmarkConfig",
    "admissible_control_specs",
    "build_control_for_budget",
    "build_control_from_spec",
    "build_manifest",
    "chebyshev_parameter_count",
    "fourier_basis_size",
    "fourier_parameter_count",
    "spline_parameter_count",
    "one_hidden_mlp_parameter_count",
    "paired_differences",
    "paired_selection_differences",
    "postprocess_runs",
    "run_benchmark",
    "run_selection_benchmark",
    "selection_profiles",
    "source_tree_sha256",
    "summarize_long",
    "summarize_selections",
    "write_manifest",
    "aggregate_selected_complexity",
    "aggregate_selection_metric",
    "audit_selection_output",
    "best_mean_cells",
    "generate_pilot_report",
]

from .reporting import (
    aggregate_selected_complexity,
    aggregate_selection_metric,
    audit_selection_output,
    best_mean_cells,
    generate_pilot_report,
)
