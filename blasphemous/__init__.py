import warnings
import logging

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")
warnings.filterwarnings("ignore", message=".*is deprecated.*")

# Suppress duplicate logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.ERROR)

from .ui import info, warn, error, success, phase, trial_log, metric

from .pipeline import run
from .extract import build_manifold
from .optimize import optimize
from .commit import commit
from .causal import run_causal_mediation, CausalReport, CausalComponent
from .benchmark import benchmark_model, compare_reports, BenchmarkReport

# Import from new analyze module (geometry is the main analysis)
from .analyze.geometry import (
    analyze,
    AnalysisReport,
    LayerGeometry,
    ResidualCollections,
)
from .analyze.visualization import plot_residuals, plot_layer_geometry
from .analyze.profiling import profile_layers, LayerProfile, ProfilingReport
from .analyze.concept_erasure import measure_erasure, ErasureMetrics

# Import LoRA ablation module
from .lora_ablation import (
    simple_lora_ablate,
    apply_projection_ablation,
    optimal_transport_ablate,
    get_layer_weights_bell_curve,
    get_layer_weights_uniform,
    get_layer_weights_selective,
)

__version__ = "0.4.0"
__all__ = [
    # UI helpers
    "info",
    "warn",
    "error",
    "success",
    "phase",
    "trial_log",
    "metric",
    # Main pipeline
    "run",
    # Legacy (for backwards compatibility)
    "analyze",
    "build_manifold",
    "optimize",
    "commit",
    "benchmark_model",
    "compare_reports",
    "BenchmarkReport",
    # Causal analysis
    "run_causal_mediation",
    "CausalReport",
    "CausalComponent",
    # New analysis modules
    "AnalysisReport",
    "LayerGeometry",
    "ResidualCollections",
    "plot_residuals",
    "plot_layer_geometry",
    "profile_layers",
    "LayerProfile",
    "ProfilingReport",
    "measure_erasure",
    "ErasureMetrics",
    # LoRA ablation methods (v0.4.0)
    "simple_lora_ablate",
    "apply_projection_ablation",
    "optimal_transport_ablate",
    "get_layer_weights_bell_curve",
    "get_layer_weights_uniform",
    "get_layer_weights_selective",
]
