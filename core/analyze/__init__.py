"""
BLASPHEMOUS Analysis Module

Comprehensive analysis tools for understanding refusal mechanisms in LLMs.

Modules:
- geometry: Core geometric analysis (silhouette, alignment detection, cone analysis)
- visualization: PaCMAP and other dimensionality reduction plots
- profiling: Detailed per-layer metrics and profiling
- concept_erasure: Before/after ablation comparison metrics

Usage:
    from blasphemous.analyze import analyze_model
    from blasphemous.analyze.visualization import plot_residuals
    from blasphemous.analyze.profiling import profile_layers
    from blasphemous.analyze.concept_erasure import measure_erasure
"""

from .geometry import analyze, AnalysisReport, LayerGeometry, ResidualCollections
from .visualization import plot_residuals, plot_layer_geometry
from .profiling import profile_layers, LayerProfile
from .concept_erasure import measure_erasure, ErasureMetrics

__all__ = [
    # Geometry (main analysis)
    "analyze",
    "AnalysisReport",
    "LayerGeometry",
    "ResidualCollections",
    # Visualization
    "plot_residuals",
    "plot_layer_geometry",
    # Profiling
    "profile_layers",
    "LayerProfile",
    # Concept Erasure
    "measure_erasure",
    "ErasureMetrics",
]

__version__ = "0.3.0"
