"""Legacy analyze module - imports from new blasphemous.analyze.geometry.

For backwards compatibility, this module re-exports the analyze function
and related classes from the new analyze module structure.
"""

# Re-export from new module for backwards compatibility
from .analyze.geometry import (
    analyze,
    AnalysisReport,
    LayerGeometry,
    ResidualCollections,
)

__all__ = [
    "analyze",
    "AnalysisReport",
    "LayerGeometry",
    "ResidualCollections",
]
