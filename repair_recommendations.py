"""
Root-level re-export of backend.repair_recommendations so callers can import
`repair_recommendations` directly from the project root or from `backend`.
"""

from backend.repair_recommendations import (
    DEFAULT_DISAGREE_THRESHOLD,
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    DEFAULT_MODERATE_CONFIDENCE_THRESHOLD,
    REPAIR_RECOMMENDATIONS_DATA,
    SUPPORTED_CLASSES,
    get_repair_recommendations,
    load_confidence_thresholds,
)

__all__ = [
    "DEFAULT_DISAGREE_THRESHOLD",
    "DEFAULT_HIGH_CONFIDENCE_THRESHOLD",
    "DEFAULT_MODERATE_CONFIDENCE_THRESHOLD",
    "REPAIR_RECOMMENDATIONS_DATA",
    "SUPPORTED_CLASSES",
    "get_repair_recommendations",
    "load_confidence_thresholds",
]
