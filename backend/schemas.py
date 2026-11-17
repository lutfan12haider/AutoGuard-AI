"""
schemas.py
----------
Pydantic models for request/response validation in AutoGuard AI API.
"""

from pydantic import BaseModel
from typing import Optional


class WindowResult(BaseModel):
    """Per-window prediction detail"""
    window:     int
    start_sec:  float
    end_sec:    float
    skipped:    bool
    prediction: Optional[str]  = None
    confidence: Optional[float] = None
    scores:     Optional[dict[str, float]] = None
    reason:     Optional[str]  = None


class RepairRecommendationPayload(BaseModel):
    """Structured repair recommendation payload for healthy, misfire, and engine_knock"""
    predicted_class:         str
    display_label:           str           = ""
    confidence:              float
    confidence_tier:         str           = "high"   # high | moderate | low | unsupported
    uncertainty_warning:     Optional[str] = None
    diagnosis_overview:      list[str]     = []
    possible_causes:         list[str]     = []
    recommended_inspections: list[str]     = []
    possible_repairs:        list[str]     = []
    safety_warning:          Optional[str] = None
    maintenance_advice:      list[str]     = []
    disclaimer:              Optional[str] = None


class AnalysisResult(BaseModel):
    """Response from POST /api/analyze and POST /api/consensus"""
    scan_id:                 int
    filename:                str
    prediction:              str           # engine_knock | healthy | misfire | uncertain
    predicted_class:         str           = ""
    confidence:              float         # 0.0 – 1.0
    class_scores:            dict[str, float]
    duration_sec:            float
    inference_ms:            float
    created_at:              str
    # windowed inference fields
    window_count:            int           = 1
    windows:                 list[WindowResult] = []
    uncertain:               bool          = False
    uncertain_reason:        Optional[str] = None
    model_prediction:        Optional[str] = None   # actual model vote before abstain
    disagree_std:            float         = 0.0
    # Repair Recommendation System fields
    confidence_tier:         str           = "high"
    uncertainty_warning:     Optional[str] = None
    diagnosis_overview:      list[str]     = []
    possible_causes:         list[str]     = []
    recommended_inspections: list[str]     = []
    possible_repairs:        list[str]     = []
    safety_warning:          Optional[str] = None
    maintenance_advice:      list[str]     = []
    disclaimer:              Optional[str] = None
    repair_recommendations:  Optional[RepairRecommendationPayload] = None


class DiagnosticRecord(BaseModel):
    """A single record from the history table"""
    id:                      int
    filename:                str
    prediction:              str
    predicted_class:         str           = ""
    confidence:              float
    knock_score:             float
    healthy_score:           float
    misfire_score:           float
    duration_sec:            float
    inference_ms:            float
    created_at:              str
    confidence_tier:         str           = "high"
    uncertainty_warning:     Optional[str] = None
    diagnosis_overview:      list[str]     = []
    possible_causes:         list[str]     = []
    recommended_inspections: list[str]     = []
    possible_repairs:        list[str]     = []
    safety_warning:          Optional[str] = None
    maintenance_advice:      list[str]     = []
    disclaimer:              Optional[str] = None


class HistoryResponse(BaseModel):
    """Response from GET /api/history"""
    records: list[DiagnosticRecord]
    total:   int
    limit:   int
    offset:  int


class StatsResponse(BaseModel):
    """Response from GET /api/stats"""
    total_scans:      int
    completed_scans:  int
    detected_faults:  int
    healthy_results:  int
    knock_results:    int
    misfire_results:  int


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status:      str
    model:       dict
    cnn:         dict = {}
    database:    str
    version:     str


class ErrorResponse(BaseModel):
    """Standard error response"""
    error:   str
    detail:  Optional[str] = None


class DiagnoseResult(BaseModel):
    """Response from POST /api/diagnose  (PyTorch CNN model)"""
    prediction:              str                  # human-readable class name
    predicted_class:         str                  = ""
    confidence:              float                = 0.0
    confidence_scores:       dict[str, float]     # {class_name: probability}  sums to 1.0
    duration_sec:            float
    inference_ms:            float
    model:                   str                  = "AutoGuardCNN"
    scan_id:                 Optional[int]        = None
    created_at:              Optional[str]        = None
    uncertain:               bool                 = False
    uncertain_reason:        Optional[str]        = None
    confidence_tier:         str                  = "high"
    uncertainty_warning:     Optional[str]        = None
    diagnosis_overview:      list[str]            = []
    possible_causes:         list[str]            = []
    recommended_inspections: list[str]            = []
    possible_repairs:        list[str]            = []
    safety_warning:          Optional[str]        = None
    maintenance_advice:      list[str]            = []
    disclaimer:              Optional[str]        = None
    repair_recommendations:  Optional[RepairRecommendationPayload] = None
