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


class AnalysisResult(BaseModel):
    """Response from POST /api/analyze"""
    scan_id:          int
    filename:         str
    prediction:       str           # engine_knock | healthy | misfire | uncertain
    confidence:       float         # 0.0 – 1.0
    class_scores:     dict[str, float]
    duration_sec:     float
    inference_ms:     float
    created_at:       str
    # windowed inference fields
    window_count:     int           = 1
    windows:          list[WindowResult] = []
    uncertain:        bool          = False
    uncertain_reason: Optional[str] = None
    model_prediction: Optional[str] = None   # actual model vote before abstain
    disagree_std:     float         = 0.0


class DiagnosticRecord(BaseModel):
    """A single record from the history table"""
    id:             int
    filename:       str
    prediction:     str
    confidence:     float
    knock_score:    float
    healthy_score:  float
    misfire_score:  float
    duration_sec:   float
    inference_ms:   float
    created_at:     str


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
    database:    str
    version:     str


class ErrorResponse(BaseModel):
    """Standard error response"""
    error:   str
    detail:  Optional[str] = None
