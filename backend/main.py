"""
main.py
-------
AutoGuard AI — FastAPI backend.

Start the server with:
    python backend/main.py
  or:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health                 — backend + model status
    POST /api/analyze            — upload WAV, get prediction
    GET  /api/history            — list all diagnostic records
    GET  /api/history/{scan_id}  — single diagnostic record
    GET  /api/stats              — aggregate statistics
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path so imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.database import init_db, insert_diagnostic, get_all_diagnostics, get_diagnostic_by_id, get_stats
from backend.model import load_model, warmup_model, is_model_loaded, get_model_info, predict_audio
from backend.schemas import AnalysisResult, DiagnosticRecord, HistoryResponse, StatsResponse, HealthResponse

# ── Supported file types ───────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3", "audio/mpeg3", "audio/x-mpeg-3",
}
ALLOWED_EXTENSIONS = {".wav", ".mp3"}
MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup: initialise DB, load model, warm up JIT.
    FastAPI's modern replacement for @app.on_event("startup").
    """
    print("[Startup] Initialising AutoGuard AI backend ...")
    init_db()
    try:
        load_model()
        warmup_model()
    except FileNotFoundError as e:
        # Server starts even without a model — /health will report it
        print(f"[Startup] WARNING: {e}")
    print("[Startup] Ready.")
    yield
    # (cleanup on shutdown goes here if needed)


app = FastAPI(
    title="AutoGuard AI API",
    description="Engine audio classification backend",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the Vite dev server (port 8443) to talk to us ────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8443",
        "http://127.0.0.1:8443",
        "http://localhost:5173",   # fallback if pnpm chooses default Vite port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile, data: bytes) -> None:
    """
    Raise HTTPException for invalid uploads.
    Checks: filename extension, file size, MIME type.
    Accepts .wav and .mp3 files.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .wav and .mp3 files are accepted.",
        )

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(data) > MAX_FILE_SIZE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CONTENT_TYPES and ct != "application/octet-stream":
        # Be lenient — browsers sometimes send generic MIME types for audio
        pass


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns backend status, model info, and database connectivity.
    Use this to confirm the server is up before sending uploads.
    """
    # Quick DB check
    try:
        stats = get_stats()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return HealthResponse(
        status="ok",
        model=get_model_info(),
        database=db_status,
        version="1.0.0",
    )


@app.post("/api/analyze", response_model=AnalysisResult, tags=["Diagnosis"])
async def analyze_audio(file: UploadFile = File(...)):
    """
    Accept a WAV file, run the ML model, save the result, and return it.

    - Validates the file (type, size, readability)
    - Extracts MFCC features using the same pipeline as training
    - Returns the predicted engine condition and per-class probabilities
    - Saves the result to SQLite for history
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    # Read file bytes
    data = await file.read()
    _validate_upload(file, data)

    # Run inference
    try:
        result = predict_audio(data, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Persist to SQLite — store the actual model prediction, not "uncertain"
    db_prediction = result.get("model_prediction") or result["prediction"]
    try:
        scan_id = insert_diagnostic(
            filename     = file.filename,
            prediction   = db_prediction,
            confidence   = result["confidence"],
            class_scores = result["class_scores"],
            duration_sec = result["duration_sec"],
            inference_ms = result["inference_ms"],
        )
    except Exception as e:
        # DB failure shouldn't block the response — just log it
        print(f"[DB] Insert failed: {e}")
        scan_id = -1

    # Fetch the created_at timestamp back from DB
    record = get_diagnostic_by_id(scan_id) if scan_id > 0 else None
    created_at = record["created_at"] if record else ""

    return AnalysisResult(
        scan_id          = scan_id,
        filename         = file.filename,
        prediction       = result["prediction"],
        confidence       = result["confidence"],
        class_scores     = result["class_scores"],
        duration_sec     = result["duration_sec"],
        inference_ms     = result["inference_ms"],
        created_at       = created_at,
        window_count     = result.get("window_count", 1),
        windows          = result.get("windows", []),
        uncertain        = result.get("uncertain", False),
        uncertain_reason = result.get("uncertain_reason"),
        model_prediction = result.get("model_prediction"),
        disagree_std     = result.get("disagree_std", 0.0),
    )


@app.get("/api/history", response_model=HistoryResponse, tags=["History"])
async def get_history(
    limit:  int = Query(default=50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Records to skip"),
):
    """
    Return saved diagnostic records, newest first.

    Supports pagination via `limit` and `offset` query parameters.
    """
    records_raw = get_all_diagnostics(limit=limit, offset=offset)
    records = [DiagnosticRecord(**r) for r in records_raw]

    # Total count for pagination
    all_records = get_all_diagnostics(limit=100000, offset=0)
    total = len(all_records)

    return HistoryResponse(
        records=records,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/history/{scan_id}", response_model=DiagnosticRecord, tags=["History"])
async def get_scan(scan_id: int):
    """
    Return a single diagnostic record by its ID.

    Returns 404 if the scan does not exist.
    """
    record = get_diagnostic_by_id(scan_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scan #{scan_id} not found.",
        )
    return DiagnosticRecord(**record)


@app.get("/api/stats", response_model=StatsResponse, tags=["Dashboard"])
async def get_dashboard_stats():
    """
    Return aggregate statistics for the dashboard.

    All numbers come directly from the SQLite database.
    """
    stats = get_stats()
    return StatsResponse(
        total_scans     = stats.get("total_scans", 0) or 0,
        completed_scans = stats.get("completed_scans", 0) or 0,
        detected_faults = stats.get("detected_faults", 0) or 0,
        healthy_results = stats.get("healthy_results", 0) or 0,
        knock_results   = stats.get("knock_results", 0) or 0,
        misfire_results = stats.get("misfire_results", 0) or 0,
    )


# ── Global error handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"[ERROR] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # set True for development hot-reload
        log_level="info",
    )
