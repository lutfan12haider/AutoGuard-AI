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

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path so imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from backend.database import init_db, insert_diagnostic, get_all_diagnostics, get_diagnostic_by_id, delete_diagnostic, get_stats
from backend.model import load_model, warmup_model, is_model_loaded, get_model_info, predict_audio
from backend.schemas import AnalysisResult, DiagnosticRecord, HistoryResponse, StatsResponse, HealthResponse, DiagnoseResult
from backend.cnn_model import load_cnn, predict_with_cnn, is_cnn_loaded, get_cnn_info

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
        print(f"[Startup] WARNING: {e}")
    try:
        load_cnn()
    except FileNotFoundError as e:
        print(f"[Startup] WARNING (CNN): {e}")
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
        "http://localhost:4173",
        "http://127.0.0.1:4173",
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
        cnn=get_cnn_info(),
        database=db_status,
        version="1.0.0",
    )


@app.post("/api/analyze", response_model=AnalysisResult, tags=["Diagnosis"])
async def analyze_audio(
    file: UploadFile = File(...),
    save: bool = Query(default=True, description="Whether to persist this scan to history"),
):
    """
    Accept a WAV/MP3 file, run the ML model, save the result, and return it.

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
    scan_id = -1
    if save:
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


@app.delete("/api/history/{scan_id}", tags=["History"])
async def delete_scan(scan_id: int):
    """Delete a diagnostic record by ID."""
    deleted = delete_diagnostic(scan_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Scan #{scan_id} not found.",
        )
    return {"deleted": True, "scan_id": scan_id}


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


@app.post("/api/diagnose", response_model=DiagnoseResult, tags=["Diagnosis"])
async def diagnose_audio(
    file: UploadFile = File(...),
    save: bool = Query(default=True, description="Whether to persist this scan to history"),
):
    """
    Accept a WAV or MP3 file, run the PyTorch CNN model, and return the
    predicted engine condition with per-class confidence scores.

    This endpoint uses the trained AutoGuardCNN (ml/autoguard_model.pt)
    rather than the SVM pipeline used by /api/analyze.
    """
    if not is_cnn_loaded():
        raise HTTPException(
            status_code=503,
            detail="PyTorch CNN model is not loaded. Run ml/train.py first.",
        )

    data = await file.read()
    _validate_upload(file, data)

    try:
        result = predict_with_cnn(data, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN inference error: {e}")

    scores = result["confidence_scores"]
    top_conf = float(max(scores.values())) if scores else 0.0
    scan_id = None
    created_at = None
    if save:
        try:
            scan_id = insert_diagnostic(
                filename     = file.filename,
                prediction   = result["prediction"],
                confidence   = top_conf,
                class_scores = scores,
                duration_sec = result["duration_sec"],
                inference_ms = result["inference_ms"],
            )
            record = get_diagnostic_by_id(scan_id) if scan_id > 0 else None
            created_at = record["created_at"] if record else None
        except Exception as e:
            print(f"[DB] Insert failed for /api/diagnose: {e}")

    return DiagnoseResult(
        prediction        = result["prediction"],
        confidence_scores = result["confidence_scores"],
        duration_sec      = result["duration_sec"],
        inference_ms      = result["inference_ms"],
        scan_id           = scan_id,
        created_at        = created_at,
    )


@app.post("/api/consensus", response_model=AnalysisResult, tags=["Diagnosis"])
async def consensus_diagnose(
    file: UploadFile = File(...),
    save: bool = Query(default=True, description="Whether to persist this scan to history"),
):
    """
    Run both AutoGuardCNN (windowed log-mel) and Acoustic SVM (windowed 288-MFCC)
    on a single upload, combine probabilities with codec-aware weighting, and
    return one definitive class: engine_knock, healthy, or misfire.
    """
    if not is_cnn_loaded() and not is_model_loaded():
        raise HTTPException(status_code=503, detail="No diagnostic models loaded.")

    data = await file.read()
    _validate_upload(file, data)

    cnn_res = None
    svm_res = None
    last_err = None

    if is_cnn_loaded():
        try:
            cnn_res = predict_with_cnn(data, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            last_err = e

    if is_model_loaded():
        try:
            svm_res = predict_audio(data, file.filename)
        except ValueError as e:
            if cnn_res is None:
                raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            last_err = e

    if cnn_res is None and svm_res is None:
        raise HTTPException(status_code=500, detail=f"Consensus inference error: {last_err}")

    classes = ("engine_knock", "healthy", "misfire")
    is_mp3 = os.path.splitext(file.filename or "")[1].lower() == ".mp3"

    if cnn_res is not None and svm_res is not None:
        # SVM was trained on WAV only; MP3 compression shifts spectral centroid toward healthy.
        w_cnn = 0.75 if is_mp3 else 0.45
        w_svm = 1.0 - w_cnn
        c_scores = cnn_res["confidence_scores"]
        s_scores = svm_res["class_scores"]
        fused = {
            c: round(w_cnn * float(c_scores.get(c, 0.0)) + w_svm * float(s_scores.get(c, 0.0)), 4)
            for c in classes
        }
        duration_sec = cnn_res["duration_sec"] or svm_res["duration_sec"]
        inference_ms = round(cnn_res["inference_ms"] + svm_res["inference_ms"], 1)
        windows = svm_res.get("windows", [])
        window_count = svm_res.get("window_count", 1)
        disagree_std = svm_res.get("disagree_std", 0.0)
    elif cnn_res is not None:
        c_scores = cnn_res["confidence_scores"]
        fused = {c: round(float(c_scores.get(c, 0.0)), 4) for c in classes}
        duration_sec = cnn_res["duration_sec"]
        inference_ms = cnn_res["inference_ms"]
        windows = []
        window_count = 1
        disagree_std = 0.0
    else:
        s_scores = svm_res["class_scores"]
        fused = {c: round(float(s_scores.get(c, 0.0)), 4) for c in classes}
        duration_sec = svm_res["duration_sec"]
        inference_ms = svm_res["inference_ms"]
        windows = svm_res.get("windows", [])
        window_count = svm_res.get("window_count", 1)
        disagree_std = svm_res.get("disagree_std", 0.0)

    pred_label = max(classes, key=lambda c: fused[c])
    confidence = round(float(fused[pred_label]), 4)

    scan_id = -1
    created_at = ""
    if save:
        try:
            scan_id = insert_diagnostic(
                filename     = file.filename,
                prediction   = pred_label,
                confidence   = confidence,
                class_scores = fused,
                duration_sec = duration_sec,
                inference_ms = inference_ms,
            )
            record = get_diagnostic_by_id(scan_id) if scan_id > 0 else None
            created_at = record["created_at"] if record else ""
        except Exception as e:
            print(f"[DB] Insert failed for /api/consensus: {e}")

    return AnalysisResult(
        scan_id          = scan_id,
        filename         = file.filename,
        prediction       = pred_label,
        confidence       = confidence,
        class_scores     = fused,
        duration_sec     = duration_sec,
        inference_ms     = inference_ms,
        created_at       = created_at,
        window_count     = window_count,
        windows          = windows,
        uncertain        = False,
        uncertain_reason = None,
        model_prediction = pred_label,
        disagree_std     = disagree_std,
    )


def _read_json_safe(rel_path: str):
    path = Path(_ROOT) / rel_path
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/api/model-telemetry", tags=["System"])
async def get_model_telemetry():
    """Return real ML evaluation metrics, CNN training log, thresholds, and dataset metadata."""
    return {
        "svm": _read_json_safe("models/metadata.json"),
        "thresholds": _read_json_safe("models/thresholds.json"),
        "cnn_info": get_cnn_info(),
        "cnn_training_log": _read_json_safe("ml/autoguard_training_log.json"),
        "feature_config": _read_json_safe("data/prepared/feature_config.json"),
        "prepared_splits": {
            "train": _read_json_safe("data/prepared/train/meta.json"),
            "validation": _read_json_safe("data/prepared/validation/meta.json"),
            "test": _read_json_safe("data/prepared/test/meta.json"),
        },
        "eval_30clips": _read_json_safe("reports/eval_30clips.json"),
    }


@app.get("/api/samples", tags=["Samples"])
async def list_sample_audio():
    """List curated real engine audio clips from data/ for 1-click live diagnosis."""
    data_root = (Path(_ROOT) / "data").resolve()
    candidates = [
        ("kwerimoto/prepared/test/healthy", "healthy", "Held-Out Test · Healthy Engine", 3),
        ("kwerimoto/prepared/test/engine_knock", "engine_knock", "Held-Out Test · Engine Knock", 3),
        ("kwerimoto/prepared/test/misfire", "misfire", "Held-Out Test · Engine Misfire", 3),
        ("kwerimoto/prepared/real_test/engine_knock", "engine_knock", "Real-World Knock Recording", 2),
        ("kaggledataset/car diagnostics dataset/idle state/normal_engine_idle", "healthy", "Kaggle · Normal Idle", 2),
        ("kaggledataset/car diagnostics dataset/startup state/bad_ignition", "misfire", "Kaggle · Bad Ignition", 2),
        ("kaggledataset/car diagnostics dataset/idle state/low_oil", "misfire", "Kaggle · Low Oil Rough Idle", 2),
        ("kaggledataset/car diagnostics dataset/idle state/serpentine_belt", "misfire", "Kaggle · Serpentine Belt", 2),
    ]
    samples = []
    for rel_dir, expected_class, group_label, limit in candidates:
        folder = (data_root / rel_dir).resolve()
        if not folder.exists() or not folder.is_dir():
            continue
        wavs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
        for p in wavs[:limit]:
            rel_file = p.relative_to(data_root).as_posix()
            samples.append({
                "id": rel_file,
                "filename": p.name,
                "path": rel_file,
                "expected_class": expected_class,
                "source": group_label,
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
    return {"samples": samples}


@app.get("/api/samples/audio", tags=["Samples"])
async def get_sample_audio(path: str = Query(..., description="Relative sample path inside data/")):
    """Stream a curated sample WAV/MP3 clip from data/."""
    data_root = (Path(_ROOT) / "data").resolve()
    try:
        target = (data_root / path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid sample path.")
    if not target.is_relative_to(data_root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample file not found.")
    if target.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported sample extension.")
    media_type = "audio/mpeg" if target.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(str(target), media_type=media_type, filename=target.name)


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
