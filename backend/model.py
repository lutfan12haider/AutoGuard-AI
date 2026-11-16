"""
model.py
--------
AutoGuard AI — model loading and inference.

Changes from v1:
  - predict_audio() now uses windowed inference for files longer than
    MAX_DURATION. It splits the audio into MAX_DURATION-second windows
    with 50% overlap, predicts each window independently, then aggregates
    probabilities by mean.  Per-window results are returned for debugging.
  - Abstain rule: return "uncertain" when
      (a) top mean probability < abstain_threshold (from thresholds.json), OR
      (b) windows disagree strongly (std of top-class proba across windows
          > disagree_threshold).
  - Thresholds are loaded from models/thresholds.json (never from test split).
"""

import os
import sys
import time
import json
import tempfile
import joblib
import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from ml.features import extract_features as extract_v3_features, TARGET_SR as SAMPLE_RATE, MAX_DURATION

MODELS_DIR = os.path.join(_ROOT, "models")

# Module-level singletons
_pipeline      = None
_label_encoder = None
_metadata      = None
_thresholds    = None   # loaded from thresholds.json


def _pool_stats(x: np.ndarray) -> np.ndarray:
    """Return [mean, std] pooled across time axis for v2 288-dim MFCC features."""
    if x.ndim == 1:
        return np.array([x.mean(), x.std()], dtype=np.float32)
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).astype(np.float32)


def _extract_v2_mfcc_features_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract the 288-dim MFCC + Delta + Delta2 + Spectral Contrast + Chroma +
    ZCR/RMS/Centroid/Rolloff/Bandwidth feature vector expected by models/best_model.pkl (v2).
    """
    if len(y) == 0:
        raise ValueError("Audio segment is empty")
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError("Audio segment appears to be silent")

    librosa = __import__("librosa")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    return np.concatenate([
        _pool_stats(mfcc),
        _pool_stats(delta_mfcc),
        _pool_stats(delta2_mfcc),
        _pool_stats(contrast),
        _pool_stats(chroma),
        _pool_stats(zcr),
        _pool_stats(rms),
        _pool_stats(centroid),
        _pool_stats(rolloff),
        _pool_stats(bandwidth),
    ]).astype(np.float32)


def _extract_v2_mfcc_features(audio_path: str) -> np.ndarray:
    """Load an audio path and extract the v2 288-dim MFCC + spectral feature vector."""
    librosa = __import__("librosa")
    y, sr = librosa.load(audio_path, sr=int(SAMPLE_RATE), duration=float(MAX_DURATION), mono=True)
    return _extract_v2_mfcc_features_from_array(y, sr)


def _extract_model_features(audio_path: str) -> np.ndarray:
    """Select v2 (288-dim) or v3 (16384-dim) feature extraction based on loaded pipeline."""
    expected_dim = getattr(_pipeline, "n_features_in_", None)
    if expected_dim is None and _metadata:
        expected_dim = _metadata.get("feature_config", {}).get("feature_dim")
    if expected_dim == 288:
        return _extract_v2_mfcc_features(audio_path)
    return extract_v3_features(audio_path)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_model() -> None:
    global _pipeline, _label_encoder, _metadata, _thresholds

    model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    meta_path    = os.path.join(MODELS_DIR, "metadata.json")
    thresh_path  = os.path.join(MODELS_DIR, "thresholds.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run: python ml/train.py"
        )

    print("[Model] Loading trained model ...")
    _pipeline      = joblib.load(model_path)
    _label_encoder = joblib.load(encoder_path)

    with open(meta_path) as f:
        _metadata = json.load(f)

    # Load thresholds — fall back to safe defaults if file missing
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            _thresholds = json.load(f)
        print(f"[Model] Thresholds: abstain={_thresholds['abstain_threshold']}  "
              f"disagree={_thresholds['disagree_threshold']}")
    else:
        _thresholds = {"abstain_threshold": 0.75, "disagree_threshold": 0.20}
        print("[Model] thresholds.json not found — using defaults.")

    print(f"[Model] Loaded: {_metadata['model_name']}")
    print(f"[Model] Classes: {_metadata['classes']}")
    print(f"[Model] Test accuracy: {_metadata['test_metrics']['accuracy']}")


def warmup_model() -> None:
    """Trigger Numba JIT so first real request isn't slow."""
    global _pipeline, _label_encoder
    if _pipeline is None:
        return
    print("[Model] Warming up ...")
    dummy = np.zeros(int(SAMPLE_RATE * 2), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, dummy, int(SAMPLE_RATE))
        # warmup will raise ValueError (silent audio) — that's fine
        try:
            feat = _extract_model_features(tmp_path)
            _pipeline.predict(feat.reshape(1, -1))
        except ValueError:
            pass
        print("[Model] Warmup complete.")
    except Exception as e:
        print(f"[Model] Warmup warning: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def is_model_loaded() -> bool:
    return _pipeline is not None


def get_model_info() -> dict:
    if _metadata is None:
        return {"loaded": False}
    return {
        "loaded":            True,
        "model_name":        _metadata["model_name"],
        "classes":           _metadata["classes"],
        "test_accuracy":     _metadata["test_metrics"]["accuracy"],
        "test_macro_f1":     _metadata["test_metrics"]["macro_f1"],
        "abstain_threshold": (_thresholds or {}).get("abstain_threshold", 0.75),
        "disagree_threshold": (_thresholds or {}).get("disagree_threshold", 0.20),
    }


# ── Windowed inference helpers ────────────────────────────────────────────────

def _window_audio(y: np.ndarray, sr: int, win_sec: float, hop_frac: float = 0.5):
    """
    Split audio array into overlapping windows.
    Returns list of (start_sec, end_sec, audio_array).
    """
    win_samples = int(sr * win_sec)
    hop_samples = int(win_samples * hop_frac)
    total       = len(y)

    if total <= win_samples:
        # Short audio: single window
        return [(0.0, total / sr, y)]

    windows = []
    start = 0
    while start < total:
        end = min(start + win_samples, total)
        seg = y[start:end]
        windows.append((start / sr, end / sr, seg))
        if end == total:
            break
        start += hop_samples
    return windows


def _predict_array(y: np.ndarray, sr: int, tmp_dir: str, window_idx: int) -> np.ndarray:
    """
    Extract features for a numpy audio window and return proba vector.
    Returns None if the window is silent or too short.
    """
    expected_dim = getattr(_pipeline, "n_features_in_", None)
    if expected_dim is None and _metadata:
        expected_dim = _metadata.get("feature_config", {}).get("feature_dim")

    if expected_dim == 288:
        try:
            feat = _extract_v2_mfcc_features_from_array(y, sr)
        except ValueError:
            return None
        proba = _pipeline.predict_proba(feat.reshape(1, -1))[0]
        return np.array(proba, dtype=np.float64)

    tmp_path = os.path.join(tmp_dir, f"win_{window_idx}.wav")
    try:
        sf.write(tmp_path, y, int(sr))
        try:
            feat = _extract_model_features(tmp_path)
        except ValueError:
            return None   # silent or too-short window
        proba = _pipeline.predict_proba(feat.reshape(1, -1))[0]
        return np.array(proba, dtype=np.float64)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Main inference entry point ────────────────────────────────────────────────

def predict_audio(audio_bytes: bytes, filename: str) -> dict:
    """
    Run inference on uploaded audio bytes.

    For files longer than MAX_DURATION seconds, the audio is split into
    overlapping windows and probabilities are averaged across windows.

    Returns
    -------
    dict:
        prediction   : str   — "engine_knock" | "healthy" | "misfire" | "uncertain"
        confidence   : float — mean top-class probability across windows
        class_scores : dict  — mean per-class probabilities
        duration_sec : float
        inference_ms : float
        windows      : list of per-window dicts (for debugging)
        window_count : int
        uncertain    : bool
        uncertain_reason : str | None
    """
    if _pipeline is None or _label_encoder is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Write upload to a temp file with the correct extension
    ext = os.path.splitext(filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Validate readability and duration
        try:
            info = sf.info(tmp_path)
            duration_sec = float(info.duration)
        except Exception as e:
            raise ValueError(f"Cannot read audio '{filename}': {e}") from e

        if duration_sec < 0.5:
            raise ValueError(
                f"Audio too short ({duration_sec:.2f}s). "
                "Please upload at least 0.5 seconds of engine audio."
            )

        # Load full audio (mono, target SR)
        y, sr = __import__("librosa").load(tmp_path, sr=int(SAMPLE_RATE), mono=True)
        if np.max(np.abs(y)) < 1e-6:
            raise ValueError(f"Audio appears to be silent: '{filename}'")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Windowed inference ────────────────────────────────────────────────────
    t0      = time.perf_counter()
    windows = _window_audio(y, sr, win_sec=MAX_DURATION, hop_frac=0.5)
    classes = list(_label_encoder.classes_)   # e.g. ['engine_knock','healthy','misfire']

    per_window   = []
    valid_probas = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, (t_start, t_end, seg) in enumerate(windows):
            proba = _predict_array(seg, sr, tmp_dir, idx)
            if proba is None:
                per_window.append({
                    "window": idx + 1,
                    "start_sec": round(t_start, 2),
                    "end_sec":   round(t_end, 2),
                    "skipped":   True,
                    "reason":    "silent or too short",
                })
                continue

            top_idx   = int(np.argmax(proba))
            top_class = classes[top_idx]
            per_window.append({
                "window":    idx + 1,
                "start_sec": round(t_start, 2),
                "end_sec":   round(t_end, 2),
                "skipped":   False,
                "prediction": top_class,
                "confidence": round(float(max(proba)), 4),
                "scores": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            })
            valid_probas.append(proba)

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    if not valid_probas:
        raise ValueError(f"All windows were silent or too short in '{filename}'.")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_proba   = np.mean(valid_probas, axis=0)   # shape (n_classes,)
    top_idx      = int(np.argmax(mean_proba))
    pred_label   = classes[top_idx]
    confidence   = round(float(mean_proba[top_idx]), 4)
    class_scores = {c: round(float(p), 4) for c, p in zip(classes, mean_proba)}

    # ── Abstain rule ──────────────────────────────────────────────────────────
    abstain_T  = _thresholds["abstain_threshold"]
    disagree_T = _thresholds["disagree_threshold"]

    uncertain        = False
    uncertain_reason = None

    if confidence < abstain_T:
        uncertain        = True
        uncertain_reason = (
            f"Low confidence ({confidence:.2f} < threshold {abstain_T:.2f}). "
            "Result may be unreliable."
        )

    if len(valid_probas) > 1:
        # Std of the top-class probability across windows
        top_class_probas = [p[top_idx] for p in valid_probas]
        disagree_std     = float(np.std(top_class_probas))
        if disagree_std > disagree_T:
            uncertain        = True
            uncertain_reason = (
                (uncertain_reason + " " if uncertain_reason else "") +
                f"Windows disagree strongly (std={disagree_std:.2f} > {disagree_T:.2f}). "
                "Different segments of the recording show different conditions."
            )
    else:
        disagree_std = 0.0

    return {
        "prediction":       "uncertain" if uncertain else pred_label,
        "confidence":       confidence,
        "class_scores":     class_scores,
        "duration_sec":     round(duration_sec, 2),
        "inference_ms":     inference_ms,
        "windows":          per_window,
        "window_count":     len(valid_probas),
        "uncertain":        uncertain,
        "uncertain_reason": uncertain_reason,
        # Always expose the actual model prediction even if abstained
        "model_prediction": pred_label,
        "disagree_std":     round(disagree_std, 4) if len(valid_probas) > 1 else 0.0,
    }
