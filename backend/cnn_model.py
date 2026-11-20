"""
backend/cnn_model.py
--------------------
AutoGuard AI — PyTorch CNN loader and inference for the /api/diagnose endpoint.

Loads ml/autoguard_model.pt once at startup and exposes predict_with_cnn()
which is called by backend/main.py.

Design rules:
  - Model weights are loaded from ml/autoguard_model.pt (produced by ml/train.py).
  - label_map.json (produced by ml/features.py) maps int indices → class names.
  - Feature extraction uses extract_features() from ml/features.py — the same
    function used during training, so train/serve preprocessing is identical.
  - The uploaded audio bytes are written to a NamedTemporaryFile so
    extract_features() can open it by path (librosa requires a file path).
  - Inference runs on CPU; torch.no_grad() ensures no gradient memory is used.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

# Resolve project root so ml.features and ml.train imports work regardless
# of where uvicorn is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from ml.features import (
    extract_features,
    FEATURE_DIM,
    TARGET_SR,
    MAX_DURATION,
    load_audio,
    clip_or_pad,
    compute_log_mel,
)   # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
_MODEL_PT       = os.path.join(_ROOT, "ml", "autoguard_model.pt")
_LABEL_MAP_JSON = os.path.join(_ROOT, "data", "prepared", "label_map.json")

# ── Module-level singletons (loaded once at startup) ──────────────────────────
_cnn_model  = None   # nn.Module
_label_map  = None   # dict[int, str]  e.g. {0: "engine_knock", 1: "healthy", 2: "misfire"}
_n_classes  = None   # int
_cnn_meta   = None   # dict from checkpoint metadata


# ── Model definition (must match ml/train.py exactly) ─────────────────────────

class _ConvBlock(nn.Module):
    """Two conv layers, BN, ReLU, MaxPool2d(2), Dropout2d."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoGuardCNN(nn.Module):
    """
    Lightweight CNN for engine sound classification.

    Input  : flat tensor (B, 16384)
    Reshape: (B, 1, 128, 128)  via .view(-1, 1, 128, 128)
    Output : logits (B, n_classes)

    Spatial: 128 → 64 → 32 → 16  (three MaxPool2d(2))
    Channels:  1 → 32 → 64 → 128
    After GAP: (B, 128) → head → (B, n_classes)
    """

    N_MELS   = 128
    N_FRAMES = 128

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1,   32),
            _ConvBlock(32,  64),
            _ConvBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept flat (B, 16384) or pre-shaped (B, 1, 128, 128)
        if x.dim() == 2:
            x = x.view(-1, 1, self.N_MELS, self.N_FRAMES)
        x = self.features(x)       # (B, 128, 16, 16)
        x = self.gap(x).flatten(1) # (B, 128)
        return self.head(x)        # (B, n_classes)


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_cnn() -> None:
    """
    Load the trained CNN weights and label map into memory.
    Called once during FastAPI startup via lifespan().

    Raises
    ------
    FileNotFoundError
        If autoguard_model.pt has not been generated yet (run ml/train.py first).
    """
    global _cnn_model, _label_map, _n_classes, _cnn_meta

    if not os.path.exists(_MODEL_PT):
        raise FileNotFoundError(
            f"PyTorch CNN weights not found at {_MODEL_PT}. "
            "Run:  python ml/train.py"
        )

    # Load label map  {str(int): class_name}  — JSON keys are always strings
    if os.path.exists(_LABEL_MAP_JSON):
        with open(_LABEL_MAP_JSON) as f:
            raw = json.load(f)
        _label_map = {int(k): v for k, v in raw.items()}
    else:
        # Fallback if data/prepared/ was not generated
        _label_map = {0: "engine_knock", 1: "healthy", 2: "misfire"}
        print("[CNN] label_map.json not found — using hard-coded defaults.")

    _n_classes = len(_label_map)

    # Load checkpoint
    checkpoint = torch.load(_MODEL_PT, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    meta       = checkpoint.get("metadata", {})

    # Instantiate and load weights
    model = AutoGuardCNN(n_classes=_n_classes)
    model.load_state_dict(state_dict)
    model.eval()   # disable dropout / batch-norm train behaviour

    _cnn_model = model
    _cnn_meta = meta

    print(f"[CNN] Loaded AutoGuardCNN  "
          f"classes={list(_label_map.values())}  "
          f"test_acc={meta.get('test_accuracy', 'unknown')}")


def is_cnn_loaded() -> bool:
    """Return True if the CNN has been loaded successfully."""
    return _cnn_model is not None


def get_cnn_info() -> dict:
    """Return summary metadata for the loaded PyTorch AutoGuardCNN model."""
    if _cnn_model is None:
        return {"loaded": False}
    meta = _cnn_meta or {}
    return {
        "loaded":        True,
        "model_name":    meta.get("model_class", "AutoGuardCNN"),
        "classes":       list(_label_map.values()) if _label_map else [],
        "test_accuracy": meta.get("test_accuracy"),
        "best_epoch":    meta.get("best_epoch"),
        "best_val_loss": meta.get("best_val_loss"),
        "feature_dim":   meta.get("feature_dim", FEATURE_DIM),
        "n_mels":        meta.get("n_mels", 128),
        "n_frames":      meta.get("n_frames", 128),
    }


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_with_cnn(audio_bytes: bytes, filename: str) -> dict:
    """
    Run CNN inference on uploaded audio bytes.

    Parameters
    ----------
    audio_bytes : Raw bytes of the uploaded audio file (.wav or .mp3).
    filename    : Original filename — used to pick the correct temp file suffix
                  so soundfile / librosa can identify the format.

    Returns
    -------
    dict:
        prediction        : str   — human-readable class name
        confidence_scores : dict  — {class_name: probability}  sums to 1.0
        duration_sec      : float — audio duration in seconds
        inference_ms      : float — total inference time in milliseconds

    Raises
    ------
    RuntimeError   if the model has not been loaded.
    ValueError     if the audio is unreadable, too short, or silent.
    """
    if _cnn_model is None or _label_map is None:
        raise RuntimeError("CNN model is not loaded. Call load_cnn() first.")

    # ── 1. Write bytes to a temp file ─────────────────────────────────────────
    ext = os.path.splitext(filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # ── 2. Validate audio readability and duration ────────────────────────
        try:
            info         = sf.info(tmp_path)
            duration_sec = float(info.duration)
        except Exception as e:
            raise ValueError(f"Cannot read audio file '{filename}': {e}") from e

        if duration_sec < 0.5:
            raise ValueError(
                f"Audio is too short ({duration_sec:.2f}s). "
                "Please upload at least 0.5 seconds of engine audio."
            )
        if duration_sec > 180.0:
            raise ValueError(
                f"Audio is too long ({duration_sec:.1f}s). "
                "Maximum supported clip length is 180 seconds for low-memory inference."
            )

        # ── 3. Extract 16 384-dim log-mel features (windowed if > MAX_DURATION) ─
        try:
            if duration_sec <= MAX_DURATION:
                feature_batch = [extract_features(tmp_path)]
            else:
                y, sr = load_audio(tmp_path)
                if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
                    raise ValueError(f"Audio file appears to be silent: '{filename}'")
                win_samples = int(sr * MAX_DURATION)
                hop_samples = max(1, win_samples // 2)
                feature_batch = []
                start = 0
                while start < len(y):
                    end = min(start + win_samples, len(y))
                    seg = y[start:end]
                    if len(seg) >= int(0.5 * sr) and np.max(np.abs(seg)) >= 1e-6:
                        seg_fixed = clip_or_pad(seg, sr)
                        feature_batch.append(compute_log_mel(seg_fixed, sr).flatten())
                    if end == len(y):
                        break
                    start += hop_samples
                if not feature_batch:
                    raise ValueError(f"All windows were silent in '{filename}'")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Feature extraction failed for '{filename}': {e}") from e

    finally:
        # Always delete the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── 4. CNN inference (batched across windows) ─────────────────────────────
    t0 = time.perf_counter()

    x = torch.from_numpy(np.stack(feature_batch)).float()   # (W, 16384)

    with torch.no_grad():
        logits = _cnn_model(x)                              # (W, n_classes)
        probs  = torch.softmax(logits, dim=1).mean(dim=0)   # (n_classes,)

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── 5. Map index → class name ─────────────────────────────────────────────
    pred_idx   = int(probs.argmax().item())
    prediction = _label_map[pred_idx]

    # Build confidence_scores dict: {class_name: rounded_probability}
    confidence_scores = {
        _label_map[i]: round(float(probs[i].item()), 4)
        for i in range(len(_label_map))
    }

    return {
        "prediction":        prediction,
        "confidence_scores": confidence_scores,
        "duration_sec":      round(duration_sec, 2),
        "inference_ms":      inference_ms,
    }
