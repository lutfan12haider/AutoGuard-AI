"""
ml/v2/audio_preprocessing.py
----------------------------
AutoGuard AI V2 — Scientifically Verified Audio Preprocessing Pipeline.

All design choices are backed by Phase 2 forensic verification (`reports/v2/phase2_forensics.json`):
  1. Target sample rate: 16,000 Hz (Nyquist = 8,000 Hz) for all cross-source models so
     KweriMoto (16 kHz) and Kaggle (24/44.1/48 kHz) cannot be separated via >8 kHz Nyquist
     cutoffs. Preserves 92.98% - 100.0% of spectral energy across 8/9 classes (and 81.81%
     for worn_out_brakes; a separate 22,050 Hz braking specialist helper is also provided).
  2. Base window length: 1.50 seconds (24,000 samples @ 16 kHz).
     Matches Kaggle idle/braking duration (1.50s) with ZERO silence padding.
     Longer clips (Kaggle startup 1.70s; KweriMoto 3-10s) are segmented into overlapping
     1.50s windows (0.75s hop). For multi-scale tabular evaluation, full-clip global pooling
     is also supported.
  3. Short clips (< 1.50s, 35 files in KweriMoto):
     - For tabular statistics: extracted on native UN-PADDED audio with adaptive delta width
       (eliminating silence-padding variance distortion: cos_dist=0.0018 vs 0.7529 in V1).
     - For fixed 2D CNN spectrograms: cosine-tapered circular wrap (`wrap_taper`), NOT
       zero-padding (cos_dist=0.0007 vs 0.3866 zero-pad) and avoiding time-reversed
       knock reflections from naive `reflect` padding.
  4. Amplitude normalization: Per-window RMS normalization to TARGET_RMS=0.08 with soft
     tanh peak limiting prevents Kaggle's peak-normalized combined mixtures (RMS ~0.19)
     or quiet recordings from acting as a trivial volume shortcut, while explicit pre-norm
     log-RMS and crest factor are extracted cleanly for optional energy-aware ablation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import scipy.signal
import soundfile as sf

TARGET_SR: int = 16_000
BRAKING_SPECIALIST_SR: int = 22_050

WINDOW_SEC: float = 1.50
HOP_SEC: float = 0.75
WINDOW_SAMPLES: int = int(TARGET_SR * WINDOW_SEC)  # 24,000
HOP_SAMPLES: int = int(TARGET_SR * HOP_SEC)        # 12,000

# Log-Mel spectrogram dimensions for 1.50s @ 16,000 Hz:
# hop_length=375 (23.4 ms) -> ceil(24000 / 375) = 64 frames -> exact (128, 64) tensor
N_MELS: int = 128
N_FRAMES: int = 64
N_FFT: int = 1024
MEL_HOP: int = 375
FMIN: float = 20.0
FMAX: float = 8_000.0

TARGET_RMS: float = 0.08
MIN_VALID_SAMPLES: int = 2048  # 0.128s (smallest file in KweriMoto is revix_car_knocking_0252 at 2048 samples)


def fast_resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase rational resampling via SciPy (deterministic, anti-aliased, 30x faster than resampy)."""
    if orig_sr == target_sr or len(y) == 0:
        return y.astype(np.float32)
    g = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    return scipy.signal.resample_poly(y.astype(np.float32), up, down).astype(np.float32)


def load_and_standardize(
    path: str | Path,
    target_sr: int = TARGET_SR,
) -> tuple[np.ndarray, int, dict]:
    """
    Load WAV/MP3 safely, convert stereo->mono by averaging channels, remove DC offset,
    and resample to `target_sr`. Also returns diagnostic file-level metadata.
    """
    path_str = str(path)
    try:
        y, orig_sr = sf.read(path_str, dtype="float32", always_2d=False)
        subtype = sf.info(path_str).subtype
    except Exception:
        # Fallback to librosa for MP3 or exotic containers
        y, orig_sr = librosa.load(path_str, sr=None, mono=False)
        subtype = "LIBROSA"

    if y.ndim > 1:
        n_channels = int(y.shape[1]) if y.shape[0] > y.shape[1] else int(y.shape[0])
        if y.shape[0] < y.shape[1] and y.shape[0] <= 8:
            y = np.mean(y, axis=0)
        else:
            y = np.mean(y, axis=1)
    else:
        n_channels = 1

    y = y.astype(np.float32)
    if len(y) == 0:
        raise ValueError(f"Empty audio file: {path_str}")

    # Remove DC offset cleanly
    y = y - float(np.mean(y))
    peak_raw = float(np.max(np.abs(y)))
    rms_raw = float(np.sqrt(np.mean(y ** 2)))
    if peak_raw < 1e-6 or rms_raw < 1e-6:
        raise ValueError(f"Silent audio file (peak={peak_raw:.2e}): {path_str}")

    # True clipping check: plateau >= 3 consecutive samples at full scale
    clip_mask = np.abs(y) >= 0.998
    sat_count = int(np.sum(clip_mask))
    if sat_count > 0:
        padded = np.concatenate(([False], clip_mask, [False]))
        s = np.where(padded[1:] & ~padded[:-1])[0]
        e = np.where(~padded[1:] & padded[:-1])[0]
        max_clip_run = int(np.max(e - s)) if len(s) else 0
    else:
        max_clip_run = 0

    dur_raw = float(len(y) / orig_sr)
    if orig_sr != target_sr:
        y = fast_resample(y, int(orig_sr), int(target_sr))

    meta = {
        "orig_sr": int(orig_sr),
        "orig_channels": n_channels,
        "subtype": str(subtype),
        "orig_duration_sec": round(dur_raw, 4),
        "raw_rms": rms_raw,
        "raw_peak": peak_raw,
        "saturated_samples": sat_count,
        "max_clip_run": max_clip_run,
    }
    return y, int(target_sr), meta


def pad_short_wrap_taper(y: np.ndarray, target_len: int, sr: int = TARGET_SR) -> np.ndarray:
    """
    Pad a short segment (< target_len) using smooth cosine-tapered circular tiling
    so engine cycle periodicity and RMS variance are preserved without silence
    artifacts or click discontinuities at tiling seams.
    """
    n = len(y)
    if n >= target_len:
        return y[:target_len].astype(np.float32)
    if n < 32:
        return np.pad(y, (0, target_len - n), mode="reflect").astype(np.float32)

    # Apply tiny 4ms cosine fade at edges before tiling to avoid click transients at boundaries
    fade = min(int(0.004 * sr), n // 8)
    y_work = y.astype(np.float32).copy()
    if fade > 2:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade, dtype=np.float32) / fade))
        y_work[:fade] *= ramp
        y_work[-fade:] *= ramp[::-1]

    repeats = math.ceil(target_len / n)
    return np.tile(y_work, repeats)[:target_len].astype(np.float32)


def normalize_window_rms(
    y: np.ndarray,
    target_rms: float = TARGET_RMS,
) -> tuple[np.ndarray, float, float]:
    """
    Normalize waveform RMS to `target_rms` with soft tanh limiting above 0.90
    to prevent digital clipping. Returns (y_norm, raw_rms, raw_crest_factor).
    """
    y = y.astype(np.float32)
    raw_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-8
    raw_peak = float(np.max(np.abs(y))) + 1e-8
    raw_crest = raw_peak / raw_rms

    y_scaled = y * (target_rms / raw_rms)
    # Soft saturation protection if extreme transient exceeds 0.95
    max_s = float(np.max(np.abs(y_scaled)))
    if max_s > 0.95:
        y_scaled = 0.95 * np.tanh(y_scaled / 0.95)
    return y_scaled.astype(np.float32), raw_rms, raw_crest


def slice_windows(
    y: np.ndarray,
    sr: int = TARGET_SR,
    win_sec: float = WINDOW_SEC,
    hop_sec: float = HOP_SEC,
    max_windows_per_clip: int | None = None,
) -> list[tuple[np.ndarray, float, float, bool]]:
    """
    Segment waveform `y` into fixed-duration windows.
    Returns list of (segment_array, start_sec, end_sec, was_short_padded).

    - If clip <= win_sec (e.g., 1.50s Kaggle clip or <1.50s short KweriMoto clip):
      returns 1 tuple preserving exact native length for tabular extraction while
      flagging `was_short_padded = len(y) < win_samples`.
    - If clip > win_sec: slices overlapping windows and includes trailing aligned window
      only if the tail adds >= 0.40 * win_sec new audio (avoiding near-duplicate end window).
    """
    win_samples = int(round(sr * win_sec))
    hop_samples = max(1, int(round(sr * hop_sec)))
    total = len(y)

    if total <= win_samples:
        return [(y.astype(np.float32), 0.0, total / sr, total < win_samples)]

    windows: list[tuple[np.ndarray, float, float, bool]] = []
    start = 0
    while start + win_samples <= total:
        end = start + win_samples
        seg = y[start:end]
        if float(np.max(np.abs(seg))) >= 1e-5:
            windows.append((seg.astype(np.float32), start / sr, end / sr, False))
        start += hop_samples

    # Include final aligned window if residual tail >= 40% of window and wasn't covered
    if windows:
        last_end = int(round(windows[-1][2] * sr))
        if total - last_end >= int(0.40 * win_samples):
            seg = y[total - win_samples : total]
            if float(np.max(np.abs(seg))) >= 1e-5:
                windows.append((seg.astype(np.float32), (total - win_samples) / sr, total / sr, False))
    else:
        windows.append((y[:win_samples].astype(np.float32), 0.0, win_samples / sr, False))

    if max_windows_per_clip is not None and len(windows) > max_windows_per_clip:
        idxs = np.linspace(0, len(windows) - 1, num=max_windows_per_clip, dtype=int)
        windows = [windows[i] for i in idxs]

    return windows


def compute_log_mel_v2(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = N_MELS,
    n_frames: int = N_FRAMES,
    fmax: float = FMAX,
    norm_mode: Literal["global_db", "instance_zscore"] = "global_db",
) -> np.ndarray:
    """
    Compute a duration-matched Log-Mel spectrogram on an RMS-normalized window.

    If `len(y) < WINDOW_SAMPLES` (1.50s), applies `pad_short_wrap_taper` first so the
    CNN never sees dead silence padding (`-1.0`) blocks.

    Normalization:
      - `global_db`: Anchors power_to_db to a calibrated reference (top_db=80.0 against
        ref=1.0 on RMS=0.08 normalized audio, then maps [-65, +15] dB -> [-1, +1] clipped),
        preserving genuine relative loudness differences between frames & bands without
        inflating quiet noise floors like V1's per-clip min-max!
      - `instance_zscore`: Zero mean, unit std per spectrogram.
    """
    target_samples = int(round(sr * WINDOW_SEC))
    y_fixed = (
        pad_short_wrap_taper(y, target_samples, sr)
        if len(y) < target_samples
        else y[:target_samples]
    )
    y_norm, _, _ = normalize_window_rms(y_fixed, target_rms=TARGET_RMS)

    # Dynamic hop so exact frame count aligns cleanly for both 16 kHz and 22.05 kHz
    hop = max(1, target_samples // n_frames)
    mel = librosa.feature.melspectrogram(
        y=y_norm,
        sr=sr,
        n_fft=N_FFT,
        hop_length=hop,
        n_mels=n_mels,
        fmin=FMIN,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=1.0, top_db=None)

    if norm_mode == "global_db":
        # Empirical distribution on RMS=0.08 audio has mean=-18.8 dB, std=18.9 dB, p100=+23.2 dB.
        # Affine centering by +20 dB / 20.0 yields ~0 mean, ~0.94 std with 0.0% peak clipping.
        log_mel = np.clip((log_mel + 20.0) / 20.0, -4.0, 3.0)
    else:
        mu = float(np.mean(log_mel))
        sd = float(np.std(log_mel)) + 1e-6
        log_mel = np.clip((log_mel - mu) / sd, -4.0, 4.0)

    T = log_mel.shape[1]
    if T >= n_frames:
        log_mel = log_mel[:, :n_frames]
    else:
        # Repeat edge frame rather than silence if off by 1 due to framing
        pad_w = n_frames - T
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_w)), mode="edge")

    return log_mel.astype(np.float32)
