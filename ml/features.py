"""
ml/features.py
==============
AutoGuard AI — audio preprocessing pipeline.

WHAT THIS MODULE DOES
---------------------
1. Loads audio from two dataset sources:
     • data/kwerimoto/prepared/   (pre-split WAV files, three classes)
     • data/kaggledataset/car diagnostics dataset/   (raw WAV files, mapped to classes)

2. Converts every audio clip to a log-mel spectrogram:
     • Resample to TARGET_SR (22 050 Hz)
     • Convert to mono
     • Clip to MAX_DURATION (10 s), pad shorter clips with silence
     • Compute mel spectrogram  (N_MELS = 128 bands, HOP = 512, WIN = 2048)
     • Convert power → dB  (log-mel)
     • Normalise to [-1, +1] per clip
     • Pad / truncate time axis to exactly N_FRAMES = 128 columns
     → shape: (128, 128) = (N_MELS, N_FRAMES)

3. Flattens to a 1-D feature vector of length N_MELS × N_FRAMES = 16 384.
   This is what extract_features() returns, and what train.py feeds to the
   SVM / RandomForest pipeline.

4. Provides three data-augmentation functions that return new audio arrays
   (seed-controlled so they are reproducible):
     • add_white_noise(y, snr_db)   — additive Gaussian noise at a given SNR
     • time_stretch(y, rate)        — speed up / slow down without pitch change
     • pitch_shift(y, sr, n_steps)  — shift pitch by n semitones

5. Provides prepare_dataset() which:
     • Scans both data sources
     • Builds train / validation / test splits from kwerimoto
       (the kaggle data is added to the training split only)
     • Applies augmentation to the training set (×4: original + 3 augmented)
     • Saves NumPy arrays and metadata to data/prepared/:
         data/prepared/train/X.npy    shape (N_train, 16384)
         data/prepared/train/y.npy    shape (N_train,)   int labels
         data/prepared/train/meta.json
         data/prepared/validation/...
         data/prepared/test/...
         data/prepared/label_map.json  int→class-name mapping

   train.py can call load_prepared_split() instead of re-extracting features
   from scratch every run.

USAGE
-----
# Single-file inference (called by backend and test scripts):
    from ml.features import extract_features
    feat = extract_features("path/to/clip.wav")  # shape (16384,)

# Batch preparation (run once):
    python ml/features.py

# In train.py — load pre-computed arrays:
    from ml.features import load_prepared_split
    X_train, y_train = load_prepared_split("train")

CONSTANTS (must stay in sync with the saved model)
-----------
    TARGET_SR   = 22050
    MAX_DURATION= 10.0
    N_MELS      = 128
    N_FRAMES    = 128
    FMAX        = 8000
    HOP_LENGTH  = 512
    WIN_LENGTH  = 2048
    FEATURE_DIM = N_MELS * N_FRAMES = 16384

KAGGLE LABEL MAPPING
--------------------
The kaggle dataset has subdirectories that do not exactly match our three
classes (healthy, engine_knock, misfire). The mapping below is deliberate
and conservative — only folders with a clear, unambiguous relationship to
one of our three classes are included. Ambiguous folders are skipped with
a warning so we never silently mislabel data.

    normal_engine_idle    → healthy
    normal_brakes         → healthy
    normal_engine_startup → healthy
    low_oil               → misfire   (low-oil causes rough running/misfire)
    serpentine_belt       → misfire   (accessory-belt noise is a mechanical fault)
    bad_ignition          → misfire   (failed ignition → misfire)

Skipped (too ambiguous or not matching our task):
    combined/*            — mixed fault combinations
    worn_out_brakes       — braking fault, not engine condition
    dead_battery          — electrical, not engine running condition
    power_steering        — hydraulic fault, not engine condition
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

warnings.filterwarnings("ignore")

# ── Root paths ─────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).resolve().parent
_ROOT       = _MODULE_DIR.parent

KWERIMOTO_PREPARED = _ROOT / "data" / "kwerimoto" / "prepared"
KAGGLE_ROOT        = _ROOT / "data" / "kaggledataset" / "car diagnostics dataset"
PREPARED_OUT       = _ROOT / "data" / "prepared"

# ── Audio / spectrogram constants ──────────────────────────────────────────────
TARGET_SR    : int   = 22_050   # resample everything to this rate
MAX_DURATION : float = 10.0     # seconds — clip / pad to this length
N_MELS       : int   = 128      # mel filterbank bands
N_FRAMES     : int   = 128      # fixed time-axis length after pad/truncate
FMAX         : int   = 8_000    # max frequency for mel filterbank
HOP_LENGTH   : int   = 512      # STFT hop in samples
WIN_LENGTH   : int   = 2_048    # STFT window in samples
FEATURE_DIM  : int   = N_MELS * N_FRAMES   # = 16 384

# ── Augmentation defaults ──────────────────────────────────────────────────────
AUG_NOISE_SNR_DB : float = 25.0   # white noise SNR for augmentation
AUG_STRETCH_RATE : float = 0.9    # time-stretch factor (0.9 = 10% slower)
AUG_PITCH_STEPS  : int   = 2      # semitones for pitch shift

# ── Kaggle folder → class label ───────────────────────────────────────────────
# Only unambiguous folders included. See module docstring for rationale.
KAGGLE_LABEL_MAP: dict[str, str] = {
    "normal_engine_idle":    "healthy",
    "normal_brakes":         "healthy",
    "normal_engine_startup": "healthy",
    "low_oil":               "misfire",
    "serpentine_belt":       "misfire",
    "bad_ignition":          "misfire",
}

KAGGLE_SKIP_FOLDERS = {
    "combined", "worn_out_brakes", "dead_battery", "power_steering",
}

# ── Supported classes ─────────────────────────────────────────────────────────
CLASSES         = ["engine_knock", "healthy", "misfire"]
CLASS_TO_INT    = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS    = {i: c for i, c in enumerate(CLASSES)}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Core preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample to TARGET_SR, convert to mono.

    Returns
    -------
    y  : np.ndarray float32, shape (n_samples,)
    sr : int — always TARGET_SR
    """
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return y.astype(np.float32), sr


def clip_or_pad(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Truncate audio longer than MAX_DURATION, or zero-pad if shorter.
    Returns array of exactly int(sr * MAX_DURATION) samples.
    """
    target_len = int(sr * MAX_DURATION)
    if len(y) >= target_len:
        return y[:target_len]
    # Pad with silence on the right
    return np.pad(y, (0, target_len - len(y)), mode="constant")


def compute_log_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a normalised log-mel spectrogram.

    Steps:
      1. mel spectrogram   → shape (N_MELS, T)
      2. power_to_db       → log scale,  range ≈ [-80, 0] dB
      3. normalise per clip → range [-1, +1]
      4. pad / truncate T  → exactly N_FRAMES columns

    Returns
    -------
    np.ndarray float32, shape (N_MELS, N_FRAMES)
    """
    # Step 1 — mel spectrogram
    mel = librosa.feature.melspectrogram(
        y          = y,
        sr         = sr,
        n_mels     = N_MELS,
        fmax       = FMAX,
        hop_length = HOP_LENGTH,
        n_fft      = WIN_LENGTH,
    )   # shape: (N_MELS, T)  dtype: float32

    # Step 2 — log scale
    log_mel = librosa.power_to_db(mel, ref=np.max)   # range ≈ [-80, 0]

    # Step 3 — per-clip normalise to [-1, +1]
    vmin, vmax = log_mel.min(), log_mel.max()
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1.0
    log_mel = 2.0 * (log_mel - vmin) / denom - 1.0   # → [-1, +1]

    # Step 4 — fixed time axis
    T = log_mel.shape[1]
    if T >= N_FRAMES:
        log_mel = log_mel[:, :N_FRAMES]
    else:
        pad_w   = N_FRAMES - T
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_w)),
                         mode="constant", constant_values=-1.0)

    return log_mel.astype(np.float32)   # shape: (N_MELS, N_FRAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Public feature extraction (called by train.py and backend)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(audio_path: str | Path) -> np.ndarray:
    """
    Load an audio file and return a flat 1-D feature vector.

    This is the SINGLE entry point used by both training (ml/train.py)
    and serving (backend/model.py). Both paths must call this function
    with no extra preprocessing so train/serve are identical.

    Parameters
    ----------
    audio_path : str or Path
        Path to a WAV or MP3 file.

    Returns
    -------
    np.ndarray float32, shape (FEATURE_DIM,) = (16384,)
        Flattened normalised log-mel spectrogram.

    Raises
    ------
    ValueError
        If the file cannot be loaded, is empty, or is silent.
    """
    path = str(audio_path)

    # Load
    try:
        y, sr = load_audio(path)
    except Exception as exc:
        raise ValueError(f"Could not load '{path}': {exc}") from exc

    if len(y) == 0:
        raise ValueError(f"Audio file is empty: '{path}'")
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError(f"Audio file appears to be silent: '{path}'")

    # Clip / pad → fixed length
    y = clip_or_pad(y, sr)

    # Log-mel spectrogram → (N_MELS, N_FRAMES)
    log_mel = compute_log_mel(y, sr)

    # Flatten → (FEATURE_DIM,)
    return log_mel.flatten()


def get_feature_config() -> dict:
    """Return feature configuration dict (saved with trained models)."""
    return {
        "target_sr":          TARGET_SR,
        "max_duration_sec":   MAX_DURATION,
        "n_mels":             N_MELS,
        "n_frames":           N_FRAMES,
        "fmax":               FMAX,
        "hop_length":         HOP_LENGTH,
        "win_length":         WIN_LENGTH,
        "feature_dim":        FEATURE_DIM,
        "representation":     "log_mel_spectrogram_flattened",
        "normalisation":      "per_clip_minmax_to_minus1_plus1",
        "version":            "v3",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Data augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def add_white_noise(
    y: np.ndarray,
    snr_db: float = AUG_NOISE_SNR_DB,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add white Gaussian noise at a given signal-to-noise ratio.

    Parameters
    ----------
    y      : audio array, float32
    snr_db : signal-to-noise ratio in decibels (higher = less noise)
    rng    : numpy random Generator for reproducibility

    Returns
    -------
    np.ndarray float32 — noisy audio, same length as y
    """
    if rng is None:
        rng = np.random.default_rng(42)

    signal_power = np.mean(y ** 2)
    if signal_power < 1e-12:
        return y.copy()

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise       = rng.normal(0.0, np.sqrt(noise_power), size=len(y)).astype(np.float32)
    return np.clip(y + noise, -1.0, 1.0)


def time_stretch(
    y: np.ndarray,
    rate: float = AUG_STRETCH_RATE,
) -> np.ndarray:
    """
    Time-stretch audio without changing pitch.

    Parameters
    ----------
    y    : audio array, float32
    rate : stretch factor
           < 1.0 → slower (e.g. 0.9 = 10% longer)
           > 1.0 → faster (e.g. 1.1 = 10% shorter)

    Returns
    -------
    np.ndarray float32 — stretched audio (different length than input)

    Note: librosa.effects.time_stretch may raise for very short clips.
    The result is clipped / padded to match the original length by
    the calling code if needed.
    """
    return librosa.effects.time_stretch(y=y.astype(np.float32), rate=rate)


def pitch_shift(
    y: np.ndarray,
    sr: int = TARGET_SR,
    n_steps: int = AUG_PITCH_STEPS,
) -> np.ndarray:
    """
    Shift audio pitch by n_steps semitones without changing duration.

    Parameters
    ----------
    y       : audio array, float32
    sr      : sample rate
    n_steps : semitones to shift (positive = up, negative = down)

    Returns
    -------
    np.ndarray float32 — pitch-shifted audio, same length as y
    """
    return librosa.effects.pitch_shift(
        y=y.astype(np.float32), sr=sr, n_steps=float(n_steps)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Dataset scanning
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_kwerimoto_split(split: str) -> list[tuple[Path, str]]:
    """
    Return list of (file_path, label) for one kwerimoto split.
    split is one of: train, validation, test, real_test
    """
    split_dir = KWERIMOTO_PREPARED / split
    if not split_dir.exists():
        print(f"  [WARN] kwerimoto split not found: {split_dir}")
        return []

    result = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        if label not in CLASSES:
            print(f"  [WARN] Unknown class folder '{label}' in {split_dir} — skipped")
            continue
        wavs = sorted(class_dir.glob("*.wav"))
        result.extend((p, label) for p in wavs)

    return result


def _scan_kaggle() -> list[tuple[Path, str]]:
    """
    Recursively scan the kaggle dataset directory and return
    (file_path, label) for every file whose parent folder name
    appears in KAGGLE_LABEL_MAP.
    """
    if not KAGGLE_ROOT.exists():
        print(f"  [WARN] Kaggle dataset not found: {KAGGLE_ROOT}")
        return []

    result    = []
    skipped   = 0
    ambiguous = set()

    for wav_path in sorted(KAGGLE_ROOT.rglob("*.wav")):
        folder_name = wav_path.parent.name

        # Skip combined or other explicitly excluded folders
        if folder_name in KAGGLE_SKIP_FOLDERS:
            skipped += 1
            ambiguous.add(folder_name)
            continue

        # Walk up parent names to find a mapped label (some files are nested)
        label = None
        for part in wav_path.parts:
            if part in KAGGLE_LABEL_MAP:
                label = KAGGLE_LABEL_MAP[part]
                break
        if label is None:
            if folder_name not in KAGGLE_SKIP_FOLDERS:
                ambiguous.add(folder_name)
            skipped += 1
            continue

        result.append((wav_path, label))

    if ambiguous:
        print(f"  [WARN] Kaggle folders skipped (ambiguous/excluded): {sorted(ambiguous)}")
    print(f"  Kaggle: {len(result)} files mapped, {skipped} skipped")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Feature extraction with augmentation
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_one(
    path: Path,
    label: str,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> list[tuple[np.ndarray, int]]:
    """
    Extract features from one audio file.
    If augment=True, also generates 3 augmented versions:
      1. White noise injection
      2. Time stretch (rate = AUG_STRETCH_RATE)
      3. Pitch shift  (n_steps = AUG_PITCH_STEPS)

    Returns list of (feature_vector, int_label) tuples.
    Original is always first; augmented follow if augment=True.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    label_int = CLASS_TO_INT[label]
    result    = []

    try:
        y, sr = load_audio(path)
    except Exception as exc:
        print(f"    SKIP (load error): {path.name} — {exc}")
        return result

    if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
        print(f"    SKIP (empty/silent): {path.name}")
        return result

    # Original
    y_fixed  = clip_or_pad(y, sr)
    log_mel  = compute_log_mel(y_fixed, sr)
    result.append((log_mel.flatten(), label_int))

    if not augment:
        return result

    # Augmentation 1 — white noise
    try:
        y_noisy = add_white_noise(y_fixed, snr_db=AUG_NOISE_SNR_DB, rng=rng)
        result.append((compute_log_mel(y_noisy, sr).flatten(), label_int))
    except Exception as exc:
        print(f"    AUG noise failed for {path.name}: {exc}")

    # Augmentation 2 — time stretch
    try:
        y_stretched = time_stretch(y, rate=AUG_STRETCH_RATE)
        y_stretched = clip_or_pad(y_stretched, sr)
        result.append((compute_log_mel(y_stretched, sr).flatten(), label_int))
    except Exception as exc:
        print(f"    AUG stretch failed for {path.name}: {exc}")

    # Augmentation 3 — pitch shift
    try:
        y_pitched = pitch_shift(y_fixed, sr=sr, n_steps=AUG_PITCH_STEPS)
        result.append((compute_log_mel(y_pitched, sr).flatten(), label_int))
    except Exception as exc:
        print(f"    AUG pitch failed for {path.name}: {exc}")

    return result


def _process_file_list(
    files: list[tuple[Path, str]],
    augment: bool,
    rng: np.random.Generator,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from a list of (path, label) pairs."""
    X_list, y_list = [], []
    total  = len(files)
    errors = 0

    for i, (path, label) in enumerate(files, 1):
        if i % 100 == 0 or i == total:
            print(f"    {i}/{total} ...", end="\r", flush=True)

        samples = _extract_one(path, label, augment=augment, rng=rng)
        if not samples:
            errors += 1
            continue
        for feat, lbl in samples:
            X_list.append(feat)
            y_list.append(lbl)

    print()   # newline after \r progress
    if errors:
        print(f"    WARNING: {errors} files skipped in {split_name}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Batch preparation (saves to data/prepared/)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    augment_train: bool = True,
    seed: int           = 42,
    include_kaggle: bool = True,
) -> None:
    """
    Scan both data sources, extract log-mel features, apply augmentation to
    the training set, and save NumPy arrays to data/prepared/.

    Directory layout created:
        data/prepared/
            label_map.json         int → class name
            feature_config.json    preprocessing constants
            train/
                X.npy   float32  (N_train, FEATURE_DIM)
                y.npy   int32    (N_train,)
                meta.json
            validation/
                X.npy   float32  (N_val, FEATURE_DIM)
                y.npy   int32    (N_val,)
                meta.json
            test/
                X.npy   float32  (N_test, FEATURE_DIM)
                y.npy   int32    (N_test,)
                meta.json

    Parameters
    ----------
    augment_train  : Apply 3 augmentation passes to training data (×4 total).
    seed           : Random seed for all augmentation functions.
    include_kaggle : Include kaggle dataset files in the training split.
    """
    rng = np.random.default_rng(seed)
    PREPARED_OUT.mkdir(parents=True, exist_ok=True)

    # ── Collect file lists ────────────────────────────────────────────────────
    print("\n[1/4] Scanning data sources ...")

    train_files = _scan_kwerimoto_split("train")
    val_files   = _scan_kwerimoto_split("validation")
    test_files  = _scan_kwerimoto_split("test")

    if include_kaggle:
        print("  Scanning kaggle dataset ...")
        kaggle_files = _scan_kaggle()
        # Kaggle data goes into training only — never validation or test
        train_files  = train_files + kaggle_files

    print(f"\n  train    : {len(train_files)} files"
          f"  {'(×4 with augmentation)' if augment_train else ''}")
    print(f"  validation: {len(val_files)} files")
    print(f"  test      : {len(test_files)} files")

    # Print class balance
    for split_name, flist in [("train", train_files),
                               ("validation", val_files),
                               ("test", test_files)]:
        from collections import Counter
        counts = Counter(lbl for _, lbl in flist)
        print(f"  {split_name:12s} class counts: "
              + "  ".join(f"{c}={counts.get(c,0)}" for c in CLASSES))

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n[2/4] Extracting features (training) ...")
    X_train, y_train = _process_file_list(
        train_files, augment=augment_train, rng=rng, split_name="train"
    )
    print(f"  X_train shape: {X_train.shape}  y_train shape: {y_train.shape}")

    print("\n[3/4] Extracting features (validation & test) ...")
    X_val,  y_val  = _process_file_list(val_files,  augment=False, rng=rng, split_name="validation")
    X_test, y_test = _process_file_list(test_files, augment=False, rng=rng, split_name="test")
    print(f"  X_val  shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n[4/4] Saving arrays to data/prepared/ ...")

    def _save(X: np.ndarray, y: np.ndarray, split: str, source_count: int) -> None:
        out = PREPARED_OUT / split
        out.mkdir(parents=True, exist_ok=True)
        np.save(str(out / "X.npy"), X)
        np.save(str(out / "y.npy"), y)
        from collections import Counter
        class_counts = {INT_TO_CLASS[i]: int(v) for i, v in Counter(y.tolist()).items()}
        meta = {
            "split":         split,
            "n_samples":     int(len(X)),
            "feature_dim":   FEATURE_DIM,
            "augmented":     augment_train if split == "train" else False,
            "seed":          seed,
            "class_counts":  class_counts,
            "source_files":  source_count,
        }
        with open(str(out / "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  → {out}/  ({len(X)} samples)")

    _save(X_train, y_train, "train",      len(train_files))
    _save(X_val,   y_val,   "validation", len(val_files))
    _save(X_test,  y_test,  "test",       len(test_files))

    # Global label map and feature config
    with open(str(PREPARED_OUT / "label_map.json"), "w") as f:
        json.dump(INT_TO_CLASS, f, indent=2)
    with open(str(PREPARED_OUT / "feature_config.json"), "w") as f:
        json.dump(get_feature_config(), f, indent=2)

    print(f"\n  label_map.json and feature_config.json saved.")
    print(f"  Done. Output → {PREPARED_OUT}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Load pre-computed arrays (called by train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_prepared_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed feature arrays from data/prepared/<split>/.

    Parameters
    ----------
    split : "train" | "validation" | "test"

    Returns
    -------
    X : np.ndarray float32, shape (N, FEATURE_DIM)
    y : np.ndarray int32,   shape (N,)   — integer class labels

    Raises
    ------
    FileNotFoundError if prepare_dataset() has not been run yet.
    """
    split_dir = PREPARED_OUT / split
    x_path    = split_dir / "X.npy"
    y_path    = split_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Pre-computed arrays not found in {split_dir}. "
            "Run:  python ml/features.py"
        )

    X = np.load(str(x_path))
    y = np.load(str(y_path))

    with open(str(split_dir / "meta.json")) as f:
        meta = json.load(f)

    print(f"  Loaded {split}: {X.shape[0]} samples, {X.shape[1]} features  "
          f"(augmented={meta.get('augmented', False)})")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare AutoGuard AI dataset: extract log-mel features "
                    "from both data sources and save NumPy arrays."
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Skip data augmentation on the training set.",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Skip the kaggle dataset (use kwerimoto only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for augmentation (default: 42).",
    )
    args = parser.parse_args()

    prepare_dataset(
        augment_train  = not args.no_augment,
        seed           = args.seed,
        include_kaggle = not args.no_kaggle,
    )
