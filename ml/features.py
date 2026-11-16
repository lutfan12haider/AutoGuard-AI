"""
features.py
-----------
Audio feature extraction for AutoGuard AI — v2 (expanded features).

Feature vector breakdown (mean + std pooling on each):
  - 40 MFCCs                         →  80 dims  (timbre / spectral envelope)
  - 40 Delta MFCCs                   →  80 dims  (rate of change — captures rhythmic knock patterns)
  - 40 Delta-Delta MFCCs             →  80 dims  (acceleration — sharp transients in knock/misfire)
  - 6  Spectral contrast bands       →  12 dims  (peak vs valley energy — knock has sharp spikes)
  - 12 Chroma features               →  24 dims  (harmonic content differs across conditions)
  - 1  Zero crossing rate            →   2 dims  (tick-heavy knock has high ZCR)
  - 1  RMS energy                    →   2 dims  (overall signal loudness)
  - 1  Spectral centroid             →   2 dims  (brightness — knock shifts spectrum up)
  - 1  Spectral rolloff              →   2 dims  (frequency below which 85% of energy sits)
  - 1  Spectral bandwidth            →   2 dims  (spread of energy around centroid)
  ─────────────────────────────────────────────
  Total                              → 286 dims

This exact function is used during BOTH training and inference.
Changing it requires retraining the model.
"""

import numpy as np
import librosa

# ── Feature configuration ─────────────────────────────────────────────────────
SAMPLE_RATE   = 22050   # Hz — all audio resampled to this
MAX_DURATION  = 10.0    # seconds — clips truncated here
N_MFCC        = 40      # MFCC coefficients
N_CHROMA      = 12      # chroma bins
N_CONTRAST    = 6       # spectral contrast bands (librosa default = 6 bands + 1 = 7, we use 6)

# Pre-computed: each feature × 2 (mean+std)
FEATURE_DIM = (
    N_MFCC * 2 +        # MFCCs
    N_MFCC * 2 +        # delta MFCCs
    N_MFCC * 2 +        # delta-delta MFCCs
    (N_CONTRAST + 1) * 2 +  # spectral contrast (6 bands + 1 mean = 7 values)
    N_CHROMA * 2 +      # chroma
    1 * 2 +             # ZCR
    1 * 2 +             # RMS
    1 * 2 +             # spectral centroid
    1 * 2 +             # spectral rolloff
    1 * 2               # spectral bandwidth
)
# = 80 + 80 + 80 + 14 + 24 + 2 + 2 + 2 + 2 + 2 = 288


def _pool(x: np.ndarray) -> np.ndarray:
    """Return [mean, std] pooled across time axis (axis=1 for 2D, axis=0 for 1D)."""
    if x.ndim == 1:
        return np.array([x.mean(), x.std()], dtype=np.float32)
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).astype(np.float32)


def extract_features(audio_path: str) -> np.ndarray:
    """
    Load an audio file (WAV or MP3) and return a rich feature vector.

    Parameters
    ----------
    audio_path : str
        Full path to the audio file (.wav or .mp3).

    Returns
    -------
    np.ndarray
        Shape (288,) — concatenated mean+std of all feature groups.

    Raises
    ------
    ValueError
        If the file cannot be loaded, is empty, or is silent.
    """
    # ── Load audio ────────────────────────────────────────────────────────────
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    except Exception as e:
        raise ValueError(f"Could not load audio file '{audio_path}': {e}") from e

    if len(y) == 0:
        raise ValueError(f"Audio file is empty: '{audio_path}'")
    if np.max(np.abs(y)) < 1e-6:
        raise ValueError(f"Audio file appears to be silent: '{audio_path}'")

    # ── MFCC features ─────────────────────────────────────────────────────────
    # MFCCs capture the spectral envelope (timbre of the engine sound)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)          # (40, T)
    mfcc_feat = _pool(mfcc)                                            # (80,)

    # Delta MFCCs — first-order temporal derivative
    # Engine knock has a rhythmic pattern; deltas capture the onset/offset rhythm
    delta_mfcc = librosa.feature.delta(mfcc, order=1)                 # (40, T)
    delta_feat = _pool(delta_mfcc)                                     # (80,)

    # Delta-delta MFCCs — second-order derivative
    # Captures sharp transients (tick/knock events have high acceleration)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)                # (40, T)
    delta2_feat = _pool(delta2_mfcc)                                   # (80,)

    # ── Spectral contrast ─────────────────────────────────────────────────────
    # Measures the difference between spectral peaks and valleys in sub-bands
    # Engine knock creates sharp high-energy spikes → high contrast in upper bands
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=N_CONTRAST)  # (7, T)
    contrast_feat = _pool(contrast)                                    # (14,)

    # ── Chroma features ───────────────────────────────────────────────────────
    # Pitch class content — healthy engines have more harmonic/tonal structure
    # vs knock/misfire which has more noisy, aperiodic content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)  # (12, T)
    chroma_feat = _pool(chroma)                                        # (24,)

    # ── Time-domain and spectral scalar features ──────────────────────────────
    # Zero crossing rate — knock ticks cause rapid sign changes in the waveform
    zcr = librosa.feature.zero_crossing_rate(y)                       # (1, T)
    zcr_feat = _pool(zcr)                                              # (2,)

    # RMS energy — overall loudness profile
    rms = librosa.feature.rms(y=y)                                    # (1, T)
    rms_feat = _pool(rms)                                              # (2,)

    # Spectral centroid — "brightness"; knock shifts energy to higher frequencies
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)          # (1, T)
    centroid_feat = _pool(centroid)                                    # (2,)

    # Spectral rolloff — frequency below which 85% of total energy lies
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)            # (1, T)
    rolloff_feat = _pool(rolloff)                                      # (2,)

    # Spectral bandwidth — spread of energy around the centroid
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)        # (1, T)
    bandwidth_feat = _pool(bandwidth)                                  # (2,)

    # ── Concatenate all features ──────────────────────────────────────────────
    feature_vector = np.concatenate([
        mfcc_feat,        # 80
        delta_feat,       # 80
        delta2_feat,      # 80
        contrast_feat,    # 14
        chroma_feat,      # 24
        zcr_feat,         #  2
        rms_feat,         #  2
        centroid_feat,    #  2
        rolloff_feat,     #  2
        bandwidth_feat,   #  2
    ])                    # = 288 total

    return feature_vector.astype(np.float32)


def get_feature_config() -> dict:
    """Return the feature extraction configuration as a plain dict for saving."""
    return {
        "sample_rate":          SAMPLE_RATE,
        "max_duration_seconds": MAX_DURATION,
        "n_mfcc":               N_MFCC,
        "n_chroma":             N_CHROMA,
        "n_contrast_bands":     N_CONTRAST,
        "feature_dim":          FEATURE_DIM,
        "feature_groups": [
            "mfcc_mean_std (80)",
            "delta_mfcc_mean_std (80)",
            "delta2_mfcc_mean_std (80)",
            "spectral_contrast_mean_std (14)",
            "chroma_mean_std (24)",
            "zcr_mean_std (2)",
            "rms_mean_std (2)",
            "spectral_centroid_mean_std (2)",
            "spectral_rolloff_mean_std (2)",
            "spectral_bandwidth_mean_std (2)",
        ],
        "pooling": "mean_std",
        "version": "v2",
    }
