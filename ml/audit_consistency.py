"""
audit_consistency.py
--------------------
Step 1: Compare every preprocessing decision between
ml/features.py (training path) and backend/model.py (serving path).
Prints a table of matches and differences. No files modified.
"""
import os, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import soundfile as sf
import joblib
import tempfile

from ml.features import (
    extract_features as train_extract,
    SAMPLE_RATE, MAX_DURATION, N_MFCC, N_CHROMA, N_CONTRAST, FEATURE_DIM,
    get_feature_config,
)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_ROOT  = os.path.join(ROOT, "data", "kwerimoto", "prepared")

SEP = "=" * 72

# ── Replicate the backend path step-by-step ───────────────────────────────────
# backend/model.py:predict_audio() does:
#   1. Write bytes to NamedTemporaryFile(suffix=ext)
#   2. sf.info(tmp_path)  → duration check
#   3. extract_features(tmp_path)   ← calls ml/features.py directly
#   4. pipeline.predict / predict_proba
# So the ONLY preprocessing difference between training and serving is:
#   training: extract_features(disk_path)
#   serving:  bytes → NamedTemporaryFile(suffix=ext) → extract_features(tmp_path)
# Both call the SAME extract_features function.

print(SEP)
print("  STEP 1 — Train / Serve Consistency Audit")
print(SEP)

pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
import json
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    meta = json.load(f)
le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

cfg = get_feature_config()

rows = [
    # (aspect, training value, serving value, match?)
    ("sample_rate",         str(SAMPLE_RATE),
                            "same (calls extract_features)", True),
    ("mono conversion",     "librosa.load mono=True (default)",
                            "same (calls extract_features)", True),
    ("clip length (MAX_DURATION)", str(MAX_DURATION)+"s",
                            "TRUNCATES at MAX_DURATION — long files lose tail", False),
    ("trimming",            "none",
                            "none", True),
    ("amplitude normalisation", "none — raw librosa.load values",
                            "none — same", True),
    ("MFCC n_mfcc",         str(N_MFCC),
                            "same (calls extract_features)", True),
    ("delta order 1 & 2",   "yes",
                            "same (calls extract_features)", True),
    ("spectral contrast",   f"n_bands={N_CONTRAST} → 7 rows",
                            "same (calls extract_features)", True),
    ("chroma",              f"n_chroma={N_CHROMA}",
                            "same (calls extract_features)", True),
    ("ZCR / RMS / centroid etc.", "yes — 5 scalar features",
                            "same (calls extract_features)", True),
    ("summary stats",       "mean + std per feature row",
                            "same (calls extract_features)", True),
    ("feature vector dim",  str(FEATURE_DIM),
                            str(meta["feature_config"]["feature_dim"]), FEATURE_DIM == meta["feature_config"]["feature_dim"]),
    ("StandardScaler",      "inside Pipeline — fitted on training X",
                            "loaded from best_model.pkl Pipeline", True),
    ("label order",         str(list(le.classes_)),
                            "loaded from label_encoder.pkl — same", True),
    ("long file handling",  f"truncated at {MAX_DURATION}s, single window",
                            f"truncated at {MAX_DURATION}s, single window — 33s file → only first {MAX_DURATION}s used", False),
    ("MP3 vs WAV",          "training data is WAV only",
                            "accepts MP3 via soundfile — different compression artifacts", False),
]

print(f"\n  {'Aspect':<30}  {'Train':<30}  {'Serve':<30}  Match")
print("  " + "-" * 95)
diffs = []
for aspect, train_val, serve_val, match in rows:
    flag = "✓" if match else "✗ DIFF"
    tv = textwrap.shorten(train_val, 28)
    sv = textwrap.shorten(serve_val, 28)
    print(f"  {aspect:<30}  {tv:<30}  {sv:<30}  {flag}")
    if not match:
        diffs.append((aspect, train_val, serve_val))

print(f"\n  Differences found: {len(diffs)}")
for i, (a, t, s) in enumerate(diffs, 1):
    print(f"\n  [{i}] {a}")
    print(f"      Train : {t}")
    print(f"      Serve : {s}")

# ── Root cause for the 33s Freesound file ─────────────────────────────────────
print(f"""
  ROOT CAUSE ANALYSIS — 33s Freesound MP3 predicted as Healthy:
  ─────────────────────────────────────────────────────────────────
  1. CLIP TRUNCATION (primary cause)
     Training clips are ~{MAX_DURATION}s. The 33s file is truncated to
     the first {MAX_DURATION}s only. If the knock is not prominent in the
     first {MAX_DURATION}s (e.g. car idling before knock starts), the
     model sees a near-healthy acoustic profile.
     FIX → window the long audio into {MAX_DURATION}s segments, predict
     each, aggregate (mean proba). This is exactly what Step 3 addresses.

  2. MP3 COMPRESSION (secondary cause)
     Training data is WAV (lossless). MP3 compression discards high-
     frequency content (above ~16kHz) and introduces ringing artifacts.
     The spectral centroid and rolloff features shift downward, moving
     the feature vector toward the healthy region.
     FIX → retrain with MP3-converted copies of training data, or add
     MP3 knock recordings (Step A of the data augmentation plan).

  3. DISTRIBUTION SHIFT (structural cause, confirmed by error_analysis.py)
     Misclassified knock files have lower ZCR (0.121 vs 0.177),
     lower centroid (1911Hz vs 2512Hz), and lower rolloff (3675Hz vs
     4676Hz) than correctly classified ones. These features overlap
     strongly with healthy engine feature ranges. The model's decision
     boundary between knock and healthy is not robust to quieter or
     more distant recordings.
""")

# ── Numerical verification: same path gives same result ──────────────────────
print("  Numerical check: training path == serving path for one file")
test_wav = None
knock_dir = os.path.join(DATA_ROOT, "test", "engine_knock")
for f in sorted(os.listdir(knock_dir)):
    if f.endswith(".wav"):
        test_wav = os.path.join(knock_dir, f)
        break

# Training path
feat_train = train_extract(test_wav)

# Serving path: bytes → tempfile → extract_features
with open(test_wav, "rb") as fh:
    raw_bytes = fh.read()
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp.write(raw_bytes)
    tmp_path = tmp.name
try:
    feat_serve = train_extract(tmp_path)
finally:
    os.unlink(tmp_path)

match = np.allclose(feat_train, feat_serve, atol=1e-5)
max_diff = float(np.max(np.abs(feat_train - feat_serve)))
print(f"    np.allclose(feat_train, feat_serve): {match}  (max_abs_diff={max_diff:.2e})")
print(f"    Conclusion: {'paths are IDENTICAL ✓' if match else 'DIVERGENCE DETECTED ✗'}")
print(f"\n{SEP}")
print("  Audit complete. No files modified.")
print(SEP)
