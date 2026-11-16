"""
diagnostic.py
-------------
Full prediction pipeline diagnostic for AutoGuard AI.
Runs Steps 1-6 from the debug spec without changing anything.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)
from collections import Counter
from ml.features import (
    extract_features, get_feature_config,
    SAMPLE_RATE, MAX_DURATION, N_MFCC, FEATURE_DIM
)

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT  = os.path.join(ROOT, "data", "kwerimoto", "prepared")
MODELS_DIR = os.path.join(ROOT, "models")

SEP  = "=" * 65
SEP2 = "-" * 65

# ── helpers ────────────────────────────────────────────────────────────────────

def first_n_wavs(split, cls, n=3):
    d = os.path.join(DATA_ROOT, split, cls)
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
    return [os.path.join(d, f) for f in files[:n]]

# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("  STEP 1 — Inspect the prediction pipeline")
print(SEP)

# Load saved model + encoder + metadata
pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    meta = json.load(f)

print(f"Model type        : {meta['model_name']}")
print(f"Saved classes     : {meta['classes']}")
print(f"class_to_index    : {meta['class_to_index']}")
print(f"LabelEncoder classes: {list(le.classes_)}")
print(f"Saved feature_dim : {meta['feature_config']['feature_dim']}")
print(f"Current FEATURE_DIM (features.py): {FEATURE_DIM}")

# Check pipeline steps
print(f"\nPipeline steps    : {[name for name, _ in pipeline.steps]}")

# Extract a dummy feature and confirm shape matches saved model
test_file = first_n_wavs("test", "engine_knock", 1)[0]
test_feat = extract_features(test_file)
print(f"Live feature shape: {test_feat.shape}  (expected {FEATURE_DIM})")

SHAPE_MATCH = test_feat.shape[0] == FEATURE_DIM == meta['feature_config']['feature_dim']
print(f"Shape consistent  : {'YES ✓' if SHAPE_MATCH else 'NO ✗ — MISMATCH DETECTED'}")

# Check sample rate and MFCC params match
config = get_feature_config()
print(f"\nFeature config (live):")
for k, v in config.items():
    if k != "feature_groups":
        print(f"  {k}: {v}")
print(f"\nFeature config (saved at training):")
for k, v in meta['feature_config'].items():
    if k != "feature_groups":
        match = str(v) == str(config.get(k, "MISSING"))
        print(f"  {k}: {v}  {'✓' if match else '✗ DIFFERS from live: ' + str(config.get(k))}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Test predictions on 3 files per class")
print(SEP)

TEST_FILES = {
    "engine_knock": first_n_wavs("test", "engine_knock", 3),
    "healthy":      first_n_wavs("test", "healthy",      3),
    "misfire":      first_n_wavs("test", "misfire",      3),
}

all_true, all_pred = [], []

for true_class, paths in TEST_FILES.items():
    print(f"\n  Class: {true_class.upper()}")
    print(SEP2)
    for path in paths:
        fname = os.path.basename(path)

        # Load audio manually for diagnostics
        y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        duration = len(y) / sr

        # Extract features
        feat = extract_features(path)

        # Predict
        X = feat.reshape(1, -1)
        t0 = time.perf_counter()
        pred_enc = pipeline.predict(X)[0]
        proba    = pipeline.predict_proba(X)[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        pred_label = le.inverse_transform([pred_enc])[0]
        class_scores = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)}
        confidence = round(float(max(proba)), 4)

        correct = "✓" if pred_label == true_class else "✗"

        print(f"  File      : {fname}")
        print(f"  Duration  : {duration:.2f}s  |  Sample rate: {sr} Hz")
        print(f"  Feat shape: {feat.shape}  |  min={feat.min():.3f}  max={feat.max():.3f}")
        print(f"  True label: {true_class}")
        print(f"  Predicted : {pred_label}  {correct}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Scores    : {class_scores}")
        print(f"  Raw enc   : {pred_enc}  |  All proba: {[round(float(p),4) for p in proba]}")
        print(f"  Infer time: {infer_ms:.1f} ms")
        print()

        all_true.append(true_class)
        all_pred.append(pred_label)

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Feature extraction consistency check")
print(SEP)

# Check that different files produce different arrays
feat_vectors = {}
for cls in ["engine_knock", "healthy", "misfire"]:
    paths = first_n_wavs("test", cls, 2)
    vecs = []
    for p in paths:
        y, sr = librosa.load(p, sr=SAMPLE_RATE, duration=MAX_DURATION)
        feat = extract_features(p)
        vecs.append((os.path.basename(p), y, feat))
    feat_vectors[cls] = vecs

print("\n  Checking features are different across files:")
for cls, vecs in feat_vectors.items():
    (fn1, y1, f1_arr), (fn2, y2, f2_arr) = vecs
    min_len = min(len(y1), len(y2))
    audio_diff = (len(y1) != len(y2)) or not np.allclose(y1[:min_len], y2[:min_len])
    feat_diff  = not np.allclose(f1_arr, f2_arr)
    print(f"  {cls}:")
    print(f"    {fn1} (samples={len(y1)}) vs {fn2} (samples={len(y2)})")
    print(f"    Audio arrays different : {audio_diff}  {'✓' if audio_diff else '✗ IDENTICAL — LOAD BUG'}")
    print(f"    Feature arrays different: {feat_diff}  {'✓' if feat_diff else '✗ IDENTICAL — EXTRACTION BUG'}")
    print(f"    Feature dims: {f1_arr.shape[0]} vs {f2_arr.shape[0]}  {'✓' if f1_arr.shape==f2_arr.shape else '✗ MISMATCH'}")

# Cross-class difference
f_knock  = extract_features(first_n_wavs("test", "engine_knock", 1)[0])
f_healthy= extract_features(first_n_wavs("test", "healthy",      1)[0])
f_misfire= extract_features(first_n_wavs("test", "misfire",      1)[0])
print(f"\n  L2 distance knock↔healthy : {np.linalg.norm(f_knock - f_healthy):.2f}")
print(f"  L2 distance knock↔misfire : {np.linalg.norm(f_knock - f_misfire):.2f}")
print(f"  L2 distance healthy↔misfire: {np.linalg.norm(f_healthy - f_misfire):.2f}")
print(f"  (Non-zero distances confirm features are class-discriminative)")

# NaN / Inf check
for cls in ["engine_knock", "healthy", "misfire"]:
    feat = extract_features(first_n_wavs("test", cls, 1)[0])
    has_nan = np.any(np.isnan(feat))
    has_inf = np.any(np.isinf(feat))
    print(f"  {cls}: NaN={has_nan}  Inf={has_inf}  {'✓' if not has_nan and not has_inf else '✗ BAD VALUES'}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Class mapping verification")
print(SEP)

print(f"  LabelEncoder.classes_ order: {list(le.classes_)}")
print(f"  Numeric mapping:")
for i, cls in enumerate(le.classes_):
    print(f"    {i} → {cls}")

# Verify by encoding and decoding known labels
for cls in ["engine_knock", "healthy", "misfire"]:
    enc = le.transform([cls])[0]
    dec = le.inverse_transform([enc])[0]
    match = dec == cls
    print(f"  encode({cls}) = {enc}  decode({enc}) = {dec}  {'✓' if match else '✗ MAPPING BUG'}")

# Verify predict_proba column order matches label encoder
feat_test = extract_features(first_n_wavs("test", "engine_knock", 1)[0])
proba_test = pipeline.predict_proba(feat_test.reshape(1, -1))[0]
print(f"\n  predict_proba columns ({len(proba_test)}) vs le.classes_ ({len(le.classes_)}): "
      f"{'✓ match' if len(proba_test) == len(le.classes_) else '✗ MISMATCH'}")
print(f"  Classes order: {list(le.classes_)}")
print(f"  Proba values : {[round(float(p),4) for p in proba_test]}")
# Highest proba should map to predicted class
pred_idx = int(np.argmax(proba_test))
pred_cls = le.classes_[pred_idx]
print(f"  argmax({pred_idx}) → {pred_cls}  (via le.classes_ indexing)")
pred_direct = le.inverse_transform([pipeline.predict(feat_test.reshape(1,-1))[0]])[0]
print(f"  pipeline.predict → {pred_direct}")
print(f"  Match: {'✓' if pred_cls == pred_direct else '✗ ORDERING BUG'}")

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Model bias check on full test split")
print(SEP)

# Load all test split files
X_test, y_test_true = [], []
for cls in ["engine_knock", "healthy", "misfire"]:
    cls_dir = os.path.join(DATA_ROOT, "test", cls)
    for fname in sorted(os.listdir(cls_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        try:
            feat = extract_features(os.path.join(cls_dir, fname))
            X_test.append(feat)
            y_test_true.append(cls)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

X_test = np.array(X_test)
y_test_pred_enc = pipeline.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred_enc)

print(f"\n  Test set size: {len(y_test_true)} samples")
print(f"\n  True class distribution:")
for cls, cnt in sorted(Counter(y_test_true).items()):
    print(f"    {cls}: {cnt}")

pred_counts = Counter(y_test_pred)
print(f"\n  Predicted class distribution:")
for cls in ["engine_knock", "healthy", "misfire"]:
    cnt = pred_counts.get(cls, 0)
    pct = cnt / len(y_test_pred) * 100
    bias_flag = " ← possible bias" if pct > 70 else ""
    print(f"    {cls}: {cnt}  ({pct:.1f}%){bias_flag}")

majority_class = Counter(y_test_true).most_common(1)[0][0]
majority_acc = Counter(y_test_true).most_common(1)[0][1] / len(y_test_true)
print(f"\n  Majority-class baseline (always predict '{majority_class}'): {majority_acc*100:.1f}%")

acc = accuracy_score(y_test_true, y_test_pred)
f1  = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
print(f"  Model test accuracy : {acc*100:.1f}%")
print(f"  Model macro F1      : {f1:.4f}")
print(f"  Above majority baseline: {'YES ✓' if acc > majority_acc else 'NO ✗ MODEL IS USELESS'}")

classes = ["engine_knock", "healthy", "misfire"]
cm = confusion_matrix(y_test_true, y_test_pred, labels=classes)
print(f"\n  Confusion matrix (rows=true, cols=pred):")
print(f"  {'':15s}  {'knock':>10}  {'healthy':>10}  {'misfire':>10}")
for i, cls in enumerate(classes):
    row = cm[i]
    recall = row[i] / row.sum() if row.sum() > 0 else 0
    print(f"  {cls:15s}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}   recall={recall*100:.0f}%")

print(f"\n  Per-class report:")
print(classification_report(y_test_true, y_test_pred, target_names=classes, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Compare direct inference vs API preprocessing")
print(SEP)

# Pick one file per class and compare direct pipeline prediction
# vs what the API model.py predict_audio() would do
import tempfile, soundfile as sf_module

test_paths = {
    "engine_knock": first_n_wavs("test", "engine_knock", 1)[0],
    "healthy":      first_n_wavs("test", "healthy",      1)[0],
    "misfire":      first_n_wavs("test", "misfire",      1)[0],
}

print("\n  Direct pipeline (no API layer):")
for true_cls, path in test_paths.items():
    feat = extract_features(path)
    pred_enc = pipeline.predict(feat.reshape(1,-1))[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    proba = pipeline.predict_proba(feat.reshape(1,-1))[0]
    print(f"  {true_cls:15s} → {pred_label:15s}  conf={max(proba)*100:.1f}%  {'✓' if pred_label==true_cls else '✗'}")

print("\n  Simulated API path (bytes→tempfile→extract_features→predict):")
for true_cls, path in test_paths.items():
    # Simulate exactly what backend/model.py:predict_audio() does
    with open(path, "rb") as f:
        audio_bytes = f.read()
    ext = os.path.splitext(path)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        info = sf_module.info(tmp_path)
        feat2 = extract_features(tmp_path)
        pred_enc2 = pipeline.predict(feat2.reshape(1,-1))[0]
        pred_label2 = le.inverse_transform([pred_enc2])[0]
        proba2 = pipeline.predict_proba(feat2.reshape(1,-1))[0]
        feat_identical = np.allclose(feat, extract_features(path))
        print(f"  {true_cls:15s} → {pred_label2:15s}  conf={max(proba2)*100:.1f}%  "
              f"feats_identical={'YES' if np.allclose(feat2, extract_features(path)) else 'NO ✗'}  "
              f"{'✓' if pred_label2==true_cls else '✗'}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  DIAGNOSTIC SUMMARY")
print(SEP)
print(f"  Model                  : {meta['model_name']}")
print(f"  Feature dim (saved)    : {meta['feature_config']['feature_dim']}")
print(f"  Feature dim (live)     : {FEATURE_DIM}")
print(f"  Shape match            : {SHAPE_MATCH}")
print(f"  LabelEncoder order     : {list(le.classes_)}")
print(f"  Test accuracy          : {acc*100:.1f}%")
print(f"  Test macro F1          : {f1:.4f}")
print(f"  Majority baseline      : {majority_acc*100:.1f}%")
print(f"  Model beats baseline   : {'YES' if acc > majority_acc else 'NO'}")
print()
# Bias check
dominant = pred_counts.most_common(1)[0]
if dominant[1] / len(y_test_pred) > 0.7:
    print(f"  ⚠ BIAS DETECTED: '{dominant[0]}' predicted {dominant[1]/len(y_test_pred)*100:.0f}% of the time")
else:
    print(f"  No severe prediction bias detected.")
print(SEP)
