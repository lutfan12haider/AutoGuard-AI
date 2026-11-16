"""
compute_threshold.py
--------------------
Choose the abstain threshold from the VALIDATION split only.
Never touches the test split.

Strategy:
  - Run the saved model on every validation file.
  - Find the top-class confidence distribution for correct vs wrong predictions.
  - Choose a threshold T such that:
      * All predictions with conf < T are considered "uncertain"
      * T maximises (correct_above_T / total_above_T) while keeping
        the fraction of correct predictions retained >= 0.90.
  - Save T and the disagreement threshold to models/thresholds.json.

Seed: 42 (not relevant here — no random ops — but noted for reproducibility).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from ml.features import extract_features

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT  = os.path.join(ROOT, "data", "kwerimoto", "prepared")
MODELS_DIR = os.path.join(ROOT, "models")

pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

VAL_DIR = os.path.join(DATA_ROOT, "validation")
CLASSES = ["engine_knock", "healthy", "misfire"]

print("Computing thresholds from VALIDATION split only...")
confs_correct, confs_wrong = [], []
all_top_confs = []

for cls in CLASSES:
    cls_dir = os.path.join(VAL_DIR, cls)
    for fname in sorted(os.listdir(cls_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(cls_dir, fname)
        try:
            feat = extract_features(path)
            proba = pipeline.predict_proba(feat.reshape(1, -1))[0]
            pred  = le.inverse_transform([pipeline.predict(feat.reshape(1,-1))[0]])[0]
            top_conf = float(max(proba))
            all_top_confs.append(top_conf)
            if pred == cls:
                confs_correct.append(top_conf)
            else:
                confs_wrong.append(top_conf)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")

print(f"\n  Validation files processed: {len(all_top_confs)}")
print(f"  Correct predictions: {len(confs_correct)}")
print(f"  Wrong predictions  : {len(confs_wrong)}")
print(f"\n  Confidence distribution (all val predictions):")
for pct in [0, 10, 25, 50, 75, 90, 100]:
    v = np.percentile(all_top_confs, pct)
    print(f"    p{pct:3d}: {v:.3f}")

print(f"\n  Confidence distribution (WRONG predictions):")
if confs_wrong:
    for pct in [0, 25, 50, 75, 100]:
        v = np.percentile(confs_wrong, pct)
        print(f"    p{pct:3d}: {v:.3f}")

# ── Choose abstain threshold ───────────────────────────────────────────────────
# Try thresholds from 0.40 to 0.80, pick the one that:
#   - retains >= 90% of CORRECT predictions above threshold
#   - maximises precision above threshold (= correct / total above)
total_correct = len(confs_correct)
best_T, best_prec, best_retained = 0.0, 0.0, 1.0

results_table = []
for T_int in range(40, 85, 5):
    T = T_int / 100.0
    above_correct = sum(1 for c in confs_correct if c >= T)
    above_wrong   = sum(1 for c in confs_wrong   if c >= T)
    above_total   = above_correct + above_wrong
    retained_frac = above_correct / total_correct if total_correct > 0 else 0
    prec_above    = above_correct / above_total if above_total > 0 else 0
    results_table.append((T, above_correct, above_wrong, retained_frac, prec_above))
    if retained_frac >= 0.90 and prec_above > best_prec:
        best_T, best_prec, best_retained = T, prec_above, retained_frac

print(f"\n  Threshold sweep (validation only):")
print(f"  {'T':>5}  {'correct≥T':>10}  {'wrong≥T':>8}  {'retained':>9}  {'prec≥T':>8}")
for T, ac, aw, rf, pa in results_table:
    flag = " ← CHOSEN" if abs(T - best_T) < 0.001 else ""
    print(f"  {T:>5.2f}  {ac:>10}  {aw:>8}  {rf:>9.3f}  {pa:>8.3f}{flag}")

# ── Disagreement threshold (for windowed mode) ────────────────────────────────
# If std(window_top_confs) > DISAGREE_T, windows disagree strongly → uncertain
# We pick 0.20 as a conservative default (20 pp std across windows).
DISAGREE_T = 0.20

print(f"\n  Selected abstain threshold: {best_T:.2f}")
print(f"  Retained correct:          {best_retained:.1%}")
print(f"  Precision above threshold: {best_prec:.1%}")
print(f"  Disagreement threshold:    {DISAGREE_T:.2f}")

# Save
out = {
    "abstain_threshold":      round(best_T, 2),
    "disagree_threshold":     DISAGREE_T,
    "chosen_from":            "validation split only",
    "retained_correct_frac":  round(best_retained, 4),
    "precision_above_T":      round(best_prec, 4),
    "note": (
        "Predict 'uncertain' when max(proba) < abstain_threshold "
        "OR std(window_top_confs) > disagree_threshold."
    ),
}
out_path = os.path.join(MODELS_DIR, "thresholds.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → {out_path}")
