"""
eval_30clips.py
---------------
Step 5: Evaluate the saved model on 30 stratified held-out test clips.

Stratification: 10 engine_knock, 10 healthy, 10 misfire (seed=42).
Never modifies the test split directory.
Saves full report to reports/eval_30clips.json and reports/eval_30clips.txt.
"""
import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
from ml.features import extract_features

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT  = os.path.join(ROOT, "data", "kwerimoto", "prepared")
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS    = os.path.join(ROOT, "reports")
os.makedirs(REPORTS, exist_ok=True)

SEED    = 42
N_PER_CLASS = 10
CLASSES = ["engine_knock", "healthy", "misfire"]

rng = random.Random(SEED)

pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    meta = json.load(f)

# ── Stratified sample ─────────────────────────────────────────────────────────
selected = {}
for cls in CLASSES:
    cls_dir = os.path.join(DATA_ROOT, "test", cls)
    all_wav = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(".wav"))
    # skip the one known-broken file
    all_wav = [f for f in all_wav if "revix_car_knocking_0252" not in f]
    chosen  = rng.sample(all_wav, min(N_PER_CLASS, len(all_wav)))
    selected[cls] = [os.path.join(cls_dir, f) for f in chosen]
    print(f"  Selected {len(chosen)} {cls} files")

# ── Predict ───────────────────────────────────────────────────────────────────
records   = []
y_true    = []
y_pred    = []
skip_count = 0

for cls, paths in selected.items():
    for path in paths:
        fname = os.path.basename(path)
        try:
            feat     = extract_features(path)
            pred_enc = pipeline.predict(feat.reshape(1, -1))[0]
            proba    = pipeline.predict_proba(feat.reshape(1, -1))[0]
            pred     = le.inverse_transform([pred_enc])[0]
            conf     = round(float(max(proba)), 4)
            scores   = {c: round(float(p), 4) for c, p in zip(le.classes_, proba)}
            correct  = pred == cls
            records.append({
                "filename": fname, "true": cls, "predicted": pred,
                "correct": correct, "confidence": conf, "scores": scores,
            })
            y_true.append(cls)
            y_pred.append(pred)
            mark = "✓" if correct else "✗"
            print(f"  {mark} {fname:<45}  true={cls:<15} pred={pred:<15} conf={conf:.3f}")
        except Exception as e:
            skip_count += 1
            print(f"  SKIP {fname}: {e}")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"  EVALUATION RESULTS — {len(y_true)} clips (seed={SEED})")
print("=" * 65)

acc    = accuracy_score(y_true, y_pred)
mac_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
wei_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"\n  Accuracy        : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Macro F1        : {mac_f1:.4f}")
print(f"  Weighted F1     : {wei_f1:.4f}")
print(f"  Skipped files   : {skip_count}")

report_str = classification_report(y_true, y_pred, target_names=CLASSES,
                                    zero_division=0)
print(f"\n  Classification report:\n{report_str}")

cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
print(f"  Confusion matrix (rows=true, cols=pred):")
print(f"  {'':17s}  {'knock':>8}  {'healthy':>8}  {'misfire':>8}")
for i, cls in enumerate(CLASSES):
    recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {cls:17s}  {cm[i,0]:>8}  {cm[i,1]:>8}  {cm[i,2]:>8}  recall={recall:.0%}")

# Per-class precision / recall / F1
print(f"\n  Per-class breakdown:")
print(f"  {'Class':17s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>7}")
for cls in CLASSES:
    p  = precision_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)
    r  = recall_score   (y_true, y_pred, labels=[cls], average="macro", zero_division=0)
    f1 = f1_score       (y_true, y_pred, labels=[cls], average="macro", zero_division=0)
    n  = y_true.count(cls)
    print(f"  {cls:17s}  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  {n:>7}")

# ── Save reports ──────────────────────────────────────────────────────────────
report_data = {
    "model":          meta["model_name"],
    "seed":           SEED,
    "n_per_class":    N_PER_CLASS,
    "total_clips":    len(y_true),
    "skipped":        skip_count,
    "accuracy":       round(acc, 4),
    "macro_f1":       round(mac_f1, 4),
    "weighted_f1":    round(wei_f1, 4),
    "classification_report": report_str,
    "confusion_matrix": cm.tolist(),
    "classes":        CLASSES,
    "per_clip":       records,
}

json_path = os.path.join(REPORTS, "eval_30clips.json")
txt_path  = os.path.join(REPORTS, "eval_30clips.txt")

with open(json_path, "w") as f:
    json.dump(report_data, f, indent=2)

lines = [
    f"AutoGuard AI — 30-clip stratified evaluation",
    f"Model: {meta['model_name']}  |  seed={SEED}",
    f"",
    f"Accuracy : {acc:.4f}",
    f"Macro F1 : {mac_f1:.4f}",
    f"",
    report_str,
    f"Confusion matrix (rows=true, cols=pred):  {CLASSES}",
]
for i, cls in enumerate(CLASSES):
    lines.append(f"  {cls}: {cm[i].tolist()}")
lines += ["", "Per-clip results:"]
for r in records:
    lines.append(
        f"  {'OK' if r['correct'] else 'XX'}  {r['filename']:<45}"
        f"  true={r['true']:<15}  pred={r['predicted']:<15}  conf={r['confidence']:.3f}"
    )

with open(txt_path, "w") as f:
    f.write("\n".join(lines))

print(f"\n  Saved → {json_path}")
print(f"  Saved → {txt_path}")
print("=" * 65)
