"""
train.py
--------
AutoGuard AI — ML training pipeline.

Steps:
  1. Load audio files from train/ and validation/ splits.
  2. Extract 80-dim MFCC feature vectors.
  3. Train SVM and Random Forest with class_weight='balanced'.
  4. Compare both models on the validation split using macro F1-score.
  5. Evaluate the winning model on the held-out test split.
  6. Evaluate separately on real_test (engine_knock only).
  7. Save the best model, label encoder, and full metrics.

Run from the project root:
    python ml/train.py
"""

import os
import sys
import time
import json
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Add parent dir so we can import ml.features
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import extract_features, get_feature_config

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kwerimoto", "prepared")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

SPLITS = {
    "train":      os.path.join(DATA_ROOT, "train"),
    "validation": os.path.join(DATA_ROOT, "validation"),
    "test":       os.path.join(DATA_ROOT, "test"),
    "real_test":  os.path.join(DATA_ROOT, "real_test"),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_split(split_path: str, classes: list[str], label_for_all: str | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Walk a split directory, extract features from every WAV file,
    and return (X, y) where y contains string class labels.

    If label_for_all is set, every file gets that label (used for real_test
    which has no 'healthy' or 'misfire' subdirs).
    """
    X_list, y_list = [], []
    errors = 0

    for class_name in os.listdir(split_path):
        class_dir = os.path.join(split_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        label = label_for_all if label_for_all else class_name

        wav_files = [f for f in os.listdir(class_dir) if f.lower().endswith(".wav")]
        print(f"  [{label}] Loading {len(wav_files)} files from {class_dir} ...")

        for wav_file in wav_files:
            path = os.path.join(class_dir, wav_file)
            try:
                feat = extract_features(path)
                X_list.append(feat)
                y_list.append(label)
            except Exception as e:
                errors += 1
                print(f"    SKIP: {wav_file} — {e}")

    if errors:
        print(f"  WARNING: Skipped {errors} files due to load errors.")

    return np.array(X_list, dtype=np.float32), y_list


def print_metrics(name: str, y_true: list, y_pred: list, classes: list[str]):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(report)
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Classes: {classes}")
    for i, row in enumerate(cm):
        print(f"  {classes[i]:15s}: {row}")
    return acc, f1, report, cm.tolist()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    CLASSES = ["engine_knock", "healthy", "misfire"]

    # ── 1. Load splits ──────────────────────────────────────────────────────
    print("\n[1/6] Loading TRAIN split ...")
    t0 = time.time()
    X_train, y_train_raw = load_split(SPLITS["train"], CLASSES)
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Class counts: { {c: y_train_raw.count(c) for c in CLASSES} }")

    print("\n[2/6] Loading VALIDATION split ...")
    X_val, y_val_raw = load_split(SPLITS["validation"], CLASSES)
    print(f"  Val: {X_val.shape[0]} samples")

    print("\n[3/6] Loading TEST split ...")
    X_test, y_test_raw = load_split(SPLITS["test"], CLASSES)
    print(f"  Test: {X_test.shape[0]} samples")

    print("\n[4/6] Loading REAL_TEST split (engine_knock only) ...")
    X_real, y_real_raw = load_split(SPLITS["real_test"], CLASSES)
    print(f"  Real_test: {X_real.shape[0]} samples")

    load_time = time.time() - t0
    print(f"\n  Feature extraction complete in {load_time:.1f}s")

    # Encode labels
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    y_test  = le.transform(y_test_raw)
    y_real  = le.transform(y_real_raw)
    print(f"\n  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── 2. Build models (Pipeline = StandardScaler + classifier) ────────────
    # StandardScaler is critical for SVM; doesn't hurt RF.
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
            ),
            cv=5,
            ensemble=False,
        ))
    ])

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # ── 3. Train & validate both models ─────────────────────────────────────
    results = {}

    for model_name, pipeline in [("SVM", svm_pipeline), ("Random Forest", rf_pipeline)]:
        print(f"\n[Training] {model_name} ...")
        t_start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t_start
        print(f"  Training time: {train_time:.1f}s")

        # Validation inference time
        t_infer = time.time()
        y_val_pred_enc = pipeline.predict(X_val)
        infer_time_per_sample = (time.time() - t_infer) / len(X_val)

        y_val_pred = le.inverse_transform(y_val_pred_enc)
        acc, f1, report, cm = print_metrics(
            f"{model_name} — Validation", y_val_raw, y_val_pred, CLASSES
        )
        results[model_name] = {
            "pipeline": pipeline,
            "val_accuracy": acc,
            "val_macro_f1": f1,
            "train_time_sec": round(train_time, 2),
            "infer_time_ms_per_sample": round(infer_time_per_sample * 1000, 3),
            "val_report": report,
            "val_cm": cm,
        }

    # ── 4. Select best model by validation macro F1 ──────────────────────────
    print("\n[5/6] Model comparison:")
    for name, r in results.items():
        print(f"  {name:15s}: Val Macro F1 = {r['val_macro_f1']:.4f}  |  Acc = {r['val_accuracy']:.4f}")

    best_name = max(results, key=lambda n: results[n]["val_macro_f1"])
    best      = results[best_name]
    print(f"\n  >> WINNER: {best_name} (Val Macro F1 = {best['val_macro_f1']:.4f})")

    # ── 5. Evaluate best model on held-out TEST split ────────────────────────
    print(f"\n[6/6] Evaluating {best_name} on TEST split ...")
    y_test_pred_enc = best["pipeline"].predict(X_test)
    y_test_pred     = le.inverse_transform(y_test_pred_enc)
    test_acc, test_f1, test_report, test_cm = print_metrics(
        f"{best_name} — Test (held-out)", y_test_raw, y_test_pred, CLASSES
    )

    # ── 6. Evaluate on real_test (engine_knock only) ─────────────────────────
    print(f"\n  Evaluating {best_name} on REAL_TEST split (engine_knock only) ...")
    y_real_pred_enc = best["pipeline"].predict(X_real)
    y_real_pred     = le.inverse_transform(y_real_pred_enc)
    # real_test contains only engine_knock files; evaluate as simple accuracy
    real_correct = sum(1 for p in y_real_pred if p == "engine_knock")
    real_acc = real_correct / len(y_real_pred)
    real_report = f"  {real_correct}/{len(y_real_pred)} correctly classified as engine_knock ({real_acc*100:.1f}%)"
    real_cm = [[real_correct, len(y_real_pred) - real_correct]]  # [correct, misclassified]
    # Show breakdown of what the model predicted
    from collections import Counter
    pred_counts = Counter(y_real_pred)
    print(f"\n{'='*55}")
    print(f"  {best_name} — Real Test (engine_knock only)")
    print(f"{'='*55}")
    print(f"  Total files  : {len(y_real_pred)}")
    print(f"  Predicted breakdown: {dict(pred_counts)}")
    print(f"  Correctly identified as engine_knock: {real_correct}/{len(y_real_pred)} ({real_acc*100:.1f}%)")
    real_f1 = real_acc  # single class, accuracy = recall for that class

    # ── 7. Save model and metadata ────────────────────────────────────────────
    model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    meta_path    = os.path.join(MODELS_DIR, "metadata.json")

    joblib.dump(best["pipeline"], model_path)
    joblib.dump(le, encoder_path)
    print(f"\n  Model saved  → {model_path}")
    print(f"  Encoder saved→ {encoder_path}")

    # Save all metadata as JSON
    metadata = {
        "model_name": best_name,
        "feature_config": get_feature_config(),
        "classes": list(le.classes_),
        "class_to_index": {c: int(i) for c, i in zip(le.classes_, le.transform(le.classes_))},
        "training": {
            "train_samples": int(X_train.shape[0]),
            "train_class_counts": {c: y_train_raw.count(c) for c in CLASSES},
            "train_time_sec": best["train_time_sec"],
            "infer_time_ms_per_sample": best["infer_time_ms_per_sample"],
        },
        "validation_metrics": {
            "accuracy": round(best["val_accuracy"], 4),
            "macro_f1": round(best["val_macro_f1"], 4),
            "confusion_matrix": best["val_cm"],
            "classification_report": best["val_report"],
        },
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_f1, 4),
            "confusion_matrix": test_cm,
            "classification_report": test_report,
        },
        "real_test_metrics": {
            "total_files": int(len(y_real_pred)),
            "correctly_identified_as_knock": int(real_correct),
            "accuracy": round(real_acc, 4),
        },
        "comparison": {
            name: {
                "val_accuracy": round(r["val_accuracy"], 4),
                "val_macro_f1": round(r["val_macro_f1"], 4),
                "train_time_sec": r["train_time_sec"],
            }
            for name, r in results.items()
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved → {meta_path}")
    print("\n  Training complete.\n")


if __name__ == "__main__":
    main()
