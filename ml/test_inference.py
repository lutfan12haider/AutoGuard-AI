"""
test_inference.py
-----------------
Smoke-test the saved model by running inference on real audio files.

Tests:
  1. Model loads correctly.
  2. Predict one file from each class (train split, quick sanity check).
  3. Predict one real_test file.
  4. Test with an invalid input (should raise a clean error, not crash).
  5. Print timing per inference call.

Run from the project root:
    python ml/test_inference.py
"""

import os
import sys
import time
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import extract_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kwerimoto", "prepared")


def load_model():
    model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    meta_path    = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun ml/train.py first.")

    pipeline = joblib.load(model_path)
    le       = joblib.load(encoder_path)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Model loaded: {meta['model_name']}")
    print(f"Classes: {meta['classes']}")
    print(f"Test Accuracy: {meta['test_metrics']['accuracy']}")
    print(f"Test Macro F1: {meta['test_metrics']['macro_f1']}")
    return pipeline, le, meta


def predict_file(pipeline, le, audio_path: str) -> dict:
    """
    Run inference on a single WAV file.
    Returns a dict with prediction, confidence, and per-class probabilities.
    """
    t0 = time.perf_counter()

    # Extract features
    features = extract_features(audio_path)
    X = features.reshape(1, -1)

    # Predict
    pred_enc  = pipeline.predict(X)[0]
    pred_label = le.inverse_transform([pred_enc])[0]

    # Probability scores (available because SVM was trained with probability=True
    # and RF has predict_proba natively)
    proba = pipeline.predict_proba(X)[0]
    class_scores = {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)}

    infer_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "prediction": pred_label,
        "confidence": round(float(max(proba)), 4),
        "class_scores": class_scores,
        "inference_ms": infer_ms,
    }


def main():
    print("=" * 55)
    print("  AutoGuard AI — Inference Smoke Test")
    print("=" * 55)

    pipeline, le, meta = load_model()

    # One file per class from the test split (held-out, never seen in training)
    test_files = [
        ("engine_knock", _first_wav(os.path.join(DATA_ROOT, "test", "engine_knock"))),
        ("healthy",      _first_wav(os.path.join(DATA_ROOT, "test", "healthy"))),
        ("misfire",      _first_wav(os.path.join(DATA_ROOT, "test", "misfire"))),
        ("engine_knock", _first_wav(os.path.join(DATA_ROOT, "real_test", "engine_knock"))),
    ]
    test_labels = ["engine_knock (test)", "healthy (test)", "misfire (test)", "engine_knock (real_test)"]

    print("\n--- Inference on test files ---")
    all_correct = True
    for (true_label, path), display_label in zip(test_files, test_labels):
        if path is None:
            print(f"  [{display_label}] No file found, skipping.")
            continue
        result = predict_file(pipeline, le, path)
        status = "✓" if result["prediction"] == true_label else "✗"
        print(f"\n  File   : {os.path.basename(path)}")
        print(f"  True   : {display_label}")
        print(f"  Pred   : {result['prediction']}  {status}")
        print(f"  Conf   : {result['confidence']*100:.1f}%")
        print(f"  Scores : {result['class_scores']}")
        print(f"  Time   : {result['inference_ms']} ms")
        if status == "✗":
            all_correct = False

    # Test error handling — pass a non-existent path
    print("\n--- Error handling test ---")
    try:
        predict_file(pipeline, le, "nonexistent_file.wav")
        print("  ERROR: Should have raised ValueError but didn't!")
    except ValueError as e:
        print(f"  Caught expected ValueError: {e}")
    except Exception as e:
        print(f"  Caught exception (type={type(e).__name__}): {e}")

    print("\n" + "=" * 55)
    print(f"  Smoke test complete. All expected predictions correct: {all_correct}")
    print("=" * 55 + "\n")


def _first_wav(directory: str) -> str | None:
    """Return the path to the first WAV file in a directory, or None."""
    if not os.path.exists(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.lower().endswith(".wav"):
            return os.path.join(directory, f)
    return None


if __name__ == "__main__":
    main()
