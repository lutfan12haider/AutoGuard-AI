"""
ml/v2/train_v2_9class_best.py
-----------------------------
AutoGuard AI V2 — Best CPU-Friendly 9-Class Librosa Ensemble.

Builds on our Phase 2/3 findings:
  - Uses strictly leakage-free session splits (`train=1385`, `validation=296`, `test=297`, `real_test=38`).
  - Extracts 1,008-dim multi-scale Librosa physics features (`win_mean 336 + win_std 336 + clip_global 336`)
    spanning MFCCs, Δ/Δ², IQRs, spectral contrast, chroma, HPSS, sub-band ratios, flux, onsets,
    RMS modulation, and pre-normalization `log10(raw_rms)` & `log10(raw_crest)`.
  - Decomposes the heterogeneous `healthy` class into its 4 true acoustic operating regimes
    (`healthy_kw`, `healthy_idle`, `healthy_startup`, `healthy_braking`) + 8 fault classes (`12` regimes),
    trains diverse calibrated/regularized base models (`SVM_RBF`, `MLP`, `LogisticRegression`, `HistGB`),
    and marginalizes the 4 healthy sub-regimes back into `healthy` at inference time.

Preserves all V1 files untouched and saves artifacts cleanly to `models/v2/track1/`.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.v2.metrics_utils import MODELS_V2_DIR, REPORTS_P3_DIR, SEED, compute_full_metrics, set_seed
from ml.v2.splitter import CLASSES_9

warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK1_DIR = ROOT / "data" / "v2_prepared" / "single_fault_9class"
OUT_DIR = MODELS_V2_DIR / "track1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUB_CLASSES_12: tuple[str, ...] = (
    "healthy_kw",
    "healthy_idle",
    "healthy_startup",
    "healthy_braking",
) + CLASSES_9[1:]


def _load_ms1008(sp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(TRACK1_DIR / f"{sp}.npz", allow_pickle=False)
    xw, wi, xg = d["X_win_tab"], d["win_clip_idx"], d["X_clip_global_tab"]
    nc = int(len(d["y_clip"]))
    xm = np.array([xw[wi == i].mean(axis=0) for i in range(nc)], dtype=np.float32)
    xs = np.array([xw[wi == i].std(axis=0) for i in range(nc)], dtype=np.float32)
    X_ms = np.hstack([xm, xs, xg]).astype(np.float32)
    return X_ms, d["y_clip"].astype(np.int32), d["clip_states"], d["clip_sources"]


def _to_sub12(y9: np.ndarray, states: np.ndarray, sources: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y9)
    for i, (yi, sti, srci) in enumerate(zip(y9, states, sources)):
        if yi != 0:
            out[i] = yi + 3
        elif str(srci).startswith("kwerimoto"):
            out[i] = 0
        elif sti == "idle":
            out[i] = 1
        elif sti == "startup":
            out[i] = 2
        else:
            out[i] = 3
    return out


def marginalize_12_to_9(p12: np.ndarray) -> np.ndarray:
    """Sum probabilities across the 4 healthy operating regimes into `healthy`."""
    p9 = np.zeros((len(p12), 9), dtype=np.float64)
    p9[:, 0] = p12[:, :4].sum(axis=1)
    p9[:, 1:] = p12[:, 4:]
    p9 /= np.maximum(p9.sum(axis=1, keepdims=True), 1e-8)
    return p9


def main() -> None:
    set_seed(SEED)
    t0_all = time.perf_counter()

    X_tr, y_tr9, st_tr, src_tr = _load_ms1008("train")
    X_va, y_va9, _, _ = _load_ms1008("validation")
    X_te, y_te9, _, _ = _load_ms1008("test")
    X_rt, y_rt9, _, _ = _load_ms1008("real_test")

    y_tr12 = _to_sub12(y_tr9, st_tr, src_tr)

    base_specs = [
        (
            "svm_rbf",
            0.35,
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", SVC(C=8.0, kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)),
            ]),
        ),
        (
            "mlp",
            0.25,
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=2.0, early_stopping=True, max_iter=200, random_state=SEED)),
            ]),
        ),
        (
            "logreg",
            0.20,
            Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=1500, class_weight="balanced", random_state=SEED)),
            ]),
        ),
        (
            "histgb",
            0.20,
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=1.5,
                class_weight="balanced",
                early_stopping=False,
                random_state=SEED,
            ),
        ),
    ]

    trained_models: dict[str, object] = {}
    weights: dict[str, float] = {}
    fit_times: dict[str, float] = {}

    for nm, w, clf in base_specs:
        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr12)
        dt = time.perf_counter() - t0
        trained_models[nm] = clf
        weights[nm] = w
        fit_times[nm] = round(dt, 2)
        p9_va = marginalize_12_to_9(clf.predict_proba(X_va))
        m_va = compute_full_metrics(y_va9, np.argmax(p9_va, axis=1), CLASSES_9)
        print(
            f"  [Fit {nm:8s}] {dt:5.1f}s | Val Acc={m_va['accuracy']:.4f} Bal={m_va['balanced_accuracy']:.4f} "
            f"F1={m_va['macro_f1']:.4f} FHR={m_va['false_healthy_rate']:.4f}",
            flush=True,
        )

    def _predict_9(X: np.ndarray) -> np.ndarray:
        p12 = sum(weights[nm] * trained_models[nm].predict_proba(X) for nm in trained_models)
        return marginalize_12_to_9(p12)

    results: dict[str, dict] = {}
    for sp, Xs, ys in (("train", X_tr, y_tr9), ("validation", X_va, y_va9), ("test", X_te, y_te9), ("real_test", X_rt, y_rt9)):
        t0 = time.perf_counter()
        p9 = _predict_9(Xs)
        inf_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(Xs))
        met = compute_full_metrics(ys, np.argmax(p9, axis=1), CLASSES_9)
        met["infer_ms_per_clip"] = round(inf_ms, 3)
        results[sp] = met

    bundle_path = OUT_DIR / "v2_9class_ensemble.joblib"
    joblib.dump(
        {
            "models": trained_models,
            "weights": weights,
            "sub_classes_12": SUB_CLASSES_12,
            "classes_9": CLASSES_9,
            "feature_dim": int(X_tr.shape[1]),
            "score_calibration": "uncalibrated_ensemble_probability",
        },
        bundle_path,
        compress=3,
    )

    report = {
        "model_name": "V2_12Subclass_Librosa_Ensemble",
        "classes": list(CLASSES_9),
        "sub_classes_12": list(SUB_CLASSES_12),
        "feature_dim": int(X_tr.shape[1]),
        "weights": weights,
        "fit_times_sec": fit_times,
        "artifact_mb": round(bundle_path.stat().st_size / (1024 ** 2), 2),
        "total_elapsed_sec": round(time.perf_counter() - t0_all, 2),
        "splits": results,
    }
    report_path = REPORTS_P3_DIR / "v2_9class_ensemble_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== SAVED BEST 9-CLASS LIBROSA ENSEMBLE SUMMARY ===", flush=True)
    for sp in ("train", "validation", "test", "real_test"):
        m = results[sp]
        print(
            f"  {sp:10s} (N={m['n_samples']:4d}): Acc={m['accuracy']:.4f} | BalAcc={m['balanced_accuracy']:.4f} | "
            f"MacroF1={m['macro_f1']:.4f} | FHR={m['false_healthy_rate']:.4f} | {m['infer_ms_per_clip']} ms/clip",
            flush=True,
        )
    print("\n=== TEST PER-CLASS PERFORMANCE (N=297) ===", flush=True)
    for c, s in results["test"]["per_class"].items():
        print(f"  {c:18s}: P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} (N={s['support']})", flush=True)
    print(f"Artifact: {bundle_path} ({report['artifact_mb']} MB)")


if __name__ == "__main__":
    main()
