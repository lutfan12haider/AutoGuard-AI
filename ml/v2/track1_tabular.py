"""
ml/v2/track1_tabular.py
-----------------------
AutoGuard AI V2 — Phase 3 Track 1 Tabular Baselines (9 Single-Fault Classes).

Trains and compares CPU-friendly models under identical leakage-free session splits:
  1. SVM_RBF          (Window 336-dim -> aggregated to Clip)
  2. RandomForest     (Window 336-dim -> aggregated to Clip)
  3. HistGB           (Window 336-dim -> aggregated to Clip; plus XGBoost if installed)
  4. SVM_Multiscale   (Clip [WinMean(336) + ClipGlobal(336)] = 672-dim)
  5. HistGB_Multiscale(Clip [WinMean(336) + ClipGlobal(336)] = 672-dim)

Strict anti-leakage discipline:
  - StandardScaler & class balancing fit STRICTLY on `train`.
  - Hyperparameter tuning selects strictly by `validation` Clip Macro-F1.
  - `test` and `real_test` are evaluated ONCE cleanly at the end and NEVER used for tuning.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.v2.features_v2 import TABULAR_FEATURE_NAMES
from ml.v2.metrics_utils import (
    MODELS_V2_DIR,
    REPORTS_P3_DIR,
    SEED,
    aggregate_windows_to_clips,
    compute_full_metrics,
    set_seed,
)
from ml.v2.splitter import CLASSES_9

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK1_DATA_DIR = ROOT / "data" / "v2_prepared" / "single_fault_9class"
OUT_MODELS_DIR = MODELS_V2_DIR / "track1"
OUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_track1_splits() -> dict[str, dict[str, np.ndarray]]:
    splits = {}
    for sp in ("train", "validation", "test", "real_test"):
        d = np.load(TRACK1_DATA_DIR / f"{sp}.npz", allow_pickle=False)
        splits[sp] = {k: d[k] for k in d.files}
    return splits


def _build_multiscale_clip_features(split_data: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate [clip_mean_window_tab (336), clip_global_tab (336)] -> (N_clips, 672)."""
    x_win = split_data["X_win_tab"]
    w_clip = split_data["win_clip_idx"]
    x_glob = split_data["X_clip_global_tab"]
    n_clips = x_glob.shape[0]
    x_win_mean = np.zeros_like(x_glob, dtype=np.float32)
    for ci in range(n_clips):
        x_win_mean[ci] = x_win[w_clip == ci].mean(axis=0)
    return np.hstack([x_win_mean, x_glob]).astype(np.float32)


def _eval_window_model(
    name: str,
    model,
    splits: dict[str, dict[str, np.ndarray]],
    train_time_sec: float,
    best_params: dict,
    rms_feat_idx: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate a trained window-level classifier across splits & all clip aggregations."""
    probs_cache: dict[str, np.ndarray] = {}
    out: dict = {
        "model_name": name,
        "input_mode": "window_1.5s_336dim",
        "best_params": best_params,
        "train_time_sec": round(train_time_sec, 3),
        "window_level": {},
        "clip_aggregations": {},
        "clip_length_breakdown": {},
    }

    t_inf_start = time.perf_counter()
    n_inf_samples = 0
    for sp in ("train", "validation", "test", "real_test"):
        d = splits[sp]
        X_w = d["X_win_tab"]
        y_w = d["y_win"]
        n_clips = int(d["y_clip"].shape[0])

        t0 = time.perf_counter()
        p_w = model.predict_proba(X_w).astype(np.float32)
        if sp in ("validation", "test"):
            n_inf_samples += len(X_w)
        inf_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(X_w))

        pred_w = np.argmax(p_w, axis=1)
        m_win = compute_full_metrics(y_w, pred_w, CLASSES_9)
        m_win["infer_ms_per_window"] = round(inf_ms, 4)
        out["window_level"][sp] = m_win
        probs_cache[f"{sp}_win_probs"] = p_w

        win_rms = np.power(10.0, X_w[:, rms_feat_idx])
        agg_dict = aggregate_windows_to_clips(p_w, d["win_clip_idx"], n_clips, win_rms=win_rms)
        out["clip_aggregations"][sp] = {}
        for agg_name, (p_c, pred_c) in agg_dict.items():
            out["clip_aggregations"][sp][agg_name] = compute_full_metrics(d["y_clip"], pred_c, CLASSES_9)
            probs_cache[f"{sp}_clip_probs_{agg_name}"] = p_c

        # Breakdown by single-window (1.5s) vs multi-window (long clips, K>=2) on val/test
        if sp in ("validation", "test"):
            counts = np.bincount(d["win_clip_idx"], minlength=n_clips)
            p_mean, pred_mean = agg_dict["mean_prob"]
            p_conf, pred_conf = agg_dict["conf_weighted"]
            p_top2, pred_top2 = agg_dict["top2_conf"]
            multi_mask = counts >= 2
            single_mask = counts == 1
            out["clip_length_breakdown"][sp] = {
                "single_window_n": int(single_mask.sum()),
                "single_window_mean_acc": round(float(np.mean(pred_mean[single_mask] == d["y_clip"][single_mask])), 4),
                "multi_window_n": int(multi_mask.sum()),
                "multi_window_win_acc": round(float(np.mean(pred_w[np.isin(d["win_clip_idx"], np.where(multi_mask)[0])] == y_w[np.isin(d["win_clip_idx"], np.where(multi_mask)[0])])), 4),
                "multi_window_mean_acc": round(float(np.mean(pred_mean[multi_mask] == d["y_clip"][multi_mask])), 4),
                "multi_window_conf_acc": round(float(np.mean(pred_conf[multi_mask] == d["y_clip"][multi_mask])), 4),
                "multi_window_top2_acc": round(float(np.mean(pred_top2[multi_mask] == d["y_clip"][multi_mask])), 4),
            }

    out["eval_infer_ms_per_window"] = round((time.perf_counter() - t_inf_start) * 1000.0 / max(1, n_inf_samples), 4)
    return out, probs_cache


def main() -> None:
    set_seed(SEED)
    t_all = time.perf_counter()
    splits = load_track1_splits()
    rms_feat_idx = TABULAR_FEATURE_NAMES.index("log10_raw_rms")

    X_tr_w, y_tr_w = splits["train"]["X_win_tab"], splits["train"]["y_win"]
    X_va_w, y_va_w = splits["validation"]["X_win_tab"], splits["validation"]["y_win"]
    n_va_clips = int(splits["validation"]["y_clip"].shape[0])
    y_va_clip = splits["validation"]["y_clip"]

    all_results: dict[str, dict] = {}
    all_prob_caches: dict[str, np.ndarray] = {}

    def _tune_window_candidates(label: str, candidates: list[tuple[dict, object]]) -> tuple[object, dict, float]:
        best_score = -1.0
        best_model = None
        best_params = {}
        best_t = 0.0
        for params, clf in candidates:
            t0 = time.perf_counter()
            clf.fit(X_tr_w, y_tr_w)
            fit_t = time.perf_counter() - t0
            p_va = clf.predict_proba(X_va_w)
            agg = aggregate_windows_to_clips(p_va, splits["validation"]["win_clip_idx"], n_va_clips)
            m_va = compute_full_metrics(y_va_clip, agg["mean_prob"][1], CLASSES_9)
            score = m_va["macro_f1"]
            print(
                f"  [Tune {label}] params={params} -> val_clip_acc={m_va['accuracy']:.4f}, "
                f"val_clip_macroF1={score:.4f}, FHR={m_va['false_healthy_rate']:.4f} ({fit_t:.2f}s)",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_model = clf
                best_params = params
                best_t = fit_t
        return best_model, best_params, best_t

    # ── 1. SVM (RBF, Window 336-dim) ───────────────────────────────────────────
    print("\n[1/5] Training SVM_RBF (Window 336-dim) ...", flush=True)
    svm_cands = [
        (
            {"C": C, "gamma": "scale"},
            Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
            ]),
        )
        for C in (1.0, 3.0, 10.0)
    ]
    svm_m, svm_p, svm_t = _tune_window_candidates("SVM_RBF", svm_cands)
    joblib.dump(svm_m, OUT_MODELS_DIR / "svm_win.joblib")
    res_svm, cache_svm = _eval_window_model("SVM_RBF (Window)", svm_m, splits, svm_t, svm_p, rms_feat_idx)
    all_results["SVM_RBF_Window"] = res_svm
    for k, v in cache_svm.items():
        all_prob_caches[f"svm_win__{k}"] = v

    # ── 2. Random Forest (Window 336-dim) ──────────────────────────────────────
    print("\n[2/5] Training RandomForest (Window 336-dim) ...", flush=True)
    rf_cands = [
        (
            {"max_features": mf, "min_samples_leaf": msl, "n_estimators": 400},
            RandomForestClassifier(
                n_estimators=400,
                max_features=mf,
                min_samples_leaf=msl,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            ),
        )
        for mf in ("sqrt", 0.15)
        for msl in (1, 2, 4)
    ]
    rf_m, rf_p, rf_t = _tune_window_candidates("RandomForest", rf_cands)
    joblib.dump(rf_m, OUT_MODELS_DIR / "rf_win.joblib")
    res_rf, cache_rf = _eval_window_model("RandomForest (Window)", rf_m, splits, rf_t, rf_p, rms_feat_idx)
    all_results["RandomForest_Window"] = res_rf
    for k, v in cache_rf.items():
        all_prob_caches[f"rf_win__{k}"] = v

    # ── 3. HistGradientBoosting (Window 336-dim) ───────────────────────────────
    print("\n[3/5] Training HistGradientBoosting (Window 336-dim) ...", flush=True)
    hgb_cands = [
        (
            {"max_iter": 250, "learning_rate": lr, "max_leaf_nodes": mln, "l2_regularization": l2},
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=lr,
                max_leaf_nodes=mln,
                l2_regularization=l2,
                class_weight="balanced",
                early_stopping=False,
                random_state=SEED,
            ),
        )
        for lr in (0.04, 0.08)
        for mln in (31, 63)
        for l2 in (0.1, 1.0)
    ]
    hgb_m, hgb_p, hgb_t = _tune_window_candidates("HistGB", hgb_cands)
    joblib.dump(hgb_m, OUT_MODELS_DIR / "histgb_win.joblib")
    res_hgb, cache_hgb = _eval_window_model("HistGradientBoosting (Window)", hgb_m, splits, hgb_t, hgb_p, rms_feat_idx)
    all_results["HistGB_Window"] = res_hgb
    for k, v in cache_hgb.items():
        all_prob_caches[f"histgb_win__{k}"] = v

    has_xgb = importlib.util.find_spec("xgboost") is not None
    print(f"  [Info] xgboost installed: {has_xgb}", flush=True)

    # ── 4 & 5. Clip-Multiscale Tabular Models ([WinMean 336 + ClipGlobal 336]) ─
    ms_feats = {sp: _build_multiscale_clip_features(splits[sp]) for sp in ("train", "validation", "test", "real_test")}
    y_tr_c = splits["train"]["y_clip"]

    def _fit_clip_model(label: str, cands: list[tuple[dict, object]], save_name: str, key_prefix: str):
        best_score = -1.0
        best_m, best_p, best_t = None, {}, 0.0
        for params, clf in cands:
            t0 = time.perf_counter()
            clf.fit(ms_feats["train"], y_tr_c)
            ft = time.perf_counter() - t0
            p_va = clf.predict_proba(ms_feats["validation"])
            m_va = compute_full_metrics(y_va_clip, np.argmax(p_va, axis=1), CLASSES_9)
            score = m_va["macro_f1"]
            print(f"  [Tune {label}] params={params} -> val_clip_acc={m_va['accuracy']:.4f}, val_clip_macroF1={score:.4f}, FHR={m_va['false_healthy_rate']:.4f} ({ft:.2f}s)", flush=True)
            if score > best_score:
                best_score, best_m, best_p, best_t = score, clf, params, ft
        joblib.dump(best_m, OUT_MODELS_DIR / save_name)
        res = {
            "model_name": label,
            "input_mode": "clip_multiscale_winmean336_plus_global336",
            "best_params": best_p,
            "train_time_sec": round(best_t, 3),
            "clip_level": {},
        }
        for sp in ("train", "validation", "test", "real_test"):
            t0 = time.perf_counter()
            p_sp = best_m.predict_proba(ms_feats[sp]).astype(np.float32)
            inf_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(ms_feats[sp]))
            m_sp = compute_full_metrics(splits[sp]["y_clip"], np.argmax(p_sp, axis=1), CLASSES_9)
            m_sp["infer_ms_per_clip"] = round(inf_ms, 4)
            res["clip_level"][sp] = m_sp
            all_prob_caches[f"{key_prefix}__{sp}_clip_probs"] = p_sp
        return res

    print("\n[4/5] Training SVM_RBF (Multiscale Clip 672-dim) ...", flush=True)
    svm_ms_cands = [
        (
            {"C": C, "gamma": "scale"},
            Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
            ]),
        )
        for C in (1.0, 3.0, 10.0)
    ]
    all_results["SVM_RBF_Multiscale"] = _fit_clip_model(
        "SVM_RBF (Multiscale Clip)", svm_ms_cands, "svm_multiscale.joblib", "svm_ms"
    )

    print("\n[5/5] Training HistGradientBoosting (Multiscale Clip 672-dim) ...", flush=True)
    hgb_ms_cands = [
        (
            {"max_iter": 250, "learning_rate": lr, "max_leaf_nodes": mln, "l2_regularization": l2},
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=lr,
                max_leaf_nodes=mln,
                l2_regularization=l2,
                class_weight="balanced",
                early_stopping=False,
                random_state=SEED,
            ),
        )
        for lr in (0.04, 0.08)
        for mln in (15, 31)
        for l2 in (0.5, 2.0)
    ]
    all_results["HistGB_Multiscale"] = _fit_clip_model(
        "HistGB (Multiscale Clip)", hgb_ms_cands, "histgb_multiscale.joblib", "histgb_ms"
    )

    np.savez_compressed(OUT_MODELS_DIR / "tabular_prob_caches.npz", **all_prob_caches)
    payload = {
        "seed": SEED,
        "xgboost_available": has_xgb,
        "total_elapsed_sec": round(time.perf_counter() - t_all, 2),
        "models": all_results,
    }
    with open(REPORTS_P3_DIR / "track1_tabular_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== TRACK 1 TABULAR SUMMARY (CLIP-LEVEL TEST) ===", flush=True)
    for k, m in all_results.items():
        if "clip_aggregations" in m:
            te_m = m["clip_aggregations"]["test"]["mean_prob"]
            va_m = m["clip_aggregations"]["validation"]["mean_prob"]
            rt_m = m["clip_aggregations"]["real_test"]["mean_prob"]
            print(
                f"  {k:24s} | Val Acc={va_m['accuracy']:.4f} F1={va_m['macro_f1']:.4f} | "
                f"Test Acc={te_m['accuracy']:.4f} BalAcc={te_m['balanced_accuracy']:.4f} F1={te_m['macro_f1']:.4f} FHR={te_m['false_healthy_rate']:.4f} | "
                f"RealTest Acc={rt_m['accuracy']:.4f}",
                flush=True,
            )
        else:
            te_m = m["clip_level"]["test"]
            va_m = m["clip_level"]["validation"]
            rt_m = m["clip_level"]["real_test"]
            print(
                f"  {k:24s} | Val Acc={va_m['accuracy']:.4f} F1={va_m['macro_f1']:.4f} | "
                f"Test Acc={te_m['accuracy']:.4f} BalAcc={te_m['balanced_accuracy']:.4f} F1={te_m['macro_f1']:.4f} FHR={te_m['false_healthy_rate']:.4f} | "
                f"RealTest Acc={rt_m['accuracy']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
