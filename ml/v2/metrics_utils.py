"""
ml/v2/metrics_utils.py
----------------------
AutoGuard AI V2 — Deterministic seeding, window-to-clip aggregation strategies,
and comprehensive Track 1 / Track 2 classification metrics (including safety-critical
False-Healthy Rate).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.v2.splitter import CLASSES_9

SEED = 42
MODELS_V2_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "v2"
REPORTS_P3_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "v2" / "phase3"
FIG_P3_DIR = REPORTS_P3_DIR / "figures"

for _d in (MODELS_V2_DIR, REPORTS_P3_DIR, FIG_P3_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Fix all Python, NumPy, and PyTorch CPU seeds for strict reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] = CLASSES_9,
    healthy_idx: int = 0,
) -> dict:
    """
    Compute accuracy, balanced accuracy, macro-F1, weighted-F1, per-class P/R/F1,
    confusion matrix, and safety-critical false-healthy metrics.
    """
    n_classes = len(class_names)
    labels = list(range(n_classes))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_p = float(np.mean(p))
    macro_r = float(np.mean(r))
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * sup) / max(1, int(np.sum(sup))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    faulty_mask = y_true != healthy_idx
    n_faulty = int(np.sum(faulty_mask))
    false_healthy_cnt = int(np.sum(y_pred[faulty_mask] == healthy_idx)) if n_faulty > 0 else 0
    false_healthy_rate = float(false_healthy_cnt / n_faulty) if n_faulty > 0 else 0.0

    healthy_mask = y_true == healthy_idx
    n_healthy = int(np.sum(healthy_mask))
    false_alarm_cnt = int(np.sum(y_pred[healthy_mask] != healthy_idx)) if n_healthy > 0 else 0
    false_alarm_rate = float(false_alarm_cnt / n_healthy) if n_healthy > 0 else 0.0

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        }

    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "n_faulty": n_faulty,
        "false_healthy_count": false_healthy_cnt,
        "false_healthy_rate": round(false_healthy_rate, 4),
        "fault_detection_recall": round(1.0 - false_healthy_rate, 4),
        "n_healthy": n_healthy,
        "false_alarm_count": false_alarm_cnt,
        "false_alarm_rate": round(false_alarm_rate, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def aggregate_windows_to_clips(
    win_probs: np.ndarray,
    win_clip_idx: np.ndarray,
    n_clips: int,
    win_rms: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Aggregate `(N_windows, C)` class probabilities into `(N_clips, C)` clip-level
    probabilities and integer predictions under multiple principled policies:
      1. `mean_prob`: Uniform arithmetic mean across all windows of the clip.
      2. `majority_vote`: Hard argmax vote across windows (tie-broken by `mean_prob`).
      3. `log_prob_geom`: Normalized geometric mean (down-weights rogue outlier zeros).
      4. `conf_weighted`: Weighted by `(1 - normalized Shannon entropy)^2` so uncertain
         ambient/quiet windows carry less weight than clear fault windows.
      5. `top2_conf`: Averages the Top-K (`K = min(2, n_win)`) lowest-entropy windows per
         clip — preserves transient faults inside long 10s recordings where fault events
         occur only during part of the clip.
      6. `energy_conf_weighted`: Multiplies confidence weight by window relative acoustic
         energy (`sqrt(rms)`) if `win_rms` is supplied.
    """
    n_classes = win_probs.shape[1]
    eps = 1e-7
    probs_c = np.clip(win_probs.astype(np.float64), eps, 1.0)
    ent = -np.sum(probs_c * np.log(probs_c), axis=1) / np.log(n_classes)
    conf_w = np.clip(1.0 - ent, 0.02, 1.0) ** 2

    if win_rms is not None:
        rms_arr = np.maximum(win_rms.astype(np.float64), 1e-5)
    else:
        rms_arr = np.ones(len(win_probs), dtype=np.float64)

    grouped: list[list[int]] = [[] for _ in range(n_clips)]
    for wi, ci in enumerate(win_clip_idx):
        grouped[int(ci)].append(wi)

    out_probs: dict[str, np.ndarray] = {
        k: np.zeros((n_clips, n_classes), dtype=np.float32)
        for k in (
            "mean_prob",
            "majority_vote",
            "log_prob_geom",
            "conf_weighted",
            "top2_conf",
            "energy_conf_weighted",
        )
    }

    for ci, idxs in enumerate(grouped):
        P = probs_c[idxs]
        cw = conf_w[idxs]

        # 1. Mean prob
        p_mean = P.mean(axis=0)
        p_mean /= p_mean.sum()
        out_probs["mean_prob"][ci] = p_mean

        # 2. Majority vote (soft fraction + tiny mean_prob tie-breaker)
        votes = np.bincount(np.argmax(P, axis=1), minlength=n_classes).astype(np.float64)
        p_vote = votes / len(idxs) + 1e-4 * p_mean
        p_vote /= p_vote.sum()
        out_probs["majority_vote"][ci] = p_vote

        # 3. Geometric mean / log-prob
        p_geo = np.exp(np.log(P).mean(axis=0))
        p_geo /= p_geo.sum()
        out_probs["log_prob_geom"][ci] = p_geo

        # 4. Confidence weighted
        w_sum = cw.sum() + eps
        p_cw = (cw[:, None] * P).sum(axis=0) / w_sum
        p_cw /= p_cw.sum()
        out_probs["conf_weighted"][ci] = p_cw

        # 5. Top-2 most confident windows (preserves intermittent events in 10s clips)
        k_top = min(2, len(idxs))
        top_local = np.argsort(cw)[-k_top:]
        p_top2 = P[top_local].mean(axis=0)
        p_top2 /= p_top2.sum()
        out_probs["top2_conf"][ci] = p_top2

        # 6. Energy x Confidence weighted
        ew = cw * np.sqrt(rms_arr[idxs] / (np.mean(rms_arr[idxs]) + eps))
        p_ew = (ew[:, None] * P).sum(axis=0) / (ew.sum() + eps)
        p_ew /= p_ew.sum()
        out_probs["energy_conf_weighted"][ci] = p_ew

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, arr in out_probs.items():
        results[k] = (arr, np.argmax(arr, axis=1).astype(np.int32))
    return results
