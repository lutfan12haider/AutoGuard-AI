"""
ml/v2/phase3_consolidate.py
---------------------------
AutoGuard AI V2 — Consolidates all Track 1 Tabular, Deep Neural, Validation-Selected
Late-Fusion Blend, V1-vs-V2 Ablation, and Track 2 Pre-Training Audit metrics into
one master report (`reports/v2/phase3/phase3_master_summary.json`).

Zero leakage: Any blend weights over `(HybridNet_V2, CNN_V2, HistGB_Multiscale, SVM_RBF_Window)`
are searched ONLY on the `validation` split (`296` clips) optimizing `validation` Macro-F1
and then evaluated untouched ONCE on `test` (`297`) and `real_test` (`38`).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from ml.v2.metrics_utils import MODELS_V2_DIR, REPORTS_P3_DIR, compute_full_metrics
from ml.v2.splitter import CLASSES_9

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK1_DIR = ROOT / "data" / "v2_prepared" / "single_fault_9class"
OUT_MODELS_DIR = MODELS_V2_DIR / "track1"


def main() -> None:
    with open(REPORTS_P3_DIR / "track1_tabular_results.json", encoding="utf-8") as f:
        tab_res = json.load(f)
    with open(REPORTS_P3_DIR / "track1_neural_results.json", encoding="utf-8") as f:
        neu_res = json.load(f)
    with open(REPORTS_P3_DIR / "v1_vs_v2_ablation_results.json", encoding="utf-8") as f:
        abl_res = json.load(f)
    with open(REPORTS_P3_DIR / "track2_pretraining_audit.json", encoding="utf-8") as f:
        t2_res = json.load(f)

    tab_cache = np.load(OUT_MODELS_DIR / "tabular_prob_caches.npz", allow_pickle=False)
    neu_cache = np.load(OUT_MODELS_DIR / "neural_prob_caches.npz", allow_pickle=False)
    splits = {sp: dict(np.load(TRACK1_DIR / f"{sp}.npz", allow_pickle=False)) for sp in ("train", "validation", "test", "real_test")}

    # Probability-level Multimodal Ensemble (Weights tuned STRICTLY on Validation Clip Macro-F1)
    # Candidate streams (Clip-level probs across 296 validation clips):
    #   p_hyb: HybridNet_V2 mean_prob
    #   p_cnn: CNN_V2 mean_prob
    #   p_hgb: HistGB_Multiscale clip_probs
    #   p_svm: SVM_RBF_Multiscale clip_probs (or SVM_RBF_Window mean_prob)
    def _get_probs(sp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_hyb = neu_cache[f"hybridnet_v2__{sp}_clip_probs_mean_prob"]
        p_cnn = neu_cache[f"cnn_v2__{sp}_clip_probs_mean_prob"]
        p_hgb = tab_cache[f"histgb_ms__{sp}_clip_probs"]
        p_svm = tab_cache[f"svm_win__{sp}_clip_probs_mean_prob"]
        return p_hyb, p_cnn, p_hgb, p_svm

    va_hyb, va_cnn, va_hgb, va_svm = _get_probs("validation")
    y_va = splits["validation"]["y_clip"]

    best_f1 = -1.0
    best_w = (0.34, 0.0, 0.33, 0.33)
    grid = [0.0, 0.15, 0.25, 0.35, 0.50]
    for wh in grid:
        for wc in [0.0, 0.15, 0.25]:
            for wg in grid:
                for ws in grid:
                    tot = wh + wc + wg + ws
                    if tot < 0.2:
                        continue
                    w = (wh / tot, wc / tot, wg / tot, ws / tot)
                    p = w[0] * va_hyb + w[1] * va_cnn + w[2] * va_hgb + w[3] * va_svm
                    m = compute_full_metrics(y_va, np.argmax(p, axis=1), CLASSES_9)
                    if m["macro_f1"] > best_f1:
                        best_f1 = m["macro_f1"]
                        best_w = tuple(round(x, 4) for x in w)

    ensemble_res = {
        "model_name": "Multimodal_Ensemble_ValTuned",
        "blend_weights": {
            "HybridNet_V2": best_w[0],
            "CNN_V2": best_w[1],
            "HistGB_Multiscale": best_w[2],
            "SVM_RBF_Window": best_w[3],
        },
        "clip_level": {},
    }
    for sp in ("train", "validation", "test", "real_test"):
        ph, pc, pg, ps = _get_probs(sp)
        p_ens = best_w[0] * ph + best_w[1] * pc + best_w[2] * pg + best_w[3] * ps
        ensemble_res["clip_level"][sp] = compute_full_metrics(splits[sp]["y_clip"], np.argmax(p_ens, axis=1), CLASSES_9)

    # Unified Model Comparison Table (all Track 1 models at primary clip-level representation)
    comparison_table = []

    def _row_from_metrics(name: str, family: str, rep: str, tr_time: float, inf_ms: float | None, m_tr: dict, m_va: dict, m_te: dict, m_rt: dict) -> dict:
        return {
            "model": name,
            "family": family,
            "representation": rep,
            "train_time_sec": tr_time,
            "infer_ms_per_unit": inf_ms,
            "train_acc": m_tr["accuracy"],
            "train_macro_f1": m_tr["macro_f1"],
            "val_acc": m_va["accuracy"],
            "val_bal_acc": m_va["balanced_accuracy"],
            "val_macro_f1": m_va["macro_f1"],
            "val_fhr": m_va["false_healthy_rate"],
            "test_acc": m_te["accuracy"],
            "test_bal_acc": m_te["balanced_accuracy"],
            "test_macro_f1": m_te["macro_f1"],
            "test_fhr": m_te["false_healthy_rate"],
            "real_test_acc": m_rt["accuracy"],
            "overfit_gap_acc": round(m_tr["accuracy"] - m_va["accuracy"], 4),
            "overfit_gap_f1": round(m_tr["macro_f1"] - m_va["macro_f1"], 4),
        }

    for k, m in tab_res["models"].items():
        if "clip_aggregations" in m:
            comparison_table.append(_row_from_metrics(
                k, "Tabular", m["input_mode"], m["train_time_sec"], m.get("eval_infer_ms_per_window"),
                m["clip_aggregations"]["train"]["mean_prob"],
                m["clip_aggregations"]["validation"]["mean_prob"],
                m["clip_aggregations"]["test"]["mean_prob"],
                m["clip_aggregations"]["real_test"]["mean_prob"],
            ))
        else:
            comparison_table.append(_row_from_metrics(
                k, "Tabular (Multiscale)", m["input_mode"], m["train_time_sec"], m["clip_level"]["test"].get("infer_ms_per_clip"),
                m["clip_level"]["train"],
                m["clip_level"]["validation"],
                m["clip_level"]["test"],
                m["clip_level"]["real_test"],
            ))

    for k, m in neu_res["models"].items():
        comparison_table.append(_row_from_metrics(
            k, "Deep Learning", m["input_mode"], m["train_time_sec"], m.get("eval_infer_ms_per_window"),
            m["clip_aggregations"]["train"]["mean_prob"],
            m["clip_aggregations"]["validation"]["mean_prob"],
            m["clip_aggregations"]["test"]["mean_prob"],
            m["clip_aggregations"]["real_test"]["mean_prob"],
        ))

    comparison_table.append(_row_from_metrics(
        "Multimodal_Ensemble_ValTuned", "Late Fusion Ensemble", f"weights={best_w}", 0.0, None,
        ensemble_res["clip_level"]["train"],
        ensemble_res["clip_level"]["validation"],
        ensemble_res["clip_level"]["test"],
        ensemble_res["clip_level"]["real_test"],
    ))

    master = {
        "track1_comparison_table": comparison_table,
        "track1_ensemble": ensemble_res,
        "v1_vs_v2_ablation_summary": abl_res,
        "track2_pretraining_audit_summary": t2_res,
    }
    out_p = REPORTS_P3_DIR / "phase3_master_summary.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print("\n=== MASTER TRACK 1 COMPARISON TABLE (9-CLASS LEAKAGE-FREE) ===", flush=True)
    for row in comparison_table:
        print(
            f"  {row['model']:28s} | Val Acc={row['val_acc']:.4f} Bal={row['val_bal_acc']:.4f} F1={row['val_macro_f1']:.4f} FHR={row['val_fhr']:.4f} | "
            f"Test Acc={row['test_acc']:.4f} Bal={row['test_bal_acc']:.4f} F1={row['test_macro_f1']:.4f} FHR={row['test_fhr']:.4f} | "
            f"RT={row['real_test_acc']:.4f} | GapF1={row['overfit_gap_f1']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
