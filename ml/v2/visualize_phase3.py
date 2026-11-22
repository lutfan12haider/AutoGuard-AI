"""
ml/v2/visualize_phase3.py
-------------------------
AutoGuard AI V2 — Phase 3 Comprehensive Diagnostic Visualizations.

Generates 6 high-DPI plots under `reports/v2/phase3/figures/`:
  p3_01_model_comparison_9class.png
  p3_02_window_vs_clip_aggregation.png
  p3_03_v1_vs_v2_ablation.png
  p3_04_learning_curves_and_overfitting.png
  p3_05_confusion_matrices_and_per_class.png
  p3_06_track2_combined_triage.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.v2.metrics_utils import FIG_P3_DIR, REPORTS_P3_DIR
from ml.v2.splitter import CLASSES_9

ROOT = Path(__file__).resolve().parent.parent.parent
PREP_DIR = ROOT / "data" / "v2_prepared"

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "axes.titlecolor": "#f8fafc",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "text.color": "#e2e8f0",
    "grid.color": "#1e293b",
    "grid.alpha": 0.7,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "font.size": 9,
})


def plot_01_model_comparison(master: dict) -> Path:
    rows = master["track1_comparison_table"]
    names = [r["model"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    w = 0.20

    te_acc = [r["test_acc"] * 100 for r in rows]
    te_bal = [r["test_bal_acc"] * 100 for r in rows]
    te_f1 = [r["test_macro_f1"] * 100 for r in rows]
    te_fhr = [r["test_fhr"] * 100 for r in rows]
    va_f1 = [r["val_macro_f1"] * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4), dpi=160, gridspec_kw={"width_ratios": [1.45, 1.0]})

    b1 = ax1.bar(x - 1.5 * w, te_acc, w, label="Test Accuracy (%)", color="#38bdf8")
    b2 = ax1.bar(x - 0.5 * w, te_bal, w, label="Test Balanced Acc (%)", color="#a78bfa")
    b3 = ax1.bar(x + 0.5 * w, te_f1, w, label="Test Macro-F1 (%)", color="#34d399")
    b4 = ax1.bar(x + 1.5 * w, va_f1, w, label="Val Macro-F1 (%)", color="#fbbf24")
    for b in (b1, b3):
        for bar in b:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylim(0, 96)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Track 1 (9 Classes): Leakage-Free Clip-Level Performance", fontweight="bold")
    ax1.grid(axis="y")
    ax1.legend(loc="upper right", fontsize=8)

    # Safety chart: Test False-Healthy Rate vs Real-Test Engine Knock Recall
    rt_acc = [r["real_test_acc"] * 100 for r in rows]
    y_pos = np.arange(len(rows))
    ax2.barh(y_pos - 0.18, te_fhr, height=0.35, label="Test False-Healthy Rate % (LOWER = SAFER)", color="#ef4444")
    ax2.barh(y_pos + 0.18, rt_acc, height=0.35, label="Real-Test Accuracy % (38 Knock Clips)", color="#10b981")
    for i, (fhr, rt) in enumerate(zip(te_fhr, rt_acc)):
        ax2.text(fhr + 1.2, i - 0.18, f"{fhr:.1f}%", va="center", fontsize=7.8, color="#fca5a5")
        ax2.text(rt + 1.2, i + 0.18, f"{rt:.1f}%", va="center", fontsize=7.8, color="#6ee7b7")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r["model"] for r in rows], fontsize=8.2)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 114)
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("Safety-Critical Fault Miss Rate & Real-World Knock Generalization", fontweight="bold")
    ax2.grid(axis="x")
    ax2.legend(loc="lower right", fontsize=7.6)

    fig.suptitle("AutoGuard AI V2 — Phase 3 Model Comparison (9 Single-Fault Classes)", fontsize=13, fontweight="bold")
    out = FIG_P3_DIR / "p3_01_model_comparison_9class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_02_window_vs_clip(tab_res: dict, neu_res: dict) -> Path:
    models = []
    for k, d in tab_res["models"].items():
        if "clip_aggregations" in d:
            models.append((k, d))
    for k, d in neu_res["models"].items():
        models.append((k, d))

    agg_keys = ["window_level", "mean_prob", "majority_vote", "log_prob_geom", "conf_weighted", "top2_conf", "energy_conf_weighted"]
    agg_labels = ["Raw\nWindow", "Mean\nProb", "Majority\nVote", "Geom.\nMean", "Conf.\nWeighted", "Top-2\nConf", "Energy×Conf\nWeighted"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=160)
    palette = ["#38bdf8", "#a78bfa", "#f59e0b", "#34d399", "#f43f5e"]

    x = np.arange(len(agg_keys))
    for i, (mname, mdict) in enumerate(models):
        f1s = []
        for ak in agg_keys:
            if ak == "window_level":
                f1s.append(mdict["window_level"]["test"]["macro_f1"] * 100)
            else:
                f1s.append(mdict["clip_aggregations"]["test"][ak]["macro_f1"] * 100)
        ax1.plot(x, f1s, "o-", linewidth=2, label=mname, color=palette[i % len(palette)])

    ax1.set_xticks(x)
    ax1.set_xticklabels(agg_labels, fontsize=8)
    ax1.set_ylabel("Test Macro-F1 (%)")
    ax1.set_title("Test Macro-F1 Across Window & Clip Aggregation Policies", fontweight="bold")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # Panel B: Long multi-window clips (K >= 2) window accuracy vs aggregated clip accuracies
    xm = np.arange(len(models))
    win_acc = [md["clip_length_breakdown"]["test"]["multi_window_win_acc"] * 100 for _, md in models]
    mean_acc = [md["clip_length_breakdown"]["test"]["multi_window_mean_acc"] * 100 for _, md in models]
    conf_acc = [md["clip_length_breakdown"]["test"]["multi_window_conf_acc"] * 100 for _, md in models]
    w = 0.24
    b1 = ax2.bar(xm - w, win_acc, w, label="Raw Window Acc (Long Clips)", color="#64748b")
    b2 = ax2.bar(xm, mean_acc, w, label="Clip Mean Prob Acc", color="#38bdf8")
    b3 = ax2.bar(xm + w, conf_acc, w, label="Clip Conf-Weighted Acc", color="#10b981")
    for b in (b1, b3):
        for bar in b:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.2)

    ax2.set_xticks(xm)
    ax2.set_xticklabels([m[0].replace("_", "\n") for m in models], fontsize=8)
    ax2.set_ylim(max(50, min(win_acc) - 10), 100)
    ax2.set_ylabel("Accuracy (%) on Multi-Window Test Clips (N=145)")
    ax2.set_title("Reliability Gain of Clip Aggregation on Longer Recordings (K >= 2)", fontweight="bold")
    ax2.grid(axis="y")
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("AutoGuard AI V2 — Window-Level vs Clip-Level Aggregation Analysis", fontsize=13, fontweight="bold")
    out = FIG_P3_DIR / "p3_02_window_vs_clip_aggregation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_03_ablation(abl: dict, master: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8), dpi=160)

    # Suite A: 3-class KweriMoto
    sa = abl["suite_a_kwerimoto_3class"]
    sa_order = [
        ("A0a_V1_Saved_CNN", "V1 Saved\nCNN"),
        ("A0b_V1_Saved_SVM_288", "V1 Saved\nSVM-288"),
        ("A1_V1_22k_10s_288MFCC_SVM", "A1: 22k 10s\n288 SVM"),
        ("A2_Std16k_10s_288MFCC_SVM", "A2: +Std 16k\n10s SVM"),
        ("A3_Std16k_1.5sWin_288MFCC_SVM", "A3: +1.5s Win\n288 SVM"),
        ("A4a_Std16k_1.5sWin_334V2_NoRawE_SVM", "A4a: +334 V2\n(No RawE)"),
        ("A4b_Std16k_1.5sWin_336V2_WithRawE_SVM", "A4b: +336 V2\n(+RawE)"),
        ("A5a_V1Arch_CNN_5ep_Kweri721", "A5a: V1 CNN\nKweri 721"),
        ("A5b_CNN_V2_15ep_OnV2Mel", "A5b: CNN V2\n15ep"),
        ("A5c_HistGB_Multiscale_3Class", "A5c: HistGB\nMultiscale"),
    ]
    labels_a = [lbl for k, lbl in sa_order if k in sa]
    keys_a = [k for k, _ in sa_order if k in sa]
    acc_a = [sa[k]["test"]["accuracy"] * 100 for k in keys_a]
    f1_a = [sa[k]["test"]["macro_f1"] * 100 for k in keys_a]
    fhr_a = [sa[k]["test"]["false_healthy_rate"] * 100 for k in keys_a]
    xa = np.arange(len(keys_a))

    ax1.plot(xa, acc_a, "o-", label="Test Accuracy (%)", color="#38bdf8", linewidth=2)
    ax1.plot(xa, f1_a, "s-", label="Test Macro-F1 (%)", color="#34d399", linewidth=2)
    ax1.bar(xa, fhr_a, width=0.45, alpha=0.28, color="#ef4444", label="False-Healthy Rate (%)")
    for i, f1v in enumerate(f1_a):
        ax1.text(i, f1v + 1.0, f"{f1v:.1f}", ha="center", fontsize=7.3, color="#6ee7b7")
    ax1.set_xticks(xa)
    ax1.set_xticklabels(labels_a, fontsize=7.5, rotation=28, ha="right")
    ax1.set_ylim(0, 102)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Suite A: Exact KweriMoto 3-Class Ablation Ladder (Test N=154)", fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=7.8)

    # Suite B: Full 9-class Track 1
    sb = abl["suite_b_track1_9class"]
    sb_order = [
        ("B0_V1Style_22k_10s_288MFCC_SVM", "B0: V1-288\n22k 10s SVM"),
        ("B0b_V1Style_CNN_5ep_OnV1Mel128", "B0b: V1 CNN\n5ep 128x128"),
        ("B1_Std16k_1.5sWin_288MFCC_SVM", "B1: +Std16k\n1.5s 288 SVM"),
        ("B2a_Std16k_1.5sWin_334V2_NoRawE_SVM", "B2a: +334 V2\nNoRawE SVM"),
        ("B2b_Std16k_1.5sWin_336V2_WithRawE_SVM", "B2b: +336 V2\n+RawE SVM"),
    ]
    labels_b = [lbl for k, lbl in sb_order if k in sb]
    keys_b = [k for k, _ in sb_order if k in sb]
    acc_b = [sb[k]["test"]["accuracy"] * 100 for k in keys_b]
    f1_b = [sb[k]["test"]["macro_f1"] * 100 for k in keys_b]
    fhr_b = [sb[k]["test"]["false_healthy_rate"] * 100 for k in keys_b]

    # Append CNN_V2 and HybridNet_V2 from master table on exact same split
    for r in master["track1_comparison_table"]:
        if r["model"] in ("CNN_V2", "HybridNet_V2"):
            labels_b.append(f"B3: {r['model']}")
            acc_b.append(r["test_acc"] * 100)
            f1_b.append(r["test_macro_f1"] * 100)
            fhr_b.append(r["test_fhr"] * 100)

    xb = np.arange(len(labels_b))
    ax2.plot(xb, acc_b, "o-", label="Test Accuracy (%)", color="#38bdf8", linewidth=2)
    ax2.plot(xb, f1_b, "s-", label="Test Macro-F1 (%)", color="#34d399", linewidth=2)
    ax2.bar(xb, fhr_b, width=0.45, alpha=0.28, color="#ef4444", label="False-Healthy Rate (%)")
    for i, f1v in enumerate(f1_b):
        ax2.text(i, f1v + 1.2, f"{f1v:.1f}", ha="center", fontsize=7.5, color="#6ee7b7")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels_b, fontsize=7.8, rotation=22, ha="right")
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Suite B: Identical 9-Class Split Ablation (Test N=297)", fontweight="bold")
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=7.8)

    fig.suptitle("AutoGuard AI V2 — Controlled V1 vs V2 Ablation (Preprocessing, Features, Raw Energy, Model)", fontsize=12.5, fontweight="bold")
    out = FIG_P3_DIR / "p3_03_v1_vs_v2_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_04_learning_curves(neu_res: dict, master: dict) -> Path:
    fig = plt.figure(figsize=(16, 8.0), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    for idx, (mkey, col_tr, col_va) in enumerate([("CNN_V2", "#38bdf8", "#fbbf24"), ("HybridNet_V2", "#a78bfa", "#34d399")]):
        hist = neu_res["models"][mkey]["history"]
        eps = [h["epoch"] for h in hist]
        best_ep = neu_res["models"][mkey]["best_epoch"]

        ax_l = fig.add_subplot(gs[0, idx])
        ax_l.plot(eps, [h["train_loss"] for h in hist], "o-", label="Train Loss", color=col_tr)
        ax_l.plot(eps, [h["val_loss"] for h in hist], "s--", label="Val Loss", color=col_va)
        ax_l.axvline(best_ep, color="#f43f5e", linestyle=":", label=f"Best Ep ({best_ep})")
        ax_l.set_title(f"{mkey}: Train vs Validation Loss", fontweight="bold")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Loss")
        ax_l.grid(True)
        ax_l.legend(fontsize=8)

        ax_m = fig.add_subplot(gs[1, idx])
        ax_m.plot(eps, [h["train_win_acc"] * 100 for h in hist], "o-", label="Train Win Acc %", color=col_tr)
        ax_m.plot(eps, [h["val_clip_acc"] * 100 for h in hist], "s-", label="Val Clip Acc %", color=col_va)
        ax_m.plot(eps, [h["val_clip_macro_f1"] * 100 for h in hist], "^-", label="Val Clip Macro-F1 %", color="#22c55e")
        ax_m.axvline(best_ep, color="#f43f5e", linestyle=":")
        ax_m.set_title(f"{mkey}: Accuracy & Macro-F1 Learning Curves", fontweight="bold")
        ax_m.set_xlabel("Epoch")
        ax_m.set_ylabel("Score (%)")
        ax_m.set_ylim(20, 102)
        ax_m.grid(True)
        ax_m.legend(fontsize=8)

    fig.suptitle("AutoGuard AI V2 — Neural Training Curves & Early-Stopping Selection", fontsize=13, fontweight="bold")
    out = FIG_P3_DIR / "p3_04_learning_curves_and_overfitting.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_05_confusion_and_per_class(tab_res: dict, neu_res: dict, master: dict) -> Path:
    # Compare Top 3 models' confusion matrices + Per-Class F1 across models
    candidates = [
        ("SVM_RBF (Window)", tab_res["models"]["SVM_RBF_Window"]["clip_aggregations"]["test"]["mean_prob"]),
        ("HistGB (Multiscale)", tab_res["models"]["HistGB_Multiscale"]["clip_level"]["test"]),
        ("HybridNet V2", neu_res["models"]["HybridNet_V2"]["clip_aggregations"]["test"]["mean_prob"]),
    ]
    short = ["hlth", "knk", "misf", "loil", "belt", "pstr", "ign", "batt", "brks"]

    fig = plt.figure(figsize=(17, 9.2), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28)

    for idx, (title, m) in enumerate(candidates):
        ax = fig.add_subplot(gs[0, idx])
        cm = np.array(m["confusion_matrix"], dtype=np.float32)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(len(short)):
            for c in range(len(short)):
                cnt = int(cm[r, c])
                if cnt > 0:
                    txt_col = "#ffffff" if cm_norm[r, c] > 0.48 else "#e2e8f0"
                    ax.text(c, r, str(cnt), ha="center", va="center", fontsize=7.5, color=txt_col, fontweight="bold" if r == c else "normal")
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(short, fontsize=8, rotation=30)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if idx == 0 else "")
        ax.set_title(f"{title}\nAcc={m['accuracy']*100:.1f}% | F1={m['macro_f1']*100:.1f}% | FHR={m['false_healthy_rate']*100:.1f}%", fontsize=9.5, fontweight="bold")

    # Bottom row spanning all 3 cols: Per-Class F1 comparison across models
    ax_pc = fig.add_subplot(gs[1, :])
    pc_models = [
        ("SVM_RBF_Win", tab_res["models"]["SVM_RBF_Window"]["clip_aggregations"]["test"]["mean_prob"]),
        ("RF_Win", tab_res["models"]["RandomForest_Window"]["clip_aggregations"]["test"]["mean_prob"]),
        ("HistGB_MS", tab_res["models"]["HistGB_Multiscale"]["clip_level"]["test"]),
        ("CNN_V2", neu_res["models"]["CNN_V2"]["clip_aggregations"]["test"]["mean_prob"]),
        ("HybridNet_V2", neu_res["models"]["HybridNet_V2"]["clip_aggregations"]["test"]["mean_prob"]),
        ("Multimodal_Ens", master["track1_ensemble"]["clip_level"]["test"]),
    ]
    mat = np.array([[md["per_class"][c]["f1"] * 100 for c in CLASSES_9] for _, md in pc_models])
    im = ax_pc.imshow(mat, cmap="YlGn", vmin=0.0, vmax=100.0, aspect="auto")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            tcol = "#0f172a" if v > 52 else "#f8fafc"
            ax_pc.text(c, r, f"{v:.1f}%", ha="center", va="center", fontsize=8.5, color=tcol, fontweight="bold")
    ax_pc.set_yticks(range(len(pc_models)))
    ax_pc.set_yticklabels([n for n, _ in pc_models], fontsize=9)
    ax_pc.set_xticks(range(9))
    ax_pc.set_xticklabels([f"{c}\n(N={candidates[0][1]['per_class'][c]['support']})" for c in CLASSES_9], fontsize=8.5)
    ax_pc.set_title("Per-Class Test F1-Score (%) Across Track 1 Models (Exposing Strong vs Bottleneck Faults)", fontweight="bold")
    plt.colorbar(im, ax=ax_pc, fraction=0.02, pad=0.02)

    fig.suptitle("AutoGuard AI V2 — Test Confusion Matrices & Per-Class Performance Audit (N=297)", fontsize=13, fontweight="bold")
    out = FIG_P3_DIR / "p3_05_confusion_matrices_and_per_class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_06_track2_triage(t2: dict) -> Path:
    with open(PREP_DIR / "track2_triage_437.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0), dpi=160)

    # Panel A: Triage disposition counts
    disp_order = [
        ("RETAINED_CLEAN_TRAIN", "Retained\nClean Train", "#10b981"),
        ("RETAINED_CLEAN_EVAL", "Retained\nClean Eval", "#38bdf8"),
        ("FLAGGED_MASKED_FOR_REVIEW", "Flagged:\nEnergy-Masked\n(Pure Split)", "#f59e0b"),
        ("EXCLUDED_CROSS_SPLIT_LEAKAGE", "Excluded:\nCross-Split\nLeakage", "#ef4444"),
    ]
    dc = t2["disposition_counts"]
    vals = [dc.get(k, 0) for k, _, _ in disp_order]
    cols = [col for _, _, col in disp_order]
    xl = np.arange(len(disp_order))
    bars = ax1.bar(xl, vals, color=cols, width=0.58)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}\n({v/437*100:.1f}%)", ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(xl)
    ax1.set_xticklabels([lbl for _, lbl, _ in disp_order], fontsize=8.5)
    ax1.set_ylim(0, max(vals) + 42)
    ax1.set_ylabel("Combined Clips (Total N = 437)")
    ax1.set_title("Track 2 Pre-Training Objective Triage Dispositions", fontweight="bold")
    ax1.grid(axis="y")

    # Panel B: Signal-to-Interference Ratio (SIR dB) vs min waveform correlation across all 437 files
    sir = np.array([float(r["min_sir_db"]) for r in rows])
    mr = np.array([float(r["min_r_wave"]) for r in rows])
    masked = mr < 0.25
    ax2.scatter(sir[~masked], mr[~masked], s=20, alpha=0.75, color="#10b981", label=f"Audible (>=0.25, N={int((~masked).sum())})")
    ax2.scatter(sir[masked], mr[masked], s=22, alpha=0.80, color="#ef4444", label=f"Energy-Masked (<0.25, N={int(masked.sum())})")
    ax2.axhline(0.25, color="#fbbf24", linestyle="--", linewidth=1.3, label="Audibility Threshold (r = 0.25)")
    ax2.axvline(-6.0, color="#94a3b8", linestyle=":", linewidth=1.1, label="SIR = -6 dB reference")
    ax2.set_xlabel("Minimum Active Fault Signal-to-Interference Ratio (SIR, dB)")
    ax2.set_ylabel("Minimum Waveform Correlation min_k |r(mix, source_k)|")
    ax2.set_title("Physical Cause of 172 Low-Correlation Clips: Loudness Masking", fontweight="bold")
    ax2.grid(True)
    ax2.legend(fontsize=8)

    fig.suptitle("AutoGuard AI V2 — Track 2 Combined-Fault Pre-Training Forensic Audit (437 Clips)", fontsize=13, fontweight="bold")
    out = FIG_P3_DIR / "p3_06_track2_combined_triage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    with open(REPORTS_P3_DIR / "phase3_master_summary.json", encoding="utf-8") as f:
        master = json.load(f)
    with open(REPORTS_P3_DIR / "track1_tabular_results.json", encoding="utf-8") as f:
        tab_res = json.load(f)
    with open(REPORTS_P3_DIR / "track1_neural_results.json", encoding="utf-8") as f:
        neu_res = json.load(f)

    outs = [
        plot_01_model_comparison(master),
        plot_02_window_vs_clip(tab_res, neu_res),
        plot_03_ablation(master["v1_vs_v2_ablation_summary"], master),
        plot_04_learning_curves(neu_res, master),
        plot_05_confusion_and_per_class(tab_res, neu_res, master),
        plot_06_track2_triage(master["track2_pretraining_audit_summary"]),
    ]
    for p in outs:
        print(f"Generated Phase 3 figure: {p}")


if __name__ == "__main__":
    main()
