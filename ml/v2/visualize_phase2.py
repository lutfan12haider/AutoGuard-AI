"""
ml/v2/visualize_phase2.py
-------------------------
AutoGuard AI V2 — Phase 2 Diagnostic Visualizations.

Generates 5 multi-panel verification plots under `reports/v2/figures/`:
  01_dataset_class_and_state_distributions.png
  02_kaggle_leakage_and_frequency_audit.png
  03_short_clip_and_combined_forensics.png
  04_waveforms_and_logmel_9classes.png
  05_extracted_feature_separability.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.v2.audio_preprocessing import (
    FMAX,
    N_FRAMES,
    N_MELS,
    TARGET_RMS,
    TARGET_SR,
    WINDOW_SEC,
    compute_log_mel_v2,
    load_and_standardize,
    normalize_window_rms,
    slice_windows,
)
from ml.v2.features_v2 import TABULAR_FEATURE_NAMES
from ml.v2.splitter import CLASSES_9

ROOT = Path(__file__).resolve().parent.parent.parent
PREP_DIR = ROOT / "data" / "v2_prepared"
REPORTS_DIR = ROOT / "reports" / "v2"
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def plot_01_distributions(split_summary: dict, mult_rows: list[dict]) -> Path:
    fig = plt.figure(figsize=(15, 8.5), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.25)

    # Panel A: Track 1 9-class split counts
    ax1 = fig.add_subplot(gs[:, 0])
    classes = list(CLASSES_9)
    splits = ["train", "validation", "test", "real_test"]
    colors = {"train": "#38bdf8", "validation": "#fbbf24", "test": "#34d399", "real_test": "#c084fc"}
    dist = split_summary["track1_class_distribution_by_split"]

    x = np.arange(len(classes))
    width = 0.20
    for idx, sp in enumerate(splits):
        vals = [dist[sp].get(c, 0) for c in classes]
        bars = ax1.bar(x + (idx - 1.5) * width, vals, width=width, label=f"{sp} (N={sum(vals)})", color=colors[sp])
        for b, v in zip(bars, vals):
            if v > 0:
                ax1.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", va="bottom", fontsize=7.2, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8.5)
    ax1.set_ylabel("Audio Clips")
    ax1.set_ylim(0, max(dist["train"].values()) + 65)
    ax1.set_title("Track 1: Single-Fault 9-Class Distribution by Split (Total N = 2,016)", fontweight="bold")
    ax1.grid(axis="y")
    ax1.legend(loc="upper right")

    # Panel B: Operating state breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    states = ["idle", "startup", "braking"]
    st_dist = split_summary["track1_operating_state_by_split"]
    bottoms = np.zeros(len(splits))
    st_colors = {"idle": "#60a5fa", "startup": "#f97316", "braking": "#f43f5e"}
    for st in states:
        vals = np.array([st_dist[sp].get(st, 0) for sp in splits])
        ax2.bar(splits, vals, bottom=bottoms, label=st, color=st_colors[st], width=0.55)
        for i, v in enumerate(vals):
            if v > 25:
                ax2.text(i, bottoms[i] + v / 2, str(v), ha="center", va="center", fontsize=8, color="#0f172a", fontweight="bold")
        bottoms += vals
    ax2.set_ylabel("Clips")
    ax2.set_title("Track 1 Operating States Across Splits", fontweight="bold")
    ax2.grid(axis="y")
    ax2.legend()

    # Panel C: Track 2 437 combined files purity & audibility
    ax3 = fig.add_subplot(gs[1, 1])
    comb_rows = [r for r in mult_rows if r["is_combined"] == "True"]
    cats = ["pure_train", "pure_validation", "pure_test", "cross_split_leakage"]
    cat_labels = ["Pure\nTrain", "Pure\nVal", "Pure\nTest", "Excluded:\nCross-Split\nLeakage"]
    aud_true, aud_false = [], []
    for c in cats:
        sub = [
            r for r in comb_rows
            if (r["split"] == "excluded_cross_split_leakage" and c == "cross_split_leakage")
            or (f"pure_{r['split']}" == c)
        ]
        t_cnt = sum(1 for r in sub if r["all_constituents_audible"] == "True")
        f_cnt = len(sub) - t_cnt
        aud_true.append(t_cnt)
        aud_false.append(f_cnt)

    xc = np.arange(len(cats))
    b1 = ax3.bar(xc, aud_true, label="All constituents audible (r >= 0.25)", color="#10b981", width=0.55)
    b2 = ax3.bar(xc, aud_false, bottom=aud_true, label="Masked/missing constituent (r < 0.25)", color="#ef4444", width=0.55)
    for i, (t, fl) in enumerate(zip(aud_true, aud_false)):
        tot = t + fl
        ax3.text(i, tot + 4, f"{tot}\n({t} aud)", ha="center", va="bottom", fontsize=8)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(cat_labels, fontsize=8.5)
    ax3.set_ylim(0, max(a + b for a, b in zip(aud_true, aud_false)) + 45)
    ax3.set_ylabel("Combined Clips (N = 437)")
    ax3.set_title("Track 2 Combined-Fault Lineage Purity & Audibility", fontweight="bold")
    ax3.grid(axis="y")
    ax3.legend(fontsize=8)

    fig.suptitle("AutoGuard AI V2 — Phase 2 Leakage-Free Split & Dataset Audit", fontsize=13, fontweight="bold", y=0.98)
    out = FIG_DIR / "01_dataset_class_and_state_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_02_leakage_and_frequency(forensics: dict, split_summary: dict) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=160)

    # Panel A: Leakage audit
    leak_f = forensics["kaggle_leakage_audit"]
    leak_s = split_summary["kaggle_session_leakage_audit"]
    folder_names = list(leak_f.keys())
    y_pos = np.arange(len(folder_names))

    rand_sim = [leak_f[k]["rand_test_to_train_1nn_sim"] for k in folder_names]
    block_sim = [leak_f[k]["block_test_to_train_1nn_sim"] for k in folder_names]
    # Look up matching key in split_summary
    sess_sim = []
    for k in folder_names:
        match_key = next(sk for sk in leak_s if sk.endswith("/" + k))
        sess_sim.append(leak_s[match_key]["test_to_train_1nn_cos"])

    h = 0.26
    ax1.barh(y_pos - h, rand_sim, height=h, label="Random 70/15/15 (up to 40% near-dups)", color="#ef4444")
    ax1.barh(y_pos, block_sim, height=h, label="Contiguous Block (bad_ignition=37.5% near-dup)", color="#f59e0b")
    ax1.barh(y_pos + h, sess_sim, height=h, label="V2 Session-Clustered (0.0% near-dup across all 9)", color="#22c55e")
    ax1.axvline(0.999, color="#f87171", linestyle="--", linewidth=1.1, label="Near-Duplicate Threshold (cos >= 0.999)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([k.replace("_", " ") for k in folder_names], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0.80, 1.015)
    ax1.set_xlabel("Mean Test-to-Train 1-NN Acoustic Cosine Similarity (Lower = Stronger Generalization Test)")
    ax1.set_title("Kaggle Split Leakage Audit Across All 9 Folders", fontweight="bold")
    ax1.grid(axis="x")
    ax1.legend(loc="lower left", fontsize=7.6)

    # Panel B: Spectral energy across frequency bands
    fb = forensics["frequency_bands"]
    labels = list(fb.keys())
    yb = np.arange(len(labels))
    b0 = np.array([fb[k]["pct_0_4kHz"] for k in labels])
    b1 = np.array([fb[k]["pct_4_8kHz"] for k in labels])
    b2 = np.array([fb[k]["pct_8_11kHz"] for k in labels])
    b3 = np.array([fb[k]["pct_above_11kHz"] for k in labels])

    ax2.barh(yb, b0, label="0 - 4 kHz", color="#3b82f6")
    ax2.barh(yb, b1, left=b0, label="4 - 8 kHz (kept @ 16 kHz SR)", color="#06b6d4")
    ax2.barh(yb, b2, left=b0 + b1, label="8 - 11 kHz (kept @ 22.05 kHz SR)", color="#f59e0b")
    ax2.barh(yb, b3, left=b0 + b1 + b2, label="> 11 kHz", color="#ec4899")
    for i, k in enumerate(labels):
        p16 = fb[k]["pct_preserved_at_16kHz"]
        ax2.text(98.5, i, f"{p16:.1f}%", va="center", ha="right", fontsize=7.5, color="#0f172a", fontweight="bold")
    ax2.set_yticks(yb)
    ax2.set_yticklabels(labels, fontsize=8.2)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Share of Spectral Energy (%)  |  Annotated: % Preserved <= 8 kHz (16 kHz SR)")
    ax2.set_title("Acoustic Energy by Frequency Band (16 kHz vs 22.05 kHz Nyquist)", fontweight="bold")
    ax2.grid(axis="x")
    ax2.legend(loc="lower left", fontsize=7.6)

    fig.suptitle("AutoGuard AI V2 — Session Leakage Elimination & Sample Rate Justification", fontsize=13, fontweight="bold")
    out = FIG_DIR / "02_kaggle_leakage_and_frequency_audit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_03_preprocessing_and_combined(forensics: dict, mult_rows: list[dict]) -> Path:
    fig = plt.figure(figsize=(15.5, 8.2), dpi=160)
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.25)

    # Panel A: Short-clip padding comparison
    ax1 = fig.add_subplot(gs[0, 0])
    sc = forensics["short_clip_strategies"]
    strats = ["wrap_taper_1.5s", "reflect_1.5s", "native_unpadded", "zero_1.5s", "zero_10.0s_V1"]
    s_labels = ["Wrap-Taper\n1.5s (CNN)", "Reflect\n1.5s", "Native Unpadded\n(Tabular)", "Zero-Pad\n1.5s", "Zero-Pad\n10.0s (V1)"]
    cos_d = [sc[s]["tabular_cosine_dist"] for s in strats]
    rms_e = [sc[s]["rms_mean_rel_error"] for s in strats]
    xs = np.arange(len(strats))
    ax1.bar(xs - 0.18, cos_d, width=0.35, label="Tabular Cosine Distortion", color="#38bdf8")
    ax1.bar(xs + 0.18, rms_e, width=0.35, label="RMS Mean Relative Error", color="#fb7185")
    for i, (c, r) in enumerate(zip(cos_d, rms_e)):
        ax1.text(i - 0.18, c + 0.018, f"{c:.4f}", ha="center", va="bottom", fontsize=7.2, rotation=25)
        ax1.text(i + 0.18, r + 0.018, f"{r:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=25)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(s_labels, fontsize=8)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Distortion / Error (Lower = Better)")
    ax1.set_title("Short Recording (<1.5s) Padding Strategy Comparison", fontweight="bold")
    ax1.grid(axis="y")
    ax1.legend(fontsize=8)

    # Panel B: Window duration policy
    ax2 = fig.add_subplot(gs[1, 0])
    wd = forensics["window_durations"]
    w_keys = ["1.5s", "3.0s", "5.0s", "10.0s", "1.5s+global_multiscale"]
    w_labels = ["1.5s\nWindow", "3.0s\nWindow", "5.0s\nWindow", "10.0s\nFull Clip", "1.5s + Global\nMultiscale"]
    acc = [wd[k]["val_accuracy"] * 100 for k in w_keys]
    f1 = [wd[k]["val_macro_f1"] * 100 for k in w_keys]
    fhr = [wd[k]["val_false_healthy_rate"] * 100 for k in w_keys]
    xw = np.arange(len(w_keys))
    ax2.plot(xw, acc, "o-", label="Val Accuracy (%)", color="#34d399", linewidth=2)
    ax2.plot(xw, f1, "s-", label="Val Macro-F1 (%)", color="#60a5fa", linewidth=2)
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax2.text(i, a + 0.45, f"{a:.2f}%", ha="center", fontsize=7.8, color="#34d399")
    ax2_r = ax2.twinx()
    ax2_r.bar(xw, fhr, width=0.35, alpha=0.25, color="#f43f5e", label="False-Healthy Rate (%)")
    ax2_r.set_ylim(0, 22)
    ax2_r.set_ylabel("False-Healthy Rate (%)", color="#fda4af")
    ax2.set_xticks(xw)
    ax2.set_xticklabels(w_labels, fontsize=8)
    ax2.set_ylim(75, 86)
    ax2.set_ylabel("Score (%)")
    ax2.set_title("Window Policy Ablation on KweriMoto (Train -> Val)", fontweight="bold")
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8)

    # Panel C: Zoomed peak-normalized combined waveform (proving peak-norm vs flat-top clipping)
    ax3 = fig.add_subplot(gs[0, 1])
    sample_comb = ROOT / next(
        r["rel_path"] for r in mult_rows
        if r["is_combined"] == "True" and r["all_constituents_audible"] == "True"
    )
    y_c, sr_c, meta_c = load_and_standardize(sample_comb, TARGET_SR)
    t_c = np.arange(len(y_c)) / sr_c
    pk_idx = int(np.argmax(np.abs(y_c)))
    lo = max(0, pk_idx - 120)
    hi = min(len(y_c), pk_idx + 120)
    ax3.plot(t_c[lo:hi] * 1000, y_c[lo:hi], color="#38bdf8", linewidth=1.2)
    ax3.axhline(1.0, color="#ef4444", linestyle="--", linewidth=0.9, label="Full-scale +/-1.0")
    ax3.axhline(-1.0, color="#ef4444", linestyle="--", linewidth=0.9)
    sat = np.abs(y_c[lo:hi]) >= 0.998
    if np.any(sat):
        ax3.scatter(t_c[lo:hi][sat] * 1000, y_c[lo:hi][sat], color="#fbbf24", zorder=5, s=38,
                    label=f"Peak-norm hit (max_run={meta_c['max_clip_run']} sample, NOT flat-top clip)")
    ax3.set_xlabel("Time around max-amplitude transient (ms)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Combined Recording Peak Forensics: Peak-Normalized, Not Clipped", fontweight="bold")
    ax3.grid(True)
    ax3.legend(fontsize=7.8, loc="lower right")

    # Panel D: Distribution of minimum constituent correlation across 437 combined files
    ax4 = fig.add_subplot(gs[1, 1])
    comb_rows = [r for r in mult_rows if r["is_combined"] == "True"]
    min_corrs = []
    for r in comb_rows:
        active_r = []
        if r["low_oil"] == "1":
            active_r.append(float(r["r_low_oil"]))
        if r["serpentine_belt"] == "1":
            active_r.append(float(r["r_serpentine_belt"]))
        if r["power_steering"] == "1":
            active_r.append(float(r["r_power_steering"]))
        min_corrs.append(min(active_r))
    ax4.hist(min_corrs, bins=24, range=(0.0, 1.0), color="#a78bfa", edgecolor="#1e293b", alpha=0.88)
    ax4.axvline(0.25, color="#ef4444", linestyle="--", linewidth=1.6,
                label=f"Audibility Cutoff (r=0.25): {sum(c < 0.25 for c in min_corrs)}/437 masked!")
    ax4.set_xlabel("Minimum Constituent Waveform Correlation min_k |r(mix, source_k)|")
    ax4.set_ylabel("Combined Clips")
    ax4.set_title("Audibility Audit Across All 437 Combined-Fault Recordings", fontweight="bold")
    ax4.grid(True)
    ax4.legend(fontsize=8)

    fig.suptitle("AutoGuard AI V2 — Short-Clip, Window Duration & Combined-Fault Forensics", fontsize=13, fontweight="bold")
    out = FIG_DIR / "03_short_clip_and_combined_forensics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_04_waveforms_and_logmel(t1_rows: list[dict], mult_rows: list[dict]) -> Path:
    samples: list[tuple[str, Path]] = []
    for cls in CLASSES_9:
        r = next(row for row in t1_rows if row["label"] == cls and row["split"] == "train" and row["short_lt_1_5s"] == "False")
        samples.append((f"{cls}\n({r['dataset_source']} / {r['operating_state']})", ROOT / r["rel_path"]))
    comb_r = next(
        row for row in mult_rows
        if row["is_combined"] == "True" and row["all_constituents_audible"] == "True"
        and row["low_oil"] == "1" and row["serpentine_belt"] == "1" and row["power_steering"] == "1"
    )
    samples.append(("combined 3-fault\n(low_oil + belt + ps)", ROOT / comb_r["rel_path"]))

    fig, axes = plt.subplots(5, 4, figsize=(16, 13), dpi=150)
    freqs_hz = np.linspace(20, FMAX, N_MELS)

    for idx, (title, p) in enumerate(samples):
        row = idx // 2
        col_pair = (idx % 2) * 2
        ax_w = axes[row, col_pair]
        ax_m = axes[row, col_pair + 1]

        y, sr, _ = load_and_standardize(p, target_sr=TARGET_SR)
        seg = slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC)[0][0]
        seg_norm, raw_rms, _ = normalize_window_rms(seg, target_rms=TARGET_RMS)
        mel = compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=FMAX, norm_mode="global_db")

        t = np.arange(len(seg_norm)) / sr
        ax_w.plot(t, seg_norm, color="#38bdf8", linewidth=0.55)
        ax_w.set_ylim(-0.6, 0.6)
        ax_w.set_title(f"{title} [raw_rms={raw_rms:.3f}]", fontsize=8.5, fontweight="bold")
        ax_w.set_ylabel("Amp" if col_pair == 0 else "", fontsize=7.5)
        ax_w.set_xlabel("Time (s)" if row == 4 else "", fontsize=7.5)
        ax_w.grid(True, alpha=0.4)

        im = ax_m.imshow(mel, origin="lower", aspect="auto", cmap="inferno", extent=[0, WINDOW_SEC, 0.02, FMAX / 1000.0], vmin=-3.0, vmax=2.0)
        ax_m.set_title("128x64 Log-Mel (global_db)", fontsize=8.5)
        ax_m.set_ylabel("Freq (kHz)" if col_pair == 0 else "", fontsize=7.5)
        ax_m.set_xlabel("Time (s)" if row == 4 else "", fontsize=7.5)

    fig.suptitle("AutoGuard AI V2 — Standardized 16 kHz / 1.50s Waveforms & Log-Mel Spectrograms (All 9 Classes + Combined)", fontsize=12.5, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = FIG_DIR / "04_waveforms_and_logmel_9classes.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_05_feature_separability() -> Path:
    tr = np.load(PREP_DIR / "single_fault_9class" / "train.npz")
    va = np.load(PREP_DIR / "single_fault_9class" / "validation.npz")

    X = np.vstack([tr["X_clip_global_tab"], va["X_clip_global_tab"]])
    y = np.concatenate([tr["y_clip"], va["y_clip"]])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_s)

    fig = plt.figure(figsize=(16, 7.5), dpi=160)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], hspace=0.35, wspace=0.28)

    palette = [
        "#22c55e", "#ef4444", "#f97316", "#eab308", "#06b6d4",
        "#3b82f6", "#a855f7", "#ec4899", "#94a3b8",
    ]
    ax_pca = fig.add_subplot(gs[:, 0])
    for ci, cname in enumerate(CLASSES_9):
        m = y == ci
        ax_pca.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.72, color=palette[ci], label=f"{cname} ({int(m.sum())})")
    ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax_pca.set_title("2D PCA of 336-Dim Clip-Global Features (Train + Val, N=1,681)", fontweight="bold")
    ax_pca.grid(True)
    ax_pca.legend(fontsize=8, loc="best", markerscale=1.4)

    biomarkers = [
        ("hpss_log10_hpr_mean", "HPSS Log10(Harmonic/Percussive)"),
        ("band_4000_fmax_ratio", "4 - 8 kHz Energy Fraction"),
        ("spec_flatness_mean", "Spectral Flatness (Tonal -> Noise)"),
        ("rms_mod_cv", "RMS Envelope CV (Unevenness)"),
    ]
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    short_names = ["hlth", "knk", "misf", "loil", "belt", "pstr", "ign", "batt", "brks"]

    for (feat_name, label), (r_i, c_i) in zip(biomarkers, positions):
        ax = fig.add_subplot(gs[r_i, c_i])
        f_idx = TABULAR_FEATURE_NAMES.index(feat_name)
        groups = [X[y == ci, f_idx] for ci in range(len(CLASSES_9))]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False)
        for patch, col in zip(bp["boxes"], palette):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("#ffffff")
            med.set_linewidth(1.5)
        ax.set_xticks(range(1, 10))
        ax.set_xticklabels(short_names, fontsize=7.8, rotation=20)
        ax.set_title(label, fontsize=9.2, fontweight="bold")
        ax.grid(axis="y")

    fig.suptitle("AutoGuard AI V2 — Extracted 336-Dim Feature Separability & Physical Biomarkers", fontsize=13, fontweight="bold")
    out = FIG_DIR / "05_extracted_feature_separability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    with open(REPORTS_DIR / "phase2_forensics.json", encoding="utf-8") as f:
        forensics = json.load(f)
    with open(PREP_DIR / "split_summary.json", encoding="utf-8") as f:
        split_summary = json.load(f)
    with open(PREP_DIR / "manifest_single_fault_9class.csv", encoding="utf-8") as f:
        t1_rows = list(csv.DictReader(f))
    with open(PREP_DIR / "manifest_multilabel_idle.csv", encoding="utf-8") as f:
        mult_rows = list(csv.DictReader(f))

    outs = [
        plot_01_distributions(split_summary, mult_rows),
        plot_02_leakage_and_frequency(forensics, split_summary),
        plot_03_preprocessing_and_combined(forensics, mult_rows),
        plot_04_waveforms_and_logmel(t1_rows, mult_rows),
        plot_05_feature_separability(),
    ]
    for p in outs:
        print(f"Generated figure: {p}")


if __name__ == "__main__":
    main()
