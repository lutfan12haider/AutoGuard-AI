"""
ml/v2/features_v2.py
--------------------
AutoGuard AI V2 — Phase 2 Feature Extraction Pipeline & Strict Sanity Suite.

Reads the leakage-free manifests produced by `ml.v2.splitter` and saves completely
isolated Phase 2 feature caches under `data/v2_prepared/`:
  1. `data/v2_prepared/single_fault_9class/{train,validation,test,real_test}.npz`
     - Window-level `X_win_tab` (N_w, 336) float32 + `X_win_mel` (N_w, 128, 64) float32
     - Clip-level full unpadded `X_clip_global_tab` (N_c, 336) float32 + mapping `win_clip_idx`
     - Metadata arrays (`y_win`, `y_clip`, `clip_rel_paths`, `clip_sources`, `clip_states`,
       `clip_sessions`, `win_is_short`, `clip_raw_rms`, `clip_raw_crest`)
  2. `data/v2_prepared/multilabel_idle/{train,validation,test,excluded_cross_split_leakage}.npz`
     - Strictly separated from Track 1 arrays; preserves `all_constituents_audible`,
       `is_combined`, and constituent correlation metadata (`r_low_oil`, etc.).
  3. `data/v2_prepared/braking_specialist_22k/{train,validation,test}.npz`
     - Compact 153-clip braking-only cache at 22,050 Hz (FMAX=11,025 Hz) so Phase 6 can
       rigorously ablate whether high-frequency braking squeal (>8 kHz) improves braking
       classification inside a hierarchical pipeline.
  4. `reports/v2/phase2_feature_report.json`
     - Exact feature names, array shapes, memory sizes, sanity-check assertions (0 NaN/Inf,
       determinism check, positive feature variance, manifest count reconciliation).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np

from ml.v2.audio_preprocessing import (
    BRAKING_SPECIALIST_SR,
    FMAX,
    HOP_SEC,
    N_FRAMES,
    N_MELS,
    TARGET_RMS,
    TARGET_SR,
    WINDOW_SEC,
    compute_log_mel_v2,
    load_and_standardize,
    normalize_window_rms,
    pad_short_wrap_taper,
    slice_windows,
)

ROOT = Path(__file__).resolve().parent.parent.parent
PREP_DIR = ROOT / "data" / "v2_prepared"
REPORTS_DIR = ROOT / "reports" / "v2"
TRACK1_DIR = PREP_DIR / "single_fault_9class"
TRACK2_DIR = PREP_DIR / "multilabel_idle"
BRAKE_DIR = PREP_DIR / "braking_specialist_22k"

for _d in (REPORTS_DIR, TRACK1_DIR, TRACK2_DIR, BRAKE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def build_tabular_feature_names() -> list[str]:
    """Return ordered list of exact 336 physics-informed tabular feature names."""
    names: list[str] = []
    for i in range(40):
        names.append(f"mfcc_{i:02d}_mean")
    for i in range(40):
        names.append(f"mfcc_{i:02d}_std")
    for i in range(40):
        names.append(f"dmfcc_{i:02d}_mean")
    for i in range(40):
        names.append(f"dmfcc_{i:02d}_std")
    for i in range(40):
        names.append(f"d2mfcc_{i:02d}_mean")
    for i in range(40):
        names.append(f"d2mfcc_{i:02d}_std")
    for i in range(20):
        names.append(f"mfcc_{i:02d}_iqr")

    for b in range(7):
        names.append(f"spec_contrast_b{b}_mean")
    for b in range(7):
        names.append(f"spec_contrast_b{b}_std")

    for c in range(12):
        names.append(f"chroma_{c:02d}_mean")
    for c in range(12):
        names.append(f"chroma_{c:02d}_std")

    for base in ("rms_norm", "zcr", "spec_centroid", "spec_bandwidth", "spec_rolloff", "spec_flatness"):
        names.extend([f"{base}_mean", f"{base}_std", f"{base}_iqr"])

    names.extend([
        "hpss_log10_hpr_mean",
        "hpss_log10_hpr_std",
        "hpss_perc_zcr_mean",
        "hpss_perc_zcr_std",
        "hpss_perc_flatness_mean",
        "hpss_harm_centroid_mean",
        "hpss_harm_centroid_std",
        "hpss_perc_crest_p95",
    ])

    names.extend([
        "band_50_300_ratio",
        "band_300_1200_ratio",
        "band_1200_4000_ratio",
        "band_4000_fmax_ratio",
        "log10_raw_rms",
        "log10_raw_crest",
        "rms_mod_cv",
        "rms_peak_to_median",
        "spec_flux_mean",
        "spec_flux_std",
        "onset_strength_mean",
        "onset_peak_to_mean",
    ])
    assert len(names) == 336, f"Expected 336 feature names, got {len(names)}"
    return names


TABULAR_FEATURE_NAMES: list[str] = build_tabular_feature_names()


def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def extract_tabular_features_from_array(
    y_norm: np.ndarray,
    raw_rms: float,
    raw_crest: float,
    sr: int = TARGET_SR,
    fmax: float = FMAX,
) -> np.ndarray:
    """
    Extract the 336-dim physics-informed acoustic vector using a single shared STFT
    (`n_fft=1024, hop_length=256`) and HPSS decomposition.

    Accepts unpadded short clips cleanly (padding ONLY below n_fft=1024 via wrap-taper
    and adapting delta widths dynamically for short frame counts).
    """
    n_fft = 1024
    hop_length = 256
    y_work = y_norm.astype(np.float32)
    if len(y_work) < n_fft:
        y_work = pad_short_wrap_taper(y_work, n_fft, sr=sr)

    D = librosa.stft(y_work, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D).astype(np.float32)
    P = (S ** 2).astype(np.float32)
    T = int(S.shape[1])

    mel = librosa.feature.melspectrogram(S=P, sr=sr, n_mels=128, fmin=20.0, fmax=fmax)
    log_mel_db = librosa.power_to_db(mel + 1e-10, ref=1.0, top_db=None)
    mfcc = librosa.feature.mfcc(S=log_mel_db, n_mfcc=40)

    w = min(9, T if T % 2 == 1 else T - 1)
    if w >= 3:
        d1 = librosa.feature.delta(mfcc, order=1, width=w)
        d2 = librosa.feature.delta(mfcc, order=2, width=w)
    else:
        d1 = np.zeros_like(mfcc)
        d2 = np.zeros_like(mfcc)

    mfcc_iqr = np.percentile(mfcc[:20], 75, axis=1) - np.percentile(mfcc[:20], 25, axis=1)

    contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=50.0, n_bands=6)
    chroma = librosa.feature.chroma_stft(S=P, sr=sr, n_chroma=12)

    rms_f = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    zcr_f = librosa.feature.zero_crossing_rate(y_work, frame_length=n_fft, hop_length=hop_length)[0]
    cent_f = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bw_f = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    roll_f = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    flat_f = librosa.feature.spectral_flatness(S=S)[0]

    desc_stats: list[float] = []
    for arr in (rms_f, zcr_f, cent_f, bw_f, roll_f, flat_f):
        desc_stats.extend([float(np.mean(arr)), float(np.std(arr)), _iqr(arr)])

    # Harmonic-Percussive Source Separation descriptors
    if T >= 4:
        H_stft, P_stft = librosa.decompose.hpss(D, margin=2.0)
        H_mag = np.abs(H_stft).astype(np.float32)
        P_mag = np.abs(P_stft).astype(np.float32)
        y_perc = librosa.istft(P_stft, hop_length=hop_length, length=len(y_work))

        h_pow = np.sum(H_mag ** 2, axis=0) + 1e-10
        p_pow = np.sum(P_mag ** 2, axis=0) + 1e-10
        log_hpr = np.log10(h_pow / p_pow)

        pzcr = librosa.feature.zero_crossing_rate(y_perc, frame_length=n_fft, hop_length=hop_length)[0]
        pflat = librosa.feature.spectral_flatness(S=P_mag + 1e-8)[0]
        hcent = librosa.feature.spectral_centroid(S=H_mag + 1e-8, sr=sr)[0]
        perc_rms = float(np.sqrt(np.mean(y_perc ** 2))) + 1e-8
        perc_p95 = float(np.percentile(np.abs(y_perc), 95))
        hpss_feats = [
            float(np.mean(log_hpr)),
            float(np.std(log_hpr)),
            float(np.mean(pzcr)),
            float(np.std(pzcr)),
            float(np.mean(pflat)),
            float(np.mean(hcent)),
            float(np.std(hcent)),
            perc_p95 / perc_rms,
        ]
    else:
        hpss_feats = [0.0, 0.0, float(np.mean(zcr_f)), 0.0, float(np.mean(flat_f)), float(np.mean(cent_f)), 0.0, float(raw_crest)]

    # Diagnostic sub-band energy ratios
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_power = np.sum(P, axis=1)
    fmax_mask = freqs <= fmax
    tot_pow = float(np.sum(band_power[fmax_mask])) + 1e-10

    def _band_ratio(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi) & fmax_mask
        return float(np.sum(band_power[m])) / tot_pow

    b1 = _band_ratio(50.0, 300.0)
    b2 = _band_ratio(300.0, 1200.0)
    b3 = _band_ratio(1200.0, 4000.0)
    b4 = _band_ratio(4000.0, fmax + 1.0)

    # Physical amplitude & modulation descriptors
    log_raw_rms = float(np.log10(max(raw_rms, 1e-7)))
    log_raw_crest = float(np.log10(max(raw_crest, 1.0)))
    rms_mean = float(np.mean(rms_f)) + 1e-8
    rms_cv = float(np.std(rms_f)) / rms_mean
    rms_p2m = float(np.percentile(rms_f, 95)) / (float(np.median(rms_f)) + 1e-8)

    if T >= 2:
        S_col = S / (np.sum(S, axis=0, keepdims=True) + 1e-8)
        flux = np.sqrt(np.sum(np.diff(S_col, axis=1) ** 2, axis=0))
        flux_mean = float(np.mean(flux))
        flux_std = float(np.std(flux))
    else:
        flux_mean, flux_std = 0.0, 0.0

    if T >= 3:
        onset_env = librosa.onset.onset_strength(S=log_mel_db, sr=sr, hop_length=hop_length)
        onset_mean = float(np.mean(onset_env))
        onset_p2m = float(np.max(onset_env)) / (onset_mean + 1e-8)
    else:
        onset_mean, onset_p2m = 0.0, 1.0

    vec = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        d1.mean(axis=1),
        d1.std(axis=1),
        d2.mean(axis=1),
        d2.std(axis=1),
        mfcc_iqr,
        contrast.mean(axis=1),
        contrast.std(axis=1),
        chroma.mean(axis=1),
        chroma.std(axis=1),
        np.array(desc_stats, dtype=np.float32),
        np.array(hpss_feats, dtype=np.float32),
        np.array(
            [
                b1,
                b2,
                b3,
                b4,
                log_raw_rms,
                log_raw_crest,
                rms_cv,
                rms_p2m,
                flux_mean,
                flux_std,
                onset_mean,
                onset_p2m,
            ],
            dtype=np.float32,
        ),
    ]).astype(np.float32)

    assert vec.shape == (336,), f"Unexpected feature dimension {vec.shape}"
    return vec


def _extract_track1_split(split_name: str, rows: list[dict]) -> dict:
    """
    Extract window-level + clip-level global features for one Track 1 split.
    Uses non-overlapping canonical `1.50s` windows (`hop_sec=1.50`) so eval clips
    never suffer window-overlap inflation and Phase 4 handles train-only augmentations.
    """
    t0 = time.perf_counter()
    win_tab_list, win_mel_list, win_y_list = [], [], []
    win_clip_idx_list, win_short_list = [], []

    clip_tab_list, clip_y_list = [], []
    clip_paths, clip_sources, clip_states, clip_sessions = [], [], [], []
    clip_raw_rms, clip_raw_crest = [], []

    for clip_i, row in enumerate(rows):
        abs_path = ROOT / row["rel_path"]
        label_int = int(row["label_int"])
        y, sr, meta = load_and_standardize(abs_path, target_sr=TARGET_SR)

        # Full-clip global tabular features (unpadded native clip length)
        y_clip_norm, raw_rms_c, raw_crest_c = normalize_window_rms(y, target_rms=TARGET_RMS)
        clip_tab = extract_tabular_features_from_array(y_clip_norm, raw_rms_c, raw_crest_c, sr=sr, fmax=FMAX)
        clip_tab_list.append(clip_tab)
        clip_y_list.append(label_int)
        clip_paths.append(row["rel_path"])
        clip_sources.append(row["dataset_source"])
        clip_states.append(row["operating_state"])
        clip_sessions.append(row["session_group"])
        clip_raw_rms.append(raw_rms_c)
        clip_raw_crest.append(raw_crest_c)

        # Window slicing (1.50s non-overlapping baseline grid)
        wins = slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC, max_windows_per_clip=7)
        for seg, _s, _e, is_short in wins:
            seg_norm, raw_rms_w, raw_crest_w = normalize_window_rms(seg, target_rms=TARGET_RMS)
            w_tab = extract_tabular_features_from_array(seg_norm, raw_rms_w, raw_crest_w, sr=sr, fmax=FMAX)
            w_mel = compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=FMAX, norm_mode="global_db")
            win_tab_list.append(w_tab)
            win_mel_list.append(w_mel)
            win_y_list.append(label_int)
            win_clip_idx_list.append(clip_i)
            win_short_list.append(bool(is_short))

        if (clip_i + 1) % 250 == 0 or (clip_i + 1) == len(rows):
            print(f"    [Track 1 | {split_name:10s}] {clip_i + 1:4d}/{len(rows)} clips -> {len(win_tab_list):5d} windows", flush=True)

    X_win_tab = np.asarray(win_tab_list, dtype=np.float32)
    X_win_mel = np.asarray(win_mel_list, dtype=np.float32)
    X_clip_global_tab = np.asarray(clip_tab_list, dtype=np.float32)

    out_path = TRACK1_DIR / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        X_win_tab=X_win_tab,
        X_win_mel=X_win_mel,
        y_win=np.asarray(win_y_list, dtype=np.int32),
        win_clip_idx=np.asarray(win_clip_idx_list, dtype=np.int32),
        win_is_short=np.asarray(win_short_list, dtype=bool),
        X_clip_global_tab=X_clip_global_tab,
        y_clip=np.asarray(clip_y_list, dtype=np.int32),
        clip_rel_paths=np.asarray(clip_paths, dtype=str),
        clip_sources=np.asarray(clip_sources, dtype=str),
        clip_states=np.asarray(clip_states, dtype=str),
        clip_sessions=np.asarray(clip_sessions, dtype=str),
        clip_raw_rms=np.asarray(clip_raw_rms, dtype=np.float32),
        clip_raw_crest=np.asarray(clip_raw_crest, dtype=np.float32),
    )

    dt = time.perf_counter() - t0
    ram_mb = (X_win_tab.nbytes + X_win_mel.nbytes + X_clip_global_tab.nbytes) / (1024 ** 2)
    return {
        "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "n_clips": len(rows),
        "n_windows": int(X_win_tab.shape[0]),
        "short_windows": int(sum(win_short_list)),
        "X_win_tab_shape": list(X_win_tab.shape),
        "X_win_mel_shape": list(X_win_mel.shape),
        "X_clip_global_tab_shape": list(X_clip_global_tab.shape),
        "arrays_ram_mb": round(ram_mb, 2),
        "file_size_mb": round(out_path.stat().st_size / (1024 ** 2), 2),
        "nan_count": int(np.isnan(X_win_tab).sum() + np.isnan(X_win_mel).sum() + np.isnan(X_clip_global_tab).sum()),
        "inf_count": int(np.isinf(X_win_tab).sum() + np.isinf(X_win_mel).sum() + np.isinf(X_clip_global_tab).sum()),
        "elapsed_sec": round(dt, 2),
    }


def _extract_track2_split(split_name: str, rows: list[dict]) -> dict:
    """Extract Track 2 multi-label idle caches (`train`, `validation`, `test`, `excluded_cross_split_leakage`)."""
    t0 = time.perf_counter()
    tab_list, mel_list, y_mh_list = [], [], []
    is_comb_list, audible_list = [], []
    paths, folders, sessions = [], [], []
    corr_arr = []

    for i, row in enumerate(rows):
        abs_path = ROOT / row["rel_path"]
        y, sr, _meta = load_and_standardize(abs_path, target_sr=TARGET_SR)
        seg = slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC)[0][0]
        seg_norm, raw_rms, raw_crest = normalize_window_rms(seg, target_rms=TARGET_RMS)
        tab = extract_tabular_features_from_array(seg_norm, raw_rms, raw_crest, sr=sr, fmax=FMAX)
        mel = compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=FMAX, norm_mode="global_db")

        tab_list.append(tab)
        mel_list.append(mel)
        y_mh_list.append([int(row["low_oil"]), int(row["serpentine_belt"]), int(row["power_steering"])])
        is_comb_list.append(row["is_combined"] == "True")
        audible_list.append(row["all_constituents_audible"] == "True")
        paths.append(row["rel_path"])
        folders.append(row["source_folder"])
        sessions.append(row["session_group"])
        corr_arr.append([float(row["r_low_oil"]), float(row["r_serpentine_belt"]), float(row["r_power_steering"])])

    X_tab = np.asarray(tab_list, dtype=np.float32)
    X_mel = np.asarray(mel_list, dtype=np.float32)
    out_path = TRACK2_DIR / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        X_tab=X_tab,
        X_mel=X_mel,
        y_multihot=np.asarray(y_mh_list, dtype=np.int32),
        is_combined=np.asarray(is_comb_list, dtype=bool),
        all_constituents_audible=np.asarray(audible_list, dtype=bool),
        rel_paths=np.asarray(paths, dtype=str),
        source_folders=np.asarray(folders, dtype=str),
        sessions=np.asarray(sessions, dtype=str),
        constituent_corrs=np.asarray(corr_arr, dtype=np.float32),
    )
    dt = time.perf_counter() - t0
    return {
        "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "n_clips": len(rows),
        "n_combined": int(sum(is_comb_list)),
        "n_single_idle": int(len(rows) - sum(is_comb_list)),
        "n_audible": int(sum(audible_list)),
        "X_tab_shape": list(X_tab.shape),
        "X_mel_shape": list(X_mel.shape),
        "file_size_mb": round(out_path.stat().st_size / (1024 ** 2), 2),
        "nan_count": int(np.isnan(X_tab).sum() + np.isnan(X_mel).sum()),
        "inf_count": int(np.isinf(X_tab).sum() + np.isinf(X_mel).sum()),
        "elapsed_sec": round(dt, 2),
    }


def _extract_braking_specialist(rows_by_split: dict[str, list[dict]]) -> dict:
    """Extract compact 22.05 kHz braking-state cache (normal_brakes vs worn_out_brakes)."""
    summary = {}
    fmax_brake = float(BRAKING_SPECIALIST_SR // 2)  # 11,025 Hz
    for sp in ("train", "validation", "test"):
        b_rows = [r for r in rows_by_split[sp] if r["operating_state"] == "braking"]
        tab_list, mel_list, y_list, paths = [], [], [], []
        for r in b_rows:
            y, sr, _ = load_and_standardize(ROOT / r["rel_path"], target_sr=BRAKING_SPECIALIST_SR)
            seg = slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC)[0][0]
            seg_norm, raw_rms, raw_crest = normalize_window_rms(seg, target_rms=TARGET_RMS)
            tab_list.append(extract_tabular_features_from_array(seg_norm, raw_rms, raw_crest, sr=sr, fmax=fmax_brake))
            mel_list.append(compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=fmax_brake, norm_mode="global_db"))
            y_list.append(1 if r["label"] == "worn_out_brakes" else 0)
            paths.append(r["rel_path"])

        X_tab = np.asarray(tab_list, dtype=np.float32)
        X_mel = np.asarray(mel_list, dtype=np.float32)
        out_path = BRAKE_DIR / f"{sp}.npz"
        np.savez_compressed(
            out_path,
            X_tab=X_tab,
            X_mel=X_mel,
            y=np.asarray(y_list, dtype=np.int32),
            rel_paths=np.asarray(paths, dtype=str),
        )
        summary[sp] = {
            "n_clips": len(b_rows),
            "worn_out_brakes": int(sum(y_list)),
            "normal_brakes": int(len(y_list) - sum(y_list)),
            "X_tab_shape": list(X_tab.shape),
            "X_mel_shape": list(X_mel.shape),
            "nan_count": int(np.isnan(X_tab).sum() + np.isnan(X_mel).sum()),
        }
    return summary


def run_sanity_checks(track1_stats: dict) -> dict:
    """Verify determinism, positive variance, value bounds, and manifest alignment."""
    print("\n=== RUNNING PHASE 2 PREPROCESSING SANITY CHECKS ===", flush=True)
    checks: dict = {}

    # 1. Determinism test on 3 representative files (KweriMoto, Kaggle normal, Kaggle worn_out_brakes)
    manifest1 = PREP_DIR / "manifest_single_fault_9class.csv"
    with open(manifest1, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    probes = [
        next(r for r in rows if r["dataset_source"].startswith("kwerimoto") and r["short_lt_1_5s"] == "True"),
        next(r for r in rows if r["dataset_source"].startswith("kwerimoto") and r["label"] == "engine_knock"),
        next(r for r in rows if r["label"] == "worn_out_brakes"),
    ]
    max_det_diff = 0.0
    for pr in probes:
        p = ROOT / pr["rel_path"]
        y1, sr1, _ = load_and_standardize(p, TARGET_SR)
        y2, sr2, _ = load_and_standardize(p, TARGET_SR)
        seg1, _, _ = normalize_window_rms(slice_windows(y1, sr1, WINDOW_SEC, WINDOW_SEC)[0][0])
        seg2, _, _ = normalize_window_rms(slice_windows(y2, sr2, WINDOW_SEC, WINDOW_SEC)[0][0])
        t1 = extract_tabular_features_from_array(seg1, 0.08, 4.0, sr1)
        t2 = extract_tabular_features_from_array(seg2, 0.08, 4.0, sr2)
        m1 = compute_log_mel_v2(y1, sr1)
        m2 = compute_log_mel_v2(y2, sr2)
        max_det_diff = max(max_det_diff, float(np.max(np.abs(t1 - t2))), float(np.max(np.abs(m1 - m2))))

    checks["determinism_max_abs_diff"] = max_det_diff
    assert max_det_diff == 0.0, f"Non-deterministic extraction! max diff={max_det_diff}"

    # 2. Check zero-variance / NaN / Inf / Mel dynamic range across saved Track 1 splits
    tr = np.load(TRACK1_DIR / "train.npz")
    stds = np.std(tr["X_win_tab"], axis=0)
    zero_var_cols = [TABULAR_FEATURE_NAMES[i] for i, s in enumerate(stds) if s < 1e-8]
    mel_min, mel_max, mel_mean, mel_std = (
        float(tr["X_win_mel"].min()),
        float(tr["X_win_mel"].max()),
        float(tr["X_win_mel"].mean()),
        float(tr["X_win_mel"].std()),
    )
    mel_sat_high = float(np.mean(tr["X_win_mel"] >= 2.999) * 100.0)
    checks["zero_variance_features"] = zero_var_cols
    checks["train_mel_range"] = [round(mel_min, 4), round(mel_max, 4)]
    checks["train_mel_mean_std"] = [round(mel_mean, 4), round(mel_std, 4)]
    checks["train_mel_ceiling_saturated_pct"] = round(mel_sat_high, 4)
    checks["total_track1_clips_extracted"] = sum(v["n_clips"] for v in track1_stats.values())
    checks["total_track1_windows_extracted"] = sum(v["n_windows"] for v in track1_stats.values())

    assert len(zero_var_cols) == 0, f"Zero-variance tabular columns found: {zero_var_cols}"
    assert checks["total_track1_clips_extracted"] == 2016, "Clip count mismatch!"
    assert mel_sat_high < 0.1, f"Mel ceiling saturation detected: {mel_sat_high:.2f}%"

    print(f"  [OK] Determinism max_abs_diff = {max_det_diff}")
    print(f"  [OK] Zero-variance features = {len(zero_var_cols)} / {len(TABULAR_FEATURE_NAMES)}")
    print(f"  [OK] Train Log-Mel range = [{mel_min:.3f}, {mel_max:.3f}], mean={mel_mean:.3f}, std={mel_std:.3f}, ceiling_sat={mel_sat_high:.4f}%")
    print(f"  [OK] Reconciled Track 1 clips = {checks['total_track1_clips_extracted']} -> {checks['total_track1_windows_extracted']} windows")
    return checks


def _inspect_cached_stats() -> tuple[dict, dict, dict]:
    """Recompute exact metadata summaries and NaN/Inf checks directly from saved `.npz` caches."""
    track1_stats = {}
    for sp in ("train", "validation", "test", "real_test"):
        out_path = TRACK1_DIR / f"{sp}.npz"
        d = np.load(out_path)
        X_wt, X_wm, X_ct = d["X_win_tab"], d["X_win_mel"], d["X_clip_global_tab"]
        ram_mb = (X_wt.nbytes + X_wm.nbytes + X_ct.nbytes) / (1024 ** 2)
        nans = int(np.isnan(X_wt).sum() + np.isnan(X_wm).sum() + np.isnan(X_ct).sum())
        infs = int(np.isinf(X_wt).sum() + np.isinf(X_wm).sum() + np.isinf(X_ct).sum())
        assert nans == 0 and infs == 0, f"Track 1 {sp} has NaN={nans} Inf={infs}"
        track1_stats[sp] = {
            "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "n_clips": int(X_ct.shape[0]),
            "n_windows": int(X_wt.shape[0]),
            "short_windows": int(np.sum(d["win_is_short"])),
            "X_win_tab_shape": list(X_wt.shape),
            "X_win_mel_shape": list(X_wm.shape),
            "X_clip_global_tab_shape": list(X_ct.shape),
            "arrays_ram_mb": round(ram_mb, 2),
            "file_size_mb": round(out_path.stat().st_size / (1024 ** 2), 2),
            "nan_count": nans,
            "inf_count": infs,
        }

    track2_stats = {}
    for sp in ("train", "validation", "test", "excluded_cross_split_leakage"):
        out_path = TRACK2_DIR / f"{sp}.npz"
        d = np.load(out_path)
        X_t, X_m = d["X_tab"], d["X_mel"]
        nans = int(np.isnan(X_t).sum() + np.isnan(X_m).sum())
        infs = int(np.isinf(X_t).sum() + np.isinf(X_m).sum())
        assert nans == 0 and infs == 0, f"Track 2 {sp} has NaN={nans} Inf={infs}"
        n_comb = int(np.sum(d["is_combined"]))
        track2_stats[sp] = {
            "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "n_clips": int(X_t.shape[0]),
            "n_combined": n_comb,
            "n_single_idle": int(X_t.shape[0] - n_comb),
            "n_audible": int(np.sum(d["all_constituents_audible"])),
            "X_tab_shape": list(X_t.shape),
            "X_mel_shape": list(X_m.shape),
            "file_size_mb": round(out_path.stat().st_size / (1024 ** 2), 2),
            "nan_count": nans,
            "inf_count": infs,
        }

    brake_stats = {}
    for sp in ("train", "validation", "test"):
        out_path = BRAKE_DIR / f"{sp}.npz"
        d = np.load(out_path)
        X_t, X_m, y = d["X_tab"], d["X_mel"], d["y"]
        nans = int(np.isnan(X_t).sum() + np.isnan(X_m).sum())
        infs = int(np.isinf(X_t).sum() + np.isinf(X_m).sum())
        assert nans == 0 and infs == 0, f"Brake 22k {sp} has NaN={nans} Inf={infs}"
        brake_stats[sp] = {
            "n_clips": int(X_t.shape[0]),
            "worn_out_brakes": int(np.sum(y)),
            "normal_brakes": int(len(y) - np.sum(y)),
            "X_tab_shape": list(X_t.shape),
            "X_mel_shape": list(X_m.shape),
            "nan_count": nans,
            "inf_count": infs,
        }
    return track1_stats, track2_stats, brake_stats


def _refresh_mel_caches() -> None:
    """Recompute ONLY Log-Mel arrays across saved Track 1 / Track 2 / Brake 22k caches using updated global_db scaling."""
    for sp in ("train", "validation", "test", "real_test"):
        p_npz = TRACK1_DIR / f"{sp}.npz"
        d = dict(np.load(p_npz, allow_pickle=False))
        mel_list = []
        for rel in d["clip_rel_paths"]:
            y, sr, _ = load_and_standardize(ROOT / str(rel), target_sr=TARGET_SR)
            for seg, _s, _e, _short in slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC, max_windows_per_clip=7):
                mel_list.append(compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=FMAX, norm_mode="global_db"))
        new_mel = np.asarray(mel_list, dtype=np.float32)
        assert new_mel.shape == d["X_win_mel"].shape, f"Shape mismatch {new_mel.shape} != {d['X_win_mel'].shape}"
        d["X_win_mel"] = new_mel
        np.savez_compressed(p_npz, **d)
        print(f"    Refreshed Track 1 {sp:10s} X_win_mel {new_mel.shape} -> mean={new_mel.mean():.3f}, max={new_mel.max():.3f}", flush=True)

    for sp in ("train", "validation", "test", "excluded_cross_split_leakage"):
        p_npz = TRACK2_DIR / f"{sp}.npz"
        d = dict(np.load(p_npz, allow_pickle=False))
        mel_list = []
        for rel in d["rel_paths"]:
            y, sr, _ = load_and_standardize(ROOT / str(rel), target_sr=TARGET_SR)
            seg = slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC)[0][0]
            mel_list.append(compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=FMAX, norm_mode="global_db"))
        new_mel = np.asarray(mel_list, dtype=np.float32)
        assert new_mel.shape == d["X_mel"].shape
        d["X_mel"] = new_mel
        np.savez_compressed(p_npz, **d)
        print(f"    Refreshed Track 2 {sp:28s} X_mel {new_mel.shape}", flush=True)

    fmax_brake = float(BRAKING_SPECIALIST_SR // 2)
    for sp in ("train", "validation", "test"):
        p_npz = BRAKE_DIR / f"{sp}.npz"
        d = dict(np.load(p_npz, allow_pickle=False))
        mel_list = []
        for rel in d["rel_paths"]:
            y, sr, _ = load_and_standardize(ROOT / str(rel), target_sr=BRAKING_SPECIALIST_SR)
            seg = slice_windows(y, sr=sr, win_sec=WINDOW_SEC, hop_sec=WINDOW_SEC)[0][0]
            mel_list.append(compute_log_mel_v2(seg, sr=sr, n_mels=N_MELS, n_frames=N_FRAMES, fmax=fmax_brake, norm_mode="global_db"))
        new_mel = np.asarray(mel_list, dtype=np.float32)
        assert new_mel.shape == d["X_mel"].shape
        d["X_mel"] = new_mel
        np.savez_compressed(p_npz, **d)
        print(f"    Refreshed Brake22k {sp:10s} X_mel {new_mel.shape}", flush=True)


def main() -> None:
    t_all = time.perf_counter()
    args = sys.argv[1:]
    refresh_mel = "--refresh-mel" in args
    from_cache = "--from-cache" in args or refresh_mel

    if refresh_mel:
        print("=== REFRESHING LOG-MEL ARRAYS ACROSS SAVED CACHES ===", flush=True)
        _refresh_mel_caches()

    if from_cache:
        print("=== VERIFYING EXISTING PHASE 2 FEATURE CACHES (--from-cache) ===", flush=True)
        track1_stats, track2_stats, brake_stats = _inspect_cached_stats()
    else:
        print("=== EXTRACTING TRACK 1 (9-CLASS SINGLE-FAULT @ 16 kHz) ===", flush=True)
        with open(PREP_DIR / "manifest_single_fault_9class.csv", encoding="utf-8") as f:
            t1_rows = list(csv.DictReader(f))

        t1_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": [], "real_test": []}
        for r in t1_rows:
            t1_by_split[r["split"]].append(r)

        track1_stats = {}
        for sp in ("train", "validation", "test", "real_test"):
            track1_stats[sp] = _extract_track1_split(sp, t1_by_split[sp])

        print("\n=== EXTRACTING TRACK 2 (MULTI-LABEL IDLE @ 16 kHz) ===", flush=True)
        with open(PREP_DIR / "manifest_multilabel_idle.csv", encoding="utf-8") as f:
            t2_rows = list(csv.DictReader(f))

        t2_by_split: dict[str, list[dict]] = {
            "train": [],
            "validation": [],
            "test": [],
            "excluded_cross_split_leakage": [],
        }
        for r in t2_rows:
            t2_by_split[r["split"]].append(r)

        track2_stats = {}
        for sp in ("train", "validation", "test", "excluded_cross_split_leakage"):
            track2_stats[sp] = _extract_track2_split(sp, t2_by_split[sp])
            s = track2_stats[sp]
            print(f"    [Track 2 | {sp:28s}] N={s['n_clips']:3d} (single={s['n_single_idle']}, comb={s['n_combined']}, audible={s['n_audible']}) -> {s['file_size_mb']} MB", flush=True)

        print("\n=== EXTRACTING BRAKING SPECIALIST CACHE (22.05 kHz, FMAX=11.025 kHz) ===", flush=True)
        brake_stats = _extract_braking_specialist(t1_by_split)
        for sp, s in brake_stats.items():
            print(f"    [Brake 22k | {sp:10s}] N={s['n_clips']:3d} (worn={s['worn_out_brakes']}, normal={s['normal_brakes']})", flush=True)

    checks = run_sanity_checks(track1_stats)

    with open(PREP_DIR / "tabular_feature_names.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(TABULAR_FEATURE_NAMES), "names": TABULAR_FEATURE_NAMES}, f, indent=2)

    report = {
        "feature_dimensions": {
            "tabular_per_window": len(TABULAR_FEATURE_NAMES),
            "tabular_clip_global": len(TABULAR_FEATURE_NAMES),
            "log_mel_spectrogram": [N_MELS, N_FRAMES],
            "target_sr_main": TARGET_SR,
            "target_sr_braking_specialist": BRAKING_SPECIALIST_SR,
            "window_sec": WINDOW_SEC,
            "hop_sec": WINDOW_SEC,
            "mel_norm_mode": "global_db",
        },
        "track1_single_fault": track1_stats,
        "track2_multilabel_idle": track2_stats,
        "braking_specialist_22k": brake_stats,
        "sanity_checks": checks,
        "total_elapsed_sec": round(time.perf_counter() - t_all, 2),
    }
    with open(REPORTS_DIR / "phase2_feature_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved complete feature report to {REPORTS_DIR / 'phase2_feature_report.json'} ({report['total_elapsed_sec']}s)")


if __name__ == "__main__":
    main()
