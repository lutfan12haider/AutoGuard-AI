"""
ml/v2/v1_vs_v2_ablation.py
--------------------------
AutoGuard AI V2 — Controlled V1 vs V2 Ablation Suite (Zero Confounds).

Isolates the individual impact of:
  1. Audio standardization (`16 kHz` resampling, DC removal, RMS=0.08, `wrap_taper`)
  2. Improved `1.50s` windowing + clip-level aggregation
  3. V2 physics feature engineering (HPSS, sub-bands, MFCC IQRs, flux, onsets)
  4. Raw-energy & crest-factor preservation (`with` vs `without` pre-norm RMS/crest)
  5. Architecture & training improvements (`AutoGuardCNN V1` vs `CNN V2` vs `HistGB` vs `HybridNet`)

Evaluated cleanly on TWO strictly controlled suites:
  - Suite A: Exact KweriMoto 3-class split (`721 train / 154 val / 154 test / 38 real_test`)
             including read-only inference of the actual saved V1 `best_model.pkl` & `autoguard_model.pt`.
  - Suite B: Full 9-class Track 1 split (`1,385 train / 296 val / 297 test / 38 real_test`)
             comparing V1-style pipelines vs incremental V2 components on identical files.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

import joblib
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from ml.v2.audio_preprocessing import (
    FMAX,
    TARGET_RMS,
    TARGET_SR,
    WINDOW_SEC,
    compute_log_mel_v2,
    load_and_standardize,
    normalize_window_rms,
    slice_windows,
)
from ml.v2.features_v2 import TABULAR_FEATURE_NAMES, extract_tabular_features_from_array
from ml.v2.metrics_utils import (
    MODELS_V2_DIR,
    REPORTS_P3_DIR,
    SEED,
    aggregate_windows_to_clips,
    compute_full_metrics,
    set_seed,
)
from ml.v2.splitter import CLASSES_9
from ml.v2.track1_neural import AutoGuardCNNv2, AutoGuardHybridNet, _compute_sqrt_class_weights

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.cnn_model import AutoGuardCNN  # noqa: E402
from ml.features import extract_features as extract_v1_cnn_features  # noqa: E402

TRACK1_DIR = ROOT / "data" / "v2_prepared" / "single_fault_9class"
RAW_E_IDXS = [
    TABULAR_FEATURE_NAMES.index("log10_raw_rms"),
    TABULAR_FEATURE_NAMES.index("log10_raw_crest"),
]
NO_RAW_MASK = np.array([i not in RAW_E_IDXS for i in range(len(TABULAR_FEATURE_NAMES))], dtype=bool)
CLASSES_3 = ("healthy", "engine_knock", "misfire")


def _pool_stats(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.array([float(x.mean()), float(x.std())], dtype=np.float32)
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).astype(np.float32)


def extract_v1_288_from_array(y: np.ndarray, sr: int) -> np.ndarray:
    """Exact replica of V1 `backend/model.py` 288-dim MFCC+Spectral vector."""
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)), mode="reflect")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    T = mfcc.shape[1]
    w = min(9, T if T % 2 == 1 else T - 1)
    if w >= 3:
        d1 = librosa.feature.delta(mfcc, order=1, width=w)
        d2 = librosa.feature.delta(mfcc, order=2, width=w)
    else:
        d1 = np.zeros_like(mfcc)
        d2 = np.zeros_like(mfcc)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.concatenate([
        _pool_stats(mfcc),
        _pool_stats(d1),
        _pool_stats(d2),
        _pool_stats(contrast),
        _pool_stats(chroma),
        _pool_stats(zcr),
        _pool_stats(rms),
        _pool_stats(cent),
        _pool_stats(roll),
        _pool_stats(bw),
    ]).astype(np.float32)


def _train_eval_svm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    splits_eval: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]],
    class_names: tuple[str, ...],
    C: float = 3.0,
) -> dict:
    """Fit StandardScaler + SVC(RBF, balanced) cleanly on `train` and report val/test/real_test."""
    t0 = time.perf_counter()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=SEED)),
    ])
    pipe.fit(X_tr, y_tr)
    fit_s = round(time.perf_counter() - t0, 3)

    res: dict = {"train_time_sec": fit_s}
    for sp, (X_sp, y_clip, w_clip_idx, n_clips) in splits_eval.items():
        probs = pipe.predict_proba(X_sp)
        if w_clip_idx is not None:
            agg = aggregate_windows_to_clips(probs, w_clip_idx, n_clips)
            pred_c = agg["mean_prob"][1]
        else:
            pred_c = np.argmax(probs, axis=1)
        res[sp] = compute_full_metrics(y_clip, pred_c, class_names)
    return res


def _eval_saved_v1_models(kw_clips: dict[str, list[tuple[Path, int]]]) -> dict:
    """Run read-only evaluation of saved V1 artifacts on identical KweriMoto val/test/real_test."""
    out: dict = {}

    # 1. V1 Pretrained SVM (models/best_model.pkl; V1 labels: 0=engine_knock, 1=healthy, 2=misfire)
    v1_svm = joblib.load(ROOT / "models" / "best_model.pkl")
    v1_le = joblib.load(ROOT / "models" / "label_encoder.pkl")
    v1_classes = list(v1_le.classes_)
    map_v1_to_c3 = {i: CLASSES_3.index(cname) for i, cname in enumerate(v1_classes)}

    svm_res = {}
    for sp in ("validation", "test", "real_test"):
        y_true, y_pred = [], []
        for p, c3_true in kw_clips[sp]:
            y, sr = librosa.load(str(p), sr=22050, duration=10.0, mono=True)
            feat = extract_v1_288_from_array(y, sr).reshape(1, -1)
            pred_v1 = int(v1_svm.predict(feat)[0])
            y_true.append(c3_true)
            y_pred.append(map_v1_to_c3[pred_v1])
        svm_res[sp] = compute_full_metrics(np.array(y_true), np.array(y_pred), CLASSES_3)
    out["A0b_V1_Saved_SVM_288"] = svm_res

    # 2. V1 Pretrained CNN (ml/autoguard_model.pt; 0=engine_knock, 1=healthy, 2=misfire)
    ckpt = torch.load(ROOT / "ml" / "autoguard_model.pt", map_location="cpu", weights_only=False)
    v1_cnn = AutoGuardCNN(n_classes=3)
    v1_cnn.load_state_dict(ckpt["model_state_dict"])
    v1_cnn.eval()

    cnn_res = {}
    for sp in ("validation", "test", "real_test"):
        y_true, y_pred = [], []
        for p, c3_true in kw_clips[sp]:
            flat = extract_v1_cnn_features(p)
            x_t = torch.from_numpy(flat).float().unsqueeze(0)
            with torch.no_grad():
                logits = v1_cnn(x_t)
                pred_v1 = int(torch.argmax(logits, dim=1).item())
            y_true.append(c3_true)
            y_pred.append(map_v1_to_c3[pred_v1])
        cnn_res[sp] = compute_full_metrics(np.array(y_true), np.array(y_pred), CLASSES_3)
    out["A0a_V1_Saved_CNN"] = cnn_res
    return out


def _train_quick_cnn_comparison(
    X_mel_tr: np.ndarray,
    y_tr: np.ndarray,
    X_mel_va: np.ndarray,
    y_va: np.ndarray,
    splits_eval: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]],
    class_names: tuple[str, ...],
    arch: str,
    epochs: int,
    use_weights: bool,
) -> dict:
    """Train V1 AutoGuardCNN vs CNNv2 on an identical spectrogram representation."""
    set_seed(SEED)
    n_c = len(class_names)
    if arch == "v1_cnn":
        # Re-use V1 architecture cleanly without modifying ml/train.py
        model = AutoGuardCNN(n_classes=n_c)
        # V1 AutoGuardCNN expects 128x128 input -> interpolate neatly ONLY if 128x64 passed
        need_flat = True
    else:
        model = AutoGuardCNNv2(n_classes=n_c)
        need_flat = False

    def _prep_tensor(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float()
        if t.ndim == 3:
            t = t.unsqueeze(1)
        if need_flat:
            if t.shape[-1] != 128:
                t = F.interpolate(t, size=(128, 128), mode="bilinear", align_corners=False)
            return t.view(t.shape[0], -1)
        return t

    xt_tr = _prep_tensor(X_mel_tr)
    yt_tr = torch.from_numpy(y_tr).long()
    ds = TensorDataset(xt_tr, yt_tr)
    loader = DataLoader(ds, batch_size=96, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    if use_weights:
        w = _compute_sqrt_class_weights(y_tr, n_c, power=0.55)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.04)
    else:
        crit = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def _batched_probs(arr: np.ndarray, bsz: int = 96) -> np.ndarray:
        xt = _prep_tensor(arr)
        ps = []
        model.eval()
        with torch.no_grad():
            for st in range(0, len(xt), bsz):
                ps.append(F.softmax(model(xt[st : st + bsz]), dim=1).cpu().numpy())
        return np.vstack(ps).astype(np.float32)

    best_f1 = -1.0
    best_sd = None
    va_w_idx = splits_eval["validation"][2]
    va_y_c = splits_eval["validation"][1]
    va_nc = splits_eval["validation"][3]

    t0 = time.perf_counter()
    for ep_i in range(epochs):
        ep_t0 = time.perf_counter()
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        p_va = _batched_probs(X_mel_va)
        if va_w_idx is not None:
            pred_c = aggregate_windows_to_clips(p_va, va_w_idx, va_nc)["mean_prob"][1]
        else:
            pred_c = np.argmax(p_va, axis=1)
        f1_va = compute_full_metrics(va_y_c, pred_c, class_names)["macro_f1"]
        print(f"    [{arch} | {n_c}-cls] Ep {ep_i+1:2d}/{epochs} va_F1={f1_va:.4f} ({time.perf_counter()-ep_t0:.1f}s)", flush=True)
        if f1_va > best_f1:
            best_f1 = f1_va
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    fit_s = round(time.perf_counter() - t0, 2)
    assert best_sd is not None
    model.load_state_dict(best_sd)

    res = {"train_time_sec": fit_s}
    for sp, (X_sp, y_clip, w_idx, nc) in splits_eval.items():
        probs = _batched_probs(X_sp)
        if w_idx is not None:
            pred_c = aggregate_windows_to_clips(probs, w_idx, nc)["mean_prob"][1]
        else:
            pred_c = np.argmax(probs, axis=1)
        res[sp] = compute_full_metrics(y_clip, pred_c, class_names)
    return res


def run_ablation() -> None:
    set_seed(SEED)
    t_all = time.perf_counter()

    # Load Track 1 V2 prepared splits (zero-leakage caches)
    t1 = {sp: dict(np.load(TRACK1_DIR / f"{sp}.npz", allow_pickle=False)) for sp in ("train", "validation", "test", "real_test")}

    # ── SUITE A: EXACT KWERIMOTO 3-CLASS CONTROLLED LADDER ──────────────────────
    print("\n=== SUITE A: CONTROLLED V1 vs V2 ABLATION ON EXACT KWERIMOTO 3-CLASS SPLIT ===", flush=True)
    kw_clips: dict[str, list[tuple[Path, int]]] = {}
    kw_c3_masks_clip: dict[str, np.ndarray] = {}
    kw_c3_masks_win: dict[str, np.ndarray] = {}
    kw_clip_remapped_win_idx: dict[str, tuple[np.ndarray, int, np.ndarray]] = {}

    for sp in ("train", "validation", "test", "real_test"):
        d = t1[sp]
        src = d["clip_sources"]
        mask_c = np.array([s.startswith("kwerimoto") for s in src], dtype=bool)
        kw_c3_masks_clip[sp] = mask_c
        # KweriMoto labels already happen to be {0: healthy, 1: engine_knock, 2: misfire}!
        assert set(d["y_clip"][mask_c].tolist()).issubset({0, 1, 2})
        kw_clips[sp] = [
            (ROOT / str(p), int(y))
            for p, y, m in zip(d["clip_rel_paths"], d["y_clip"], mask_c)
            if m
        ]
        old_to_new = {}
        new_i = 0
        for oi, m in enumerate(mask_c):
            if m:
                old_to_new[oi] = new_i
                new_i += 1
        w_mask = np.array([int(ci) in old_to_new for ci in d["win_clip_idx"]], dtype=bool)
        kw_c3_masks_win[sp] = w_mask
        new_w_idx = np.array([old_to_new[int(ci)] for ci in d["win_clip_idx"][w_mask]], dtype=np.int32)
        kw_clip_remapped_win_idx[sp] = (new_w_idx, new_i, d["y_clip"][mask_c].astype(np.int32))

    suite_a = _eval_saved_v1_models(kw_clips)

    # Extract V1 288-dim @22.05kHz whole-clip vs Standardized 16kHz whole-clip vs 16kHz 1.5s windows for KweriMoto
    print("  Extracting V1-288 features across KweriMoto stages (A1/A2/A3) ...", flush=True)
    v1_22k_clip, v2_16k_clip, v1_288_16k_win = {}, {}, {}
    for sp in ("train", "validation", "test", "real_test"):
        f_22, f_16_c, f_16_w = [], [], []
        for p, _y in kw_clips[sp]:
            y22, sr22 = librosa.load(str(p), sr=22050, duration=10.0, mono=True)
            f_22.append(extract_v1_288_from_array(y22, 22050))

            y16, sr16, _ = load_and_standardize(p, TARGET_SR)
            yn_c, _, _ = normalize_window_rms(y16, TARGET_RMS)
            f_16_c.append(extract_v1_288_from_array(yn_c, TARGET_SR))

            for seg, _s, _e, _sh in slice_windows(y16, TARGET_SR, WINDOW_SEC, WINDOW_SEC, max_windows_per_clip=7):
                yn_w, _, _ = normalize_window_rms(seg, TARGET_RMS)
                f_16_w.append(extract_v1_288_from_array(yn_w, TARGET_SR))

        v1_22k_clip[sp] = np.asarray(f_22, dtype=np.float32)
        v2_16k_clip[sp] = np.asarray(f_16_c, dtype=np.float32)
        v1_288_16k_win[sp] = np.asarray(f_16_w, dtype=np.float32)

    # Run A1..A4b with identical SVC(C=3.0, rbf, balanced)
    eval_clip_a = lambda feat_dict: {
        sp: (feat_dict[sp], kw_clip_remapped_win_idx[sp][2], None, kw_clip_remapped_win_idx[sp][1])
        for sp in ("train", "validation", "test", "real_test")
    }
    eval_win_a = lambda feat_dict: {
        sp: (feat_dict[sp], kw_clip_remapped_win_idx[sp][2], kw_clip_remapped_win_idx[sp][0], kw_clip_remapped_win_idx[sp][1])
        for sp in ("train", "validation", "test", "real_test")
    }
    y_tr_c3_clip = kw_clip_remapped_win_idx["train"][2]
    y_tr_c3_win = t1["train"]["y_win"][kw_c3_masks_win["train"]]

    suite_a["A1_V1_22k_10s_288MFCC_SVM"] = _train_eval_svm(v1_22k_clip["train"], y_tr_c3_clip, eval_clip_a(v1_22k_clip), CLASSES_3)
    suite_a["A2_Std16k_10s_288MFCC_SVM"] = _train_eval_svm(v2_16k_clip["train"], y_tr_c3_clip, eval_clip_a(v2_16k_clip), CLASSES_3)
    suite_a["A3_Std16k_1.5sWin_288MFCC_SVM"] = _train_eval_svm(v1_288_16k_win["train"], y_tr_c3_win, eval_win_a(v1_288_16k_win), CLASSES_3)

    v2_334_noraw_a = {sp: t1[sp]["X_win_tab"][kw_c3_masks_win[sp]][:, NO_RAW_MASK] for sp in ("train", "validation", "test", "real_test")}
    v2_336_withraw_a = {sp: t1[sp]["X_win_tab"][kw_c3_masks_win[sp]] for sp in ("train", "validation", "test", "real_test")}
    suite_a["A4a_Std16k_1.5sWin_334V2_NoRawE_SVM"] = _train_eval_svm(v2_334_noraw_a["train"], y_tr_c3_win, eval_win_a(v2_334_noraw_a), CLASSES_3)
    suite_a["A4b_Std16k_1.5sWin_336V2_WithRawE_SVM"] = _train_eval_svm(v2_336_withraw_a["train"], y_tr_c3_win, eval_win_a(v2_336_withraw_a), CLASSES_3)

    # Precompute V1 128x128 min-max spectrograms once across all 2,016 clips for A5a & B0b
    print("  Extracting V1 22.05kHz 128x128 Mel & 288-dim clip features once across splits ...", flush=True)
    v1_9_clip_288, v1_9_mel128 = {}, {}
    for sp in ("train", "validation", "test", "real_test"):
        f_clip288, f_mel = [], []
        for rel in t1[sp]["clip_rel_paths"]:
            p = ROOT / str(rel)
            y22, _sr22 = librosa.load(str(p), sr=22050, duration=10.0, mono=True)
            f_clip288.append(extract_v1_288_from_array(y22, 22050))
            f_mel.append(extract_v1_cnn_features(p).reshape(128, 128))
        v1_9_clip_288[sp] = np.asarray(f_clip288, dtype=np.float32)
        v1_9_mel128[sp] = np.asarray(f_mel, dtype=np.float32)

    # A5: Architecture & Training comparison on identical KweriMoto 3-class split
    v1_kw_mel128 = {sp: v1_9_mel128[sp][kw_c3_masks_clip[sp]] for sp in ("train", "validation", "test", "real_test")}
    suite_a["A5a_V1Arch_CNN_5ep_Kweri721"] = _train_quick_cnn_comparison(
        v1_kw_mel128["train"], y_tr_c3_clip, v1_kw_mel128["validation"], kw_clip_remapped_win_idx["validation"][2],
        eval_clip_a(v1_kw_mel128), CLASSES_3, arch="v1_cnn", epochs=5, use_weights=False
    )
    mel_kw = {sp: t1[sp]["X_win_mel"][kw_c3_masks_win[sp]] for sp in ("train", "validation", "test", "real_test")}
    suite_a["A5b_CNN_V2_15ep_OnV2Mel"] = _train_quick_cnn_comparison(
        mel_kw["train"], y_tr_c3_win, mel_kw["validation"], y_tr_c3_win[:len(mel_kw["validation"])],
        eval_win_a(mel_kw), CLASSES_3, arch="v2_cnn", epochs=15, use_weights=True
    )

    # A5c: Clip-Multiscale HistGB on exact 3-class KweriMoto
    def _ms3(sp: str):
        xw = t1[sp]["X_win_tab"][kw_c3_masks_win[sp]]
        wi, nc, _ = kw_clip_remapped_win_idx[sp]
        xg = t1[sp]["X_clip_global_tab"][kw_c3_masks_clip[sp]]
        xm = np.array([xw[wi == i].mean(axis=0) for i in range(nc)], dtype=np.float32)
        return np.hstack([xm, xg]).astype(np.float32)

    ms3 = {sp: _ms3(sp) for sp in ("train", "validation", "test", "real_test")}
    hgb3 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=SEED)
    t0 = time.perf_counter()
    hgb3.fit(ms3["train"], y_tr_c3_clip)
    hgb3_res = {"train_time_sec": round(time.perf_counter() - t0, 2)}
    for sp in ("train", "validation", "test", "real_test"):
        hgb3_res[sp] = compute_full_metrics(kw_clip_remapped_win_idx[sp][2], hgb3.predict(ms3[sp]), CLASSES_3)
    suite_a["A5c_HistGB_Multiscale_3Class"] = hgb3_res

    for k, m in suite_a.items():
        va_m, te_m, rt_m = m.get("validation", {}), m.get("test", {}), m.get("real_test", {})
        print(
            f"  [Suite A] {k:38s} | Val={va_m.get('accuracy',0):.4f} (F1={va_m.get('macro_f1',0):.4f}) | "
            f"Test={te_m.get('accuracy',0):.4f} (F1={te_m.get('macro_f1',0):.4f}, FHR={te_m.get('false_healthy_rate',0):.4f}) | "
            f"RealTest={rt_m.get('accuracy',0):.4f}",
            flush=True,
        )

    # Save Suite A checkpoint before Suite B
    with open(REPORTS_P3_DIR / "v1_vs_v2_ablation_suite_a.json", "w", encoding="utf-8") as f:
        json.dump(suite_a, f, indent=2)

    # ── SUITE B: FULL 9-CLASS IDENTICAL LEAKAGE-FREE SPLIT ABLATION ───────────
    print("\n=== SUITE B: CONTROLLED V1-STYLE vs V2 ABLATION ON FULL 9-CLASS TRACK 1 ===", flush=True)
    v1_9_win_288_16k = {}
    # Extract 16 kHz V1-288 on 1.5s windows for B1
    for sp in ("train", "validation", "test", "real_test"):
        fw = []
        for rel in t1[sp]["clip_rel_paths"]:
            y16, sr16, _ = load_and_standardize(ROOT / str(rel), TARGET_SR)
            for seg, _s, _e, _sh in slice_windows(y16, TARGET_SR, WINDOW_SEC, WINDOW_SEC, max_windows_per_clip=7):
                yn_w, _, _ = normalize_window_rms(seg, TARGET_RMS)
                fw.append(extract_v1_288_from_array(yn_w, TARGET_SR))
        v1_9_win_288_16k[sp] = np.asarray(fw, dtype=np.float32)

    eval_clip_b = lambda fd: {sp: (fd[sp], t1[sp]["y_clip"], None, len(t1[sp]["y_clip"])) for sp in ("train", "validation", "test", "real_test")}
    eval_win_b = lambda fd: {sp: (fd[sp], t1[sp]["y_clip"], t1[sp]["win_clip_idx"], len(t1[sp]["y_clip"])) for sp in ("train", "validation", "test", "real_test")}

    suite_b = {}
    suite_b["B0_V1Style_22k_10s_288MFCC_SVM"] = _train_eval_svm(v1_9_clip_288["train"], t1["train"]["y_clip"], eval_clip_b(v1_9_clip_288), CLASSES_9)
    suite_b["B0b_V1Style_CNN_5ep_OnV1Mel128"] = _train_quick_cnn_comparison(
        v1_9_mel128["train"], t1["train"]["y_clip"], v1_9_mel128["validation"], t1["validation"]["y_clip"],
        eval_clip_b(v1_9_mel128), CLASSES_9, arch="v1_cnn", epochs=5, use_weights=False
    )
    suite_b["B1_Std16k_1.5sWin_288MFCC_SVM"] = _train_eval_svm(v1_9_win_288_16k["train"], t1["train"]["y_win"], eval_win_b(v1_9_win_288_16k), CLASSES_9)

    b_noraw = {sp: t1[sp]["X_win_tab"][:, NO_RAW_MASK] for sp in ("train", "validation", "test", "real_test")}
    b_withraw = {sp: t1[sp]["X_win_tab"] for sp in ("train", "validation", "test", "real_test")}
    suite_b["B2a_Std16k_1.5sWin_334V2_NoRawE_SVM"] = _train_eval_svm(b_noraw["train"], t1["train"]["y_win"], eval_win_b(b_noraw), CLASSES_9)
    suite_b["B2b_Std16k_1.5sWin_336V2_WithRawE_SVM"] = _train_eval_svm(b_withraw["train"], t1["train"]["y_win"], eval_win_b(b_withraw), CLASSES_9)

    for k, m in suite_b.items():
        va_m, te_m, rt_m = m.get("validation", {}), m.get("test", {}), m.get("real_test", {})
        print(
            f"  [Suite B] {k:38s} | Val={va_m['accuracy']:.4f} (F1={va_m['macro_f1']:.4f}) | "
            f"Test={te_m['accuracy']:.4f} (BalAcc={te_m['balanced_accuracy']:.4f}, F1={te_m['macro_f1']:.4f}, FHR={te_m['false_healthy_rate']:.4f}) | "
            f"RealTest={rt_m['accuracy']:.4f}",
            flush=True,
        )

    out_path = REPORTS_P3_DIR / "v1_vs_v2_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "total_elapsed_sec": round(time.perf_counter() - t_all, 2),
                "suite_a_kwerimoto_3class": suite_a,
                "suite_b_track1_9class": suite_b,
            },
            f,
            indent=2,
        )
    print(f"\nSaved controlled ablation results to {out_path}")


if __name__ == "__main__":
    run_ablation()
