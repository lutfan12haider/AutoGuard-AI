"""
ml/v2/track1_neural.py
----------------------
AutoGuard AI V2 — Phase 3 Deep Learning & Hybrid Multimodal Models (9 Classes).

Trains and evaluates:
  1. `AutoGuardCNNv2`    : Frequency-aware SE-ResNet CNN on `(128, 64)` global_db Log-Mel
                           with explicit normalized frequency coordinate channel & dual
                           band statistics pooling (~340k params).
  2. `AutoGuardHybridNet`: End-to-end Multimodal Fusion combining `AutoGuardCNNv2` encoder
                           with a LayerNorm Residual MLP over the 336-dim physics/HPSS/MFCC
                           tabular features (scaler fit strictly on `train`).

Strict anti-leakage & reproducibility:
  - Fixed seed (`SEED = 42`), deterministic PyTorch CPU mode.
  - Pure Phase 2 features without synthetic augmentation (Phase 4 isolates augmentation).
  - Checkpoint selection strictly by `validation` Clip Macro-F1 (never `test`).
  - Logs per-epoch train & validation loss, accuracy, Macro-F1, and False-Healthy Rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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


class SEBlock(nn.Module):
    """Channel Squeeze-and-Excitation recalibration."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ConvSEResBlock(nn.Module):
    """
    MKLDNN-accelerated 2D Conv block with Squeeze-and-Excitation and residual skip.
    Uses standard dense Conv2d (`groups=1`) for optimal CPU oneDNN throughput (~10x faster
    than depthwise grouped convs on CPU).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(F.silu(self.bn(self.conv(x))))
        out = F.silu(out + self.skip(x))
        return self.drop(out)


class MelBackbone(nn.Module):
    """CPU-friendly frequency-aware CNN backbone returning 512-dim band-pooled embedding."""

    def __init__(self) -> None:
        super().__init__()
        # Fixed frequency coordinate grid [-1, +1] across 128 Mel bins
        freq_grid = torch.linspace(-1.0, 1.0, steps=128).view(1, 1, 128, 1).repeat(1, 1, 1, 64)
        self.register_buffer("freq_grid", freq_grid)

        self.stages = nn.Sequential(
            ConvSEResBlock(2, 24, stride=2, dropout=0.05),    # -> (24, 64, 32)
            ConvSEResBlock(24, 48, stride=2, dropout=0.10),   # -> (48, 32, 16)
            ConvSEResBlock(48, 96, stride=2, dropout=0.15),   # -> (96, 16, 8)
            ConvSEResBlock(96, 128, stride=1, dropout=0.20),  # -> (128, 16, 8)
        )
        # Dual pooling over Low (0-4k) vs High (4-8k) band split (H=2, W=1 -> 2*128*2 = 512)
        self.band_avg = nn.AdaptiveAvgPool2d((2, 1))
        self.band_max = nn.AdaptiveMaxPool2d((2, 1))
        self.out_dim = 512

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, 128, 64)
        b = mel.shape[0]
        fg = self.freq_grid.expand(b, -1, -1, -1)
        x = torch.cat([mel, fg], dim=1)  # (B, 2, 128, 64)
        feat = self.stages(x)
        p_avg = self.band_avg(feat).flatten(1)
        p_max = self.band_max(feat).flatten(1)
        return torch.cat([p_avg, p_max], dim=1)


class AutoGuardCNNv2(nn.Module):
    """Frequency-aware SE-ResNet classifier on 128x64 Log-Mel spectrograms."""

    def __init__(self, n_classes: int = 9) -> None:
        super().__init__()
        self.backbone = MelBackbone()
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.out_dim),
            nn.Dropout(0.35),
            nn.Linear(self.backbone.out_dim, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(mel))


class AutoGuardHybridNet(nn.Module):
    """
    Multimodal Hybrid classifier fusing:
      - Frequency-aware SE-ResNet (`MelBackbone`, 512 -> 128)
      - Physics Tabular Residual MLP (336 -> 128)
      - Gated fusion layer & classification head
    """

    def __init__(self, tab_dim: int = 336, n_classes: int = 9) -> None:
        super().__init__()
        self.mel_backbone = MelBackbone()
        self.mel_proj = nn.Sequential(
            nn.LayerNorm(self.mel_backbone.out_dim),
            nn.Dropout(0.35),
            nn.Linear(self.mel_backbone.out_dim, 128),
            nn.SiLU(inplace=True),
        )
        self.tab_in = nn.Sequential(
            nn.Linear(tab_dim, 160),
            nn.LayerNorm(160),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )
        # Modality gate weights contribution dynamically per acoustic sample
        self.gate = nn.Sequential(
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, n_classes),
        )

    def forward(self, mel: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        z_m = self.mel_proj(self.mel_backbone(mel))
        z_t = self.tab_in(tab)
        z = torch.cat([z_m, z_t], dim=1)
        z = z * (0.5 + self.gate(z))  # residual modality scaling in [0.5, 1.5]
        return self.head(z)


def _compute_sqrt_class_weights(y: np.ndarray, n_classes: int, power: float = 0.55) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = (counts.max() / np.maximum(counts, 1.0)) ** power
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32)


def _predict_neural(
    model: nn.Module,
    loader: DataLoader,
    is_hybrid: bool,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, float]:
    model.eval()
    probs_all = []
    tot_loss = 0.0
    tot_n = 0
    with torch.no_grad():
        for batch in loader:
            if is_hybrid:
                mel, tab, yb = batch
                logits = model(mel, tab)
            else:
                mel, yb = batch
                logits = model(mel)
            if criterion is not None:
                loss = criterion(logits, yb)
                tot_loss += float(loss.item()) * len(yb)
                tot_n += len(yb)
            probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
    avg_loss = tot_loss / max(1, tot_n)
    return np.vstack(probs_all).astype(np.float32), avg_loss


def train_and_evaluate_neural(
    model_key: str,
    model: nn.Module,
    is_hybrid: bool,
    splits: dict[str, dict[str, np.ndarray]],
    tab_scaled: dict[str, np.ndarray] | None,
    rms_feat_idx: int,
    n_epochs: int = 18,
    batch_size: int = 96,
    lr: float = 2.0e-3,
    weight_decay: float = 2e-3,
    patience: int = 6,
) -> tuple[dict, dict[str, np.ndarray]]:
    set_seed(SEED)
    n_classes = len(CLASSES_9)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _make_loader(sp: str, shuffle: bool) -> DataLoader:
        mel_t = torch.from_numpy(splits[sp]["X_win_mel"][:, None, :, :]).float()
        y_t = torch.from_numpy(splits[sp]["y_win"]).long()
        if is_hybrid:
            assert tab_scaled is not None
            tab_t = torch.from_numpy(tab_scaled[sp]).float()
            ds = TensorDataset(mel_t, tab_t, y_t)
        else:
            ds = TensorDataset(mel_t, y_t)
        g = torch.Generator().manual_seed(SEED) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)

    train_loader_shuf = _make_loader("train", shuffle=True)
    eval_loaders = {sp: _make_loader(sp, shuffle=False) for sp in ("train", "validation", "test", "real_test")}

    cls_w = _compute_sqrt_class_weights(splits["train"]["y_win"], n_classes, power=0.55)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.05)

    n_va_clips = int(splits["validation"]["y_clip"].shape[0])
    y_va_clip = splits["validation"]["y_clip"]
    win_clip_va = splits["validation"]["win_clip_idx"]

    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []
    t_train_start = time.perf_counter()

    print(f"\n=== TRAINING {model_key} (params={n_params:,}) ===", flush=True)
    for epoch in range(1, n_epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0

        for batch in train_loader_shuf:
            optimizer.zero_grad()
            if is_hybrid:
                mel, tab, yb = batch
                logits = model(mel, tab)
            else:
                mel, yb = batch
                logits = model(mel)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            bs = len(yb)
            tr_loss_sum += float(loss.item()) * bs
            tr_correct += int((torch.argmax(logits, dim=1) == yb).sum().item())
            tr_total += bs

        scheduler.step()
        tr_loss = tr_loss_sum / max(1, tr_total)
        tr_acc = tr_correct / max(1, tr_total)

        p_va_w, va_loss = _predict_neural(model, eval_loaders["validation"], is_hybrid, criterion)
        va_win_acc = float(np.mean(np.argmax(p_va_w, axis=1) == splits["validation"]["y_win"]))
        agg_va = aggregate_windows_to_clips(p_va_w, win_clip_va, n_va_clips)
        m_va_c = compute_full_metrics(y_va_clip, agg_va["mean_prob"][1], CLASSES_9)

        ep_sec = time.perf_counter() - ep_t0
        rec = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_win_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_win_acc": round(va_win_acc, 4),
            "val_clip_acc": m_va_c["accuracy"],
            "val_clip_macro_f1": m_va_c["macro_f1"],
            "val_false_healthy_rate": m_va_c["false_healthy_rate"],
            "time_sec": round(ep_sec, 2),
        }
        history.append(rec)
        print(
            f"  Ep {epoch:2d}/{n_epochs} | tr_loss={tr_loss:.4f} tr_wAcc={tr_acc:.4f} | "
            f"va_loss={va_loss:.4f} va_cAcc={m_va_c['accuracy']:.4f} va_F1={m_va_c['macro_f1']:.4f} "
            f"FHR={m_va_c['false_healthy_rate']:.4f} ({ep_sec:.1f}s)",
            flush=True,
        )

        # Model selection & early stopping strictly on Validation Clip Macro-F1
        if m_va_c["macro_f1"] > best_score:
            best_score = m_va_c["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"  [EarlyStop] No val Macro-F1 improvement for {patience} epochs (best_ep={best_epoch}).", flush=True)
            break

    train_time = time.perf_counter() - t_train_start
    assert best_state is not None
    model.load_state_dict(best_state)

    ckpt_path = OUT_MODELS_DIR / f"{model_key.lower()}.pt"
    torch.save(
        {
            "model_key": model_key,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_clip_macro_f1": best_score,
            "classes": list(CLASSES_9),
            "n_params": n_params,
        },
        ckpt_path,
    )

    out = {
        "model_name": model_key,
        "input_mode": "mel_128x64_plus_tab336" if is_hybrid else "mel_128x64_freq_coord",
        "n_params": n_params,
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 2),
        "history": history,
        "window_level": {},
        "clip_aggregations": {},
        "clip_length_breakdown": {},
    }
    probs_cache: dict[str, np.ndarray] = {}
    t_inf_tot = 0.0
    n_inf_tot = 0

    for sp in ("train", "validation", "test", "real_test"):
        d = splits[sp]
        t0 = time.perf_counter()
        p_w, _ = _predict_neural(model, eval_loaders[sp], is_hybrid, None)
        dt_inf = time.perf_counter() - t0
        if sp in ("validation", "test"):
            t_inf_tot += dt_inf
            n_inf_tot += len(p_w)

        pred_w = np.argmax(p_w, axis=1)
        m_w = compute_full_metrics(d["y_win"], pred_w, CLASSES_9)
        m_w["infer_ms_per_window"] = round(dt_inf * 1000.0 / max(1, len(p_w)), 4)
        out["window_level"][sp] = m_w
        probs_cache[f"{model_key.lower()}__{sp}_win_probs"] = p_w

        n_clips = int(d["y_clip"].shape[0])
        win_rms = np.power(10.0, d["X_win_tab"][:, rms_feat_idx])
        agg_dict = aggregate_windows_to_clips(p_w, d["win_clip_idx"], n_clips, win_rms=win_rms)
        out["clip_aggregations"][sp] = {}
        for agg_name, (p_c, pred_c) in agg_dict.items():
            out["clip_aggregations"][sp][agg_name] = compute_full_metrics(d["y_clip"], pred_c, CLASSES_9)
            probs_cache[f"{model_key.lower()}__{sp}_clip_probs_{agg_name}"] = p_c

        if sp in ("validation", "test"):
            counts = np.bincount(d["win_clip_idx"], minlength=n_clips)
            _, pred_mean = agg_dict["mean_prob"]
            _, pred_conf = agg_dict["conf_weighted"]
            _, pred_top2 = agg_dict["top2_conf"]
            multi_mask = counts >= 2
            single_mask = counts == 1
            out["clip_length_breakdown"][sp] = {
                "single_window_n": int(single_mask.sum()),
                "single_window_mean_acc": round(float(np.mean(pred_mean[single_mask] == d["y_clip"][single_mask])), 4),
                "multi_window_n": int(multi_mask.sum()),
                "multi_window_win_acc": round(float(np.mean(pred_w[np.isin(d["win_clip_idx"], np.where(multi_mask)[0])] == d["y_win"][np.isin(d["win_clip_idx"], np.where(multi_mask)[0])])), 4),
                "multi_window_mean_acc": round(float(np.mean(pred_mean[multi_mask] == d["y_clip"][multi_mask])), 4),
                "multi_window_conf_acc": round(float(np.mean(pred_conf[multi_mask] == d["y_clip"][multi_mask])), 4),
                "multi_window_top2_acc": round(float(np.mean(pred_top2[multi_mask] == d["y_clip"][multi_mask])), 4),
            }

    out["eval_infer_ms_per_window"] = round(t_inf_tot * 1000.0 / max(1, n_inf_tot), 4)
    return out, probs_cache


def main() -> None:
    set_seed(SEED)
    t_all = time.perf_counter()

    splits = {}
    for sp in ("train", "validation", "test", "real_test"):
        d = np.load(TRACK1_DATA_DIR / f"{sp}.npz", allow_pickle=False)
        splits[sp] = {k: d[k] for k in d.files}
    rms_feat_idx = TABULAR_FEATURE_NAMES.index("log10_raw_rms")

    # Fit StandardScaler STRICTLY on training tabular windows for HybridNet
    tab_scaler = StandardScaler()
    tab_scaler.fit(splits["train"]["X_win_tab"])
    joblib.dump(tab_scaler, OUT_MODELS_DIR / "hybrid_tab_scaler.joblib")
    tab_scaled = {
        sp: np.clip(tab_scaler.transform(splits[sp]["X_win_tab"]), -8.0, 8.0).astype(np.float32)
        for sp in ("train", "validation", "test", "real_test")
    }

    cnn_v2 = AutoGuardCNNv2(n_classes=len(CLASSES_9))
    res_cnn, cache_cnn = train_and_evaluate_neural(
        "CNN_V2", cnn_v2, False, splits, None, rms_feat_idx, n_epochs=18, batch_size=96, lr=2.0e-3, patience=6
    )

    hybrid_v2 = AutoGuardHybridNet(tab_dim=splits["train"]["X_win_tab"].shape[1], n_classes=len(CLASSES_9))
    res_hyb, cache_hyb = train_and_evaluate_neural(
        "HybridNet_V2", hybrid_v2, True, splits, tab_scaled, rms_feat_idx, n_epochs=18, batch_size=96, lr=1.5e-3, patience=6
    )

    all_caches = {}
    all_caches.update(cache_cnn)
    all_caches.update(cache_hyb)
    np.savez_compressed(OUT_MODELS_DIR / "neural_prob_caches.npz", **all_caches)

    payload = {
        "seed": SEED,
        "total_elapsed_sec": round(time.perf_counter() - t_all, 2),
        "models": {"CNN_V2": res_cnn, "HybridNet_V2": res_hyb},
    }
    with open(REPORTS_P3_DIR / "track1_neural_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== TRACK 1 NEURAL SUMMARY (CLIP-LEVEL TEST) ===", flush=True)
    for k, m in payload["models"].items():
        te_m = m["clip_aggregations"]["test"]["mean_prob"]
        va_m = m["clip_aggregations"]["validation"]["mean_prob"]
        rt_m = m["clip_aggregations"]["real_test"]["mean_prob"]
        print(
            f"  {k:14s} | BestEp={m['best_epoch']:2d} | Val Acc={va_m['accuracy']:.4f} F1={va_m['macro_f1']:.4f} | "
            f"Test Acc={te_m['accuracy']:.4f} BalAcc={te_m['balanced_accuracy']:.4f} F1={te_m['macro_f1']:.4f} FHR={te_m['false_healthy_rate']:.4f} | "
            f"RealTest Acc={rt_m['accuracy']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
