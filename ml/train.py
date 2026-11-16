"""
ml/train.py
===========
AutoGuard AI — PyTorch CNN training script.

Run:
    python ml/train.py

Requires:
    data/prepared/train/, validation/, test/
    (generate with: python ml/features.py)
"""

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.features import load_prepared_split, CLASSES  # noqa: E402

MODEL_OUT = Path(__file__).resolve().parent / "autoguard_model.pt"
LOG_OUT   = Path(__file__).resolve().parent / "autoguard_training_log.json"

# ── Config ─────────────────────────────────────────────────────────────────────
N_EPOCHS    = 5
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY= 1e-4
SEED        = 42
N_MELS      = 128
N_FRAMES    = 128
FEATURE_DIM = N_MELS * N_FRAMES  # 16 384


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Wrap numpy arrays in a TensorDataset.
    num_workers=0  — no multiprocessing, safe on Windows CPU.
    pin_memory=False — not needed without GPU.
    """
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


# ── Class-weight helper ────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights: weight[c] = N / (n_classes * count[c])."""
    counts  = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ── Model ──────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two conv layers, BN, ReLU, MaxPool2d(2), Dropout2d."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoGuardCNN(nn.Module):
    """
    Lightweight CNN for engine sound classification.

    Input  : flat tensor (B, 16384)
    Reshape: (B, 1, 128, 128)  via .view(-1, 1, 128, 128)
    Output : logits (B, n_classes)

    Spatial: 128 → 64 → 32 → 16  (three MaxPool2d(2))
    Channels:  1 → 32 → 64 → 128
    After GAP: (B, 128) → head → (B, n_classes)
    ~321 k trainable parameters.
    """

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, N_MELS, N_FRAMES)  # (B, 1, 128, 128)
        x = self.features(x)                   # (B, 128, 16, 16)
        x = self.gap(x).flatten(1)            # (B, 128)
        return self.head(x)                   # (B, n_classes)


# ── Training / evaluation ──────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()

        # Print batch progress every 20 batches or on the last batch
        if batch_idx % 20 == 0 or batch_idx == n_batches:
            print(
                f"  epoch {epoch}/{total_epochs}  "
                f"batch {batch_idx}/{n_batches}  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_correct  = 0
    n_seen     = 0
    n_batches  = 0

    for xb, yb in loader:
        xb, yb  = xb.to(device), yb.to(device)
        logits   = model(xb)
        total_loss += criterion(logits, yb).item()
        n_correct  += (logits.argmax(dim=1) == yb).sum().item()
        n_seen     += len(yb)
        n_batches  += 1

    mean_loss = total_loss / max(n_batches, 1)
    accuracy  = n_correct  / max(n_seen, 1)
    return mean_loss, accuracy


@torch.no_grad()
def predict_all(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(preds.tolist())
    return np.array(y_true), np.array(y_pred)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60, flush=True)
    print("  AutoGuard AI — CNN Training", flush=True)
    print("=" * 60, flush=True)
    print(f"  device      : {device}", flush=True)
    print(f"  epochs      : {N_EPOCHS}", flush=True)
    print(f"  batch size  : {BATCH_SIZE}", flush=True)
    print(f"  lr          : {LR}  weight_decay={WEIGHT_DECAY}", flush=True)
    print(f"  seed        : {SEED}", flush=True)
    print(flush=True)

    # 1. Load pre-computed numpy arrays ───────────────────────────────────────
    print("[1/5] Loading prepared feature arrays ...", flush=True)
    X_tr, y_tr = load_prepared_split("train")
    X_va, y_va = load_prepared_split("validation")
    X_te, y_te = load_prepared_split("test")

    for name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
        counts = {CLASSES[k]: v for k, v in sorted(Counter(y.tolist()).items())}
        print(f"  {name:6s}  shape={X.shape}  classes={counts}", flush=True)
    print(flush=True)

    # 2. Data loaders (num_workers=0, no multiprocessing) ─────────────────────
    train_loader = make_loader(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_va, y_va, BATCH_SIZE, shuffle=False)
    test_loader  = make_loader(X_te, y_te, BATCH_SIZE, shuffle=False)

    # 3. Model, loss, optimiser ───────────────────────────────────────────────
    print("[2/5] Building model ...", flush=True)
    model    = AutoGuardCNN(n_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters: {n_params:,}", flush=True)

    cw        = compute_class_weights(y_tr, len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=N_EPOCHS, eta_min=1e-5
    )

    print(
        "  class weights: "
        + "  ".join(f"{CLASSES[i]}={cw[i].item():.3f}" for i in range(len(CLASSES))),
        flush=True,
    )
    print(flush=True)

    # 4. Training loop ─────────────────────────────────────────────────────────
    print(f"[3/5] Training for {N_EPOCHS} epochs ...", flush=True)
    print(flush=True)

    best_val_loss = float("inf")
    best_epoch    = 0
    best_state    = None
    epoch_log     = []
    t_total       = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        print(f"--- Epoch {epoch}/{N_EPOCHS} ---", flush=True)
        t_ep = time.time()

        tr_loss            = train_epoch(model, train_loader, criterion, optimiser, device, epoch, N_EPOCHS)
        va_loss, va_acc    = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        ep_time = time.time() - t_ep

        print(
            f"  EPOCH {epoch:>2}  "
            f"train_loss={tr_loss:.4f}  "
            f"val_loss={va_loss:.4f}  "
            f"val_acc={va_acc:.4f}  "
            f"time={ep_time:.1f}s",
            flush=True,
        )
        print(flush=True)

        epoch_log.append({
            "epoch":      epoch,
            "train_loss": round(float(tr_loss), 6),
            "val_loss":   round(float(va_loss), 6),
            "val_acc":    round(float(va_acc),  6),
            "time_sec":   round(ep_time, 1),
        })

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ** New best checkpoint saved (epoch {epoch}) **", flush=True)
            print(flush=True)

    total_time = time.time() - t_total
    print(f"[3/5] Training complete.  best_epoch={best_epoch}  val_loss={best_val_loss:.4f}  total={total_time:.1f}s", flush=True)
    print(flush=True)

    # 5. Restore best checkpoint ───────────────────────────────────────────────
    print(f"[4/5] Restoring best checkpoint (epoch {best_epoch}) ...", flush=True)
    model.load_state_dict(best_state)
    print(flush=True)

    # 6. Evaluate on held-out test split ───────────────────────────────────────
    print("[5/5] Evaluating on held-out test split ...", flush=True)
    y_true, y_pred      = predict_all(model, test_loader, device)
    te_loss, te_acc     = eval_epoch(model, test_loader, criterion, device)

    print(f"  test_loss    = {te_loss:.4f}", flush=True)
    print(f"  test_accuracy= {te_acc:.4f}  ({te_acc * 100:.1f}%)", flush=True)
    print(flush=True)

    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    print("  Classification report:", flush=True)
    print(report, flush=True)

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion matrix (rows=true, cols=pred):", flush=True)
    header = "  " + "  ".join(f"{c:>14}" for c in CLASSES)
    print(header, flush=True)
    for i, row in enumerate(cm):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
        line   = f"  {CLASSES[i]:14s}  " + "  ".join(f"{v:>14}" for v in row)
        line  += f"   recall={recall:.0%}"
        print(line, flush=True)
    print(flush=True)

    # 7. Save model weights ────────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict": best_state,
            "metadata": {
                "model_class":   "AutoGuardCNN",
                "classes":       CLASSES,
                "n_mels":        N_MELS,
                "n_frames":      N_FRAMES,
                "feature_dim":   FEATURE_DIM,
                "best_epoch":    best_epoch,
                "best_val_loss": round(float(best_val_loss), 6),
                "test_accuracy": round(float(te_acc), 6),
                "hyperparams": {
                    "n_epochs":     N_EPOCHS,
                    "batch_size":   BATCH_SIZE,
                    "lr":           LR,
                    "weight_decay": WEIGHT_DECAY,
                    "seed":         SEED,
                    "optimiser":    "Adam",
                    "scheduler":    "CosineAnnealingLR",
                    "loss":         "CrossEntropyLoss(class-weighted)",
                },
                "training_time_sec": round(total_time, 1),
            },
        },
        str(MODEL_OUT),
    )

    with open(str(LOG_OUT), "w") as f:
        json.dump(
            {
                "best_epoch":    best_epoch,
                "best_val_loss": round(float(best_val_loss), 6),
                "test_accuracy": round(float(te_acc), 6),
                "per_epoch":     epoch_log,
            },
            f,
            indent=2,
        )

    print(f"  Model saved      → {MODEL_OUT}", flush=True)
    print(f"  Training log     → {LOG_OUT}", flush=True)
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  Done.  best_epoch={best_epoch}  test_acc={te_acc * 100:.1f}%", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
