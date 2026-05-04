import os
import time
import torch
import torch.nn as nn
from sklearn import metrics

from src.loader import load_data
from src.model import TripleMRNet


def run_epoch(model, loader, optimizer=None, scaler=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    preds, labels = [], []

    for ax, sa, co, y in loader:
        ax = ax.to(device, non_blocking=True)
        sa = sa.to(device, non_blocking=True)
        co = co.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logit = model(ax, sa, co)
                    loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logit = model(ax, sa, co)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
                if train:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        p = torch.sigmoid(logit).detach().cpu().numpy()[0][0]
        t = y.detach().cpu().numpy()[0][0]
        preds.append(p)
        labels.append(t)

    avg_loss = total_loss / max(1, len(loader))
    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)
    return avg_loss, auc


def train(
    rundir="runs_acl",
    task="acl",
    epochs=10,
    lr=3e-4,
    weight_decay=1e-5,
    data_dir="data",
    labels_dir="labels",
    num_workers=2,
    use_amp=True
):
    os.makedirs(rundir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, valid_loader = load_data(
        task=task, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    model = TripleMRNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    best_auc = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, optimizer, scaler, device)
        va_loss, va_auc = run_epoch(model, valid_loader, None, None, device)

        scheduler.step(va_loss)

        print(f"[Epoch {ep}] "
              f"Train Loss: {tr_loss:.4f} | Train AUC: {tr_auc:.4f} || "
              f"Valid Loss: {va_loss:.4f} | Valid AUC: {va_auc:.4f}")

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), os.path.join(rundir, f"best_auc_{va_auc:.4f}.pth"))
            print("💾 Save BEST")

    print(f"Done. Best Val AUC: {best_auc:.4f}")
