import os
import time
import json
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from src.loader import load_data
from src.model import TripleMRNet


# ================= RUN 1 EPOCH =================
def run_epoch(model, loader, optimizer=None, scaler=None, device="cuda"):
    train = optimizer is not None

    model.train() if train else model.eval()

    total_loss = 0
    preds, labels = [], []

    loop = tqdm(loader, leave=False)

    for ax, sa, co, y in loop:
        ax = ax.to(device, non_blocking=True)
        sa = sa.to(device, non_blocking=True)
        co = co.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    logit = model(ax, sa, co)
                    loss = F.binary_cross_entropy_with_logits(logit, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logit = model(ax, sa, co)
                loss = F.binary_cross_entropy_with_logits(logit, y)

                if train:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()

        pred = torch.sigmoid(logit).detach().cpu().numpy()[0][0]
        label = y.detach().cpu().numpy()[0][0]

        preds.append(pred)
        labels.append(label)

        loop.set_description(f"{'Train' if train else 'Valid'} Loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)

    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc


# ================= TRAIN =================
def train(
    rundir="runs_acl",
    task="acl",
    epochs=8,
    lr=3e-4,
    weight_decay=1e-5,
    data_dir="data",
    labels_dir="labels",
    num_workers=2,
    use_amp=True
):
    os.makedirs(rundir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ===== LOAD DATA =====
    train_loader, valid_loader = load_data(
        task=task,
        data_dir=data_dir,
        labels_dir=labels_dir,
        num_workers=num_workers
    )

    # ===== MODEL =====
    model = TripleMRNet().to(device)

    # ===== OPTIMIZER =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.3
    )

    scaler = torch.amp.GradScaler(
        enabled=(use_amp and device == "cuda")
    )

    best_auc = 0.0

    # ===== SAVE CONFIG =====
    with open(os.path.join(rundir, "config.json"), "w") as f:
        json.dump({
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay
        }, f, indent=4)

    # ================= LOOP =================
    for epoch in range(1, epochs + 1):
        start = time.time()

        print(f"\n🚀 Epoch {epoch}/{epochs}")

        train_loss, train_auc = run_epoch(
            model, train_loader, optimizer, scaler, device
        )

        valid_loss, valid_auc = run_epoch(
            model, valid_loader, None, None, device
        )

        scheduler.step(valid_loss)

        print(f"[Epoch {epoch}] "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} || "
              f"Valid Loss: {valid_loss:.4f} | Valid AUC: {valid_auc:.4f}")

        # ===== SAVE BEST =====
        if valid_auc > best_auc:
            best_auc = valid_auc
            save_path = os.path.join(rundir, f"best_auc_{valid_auc:.4f}.pth")
            torch.save(model.state_dict(), save_path)
            print("💾 Save BEST model")

        print(f"⏱ Time: {time.time() - start:.2f}s")

    print("\n✅ DONE")
    print(f"🔥 Best Val AUC: {best_auc:.4f}")
