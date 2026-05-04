import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path

from src.evaluate import run_model
from src.loader import load_data
from src.model import TripleMRNet


# ================= FOCAL LOSS =================
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def train(
    rundir, task, backbone, epochs, learning_rate, weight_decay, use_gpu,
    abnormal_model_path=None, data_dir="data", labels_dir=None,
    num_workers=4, use_amp=False
):
    # ================= LOAD DATA =================
    train_loader, valid_loader = load_data(
        task, use_gpu, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    # ================= MODEL =================
    model = TripleMRNet(backbone=backbone)

    # 🔥 Freeze backbone ban đầu
    for name, param in model.named_parameters():
        param.requires_grad = ("classifier" in name)

    if use_gpu:
        model = model.cuda()

    # ================= LOSS =================
    criterion = FocalLoss()

    # ================= OPTIMIZER (2 LR) =================
    def make_optimizer(unfreeze=False):
        head = [p for n, p in model.named_parameters() if "classifier" in n]
        body = [p for n, p in model.named_parameters() if "classifier" not in n]

        if not unfreeze:
            return torch.optim.AdamW(head, lr=1e-4, weight_decay=weight_decay)

        return torch.optim.AdamW([
            {"params": head, "lr": 1e-4},
            {"params": body, "lr": 1e-6}
        ], weight_decay=weight_decay)

    optimizer = make_optimizer(unfreeze=False)

    # ================= SCHEDULER =================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.3
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(use_gpu and use_amp))

    # ================= LOG FILE =================
    log_path = Path(rundir) / "train_log.csv"
    log_file = open(log_path, "w")
    log_file.write("epoch,train_loss,train_auc,val_loss,val_auc\n")

    # ================= CHECKPOINT =================
    checkpoint_dir = Path(rundir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0
    patience = 4
    counter = 0
    backbone_unfrozen = False

    start_time = datetime.now()

    # ================= TRAIN LOOP =================
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs} | Time: {datetime.now() - start_time}")

        # 🔥 UNFREEZE SAU 4 EPOCH
        if epoch == 4 and not backbone_unfrozen:
            print("🔥 Unfreezing backbone...")
            backbone_unfrozen = True
            optimizer = make_optimizer(unfreeze=True)

        # ===== TRAIN =====
        train_loss, train_auc, _, _ = run_model(
            model,
            train_loader,
            train=True,
            optimizer=optimizer,
            external_criterion=criterion,
            use_amp=(use_gpu and use_amp),
            scaler=scaler,
            grad_clip=1.0
        )

        # ===== VALID =====
        val_loss, val_auc, _, _ = run_model(
            model,
            valid_loader,
            external_criterion=criterion,
            use_amp=(use_gpu and use_amp)
        )

        # ===== PRINT =====
        print(f"[Epoch {epoch+1}]")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid AUC: {val_auc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ===== LOG =====
        log_file.write(f"{epoch+1},{train_loss:.4f},{train_auc:.4f},{val_loss:.4f},{val_auc:.4f}\n")
        log_file.flush()

        # ===== SCHEDULER =====
        scheduler.step(val_auc)

        # ===== SAVE BEST =====
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_dir / "best_model.pth"
            )
            print("💾 Save BEST model")

        else:
            counter += 1
            print(f"⏳ No improve ({counter}/{patience})")

        # ===== EARLY STOP =====
        if counter >= patience:
            print("⛔ Early stopping triggered")
            break

    log_file.close()
    print(f"\n✅ Done. Best Val AUC: {best_val_auc:.4f}")
