import argparse, json, os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from evaluate import run_model
from loader import load_data
from model import TripleMRNet


def train(
    rundir, task, backbone, epochs, learning_rate, weight_decay, use_gpu,
    data_dir="data", labels_dir=None, num_workers=2, use_amp=True,
    patience=7
):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    train_loader, valid_loader = load_data(
        task, use_gpu, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    model = TripleMRNet(backbone=backbone).to(device)

    # ===== class imbalance (pos_weight) =====
    # ước lượng nhanh từ train_loader
    pos = 0
    neg = 0
    for _, _, _, y, _ in train_loader:
        y = y.view(-1)
        pos += (y == 1).sum().item()
        neg += (y == 0).sum().item()
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Cosine schedule (mượt hơn)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))

    best_auc = 0.0
    wait = 0

    out_dir = Path(rundir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        # ===== TRAIN =====
        model.train()
        train_loss, train_auc, _, _ = run_model(
            model, train_loader, train=True, optimizer=optimizer,
            use_amp=(device.type == "cuda" and use_amp),
            scaler=scaler,
            external_criterion=criterion,
            grad_clip=1.0
        )
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

        # ===== VALID =====
        model.eval()
        val_loss, val_auc, _, _ = run_model(
            model, valid_loader,
            use_amp=(device.type == "cuda" and use_amp),
            external_criterion=criterion
        )
        print(f"Valid Loss: {val_loss:.4f} | Valid AUC: {val_auc:.4f}")

        scheduler.step()

        # ===== SAVE BEST (theo AUC) =====
        if val_auc > best_auc:
            best_auc = val_auc
            wait = 0
            save_path = out_dir / f"best_epoch{epoch+1}_auc{val_auc:.4f}.pth"
            torch.save(model.state_dict(), save_path)
            print("💾 Save BEST model")
        else:
            wait += 1
            print(f"⚠ No improve ({wait}/{patience})")

        # ===== EARLY STOP =====
        if wait >= patience:
            print("🛑 Early stopping triggered")
            break


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--rundir', type=str, required=True)
    p.add_argument('--task', type=str, required=True)
    p.add_argument('--data-dir', type=str, default="data")
    p.add_argument('--labels-dir', type=str, default=None)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--learning_rate', default=3e-5, type=float)
    p.add_argument('--weight_decay', default=1e-5, type=float)
    p.add_argument('--epochs', default=30, type=int)
    p.add_argument('--backbone', default="efficientnet_b0", type=str)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--patience', default=7, type=int)
    return p


if __name__ == '__main__':
    args = get_parser().parse_args()

    train(
        rundir=args.rundir,
        task=args.task,
        backbone=args.backbone,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_gpu=args.gpu,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        num_workers=args.num_workers,
        use_amp=args.amp,
        patience=args.patience
    )
