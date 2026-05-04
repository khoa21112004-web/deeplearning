import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path

from evaluate import run_model
from loader import load_data
from model import TripleMRNet


def train(
    rundir, task, backbone, epochs, learning_rate, weight_decay, use_gpu,
    data_dir="data", labels_dir=None,
    num_workers=2, use_amp=False
):
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

    print(f"🚀 Using device: {device}")

    # ===== LOAD DATA =====
    train_loader, valid_loader = load_data(
        task, use_gpu,
        data_dir=data_dir,
        labels_dir=labels_dir,
        num_workers=num_workers
    )

    print(f"Loaded {len(train_loader.dataset)} train samples")
    print(f"Loaded {len(valid_loader.dataset)} valid samples")

    # ===== MODEL =====
    model = TripleMRNet(backbone=backbone).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.3
    )

    scaler = torch.amp.GradScaler(enabled=(device == "cuda" and use_amp))

    # ===== TRAIN CONFIG =====
    best_val_auc = 0.0
    patience = 4
    counter = 0

    checkpoint_dir = Path(rundir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ===== TRAIN LOOP =====
    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        # ===== TRAIN =====
        train_loss, train_auc, _, _ = run_model(
            model,
            train_loader,
            train=True,
            optimizer=optimizer,
            use_amp=(device == "cuda" and use_amp),
            scaler=scaler
        )

        print(f"[Epoch {epoch+1}]")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

        # ===== VALID =====
        val_loss, val_auc, _, _ = run_model(
            model,
            valid_loader,
            use_amp=(device == "cuda" and use_amp)
        )

        print(f"Valid Loss: {val_loss:.4f} | Valid AUC: {val_auc:.4f}")

        scheduler.step(val_loss)

        # ===== SAVE BEST =====
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0

            save_path = checkpoint_dir / f"best_epoch{epoch+1}_auc{val_auc:.4f}.pth"
            torch.save(model.state_dict(), save_path)
            print("💾 Save BEST model")

        else:
            counter += 1
            print(f"⏳ No improve ({counter}/{patience})")

        # ===== EARLY STOP =====
        if counter >= patience:
            print("🛑 Early stopping")
            break

    print(f"\n✅ Done. Best Val AUC: {best_val_auc:.4f}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--labels-dir', type=str, default=None)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--backbone', default="efficientnet_b0", type=str)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)

    with open(Path(args.rundir) / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

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
    )
