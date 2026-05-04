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
    abnormal_model_path=None, data_dir="data", labels_dir=None,
    num_workers=4, use_amp=False, checkpoint_every=1
):
    train_loader, valid_loader = load_data(
        task, use_gpu, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    model = TripleMRNet(backbone=backbone)

    # ===== LOAD CHECKPOINT (FIX PYTORCH 2.6) =====
    max_epoch = 0
    model_path = None
    for dirpath, _, files in os.walk(rundir):
        for fname in files:
            if "epoch" in fname:
                try:
                    ep = int(fname.rsplit("epoch", 1)[1])
                    if ep > max_epoch:
                        max_epoch = ep
                        model_path = os.path.join(dirpath, fname)
                except:
                    continue

    if model_path:
        print("🔁 Resume training from checkpoint:", model_path)
        state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.3
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(use_gpu and use_amp))

    # ===== EARLY STOPPING =====
    best_val_auc = 0.0
    patience = 5
    counter = 0

    checkpoint_dir = Path(rundir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    epoch = max_epoch

    while epoch < epochs:
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        train_loss, train_auc, _, _ = run_model(
            model, train_loader, train=True,
            optimizer=optimizer,
            abnormal_model_path=abnormal_model_path,
            use_amp=(use_gpu and use_amp),
            scaler=scaler
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train AUC: {train_auc:.4f}")

        val_loss, val_auc, _, _ = run_model(
            model, valid_loader,
            abnormal_model_path=abnormal_model_path,
            use_amp=(use_gpu and use_amp)
        )

        print(f"Valid Loss: {val_loss:.4f}")
        print(f"Valid AUC: {val_auc:.4f}")

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
            print(f"⚠ No improvement ({counter}/{patience})")

        # ===== EARLY STOP =====
        if counter >= patience:
            print("🛑 Early stopping triggered")
            break

        epoch += 1


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
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--backbone', default="efficientnet_b0", type=str)
    parser.add_argument('--abnormal_model', default=None, type=str)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.rundir, exist_ok=True)

    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    train(
        rundir=args.rundir,
        task=args.task,
        backbone=args.backbone,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_gpu=args.gpu,
        abnormal_model_path=args.abnormal_model,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        num_workers=args.num_workers,
        use_amp=args.amp,
    )
