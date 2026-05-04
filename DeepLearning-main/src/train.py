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
    num_workers=4, use_amp=False
):
    # ================= LOAD DATA =================
    train_loader, valid_loader = load_data(
        task, use_gpu, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    # ================= MODEL =================
    model = TripleMRNet(backbone=backbone)

    # 🔥 Freeze backbone nhưng mở layer cuối
    if hasattr(model, 'features'):
        for param in model.features.parameters():
            param.requires_grad = False

        # mở layer cuối để fine-tune
        try:
            for param in model.features[-1].parameters():
                param.requires_grad = True
        except:
            pass

    if use_gpu:
        model = model.cuda()

    # ================= OPTIMIZER =================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 🔥 Scheduler ổn định hơn
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=2, factor=0.3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(use_gpu and use_amp))

    # ================= CHECKPOINT =================
    checkpoint_dir = Path(rundir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_auc = 0

    latest_ckpt = checkpoint_dir / "last_checkpoint.pth"
    if latest_ckpt.exists():
        print("🔄 Resume training from checkpoint...")
        ckpt = torch.load(latest_ckpt)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_val_auc = ckpt["best_val_auc"]

    # ================= EARLY STOP =================
    patience = 3
    patience_counter = 0

    start_time = datetime.now()

    # ================= TRAIN LOOP =================
    for epoch in range(start_epoch, epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs} | Time: {datetime.now() - start_time}")

        # ===== TRAIN =====
        train_loss, train_auc, _, _ = run_model(
            model,
            train_loader,
            train=True,
            optimizer=optimizer,
            abnormal_model_path=abnormal_model_path,
            use_amp=(use_gpu and use_amp),
            scaler=scaler
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train AUC: {train_auc:.4f}")

        if use_gpu:
            torch.cuda.empty_cache()

        # ===== VALID =====
        val_loss, val_auc, _, _ = run_model(
            model,
            valid_loader,
            abnormal_model_path=abnormal_model_path,
            use_amp=(use_gpu and use_amp)
        )

        print(f"Valid Loss: {val_loss:.4f}")
        print(f"Valid AUC: {val_auc:.4f}")

        # 🔥 Log learning rate
        print(f"LR: {optimizer.param_groups[0]['lr']}")

        # ===== SCHEDULER =====
        scheduler.step(val_auc)

        # ===== SAVE BEST =====
        if val_auc > best_val_auc:
            print("💾 Save BEST model")
            best_val_auc = val_auc
            patience_counter = 0

            torch.save(
                model.state_dict(),
                Path(rundir) / "best_model.pth"
            )
        else:
            patience_counter += 1

        # ===== SAVE CHECKPOINT =====
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_auc": best_val_auc,
        }, latest_ckpt)

        # ===== EARLY STOP =====
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default="data")
    parser.add_argument('--labels-dir', type=str, default=None)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--backbone', default="alexnet", type=str)
    parser.add_argument('--abnormal_model', default=None, type=str)
    parser.add_argument('--num_workers', type=int, default=4)
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
        abnormal_model_path=args.abnormal_model,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        num_workers=args.num_workers,
        use_amp=args.amp,
    )
