import argparse
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_image_standardize(img: torch.Tensor) -> torch.Tensor:
    # img shape: [C, H, W]
    return (img - img.mean()) / (img.std() + 1e-6)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Lambda(per_image_standardize),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(per_image_standardize),
        ]
    )
    return train_tf, val_tf


def build_model(num_classes: int) -> nn.Module:
    # Train from scratch: do NOT load ImageNet weights
    try:
        model = models.efficientnet_b0(weights=None)
    except Exception:
        model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return float((pred == target).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    accs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits, labels))

    if len(losses) == 0:
        return 0.0, 0.0
    return float(np.mean(losses)), float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data_pretrained")
    parser.add_argument("--output-dir", type=str, default="weights_pretrained")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf, val_tf = build_transforms(args.image_size)
    full_ds_train_tf = datasets.ImageFolder(root=args.data_dir, transform=train_tf)
    full_ds_val_tf = datasets.ImageFolder(root=args.data_dir, transform=val_tf)

    targets = np.array(full_ds_train_tf.targets)
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=targets,
    )

    train_set = Subset(full_ds_train_tf, train_idx.tolist())
    val_set = Subset(full_ds_val_tf, val_idx.tolist())

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    num_classes = len(full_ds_train_tf.classes)
    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "efficientnetb0_scratch_best.pth")
    last_ckpt_path = os.path.join(args.output_dir, "efficientnetb0_scratch_last.pth")
    features_ckpt_path = os.path.join(args.output_dir, "efficientnetb0_scratch_features.pth")

    print(f"Device: {device}")
    print(f"Classes: {full_ds_train_tf.classes}")
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")
    print("Training EfficientNetB0 from scratch (weights=None)...")

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        print(
            "Epoch [{}/{}] | train loss {:.4f} acc {:.4f} | val loss {:.4f} acc {:.4f} | lr {:.6f}".format(
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                optimizer.param_groups[0]["lr"],
            )
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": full_ds_train_tf.classes,
            "class_to_idx": full_ds_train_tf.class_to_idx,
            "val_acc": val_acc,
            "val_loss": val_loss,
        }
        torch.save(state, last_ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(state, best_ckpt_path)
            torch.save(model.features.state_dict(), features_ckpt_path)
            print(f"  New best val acc: {best_val_acc:.4f} -> saved")

    elapsed = time.time() - started
    print(f"Done. Total time: {elapsed:.2f}s")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Backbone features checkpoint: {features_ckpt_path}")


if __name__ == "__main__":
    main()
