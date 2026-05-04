import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models as tv_models

from config import config as base_config
from utils import _get_lr


def per_image_standardize(img: torch.Tensor) -> torch.Tensor:
    return (img - img.mean()) / (img.std() + 1e-6)


def _build_efficientnet_b0(use_imagenet_pretrained: bool = False):
    if use_imagenet_pretrained:
        try:
            return tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            return tv_models.efficientnet_b0(pretrained=True)
    try:
        return tv_models.efficientnet_b0(weights=None)
    except Exception:
        return tv_models.efficientnet_b0(pretrained=False)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.avg_pool(x))
        return x * scale


class EfficientNetB0(nn.Module):
    def __init__(self, use_imagenet_pretrained: bool = False):
        super().__init__()
        self.backbone = _build_efficientnet_b0(use_imagenet_pretrained=use_imagenet_pretrained).features
        self.se = SqueezeExcite(1280, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_norm = nn.LayerNorm(1280)
        self.feat_dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def _encode_plane(self, net, x):
        if x.dim() == 4:
            feat = net(x)
            feat = self.se(feat)
            feat = self.pool(feat).view(feat.size(0), -1)
            feat = self.feat_norm(feat)
            feat = self.feat_dropout(feat)
            feat, _ = torch.max(feat, dim=0, keepdim=True)
            return feat

        if x.dim() != 5:
            raise ValueError(f"Unexpected input shape for plane: {x.shape}")

        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = net(x)
        feat = self.se(feat)
        feat = self.pool(feat).view(feat.size(0), -1)
        feat = self.feat_norm(feat)
        feat = self.feat_dropout(feat)
        feat = feat.view(b, s, -1)
        feat, _ = torch.max(feat, dim=1)
        return feat

    def forward(self, x):
        axial = self._encode_plane(self.backbone, x[0])
        coronal = self._encode_plane(self.backbone, x[1])
        sagittal = self._encode_plane(self.backbone, x[2])
        planes = torch.stack([axial, coronal, sagittal], dim=1)
        feats, _ = torch.max(planes, dim=1)
        output = self.fc(feats)
        return output


def _build_grad_scaler(device: str):
    enabled = device == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(device: str, enabled: bool):
    if not enabled or device == "cpu":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _sample_indices(num_items: int, target: int) -> List[int]:
    if num_items <= 0:
        return []
    if target <= 0:
        return list(range(num_items))
    if num_items == target:
        return list(range(num_items))
    idx = np.linspace(0, num_items - 1, num=target)
    idx = np.clip(np.round(idx).astype(np.int64), 0, num_items - 1)
    return idx.tolist()


def _read_split_labels(labels_dir: str, task: str, split: str):
    path = os.path.join(labels_dir, f"{split}-{task}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing labels file: {path}")
    data = np.genfromtxt(path, delimiter=",", dtype=int)
    if data.ndim == 1 and data.size == 2:
        data = np.array([data])
    ids = []
    labels_map = {}
    for rid, lab in data:
        rid = int(rid)
        labels_map[rid] = int(lab)
        ids.append(rid)
    return ids, labels_map


def _extract_features_state(pretrained_state) -> Dict[str, torch.Tensor]:
    if isinstance(pretrained_state, dict) and "model_state_dict" in pretrained_state:
        model_state = pretrained_state["model_state_dict"]
    else:
        model_state = pretrained_state

    if not isinstance(model_state, dict):
        raise ValueError("Unsupported checkpoint format.")

    direct_features_keys = [k for k in model_state.keys() if k and k[0].isdigit()]
    if len(direct_features_keys) > 0:
        return model_state

    features_state = {}
    for key, val in model_state.items():
        k = key
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("features."):
            features_state[k[len("features.") :]] = val

    if len(features_state) == 0:
        raise ValueError("Could not extract EfficientNet features weights from checkpoint.")
    return features_state


def _load_pretrained_backbone(model: EfficientNetB0, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    features_state = _extract_features_state(ckpt)
    missing, unexpected = model.backbone.load_state_dict(features_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading backbone: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading backbone: {len(unexpected)}")
    print(f"[OK] Loaded pretrained backbone from: {ckpt_path}")


class MRPngDataset(Dataset):
    def __init__(
        self,
        ids: List[int],
        labels_map: Dict[int, int],
        split: str,
        data_dir: str,
        target_slices: int = 32,
        image_size: int = 224,
        transform=None,
    ):
        super().__init__()
        self.planes = ["axial", "coronal", "sagittal"]
        self.ids = list(ids)
        self.labels_map = labels_map
        self.split = split
        self.data_dir = data_dir
        self.target_slices = target_slices
        self.image_size = image_size
        self.transform = transform
        self.resample = Image.BILINEAR if not hasattr(Image, "Resampling") else Image.Resampling.BILINEAR

        self.slice_paths = {p: [] for p in self.planes}
        for rid in self.ids:
            rid_str = str(rid).zfill(4)
            for plane in self.planes:
                plane_dir = os.path.join(self.data_dir, self.split, plane, rid_str)
                if not os.path.isdir(plane_dir):
                    raise FileNotFoundError(f"Missing plane directory: {plane_dir}")
                files = sorted(
                    [
                        os.path.join(plane_dir, fn)
                        for fn in os.listdir(plane_dir)
                        if fn.lower().endswith(".png")
                    ]
                )
                if len(files) == 0:
                    raise FileNotFoundError(f"No PNG files in: {plane_dir}")
                self.slice_paths[plane].append(files)

        self.labels = [int(self.labels_map[rid]) for rid in self.ids]
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([neg / pos]) if pos > 0 else torch.FloatTensor([1.0])

    def __len__(self):
        return len(self.ids)

    def _center_crop_or_resize(self, img: Image.Image) -> Image.Image:
        target = self.image_size
        if target is None or target <= 0:
            return img
        w, h = img.size
        if h >= target and w >= target:
            left = (w - target) // 2
            top = (h - target) // 2
            return img.crop((left, top, left + target, top + target))
        return img.resize((target, target), resample=self.resample)

    def _load_plane(self, files: List[str]) -> torch.Tensor:
        idxs = _sample_indices(len(files), self.target_slices)
        slices = []
        for i in idxs:
            with Image.open(files[i]) as pil_img:
                pil_img = pil_img.convert("L")
                pil_img = self._center_crop_or_resize(pil_img)
                arr = np.asarray(pil_img, dtype=np.float32)

            img = torch.from_numpy(arr)  # [H, W]
            img = torch.stack((img, img, img), dim=0)  # [3, H, W]
            if self.transform:
                img = self.transform(img)
            slices.append(img)
        return torch.stack(slices, dim=0)  # [S, 3, H, W]

    def __getitem__(self, index):
        images = [self._load_plane(self.slice_paths[plane][index]) for plane in self.planes]
        label = torch.FloatTensor([1.0 if self.labels[index] == 1 else 0.0])
        return images, label


def _build_loaders(
    data_dir: str,
    labels_dir: str,
    task_name: str,
    batch_size: int,
    num_workers: int,
    target_slices: int,
    image_size: int,
):
    train_ids, train_map = _read_split_labels(labels_dir, task_name, "train")
    val_ids, val_map = _read_split_labels(labels_dir, task_name, "valid")
    test_ids, test_map = _read_split_labels(labels_dir, task_name, "test")

    train_ds = MRPngDataset(
        ids=train_ids,
        labels_map=train_map,
        split="train",
        data_dir=data_dir,
        target_slices=target_slices,
        image_size=image_size,
        transform=per_image_standardize,
    )
    val_ds = MRPngDataset(
        ids=val_ids,
        labels_map=val_map,
        split="valid",
        data_dir=data_dir,
        target_slices=target_slices,
        image_size=image_size,
        transform=per_image_standardize,
    )
    test_ds = MRPngDataset(
        ids=test_ids,
        labels_map=test_map,
        split="test",
        data_dir=data_dir,
        target_slices=target_slices,
        image_size=image_size,
        transform=per_image_standardize,
    )

    labels = np.asarray(train_ds.labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.weights


def _best_threshold(y_true, y_prob):
    try:
        fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
        if len(thr) == 0:
            return 0.5
        idx = int(np.argmax(tpr - fpr))
        return float(thr[idx])
    except Exception:
        return 0.5


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    threshold=0.5,
    auto_threshold=False,
    scaler=None,
    scheduler=None,
    log_interval=0,
    epoch_idx=None,
    max_epoch=None,
    phase="train",
    grad_accum_steps=1,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    y_true = []
    y_prob = []
    losses = []

    grad_accum_steps = max(1, int(grad_accum_steps))
    num_steps = len(loader)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(loader, 1):
        images, label = batch

        if device != "cpu":
            images = [img.to(device, non_blocking=True) for img in images]
            label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            step_was_skipped = False
            should_step = is_train and (step_idx % grad_accum_steps == 0 or step_idx == num_steps)
            if scaler is not None and device != "cpu":
                with _autocast_ctx(device=device, enabled=True):
                    output = model(images)
                    loss = criterion(output, label) / grad_accum_steps
                if is_train:
                    scale_before = scaler.get_scale()
                    scaler.scale(loss).backward()
                    if should_step:
                        scaler.step(optimizer)
                        scaler.update()
                        scale_after = scaler.get_scale()
                        step_was_skipped = scale_after < scale_before
                        optimizer.zero_grad(set_to_none=True)
            else:
                output = model(images)
                loss = criterion(output, label) / grad_accum_steps
                if is_train:
                    loss.backward()
                    if should_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            if is_train and should_step and scheduler is not None and not step_was_skipped:
                scheduler.step()

        losses.append(loss.item() * grad_accum_steps)
        probas = torch.sigmoid(output).detach().cpu().numpy()
        labels = label.detach().cpu().numpy()
        y_prob.extend(probas.tolist())
        y_true.extend(labels.tolist())

        if log_interval and (step_idx % log_interval == 0 or step_idx == num_steps):
            running_loss = float(np.mean(losses))
            if epoch_idx is not None and max_epoch is not None:
                prefix = f"[{phase.upper()}] Epoch [{epoch_idx + 1}/{max_epoch}] Step [{step_idx}/{num_steps}]"
            else:
                prefix = f"[{phase.upper()}] Step [{step_idx}/{num_steps}]"
            if is_train:
                print(f"{prefix} | loss {running_loss:.4f} | lr {_get_lr(optimizer):.6f}")
            else:
                print(f"{prefix} | loss {running_loss:.4f}")

    if len(losses) == 0:
        return 0.0, 0.5, 0.0, float(threshold)

    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    try:
        auc = metrics.roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5

    if auto_threshold:
        threshold = _best_threshold(y_true, y_prob)

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    acc = metrics.accuracy_score(y_true, y_pred)
    loss_mean = float(np.mean(losses))
    return loss_mean, float(auc), float(acc), float(threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="abnormal / acl / meniscus")
    parser.add_argument("--data-dir", type=str, default=r"F:\NgDuyLinh\datasetDeep\data_png")
    parser.add_argument("--labels-dir", type=str, default=r"F:\NgDuyLinh\datasetDeep\labels")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--target-slices", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-dir", type=str, default=None)
    parser.add_argument("--pretrained-features", type=str, default=None)
    parser.add_argument("--use-imagenet-pretrained", dest="use_imagenet_pretrained", action="store_true")
    parser.add_argument("--no-imagenet-pretrained", dest="use_imagenet_pretrained", action="store_false")
    parser.set_defaults(use_imagenet_pretrained=False)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = dict(base_config)
    if args.task is not None:
        cfg["task"] = args.task
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.max_epoch is not None:
        cfg["max_epoch"] = args.max_epoch
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay
    if args.target_slices is not None:
        cfg["target_slices"] = args.target_slices
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.grad_accum_steps is not None:
        cfg["grad_accum_steps"] = max(1, int(args.grad_accum_steps))

    task_name = cfg.get("task", "abnormal")
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))

    if args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, train_wts = _build_loaders(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        task_name=task_name,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        target_slices=cfg["target_slices"],
        image_size=cfg["image_size"],
    )

    model = EfficientNetB0(use_imagenet_pretrained=bool(args.use_imagenet_pretrained))
    pretrained_features_path = None
    if args.pretrained_features is not None:
        pretrained_features_path = args.pretrained_features
    elif args.pretrained_dir is not None:
        p1 = os.path.join(args.pretrained_dir, "efficientnetb0_scratch_features.pth")
        p2 = os.path.join(args.pretrained_dir, "efficientnetb0_scratch_best.pth")
        if os.path.exists(p1):
            pretrained_features_path = p1
        elif os.path.exists(p2):
            pretrained_features_path = p2
    if pretrained_features_path is not None:
        if not os.path.exists(pretrained_features_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_features_path}")
        _load_pretrained_backbone(model, pretrained_features_path, device="cpu")

    if device != "cpu":
        model = model.to(device)
        train_wts = train_wts.to(device)

    use_dataparallel = False
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        use_dataparallel = True

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    if device != "cpu":
        criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        epochs=cfg["max_epoch"],
        steps_per_epoch=max(1, math.ceil(len(train_loader) / grad_accum_steps)),
        pct_start=0.3,
        anneal_strategy="cos",
    )
    scaler = _build_grad_scaler(device)

    save_folder = os.path.join("weights_png", task_name)
    os.makedirs(save_folder, exist_ok=True)
    best_model_path = os.path.join(save_folder, "efficientnetb0_png_best.pth")
    last_model_path = os.path.join(save_folder, "efficientnetb0_png_last.pth")

    start_epoch = cfg["starting_epoch"]
    best_val_auc = 0.0
    if args.resume and os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", start_epoch) + 1
        best_val_auc = checkpoint.get("best_val_auc", best_val_auc)
        print(f"[RESUME] start_epoch={start_epoch}, best_val_auc={best_val_auc:.4f}")

    print("Training Configuration")
    print(cfg)
    print(
        f"Task: {task_name} | Device: {device} | Batch size: {cfg['batch_size']} | "
        f"GradAccum: {grad_accum_steps} | Effective BS: {cfg['batch_size'] * grad_accum_steps}"
    )
    print(f"Data dir: {args.data_dir}")
    print(f"Labels dir: {args.labels_dir}")
    print(f"ImageNet pretrained: {bool(args.use_imagenet_pretrained)}")
    if pretrained_features_path is not None:
        print(f"Backbone pretrained from checkpoint: {pretrained_features_path}")
    if use_dataparallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

    patience = cfg.get("patience", 5)
    epochs_no_improve = 0
    threshold = 0.5

    t0 = time.time()
    for epoch in range(start_epoch, cfg["max_epoch"]):
        lr_now = _get_lr(optimizer)
        e0 = time.time()

        train_loss, train_auc, train_acc, _ = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=threshold,
            scaler=scaler,
            scheduler=scheduler,
            log_interval=cfg.get("log_train", 100),
            epoch_idx=epoch,
            max_epoch=cfg["max_epoch"],
            phase="train",
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_auc, val_acc, best_thr = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            threshold=threshold,
            auto_threshold=True,
            scaler=scaler,
            scheduler=None,
            log_interval=cfg.get("log_val", 10),
            epoch_idx=epoch,
            max_epoch=cfg["max_epoch"],
            phase="val",
            grad_accum_steps=1,
        )

        print(
            "Epoch [{}/{}] | train loss {:.4f} auc {:.4f} acc {:.4f} | "
            "val loss {:.4f} auc {:.4f} acc {:.4f} | thr {:.4f} | lr {:.6f} | time {:.2f}s".format(
                epoch + 1,
                cfg["max_epoch"],
                train_loss,
                train_auc,
                train_acc,
                val_loss,
                val_auc,
                val_acc,
                best_thr,
                lr_now,
                time.time() - e0,
            )
        )

        model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        state = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_auc": max(best_val_auc, val_auc),
            "task": task_name,
            "use_imagenet_pretrained": bool(args.use_imagenet_pretrained),
            "pretrained_features_path": pretrained_features_path,
            "batch_size": cfg["batch_size"],
            "grad_accum_steps": grad_accum_steps,
            "target_slices": cfg["target_slices"],
            "image_size": cfg["image_size"],
        }
        torch.save(state, last_model_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(state, best_model_path)
            print(f"[SAVE] New best AUC={best_val_auc:.4f} -> {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs.")
                break

    print(f"Done. Total training time: {time.time() - t0:.2f}s")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Last checkpoint: {last_model_path}")

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        best_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
    test_loss, test_auc, test_acc, _ = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        threshold=0.5,
        auto_threshold=False,
        scaler=scaler,
        scheduler=None,
        log_interval=0,
        phase="test",
        grad_accum_steps=1,
    )
    print("Test | loss {:.4f} auc {:.4f} acc {:.4f}".format(test_loss, test_auc, test_acc))


if __name__ == "__main__":
    main()
