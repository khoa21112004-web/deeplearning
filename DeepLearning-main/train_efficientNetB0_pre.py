import argparse
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import config as base_config
from dataset.dataset import INPUT_DIM
from models import EfficientNetB0
from preprocessing.slice_sampling import uniform_slice_sampling
from utils import _get_lr


def per_image_standardize(img: torch.Tensor) -> torch.Tensor:
    return (img - img.mean()) / (img.std() + 1e-6)


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


class MRDataByIds(Dataset):
    def __init__(
        self,
        ids: List[int],
        labels_map: Dict[int, int],
        data_dir: str = "data",
        transform=None,
        target_slices: int = 32,
        input_dim: int = INPUT_DIM,
    ):
        super().__init__()
        self.planes = ["axial", "coronal", "sagittal"]
        self.ids = list(ids)
        self.labels_map = labels_map
        self.data_dir = data_dir
        self.target_slices = target_slices
        self.input_dim = input_dim
        self.transform = transform

        self.paths = {p: [] for p in self.planes}
        for rid in self.ids:
            rid_str = str(rid).zfill(4)
            src_split = _find_source_split(self.data_dir, rid_str)
            if src_split is None:
                raise FileNotFoundError(f"Missing data for id={rid}")
            for plane in self.planes:
                self.paths[plane].append(os.path.join(self.data_dir, src_split, plane, f"{rid_str}.npy"))

        self.labels = [int(self.labels_map[rid]) for rid in self.ids]
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        self.weights = torch.FloatTensor([neg / pos]) if pos > 0 else torch.FloatTensor([1.0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_raw = {}
        for plane in self.planes:
            volume = np.load(self.paths[plane][index])
            img_raw[plane] = self._prepare_volume(volume)

        label = self.labels[index]
        label = torch.FloatTensor([1]) if label == 1 else torch.FloatTensor([0])
        return [img_raw[plane] for plane in self.planes], label

    def _prepare_volume(self, image: np.ndarray) -> torch.Tensor:
        image = uniform_slice_sampling(image, target_slices=self.target_slices)

        target = self.input_dim
        if target is not None and target <= image.shape[1] and target <= image.shape[2]:
            pad = int((image.shape[2] - target) / 2)
            if pad > 0:
                image = image[:, pad:-pad, pad:-pad]

        image = torch.FloatTensor(image)
        image = torch.stack((image,) * 3, axis=1)  # [S, 3, H, W]

        # FIX: normalize per-slice (nhất quán với pretrained.py normalize từng ảnh 2D)
        # thay vì normalize toàn volume → mất thông tin tương đối giữa các slice
        if self.transform:
            image = torch.stack([self.transform(image[s]) for s in range(image.shape[0])], dim=0)
        return image


def _find_source_split(data_dir: str, rid_str: str):
    for split in ["train", "valid", "test"]:
        ok = True
        for plane in ["axial", "coronal", "sagittal"]:
            path = os.path.join(data_dir, split, plane, f"{rid_str}.npy")
            if not os.path.exists(path):
                ok = False
                break
        if ok:
            return split
    return None


def _read_split_labels(labels_dir: str, task: str, split: str):
    path = os.path.join(labels_dir, f"{split}-{task}.csv")
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


def _load_data_from_ids(
    ids_train,
    ids_val,
    labels_map,
    batch_size,
    num_workers,
    target_slices,
    image_size,
    data_dir,
):
    train_data = MRDataByIds(
        ids_train,
        labels_map,
        data_dir=data_dir,
        transform=per_image_standardize,
        target_slices=target_slices,
        input_dim=image_size,
    )
    labels = train_data.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, sampler=sampler)

    val_data = MRDataByIds(
        ids_val,
        labels_map,
        data_dir=data_dir,
        transform=per_image_standardize,
        target_slices=target_slices,
        input_dim=image_size,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, train_data.weights


def _extract_features_state(pretrained_state) -> Dict[str, torch.Tensor]:
    if isinstance(pretrained_state, dict) and "model_state_dict" in pretrained_state:
        model_state = pretrained_state["model_state_dict"]
    else:
        model_state = pretrained_state

    if not isinstance(model_state, dict):
        raise ValueError("Unsupported checkpoint format.")

    # Case A: features checkpoint saved directly from torchvision model.features
    direct_features_keys = [k for k in model_state.keys() if k and k[0].isdigit()]
    if len(direct_features_keys) > 0:
        return model_state

    # Case B: full model checkpoint with prefix "features."
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


def _get_stateful_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _load_model_state_flexible(model, state_dict: Dict[str, torch.Tensor]) -> None:
    target = _get_stateful_model(model)
    try:
        target.load_state_dict(state_dict)
        return
    except Exception:
        pass

    # Try removing "module." prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k[len("module.") :]: v for k, v in state_dict.items()}
        try:
            target.load_state_dict(stripped)
            return
        except Exception:
            pass

    # Try adding "module." prefix
    added = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(added)


def _best_threshold(y_true, y_prob):
    try:
        fpr, tpr, thr = metrics.roc_curve(y_true, y_prob)
        if len(thr) == 0:
            return 0.5
        idx = int(np.argmax(tpr - fpr))
        return float(thr[idx])
    except Exception:
        return 0.5


def _weighted_bce_prob(prob, target, pos_weight=None, eps=1e-7):
    prob = prob.clamp(eps, 1.0 - eps)
    if pos_weight is None:
        return F.binary_cross_entropy(prob, target)
    return (-(pos_weight * target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))).mean()


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
            images = [img.to(device) for img in images]
            label = label.to(device)

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
    parser.add_argument("--labels-dir", type=str, default="labels")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--task", type=str, default=None, help="abnormal / acl / meniscus")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pretrained-dir", type=str, default="model_pretrained")
    parser.add_argument("--pretrained-features", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8, help="Keep same as pretraining by default")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    args = parser.parse_args()

    cfg = dict(base_config)
    if args.task:
        cfg["task"] = args.task
    cfg["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["grad_accum_steps"] = max(1, int(args.grad_accum_steps))
    task_name = cfg.get("task", "acl")
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))

    if args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ids, train_map = _read_split_labels(args.labels_dir, task_name, "train")
    val_ids, val_map = _read_split_labels(args.labels_dir, task_name, "valid")
    labels_map = {**train_map, **val_map}

    train_loader, val_loader, train_wts = _load_data_from_ids(
        train_ids,
        val_ids,
        labels_map,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        target_slices=cfg["target_slices"],
        image_size=cfg["image_size"],
        data_dir=args.data_dir,
    )

    model = EfficientNetB0(use_imagenet_pretrained=False)
    if args.pretrained_features is not None:
        pretrained_path = args.pretrained_features
    else:
        pretrained_path = os.path.join(args.pretrained_dir, "efficientnetb0_scratch_features.pth")
        if not os.path.exists(pretrained_path):
            pretrained_path = os.path.join(args.pretrained_dir, "efficientnetb0_scratch_best.pth")
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found. Checked: {args.pretrained_features} and {args.pretrained_dir}"
        )
    _load_pretrained_backbone(model, pretrained_path, device="cpu")

    if device != "cpu":
        model = model.to(device)
        train_wts = train_wts.to(device)

    use_dataparallel = False
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        use_dataparallel = True

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_wts)
    if device != "cpu":
        criterion = criterion.to(device)
        val_criterion = val_criterion.to(device)

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

    save_folder = os.path.join("weights", task_name)
    os.makedirs(save_folder, exist_ok=True)
    best_model_path = os.path.join(save_folder, "efficientnetb0_pre_best_model.pth")
    last_model_path = os.path.join(save_folder, "efficientnetb0_pre_last_checkpoint.pth")

    start_epoch = cfg["starting_epoch"]
    best_val_auc = 0.0
    if args.resume and os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        _load_model_state_flexible(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_auc = checkpoint.get("best_val_auc", 0.0)
        print(f"[RESUME] start_epoch={start_epoch}, best_val_auc={best_val_auc:.4f}")

    print("Training Configuration")
    print(cfg)
    print(
        f"Task: {task_name} | Device: {device} | Batch size: {cfg['batch_size']} | "
        f"GradAccum: {grad_accum_steps} | Effective BS: {cfg['batch_size'] * grad_accum_steps}"
    )
    if use_dataparallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    print(f"Pretrained loaded from: {pretrained_path}")

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
            log_interval=cfg.get("log_train", 0),
            epoch_idx=epoch,
            max_epoch=cfg["max_epoch"],
            phase="train",
            grad_accum_steps=grad_accum_steps,
        )
        val_loss, val_auc, val_acc, best_thr = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=val_criterion,
            optimizer=None,
            device=device,
            threshold=threshold,
            auto_threshold=True,
            scaler=scaler,
            scheduler=None,
            log_interval=cfg.get("log_val", 0),
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

        improved = val_auc > best_val_auc
        model_state = _get_stateful_model(model).state_dict()
        state = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_auc": max(best_val_auc, val_auc),
            "task": task_name,
            "pretrained_path": pretrained_path,
            "batch_size": cfg["batch_size"],
            "grad_accum_steps": grad_accum_steps,
        }
        torch.save(state, last_model_path)

        if improved:
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


if __name__ == "__main__":
    main()
