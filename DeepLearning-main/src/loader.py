import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73
TARGET_SLICES = 32


def _normalize_id(raw_id):
    base = os.path.splitext(os.path.basename(str(raw_id).strip()))[0]
    if base.isdigit():
        return str(int(base))
    return base


# ===== FIX SLICE =====
def resize_slices(vol, target=TARGET_SLICES):
    current = vol.shape[0]

    if current > target:
        start = (current - target) // 2
        vol = vol[start:start + target]

    elif current < target:
        pad = target - current
        vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)), mode='constant')

    return vol


# ===== PREPROCESS (FIX QUAN TRỌNG) =====
def preprocess(vol):
    vol = vol.astype(np.float32)

    pad = int((vol.shape[2] - INPUT_DIM) / 2)
    vol = vol[:, pad:-pad, pad:-pad]

    # normalize
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-6)
    vol = vol * MAX_PIXEL_VAL
    vol = (vol - MEAN) / STDDEV

    # 🔥 FIX: dùng nhiều slice
    vol = resize_slices(vol, TARGET_SLICES)

    # convert thành (N, 3, H, W)
    vol = np.stack((vol,) * 3, axis=1)

    return torch.FloatTensor(vol)


class Dataset(data.Dataset):
    def __init__(self, datadir, tear_type, use_gpu, labels_dir=None, augment=False):
        self.use_gpu = use_gpu
        self.augment = augment
        self.datadir = datadir.rstrip('/')

        label_root = labels_dir if labels_dir else datadir

        label_dict = {}
        abnormal_dict = {}

        # ===== LOAD LABEL =====
        for line in open(label_root + '-' + tear_type + '.csv'):
            fid, lab = line.strip().split(',')
            label_dict[_normalize_id(fid)] = int(lab)

        for line in open(label_root + '-abnormal.csv'):
            fid, lab = line.strip().split(',')
            abnormal_dict[_normalize_id(fid)] = int(lab)

        # ===== LOAD FILE =====
        self.paths = []
        for f in os.listdir(os.path.join(datadir, "axial")):
            if f.endswith(".npy"):
                pid = _normalize_id(f)
                if pid in label_dict and pid in abnormal_dict:
                    self.paths.append(f)

        self.paths.sort()

        if len(self.paths) == 0:
            raise ValueError("❌ No valid data found")

        self.labels = [label_dict[_normalize_id(p)] for p in self.paths]
        self.abnormal_labels = [abnormal_dict[_normalize_id(p)] for p in self.paths]

        pos = np.mean(self.labels)
        self.weights = [pos, 1 - pos]

    # ===== WEIGHTED LOSS =====
    def weighted_loss(self, pred, target):
        weights = torch.FloatTensor([self.weights[int(t[0])] for t in target])
        if self.use_gpu:
            weights = weights.cuda()
        return F.binary_cross_entropy_with_logits(pred, target, weight=weights)

    def __getitem__(self, idx):
        fname = self.paths[idx]

        vol_axial = np.load(os.path.join(self.datadir, "axial", fname))
        vol_sagit = np.load(os.path.join(self.datadir, "sagittal", fname))
        vol_coron = np.load(os.path.join(self.datadir, "coronal", fname))

        # ===== AUGMENT =====
        if self.augment:
            if np.random.rand() < 0.5:
                vol_axial = vol_axial[:, :, ::-1]

            if np.random.rand() < 0.3:
                vol_sagit = vol_sagit.astype(np.float32)
                vol_sagit += np.random.normal(0, 0.01, vol_sagit.shape)

            if np.random.rand() < 0.3:
                vol_coron = np.rot90(vol_coron, k=1, axes=(1, 2))

        vol_axial = preprocess(vol_axial)
        vol_sagit = preprocess(vol_sagit)
        vol_coron = preprocess(vol_coron)

        label = torch.FloatTensor([self.labels[idx]])

        return vol_axial, vol_sagit, vol_coron, label, self.abnormal_labels[idx]

    def __len__(self):
        return len(self.paths)


# ===== LOAD DATA =====
def load_data(task="abnormal", use_gpu=False, data_dir="data", labels_dir=None, num_workers=2):

    train_ds = Dataset(
        os.path.join(data_dir, "train"),
        task,
        use_gpu,
        labels_dir=os.path.join(labels_dir, "train") if labels_dir else None,
        augment=True
    )

    valid_ds = Dataset(
        os.path.join(data_dir, "valid"),
        task,
        use_gpu,
        labels_dir=os.path.join(labels_dir, "valid") if labels_dir else None,
        augment=False
    )

    # ===== BALANCE =====
    labels = np.array(train_ds.labels)
    class_count = np.bincount(labels)
    weights = 1. / (class_count + 1e-6)
    sample_weights = weights[labels]

    sampler = data.WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    train_loader = data.DataLoader(
        train_ds,
        batch_size=1,  # 🔥 bắt buộc
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = data.DataLoader(
        valid_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader
