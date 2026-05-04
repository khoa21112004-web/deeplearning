import numpy as np
import os
import torch
import torch.utils.data as data

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73
TARGET_SLICES = 32


def _normalize_id(raw_id):
    base = os.path.splitext(os.path.basename(str(raw_id).strip()))[0]
    return str(int(base)) if base.isdigit() else base


def resize_slices(vol, target=TARGET_SLICES):
    current = vol.shape[0]

    if current > target:
        start = (current - target) // 2
        vol = vol[start:start + target]

    elif current < target:
        pad = target - current
        vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)), mode='constant')

    return vol


# ================== PREPROCESS ==================
def preprocess(vol):
    vol = vol.astype(np.float32)

    # center crop
    pad = int((vol.shape[2] - INPUT_DIM) / 2)
    vol = vol[:, pad:-pad, pad:-pad]

    # normalize
    vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-6)
    vol = vol * MAX_PIXEL_VAL
    vol = (vol - MEAN) / STDDEV

    # 🔥 FIX QUAN TRỌNG: lấy 3 slice thay vì 1
    mid = vol.shape[0] // 2
    vol = vol[mid-1:mid+2]   # (3, H, W)

    return torch.from_numpy(vol).float()


# ================== DATASET ==================
class Dataset(data.Dataset):
    def __init__(self, datadir, tear_type, use_gpu, labels_dir=None, augment=False):
        self.use_gpu = use_gpu
        self.augment = augment
        self.datadir = datadir.rstrip('/')

        label_root = labels_dir if labels_dir else datadir

        label_dict = {}
        abnormal_dict = {}

        # ===== LOAD LABEL =====
        with open(label_root + '-' + tear_type + '.csv') as f:
            for line in f:
                fid, lab = line.strip().split(',')
                label_dict[_normalize_id(fid)] = int(lab)

        with open(label_root + '-abnormal.csv') as f:
            for line in f:
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
                vol_sagit += np.random.normal(0, 0.01, vol_sagit.shape).astype(np.float32)

        vol_axial = preprocess(vol_axial)
        vol_sagit = preprocess(vol_sagit)
        vol_coron = preprocess(vol_coron)

        # ✅ FIX LABEL SHAPE CHUẨN
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return vol_axial, vol_sagit, vol_coron, label, self.abnormal_labels[idx]

    def __len__(self):
        return len(self.paths)


# ================== LOAD DATA ==================
def load_data(task="acl", use_gpu=False, data_dir="data", labels_dir=None, num_workers=2):

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
        batch_size=1,
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
