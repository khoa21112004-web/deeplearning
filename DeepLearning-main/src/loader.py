import torch
import torch.utils.data as data
import numpy as np
import os

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


def _normalize_id(raw_id):
    base = os.path.splitext(os.path.basename(str(raw_id).strip()))[0]
    if base.isdigit():
        return str(int(base))
    return base


class Dataset(data.Dataset):
    def __init__(self, datadir, tear_type, use_gpu, labels_dir=None, augment=False):
        super().__init__()

        self.datadir = datadir.rstrip("/")
        self.use_gpu = use_gpu
        self.augment = augment

        label_root = labels_dir if labels_dir is not None else datadir

        # ===== LOAD LABEL =====
        label_dict = {}
        abnormal_dict = {}

        with open(label_root + '-' + tear_type + '.csv') as f:
            for line in f:
                pid, label = line.strip().split(',')
                label_dict[_normalize_id(pid)] = int(label)

        with open(label_root + '-abnormal.csv') as f:
            for line in f:
                pid, label = line.strip().split(',')
                abnormal_dict[_normalize_id(pid)] = int(label)

        # ===== LOAD PATHS =====
        self.paths = []
        self.labels = []
        self.abnormal = []

        axial_dir = os.path.join(self.datadir, "axial")

        for fname in os.listdir(axial_dir):
            if not fname.endswith(".npy"):
                continue

            pid = _normalize_id(fname)

            if pid in label_dict and pid in abnormal_dict:
                self.paths.append(fname)
                self.labels.append(label_dict[pid])
                self.abnormal.append(abnormal_dict[pid])

        self.labels = np.array(self.labels)

        if len(self.paths) == 0:
            raise ValueError("❌ No data found. Check dataset path!")

    def __getitem__(self, idx):
        fname = self.paths[idx]

        # ===== LOAD 3 VIEW =====
        vol_axial = np.load(os.path.join(self.datadir, "axial", fname))
        vol_sagit = np.load(os.path.join(self.datadir, "sagittal", fname))
        vol_coron = np.load(os.path.join(self.datadir, "coronal", fname))

        # ===== PREPROCESS =====
        def process(vol):
            pad = int((vol.shape[2] - INPUT_DIM) / 2)
            vol = vol[:, pad:-pad, pad:-pad]

            vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-6) * MAX_PIXEL_VAL
            vol = (vol - MEAN) / STDDEV

            vol = np.stack((vol,) * 3, axis=1)
            return vol

        vol_axial = process(vol_axial)
        vol_sagit = process(vol_sagit)
        vol_coron = process(vol_coron)

        # ===== AUGMENT =====
        if self.augment:
            if np.random.rand() < 0.5:
                vol_axial = vol_axial[:, :, ::-1, :]
            if np.random.rand() < 0.3:
                vol_sagit = vol_sagit + np.random.normal(0, 0.01, vol_sagit.shape)

        # ===== TO TENSOR =====
        vol_axial = torch.FloatTensor(vol_axial)
        vol_sagit = torch.FloatTensor(vol_sagit)
        vol_coron = torch.FloatTensor(vol_coron)

        label = torch.FloatTensor([self.labels[idx]])
        abnormal = self.abnormal[idx]

        return vol_axial, vol_sagit, vol_coron, label, abnormal

    def __len__(self):
        return len(self.paths)


def load_data(task="abnormal", use_gpu=False, data_dir="data", labels_dir=None, num_workers=2):

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    labels_train = os.path.join(labels_dir, "train") if labels_dir else None
    labels_valid = os.path.join(labels_dir, "valid") if labels_dir else None

    train_dataset = Dataset(train_dir, task, use_gpu, labels_train, augment=True)
    valid_dataset = Dataset(valid_dir, task, use_gpu, labels_valid, augment=False)

    # ===== BALANCE DATA =====
    labels = train_dataset.labels
    class_count = np.bincount(labels)
    weights = 1. / class_count
    sample_weights = weights[labels]

    sampler = data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader
