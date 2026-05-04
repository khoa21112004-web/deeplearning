import numpy as np
import os
import torch
import torch.utils.data as data

INPUT_DIM = 224
TARGET_SLICES = 32
MEAN = 58.09
STD = 49.73


def _normalize_id(x):
    base = os.path.splitext(os.path.basename(str(x).strip()))[0]
    return str(int(base)) if base.isdigit() else base


def resize_slices(vol, target=TARGET_SLICES):
    S = vol.shape[0]
    if S > target:
        st = (S - target) // 2
        vol = vol[st:st + target]
    elif S < target:
        pad = target - S
        last = vol[-1:]
        vol = np.concatenate([vol, np.repeat(last, pad, axis=0)], axis=0)
    return vol


def preprocess(vol):
    vol = vol.astype(np.float32)

    # crop center
    pad = (vol.shape[2] - INPUT_DIM) // 2
    vol = vol[:, pad:-pad, pad:-pad]

    # 🔥 FIX: normalize cố định (KHÔNG min-max)
    vol = (vol - MEAN) / STD

    vol = resize_slices(vol)

    # to 3 channel
    vol = np.stack([vol, vol, vol], axis=1)

    return torch.from_numpy(vol).float()


class Dataset(data.Dataset):
    def __init__(self, datadir, task, labels_dir=None):
        self.datadir = datadir.rstrip("/")

        label_root = labels_dir if labels_dir else datadir
        label_dict = {}
        abnormal_dict = {}

        for line in open(label_root + f"-{task}.csv"):
            f, l = line.strip().split(",")
            label_dict[_normalize_id(f)] = int(l)

        for line in open(label_root + "-abnormal.csv"):
            f, l = line.strip().split(",")
            abnormal_dict[_normalize_id(f)] = int(l)

        self.paths = []
        for f in os.listdir(os.path.join(self.datadir, "axial")):
            if f.endswith(".npy"):
                pid = _normalize_id(f)
                if pid in label_dict and pid in abnormal_dict:
                    self.paths.append(f)

        self.paths.sort()
        self.labels = [label_dict[_normalize_id(p)] for p in self.paths]

        print(f"Loaded {len(self.paths)} samples from {self.datadir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        f = self.paths[i]

        ax = np.load(os.path.join(self.datadir, "axial", f))
        sa = np.load(os.path.join(self.datadir, "sagittal", f))
        co = np.load(os.path.join(self.datadir, "coronal", f))

        ax = preprocess(ax)
        sa = preprocess(sa)
        co = preprocess(co)

        y = torch.tensor([self.labels[i]], dtype=torch.float32)

        return ax, sa, co, y


def load_data(task="acl", data_dir="data", labels_dir=None, num_workers=2):

    train_ds = Dataset(
        os.path.join(data_dir, "train"),
        task,
        labels_dir=os.path.join(labels_dir, "train")
    )

    valid_ds = Dataset(
        os.path.join(data_dir, "valid"),
        task,
        labels_dir=os.path.join(labels_dir, "valid")
    )

    train_loader = data.DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
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
