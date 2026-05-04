import numpy as np
import os
import torch
import torch.utils.data as data

INPUT_DIM = 224
TARGET_SLICES = 32
MEAN = 58.09
STD = 49.73


def _normalize_id(x):
    return os.path.splitext(os.path.basename(str(x).strip()))[0]


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

    pad = (vol.shape[2] - INPUT_DIM) // 2
    vol = vol[:, pad:-pad, pad:-pad]

    vol = (vol - MEAN) / STD

    vol = resize_slices(vol)

    vol = np.stack([vol, vol, vol], axis=1)
    return torch.from_numpy(vol).float()


class Dataset(data.Dataset):
    def __init__(self, datadir, task, labels_root):
        self.datadir = datadir

        split = os.path.basename(datadir)  # train / valid

        # 🔥 FIX CSV PATH ĐÚNG FORMAT DATASET BẠN
        label_path = os.path.join(labels_root, f"{split}-{task}.csv")
        abnormal_path = os.path.join(labels_root, f"{split}-abnormal.csv")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing {label_path}")

        label_dict = {}
        abnormal_dict = {}

        with open(label_path) as f:
            for line in f:
                k, v = line.strip().split(',')
                label_dict[_normalize_id(k)] = int(v)

        with open(abnormal_path) as f:
            for line in f:
                k, v = line.strip().split(',')
                abnormal_dict[_normalize_id(k)] = int(v)

        self.paths = []
        for f in os.listdir(os.path.join(datadir, "axial")):
            if f.endswith(".npy"):
                pid = _normalize_id(f)
                if pid in label_dict and pid in abnormal_dict:
                    self.paths.append(f)

        self.paths.sort()
        self.labels = [label_dict[_normalize_id(p)] for p in self.paths]

        print(f"Loaded {len(self.paths)} samples from {datadir}")
        for i in range(min(3, len(self.paths))):
            print(self.paths[i], self.labels[i])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        f = self.paths[i]

        ax = preprocess(np.load(os.path.join(self.datadir, "axial", f)))
        sa = preprocess(np.load(os.path.join(self.datadir, "sagittal", f)))
        co = preprocess(np.load(os.path.join(self.datadir, "coronal", f)))

        y = torch.tensor([self.labels[i]], dtype=torch.float32)

        return ax, sa, co, y


def load_data(task="acl", data_dir="data", labels_root="labels", num_workers=2):

    train_ds = Dataset(
        os.path.join(data_dir, "train"),
        task,
        labels_root
    )

    valid_ds = Dataset(
        os.path.join(data_dir, "valid"),
        task,
        labels_root
    )

    train_loader = data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader
