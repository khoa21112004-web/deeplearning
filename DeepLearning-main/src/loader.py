import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

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
    def __init__(self, datadir, tear_type, use_gpu, labels_dir=None):
        super().__init__()
        self.use_gpu = use_gpu

        label_dict = {}
        abnormal_label_dict = {}
        self.paths = []

        self.datadir = datadir.rstrip("/")

        label_root = labels_dir if labels_dir else datadir

        # ===== LOAD LABEL =====
        for line in open(label_root + '-' + tear_type + '.csv'):
            f, l = line.strip().split(',')
            label_dict[_normalize_id(f)] = int(l)

        for line in open(label_root + '-abnormal.csv'):
            f, l = line.strip().split(',')
            abnormal_label_dict[_normalize_id(f)] = int(l)

        # ===== LOAD FILE =====
        for f in os.listdir(os.path.join(self.datadir, "axial")):
            if f.endswith(".npy"):
                self.paths.append(f)

        self.paths.sort()

        # ===== FILTER =====
        self.paths = [
            p for p in self.paths
            if _normalize_id(p) in label_dict and _normalize_id(p) in abnormal_label_dict
        ]

        if len(self.paths) == 0:
            raise ValueError("❌ Không tìm thấy data hợp lệ")

        self.labels = [label_dict[_normalize_id(p)] for p in self.paths]
        self.abnormal_labels = [abnormal_label_dict[_normalize_id(p)] for p in self.paths]

        # ===== WEIGHTED LOSS =====
        if tear_type != "abnormal":
            temp = [self.labels[i] for i in range(len(self.labels)) if self.abnormal_labels[i] == 1]
            neg_weight = float(np.mean(temp)) if temp else 0.5
        else:
            neg_weight = float(np.mean(self.labels)) if self.labels else 0.5

        self.weights = [neg_weight, 1 - neg_weight]

        # ===== AUGMENTATION ===== 🔥
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])

    def weighted_loss(self, prediction, target):
        weights = torch.FloatTensor([self.weights[int(t[0])] for t in target])
        if self.use_gpu:
            weights = weights.cuda()
        return F.binary_cross_entropy_with_logits(prediction, target, weight=weights)

    def process_volume(self, vol):
        # crop center
        pad = int((vol.shape[2] - INPUT_DIM) / 2)
        vol = vol[:, pad:-pad, pad:-pad]

        # normalize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1e-6)
        vol = vol * MAX_PIXEL_VAL
        vol = (vol - MEAN) / STDDEV

        # to 3 channel
        vol = np.stack((vol,) * 3, axis=1)

        # 🔥 AUGMENT từng slice
        for i in range(vol.shape[0]):
            img = torch.FloatTensor(vol[i])
            vol[i] = self.augment(img).numpy()

        return torch.FloatTensor(vol)

    def __getitem__(self, index):
        filename = self.paths[index]

        vol_axial = np.load(os.path.join(self.datadir, "axial", filename))
        vol_sagit = np.load(os.path.join(self.datadir, "sagittal", filename))
        vol_coron = np.load(os.path.join(self.datadir, "coronal", filename))

        vol_axial = self.process_volume(vol_axial)
        vol_sagit = self.process_volume(vol_sagit)
        vol_coron = self.process_volume(vol_coron)

        label = torch.FloatTensor([self.labels[index]])

        return vol_axial, vol_sagit, vol_coron, label, self.abnormal_labels[index]

    def __len__(self):
        return len(self.paths)


def load_data(task="acl", use_gpu=False, data_dir="data", labels_dir=None, num_workers=4):

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    labels_train = None if labels_dir is None else os.path.join(labels_dir, "train")
    labels_valid = None if labels_dir is None else os.path.join(labels_dir, "valid")

    train_dataset = Dataset(train_dir, task, use_gpu, labels_train)
    valid_dataset = Dataset(valid_dir, task, use_gpu, labels_valid)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        persistent_workers=(num_workers > 0),
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, valid_loader
