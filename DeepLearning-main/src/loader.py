import torch
import torch.utils.data as data
import numpy as np
import os

class Dataset(data.Dataset):
    def __init__(self, datadir, tear_type, use_gpu, labels_dir=None, augment=False):
        self.datadir = datadir
        self.use_gpu = use_gpu
        self.augment = augment

        # load labels như bạn đã có (giữ nguyên)
        # self.paths, self.labels ...

    def __getitem__(self, idx):
        # load npy như bạn đã làm
        # vol_axial, vol_sagit, vol_coron

        # ===== AUGMENT (nhẹ) =====
        if self.augment and np.random.rand() < 0.5:
            vol_axial = vol_axial[:, ::-1, :]   # flip
        if self.augment and np.random.rand() < 0.3:
            vol_sagit = vol_sagit + np.random.normal(0, 0.01, vol_sagit.shape)

        return vol_axial, vol_sagit, vol_coron, label, abnormal


def load_data(task="abnormal", use_gpu=False, data_dir="data", labels_dir=None, num_workers=2):
    train_dataset = Dataset(os.path.join(data_dir, "train"), task, use_gpu, labels_dir, augment=True)
    valid_dataset = Dataset(os.path.join(data_dir, "valid"), task, use_gpu, labels_dir, augment=False)

    # ===== sampler cân bằng class =====
    labels = np.array(train_dataset.labels)
    class_count = np.bincount(labels)
    weights = 1. / class_count
    sample_weights = weights[labels]
    sampler = data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader
