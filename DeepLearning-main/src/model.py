import torch
import torch.nn as nn
from torchvision import models


class TripleMRNet(nn.Module):
    def __init__(self):
        super().__init__()

        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(m.children())[:-1])

        self.fc = nn.Linear(512 * 3, 1)

    def encode(self, vol):
        vol = vol.squeeze(0)
        f = self.encoder(vol)
        f = f.view(f.size(0), -1)
        return f.mean(dim=0, keepdim=True)  # 🔥 mean pooling

    def forward(self, ax, sa, co):
        x = self.encode(ax)
        y = self.encode(sa)
        z = self.encode(co)

        w = torch.cat([x, y, z], dim=1)
        return self.fc(w)
