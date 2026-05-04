import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from src.loader import load_data
from src.model import TripleMRNet


def run_epoch(model, loader, optimizer=None, device="cuda"):
    train = optimizer is not None
    model.train() if train else model.eval()

    preds, labels = [], []
    total_loss = 0

    loop = tqdm(loader)

    for ax, sa, co, y in loop:
        ax, sa, co, y = ax.to(device), sa.to(device), co.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        logit = model(ax, sa, co)
        loss = F.binary_cross_entropy_with_logits(logit, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        pred = torch.sigmoid(logit).item()
        preds.append(pred)
        labels.append(y.item())

        loop.set_description(f"Loss {loss.item():.4f}")

    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return total_loss / len(loader), auc


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, valid_loader = load_data(
        task="acl",
        data_dir="data",
        labels_root="labels"
    )

    model = TripleMRNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, 6):
        print(f"\n🚀 Epoch {epoch}")

        train_loss, train_auc = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_auc = run_epoch(model, valid_loader, None, device)

        print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
