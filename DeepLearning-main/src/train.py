import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm


def run_epoch(model, loader, optimizer=None, scaler=None, device="cuda"):
    train = optimizer is not None

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    preds, labels = [], []

    loop = tqdm(loader, leave=False)  # 🔥 thanh %

    for ax, sa, co, y in loop:
        ax = ax.to(device)
        sa = sa.to(device)
        co = co.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            if scaler:
                with torch.cuda.amp.autocast():
                    logit = model(ax, sa, co)
                    loss = F.binary_cross_entropy_with_logits(logit, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logit = model(ax, sa, co)
                loss = F.binary_cross_entropy_with_logits(logit, y)

                if train:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()

        pred = torch.sigmoid(logit).detach().cpu().numpy()[0][0]
        label = y.detach().cpu().numpy()[0][0]

        preds.append(pred)
        labels.append(label)

        # 🔥 update thanh %
        loop.set_description(f"Loss {loss.item():.4f}")

    fpr, tpr, _ = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return total_loss / len(loader), auc
