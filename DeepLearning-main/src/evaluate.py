import argparse
import torch
from tqdm import tqdm
from sklearn import metrics

from loader import load_data
from model import TripleMRNet


# ================= RUN MODEL =================
def run_model(
    model,
    loader,
    train=False,
    optimizer=None,
    use_amp=False,
    scaler=None,
    external_criterion=None,
    grad_clip=None
):
    preds, labels = [], []
    device = next(model.parameters()).device

    if external_criterion is None:
        external_criterion = torch.nn.BCEWithLogitsLoss()

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    n = 0

    for batch in tqdm(loader):
        x1, x2, x3, y, _ = batch

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        x3 = x3.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logit = model(x1, x2, x3)
                loss = external_criterion(logit, y)

        total_loss += loss.item()

        prob = torch.sigmoid(logit).detach().cpu().numpy().ravel()
        lab  = y.detach().cpu().numpy().ravel()

        preds.extend(prob.tolist())
        labels.extend(lab.tolist())

        if train:
            if use_amp:
                scaler.scale(loss).backward()

                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

        n += 1

    avg_loss = total_loss / max(n, 1)

    # 🔥 FIX: tránh crash nếu chỉ có 1 class
    if len(set(labels)) > 1:
        fpr, tpr, _ = metrics.roc_curve(labels, preds)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = 0.5

    return avg_loss, auc, preds, labels


# ================= EVALUATE =================
def evaluate(split, model_path, diagnosis, use_gpu, data_dir="data", labels_dir=None, num_workers=4):
    train_loader, valid_loader = load_data(
        diagnosis, use_gpu, data_dir=data_dir, labels_dir=labels_dir, num_workers=num_workers
    )

    model = TripleMRNet()

    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    else:
        raise ValueError("split must be 'train' or 'valid'")

    loss, auc, preds, labels = run_model(model, loader)

    print(f"{split} loss: {loss:.4f}")
    print(f"{split} AUC: {auc:.4f}")

    return preds, labels


# ================= MAIN =================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=str, required=True)  # 🔥 FIX: string
    parser.add_argument('--gpu', action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)
