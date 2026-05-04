import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

from loader import load_data
from model import TripleMRNet


# ===== ARG =====
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--diagnosis', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser


# ===== RUN MODEL =====
def run_model(model, loader, train=False, optimizer=None,
              use_amp=False, scaler=None,
              external_criterion=None, grad_clip=None):

    preds, labels = [], []
    device = next(model.parameters()).device

    total_loss, n = 0.0, 0

    if train:
        model.train()
    else:
        model.eval()

    for batch in tqdm(loader, ncols=100):  # thanh progress %
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

                # ===== FIX SHAPE (QUAN TRỌNG) =====
                y = y.view_as(logit)

                # ===== FIX LOSS =====
                if external_criterion is not None:
                    loss = F.binary_cross_entropy_with_logits(logit, y)
                else:
                    loss = F.binary_cross_entropy_with_logits(logit, y)

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

    # ===== AUC =====
    try:
        fpr, tpr, _ = metrics.roc_curve(labels, preds)
        auc = metrics.auc(fpr, tpr)
    except:
        auc = 0.5

    return avg_loss, auc, preds, labels


# ===== EVALUATE =====
def evaluate(split, model_path, diagnosis, use_gpu,
             data_dir="data", labels_dir=None, num_workers=4):

    train_loader, valid_loader = load_data(
        diagnosis, use_gpu,
        data_dir=data_dir,
        labels_dir=labels_dir,
        num_workers=num_workers
    )

    model = TripleMRNet()
    state_dict = torch.load(model_path, map_location=("cuda" if use_gpu else "cpu"))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    loader = train_loader if split == 'train' else valid_loader

    loss, auc, preds, labels = run_model(model, loader)

    print(f'{split} loss: {loss:.4f}')
    print(f'{split} AUC: {auc:.4f}')

    return preds, labels


# ===== MAIN =====
if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.diagnosis, args.gpu)
