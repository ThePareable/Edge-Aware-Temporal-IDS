import argparse
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Batch

from src.io.artifacts import ensure_dir, save_json
from src.models.tgat import TGATModel

class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0, alpha=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.alpha = alpha  # optional scalar or per-class tensor

    def forward(self, logits, targets):
        # logits, targets: [B, C]
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)  # prob of the true class
        focal = (1 - pt).pow(self.gamma)

        if self.alpha is not None:
            # alpha can be scalar or [C]
            alpha = self.alpha
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=logits.device)
            if alpha.ndim == 1:
                alpha = alpha.view(1, -1)
            at = alpha * targets + (1 - alpha) * (1 - targets)
            loss = at * focal * bce
        else:
            loss = focal * bce

        return loss.mean()


class GraphLabelDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels  # list[list[float]] multi-hot

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return g, y


def collate_multilabel(batch):
    graphs, ys = zip(*batch)
    bg = Batch.from_data_list(list(graphs))
    y = torch.stack(list(ys), dim=0)  # [B, C]
    return bg, y


def compute_pos_weight(train_labels: torch.Tensor):
    # train_labels: [N, C] in {0,1}
    pos = train_labels.sum(dim=0)
    neg = train_labels.shape[0] - pos
    # pos_weight = neg/pos
    pos_weight = neg / (pos + 1e-6)
    # clamp to avoid crazy values for ultra-rare classes
    pos_weight = torch.clamp(pos_weight, 1.0, 50.0)
    return pos_weight


def compute_sample_weights(labels):
    # labels: list[list[0/1]] size N x C
    import numpy as np
    Y = np.array(labels, dtype=np.float32)
    # per-class freq
    freq = Y.mean(axis=0) + 1e-6
    inv = 1.0 / freq
    inv = np.clip(inv, 1.0, 50.0)
    # weight per sample = sum(inv[c] for positive labels)
    w = (Y * inv[None, :]).sum(axis=1)
    # if a sample has no labels (shouldn't for attack-only), give small weight
    w = np.where(w > 0, w, 1.0)
    return w.astype(np.float64)



def f1_micro_macro(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_true/y_pred: [N, C] in {0,1}
    eps = 1e-9
    tp = (y_true * y_pred).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    f1_per_class = (2 * tp) / (2 * tp + fp + fn + eps)
    f1_macro = f1_per_class.mean().item()

    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    f1_micro = (2 * tp_micro) / (2 * tp_micro + fp_micro + fn_micro + eps)
    return f1_micro.item(), f1_macro, f1_per_class.cpu().tolist()


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_true = []
    all_pred = []

    for bg, y in loader:
        bg = bg.to(device)
        y = y.to(device)

        logits = model(bg.x, bg.edge_index, bg.node_time, bg.edge_time, bg.edge_attr, bg.batch)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).float()

        total_loss += loss.item()
        n_batches += 1

        all_true.append(y.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y_true = torch.cat(all_true, dim=0)
    y_pred = torch.cat(all_pred, dim=0)

    f1_micro, f1_macro, f1_per_class = f1_micro_macro(y_true, y_pred)

    # exact match ratio (subset accuracy) — strict
    exact_match = (y_true.eq(y_pred).all(dim=1).float().mean().item())

    return {
        "loss": total_loss / max(1, n_batches),
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "exact_match": exact_match,
        "f1_per_class": f1_per_class,
    }


def train_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for bg, y in loader:
        bg = bg.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(bg.x, bg.edge_index, bg.node_time, bg.edge_time, bg.edge_attr, bg.batch)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def load_payload(path: Path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["graphs"], payload["labels"], payload.get("meta", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--time-dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "runs" / args.run_id
    out_dir = ensure_dir(project_root / "outputs" / f"multilabel_{args.run_id}")

    tr_path = run_dir / "ml_train_graphs.pkl"
    va_path = run_dir / "ml_val_graphs.pkl"
    te_path = run_dir / "ml_test_graphs.pkl"

    train_g, train_y, meta = load_payload(tr_path)
    val_g, val_y, _ = load_payload(va_path)
    test_g, test_y, _ = load_payload(te_path)

    classes = meta.get("classes", [])
    num_classes = len(classes) if classes else (len(train_y[0]) if train_y else 0)
    if num_classes <= 1:
        raise RuntimeError("num_classes invalid. Did you prepare multilabel data?")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dynamic dims from data
    in_dim = int(train_g[0].x.size(1))
    edge_dim = int(train_g[0].edge_attr.size(1))
    model = TGATModel(
        in_dim=in_dim,
        hidden_dim=args.hidden,
        out_dim=num_classes,
        edge_feat_dim=edge_dim,
        time_dim=args.time_dim,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    # compute pos_weight from training labels
    train_Y = torch.tensor(train_y, dtype=torch.float32)
    pos_weight = compute_pos_weight(train_Y).to(device)

    criterion = FocalLoss(pos_weight=pos_weight, gamma=2.0, alpha=None)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    # sampler disabled (OOM)
    train_loader = DataLoader(GraphLabelDataset(train_g, train_y), batch_size=args.batch, shuffle=True, collate_fn=collate_multilabel)
    val_loader   = DataLoader(GraphLabelDataset(val_g,   val_y),   batch_size=args.batch, shuffle=False, collate_fn=collate_multilabel)
    test_loader  = DataLoader(GraphLabelDataset(test_g,  test_y),  batch_size=args.batch, shuffle=False, collate_fn=collate_multilabel)

    best = -1.0
    patience = 0
    best_path = out_dir / "best_model.pt"

    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=args.grad_clip)
        vm = eval_epoch(model, val_loader, criterion, device, threshold=args.threshold)

        scheduler.step(vm["f1_micro"])

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={tr_loss:.4f} "
              f"| val_loss={vm['loss']:.4f} val_f1_micro={vm['f1_micro']:.4f} val_f1_macro={vm['f1_macro']:.4f} val_exact={vm['exact_match']:.4f}")

        score = vm["f1_micro"]
        if score > best:
            best = score
            patience = 0
            torch.save(model.state_dict(), best_path)
            print("✓ saved best")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stop.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    tm = eval_epoch(model, test_loader, criterion, device, threshold=args.threshold)

    save_json(out_dir / "metrics.json", {
        "best_val_f1_micro": best,
        "test": tm,
        "classes": classes,
        "threshold": args.threshold,
        "pos_weight": pos_weight.detach().cpu().tolist(),
    })

    print(f"\n✓ DONE. outputs: {out_dir}")
    print("Test metrics:", {k: tm[k] for k in ['loss','f1_micro','f1_macro','exact_match']})


if __name__ == "__main__":
    main()