import argparse
from pathlib import Path
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Batch

from src.io.artifacts import ensure_dir, save_json
from src.models.gclstm import GCLSTMClassifier


def load_pack(run_dir: Path, split: str):
    p = run_dir / f"gclstm_{split}.pkl"
    d = pickle.load(open(p, "rb"))
    seqs = d["seqs"]      # list[sample] -> list[t] -> Data
    labels = d["labels"]  # list[np.ndarray] (C,)
    meta = d.get("meta", {})
    return seqs, labels, meta


class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        steps = self.seqs[idx]  # list[Data]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # (C,)
        return steps, y


def collate_fn(batch):
    # batch: list[(steps, y)]
    steps_list, ys = zip(*batch)
    T = len(steps_list[0])
    if any(len(s) != T for s in steps_list):
        raise ValueError("Variable-length sequences found. Ensure prepare script creates fixed-length sequences per sample.")

    # time-major: steps[t] is a Batch of graphs at time t
    steps = []
    for t in range(T):
        graphs_t = [s[t] for s in steps_list]
        steps.append(Batch.from_data_list(graphs_t))

    Y = torch.stack(list(ys), dim=0)  # (B, C)
    return steps, Y


@torch.no_grad()
def eval_epoch(model, loader, device, threshold: float):
    model.eval()
    total_loss = 0.0
    total = 0

    all_pred = []
    all_true = []

    crit = nn.BCEWithLogitsLoss()

    for steps, Y in loader:
        steps = [b.to(device) for b in steps]
        Y = Y.to(device)

        logits = model(steps)  # (B, C)
        loss = crit(logits, Y)

        B = Y.size(0)
        total_loss += float(loss.item()) * B
        total += B

        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).float()

        all_pred.append(pred.detach().cpu())
        all_true.append(Y.detach().cpu())

    Yp = torch.cat(all_pred, dim=0).numpy()
    Yt = torch.cat(all_true, dim=0).numpy()

    # micro F1
    tp = float(((Yp == 1) & (Yt == 1)).sum())
    fp = float(((Yp == 1) & (Yt == 0)).sum())
    fn = float(((Yp == 0) & (Yt == 1)).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec  = tp / max(tp + fn, 1e-9)
    f1_micro = 2 * prec * rec / max(prec + rec, 1e-9)

    # macro F1 (per class)
    C = Yt.shape[1]
    f1s = []
    for c in range(C):
        tp_c = float(((Yp[:,c] == 1) & (Yt[:,c] == 1)).sum())
        fp_c = float(((Yp[:,c] == 1) & (Yt[:,c] == 0)).sum())
        fn_c = float(((Yp[:,c] == 0) & (Yt[:,c] == 1)).sum())
        p_c = tp_c / max(tp_c + fp_c, 1e-9)
        r_c = tp_c / max(tp_c + fn_c, 1e-9)
        f1_c = 2 * p_c * r_c / max(p_c + r_c, 1e-9)
        f1s.append(f1_c)
    f1_macro = float(np.mean(f1s))

    exact = float((Yp == Yt).all(axis=1).mean())

    return {
        "loss": total_loss / max(total, 1),
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "exact_match": exact,
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    crit = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total = 0

    for steps, Y in loader:
        steps = [b.to(device) for b in steps]
        Y = Y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(steps)  # (B, C)
        loss = crit(logits, Y)
        loss.backward()
        optimizer.step()

        B = Y.size(0)
        total_loss += float(loss.item()) * B
        total += B

    return total_loss / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gcn-hidden", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)  # LSTM hidden
    ap.add_argument("--layers", type=int, default=2)    # gcn_layers
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = Path("src/runs") / args.run_id
    out_dir = ensure_dir(Path("src/outputs") / f"gclstm_ml_{args.run_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs, train_labels, meta = load_pack(run_dir, "train")
    val_seqs, val_labels, _ = load_pack(run_dir, "val")
    test_seqs, test_labels, _ = load_pack(run_dir, "test")

    x0 = train_seqs[0][0].x
    in_dim = int(x0.size(-1))
    classes = meta.get("classes", [])
    num_classes = int(meta.get("num_classes", len(classes) if classes else train_labels[0].shape[0]))

    print(f"[gclstm-ml] in_dim={in_dim} num_classes={num_classes} device={device}")

    train_ds = SeqDataset(train_seqs, train_labels)
    val_ds = SeqDataset(val_seqs, val_labels)
    test_ds = SeqDataset(test_seqs, test_labels)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_ld  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = GCLSTMClassifier(
        in_dim=in_dim,
        hidden=args.gcn_hidden,
        gcn_layers=int(args.layers),
        lstm_hidden=args.hidden,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 1e18
    best_path = run_dir / "gclstm_multilabel_best.pt"
    bad = 0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_ld, optimizer, device)
        vm = eval_epoch(model, val_ld, device, threshold=args.threshold)

        print(
            f"Epoch {ep}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={vm['loss']:.4f} "
            f"val_f1_micro={vm['f1_micro']:.4f} val_f1_macro={vm['f1_macro']:.4f} val_exact={vm['exact_match']:.4f}"
        )

        if vm["loss"] < best_val:
            best_val = vm["loss"]
            torch.save(model.state_dict(), best_path)
            bad = 0
            print("✓ saved best")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stop.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    tm = eval_epoch(model, test_ld, device, threshold=args.threshold)

    save_json(out_dir / "metrics.json", {"test": tm, "classes": classes})
    print("✓ DONE. outputs:", out_dir)
    print("Test metrics:", tm)


if __name__ == "__main__":
    main()