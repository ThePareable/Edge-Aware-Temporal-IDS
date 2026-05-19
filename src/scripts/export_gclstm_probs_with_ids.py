import argparse, pickle, inspect
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

from src.models.gclstm import GCLSTMClassifier

def dummy_graph(in_dim: int) -> Data:
    return Data(
        x=torch.zeros((1, in_dim), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 1), dtype=torch.float32),
    )

class SeqDS(Dataset):
    def __init__(self, seqs, labels, ids):
        self.seqs = seqs
        self.labels = [np.asarray(y, dtype=np.float32) for y in labels]
        self.ids = list(ids)

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], torch.tensor(self.labels[i], dtype=torch.float32), self.ids[i]

def collate(batch):
    seqs, ys, ids = zip(*batch)
    T = len(seqs[0])
    steps = []
    for t in range(T):
        dl = []
        for s in seqs:
            g = s[t]
            if not isinstance(g, Data):
                raise TypeError("step is not Data")
            dl.append(g)
        steps.append(Batch.from_data_list(dl))
    y = torch.stack(list(ys), dim=0)
    return steps, y, list(ids)

def build_model(in_dim, num_classes, dropout=0.3):
    """
    GCLSTMClassifier constructor farklı projelerde farklı param isimleriyle gelebiliyor.
    Burada signature'ı okuyup uygun arg'larla instantiate ediyoruz.
    """
    sig = inspect.signature(GCLSTMClassifier)
    params = set(sig.parameters.keys())

    kwargs = {}
    # common
    if "in_dim" in params: kwargs["in_dim"] = in_dim
    if "input_dim" in params: kwargs["input_dim"] = in_dim

    if "num_classes" in params: kwargs["num_classes"] = num_classes
    if "out_dim" in params: kwargs["out_dim"] = num_classes
    if "n_classes" in params: kwargs["n_classes"] = num_classes

    if "dropout" in params: kwargs["dropout"] = dropout

    # try to set hidden sizes if available (defaults OK if not)
    if "hidden" in params: kwargs["hidden"] = 128
    if "gcn_layers" in params: kwargs["gcn_layers"] = 2
    if "lstm_hidden" in params: kwargs["lstm_hidden"] = 128

    if "gcn_layers" in params: kwargs["gcn_layers"] = 2
    if "lstm_hidden" in params: kwargs["lstm_hidden"] = 128

    if "hidden_dim" in params: kwargs["hidden_dim"] = 128

    # gcn hidden naming variants
    if "gcn_hidden" in params: kwargs["gcn_hidden"] = 128
    if "gcn_hidden_dim" in params: kwargs["gcn_hidden_dim"] = 128
    if "gcn_dim" in params: kwargs["gcn_dim"] = 128

    # layers naming variants
    if "layers" in params: kwargs["layers"] = 2
    if "num_layers" in params: kwargs["num_layers"] = 2
    if "n_layers" in params: kwargs["n_layers"] = 2

    try:
        return GCLSTMClassifier(**kwargs)
    except TypeError as e:
        # debug-friendly error
        raise TypeError(f"GCLSTMClassifier init failed.\n"
                        f"Signature: {sig}\n"
                        f"Tried kwargs: {kwargs}\n"
                        f"Error: {e}")

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path("src/runs") / args.run_id
    d = pickle.load(open(run_dir / f"gclstm_{args.split}.pkl", "rb"))
    seqs, labels = d["seqs"], d["labels"]
    ids = d.get("ids", [])
    meta = d.get("meta", {})
    classes = meta.get("classes", [])
    C = len(classes) if classes else len(labels[0])

    # infer in_dim from first graph
    g0 = seqs[0][0]
    in_dim = int(g0.x.size(1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(in_dim=in_dim, num_classes=C, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    loader = DataLoader(SeqDS(seqs, labels, ids), batch_size=args.batch, shuffle=False, collate_fn=collate)

    all_ids, all_probs, all_true = [], [], []
    for steps, y, bid in loader:
        steps = [b.to(device) for b in steps]
        y = y.to(device)
        logits = model(steps)  # [B,C]
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_true.append(y.detach().cpu().numpy())
        all_ids.extend(bid)

    out = {
        "ids": all_ids,
        "probs": np.concatenate(all_probs, axis=0),
        "y": np.concatenate(all_true, axis=0),
        "classes": classes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(args.out, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ wrote", args.out, "N=", len(all_ids), "C=", C)

if __name__ == "__main__":
    main()
