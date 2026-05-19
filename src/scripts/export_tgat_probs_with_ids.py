import argparse, json, pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from src.models.tgat import TGATModel

class GraphLabelIdDS(Dataset):
    def __init__(self, graphs, labels, ids):
        self.graphs = graphs
        self.labels = [np.asarray(y, dtype=np.float32) for y in labels]
        self.ids = list(ids)

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return g, y, self.ids[idx]

def collate(batch):
    graphs, ys, ids = zip(*batch)
    bg = Batch.from_data_list(list(graphs))
    y = torch.stack(list(ys), dim=0)
    return bg, y, list(ids)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="test")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = Path("src/runs") / args.run_id
    payload = pickle.load(open(run_dir / f"ml_{args.split}_graphs.pkl","rb"))
    graphs, labels, ids = payload["graphs"], payload["labels"], payload["ids"]
    meta = payload.get("meta", {})
    classes = meta.get("classes", [])
    C = len(classes) if classes else len(labels[0])

    # dynamic dims
    in_dim = int(graphs[0].x.size(1))
    edge_dim = int(graphs[0].edge_attr.size(1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TGATModel(
        in_dim=in_dim,
        hidden_dim=256,
        out_dim=C,
        edge_feat_dim=edge_dim,
        time_dim=64,
        num_layers=3,
        dropout=0.3,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    ld = DataLoader(GraphLabelIdDS(graphs, labels, ids), batch_size=args.batch, shuffle=False, collate_fn=collate)

    all_ids = []
    all_probs = []
    all_y = []
    for bg, y, bid in ld:
        bg = bg.to(device)
        logits = model(bg.x, bg.edge_index, bg.node_time, bg.edge_time, bg.edge_attr, bg.batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())
        all_ids.extend(bid)

    out = {
        "run_id": args.run_id,
        "split": args.split,
        "classes": classes,
        "ids": all_ids,
        "probs": np.vstack(all_probs).astype(np.float32),
        "labels": np.vstack(all_y).astype(np.float32),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ wrote", args.out, "N=", len(all_ids), "C=", out["probs"].shape[1])

if __name__ == "__main__":
    main()
