import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.io.artifacts import ensure_dir, load_pickle
from src.pipeline.step3_fit_normalizer import apply_normalizer

EXCLUDE = {
    "src_node","dst_node","src_ip","dst_ip",
    "label","binary",
    "time_window_id","window_size",
    "start_time","end_time",
    "protocol",
}

def pick_feature_cols(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def build_snapshot_graph(gdf: pd.DataFrame, feat_cols):
    # local node map (RAM-friendly)
    nodes = pd.unique(pd.concat([gdf["src_node"], gdf["dst_node"]], axis=0))
    node_map = {n:i for i,n in enumerate(nodes)}
    src = gdf["src_node"].map(node_map).astype(np.int64).values
    dst = gdf["dst_node"].map(node_map).astype(np.int64).values

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # edge features
    E = gdf[feat_cols].to_numpy(dtype=np.float32)
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.tensor(E, dtype=torch.float32)

    # node features = mean incident edge features (simple & stable)
    N = len(node_map)
    F = len(feat_cols)
    X = np.zeros((N, F), dtype=np.float32)
    C = np.zeros((N,), dtype=np.float32)

    for i in range(len(gdf)):
        s = src[i]; d = dst[i]
        X[s] += E[i]; C[s] += 1
        X[d] += E[i]; C[d] += 1
    C = np.maximum(C, 1.0)
    X = X / C[:, None]
    x = torch.tensor(X, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _dummy_graph(in_dim: int):
    # 1 node, 0 edge (padding for missing timesteps)
    import torch
    from torch_geometric.data import Data
    x = torch.zeros((1, in_dim), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.zeros((0, in_dim), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def build_sequences(df: pd.DataFrame, feat_cols, max_steps: int, norm_stats, window_sizes=None):
    # normalize numeric cols (same as TGAT pipeline)
    df = apply_normalizer(df, norm_stats)

    if window_sizes:
        df = df[df["window_size"].isin(window_sizes)].copy()

    groups = df.groupby(["window_size", "time_window_id"], sort=False)

    seqs = []
    labels = []
    ids = []
    meta = {
        "feat_cols": feat_cols,
        "num_classes": int(df["label"].nunique()),
        "classes": sorted(df["label"].unique().tolist()),
    }
    cls_to_idx = {c:i for i,c in enumerate(meta["classes"])}

    for (ws, tw), g in groups:
        g = g.sort_values("start_time", kind="mergesort")

        # label vec: multi-label değilse bile one-hot yapıyoruz (binary/multiclass aynı kod)
        y = np.zeros((meta["num_classes"],), dtype=np.float32)
        # window içinde birden fazla label varsa multi-hot olur
        for lab in pd.unique(g["label"]):
            y[cls_to_idx[lab]] = 1.0

        # split into steps
        n = len(g)
        if n == 0:
            continue
        step = max(1, int(np.ceil(n / max_steps)))
        snaps = []
        for i in range(0, n, step):
            chunk = g.iloc[i:i+step]
            if len(chunk) == 0:
                continue
            snaps.append(build_snapshot_graph(chunk, feat_cols))
            if len(snaps) >= max_steps:
                break

        if len(snaps) >= 1:
            # hybrid id: per sequence/window -> t{time_window_id}
            try:
                _tw = int(tw)
            except Exception:
                # fallback: try from current group df (gdf) or df slice
                if 'gdf' in locals() and 'time_window_id' in gdf.columns:
                    _tw = int(gdf['time_window_id'].iloc[0])
                else:
                    _tw = len(ids)
            ids.append(f"t{_tw}")
#             ids.append(str(len(ids)))  # disabled duplicate ids.append
            seqs.append(snaps)
            # attack-only: drop all-zero labels

            if float(getattr(y, 'sum', lambda: sum(y))()) == 0.0:

                continue

            labels.append(y)

    # --- enforce fixed-length sequences: T = len(window_sizes) ---

    T_target = len(window_sizes) if window_sizes is not None else None

    if T_target is not None:

        in_dim = len(feat_cols)

        fixed = []

        for steps in seqs:

            # steps: list[Data]

            if not isinstance(steps, (list, tuple)):

                # unexpected; keep as-is

                fixed.append(steps)

                continue

            if len(steps) < T_target:

                steps = list(steps) + [_dummy_graph(in_dim) for _ in range(T_target - len(steps))]

            elif len(steps) > T_target:

                steps = list(steps)[:T_target]

            fixed.append(steps)

        seqs = fixed


    if len(ids) != len(seqs):
        # fallback: enforce 1 id per sample
        ids = [str(i) for i in range(len(seqs))]
    return seqs, labels, ids, meta

def write_split(run_dir: Path, name: str, max_steps: int, window_sizes):
    df = pd.read_csv(run_dir / f"{name}_flows.csv")
    norm_stats = load_pickle(run_dir / "normalization_stats.pkl")
    feat_cols = pick_feature_cols(df)

    print(f"[prepare_gclstm_seq] {name}: rows={len(df)} feat_cols={len(feat_cols)}")
    print("[prepare_gclstm_seq] first cols:", feat_cols[:20])

    seqs, labels, ids, meta = build_sequences(df, feat_cols, max_steps=max_steps, norm_stats=norm_stats, window_sizes=window_sizes)

    out = {"seqs": seqs, "labels": labels, "ids": ids, "meta": meta}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-run-id", required=True)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--window-sizes", default="", help="comma list e.g. 1,5,30 (empty=all)")
    args = ap.parse_args()

    run_dir = Path("src/runs") / args.run_id
    out_dir = Path("src/runs") / args.out_run_id
    ensure_dir(out_dir)

    window_sizes = None
    if args.window_sizes.strip():
        window_sizes = [float(x) for x in args.window_sizes.split(",")]

    for split in ["train", "val", "test"]:
        pack = write_split(run_dir, split, max_steps=args.max_steps, window_sizes=window_sizes)
        p = out_dir / f"gclstm_{split}.pkl"
        with open(p, "wb") as f:
            pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ wrote {split}: {p} | samples={len(pack['seqs'])}")

if __name__ == "__main__":
    main()
