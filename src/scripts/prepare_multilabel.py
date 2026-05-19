import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io.artifacts import ensure_dir, save_pickle, load_pickle
from src.pipeline.step3_fit_normalizer import apply_normalizer
from src.graph.graph_builder import SimpleGraphBuilder


def pick_edge_feature_cols(df: pd.DataFrame):
    exclude = {
        "src_node","dst_node","src_ip","dst_ip",
        "label","binary",
        "time_window_id","window_size",
        "start_time","end_time",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def labelvec_from_labels(labels: pd.Series, class_to_idx: dict, C: int) -> np.ndarray:
    # window içindeki tüm flow label'larını OR ile birleştir
    v = np.zeros((C,), dtype=np.float32)
    # labels bazen NaN olabilir
    for lab in labels.dropna().astype(str).values:
        idx = class_to_idx.get(lab)
        if idx is not None:
            v[idx] = 1.0
    return v


def build_split(df: pd.DataFrame, name: str, builder: SimpleGraphBuilder, keep_ws: set, class_to_idx: dict, C: int):
    graphs, labels, ids = [], [], []
    grouped = df.groupby(["window_size", "time_window_id"], sort=False)

    for (ws, tw), gdf in tqdm(grouped, desc=f"multilabel_graphs[{name}]"):
        ws = float(ws)
        if ws not in keep_ws:
            continue

        # label_vec'i label kolonundan üret
        if "label" not in gdf.columns:
            raise KeyError("flows.csv içinde 'label' kolonu yok. Bu script label'dan label_vec üretiyor.")

        label_vec = labelvec_from_labels(gdf["label"], class_to_idx, C)
        # ---- BENIGN-aware multilabel ----
        benign_name = "BENIGN"
        benign_idx = class_to_idx.get(benign_name, None)

        if benign_idx is None:
            # Attack-only mod (eski davranış): hiç class yoksa sampleı at.
            if not np.any(label_vec):
                continue
        else:
            # Attack var mı? (BENIGN hariç)
            tmp = label_vec.copy()
            tmp[benign_idx] = 0.0
            attack_active = bool(tmp.any())

            if attack_active:
                # Attack varsa benign kesin 0 olmalı
                label_vec[benign_idx] = 0.0
            else:
                # Attack yoksa benign sample: benign=1 ve diğerleri 0
                label_vec[:] = 0.0
                label_vec[benign_idx] = 1.0

        graph_id = f"w{ws}_t{int(tw)}"
        data, _ = builder.build_graph(
            gdf,
            window_id=int(tw),
            window_size=ws,
            label_vec=label_vec
        )

        graphs.append(data)
        labels.append(label_vec.astype(np.float32))
        ids.append(graph_id)

    return graphs, labels, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-run-id", required=True)
    ap.add_argument("--windows", default="1,5,30")
    ap.add_argument("--keep-classes-file", required=True)
    args = ap.parse_args()

    run_dir = Path("src/runs") / args.run_id
    out_dir = Path("src/runs") / args.out_run_id
    ensure_dir(out_dir)

    keep_ws = set(float(x.strip()) for x in args.windows.split(",") if x.strip())
    classes = [l.strip() for l in open(args.keep_classes_file) if l.strip()]
    if len(classes) == 0:
        raise RuntimeError("keep-classes-file boş görünüyor.")

    class_to_idx = {c:i for i,c in enumerate(classes)}
    C = len(classes)

    train_df = pd.read_csv(run_dir / "train_flows.csv")
    val_df   = pd.read_csv(run_dir / "val_flows.csv")
    test_df  = pd.read_csv(run_dir / "test_flows.csv")

    # normalization_stats varsa uygula (yoksa direkt geç)
    stats_path = run_dir / "normalization_stats.pkl"
    if stats_path.exists():
        norm_stats = load_pickle(stats_path)
        train_df = apply_normalizer(train_df, norm_stats)
        val_df   = apply_normalizer(val_df,   norm_stats)
        test_df  = apply_normalizer(test_df,  norm_stats)

    edge_feature_cols = pick_edge_feature_cols(train_df)
    print(f"[prepare_multilabel] edge_feature_cols={len(edge_feature_cols)}")
    if edge_feature_cols:
        print("[prepare_multilabel] first cols:", edge_feature_cols[:20])

    builder = SimpleGraphBuilder(edge_feature_cols)

    tr_g, tr_y, tr_ids = build_split(train_df, "train", builder, keep_ws, class_to_idx, C)
    va_g, va_y, va_ids = build_split(val_df,   "val",   builder, keep_ws, class_to_idx, C)
    te_g, te_y, te_ids = build_split(test_df,  "test",  builder, keep_ws, class_to_idx, C)

    meta = {"classes": classes, "num_classes": C}

    # IMPORTANT: save_pickle signature = save_pickle(path, obj)
    save_pickle(out_dir / "ml_train_graphs.pkl", {"graphs": tr_g, "labels": tr_y, "ids": tr_ids, "meta": meta})
    save_pickle(out_dir / "ml_val_graphs.pkl",   {"graphs": va_g, "labels": va_y, "ids": va_ids, "meta": meta})
    save_pickle(out_dir / "ml_test_graphs.pkl",  {"graphs": te_g, "labels": te_y, "ids": te_ids, "meta": meta})

    print("✓ wrote:")
    print(" ", out_dir / "ml_train_graphs.pkl")
    print(" ", out_dir / "ml_val_graphs.pkl")
    print(" ", out_dir / "ml_test_graphs.pkl")
    print(f"Counts: train={len(tr_g)} val={len(va_g)} test={len(te_g)}")
    print("OUT RUN ID:", args.out_run_id)
    print("OUT RUN DIR:", out_dir.resolve())


if __name__ == "__main__":
    main()
