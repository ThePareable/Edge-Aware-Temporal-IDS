"""
Hybrid TGAT-GCLSTM Inference Script — Raspberry Pi 5
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


class TimeEncoding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.lin = nn.Linear(1, time_dim)

    def forward(self, delta_t):
        return torch.tanh(self.lin(delta_t.unsqueeze(-1)))


class TGATLayer(MessagePassing):
    def __init__(self, in_dim, time_dim, out_dim, edge_feat_dim, dropout=0.0):
        super().__init__(aggr="add", node_dim=0)
        self.lin_h    = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_t    = nn.Linear(time_dim, out_dim, bias=False)
        self.edge_lin = nn.Linear(edge_feat_dim, out_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(4 * out_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, edge_index, node_time, edge_time, edge_attr, time_encoder):
        h = self.lin_h(x)
        return self.propagate(
            edge_index, x=h,
            node_time=node_time, edge_time=edge_time,
            edge_attr=edge_attr, time_encoder=time_encoder,
            size=(h.size(0), h.size(0)),
        )

    def message(self, x_j, x_i, node_time_i, edge_time, edge_attr,
                time_encoder, index, ptr, size_i):
        t_emb    = time_encoder(node_time_i - edge_time)
        t_proj   = self.lin_t(t_emb)
        edge_emb = self.edge_lin(edge_attr)
        cat      = torch.cat([x_i, x_j, t_proj, edge_emb], dim=-1)
        e_ij     = self.leaky_relu((cat * self.attn_vec).sum(dim=-1))
        alpha    = self.dropout(softmax(e_ij, index))
        return (x_j + edge_emb) * alpha.unsqueeze(-1)


class TGATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_feat_dim,
                 time_dim=64, num_layers=3, dropout=0.0):
        super().__init__()
        self.time_encoder = TimeEncoding(time_dim)
        self.input_proj   = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TGATLayer(hidden_dim, time_dim, hidden_dim, edge_feat_dim, dropout)
            for _ in range(num_layers)
        ])
        self.activation = nn.ReLU()
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, node_time, edge_time, edge_attr, batch=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, node_time, edge_time, edge_attr, self.time_encoder)
            h = self.activation(h)
            h = self.dropout(h)
        g = global_mean_pool(h, batch) if batch is not None else h.mean(dim=0, keepdim=True)
        return self.classifier(g)


class SnapshotGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_dim, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return global_mean_pool(h, batch)


class GCLSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden, gcn_layers, lstm_hidden, num_classes, dropout):
        super().__init__()
        self.encoder = SnapshotGCNEncoder(in_dim, hidden, gcn_layers, dropout)
        self.lstm    = nn.LSTM(hidden, lstm_hidden, num_layers=1, batch_first=True)
        self.head    = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, seq_batch):
        embs   = [self.encoder(bt.x, bt.edge_index, bt.batch) for bt in seq_batch]
        E      = torch.stack(embs, dim=1)
        out, _ = self.lstm(E)
        return self.head(out[:, -1, :])


def load_models(meta, tgat_path, gclstm_path, stacker_path, device):
    C  = meta["num_classes"]
    tc = meta["tgat"]
    gc = meta["gclstm"]

    tgat = TGATModel(
        in_dim       = tc["in_dim"],
        hidden_dim   = tc["hidden_dim"],
        out_dim      = C,
        edge_feat_dim= tc["edge_dim"],
        time_dim     = tc["time_dim"],
        num_layers   = tc["num_layers"],
        dropout      = 0.0,
    )
    tgat.load_state_dict(torch.load(tgat_path, map_location=device))
    tgat.to(device).eval()

    gclstm = GCLSTMClassifier(
        in_dim     = gc["in_dim"],
        hidden     = gc["hidden"],
        gcn_layers = gc["gcn_layers"],
        lstm_hidden= gc["lstm_hidden"],
        num_classes= C,
        dropout    = 0.0,
    )
    gclstm.load_state_dict(torch.load(gclstm_path, map_location=device))
    gclstm.to(device).eval()

    stacker = nn.Linear(2 * C, C)
    stacker.load_state_dict(torch.load(stacker_path, map_location=device))
    stacker.to(device).eval()

    return tgat, gclstm, stacker


def run_inference(tgat, gclstm, stacker, tgat_graph, gclstm_seq, thresholds, device):
    with torch.no_grad():
        bg           = Batch.from_data_list([tgat_graph]).to(device)
        tgat_probs   = torch.sigmoid(
            tgat(bg.x, bg.edge_index, bg.node_time, bg.edge_time, bg.edge_attr, bg.batch)
        )
        seq_batch    = [Batch.from_data_list([s]).to(device) for s in gclstm_seq]
        gclstm_probs = torch.sigmoid(gclstm(seq_batch))

        eps   = 1e-6
        p1    = tgat_probs.clamp(eps, 1 - eps)
        p2    = gclstm_probs.clamp(eps, 1 - eps)
        X     = torch.cat([torch.log(p1 / (1 - p1)),
                           torch.log(p2 / (1 - p2))], dim=1)
        probs = torch.sigmoid(stacker(X)).cpu().numpy()[0]

    preds = (probs >= thresholds).astype(int)
    return probs, preds


def make_dummy_graph(in_dim, n_nodes=5, n_edges=4, edge_dim=None):
    return Data(
        x          = torch.zeros(n_nodes, in_dim),
        edge_index = torch.zeros(2, n_edges, dtype=torch.long),
        edge_attr  = torch.zeros(n_edges, edge_dim if edge_dim is not None else in_dim),
        node_time  = torch.zeros(n_nodes),
        edge_time  = torch.zeros(n_edges),
    )


def measure_latency(tgat, gclstm, stacker, tg, gs, thresholds, device, runs=20):
    for _ in range(3):
        run_inference(tgat, gclstm, stacker, tg, gs, thresholds, device)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_inference(tgat, gclstm, stacker, tg, gs, thresholds, device)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\n── Latency Sonuçları ({runs} run) ──────────────────")
    print(f"  p50        : {np.percentile(times, 50):.1f} ms")
    print(f"  p95        : {np.percentile(times, 95):.1f} ms")
    print(f"  p99        : {np.percentile(times, 99):.1f} ms")
    print(f"  mean       : {times.mean():.1f} ms")
    print(f"  min        : {times.min():.1f} ms")
    print(f"  max        : {times.max():.1f} ms")
    print(f"  throughput : {1000/times.mean():.2f} sample/s")
    return times


def main():
    ap = argparse.ArgumentParser(description="Hybrid TGAT-GCLSTM Inference — Pi 5")
    ap.add_argument("--tgat-model",    default="best_model.pt")
    ap.add_argument("--gclstm-model",  default="gclstm_multilabel_best.pt")
    ap.add_argument("--stacker-model", default="hybrid_stacker.pt")
    ap.add_argument("--meta",          default="meta.json")
    ap.add_argument("--input",         default=None)
    ap.add_argument("--latency",       action="store_true")
    ap.add_argument("--runs",          type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[inference] device={device}")

    with open(args.meta) as f:
        meta = json.load(f)

    classes    = meta["classes"]
    C          = meta["num_classes"]
    thresholds = np.array(meta.get("thresholds", [0.5] * C), dtype=np.float32)
    T          = meta.get("gclstm_steps", 3)
    tgat_in    = meta["tgat"]["in_dim"]
    gclstm_in  = meta["gclstm"]["in_dim"]

    print(f"[inference] {C} sınıf: {classes}")

    print("[inference] Modeller yükleniyor...")
    tgat, gclstm, stacker = load_models(
        meta, args.tgat_model, args.gclstm_model, args.stacker_model, device
    )
    print("[inference] ✓ Modeller hazır\n")

    tgat_graphs = []
    gclstm_seqs = []

    if args.input:
        with open(args.input, "rb") as f:
            pack = pickle.load(f)

        tgat_graphs = pack["tgat_graphs"]
        gclstm_seqs = pack["gclstm_seqs"]
        ids         = pack.get("ids", [str(i) for i in range(len(tgat_graphs))])

        print(f"[inference] {len(tgat_graphs)} sample işleniyor...")
        results = []
        t_start = time.perf_counter()

        for i, (tg, gs) in enumerate(zip(tgat_graphs, gclstm_seqs)):
            probs, preds = run_inference(tgat, gclstm, stacker, tg, gs, thresholds, device)
            detected     = [classes[c] for c, p in enumerate(preds) if p == 1]
            results.append({
                "id"      : ids[i],
                "detected": detected if detected else ["BENIGN"],
                "probs"   : {classes[c]: round(float(probs[c]), 4) for c in range(C)},
            })
            if (i + 1) % 10 == 0:
                elapsed = (time.perf_counter() - t_start) * 1000
                print(f"  [{i+1}/{len(tgat_graphs)}] {elapsed:.0f}ms | son: {results[-1]['detected']}")

        total_ms = (time.perf_counter() - t_start) * 1000
        print(f"\n[inference] ✓ {len(results)} sample | {total_ms:.0f}ms toplam | {total_ms/len(results):.1f}ms/sample")

        out_path = Path(args.input).stem + "_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[inference] Sonuçlar: {out_path}")

    if args.latency:
        tg = make_dummy_graph(tgat_in, edge_dim=60)
        gs = [make_dummy_graph(gclstm_in)] * T
        if args.input and tgat_graphs:
            tg = tgat_graphs[0]
            gs = gclstm_seqs[0]
        measure_latency(tgat, gclstm, stacker, tg, gs, thresholds, device, runs=args.runs)

    if not args.input and not args.latency:
        print("[inference] Smoke test (dummy input)...")
        tg           = make_dummy_graph(tgat_in, edge_dim=60)
        gs           = [make_dummy_graph(gclstm_in)] * T
        probs, preds = run_inference(tgat, gclstm, stacker, tg, gs, thresholds, device)
        detected     = [classes[c] for c, p in enumerate(preds) if p == 1]
        print(f"  probs   : { {classes[c]: round(float(probs[c]), 3) for c in range(C)} }")
        print(f"  detected: {detected if detected else ['BENIGN']}")
        print("[inference] ✓ Smoke test başarılı")


if __name__ == "__main__":
    main()
