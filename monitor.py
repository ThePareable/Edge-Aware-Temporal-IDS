"""
Hybrid TGAT-GCLSTM Canlı Trafik İzleme — Raspberry Pi 5
=========================================================
Kullanım: sudo python3 monitor.py
"""

import os, sys, re, time, pickle, logging, subprocess, threading
from collections import deque
from pathlib import Path

BASE_DIR = "/home/raspberrypi/tez"
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, DNS, Raw, Ether
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

INTERFACE        = "eth0"
WINDOW_SIZES     = [1.0, 5.0, 30.0]
CAPTURE_INTERVAL = 5.0
BUFFER_DURATION  = 30.0
CONFIDENCE_THR   = 0.5
MIN_ALERTS       = 10
VICTIM_PC_IP     = "192.168.10.80"
ADMIN_MAC        = "50:eb:f6:d2:88:05"

CLASSES = [
    "ARP-SPOOF", "BENIGN", "DDOS-ICMP", "EXPLOITING-FTP",
    "FTP-BRUTE-FORCE", "PORT-SCANNING", "SQL-INJECTION",
    "SSH-BRUTE-FORCE", "SYN-FLOOD", "XSS"
]
BENIGN_IDX = 1

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(BASE_DIR, "monitor.log")),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────

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
            edge_index, x=h, node_time=node_time, edge_time=edge_time,
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
        alpha    = self.dropout(pyg_softmax(e_ij, index))
        return (x_j + edge_emb) * alpha.unsqueeze(-1)

class TGATModel(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=256, out_dim=10,
                 edge_feat_dim=60, time_dim=64, num_layers=3, dropout=0.0):
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

# ─────────────────────────────────────────────────────────────
# Paket Buffer
# ─────────────────────────────────────────────────────────────

class PacketBuffer:
    def __init__(self, duration=30.0):
        self.duration = duration
        self.packets  = deque()
        self.lock     = threading.Lock()

    def add(self, pkt_dict):
        with self.lock:
            self.packets.append(pkt_dict)
            cutoff = time.time() - self.duration
            while self.packets and self.packets[0]["Time"] < cutoff:
                self.packets.popleft()

    def get_dataframe(self):
        with self.lock:
            if not self.packets:
                return pd.DataFrame()
            return pd.DataFrame(list(self.packets))

    def size(self):
        with self.lock:
            return len(self.packets)

# ─────────────────────────────────────────────────────────────
# Paket → Dict
# ─────────────────────────────────────────────────────────────

def packet_to_dict(pkt):
    d = {"Time": float(pkt.time)}
    if IP in pkt:
        d["IP Source"]          = pkt[IP].src
        d["IP Destination"]     = pkt[IP].dst
        d["IP TTL"]             = pkt[IP].ttl
        d["IP Flags"]           = int(pkt[IP].flags)
        d["IP Fragment Offset"] = pkt[IP].frag
        d["IP DSCP Field"]      = pkt[IP].tos >> 2
        d["Length"]             = len(pkt)
    if TCP in pkt:
        d["Protocol"]             = "TCP"
        d["TCP Source Port"]      = pkt[TCP].sport
        d["TCP Destination Port"] = pkt[TCP].dport
        flags = pkt[TCP].flags
        d["TCP SYN Flag"]         = "Set" if flags & 0x02 else "Not set"
        d["TCP ACK Flag"]         = "Set" if flags & 0x10 else "Not set"
        d["TCP FIN Flag"]         = "Set" if flags & 0x01 else "Not set"
        d["TCP RST Flag"]         = "Set" if flags & 0x04 else "Not set"
        d["TCP Window Size"]      = pkt[TCP].window
        d["TCP Sequence Number"]  = pkt[TCP].seq
        if Raw in pkt:
            payload = pkt[Raw].load.decode("utf-8", errors="ignore").lower()
            d["Payload Length"]   = len(pkt[Raw].load)
            d["HTTP Request URI"] = payload[:500]
    elif UDP in pkt:
        d["Protocol"]             = "UDP"
        d["UDP Source Port"]      = pkt[UDP].sport
        d["UDP Destination Port"] = pkt[UDP].dport
        if DNS in pkt:
            d["has_dns"] = 1
    elif ICMP in pkt:
        d["Protocol"]  = "ICMP"
        d["ICMP Type"] = pkt[ICMP].type
    elif ARP in pkt:
        d["Protocol"] = "ARP"
    else:
        d["Protocol"] = "OTHER"
    return d

# ─────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────

def apply_normalizer(flows_df, norm_stats):
    df = flows_df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
    feature_cols = norm_stats["feature_cols"]
    available = [c for c in feature_cols if c in df.columns]
    df[available] = np.log1p(df[available].clip(lower=0))
    df[available] = (df[available] - norm_stats["mean"][available]) / norm_stats["std"][available]
    return df

# ─────────────────────────────────────────────────────────────
# Bağlantı Kesme / Açma
# ─────────────────────────────────────────────────────────────

def block_traffic():
    log.warning("🚨 SALDIRI TESPİT EDİLDİ — eth0 trafiği kesiliyor!")
    # subprocess.run(["iptables", "-I", "FORWARD", "-i", "eth0", "-j", "DROP"], check=False)
    # subprocess.run(["iptables", "-I", "FORWARD", "-o", "eth0", "-j", "DROP"], check=False)

def unblock_traffic():
    log.info("✅ eth0 trafiği yeniden açılıyor...")
    # subprocess.run(["iptables", "-D", "FORWARD", "-i", "eth0", "-j", "DROP"], check=False)
    # subprocess.run(["iptables", "-D", "FORWARD", "-o", "eth0", "-j", "DROP"], check=False)

# ─────────────────────────────────────────────────────────────
# MAC Discovery
# ─────────────────────────────────────────────────────────────

def discover_trusted_macs():
    trusted = set()
    try:
        subprocess.run(["ping", "-c", "3", "-W", "1", VICTIM_PC_IP], capture_output=True)
        result = subprocess.run(["arp", "-n"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "eth0" in line and "ether" in line:
                mac = re.search(
                    r"([0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2}:[0-9a-f]{2})",
                    line
                )
                if mac:
                    trusted.add(mac.group(1))
                    log.info(f"Güvenilen MAC bulundu: {mac.group(1)}")
    except Exception as e:
        log.warning(f"MAC discovery hatası: {e}")
    return trusted

# ─────────────────────────────────────────────────────────────
# Ana Monitor Sınıfı
# ─────────────────────────────────────────────────────────────

class HybridMonitor:
    def __init__(self):
        self.device      = torch.device("cpu")
        self.buffer      = PacketBuffer(duration=BUFFER_DURATION)
        self.alert_count = 0
        self.blocked     = False
        self.running     = True

        # TGAT
        log.info("TGAT yükleniyor...")
        self.tgat = TGATModel(
            in_dim=16, hidden_dim=256, out_dim=len(CLASSES),
            edge_feat_dim=60, time_dim=64, num_layers=3, dropout=0.0,
        )
        self.tgat.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "best_model.pt"), map_location=self.device
        ))
        self.tgat.eval()
        log.info("✓ TGAT hazır")

        # GCLSTM
        log.info("GCLSTM yükleniyor...")
        self.gclstm = GCLSTMClassifier(
            in_dim=60, hidden=128, gcn_layers=2,
            lstm_hidden=128, num_classes=len(CLASSES), dropout=0.0,
        )
        self.gclstm.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "gclstm_multilabel_best.pt"), map_location=self.device
        ))
        self.gclstm.eval()
        log.info("✓ GCLSTM hazır")

        # Stacker
        log.info("Stacker yükleniyor...")
        self.stacker = nn.Linear(2 * len(CLASSES), len(CLASSES))
        self.stacker.load_state_dict(torch.load(
            os.path.join(BASE_DIR, "hybrid_stacker.pt"), map_location=self.device
        ))
        self.stacker.eval()
        log.info("✓ Stacker hazır")

        # Normalizer
        with open(os.path.join(BASE_DIR, "normalization_stats.pkl"), "rb") as f:
            self.norm_stats = pickle.load(f)
        log.info("✓ Normalizer hazır")

        # Graph Builder & FlowBuilders
        from graph_builder import SimpleGraphBuilder
        self.gb = SimpleGraphBuilder(node_feature_dim=16, edge_feature_dim=60)
        self.gb_gclstm = SimpleGraphBuilder(node_feature_dim=60, edge_feature_dim=60)

        from build_flows import FlowBuilder
        self.flow_builders = {ws: FlowBuilder(time_window=ws) for ws in WINDOW_SIZES}
        log.info(f"✓ FlowBuilder hazır: {WINDOW_SIZES}")

        # Trusted MACs
        self.trusted_macs = discover_trusted_macs()
        log.info(f"Güvenilen MAC adresleri: {self.trusted_macs}")

    def packet_callback(self, pkt):
        try:
            if not (IP in pkt or ARP in pkt):
                return
            src_ip = pkt[IP].src if IP in pkt else ""
            dst_ip = pkt[IP].dst if IP in pkt else ""
            src_mac = pkt[Ether].src.lower() if Ether in pkt else ""
            # Pi kendi trafiğini gösterme (192.168.10.1)
            if src_ip == "192.168.10.1" or dst_ip == "192.168.10.1":
                return
            # Güvenilen MAC + SSH → yönetim trafiği, filtrele
            if TCP in pkt and (pkt[TCP].sport == 22 or pkt[TCP].dport == 22):
                if src_mac in {m.lower() for m in self.trusted_macs}:
                    return
            # Sadece Victim PC trafiğini izle
            if src_ip != VICTIM_PC_IP and dst_ip != VICTIM_PC_IP:
                return
            d = packet_to_dict(pkt)
            self.buffer.add(d)
        except Exception as e:
            log.warning(f"Callback hatası: {e}")

    def run_inference(self):
        df = self.buffer.get_dataframe()
        if df is None or len(df) < 5:
            return None, None

        # Her pencere için flow oluştur
        all_flows = []
        for ws, builder in self.flow_builders.items():
            try:
                flows = builder.build_flows(df)
                if flows is not None and len(flows) > 0:
                    all_flows.append(flows)
            except Exception as e:
                log.warning(f"Flow hatası (ws={ws}): {e}")

        if not all_flows:
            return None, None

        try:
            combined  = pd.concat(all_flows, ignore_index=True)
            norm_df   = apply_normalizer(combined, self.norm_stats)
            tgat_graph = self.gb.build_graph(norm_df)
        except Exception as e:
            log.warning(f"Graph hatası: {e}")
            return None, None

        # GCLSTM için her pencere ayrı graph
        gclstm_seq = []
        for ws, builder in self.flow_builders.items():
            try:
                flows    = builder.build_flows(df)
                norm_f   = apply_normalizer(flows, self.norm_stats)
                g        = self.gb_gclstm.build_graph(norm_f)
                gclstm_seq.append(g)
            except Exception as e:
                log.warning(f"GCLSTM graph hatası (ws={ws}): {e}")

        if len(gclstm_seq) == 0:
            return None, None

        with torch.no_grad():
            # TGAT
            bg = Batch.from_data_list([tgat_graph]).to(self.device)
            tgat_logits = self.tgat(
                bg.x, bg.edge_index, bg.node_time,
                bg.edge_time, bg.edge_attr, bg.batch
            )
            tgat_probs = torch.sigmoid(tgat_logits)  # [1, C]

            # GCLSTM
            seq_batch    = [Batch.from_data_list([g]).to(self.device) for g in gclstm_seq]
            gclstm_probs = torch.sigmoid(self.gclstm(seq_batch))  # [1, C]

            # Hybrid Stacker — logit dönüşüm
            eps  = 1e-6
            p1   = tgat_probs.clamp(eps, 1 - eps)
            p2   = gclstm_probs.clamp(eps, 1 - eps)
            X    = torch.cat([torch.log(p1 / (1 - p1)),
                              torch.log(p2 / (1 - p2))], dim=1)  # [1, 2C]
            probs = torch.sigmoid(self.stacker(X)).cpu().numpy()[0]  # (C,)

        return probs, CLASSES

    def inference_loop(self):
        log.info(f"Inference döngüsü başladı (her {CAPTURE_INTERVAL}sn)")
        time.sleep(CAPTURE_INTERVAL)

        while self.running:
            t0 = time.perf_counter()
            try:
                probs, classes = self.run_inference()
                if probs is not None:
                    top_idx  = int(np.argmax(probs))
                    top_prob = float(probs[top_idx])
                    top_cls  = classes[top_idx]

                    attack_probs = {
                        cls: round(float(p), 3)
                        for cls, p in zip(classes, probs)
                        if p > 0.3
                    }
                    log.info(
                        f"Tahmin: {top_cls} ({top_prob:.3f}) | "
                        f"Buffer: {self.buffer.size()} pkt | "
                        f"BENIGN={probs[BENIGN_IDX]:.3f} | "
                        f"Aktif: {attack_probs}"
                    )

                    is_attack = (top_idx != BENIGN_IDX and top_prob >= CONFIDENCE_THR)

                    if is_attack:
                        self.alert_count += 1
                        log.warning(
                            f"⚠️  SALDIRI: {top_cls} ({top_prob:.3f}) "
                            f"[{self.alert_count}/{MIN_ALERTS}]"
                        )
                        if self.alert_count >= MIN_ALERTS and not self.blocked:
                            block_traffic()
                            self.blocked = True
                    else:
                        if self.alert_count > 0:
                            self.alert_count = max(0, self.alert_count - 1)
                        if self.blocked and self.alert_count == 0:
                            unblock_traffic()
                            self.blocked = False

            except Exception as e:
                log.warning(f"Inference hatası: {e}")

            elapsed = time.perf_counter() - t0
            time.sleep(max(0, CAPTURE_INTERVAL - elapsed))

    def start(self):
        log.info(f"🔍 Hybrid IDS başladı — interface: {INTERFACE}")
        log.info(f"   Pencereler: {WINDOW_SIZES}sn")
        log.info(f"   Inference aralığı: {CAPTURE_INTERVAL}sn")
        log.info(f"   Saldırı eşiği: {CONFIDENCE_THR} (min {MIN_ALERTS} ardışık)")

        t = threading.Thread(target=self.inference_loop, daemon=True)
        t.start()

        try:
            sniff(
                iface=INTERFACE,
                prn=self.packet_callback,
                store=False,
                promisc=True,
                stop_filter=lambda _: not self.running,
            )
        except KeyboardInterrupt:
            log.info("Durduruluyor...")
            self.running = False
            if self.blocked:
                unblock_traffic()

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("HATA: sudo ile çalıştır!")
        sys.exit(1)

    monitor = HybridMonitor()
    monitor.start()
