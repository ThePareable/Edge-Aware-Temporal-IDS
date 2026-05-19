"""
Canlı Trafik İzleme ve TGAT Saldırı Tespit Scripti — Raspberry Pi 5
====================================================================
Çalışma prensibi:
  - Her 5 saniyede bir eth0'dan geçen trafiği yakala
  - 1sn, 5sn, 30sn pencerelerinde flow oluştur
  - Hepsini tek graph'a birleştir
  - TGAT modeline ver
  - Saldırı tespit edilince eth0 bağlantısını kes

Kullanım:
  sudo python3 monitor.py

NOT: root yetkisi gerekiyor (tcpdump ve iptables için)
"""

import os
import sys
import time
import pickle
import logging
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, DNS, Raw
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import softmax

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/home/raspberrypi/tez/monitor.log"),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

INTERFACE        = "eth0"
WINDOW_SIZES     = [1.0, 5.0, 30.0]   # saniye
CAPTURE_INTERVAL = 5.0                 # kaç saniyede bir inference
BUFFER_DURATION  = 30.0               # kaç saniyelik paket tutulur
CONFIDENCE_THR   = 0.7                # saldırı eşiği
MIN_ALERTS       = 3                  # kaç ardışık tespitte bağlantı kesilir
MODEL_DIR        = Path("/home/raspberrypi/tez")

CLASSES = [
    "ARP-SPOOF", "BENIGN", "DDOS-ICMP", "EXPLOITING-FTP",
    "FTP-BRUTE-FORCE", "PORT-SCANNING", "SQL-INJECTION",
    "SSH-BRUTE-FORCE", "SYN-FLOOD", "XSS"
]
BENIGN_IDX = 1

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
        alpha    = self.dropout(softmax(e_ij, index))
        return (x_j + edge_emb) * alpha.unsqueeze(-1)


class TGATModel(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=128, out_dim=10,
                 edge_feat_dim=59, time_dim=32, num_layers=3, dropout=0.0):
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

# ─────────────────────────────────────────────────────────────
# Paket Buffer
# ─────────────────────────────────────────────────────────────

class PacketBuffer:
    """Thread-safe paket tamponu."""
    def __init__(self, duration=30.0):
        self.duration = duration
        self.packets  = deque()
        self.lock     = threading.Lock()

    def add(self, pkt_dict):
        with self.lock:
            self.packets.append(pkt_dict)
            # Eski paketleri temizle
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
# Paket → Dict dönüşümü
# ─────────────────────────────────────────────────────────────

def packet_to_dict(pkt):
    """Scapy paketini FlowBuilder'ın beklediği dict formatına çevir."""
    d = {"Time": float(pkt.time)}

    if IP in pkt:
        d["IP Source"]      = pkt[IP].src
        d["IP Destination"] = pkt[IP].dst
        d["IP TTL"]         = pkt[IP].ttl
        d["IP Flags"]       = int(pkt[IP].flags)
        d["IP Fragment Offset"] = pkt[IP].frag
        d["IP DSCP Field"]  = pkt[IP].tos >> 2
        d["Length"]         = len(pkt)

    if TCP in pkt:
        d["Protocol"]           = "TCP"
        d["TCP Source Port"]    = pkt[TCP].sport
        d["TCP Destination Port"] = pkt[TCP].dport
        flags = pkt[TCP].flags
        d["TCP SYN Flag"] = "Set" if flags & 0x02 else "Not set"
        d["TCP ACK Flag"] = "Set" if flags & 0x10 else "Not set"
        d["TCP FIN Flag"] = "Set" if flags & 0x01 else "Not set"
        d["TCP RST Flag"] = "Set" if flags & 0x04 else "Not set"
        d["TCP Window Size"]    = pkt[TCP].window
        d["TCP Sequence Number"] = pkt[TCP].seq
        if Raw in pkt:
            d["TCP Length"] = len(pkt[Raw].load)
            payload = pkt[Raw].load.decode("utf-8", errors="ignore").lower()
            d["Payload Length"] = len(pkt[Raw].load)
            # HTTP detection
            if "http" in payload or "get " in payload or "post " in payload:
                d["HTTP Request Method"] = "GET" if "get " in payload else "POST"
            # SQL/XSS keywords
            d["HTTP Request URI"] = payload[:500]

    elif UDP in pkt:
        d["Protocol"]           = "UDP"
        d["UDP Source Port"]    = pkt[UDP].sport
        d["UDP Destination Port"] = pkt[UDP].dport
        if DNS in pkt:
            d["has_dns"] = 1
            try:
                d["DNS Query Name"] = pkt[DNS].qd.qname.decode() if pkt[DNS].qd else None
            except Exception:
                pass

    elif ICMP in pkt:
        d["Protocol"] = "ICMP"
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
    # Sadece mevcut kolonlara uygula
    available = [c for c in feature_cols if c in df.columns]
    df[available] = np.log1p(df[available].clip(lower=0))
    df[available] = (df[available] - norm_stats["mean"][available]) / norm_stats["std"][available]
    return df

# ─────────────────────────────────────────────────────────────
# Graph Oluşturma
# ─────────────────────────────────────────────────────────────

def flows_to_graph(flows_df, graph_builder, norm_stats):
    """
    flows_df → normalize → SimpleGraphBuilder → Data
    """
    if flows_df is None or len(flows_df) == 0:
        return None
    try:
        norm_df = apply_normalizer(flows_df, norm_stats)
        graph   = graph_builder.build_graph(norm_df)
        return graph
    except Exception as e:
        log.warning(f"Graph oluşturma hatası: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# Bağlantı Kesme / Açma
# ─────────────────────────────────────────────────────────────

def block_traffic():
    """eth0 üzerinden gelen trafiği kes."""
    log.warning("🚨 SALDIRI TESPİT EDİLDİ — eth0 trafiği kesiliyor!")
    subprocess.run(["iptables", "-I", "FORWARD", "-i", "eth0", "-j", "DROP"], check=False)
    subprocess.run(["iptables", "-I", "FORWARD", "-o", "eth0", "-j", "DROP"], check=False)

def unblock_traffic():
    """eth0 trafiğini geri aç."""
    log.info("✅ eth0 trafiği yeniden açılıyor...")
    subprocess.run(["iptables", "-D", "FORWARD", "-i", "eth0", "-j", "DROP"], check=False)
    subprocess.run(["iptables", "-D", "FORWARD", "-o", "eth0", "-j", "DROP"], check=False)

# ─────────────────────────────────────────────────────────────
# Ana Monitor Sınıfı
# ─────────────────────────────────────────────────────────────

class TGATMonitor:
    def __init__(self):
        self.device      = torch.device("cpu")
        self.buffer      = PacketBuffer(duration=BUFFER_DURATION)
        self.alert_count = 0
        self.blocked     = False
        self.running     = True

        # Model yükle
        log.info("Model yükleniyor...")
        self.model = TGATModel(
            in_dim=16, hidden_dim=128, out_dim=len(CLASSES),
            edge_feat_dim=59, time_dim=32, num_layers=3, dropout=0.0,
        )
        self.model.load_state_dict(
            torch.load(MODEL_DIR / "best_model.pt", map_location=self.device)
        )
        self.model.eval()
        log.info("✓ Model hazır")

        # Normalizer yükle
        with open(MODEL_DIR / "normalization_stats.pkl", "rb") as f:
            self.norm_stats = pickle.load(f)
        log.info("✓ Normalizer hazır")

        # Graph builder
        from graph_builder import SimpleGraphBuilder
        self.graph_builder = SimpleGraphBuilder(node_feature_dim=16, edge_feature_dim=59)
        log.info("✓ Graph builder hazır")

        # FlowBuilder
        from build_flows import FlowBuilder
        self.flow_builders = {ws: FlowBuilder(time_window=ws) for ws in WINDOW_SIZES}
        log.info(f"✓ FlowBuilder hazır: {WINDOW_SIZES}")

    def packet_callback(self, pkt):
        """Scapy'nin her yakaladığı pakette çağrılır."""
        try:
            if IP in pkt or ARP in pkt:
                d = packet_to_dict(pkt)
                self.buffer.add(d)
        except Exception:
            pass

    def run_inference(self):
        """Buffer'daki paketlerden graph oluştur, modeli çalıştır."""
        df = self.buffer.get_dataframe()
        if df is None or len(df) < 5:
            log.debug(f"Yetersiz paket: {len(df) if df is not None else 0}")
            return None, None

        all_flows = []
        for ws, builder in self.flow_builders.items():
            try:
                flows = builder.build_flows(df)
                if flows is not None and len(flows) > 0:
                    all_flows.append(flows)
            except Exception as e:
                log.debug(f"Flow oluşturma hatası (ws={ws}): {e}")

        if not all_flows:
            log.debug("Hiç flow oluşturulamadı")
            return None, None

        combined = pd.concat(all_flows, ignore_index=True)
        graph    = flows_to_graph(combined, self.graph_builder, self.norm_stats)

        if graph is None:
            return None, None

        with torch.no_grad():
            bg     = Batch.from_data_list([graph]).to(self.device)
            logits = self.model(
                bg.x, bg.edge_index, bg.node_time,
                bg.edge_time, bg.edge_attr, bg.batch
            )
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return probs, CLASSES

    def inference_loop(self):
        """Her CAPTURE_INTERVAL saniyede bir inference çalıştır."""
        log.info(f"Inference döngüsü başladı (her {CAPTURE_INTERVAL}sn)")
        time.sleep(CAPTURE_INTERVAL)  # buffer dolsun

        while self.running:
            t0 = time.perf_counter()

            probs, classes = self.run_inference()

            if probs is not None:
                top_idx  = int(np.argmax(probs))
                top_prob = float(probs[top_idx])
                top_cls  = classes[top_idx]

                # Sonucu logla
                attack_probs = {
                    cls: round(float(p), 3)
                    for cls, p in zip(classes, probs)
                    if cls != "BENIGN" and p > 0.05
                }
                log.info(
                    f"Tahmin: {top_cls} ({top_prob:.3f}) | "
                    f"Paket buffer: {self.buffer.size()} | "
                    f"Şüpheli: {attack_probs}"
                )

                # Saldırı kontrolü
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

            elapsed = time.perf_counter() - t0
            sleep_t = max(0, CAPTURE_INTERVAL - elapsed)
            time.sleep(sleep_t)

    def start(self):
        log.info(f"🔍 İzleme başladı — interface: {INTERFACE}")
        log.info(f"   Pencereler: {WINDOW_SIZES}sn")
        log.info(f"   Inference aralığı: {CAPTURE_INTERVAL}sn")
        log.info(f"   Saldırı eşiği: {CONFIDENCE_THR} (min {MIN_ALERTS} ardışık tespit)")

        # Inference thread
        t = threading.Thread(target=self.inference_loop, daemon=True)
        t.start()

        # Paket yakalama (ana thread)
        try:
            sniff(
                iface=INTERFACE,
                prn=self.packet_callback,
                store=False,
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
        print("HATA: Bu script root yetkisi gerektiriyor.")
        print("Kullanım: sudo python3 monitor.py")
        sys.exit(1)

    monitor = TGATMonitor()
    monitor.start()
