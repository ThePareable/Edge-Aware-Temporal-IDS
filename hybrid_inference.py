import torch, pickle, sys, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from scapy.all import sniff, IP, TCP, Ether

# --- KONFİGÜRASYON ---
ADMIN_MAC = "aa:bb:cc:dd:ee:ff"  # <--- BURAYA KENDİ PC MAC ADRESİNİ YAZ
VICTIM_IP = "192.168.10.80"
CLASSES = ['ARP-SPOOF','BENIGN','DDOS-ICMP','EXPLOITING-FTP','FTP-BRUTE-FORCE','PORT-SCANNING','SQL-INJECTION','SSH-BRUTE-FORCE','SYN-FLOOD','XSS']

sys.path.insert(0, '/home/raspberrypi/tez')
from build_flows import FlowBuilder
from graph_builder import SimpleGraphBuilder
from monitor import TGATModel

# --- MODEL TANIMLARI ---
class SnapshotGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)] + [GCNConv(hidden, hidden) for _ in range(layers-1)])
        self.dropout = dropout
    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=False)
        return global_mean_pool(h, batch)

class GCLSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden, gcn_layers, lstm_hidden, num_classes, dropout):
        super().__init__()
        self.encoder = SnapshotGCNEncoder(in_dim, hidden, gcn_layers, dropout)
        self.lstm = nn.LSTM(hidden, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(lstm_hidden), nn.Dropout(dropout), nn.Linear(lstm_hidden, num_classes))
    def forward(self, seq_batch):
        embs = []
        for bt in seq_batch:
            # HATAYI ÇÖZEN KISIM: 16 -> 60 Padding
            x = bt.x
            if x.shape[1] < 60:
                pad = torch.zeros((x.shape[0], 60 - x.shape[1])).to(x.device)
                x = torch.cat([x, pad], dim=1)
            embs.append(self.encoder(x, bt.edge_index, bt.batch))
        E = torch.stack(embs, dim=1)
        out, _ = self.lstm(E)
        return self.head(out[:, -1, :])

# --- YÜKLEME ---
C = len(CLASSES)
tgat = TGATModel(in_dim=16, hidden_dim=256, out_dim=C, edge_feat_dim=60, time_dim=64, num_layers=3, dropout=0.0)
tgat.load_state_dict(torch.load('best_model.pt', map_location='cpu'))
tgat.eval()

gclstm = GCLSTMClassifier(in_dim=60, hidden=128, gcn_layers=2, lstm_hidden=128, num_classes=C, dropout=0.0)
gclstm.load_state_dict(torch.load('gclstm_multilabel_best.pt', map_location='cpu'))
gclstm.eval()

stacker = nn.Linear(2*C, C)
stacker.load_state_dict(torch.load('hybrid_stacker.pt', map_location='cpu'))
stacker.eval()

# --- PACKET HANDLER (MAC FILTER DAHİL) ---
captured_packets = []
def packet_handler(pkt):
    if not pkt.haslayer(IP): return
    
    # Kendi SSH trafiğini ele
    if pkt.haslayer(Ether) and pkt.haslayer(TCP):
        is_ssh = (pkt[TCP].sport == 22 or pkt[TCP].dport == 22)
        if is_ssh and (pkt[Ether].src.lower() == ADMIN_MAC.lower() or pkt[Ether].dst.lower() == ADMIN_MAC.lower()):
            return
            
    # Sadece Victim IP odaklı topla
    if pkt[IP].src == VICTIM_IP or pkt[IP].dst == VICTIM_IP:
        captured_packets.append(pkt)

print(f"[*] Dinleme başlıyor... (MAC Filtresi: {ADMIN_MAC})")
sniff(iface=["wlan0", "eth0"], prn=packet_handler, timeout=15, store=0)

# --- INFERENCE ---
if len(captured_packets) < 5:
    print("[!] Yeterli paket yakalanamadı.")
    sys.exit()

gb = SimpleGraphBuilder()
windows = [1.0, 5.0, 30.0]
seq_batch = []

with torch.no_grad():
    for w in windows:
        fb = FlowBuilder(time_window=w)
        flows = fb.build_from_packets(captured_packets)
        bg = gb.build_graph(flows)
        # TGAT için dummy time_input (Eğitimdeki yapıya göre)
        bg.time = torch.zeros(bg.num_nodes).float() 
        seq_batch.append(bg)

    # Model Outputs
    # TGAT için son window'u kullanıyoruz
    last_bg = seq_batch[-1]
    # TGAT node feature uyumu (16 dim)
    tgat_out = tgat(last_bg.x[:, :16], last_bg.edge_index, last_bg.edge_attr, last_bg.time, last_bg.batch)
    tgat_probs = torch.sigmoid(tgat_out)

    # GCLSTM sequence
    gclstm_out = gclstm(seq_batch)
    gclstm_probs = torch.sigmoid(gclstm_out)

    # Hybrid Stacker
    combined = torch.cat([tgat_probs, gclstm_probs], dim=1)
    final_probs = torch.sigmoid(stacker(combined)).squeeze().numpy()

    # Sonuçları Bas
    print("\n--- TESPİT SONUÇLARI ---")
    res = {CLASSES[i]: final_probs[i] for i in range(len(CLASSES))}
    for c, p in sorted(res.items(), key=lambda x: x[1], reverse=True):
        print(f"{c:20}: {p:.4f}")

