import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SnapshotGCNEncoder(nn.Module):
    """
    Encodes a single snapshot graph into a fixed-dim graph embedding.
    """
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        assert layers >= 1
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)  # [B, hidden]
        return g

class GCLSTMClassifier(nn.Module):
    """
    GCN per snapshot -> LSTM over time -> classification.
    """
    def __init__(self, in_dim: int, hidden: int, gcn_layers: int, lstm_hidden: int,
                 num_classes: int, dropout: float):
        super().__init__()
        self.encoder = SnapshotGCNEncoder(in_dim, hidden, gcn_layers, dropout)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, seq_batch):
        """
        seq_batch: list length T, each element is a torch_geometric Batch (same B across T)
        Returns logits [B, C]
        """
        T = len(seq_batch)
        assert T >= 1
        emb_list = []
        for bt in seq_batch:
            g = self.encoder(bt.x, bt.edge_index, bt.batch)  # [B, hidden]
            emb_list.append(g)
        E = torch.stack(emb_list, dim=1)  # [B, T, hidden]
        out, _ = self.lstm(E)             # [B, T, lstm_hidden]
        last = out[:, -1, :]              # last step
        logits = self.head(last)          # [B, C]
        return logits
