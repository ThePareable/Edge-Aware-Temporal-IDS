import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


class TimeEncoding(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.lin = nn.Linear(1, time_dim)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        dt = delta_t.unsqueeze(-1)
        return torch.tanh(self.lin(dt))


class TGATLayer(MessagePassing):
    def __init__(self, in_dim: int, time_dim: int, out_dim: int, edge_feat_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        self.lin_h = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_t = nn.Linear(time_dim, out_dim, bias=False)
        self.edge_lin = nn.Linear(edge_feat_dim, out_dim, bias=False)

        self.attn_vec = nn.Parameter(torch.empty(4 * out_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, node_time, edge_time, edge_attr, time_encoder: TimeEncoding):
        h = self.lin_h(x)
        out = self.propagate(
            edge_index,
            x=h,
            node_time=node_time,
            edge_time=edge_time,
            edge_attr=edge_attr,
            time_encoder=time_encoder,
            size=(h.size(0), h.size(0)),
        )
        return out

    def message(self, x_j, x_i, node_time_i, edge_time, edge_attr, time_encoder, index, ptr, size_i):
        delta_t = node_time_i - edge_time
        t_emb = time_encoder(delta_t)
        t_emb_proj = self.lin_t(t_emb)

        edge_emb = self.edge_lin(edge_attr)

        cat = torch.cat([x_i, x_j, t_emb_proj, edge_emb], dim=-1)
        e_ij = (cat * self.attn_vec).sum(dim=-1)
        e_ij = self.leaky_relu(e_ij)

        alpha = softmax(e_ij, index)
        alpha = self.dropout(alpha)

        msg = x_j + edge_emb
        return msg * alpha.unsqueeze(-1)


class TGATModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, edge_feat_dim: int,
                 time_dim: int = 16, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.time_encoder = TimeEncoding(time_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TGATLayer(hidden_dim, time_dim, hidden_dim, edge_feat_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, node_time, edge_time, edge_attr, batch=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, node_time, edge_time, edge_attr, self.time_encoder)
            h = self.activation(h)
            h = self.dropout(h)

        if batch is None:
            g_emb = h.mean(dim=0, keepdim=True)
        else:
            g_emb = global_mean_pool(h, batch)

        return self.classifier(g_emb)
