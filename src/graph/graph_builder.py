from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass
class GraphMeta:
    node_map: Dict[str, int]
    window_id: int
    window_size: float


def _parse_node_str(s: str) -> Tuple[str, int]:
    # "185.125.190.18:80" -> ("185.125.190.18", 80)
    ip, port = s.rsplit(":", 1)
    return ip, int(port)


class SimpleGraphBuilder:
    """
    TGAT-friendly builder:
    - node features are NOT zeros (basic traffic stats + port category flags)
    - node_time is computed (max end_time per node)
    - edge_attr uses numeric columns (either provided or default list of 59)
    - edge_time uses start_time
    """

    def __init__(self, edge_feature_cols: Optional[List[str]] = None, node_feature_dim: int = 16):
        self.node_feature_dim = node_feature_dim

        if edge_feature_cols is not None:
            self.edge_feature_cols = edge_feature_cols
        else:
            # default 59 list (from earlier stable version)
            self.edge_feature_cols = [
                "duration","packet_count","total_bytes","mean_packet_size","std_packet_size",
                "min_packet_size","max_packet_size","packet_rate","byte_rate","mean_iat","std_iat",
                "syn_count","ack_count","fin_count","rst_count",
                "syn_ratio","ack_ratio","fin_ratio","rst_ratio",
                "tcp_window_size_mean","tcp_window_size_std",
                "ip_ttl_mean","ip_ttl_std","ip_dscp_mean","ip_dscp_std","ip_dscp_unique",
                "has_fragmented_packets","fragment_count","has_fragments",
                "fragment_offset_max","tcp_seq_std","tcp_seq_is_random",
                "tcp_payload_mean","tcp_payload_total","tcp_payload_ratio",
                "has_http","has_sql_keywords","has_script_tags",
                "http_response_200_ratio","http_response_4xx_ratio","http_response_5xx_ratio","http_response_diversity",
                "has_sqlmap","has_nikto","has_burp","has_normal_browser","user_agent_changes",
                "http_content_length_mean","http_content_length_total",
                "icmp_type_mode","icmp_type_8_ratio","icmp_type_3_ratio","icmp_type_diversity",
                "has_dns","dns_type_A_ratio","dns_type_TXT_ratio","dns_type_diversity",
                "payload_length_var",
            ]

    def build_graph(
        self,
        flows_df: pd.DataFrame,
        window_id: int,
        window_size: float,
        label_vec=None,
    ) -> Tuple[Data, GraphMeta]:

        if label_vec is None:
            if "label_vec" in flows_df.columns:
                label_vec = flows_df.iloc[0]["label_vec"]
            else:
                raise ValueError("label_vec missing (pass it from prepare_multilabel.py)")

        node_map = self._create_node_map(flows_df)
        x = self._create_node_features(flows_df, node_map)
        node_time = self._compute_node_times(flows_df, node_map)

        edge_index, edge_attr, edge_time = self._create_edges(flows_df, node_map)

        y = torch.tensor(label_vec, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_time,
            node_time=node_time,
            y=y,
        )
        meta = GraphMeta(node_map=node_map, window_id=window_id, window_size=window_size)
        return data, meta

    def _create_node_map(self, df: pd.DataFrame) -> Dict[str, int]:
        nodes = pd.unique(pd.concat([df["src_node"], df["dst_node"]], axis=0))
        nodes = [n for n in nodes if isinstance(n, str)]
        # stable ordering
        nodes = sorted(nodes, key=lambda s: _parse_node_str(s))
        return {n: i for i, n in enumerate(nodes)}

    def _create_edges(self, df: pd.DataFrame, node_map: Dict[str, int]):
        src_idx = df["src_node"].map(node_map)
        dst_idx = df["dst_node"].map(node_map)

        # drop NA edges safely
        ok = src_idx.notna() & dst_idx.notna()
        src = src_idx[ok].astype(np.int64).values
        dst = dst_idx[ok].astype(np.int64).values
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # edge_attr
        if self.edge_feature_cols:
            E = df.loc[ok, self.edge_feature_cols].to_numpy(dtype=np.float32)
            E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            E = np.zeros((ok.sum(), 1), dtype=np.float32)
        edge_attr = torch.tensor(E, dtype=torch.float32)

        edge_time_np = df.loc[ok, "start_time"].to_numpy(dtype=np.float32)
        edge_time_np = np.nan_to_num(edge_time_np, nan=0.0, posinf=0.0, neginf=0.0)
        edge_time = torch.tensor(edge_time_np, dtype=torch.float32)

        return edge_index, edge_attr, edge_time

    def _create_node_features(self, df: pd.DataFrame, node_map: Dict[str, int]) -> torch.Tensor:
        N = len(node_map)
        feats = np.zeros((N, self.node_feature_dim), dtype=np.float32)

        stats = defaultdict(lambda: {
            "in_packets": 0.0, "out_packets": 0.0,
            "in_bytes": 0.0, "out_bytes": 0.0,
            "in_conns": 0.0, "out_conns": 0.0,
            "syn": 0.0, "rst": 0.0,
        })

        for _, flow in df.iterrows():
            s = flow["src_node"]; d = flow["dst_node"]
            if not isinstance(s, str) or not isinstance(d, str):
                continue

            pc = float(flow.get("packet_count", 0.0))
            tb = float(flow.get("total_bytes", 0.0))
            syn = float(flow.get("syn_count", 0.0))
            rst = float(flow.get("rst_count", 0.0))

            stats[s]["out_packets"] += pc
            stats[s]["out_bytes"] += tb
            stats[s]["out_conns"] += 1.0
            stats[s]["syn"] += syn

            stats[d]["in_packets"] += pc
            stats[d]["in_bytes"] += tb
            stats[d]["in_conns"] += 1.0
            stats[d]["rst"] += rst

        for node_str, idx in node_map.items():
            ip, port = _parse_node_str(node_str)
            st = stats[node_str]
            outc = st["out_conns"]; inc = st["in_conns"]
            activity_ratio = outc / (inc + 1.0)

            base = [
                st["out_packets"], st["in_packets"],
                st["out_bytes"], st["in_bytes"],
                outc, inc,
                1.0 if port < 1024 else 0.0,
                1.0 if port == 22 else 0.0,
                1.0 if port in (80, 443) else 0.0,
                st["syn"], st["rst"],
                activity_ratio,
            ]
            if len(base) < self.node_feature_dim:
                base += [0.0] * (self.node_feature_dim - len(base))
            feats[idx] = np.array(base[: self.node_feature_dim], dtype=np.float32)

        return torch.tensor(feats, dtype=torch.float32)

    def _compute_node_times(self, df: pd.DataFrame, node_map: Dict[str, int]) -> torch.Tensor:
        # node_time = max end_time touching that node
        N = len(node_map)
        node_t = np.zeros((N,), dtype=np.float32)

        for _, flow in df.iterrows():
            s = flow["src_node"]; d = flow["dst_node"]
            if not isinstance(s, str) or not isinstance(d, str):
                continue
            t = float(flow.get("end_time", flow.get("start_time", 0.0)))
            si = node_map.get(s); di = node_map.get(d)
            if si is not None: node_t[si] = max(node_t[si], t)
            if di is not None: node_t[di] = max(node_t[di], t)

        node_t = np.nan_to_num(node_t, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(node_t, dtype=torch.float32)
