import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    ip: str
    port: int

    def __str__(self):
        return f"{self.ip}:{self.port}"

    def __repr__(self):
        return f"Node({self.ip}:{self.port})"


def extract_timestamp(row: pd.Series) -> float:
    timestamp = row.get("Time", 0.0)
    if pd.isna(timestamp):  # type: ignore
        return 0.0
    return float(np.float64(timestamp))


def extract_protocol(row: pd.Series) -> str:
    protocol = row.get("Protocol", "UNKNOWN")
    if pd.isna(protocol):  # type: ignore
        return "UNKNOWN"
    return str(protocol).upper()


def get_time_window_id(timestamp: float, window_size: float) -> int:
    return int(timestamp // window_size)


def are_in_same_window(t1: float, t2: float, window_size: float) -> bool:
    return get_time_window_id(t1, window_size) == get_time_window_id(t2, window_size)


def extract_node(row: pd.Series, direction: str) -> Node:
    if direction == "src":
        ip = row.get("IP Source", "0.0.0.0")
        port = row.get("TCP Source Port")
        if pd.isna(port):  # type: ignore
            port = row.get("UDP Source Port")
        if pd.isna(port):  # type: ignore
            protocol = extract_protocol(row)
            if protocol == "ICMP":
                port = -1
            elif protocol == "ARP":
                port = -2
            else:
                port = 0
    else:
        ip = row.get("IP Destination", "0.0.0.0")
        port = row.get("TCP Destination Port")
        if pd.isna(port):  # type: ignore
            port = row.get("UDP Destination Port")
        if pd.isna(port):  # type: ignore
            protocol = extract_protocol(row)
            if protocol == "ICMP":
                port = -1
            elif protocol == "ARP":
                port = -2
            else:
                port = 0

    try:
        port = int(float(port)) if not pd.isna(port) else 0  # type: ignore
    except Exception:
        port = 0

    return Node(ip=str(ip), port=port)
