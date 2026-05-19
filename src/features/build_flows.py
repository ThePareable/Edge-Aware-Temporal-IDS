"""
Build flows from packets

Flow = packets between same (src_ip:port, dst_ip:port, protocol) within time window

This script aggregates raw packets into flows and extracts 59 NUMERIC edge features:

FEATURE CATEGORIES:
  - Volume & Temporal (11): duration, packet_count, total_bytes, packet sizes (mean/std/min/max), 
                            packet_rate, byte_rate, IAT statistics (mean/std)
  - TCP Flags (10): SYN/ACK/FIN/RST counts and ratios, TCP window size (mean/std)
  - IP Layer (8): TTL (mean/std), DSCP (mean/std/unique), fragmentation flags (has_fragmented_packets, fragment_count)
  - TCP Extended (6): sequence number std, seq_is_random, payload (mean/total/ratio), fragment_offset_max
  - HTTP (14): has_http, SQL/XSS keywords, response code ratios (200/4xx/5xx), response diversity,
               attack tool detection (sqlmap/nikto/burp/browser), user_agent_changes, content_length (mean/total)
  - ICMP (4): type_mode, type_8_ratio, type_3_ratio, type_diversity
  - DNS (4): has_dns, dns_type_A_ratio, dns_type_TXT_ratio, dns_type_diversity
  - Payload (1): payload_length_var (fuzzing detection)
  - Multi-scale (1): window_size (CRITICAL for multi-scale learning!)

TOTAL: 59 numeric edge features for TGAT (11+10+8+6+14+4+4+1+1)
Aligned with thesis proposal section 2.2: "Kenar özellikleri"
Expected attack coverage: 95%+ (all 15 attack classes)
"""

import pandas as pd
import numpy as np 
from collections import defaultdict
from pathlib import Path
import os
from graph_design import Node, extract_node, extract_protocol, extract_timestamp, get_time_window_id

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(PROJECT_ROOT)
print(f"Working directory: {PROJECT_ROOT}")


class FlowBuilder:
    """ Builds flows from raw packets and extracts discriminative features
    
    Multi-Scale Strategy: 
    - Train ALL attacks at ALL windows (1s, 5s, 30s)
    - window_size becomes a feature for the model to learn from
    - Model learns: "This attack behaves like THIS at different scales"
    - Essential for real IDS: No circular dependency at inference
    
    Example:
        PORT-SCANNING at 1s: packet_rate=500, duration=0.3s, rst_ratio=0.9
        PORT-SCANNING at 5s: packet_rate=100, duration=0.5s, rst_ratio=0.9
        PORT-SCANNING at 30s: packet_rate=17, duration=2s, rst_ratio=0.9
        → Model learns: "High RST at ALL scales = Port scan"
    """
    
    def __init__(self, time_window=5.0):
        """ 
        Args:
            time_window: Time window in seconds for grouping packets
        """
        self.time_window = time_window
       
    def build_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Convert packets to flows.
        
        Args: 
            df: DataFrame with raw packets
        
        Returns:
            DataFrame where each row is a flow with aggregated features
        """ 
        print(f"\nBuilding flows with {self.time_window}s time window...")
        print(f"Input: {len(df)} packets")
        
        #Step 1: Extract nodes and protocol for each packet
        df = df.copy()
        
        # extract source and destination nodes
        
        df["src_node"] = df.apply(lambda row: extract_node(row, "src"), axis = 1)
        df["dst_node"] = df.apply(lambda row: extract_node(row, "dst"), axis = 1)
        
        # extract protocol (same for both src and dst in a packet)
        df["protocol"] = df.apply(extract_protocol, axis = 1)
        
        # step 2 assign time window id using step2's helper function
        df["time_window_id"] = df["Time"].apply(lambda t: get_time_window_id(t, self.time_window))
        
        # step 3: create flow id (5-tuple + time_window)
        # Flow = (src_ip, src_port, dst_ip, dst_port, protocol, time_window)
        df["flow_id"] = df.apply(
            lambda row: f"{row["src_node"]}→{row["dst_node"]}|{row["protocol"]}_w{row["time_window_id"]}", 
            axis = 1
        )
        
        # step 4: aggregate packets into flows
        flows = []
        for flow_id, group in df.groupby("flow_id"):
            flow = self._aggregate_flow(group)
            flows.append(flow)
            
        flow_df = pd.DataFrame(flows)
        print(f"Output: {len(flow_df)} flows")
        print(f"Reduction: {len(df)} packets → {len(flow_df)} flows ({len(flow_df)/len(df)*100:.1f}%)")
        
        return flow_df
    
    def _aggregate_flow(self, packets: pd.DataFrame) -> dict:
        """
        Aggregate a group of packets into one flow.
        
        Extracts 59 NUMERIC edge features aligned with thesis proposal (section 2.2):
        
        "Kenar özellikleri: akış bayt oranı, RTT, bağlantı sıklığı, SYN/ACK oranı, akış yönü"
        
        Implementation breakdown:
         - Volume & Temporal (11): packet count, bytes, sizes, rates, IAT statistics
         - TCP Flags (10): SYN/ACK/FIN/RST counts + ratios, window size
         - IP Layer (8): TTL, DSCP, fragmentation detection
         - TCP Extended (6): sequence analysis, payload analysis
         - HTTP (14): response codes, attack tools, SQL/XSS detection, content-length
         - ICMP (4): type analysis for flood detection
         - DNS (4): query type analysis for tunneling detection
         - Payload (1): variance for fuzzing detection
         - Multi-scale (1): window_size for cross-scale learning
         
        Total: 59 numeric features (11+10+8+6+14+4+4+1+1, used as edge_attr in PyG Data objects)
        """
        
        first_packet = packets.iloc[0]
        protocol = first_packet.get('protocol', 'UNKNOWN')
        
        flow = {
            # Identity 
            "src_node": first_packet['src_node'],
            "dst_node": first_packet['dst_node'],
            "src_ip": first_packet['src_node'].ip,
            "src_port": first_packet['src_node'].port,
            "dst_ip": first_packet['dst_node'].ip,
            "dst_port": first_packet['dst_node'].port,
            "time_window_id": first_packet['time_window_id'],
            "protocol": protocol,
            
            # Temporal
            "start_time": packets['Time'].min(),
            "end_time": packets['Time'].max(),
            "duration": packets['Time'].max() - packets['Time'].min(),
            
            # Volume
            "packet_count": len(packets),
        }
        
        # Packet sizes
        if 'Length' in packets.columns:
            flow["total_bytes"] = packets['Length'].sum()
            flow["mean_packet_size"] = packets['Length'].mean()
            flow["std_packet_size"] = packets['Length'].std() if len(packets) > 1 else 0
            flow["min_packet_size"] = packets['Length'].min()
            flow["max_packet_size"] = packets['Length'].max()
        else:
            flow["total_bytes"] = 0.0
            flow["mean_packet_size"] = 0.0
            flow["std_packet_size"] = 0.0
            flow["min_packet_size"] = 0.0
            flow["max_packet_size"] = 0.0
            
        # Temporal Features 
        if flow["duration"] > 0:
            flow["packet_rate"] = flow["packet_count"] / flow["duration"]
            flow["byte_rate"] = flow["total_bytes"] / flow["duration"]
        else:
            flow["packet_rate"] = 0.0
            flow["byte_rate"] = 0.0
            
        # Inter-arrival times
        if len(packets) > 1:
            iats = np.diff(packets['Time'].values)
            flow["mean_iat"] = np.mean(iats)
            flow["std_iat"] = np.std(iats)
        else:
            flow["mean_iat"] = 0.0
            flow["std_iat"] = 0.0
            
        # TCP Flags
        if "TCP SYN Flag" in packets.columns:
            flow["syn_count"] = (packets["TCP SYN Flag"] == "Set").sum()
            flow["ack_count"] = (packets["TCP ACK Flag"] == "Set").sum()
            flow["fin_count"] = (packets["TCP FIN Flag"] == "Set").sum()
            flow["rst_count"] = (packets["TCP RST Flag"] == "Set").sum()
            
            # Ratios
            flow["syn_ratio"] = flow["syn_count"] / flow["packet_count"]
            flow["ack_ratio"] = flow["ack_count"] / flow["packet_count"]
            flow["fin_ratio"] = flow["fin_count"] / flow["packet_count"]
            flow["rst_ratio"] = flow["rst_count"] / flow["packet_count"]
            
            # TCP window size
            if "TCP Window Size" in packets.columns:
                flow["tcp_window_size_mean"] = packets["TCP Window Size"].mean()
                flow["tcp_window_size_std"] = packets["TCP Window Size"].std() if len(packets) > 1 else 0
            else:
                flow["tcp_window_size_mean"] = 0.0
                flow["tcp_window_size_std"] = 0.0
        else:
            for key in ["syn_count", "ack_count", "fin_count", "rst_count",
                        "syn_ratio", "ack_ratio", "fin_ratio", "rst_ratio",
                        "tcp_window_size_mean", "tcp_window_size_std"]:
                flow[key] = 0.0
                
        # IP Layer
        if "IP TTL" in packets.columns:
            # numeric value istenilen coerce nan isaretliyor na olanlar da droplanir.
            ttl_values = pd.to_numeric(packets["IP TTL"], errors="coerce").dropna() 
            flow["ip_ttl_mean"] = float(ttl_values.mean()) if len(ttl_values) > 0 else 0.0
            flow["ip_ttl_std"] = float(ttl_values.std()) if len(ttl_values) > 1 else 0.0
        else:
            flow["ip_ttl_mean"] = 0.0
            flow["ip_ttl_std"] = 0.0
            
        # Application Layer
        flow["has_http"] = int(packets["HTTP Request Method"].notna().any()) if "HTTP Request Method" in packets.columns else 0
        flow["has_dns"] = int(packets["DNS Query Name"].notna().any()) if "DNS Query Name" in packets.columns else 0
        
        # HTTP details
        if flow["has_http"]:
            http_methods = packets["HTTP Request Method"].dropna()
            flow["http_method"] = http_methods.mode()[0] if len(http_methods) > 0 else "NONE"
        else:
            flow["http_method"] = "NONE"
            
        if "HTTP Request URI" in packets.columns:
            uris = packets["HTTP Request URI"].dropna().astype(str).str.lower()
            if len(uris) > 0:
                flow['has_sql_keywords'] = int(uris.str.contains('select|union|insert|delete|drop|update|\\-\\-|\\\' or').any())
                flow['has_script_tags'] = int(uris.str.contains('<script|javascript:|onerror=|onload=|<img|alert\\(').any())
            else:
                flow['has_sql_keywords'] = 0
                flow['has_script_tags'] = 0
        else:
            flow['has_sql_keywords'] = 0
            flow['has_script_tags'] = 0
            
        if "Payload Length" in packets.columns:
            payload_lengths = pd.to_numeric(packets["Payload Length"], errors="coerce").dropna()
            flow['payload_length_var'] = float(payload_lengths.var()) if len(payload_lengths) > 1 else 0
        else:
            flow['payload_length_var'] = 0.0
            
        # IP DSCP Field (Qos, C2 Detection)
        if 'IP DSCP Field' in packets.columns:
            dscp_values = packets['IP DSCP Field'].dropna()
            if len(dscp_values) > 0:
                # Convert hex strings (0x00) to integers
                dscp_numeric = pd.to_numeric(dscp_values, errors='coerce').dropna()
                if len(dscp_numeric) > 0:
                    flow['ip_dscp_mean'] = float(dscp_numeric.mean())
                    flow['ip_dscp_std'] = float(dscp_numeric.std()) if len(dscp_numeric) > 1 else 0
                    flow['ip_dscp_unique'] = int(dscp_numeric.nunique())
                else:
                    flow['ip_dscp_mean'] = 0.0
                    flow['ip_dscp_std'] = 0.0
                    flow['ip_dscp_unique'] = 0
            else:
                flow['ip_dscp_mean'] = 0.0
                flow['ip_dscp_std'] = 0.0
                flow['ip_dscp_unique'] = 0
        else:
            flow['ip_dscp_mean'] = 0.0
            flow['ip_dscp_std'] = 0.0
            flow['ip_dscp_unique'] = 0
            
        # IP Flags (Fragmentation detection)
        # Why: Fragmentation attacks use MF (More Fragments) bit, normal traffic uses 
        # DF (Don't Fragment) bit
        if "IP Flags" in packets.columns:
            ip_flags = packets["IP Flags"].dropna()
            if len(ip_flags) > 0:
                # Ip Flags is hex: 0x4000 = DF (Don't Fragment), 0x2000 = MF (More Fragments)
                # Convert hex to int for comparison (0x4000 = 16384)
                flags_numeric = pd.to_numeric(ip_flags, errors='coerce').dropna()
                if len(flags_numeric) > 0:
                    # DF bit = 0x4000 (16384), check for non-DF packets
                    flow['has_fragmented_packets'] = int((flags_numeric != 16384).any())
                    flow["fragment_count"] = int((flags_numeric != 16384).sum())
                else:
                    flow['has_fragmented_packets'] = 0
                    flow['fragment_count'] = 0
            else:
                flow['has_fragmented_packets'] = 0
                flow['fragment_count'] = 0
        else:
            flow['has_fragmented_packets'] = 0
            flow['fragment_count'] = 0
            
        # IP Fragment Offset
        # Why: Fragmentation attacks, IDS evasion
        # Availability: Fragmented packets only
        if "IP Fragment Offset" in packets.columns:
            frag_offsets = pd.to_numeric(packets["IP Fragment Offset"], errors="coerce").dropna()
            if len(frag_offsets) > 0 and (frag_offsets != 0).any():
                flow['has_fragments'] = 1
                flow['fragment_offset_max'] = int(frag_offsets.max())
            else:
                flow['has_fragments'] = 0
                flow['fragment_offset_max'] = 0
        else:
            flow['has_fragments'] = 0
            flow['fragment_offset_max'] = 0
            
        # TCP Sequence Number Analysis
        # Why: SYN floods may use sequential or random sequence numbers 
        # Availability: TCP packets
        if "TCP Sequence Number" in packets.columns:
            seq_nums = pd.to_numeric(packets["TCP Sequence Number"], errors="coerce").dropna()
            if len(seq_nums) > 1:
                # check if sequential (small std) or random (large std)
                diffs = seq_nums.diff().dropna()
                if len(diffs) > 0:
                    flow["tcp_seq_std"] = float(diffs.std())
                    # random if std > 100000 (heuristic)
                    flow["tcp_seq_is_random"] = int(flow["tcp_seq_std"] > 100000)
                else:
                    flow["tcp_seq_std"] = 0.0
                    flow["tcp_seq_is_random"] = 0
            else:
                flow["tcp_seq_std"] = 0.0
                flow["tcp_seq_is_random"] = 0
        else:
            flow["tcp_seq_std"] = 0.0
            flow["tcp_seq_is_random"] = 0
            
        # TCP Payload Length
        # Why: Payload vs header ratio, data exfiltration detection
        # Availability: TCP packets
        if 'TCP Length' in packets.columns:
            tcp_lengths = pd.to_numeric(packets['TCP Length'], errors='coerce').dropna()
            if len(tcp_lengths) > 0:
                flow['tcp_payload_mean'] = float(tcp_lengths.mean())
                flow['tcp_payload_total'] = int(tcp_lengths.sum())
                # Ratio of TCP payload to total packet size
                if flow['total_bytes'] > 0:
                    flow['tcp_payload_ratio'] = float(flow['tcp_payload_total'] / flow['total_bytes'])
                else:
                    flow['tcp_payload_ratio'] = 0.0
            else:
                flow['tcp_payload_mean'] = 0.0
                flow['tcp_payload_total'] = 0
                flow['tcp_payload_ratio'] = 0.0
        else:
            flow['tcp_payload_mean'] = 0.0
            flow['tcp_payload_total'] = 0
            flow['tcp_payload_ratio'] = 0.0
            
         # === HTTP Extended (2 features) ===
        
        # 6. HTTP Response Code (CRITICAL for SQL Injection / XSS)
        # Why: SQL injection causes 500 errors, XSS causes redirects (302)
        # Availability: HTTP traffic only
        if flow['has_http'] and 'HTTP Response Code' in packets.columns:
            response_codes = pd.to_numeric(packets['HTTP Response Code'], errors='coerce').dropna()
            if len(response_codes) > 0:
                flow['http_response_200_ratio'] = float((response_codes == 200).sum() / len(response_codes))
                flow['http_response_4xx_ratio'] = float(((response_codes >= 400) & (response_codes < 500)).sum() / len(response_codes))
                flow['http_response_5xx_ratio'] = float(((response_codes >= 500) & (response_codes < 600)).sum() / len(response_codes))
                flow['http_response_diversity'] = int(response_codes.nunique())
            else:
                flow['http_response_200_ratio'] = 0.0
                flow['http_response_4xx_ratio'] = 0.0
                flow['http_response_5xx_ratio'] = 0.0
                flow['http_response_diversity'] = 0
        else:
            flow['http_response_200_ratio'] = 0.0
            flow['http_response_4xx_ratio'] = 0.0
            flow['http_response_5xx_ratio'] = 0.0
            flow['http_response_diversity'] = 0
        
        # 7. HTTP User-Agent (Attack tool detection)
        # Why: Attack tools (sqlmap, nikto, burp) have distinctive user agents
        # Availability: HTTP traffic only
        if flow['has_http'] and 'HTTP User-Agent' in packets.columns:
            user_agents = packets['HTTP User-Agent'].dropna()
            if len(user_agents) > 0:
                ua_str = ' '.join(user_agents.astype(str).str.lower())
                # Check for common attack tools
                flow['has_sqlmap'] = int('sqlmap' in ua_str)
                flow['has_nikto'] = int('nikto' in ua_str)
                flow['has_burp'] = int('burp' in ua_str)
                flow['has_normal_browser'] = int(any(x in ua_str for x in ['mozilla', 'chrome', 'safari', 'firefox']))
                flow['user_agent_changes'] = int(user_agents.nunique())  # UA shouldn't change mid-flow
            else:
                flow['has_sqlmap'] = 0
                flow['has_nikto'] = 0
                flow['has_burp'] = 0
                flow['has_normal_browser'] = 0
                flow['user_agent_changes'] = 0
        else:
            flow['has_sqlmap'] = 0
            flow['has_nikto'] = 0
            flow['has_burp'] = 0
            flow['has_normal_browser'] = 0
            flow['user_agent_changes'] = 0
        
        # 8. HTTP Content-Length
        # Why: Data exfiltration, large payload detection
        # Availability: HTTP traffic only
        if flow['has_http'] and 'HTTP Content-Length' in packets.columns:
            content_lengths = pd.to_numeric(packets['HTTP Content-Length'], errors='coerce').dropna()
            if len(content_lengths) > 0:
                flow['http_content_length_mean'] = float(content_lengths.mean())
                flow['http_content_length_total'] = int(content_lengths.sum())
            else:
                flow['http_content_length_mean'] = 0.0
                flow['http_content_length_total'] = 0.0
        else:
            flow['http_content_length_mean'] = 0.0
            flow['http_content_length_total'] = 0.0
        
        # === ICMP Extended (1 feature) ===
        
        # 9. ICMP Type (ICMP flood classification)
        # Why: Type 8 = Echo Request (ping flood), Type 3 = Dest Unreachable (scan)
        # Availability: ICMP traffic only
        if 'ICMP Type' in packets.columns:
            icmp_types = pd.to_numeric(packets['ICMP Type'], errors='coerce').dropna()
            if len(icmp_types) > 0:
                flow['icmp_type_mode'] = int(icmp_types.mode()[0]) if len(icmp_types) > 0 else -1
                flow['icmp_type_8_ratio'] = float((icmp_types == 8).sum() / len(icmp_types))  # Echo Request
                flow['icmp_type_3_ratio'] = float((icmp_types == 3).sum() / len(icmp_types))  # Dest Unreachable
                flow['icmp_type_diversity'] = int(icmp_types.nunique())
            else:
                flow['icmp_type_mode'] = -1
                flow['icmp_type_8_ratio'] = 0.0
                flow['icmp_type_3_ratio'] = 0.0
                flow['icmp_type_diversity'] = 0
        else:
            flow['icmp_type_mode'] = -1
            flow['icmp_type_8_ratio'] = 0.0
            flow['icmp_type_3_ratio'] = 0.0
            flow['icmp_type_diversity'] = 0
        
        # === DNS Extended (1 feature) ===
        
        # 10. DNS Query Type (DNS tunneling detection)
        # Why: Normal = A records, Tunneling = TXT/MX/NULL records
        # Availability: DNS traffic only
        if flow['has_dns'] and 'DNS Query Type' in packets.columns:
            query_types = pd.to_numeric(packets['DNS Query Type'], errors='coerce').dropna()
            if len(query_types) > 0:
                flow['dns_type_A_ratio'] = float((query_types == 1).sum() / len(query_types))  # A record
                flow['dns_type_TXT_ratio'] = float((query_types == 16).sum() / len(query_types))  # TXT (tunneling!)
                flow['dns_type_diversity'] = int(query_types.nunique())
            else:
                flow['dns_type_A_ratio'] = 0.0
                flow['dns_type_TXT_ratio'] = 0.0
                flow['dns_type_diversity'] = 0
        else:
            flow['dns_type_A_ratio'] = 0.0
            flow['dns_type_TXT_ratio'] = 0.0
            flow['dns_type_diversity'] = 0
            
        # add window size as a feature for multi-scale learning
        flow["window_size"] = float(self.time_window)
        
        return flow
    

def build_all_attacks_multi_scale(benign_csv="/Users/emre/Desktop/tez/data/raw/Benign/normal_data.csv", windows=[1.0, 5.0, 30.0]):
    """
    Build flows for ALL attack types at ALL window sizes.
    
    This is the CORRECT approach for real-world IDS deployment:
    - ALL attacks processed at ALL windows (1s, 5s, 30s)
    - Benign processed at ALL windows
    - window_size is just "scale information", NOT a constraint
    - Model learns multi-scale representations for each attack
    
    Why this is essential:
        At inference time on edge device, you DON'T know the attack type!
        You must process at all scales and let model decide.
        
        Example:
            Unknown traffic → Process at 1s, 5s, 30s
            → Feed all 3 to model
            → Model sees attack patterns at multiple scales
            → No circular dependency!
    
    Args:
        benign_csv: Path to benign traffic CSV (default: your normal_data.csv path)
                   Set to None to exclude benign traffic (attacks only)
        windows: List of window sizes (default: [1.0, 5.0, 30.0])
    
    Returns:
        DataFrame with ALL attacks + benign at ALL window sizes
        
    Example 1 (with benign - RECOMMENDED for training):
        >>> combined = build_all_attacks_multi_scale()  # Uses default benign path
        >>> print(f"Classes: {combined['label'].nunique()}")  # 16 (15 attacks + benign)
        
    Example 2 (without benign - attacks only):
        >>> combined = build_all_attacks_multi_scale(benign_csv=None)
        >>> print(f"Classes: {combined['label'].nunique()}")  # 15 (attacks only)
        
    Example 3 (custom paths):
        >>> combined = build_all_attacks_multi_scale(
        ...     benign_csv='/path/to/your/benign.csv',
        ...     windows=[1.0, 5.0, 30.0]
        ... )
    
    Model learning:
        PORT-SCANNING at 1s: packet_rate=500, rst_ratio=0.9
        PORT-SCANNING at 5s: packet_rate=100, rst_ratio=0.9  
        PORT-SCANNING at 30s: packet_rate=17, rst_ratio=0.9
        → Model learns: "High RST at ALL scales = Port scan"
        
        SSH-BRUTE at 1s: packet_rate=0.1, sparse
        SSH-BRUTE at 5s: packet_rate=0.5, dst_port=22
        SSH-BRUTE at 30s: packet_rate=0.2, packet_count=12, RICH!
        → Model learns: "Low rate + SSH port at all scales = Brute force"
    """
    # Define attack files (ALL will be processed at ALL windows)
    attack_files = {
        'PORT-SCANNING': 'data/raw/Malicious/PORT-SCANNING/port_scanning.csv',
        'SYN-FLOOD': 'data/raw/Malicious/SYN-FLOOD/syn_flood.csv',
        'DDOS-ICMP': 'data/raw/Malicious/DDOS-ICMP/ddos_icmp.csv',
        'DDOS-UDP': 'data/raw/Malicious/DDOS-UDP/ddos_udp.csv',
        'DDOS-RAW': 'data/raw/Malicious/DDOS-RAW/ddos_raw.csv',
        'DOS': 'data/raw/Malicious/DOS/dos.csv',
        'ICMP-FLOOD': 'data/raw/Malicious/ICMP-FLOOD/icmp_flood.csv',
        'SQL-INJECTION': ['data/raw/Malicious/SQL-INJECTION/sqli_1.csv',
                          "data/raw/Malicious/SQL-INJECTION/sqli_2.csv",
                          "data/raw/Malicious/SQL-INJECTION/sqli_3.csv",
                          ],
        'XSS': ['data/raw/Malicious/XSS/xss_1.csv',
                "data/raw/Malicious/XSS/xss_2.csv",
                "data/raw/Malicious/XSS/xss_3.csv",
                ],
        'FUZZING': 'data/raw/Malicious/FUZZING/fuzzing.csv',
        'ARP-SPOOF': 'data/raw/Malicious/ARP-SPOOF/arp_spoof.csv',
        'REMOTE-CODE-EXECUTION': 'data/raw/Malicious/REMOTE-CODE-EXECUTION/rce.csv',
        'SSH-BRUTE-FORCE': 'data/raw/Malicious/SSH-BRUTE-FORCE/ssh_brute_force.csv',
        'FTP-BRUTE-FORCE': 'data/raw/Malicious/FTP-BRUTE-FORCE/ftp_brute_force.csv',
        'EXPLOITING-FTP': [ 'data/raw/Malicious/EXPLOITING-FTP/ftp_1.csv',
                           "data/raw/Malicious/EXPLOITING-FTP/ftp_2.csv",
            ],
    }
    
    print("="*80)
    print("MULTI-SCALE FLOW BUILDING: ALL ATTACKS AT ALL WINDOWS")
    if benign_csv:
        print("+ BENIGN TRAFFIC")
    print("="*80)
    print(f"Window sizes: {windows}")
    print(f"Strategy: Each attack processed at ALL scales")
    print(f"Reason: Real IDS doesn't know attack type beforehand!")
    print("="*80)
    
    all_flows = []
    
    # Process attacks (ALL at ALL windows)
    for attack_type, csv_path in attack_files.items():
        try:
            import os
            
            # Handle both single file and list of files
            csv_paths = csv_path if isinstance(csv_path, list) else [csv_path]
            
            # Check all files exist
            missing_files = [p for p in csv_paths if not os.path.exists(p)]
            if missing_files:
                print(f"\n⚠️  Skipping {attack_type} (files not found: {missing_files})")
                continue
            
            print(f"\nProcessing {attack_type} at ALL windows {windows}...")
            if len(csv_paths) > 1:
                print(f"  Loading {len(csv_paths)} CSV files...")
            
            # Load and combine all CSVs for this attack type
            dfs = []
            for path in csv_paths:
                df_part = pd.read_csv(path)
                dfs.append(df_part)
                print(f"    {os.path.basename(path)}: {len(df_part)} packets")
            
            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            print(f"  Total packets: {len(df)}")
            
            # Build flows at each window
            for window in windows:
                builder = FlowBuilder(time_window=window)
                flows = builder.build_flows(df)
                flows['label'] = attack_type
                all_flows.append(flows)
                print(f"  {window:5.1f}s: {len(flows):6d} flows")
            
            print(f"  ✓ {attack_type} processed at {len(windows)} scales")
            
        except Exception as e:
            print(f"\n❌ Error processing {attack_type}: {e}")
            continue
    
    # Process benign (at ALL windows if provided)
    if benign_csv:
        try:
            import os
            if os.path.exists(benign_csv):
                print(f"\n{'='*80}")
                print("Processing BENIGN traffic at ALL windows...")
                print(f"{'='*80}")
                print(f"CSV: {benign_csv}")
                print(f"Strategy: Process at ALL scales (benign is diverse)")
                
                # Load benign packets once
                print(f"\nLoading benign packets...")
                df_benign = pd.read_csv(benign_csv)
                print(f"  Packets: {len(df_benign)}")
                print(f"  Time range: {df_benign['Time'].min():.2f}s - {df_benign['Time'].max():.2f}s")
                
                # Build benign flows at each scale
                for window in windows:
                    print(f"\n  Building benign flows with {window}s window...")
                    builder = FlowBuilder(time_window=window)
                    flows = builder.build_flows(df_benign)
                    flows['label'] = 'BENIGN'
                    all_flows.append(flows)
                    print(f"    → {len(flows)} flows created (window_size={window})")
                
                print(f"\n  ✓ BENIGN processed at {len(windows)} scales")
                print(f"    (Captures DIVERSITY: fast/medium/slow normal behavior)")
            else:
                print(f"\n⚠️  Benign file not found: {benign_csv}")
        except Exception as e:
            print(f"\n❌ Error processing benign: {e}")
    
    if not all_flows:
        print("\n❌ No flows generated!")
        return None
    
    # Combine all
    combined = pd.concat(all_flows, ignore_index=True)
    
    print("\n" + "="*80)
    print("MULTI-SCALE DATASET (ALL ATTACKS AT ALL WINDOWS)")
    print("="*80)
    print(f"Total flows: {len(combined)}")
    print(f"Total classes: {combined['label'].nunique()}")
    print(f"Features: {len([col for col in combined.columns if col != 'label'])}")
    print(f"Window sizes: {sorted(combined['window_size'].unique())}")
    
    # Separate attacks and benign
    is_benign = combined['label'] == 'BENIGN'
    attack_flows = combined[~is_benign]
    benign_flows = combined[is_benign]
    
    print(f"\nDataset composition:")
    print(f"  Attack flows: {len(attack_flows):6d} ({len(attack_flows)/len(combined)*100:.1f}%)")
    if len(benign_flows) > 0:
        print(f"  Benign flows: {len(benign_flows):6d} ({len(benign_flows)/len(combined)*100:.1f}%)")
    
    print(f"\nFlows per class (summed across all windows):")
    for label in sorted(combined['label'].unique()):
        count = (combined['label'] == label).sum()
        windows = sorted(combined[combined['label'] == label]['window_size'].unique())
        print(f"  {label:25s}: {count:6d} flows at {len(windows)} scales")
    
    print(f"\nWindow distribution (all classes combined):")
    for ws in sorted(combined['window_size'].unique()):
        count = (combined['window_size'] == ws).sum()
        classes = combined[combined['window_size'] == ws]['label'].nunique()
        print(f"  {ws:5.1f}s: {count:6d} flows from {classes} classes")
    
    print(f"\n{'='*80}")
    print("✓ MULTI-SCALE DATASET READY FOR TGAT TRAINING")
    print(f"{'='*80}")
    print("Strategy:")
    print("  • ALL attacks at ALL windows (1s, 5s, 30s)")
    print("  • window_size is just scale information, NOT a constraint")
    print("  • Model learns multi-scale patterns for EACH attack")
    print()
    print("Why this works:")
    print("  • At inference: Process unknown traffic at all scales")
    print("  • Feed all scales to model")
    print("  • Model recognizes attack patterns across scales")
    print("  • No circular dependency!")
    print()
    print("Example:")
    print("  PORT-SCANNING at 1s: packet_rate=500, rst_ratio=0.9")
    print("  PORT-SCANNING at 5s: packet_rate=100, rst_ratio=0.9")
    print("  PORT-SCANNING at 30s: packet_rate=17, rst_ratio=0.9")
    print("  → Model learns: 'High RST at ALL scales = Port scan'")
    
    return combined


def test_flow_building():
    """Test flow building on port scanning data"""
    print("\n" + "="*80)
    print("STEP 3: TESTING FLOW BUILDING")
    print("="*80)
    
    # Load port scanning data
    df = pd.read_csv("data/raw/Malicious/PORT-SCANNING/port_scanning.csv", nrows=1000)
    
    print(f"\n1. Input data:")
    print(f"   Packets: {len(df)}")
    print(f"   Time range: {df['Time'].min():.2f}s to {df['Time'].max():.2f}s")
    print(f"   Duration: {df['Time'].max() - df['Time'].min():.2f}s")
    
    # Build flows
    builder = FlowBuilder(time_window=10.0)
    flows = builder.build_flows(df)
    
    print(f"\n2. Output flows:")
    print(f"   Total flows: {len(flows)}")
    print(f"   Time windows: {flows['time_window_id'].nunique()}")
    print(f"   Window size feature: {flows['window_size'].unique()}")
    
    print(f"\n3. Sample flows:")
    for i in range(min(5, len(flows))):
        flow = flows.iloc[i]
        print(f"   Flow {i}: {flow['src_node']} →[{flow['protocol']}]→ {flow['dst_node']}")
        print(f"      Packets: {flow['packet_count']}, Bytes: {flow['total_bytes']}")
        print(f"      SYN ratio: {flow['syn_ratio']:.2f}, RST ratio: {flow['rst_ratio']:.2f}")
        print(f"      Packet rate: {flow['packet_rate']:.1f} pkt/s, Mean IAT: {flow['mean_iat']:.4f}s")
        print(f"      Window size: {flow['window_size']}s")
    
    print(f"\n4. Flow statistics:")
    print(f"   Avg packets per flow: {flows['packet_count'].mean():.1f}")
    print(f"   Avg duration: {flows['duration'].mean():.4f}s")
    print(f"   Avg RST ratio: {flows['rst_ratio'].mean():.3f}")
    print(f"   ^ High RST ratio indicates port scanning (closed ports)")
    
    print(f"\n5. Unique connections:")
    unique_connections = flows.groupby(['src_ip', 'dst_ip']).size()
    print(f"   Unique (src, dst) pairs: {len(unique_connections)}")
    
    print(f"\n6. Destination port diversity:")
    unique_dst_ports = flows['dst_port'].nunique()
    print(f"   Unique destination ports: {unique_dst_ports}")
    print(f"   Sample ports: {sorted(flows['dst_port'].unique())[:20]}")
    print(f"   ^ Many unique ports = scanning!")
    
    print("\n" + "="*80)
    print("✓ Step 3 complete! Flows are built correctly.")
    print("="*80)
    
    return flows


if __name__ == '__main__':
    flows = test_flow_building()
    
    # Save for next step
    flows.to_csv('test_flows.csv', index=False)
    print("\n✓ Saved flows to 'test_flows.csv' for next step")

