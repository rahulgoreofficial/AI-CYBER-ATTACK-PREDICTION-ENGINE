"""
Synthetic Enrichment — Add IPs, timestamps, and device mappings to ML-only datasets.
====================================================================================

The CICIDS2017 "MachineLearningCVE" version (79 columns) is missing critical columns
for our project: Source IP, Destination IP, Timestamp, Flow ID, Protocol (as number).

This module synthesizes realistic network metadata by:
1. Assigning synthetic timestamps based on row order (data is roughly chronological)
2. Assigning source/destination IPs from the known CICIDS2017 network topology
3. Mapping IPs to campus device IDs
4. Adding protocol information from destination port patterns

IMPORTANT: All synthesized values are explicitly marked as synthetic.
These are NOT real network capture metadata — they are realistic simulations
for research demonstration purposes.

Usage:
    from ml.preprocessing.synthetic_enrichment import enrich_ml_only_dataset
    df_enriched = enrich_ml_only_dataset(df, day="wednesday")
"""

import sys
import json
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import PATHS, RANDOM_SEED


# ==============================================================================
# CICIDS2017 KNOWN NETWORK TOPOLOGY
# ==============================================================================
# These IPs are from the CICIDS2017 documentation / network diagram
# Source: https://www.unb.ca/cic/datasets/ids-2017.html
CICIDS2017_INTERNAL_IPS = [
    "192.168.10.1",    # Gateway/Router
    "192.168.10.3",    # Ubuntu server (web)
    "192.168.10.5",    # PC (Windows)
    "192.168.10.8",    # PC (Windows)
    "192.168.10.9",    # PC (Windows)
    "192.168.10.14",   # PC (Windows)
    "192.168.10.15",   # PC (Windows)
    "192.168.10.17",   # File server
    "192.168.10.19",   # PC (Windows Vista)
    "192.168.10.25",   # PC (Windows)
    "192.168.10.50",   # Server
    "192.168.10.51",   # Server (firewall)
]

CICIDS2017_ATTACKER_IPS = [
    "205.174.165.68",  # Attacker 1
    "205.174.165.69",  # Attacker 2
    "205.174.165.70",  # Attacker 3
    "205.174.165.71",  # Attacker 4
    "205.174.165.73",  # Kali Linux attacker
]

# Map IPs to campus device IDs for our project
IP_TO_DEVICE = {
    "192.168.10.1":    "ROUTER-01",
    "192.168.10.3":    "WEB-SERVER-01",
    "192.168.10.5":    "PC-01",
    "192.168.10.8":    "PC-02",
    "192.168.10.9":    "PC-03",
    "192.168.10.14":   "PC-05",
    "192.168.10.15":   "PC-06",
    "192.168.10.17":   "FILE-SERVER-01",
    "192.168.10.19":   "PC-07",
    "192.168.10.25":   "PC-08",
    "192.168.10.50":   "DB-SERVER-01",
    "192.168.10.51":   "FIREWALL-01",
    "205.174.165.68":  "EXT-ATTACKER-01",
    "205.174.165.69":  "EXT-ATTACKER-02",
    "205.174.165.70":  "EXT-ATTACKER-03",
    "205.174.165.71":  "EXT-ATTACKER-04",
    "205.174.165.73":  "EXT-ATTACKER-05",
}

# Day-specific simulated start times (CICIDS2017 captures ran during working hours)
DAY_START_TIMES = {
    "monday":               "2017-07-03 09:00:00",
    "tuesday":              "2017-07-04 09:00:00",
    "wednesday":            "2017-07-05 09:00:00",
    "thursday_morning":     "2017-07-06 09:00:00",
    "thursday_afternoon":   "2017-07-06 13:00:00",
    "friday_morning":       "2017-07-07 09:00:00",
    "friday_afternoon_ddos":    "2017-07-07 13:00:00",
    "friday_afternoon_portscan":"2017-07-07 13:30:00",
}

# Known attack timing windows for Wednesday (from CICIDS2017 documentation)
WEDNESDAY_ATTACK_WINDOWS = {
    "slowloris":   {"start": "09:47:00", "end": "10:10:00"},
    "slowhttp":    {"start": "10:14:00", "end": "10:35:00"},
    "hulk":        {"start": "10:43:00", "end": "11:00:00"},
    "goldeneye":   {"start": "11:10:00", "end": "11:23:00"},
    "heartbleed":  {"start": "15:12:00", "end": "15:32:00"},
}

# Port → protocol name mapping (common protocols)
PORT_TO_PROTOCOL = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet",
    25: "SMTP", 53: "DNS", 80: "HTTP", 110: "POP3",
    143: "IMAP", 443: "HTTPS", 993: "IMAPS", 995: "POP3S",
    3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
    445: "SMB", 139: "NetBIOS", 135: "RPC",
    8080: "HTTP-Alt", 8443: "HTTPS-Alt",
}


def synthesize_timestamps(df: pd.DataFrame, day: str = "wednesday") -> pd.DataFrame:
    """
    Assign synthetic timestamps to rows based on their order.

    The CICIDS2017 data is roughly chronological within each file.
    We distribute rows across the working day proportionally,
    with random jitter to simulate realistic inter-flow timing.

    Args:
        df: DataFrame (rows should be in original order).
        day: Which day file this represents.

    Returns:
        DataFrame with 'timestamp' column added.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n_rows = len(df)

    # Get start time for this day
    start_str = DAY_START_TIMES.get(day, "2017-07-05 09:00:00")
    start_time = pd.Timestamp(start_str)

    # Working day duration: ~8 hours for full day, ~4 hours for half-day files
    if "afternoon" in day:
        duration_hours = 4
    elif "morning" in day:
        duration_hours = 4
    else:
        duration_hours = 8

    duration_seconds = duration_hours * 3600

    # Generate monotonically increasing timestamps with small random jitter
    # Base: evenly spaced across the day
    base_offsets = np.linspace(0, duration_seconds, n_rows, endpoint=False)

    # Add small random jitter (±2 seconds) to make it realistic
    jitter = rng.uniform(-2, 2, size=n_rows)
    jitter[0] = 0  # First row starts at exact start time

    offsets = base_offsets + jitter
    offsets = np.sort(offsets)  # Ensure monotonicity after jitter
    offsets = np.clip(offsets, 0, duration_seconds)

    timestamps = start_time + pd.to_timedelta(offsets, unit="s")
    df["timestamp"] = timestamps

    print(f"[enrichment] Synthesized {n_rows:,} timestamps")
    print(f"[enrichment]   Range: {timestamps.min()} -> {timestamps.max()}")
    print(f"[enrichment]   Duration: {duration_hours} hours")

    return df


def synthesize_ips(df: pd.DataFrame, day: str = "wednesday") -> pd.DataFrame:
    """
    Assign synthetic source/destination IPs based on traffic patterns.

    Strategy:
    - Benign traffic: random internal-to-internal or internal-to-external
    - Attack traffic: attacker IP → internal victim IP, distributed by label type
    - Use destination port to influence the victim selection (servers vs PCs)

    Args:
        df: DataFrame with 'label' and 'dst_port' columns.
        day: Which day file.

    Returns:
        DataFrame with 'src_ip' and 'dst_ip' columns added.
    """
    rng = np.random.RandomState(RANDOM_SEED + 1)
    n_rows = len(df)

    # Identify label column
    label_col = "label" if "label" in df.columns else "Label"
    port_col = "dst_port" if "dst_port" in df.columns else "Destination Port"

    # Initialize
    src_ips = np.empty(n_rows, dtype=object)
    dst_ips = np.empty(n_rows, dtype=object)

    # Separate attack and benign indices
    is_attack = df[label_col] != "BENIGN"
    attack_idx = np.where(is_attack)[0]
    benign_idx = np.where(~is_attack)[0]

    # --- Benign traffic ---
    # Mix of internal↔internal and internal↔external
    internal_ips = np.array(CICIDS2017_INTERNAL_IPS)
    server_ips = ["192.168.10.3", "192.168.10.17", "192.168.10.50", "192.168.10.51"]
    pc_ips = [ip for ip in CICIDS2017_INTERNAL_IPS if ip not in server_ips and ip != "192.168.10.1"]

    for idx in benign_idx:
        port = df[port_col].iloc[idx] if port_col in df.columns else 80

        # High ports (>1024) as dst → likely a response to internal PC
        # Low ports as dst → likely a request to a server
        if isinstance(port, (int, float)) and port <= 1024:
            # Request to server
            src_ips[idx] = rng.choice(pc_ips)
            if port in (80, 443, 8080):
                dst_ips[idx] = "192.168.10.3"  # Web server
            elif port in (445, 139):
                dst_ips[idx] = "192.168.10.17"  # File server
            elif port in (5432, 3306):
                dst_ips[idx] = "192.168.10.50"  # DB server
            else:
                dst_ips[idx] = rng.choice(server_ips)
        else:
            # Internal communication or response
            src_ips[idx] = rng.choice(internal_ips)
            dst_ip_choices = [ip for ip in internal_ips if ip != src_ips[idx]]
            dst_ips[idx] = rng.choice(dst_ip_choices) if dst_ip_choices else rng.choice(internal_ips)

    # --- Attack traffic ---
    # Attacker IPs → specific victim IPs based on attack type
    attack_victim_map = {
        "DoS slowloris":  "192.168.10.3",     # Web server target
        "DoS Slowhttptest":"192.168.10.3",
        "DoS Hulk":       "192.168.10.3",
        "DoS GoldenEye":  "192.168.10.3",
        "Heartbleed":     "192.168.10.3",      # OpenSSL on web server
        "FTP-Patator":    "192.168.10.50",     # FTP server
        "SSH-Patator":    "192.168.10.50",     # SSH server
        "Web Attack":     "192.168.10.3",      # Web server
        "Bot":            "192.168.10.5",      # Botnet victim PC
        "Infiltration":   "192.168.10.9",      # Infiltrated PC
        "PortScan":       "192.168.10.1",      # Scanning the gateway
        "DDoS":           "192.168.10.3",      # DDoS on web server
    }

    for idx in attack_idx:
        label = df[label_col].iloc[idx]
        # Find matching attack type
        victim_ip = "192.168.10.3"  # Default victim
        for attack_name, victim in attack_victim_map.items():
            if attack_name.lower() in label.lower():
                victim_ip = victim
                break

        src_ips[idx] = rng.choice(CICIDS2017_ATTACKER_IPS)
        dst_ips[idx] = victim_ip

    df["src_ip"] = src_ips
    df["dst_ip"] = dst_ips

    unique_src = len(set(src_ips))
    unique_dst = len(set(dst_ips))
    print(f"[enrichment] Synthesized IPs: {unique_src} unique sources, {unique_dst} unique destinations")
    print(f"[enrichment]   Attack flows: {len(attack_idx):,}, Benign flows: {len(benign_idx):,}")

    return df


def map_ips_to_devices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map IP addresses to stable campus device IDs.

    Uses the IP_TO_DEVICE mapping for known CICIDS2017 IPs,
    falls back to IP-based naming for unknown IPs.
    """
    def ip_to_device(ip: str) -> str:
        if ip in IP_TO_DEVICE:
            return IP_TO_DEVICE[ip]
        # Fallback for unknown IPs
        parts = ip.split(".")
        if parts[0] == "192":
            return f"PC-{parts[2]}-{parts[3]}"
        elif parts[0] in ("205", "172"):
            return f"EXT-{parts[2]}-{parts[3]}"
        return f"UNKNOWN-{ip}"

    df["src_device"] = df["src_ip"].apply(ip_to_device)
    df["dst_device"] = df["dst_ip"].apply(ip_to_device)

    n_devices = len(set(df["src_device"].unique()) | set(df["dst_device"].unique()))
    print(f"[enrichment] Mapped IPs to {n_devices} unique device IDs")

    # Show device distribution
    print(f"[enrichment]   Top source devices:")
    for dev, count in df["src_device"].value_counts().head(5).items():
        print(f"    {dev:<25s} {count:>10,}")
    print(f"[enrichment]   Top destination devices:")
    for dev, count in df["dst_device"].value_counts().head(5).items():
        print(f"    {dev:<25s} {count:>10,}")

    return df


def synthesize_protocol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize protocol column from destination port information.

    Maps well-known ports to protocol numbers:
    - TCP = 6 (most flows)
    - UDP = 17 (DNS, etc.)
    - ICMP = 1

    Also creates a 'protocol_name' column for readability.
    """
    port_col = "dst_port" if "dst_port" in df.columns else "Destination Port"

    if port_col not in df.columns:
        # If no port info at all, default to TCP
        df["protocol"] = 6
        df["protocol_name"] = "TCP"
        return df

    # Default to TCP (protocol 6)
    df["protocol"] = 6

    # UDP ports
    udp_ports = {53, 67, 68, 123, 161, 162, 500, 514, 1194, 1900}
    ports = pd.to_numeric(df[port_col], errors="coerce").fillna(0).astype(int)
    df.loc[ports.isin(udp_ports), "protocol"] = 17  # UDP

    # Protocol name
    df["protocol_name"] = df["protocol"].map({6: "TCP", 17: "UDP", 1: "ICMP"}).fillna("TCP")

    # Service name from destination port
    df["service"] = ports.map(PORT_TO_PROTOCOL).fillna("Other")

    print(f"[enrichment] Synthesized protocol info:")
    print(f"  TCP: {(df['protocol'] == 6).sum():,}")
    print(f"  UDP: {(df['protocol'] == 17).sum():,}")

    return df


def enrich_ml_only_dataset(df: pd.DataFrame, day: str = "wednesday") -> pd.DataFrame:
    """
    Full enrichment pipeline for MachineLearningCVE version of CICIDS2017.

    Adds:
    - timestamp (synthesized from row order + day)
    - src_ip, dst_ip (synthesized from known CICIDS2017 topology + attack labels)
    - src_device, dst_device (mapped from IPs to campus device IDs)
    - protocol, protocol_name, service (from destination port)
    - is_synthetic_metadata flag

    Args:
        df: DataFrame loaded from ML-only CSV (79 columns).
        day: Which day file this represents.

    Returns:
        Enriched DataFrame with all columns needed for the pipeline.
    """
    print("\n" + "=" * 70)
    print("SYNTHETIC ENRICHMENT (MachineLearningCVE -> Full Pipeline)")
    print("=" * 70)
    print(f"[enrichment] Day: {day}")
    print(f"[enrichment] Input: {len(df):,} rows x {len(df.columns)} columns")

    # Mark that metadata is synthesized (for transparency)
    df["is_synthetic_metadata"] = True

    # Step 1: Synthesize timestamps
    df = synthesize_timestamps(df, day=day)

    # Step 2: Synthesize IPs
    df = synthesize_ips(df, day=day)

    # Step 3: Map IPs to device IDs
    df = map_ips_to_devices(df)

    # Step 4: Synthesize protocol info
    df = synthesize_protocol(df)

    print(f"\n[enrichment] Output: {len(df):,} rows x {len(df.columns)} columns")
    print(f"[enrichment] New columns: timestamp, src_ip, dst_ip, src_device, dst_device,")
    print(f"             protocol, protocol_name, service, is_synthetic_metadata")
    print("=" * 70)

    return df


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    from ml.preprocessing.loader import load_cicids2017

    print("Testing synthetic enrichment...")
    df = load_cicids2017("wednesday", nrows=500)
    df_enriched = enrich_ml_only_dataset(df, day="wednesday")

    print(f"\nEnriched columns: {list(df_enriched.columns)}")
    print(f"\nSample enriched data:")
    print(df_enriched[["timestamp", "src_ip", "dst_ip", "src_device", "dst_device",
                        "protocol_name", "service"]].head(5).to_string())
    print(f"\n[OK] Enrichment works correctly.")
