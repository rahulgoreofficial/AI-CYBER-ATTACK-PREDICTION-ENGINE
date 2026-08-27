"""
Graph Construction — Build per-window NetworkX directed graphs.
================================================================

For each time window, constructs a directed graph where:
- Nodes = devices (active in traffic + topology-only devices)
- Edges = communication links (src_device → dst_device), aggregated per window

Edge attributes:
    frequency, total_bytes, avg_duration, attack_count, protocols, recency

Node attributes (from topology):
    device_type, criticality, vulnerability, vlan, open_ports

Usage:
    from graph.construction import build_all_window_graphs, load_topology
    topology = load_topology()
    window_graphs = build_all_window_graphs(df_processed, topology)
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PATHS, GRAPH_CONFIG


# ==============================================================================
# TOPOLOGY LOADING
# ==============================================================================

def load_topology(topology_path: Path | None = None) -> dict:
    """
    Load and parse the campus topology JSON.

    Returns:
        Dictionary with keys:
        - 'devices': dict mapping device_id → device attributes
        - 'connections': list of connection dicts
        - 'segments': list of network segments
    """
    if topology_path is None:
        topology_path = PATHS["campus_topology"]

    if not topology_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    with open(topology_path) as f:
        raw = json.load(f)

    # Build device lookup
    devices = {}
    for dev in raw["devices"]:
        devices[dev["id"]] = {
            "device_type": dev.get("type", "unknown"),
            "department": dev.get("department", "unknown"),
            "vlan": dev.get("vlan", "unknown"),
            "os": dev.get("os", "unknown"),
            "criticality": dev.get("criticality", 0.5),
            "vulnerability": dev.get("vulnerability", 0.5),
            "open_ports": dev.get("open_ports", []),
            "description": dev.get("description", ""),
        }

    topology = {
        "devices": devices,
        "connections": raw.get("connections", []),
        "segments": raw.get("network", {}).get("segments", []),
    }

    print(f"[graph] Loaded topology: {len(devices)} devices, "
          f"{len(topology['connections'])} connections")

    return topology


# ==============================================================================
# PER-WINDOW GRAPH CONSTRUCTION
# ==============================================================================

def build_graph_for_window(
    df_window: pd.DataFrame,
    topology: dict,
    include_topology_nodes: bool = True,
) -> nx.DiGraph:
    """
    Build a directed graph for a single time window.

    Args:
        df_window: DataFrame filtered to a single window_id.
            Required columns: src_device, dst_device, is_attack
            Optional: flow_duration, total_len_fwd_packets, total_len_bwd_packets,
                      protocol, timestamp, label
        topology: Output from load_topology().
        include_topology_nodes: If True, add nodes from topology even if
            they have no traffic in this window.

    Returns:
        nx.DiGraph with node and edge attributes.
    """
    G = nx.DiGraph()

    # ── Step 1: Add topology nodes (with metadata) ──
    if include_topology_nodes:
        for device_id, attrs in topology["devices"].items():
            G.add_node(device_id, **attrs, has_traffic=False)

    # ── Step 2: Aggregate edges from flow data ──
    if len(df_window) == 0:
        return G

    # Identify available columns for edge aggregation
    has_duration = "flow_duration" in df_window.columns
    has_bytes_fwd = "total_len_fwd_packets" in df_window.columns
    has_bytes_bwd = "total_len_bwd_packets" in df_window.columns
    has_protocol = "protocol" in df_window.columns
    has_timestamp = "timestamp" in df_window.columns
    has_label = "label" in df_window.columns

    # Group by (src_device, dst_device) to aggregate edges
    edge_groups = df_window.groupby(["src_device", "dst_device"])

    for (src, dst), group in edge_groups:
        # Compute edge attributes
        frequency = len(group)
        attack_count = int(group["is_attack"].sum())

        total_bytes = 0.0
        if has_bytes_fwd:
            total_bytes += group["total_len_fwd_packets"].sum()
        if has_bytes_bwd:
            total_bytes += group["total_len_bwd_packets"].sum()

        avg_duration = group["flow_duration"].mean() if has_duration else 0.0

        protocols = set()
        if has_protocol:
            protocols = set(group["protocol"].unique())

        recency = None
        if has_timestamp:
            recency = str(group["timestamp"].max())

        attack_types = set()
        if has_label:
            attack_types = set(group["label"].unique()) - {"BENIGN"}

        # Add edge
        G.add_edge(
            src, dst,
            frequency=frequency,
            total_bytes=float(total_bytes),
            avg_duration=float(avg_duration) if not pd.isna(avg_duration) else 0.0,
            attack_count=attack_count,
            protocols=protocols,
            recency=recency,
            attack_types=attack_types,
        )

        # Mark nodes as having traffic
        if src in G.nodes:
            G.nodes[src]["has_traffic"] = True
        if dst in G.nodes:
            G.nodes[dst]["has_traffic"] = True

    # ── Step 3: Add any traffic-only nodes not in topology ──
    traffic_devices = set(df_window["src_device"].unique()) | set(df_window["dst_device"].unique())
    for device_id in traffic_devices:
        if device_id not in G.nodes:
            # Device appears in traffic but not in topology (e.g., external attackers)
            default_attrs = topology["devices"].get(device_id, {
                "device_type": "external",
                "department": "external",
                "vlan": "external",
                "os": "unknown",
                "criticality": 0.1,
                "vulnerability": 0.5,
                "open_ports": [],
                "description": "External/unmapped device",
            })
            G.add_node(device_id, **default_attrs, has_traffic=True)

    # ── Step 4: Add topology connections as structural edges ──
    # These represent physical/logical links even without active traffic
    for conn in topology["connections"]:
        src, dst = conn["from"], conn["to"]
        if src in G.nodes and dst in G.nodes:
            if not G.has_edge(src, dst):
                G.add_edge(
                    src, dst,
                    frequency=0,
                    total_bytes=0.0,
                    avg_duration=0.0,
                    attack_count=0,
                    protocols=set(),
                    recency=None,
                    attack_types=set(),
                    is_topology_link=True,
                )

    return G


def build_all_window_graphs(
    df_processed: pd.DataFrame,
    topology: dict | None = None,
    include_topology_nodes: bool = True,
    verbose: bool = True,
) -> dict[int, nx.DiGraph]:
    """
    Build directed graphs for ALL time windows.

    Args:
        df_processed: Fully preprocessed DataFrame with window_id.
        topology: Output from load_topology(). Loaded automatically if None.
        include_topology_nodes: Include devices from topology with no traffic.
        verbose: Print progress.

    Returns:
        dict mapping window_id → nx.DiGraph
    """
    if topology is None:
        topology = load_topology()

    all_windows = sorted(df_processed["window_id"].unique())

    if verbose:
        print(f"\n{'='*70}")
        print("GRAPH CONSTRUCTION — ALL WINDOWS")
        print(f"{'='*70}")
        print(f"  Total windows: {len(all_windows)}")
        print(f"  Include topology nodes: {include_topology_nodes}")

    window_graphs = {}
    total_nodes = 0
    total_edges = 0
    attack_edge_windows = 0

    for window_id in all_windows:
        df_window = df_processed[df_processed["window_id"] == window_id]
        G = build_graph_for_window(df_window, topology, include_topology_nodes)
        window_graphs[window_id] = G

        total_nodes += G.number_of_nodes()
        total_edges += G.number_of_edges()

        # Check for attack edges
        has_attack_edges = any(
            G.edges[e].get("attack_count", 0) > 0
            for e in G.edges
        )
        if has_attack_edges:
            attack_edge_windows += 1

    if verbose:
        avg_nodes = total_nodes / len(all_windows) if all_windows else 0
        avg_edges = total_edges / len(all_windows) if all_windows else 0
        print(f"\n  --- Graph Statistics ---")
        print(f"  Graphs built:         {len(window_graphs)}")
        print(f"  Avg nodes per graph:  {avg_nodes:.1f}")
        print(f"  Avg edges per graph:  {avg_edges:.1f}")
        print(f"  Windows with attacks: {attack_edge_windows}/{len(all_windows)}")

        # Sample graph detail
        if all_windows:
            sample_id = all_windows[len(all_windows) // 2]
            G_sample = window_graphs[sample_id]
            print(f"\n  Sample graph (window {sample_id}):")
            print(f"    Nodes: {G_sample.number_of_nodes()}")
            print(f"    Edges: {G_sample.number_of_edges()}")
            traffic_nodes = sum(
                1 for n in G_sample.nodes
                if G_sample.nodes[n].get("has_traffic", False)
            )
            print(f"    Nodes with traffic: {traffic_nodes}")

        print(f"{'='*70}")

    return window_graphs


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testing graph construction...")

    # Load topology
    topology = load_topology()
    print(f"\nTopology devices: {list(topology['devices'].keys())}")

    # Load processed data
    processed_path = PATHS["data_processed"] / "cicids2017_wednesday_processed.csv"
    if not processed_path.exists():
        print(f"[ERROR] Processed data not found: {processed_path}")
        print("Run: python -m ml.preprocessing.pipeline --day wednesday")
        sys.exit(1)

    print(f"\nLoading processed data: {processed_path}")
    df = pd.read_csv(processed_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} rows, {df['window_id'].nunique()} windows")

    # Build all graphs
    window_graphs = build_all_window_graphs(df, topology)

    # Detailed check on one graph
    sample_window = sorted(window_graphs.keys())[len(window_graphs) // 2]
    G = window_graphs[sample_window]
    print(f"\n--- Sample Graph (window {sample_window}) ---")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    for node in sorted(G.nodes)[:5]:
        attrs = G.nodes[node]
        print(f"  Node '{node}': type={attrs.get('device_type', '?')}, "
              f"crit={attrs.get('criticality', '?')}, "
              f"traffic={attrs.get('has_traffic', '?')}")

    for edge in list(G.edges)[:3]:
        attrs = G.edges[edge]
        print(f"  Edge {edge[0]} -> {edge[1]}: "
              f"freq={attrs.get('frequency', 0)}, "
              f"attacks={attrs.get('attack_count', 0)}")

    print("\n[OK] Graph construction test passed.")
