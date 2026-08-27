"""
Graph Feature Extraction — Compute per-node graph metrics per time window.
===========================================================================

Extracts structural and attack-aware features from each time-window graph:
- Centrality metrics: degree, betweenness, closeness, PageRank
- Attack-context metrics: neighbor attack count, attack edge ratio
- Structural: in/out degree

Usage:
    from graph.features import extract_all_graph_features
    graph_features_df = extract_all_graph_features(window_graphs)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.config import PATHS, GRAPH_CONFIG


def extract_graph_features(G: nx.DiGraph, window_id: int) -> pd.DataFrame:
    """
    Extract per-node graph features from a single time-window graph.

    Args:
        G: NetworkX DiGraph for one time window.
        window_id: The window identifier.

    Returns:
        DataFrame with one row per node, columns:
            window_id, device_id, degree, in_degree, out_degree,
            betweenness_centrality, closeness_centrality, pagerank,
            neighbor_attack_count, is_attack_source, is_attack_destination,
            attack_edge_ratio, weighted_degree, traffic_node
    """
    nodes = list(G.nodes)
    if len(nodes) == 0:
        return pd.DataFrame(columns=[
            "window_id", "device_id", "degree", "in_degree", "out_degree",
            "betweenness_centrality", "closeness_centrality", "pagerank",
            "neighbor_attack_count", "is_attack_source", "is_attack_destination",
            "attack_edge_ratio", "weighted_degree", "traffic_node",
        ])

    # ── Basic degree metrics ──
    degree_dict = dict(G.degree())
    in_degree_dict = dict(G.in_degree())
    out_degree_dict = dict(G.out_degree())

    # ── Centrality metrics ──
    # Use the undirected view for betweenness and closeness (standard practice)
    G_undirected = G.to_undirected()

    try:
        betweenness = nx.betweenness_centrality(G_undirected)
    except Exception:
        betweenness = {n: 0.0 for n in nodes}

    try:
        closeness = nx.closeness_centrality(G_undirected)
    except Exception:
        closeness = {n: 0.0 for n in nodes}

    # PageRank on the directed graph (as intended — influence propagation)
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
    except Exception:
        # Fallback for disconnected graphs
        pagerank = {n: 1.0 / len(nodes) for n in nodes}

    # ── Attack-context metrics ──
    neighbor_attack_count = {}
    is_attack_source = {}
    is_attack_destination = {}
    attack_edge_ratio = {}
    weighted_degree = {}

    for node in nodes:
        # Count attack traffic on adjacent edges (in + out)
        attack_neighbors = 0
        total_edges = 0
        attack_edges = 0
        w_degree = 0.0

        # Outgoing edges
        for _, dst, data in G.out_edges(node, data=True):
            total_edges += 1
            ac = data.get("attack_count", 0)
            freq = data.get("frequency", 0)
            w_degree += freq
            if ac > 0:
                attack_edges += 1
                attack_neighbors += ac

        # Incoming edges
        for src, _, data in G.in_edges(node, data=True):
            total_edges += 1
            ac = data.get("attack_count", 0)
            freq = data.get("frequency", 0)
            w_degree += freq
            if ac > 0:
                attack_edges += 1
                attack_neighbors += ac

        neighbor_attack_count[node] = attack_neighbors

        # Is this node a source of attack traffic?
        outbound_attacks = sum(
            G.edges[(node, dst)].get("attack_count", 0)
            for dst in G.successors(node)
        )
        is_attack_source[node] = int(outbound_attacks > 0)

        # Is this node a destination of attack traffic?
        inbound_attacks = sum(
            G.edges[(src, node)].get("attack_count", 0)
            for src in G.predecessors(node)
        )
        is_attack_destination[node] = int(inbound_attacks > 0)

        # Ratio of edges carrying attack traffic
        attack_edge_ratio[node] = (
            attack_edges / total_edges if total_edges > 0 else 0.0
        )

        weighted_degree[node] = w_degree

    # ── Build DataFrame ──
    records = []
    for node in nodes:
        records.append({
            "window_id": window_id,
            "device_id": node,
            "degree": degree_dict.get(node, 0),
            "in_degree": in_degree_dict.get(node, 0),
            "out_degree": out_degree_dict.get(node, 0),
            "betweenness_centrality": betweenness.get(node, 0.0),
            "closeness_centrality": closeness.get(node, 0.0),
            "pagerank": pagerank.get(node, 0.0),
            "neighbor_attack_count": neighbor_attack_count.get(node, 0),
            "is_attack_source": is_attack_source.get(node, 0),
            "is_attack_destination": is_attack_destination.get(node, 0),
            "attack_edge_ratio": attack_edge_ratio.get(node, 0.0),
            "weighted_degree": weighted_degree.get(node, 0.0),
            "traffic_node": int(G.nodes[node].get("has_traffic", False)),
        })

    return pd.DataFrame(records)


def extract_all_graph_features(
    window_graphs: dict[int, nx.DiGraph],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract graph features for ALL time-window graphs.

    Args:
        window_graphs: dict mapping window_id → nx.DiGraph
        verbose: Print progress.

    Returns:
        DataFrame with one row per (window_id, device_id) containing all
        graph-derived features.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("GRAPH FEATURE EXTRACTION")
        print(f"{'='*70}")
        print(f"  Windows to process: {len(window_graphs)}")

    all_features = []

    for i, (window_id, G) in enumerate(sorted(window_graphs.items())):
        features_df = extract_graph_features(G, window_id)
        all_features.append(features_df)

        if verbose and (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(window_graphs)} windows...")

    graph_features = pd.concat(all_features, ignore_index=True)

    if verbose:
        print(f"\n  --- Graph Feature Statistics ---")
        print(f"  Total rows:           {len(graph_features):,}")
        print(f"  Unique devices:       {graph_features['device_id'].nunique()}")
        print(f"  Unique windows:       {graph_features['window_id'].nunique()}")
        print(f"  Feature columns:      {len(graph_features.columns) - 2}")

        # Summary statistics for key metrics
        for col in ["degree", "betweenness_centrality", "pagerank",
                     "neighbor_attack_count"]:
            if col in graph_features.columns:
                vals = graph_features[col]
                print(f"  {col}: "
                      f"mean={vals.mean():.4f}, "
                      f"max={vals.max():.4f}, "
                      f"nonzero={(vals > 0).sum():,}")

        # Devices with highest average PageRank
        avg_pr = graph_features.groupby("device_id")["pagerank"].mean()
        avg_pr = avg_pr.sort_values(ascending=False)
        print(f"\n  --- Top Devices by Avg PageRank ---")
        for dev, pr in avg_pr.head(8).items():
            print(f"    {dev:<25s} PageRank={pr:.6f}")

        # Devices most often attack sources
        src_rate = graph_features.groupby("device_id")["is_attack_source"].mean()
        src_rate = src_rate[src_rate > 0].sort_values(ascending=False)
        if len(src_rate) > 0:
            print(f"\n  --- Devices as Attack Sources (% of windows) ---")
            for dev, rate in src_rate.head(5).items():
                print(f"    {dev:<25s} {rate*100:.1f}%")

        print(f"{'='*70}")

    return graph_features


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    from graph.construction import build_all_window_graphs, load_topology

    print("Testing graph feature extraction...")

    # Load data
    topology = load_topology()

    processed_path = PATHS["data_processed"] / "cicids2017_wednesday_processed.csv"
    if not processed_path.exists():
        print(f"[ERROR] Processed data not found: {processed_path}")
        sys.exit(1)

    df = pd.read_csv(processed_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} rows")

    # Build graphs
    window_graphs = build_all_window_graphs(df, topology, verbose=False)
    print(f"Built {len(window_graphs)} graphs")

    # Extract features
    graph_features = extract_all_graph_features(window_graphs)

    # Verify
    assert len(graph_features) > 0, "No features extracted!"
    assert "pagerank" in graph_features.columns, "Missing pagerank!"
    assert graph_features["pagerank"].notna().all(), "NaN in pagerank!"

    # Save for inspection
    output_path = PATHS["data_processed"] / "graph_features_wednesday.csv"
    graph_features.to_csv(output_path, index=False)
    print(f"\n[OK] Graph features saved: {output_path}")
    print(f"  Shape: {graph_features.shape}")
    print(f"\n[OK] Graph feature extraction test passed.")
