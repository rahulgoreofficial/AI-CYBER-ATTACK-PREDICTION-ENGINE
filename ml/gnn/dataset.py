"""
PyTorch Geometric Dataset Converter — M5.1
============================================

Converts the per-window NetworkX DiGraphs (from graph/construction.py) into
PyTorch Geometric ``Data`` objects suitable for GNN training.

Each time-window graph becomes one ``Data`` object containing:
- ``x``: Node feature matrix  [num_nodes, num_features]
- ``edge_index``: COO adjacency [2, num_edges]
- ``edge_attr``: Edge feature matrix [num_edges, num_edge_features]
- ``y``: Per-node binary target (is_future_target) [num_nodes]
- ``device_ids``: List of device_id strings (for evaluation mapping)
- ``window_id``: Integer window identifier

Anti-leakage:
    - Node features are computed from current + past windows only
    - Target labels come from strictly future windows
    - Train/val/test split is temporal (same as XGBoost)

Usage:
    from ml.gnn.dataset import build_pyg_dataset
    dataset, splits = build_pyg_dataset(day="wednesday")
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    raise ImportError(
        "PyTorch Geometric is required for the GNN module.\n"
        "Install: pip install torch torch-geometric\n"
        "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )

from ml.config import PATHS, MODEL_CONFIG, RANDOM_SEED
from graph.construction import build_all_window_graphs, load_topology
from graph.features import extract_all_graph_features
from ml.feature_engineering.feature_combiner import build_complete_feature_matrix


# ==============================================================================
# NODE FEATURE EXTRACTION
# ==============================================================================

def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify the numeric feature columns from the feature matrix.
    Mirrors the logic in ml/xgboost_model/train.py::prepare_features().
    """
    exclude_cols = {
        "window_id", "device_id",
        "is_future_target", "target_attack_count",
        "target_attack_types", "earliest_target_window",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]
    ]
    return feature_cols


def _build_node_id_mapping(G) -> dict[str, int]:
    """Create a deterministic node → index mapping (sorted for reproducibility)."""
    return {node: idx for idx, node in enumerate(sorted(G.nodes))}


# ==============================================================================
# EDGE FEATURE EXTRACTION
# ==============================================================================

def _extract_edge_features(G, node_map: dict[str, int]) -> tuple:
    """
    Extract edge_index and edge_attr from a NetworkX DiGraph.

    Returns:
        edge_index: Tensor [2, num_edges] (COO format)
        edge_attr: Tensor [num_edges, 4] — (frequency, total_bytes, avg_duration, attack_count)
    """
    if G.number_of_edges() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
        return edge_index, edge_attr

    src_list = []
    dst_list = []
    attrs = []

    for u, v, data in G.edges(data=True):
        if u not in node_map or v not in node_map:
            continue
        src_list.append(node_map[u])
        dst_list.append(node_map[v])
        attrs.append([
            float(data.get("frequency", 0)),
            float(data.get("total_bytes", 0.0)),
            float(data.get("avg_duration", 0.0)),
            float(data.get("attack_count", 0)),
        ])

    if len(src_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
        return edge_index, edge_attr

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)

    return edge_index, edge_attr


# ==============================================================================
# PyG DATA CONSTRUCTION
# ==============================================================================

def build_pyg_data_for_window(
    window_id: int,
    G,
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
) -> Data | None:
    """
    Build a single PyG Data object for one time-window graph.

    Args:
        window_id: The time window identifier.
        G: NetworkX DiGraph for this window.
        feature_matrix: Full feature matrix (all windows).
        feature_cols: List of numeric feature column names.

    Returns:
        PyG Data object, or None if the window has no data.
    """
    # Filter feature matrix to this window
    wf = feature_matrix[feature_matrix["window_id"] == window_id].copy()
    if len(wf) == 0:
        return None

    # Build deterministic node ordering
    node_map = _build_node_id_mapping(G)
    sorted_nodes = sorted(G.nodes)
    num_nodes = len(sorted_nodes)

    if num_nodes == 0:
        return None

    # ── Node features ──
    # Create a lookup from device_id to feature vector
    device_to_features = {}
    for _, row in wf.iterrows():
        device_to_features[row["device_id"]] = row[feature_cols].values.astype(np.float32)

    # Build feature matrix: one row per node (sorted order)
    num_features = len(feature_cols)
    x = np.zeros((num_nodes, num_features), dtype=np.float32)
    y = np.zeros(num_nodes, dtype=np.float32)
    device_ids = []

    for idx, node in enumerate(sorted_nodes):
        device_ids.append(node)
        if node in device_to_features:
            features = device_to_features[node]
            # Handle NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            x[idx] = features

        # Target label
        target_row = wf[wf["device_id"] == node]
        if len(target_row) > 0:
            y[idx] = float(target_row["is_future_target"].values[0])

    # ── Edge features ──
    edge_index, edge_attr = _extract_edge_features(G, node_map)

    # ── Build Data object ──
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y, dtype=torch.float32),
    )

    # Store metadata (not tensors — for evaluation)
    data.device_ids = device_ids
    data.window_id = window_id
    data.num_nodes = num_nodes

    return data


# ==============================================================================
# DATASET BUILDER
# ==============================================================================

def build_pyg_dataset(
    day: str = "wednesday",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    verbose: bool = True,
) -> tuple[list, dict]:
    """
    Build the complete PyG dataset from the CICIDS2017 processed data.

    Pipeline:
    1. Load processed CSV → build feature matrix (reuses feature_combiner)
    2. Build per-window NetworkX graphs (reuses graph/construction)
    3. Convert each window graph to a PyG Data object
    4. Split by temporal ordering (same as XGBoost)

    Args:
        day: CICIDS2017 day identifier.
        train_ratio: Fraction of windows for training.
        val_ratio: Fraction of windows for validation.
        verbose: Print progress.

    Returns:
        (dataset, splits) where:
        - dataset: list of PyG Data objects (one per window)
        - splits: dict with 'train', 'val', 'test' lists of indices
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"BUILDING PyG DATASET — {day.upper()}")
        print(f"{'='*70}")

    # ── Step 1: Load processed data ──
    processed_path = PATHS["data_processed"] / f"cicids2017_{day}_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {processed_path}\n"
            f"Run: python -m ml.preprocessing.pipeline --day {day}"
        )

    print(f"[gnn] Loading processed data: {processed_path.name}")
    df = pd.read_csv(processed_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Step 2: Build feature matrix (reuse existing pipeline) ──
    # Check if cached feature matrix exists
    fm_path = PATHS["data_processed"] / f"ml_feature_matrix_{day}.csv"
    if fm_path.exists():
        print(f"[gnn] Loading cached feature matrix: {fm_path.name}")
        feature_matrix = pd.read_csv(fm_path, low_memory=False)
    else:
        print("[gnn] Building feature matrix from scratch...")
        feature_matrix = build_complete_feature_matrix(df)

    feature_cols = _get_feature_columns(feature_matrix)
    print(f"[gnn] Feature columns: {len(feature_cols)}")

    # ── Step 3: Build window graphs ──
    topology = load_topology()
    window_graphs = build_all_window_graphs(df, topology, verbose=False)
    print(f"[gnn] Built {len(window_graphs)} window graphs")

    # ── Step 4: Convert to PyG Data objects ──
    if verbose:
        print("[gnn] Converting to PyG Data objects...")

    dataset = []
    window_ids = sorted(window_graphs.keys())

    for i, wid in enumerate(window_ids):
        G = window_graphs[wid]
        data = build_pyg_data_for_window(wid, G, feature_matrix, feature_cols)
        if data is not None:
            dataset.append(data)

        if verbose and (i + 1) % 25 == 0:
            print(f"  Converted {i+1}/{len(window_ids)} windows...")

    print(f"[gnn] Total PyG Data objects: {len(dataset)}")

    # ── Step 5: Temporal split ──
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, n)),
    }

    if verbose:
        for split_name, indices in splits.items():
            n_graphs = len(indices)
            total_nodes = sum(dataset[i].num_nodes for i in indices)
            total_pos = sum(int(dataset[i].y.sum().item()) for i in indices)
            total_samples = sum(dataset[i].y.shape[0] for i in indices)
            print(f"  {split_name.upper():>5s}: {n_graphs:3d} graphs, "
                  f"{total_samples:5d} node-samples, "
                  f"{total_pos:4d} positive ({total_pos/max(total_samples,1)*100:.1f}%)")

    # ── Summary ──
    if verbose and len(dataset) > 0:
        sample = dataset[0]
        print(f"\n  --- Sample Data Object (window {sample.window_id}) ---")
        print(f"  Nodes:         {sample.num_nodes}")
        print(f"  Node features: {sample.x.shape[1]}")
        print(f"  Edges:         {sample.edge_index.shape[1]}")
        print(f"  Edge features: {sample.edge_attr.shape[1] if sample.edge_attr.shape[0] > 0 else 0}")
        print(f"  Positive (y):  {int(sample.y.sum().item())}")
        print(f"{'='*70}")

    return dataset, splits


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M5.1 — Build PyTorch Geometric dataset from CICIDS2017"
    )
    parser.add_argument("--day", default="wednesday", help="CICIDS2017 day")
    args = parser.parse_args()

    dataset, splits = build_pyg_dataset(day=args.day)

    # Verify
    assert len(dataset) > 0, "No data objects created!"
    assert len(splits["train"]) > 0, "Empty training set!"
    assert len(splits["test"]) > 0, "Empty test set!"

    sample = dataset[0]
    assert sample.x.dim() == 2, f"Bad x shape: {sample.x.shape}"
    assert sample.edge_index.dim() == 2, f"Bad edge_index: {sample.edge_index.shape}"
    assert sample.y.dim() == 1, f"Bad y shape: {sample.y.shape}"
    assert sample.x.shape[0] == sample.y.shape[0], "Node count mismatch!"

    print("\n[OK] PyG dataset construction verified successfully.")
"""
    print(f"  Total graphs:     {len(dataset)}")
    print(f"  Train:            {len(splits['train'])}")
    print(f"  Val:              {len(splits['val'])}")
    print(f"  Test:             {len(splits['test'])}")
    print(f"  Feature dim:      {dataset[0].x.shape[1]}")
"""
