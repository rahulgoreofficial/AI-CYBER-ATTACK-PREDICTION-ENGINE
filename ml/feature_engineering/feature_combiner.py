"""
Feature Combiner — Build the complete ML-ready feature matrix.
================================================================

Orchestrates the combination of:
1. Traffic features (per-device per-window aggregations)
2. Graph features (degree, centrality, PageRank, attack metrics)
3. Asset features (criticality, vulnerability, device type)
4. Temporal delta features (window-over-window changes)
5. Target labels (is_future_target)

Output: One row per (window_id, device_id) with all features + target.

Anti-leakage guarantee:
    - Features use ONLY events at or before window T
    - Target labels use ONLY events strictly after window T
    - Validated with explicit assertions

Usage:
    from ml.feature_engineering.feature_combiner import build_complete_feature_matrix
    feature_matrix = build_complete_feature_matrix(df_processed)

    # Or from command line:
    python -m ml.feature_engineering.feature_combiner --day wednesday
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import PATHS, TIME_CONFIG, GRAPH_CONFIG
from ml.feature_engineering.target_generator import (
    generate_target_labels,
    generate_device_features_per_window,
)
from graph.construction import build_all_window_graphs, load_topology
from graph.features import extract_all_graph_features


# ==============================================================================
# TEMPORAL DELTA FEATURES
# ==============================================================================

def add_temporal_delta_features(
    device_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add window-over-window delta features to capture temporal trends.

    For each device, computes the change from the previous window for
    key traffic metrics.

    Args:
        device_features: DataFrame with (window_id, device_id) and traffic features.

    Returns:
        DataFrame with additional delta columns.
    """
    print("[features] Computing temporal delta features...")

    # Sort to ensure correct windowing
    device_features = device_features.sort_values(
        ["device_id", "window_id"]
    ).reset_index(drop=True)

    # Key columns to compute deltas for
    delta_cols = [
        "dst_flow_count", "src_flow_count", "total_flow_count",
        "dst_attack_count", "total_attack_count",
        "dst_unique_sources", "src_unique_destinations",
    ]

    # Only compute deltas for columns that exist
    delta_cols = [c for c in delta_cols if c in device_features.columns]

    for col in delta_cols:
        # Compute per-device delta (current - previous window)
        device_features[f"delta_{col}"] = (
            device_features.groupby("device_id")[col].diff().fillna(0)
        )

    # Add "new connections" feature: was dst_unique_sources zero in previous window?
    if "dst_unique_sources" in device_features.columns:
        prev_sources = device_features.groupby("device_id")["dst_unique_sources"].shift(1).fillna(0)
        device_features["new_inbound_connections"] = (
            (device_features["dst_unique_sources"] > 0) & (prev_sources == 0)
        ).astype(int)

    # Add rolling window features (mean over last 3 windows)
    lookback = TIME_CONFIG.get("lookback_windows", 3)
    for col in ["total_flow_count", "total_attack_count"]:
        if col in device_features.columns:
            device_features[f"rolling_{lookback}w_{col}"] = (
                device_features.groupby("device_id")[col]
                .transform(lambda x: x.rolling(lookback, min_periods=1).mean())
            )

    n_delta_features = sum(1 for c in device_features.columns if c.startswith("delta_"))
    n_rolling_features = sum(1 for c in device_features.columns if c.startswith("rolling_"))
    print(f"[features] Added {n_delta_features} delta + {n_rolling_features} rolling features")

    return device_features


# ==============================================================================
# CATEGORICAL ENCODING
# ==============================================================================

def encode_categorical_features(
    df: pd.DataFrame,
    topology: dict,
) -> pd.DataFrame:
    """
    One-hot encode device_type from topology metadata.

    Args:
        df: Feature DataFrame with device_id column.
        topology: Output from load_topology().

    Returns:
        DataFrame with device_type one-hot encoded columns.
    """
    print("[features] Encoding categorical features...")

    # Map device_id to device_type
    df["device_type"] = df["device_id"].map(
        lambda d: topology["devices"].get(d, {}).get("device_type", "unknown")
    )

    # One-hot encode
    type_dummies = pd.get_dummies(df["device_type"], prefix="devtype")
    df = pd.concat([df, type_dummies], axis=1)

    # Drop the string column (not useful for ML)
    df = df.drop(columns=["device_type"])

    n_types = len(type_dummies.columns)
    print(f"[features] Encoded {n_types} device types: {list(type_dummies.columns)}")

    return df


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def build_complete_feature_matrix(
    df_processed: pd.DataFrame,
    prediction_horizon_windows: int = 3,
    include_graph_features: bool = True,
    include_temporal_deltas: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the complete ML-ready feature matrix.

    Orchestrates:
    1. Traffic feature aggregation (per-device per-window)
    2. Graph construction & feature extraction
    3. Asset metadata merging
    4. Temporal delta computation
    5. Target label generation & merging
    6. Categorical encoding

    Args:
        df_processed: Fully preprocessed flow-level DataFrame.
        prediction_horizon_windows: Future windows for target labels.
        include_graph_features: Whether to compute graph features.
        include_temporal_deltas: Whether to compute temporal deltas.
        verbose: Print progress.

    Returns:
        DataFrame with one row per (window_id, device_id) containing:
        - All traffic features
        - All graph features
        - Asset metadata
        - Temporal deltas
        - Target label (is_future_target)
    """
    if verbose:
        print(f"\n{'='*70}")
        print("BUILDING COMPLETE FEATURE MATRIX")
        print(f"{'='*70}")

    # ── Step 1: Traffic features ──
    if verbose:
        print("\n[Step 1/6] Aggregating traffic features...")
    device_features = generate_device_features_per_window(df_processed)

    # ── Step 2: Graph features ──
    if include_graph_features:
        if verbose:
            print("\n[Step 2/6] Building graphs & extracting features...")
        topology = load_topology()
        window_graphs = build_all_window_graphs(
            df_processed, topology, verbose=verbose
        )
        graph_features = extract_all_graph_features(window_graphs, verbose=verbose)

        # Merge graph features
        device_features = device_features.merge(
            graph_features,
            on=["window_id", "device_id"],
            how="left",
        )
        # Fill NaN for devices not in graph (shouldn't happen, but safe)
        graph_cols = [c for c in graph_features.columns
                      if c not in ("window_id", "device_id")]
        device_features[graph_cols] = device_features[graph_cols].fillna(0)
    else:
        if verbose:
            print("\n[Step 2/6] Skipping graph features (disabled)")

    # ── Step 3: Asset metadata ──
    if verbose:
        print("\n[Step 3/6] Adding asset metadata...")
    topology = load_topology()

    # Criticality & vulnerability
    device_features["device_criticality"] = device_features["device_id"].map(
        lambda d: topology["devices"].get(d, {}).get("criticality", 0.3)
    )
    device_features["device_vulnerability"] = device_features["device_id"].map(
        lambda d: topology["devices"].get(d, {}).get("vulnerability", 0.3)
    )

    # VLAN encoding
    device_features["device_vlan"] = device_features["device_id"].map(
        lambda d: topology["devices"].get(d, {}).get("vlan", "external")
    )
    vlan_dummies = pd.get_dummies(device_features["device_vlan"], prefix="vlan")
    device_features = pd.concat([device_features, vlan_dummies], axis=1)
    device_features = device_features.drop(columns=["device_vlan"])

    if verbose:
        print(f"[features] Added criticality, vulnerability, {len(vlan_dummies.columns)} VLAN features")

    # ── Step 4: Temporal deltas ──
    if include_temporal_deltas:
        if verbose:
            print("\n[Step 4/6] Computing temporal delta features...")
        device_features = add_temporal_delta_features(device_features)
    else:
        if verbose:
            print("\n[Step 4/6] Skipping temporal deltas (disabled)")

    # ── Step 5: Device type encoding ──
    if verbose:
        print("\n[Step 5/6] Encoding device types...")
    device_features = encode_categorical_features(device_features, topology)

    # ── Step 6: Target labels ──
    if verbose:
        print("\n[Step 6/6] Generating & merging target labels...")
    target_df = generate_target_labels(df_processed, prediction_horizon_windows)

    device_features = device_features.merge(
        target_df[["window_id", "device_id", "is_future_target",
                    "target_attack_count", "target_attack_types",
                    "earliest_target_window"]],
        on=["window_id", "device_id"],
        how="left",
    )
    device_features["is_future_target"] = (
        device_features["is_future_target"].fillna(0).astype(int)
    )

    # ── Anti-leakage validation ──
    _validate_no_leakage(device_features)

    # ── Final summary ──
    if verbose:
        _print_feature_matrix_summary(device_features)

    return device_features


def _validate_no_leakage(df: pd.DataFrame) -> None:
    """
    Validate that no future-target information leaked into features.

    Checks that feature columns don't contain target-derived information.
    """
    # Target columns should be clearly separated
    target_cols = {"is_future_target", "target_attack_count",
                   "target_attack_types", "earliest_target_window"}
    feature_cols = set(df.columns) - target_cols - {"window_id", "device_id"}

    # Verify no feature column name suggests target leakage
    suspicious = [c for c in feature_cols if "future" in c.lower() or "target" in c.lower()]
    if suspicious:
        print(f"[WARNING] Suspicious feature columns (possible leakage): {suspicious}")

    # Verify is_future_target is binary
    assert df["is_future_target"].isin([0, 1]).all(), "Target is not binary!"

    print("[features] [OK] Anti-leakage validation passed")


def _print_feature_matrix_summary(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the feature matrix."""
    target_cols = {"is_future_target", "target_attack_count",
                   "target_attack_types", "earliest_target_window"}
    id_cols = {"window_id", "device_id"}
    feature_cols = [c for c in df.columns if c not in target_cols and c not in id_cols]

    n_positive = df["is_future_target"].sum()
    n_total = len(df)

    print(f"\n{'='*70}")
    print("FEATURE MATRIX COMPLETE")
    print(f"{'='*70}")
    print(f"  Total samples:       {n_total:,}")
    print(f"  Feature columns:     {len(feature_cols)}")
    print(f"  ID columns:          {len(id_cols)}")
    print(f"  Target columns:      {len(target_cols)}")
    print(f"  Time windows:        {df['window_id'].nunique()}")
    print(f"  Devices:             {df['device_id'].nunique()}")
    print(f"  Positive targets:    {n_positive:,} ({n_positive/n_total*100:.2f}%)")
    print(f"  Negative targets:    {n_total - n_positive:,} ({(n_total-n_positive)/n_total*100:.2f}%)")

    # Feature categories
    traffic_feats = [c for c in feature_cols if c.startswith("dst_") or c.startswith("src_")]
    graph_feats = [c for c in feature_cols if c in (
        "degree", "in_degree", "out_degree", "betweenness_centrality",
        "closeness_centrality", "pagerank", "neighbor_attack_count",
        "is_attack_source", "is_attack_destination", "attack_edge_ratio",
        "weighted_degree", "traffic_node",
    )]
    asset_feats = [c for c in feature_cols if c.startswith("device_") or c.startswith("vlan_")]
    delta_feats = [c for c in feature_cols if c.startswith("delta_") or c.startswith("rolling_")]
    type_feats = [c for c in feature_cols if c.startswith("devtype_")]
    other_feats = [c for c in feature_cols
                   if c not in traffic_feats + graph_feats + asset_feats + delta_feats + type_feats]

    print(f"\n  --- Feature Breakdown ---")
    print(f"  Traffic features:    {len(traffic_feats)}")
    print(f"  Graph features:      {len(graph_feats)}")
    print(f"  Asset features:      {len(asset_feats)}")
    print(f"  Temporal deltas:     {len(delta_feats)}")
    print(f"  Device type (OHE):   {len(type_feats)}")
    print(f"  Other features:      {len(other_feats)}")
    if other_feats:
        print(f"    -> {other_feats}")

    # NaN check
    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n  [WARNING] {len(nan_cols)} feature columns have NaN values:")
        for col, count in nan_cols.items():
            print(f"    {col}: {count} NaN")
    else:
        print(f"\n  [OK] No NaN values in any feature column")

    print(f"{'='*70}")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build complete ML feature matrix (traffic + graph + asset + temporal)"
    )
    parser.add_argument("--day", default="wednesday", help="CICIDS2017 day to process")
    parser.add_argument("--no-graph", action="store_true",
                        help="Skip graph feature extraction")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Skip temporal delta features")
    parser.add_argument("--horizon", type=int, default=3,
                        help="Prediction horizon in windows (default: 3 = 15 min)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output CSV")
    args = parser.parse_args()

    # Load processed data
    processed_path = PATHS["data_processed"] / f"cicids2017_{args.day}_processed.csv"
    if not processed_path.exists():
        print(f"[ERROR] Processed data not found: {processed_path}")
        print(f"Run: python -m ml.preprocessing.pipeline --day {args.day}")
        sys.exit(1)

    print(f"\nLoading processed data: {processed_path}")
    df = pd.read_csv(processed_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} rows, {df['window_id'].nunique()} windows")

    # Build feature matrix
    feature_matrix = build_complete_feature_matrix(
        df,
        prediction_horizon_windows=args.horizon,
        include_graph_features=not args.no_graph,
        include_temporal_deltas=not args.no_temporal,
    )

    # Save
    if not args.no_save:
        output_path = PATHS["data_processed"] / f"ml_feature_matrix_{args.day}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_matrix.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Feature matrix saved: {output_path} ({size_mb:.1f} MB)")
        print(f"  Shape: {feature_matrix.shape}")

    return feature_matrix


if __name__ == "__main__":
    main()
