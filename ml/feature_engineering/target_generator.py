"""
Target Label Generator — Create temporal next-target prediction labels.
========================================================================

This is the MOST CRITICAL module differentiating this project from a
standard IDS. It defines the prediction target as:

    "Will device D become an attack destination within the next H minutes?"

Anti-leakage guarantees:
1. Features use ONLY events at or before time window T
2. Target labels use ONLY events strictly after time window T
3. No shuffling across time boundaries

Usage:
    from ml.feature_engineering.target_generator import generate_targets
    df_targets = generate_targets(df_processed)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import TIME_CONFIG, PATHS


def generate_target_labels(
    df: pd.DataFrame,
    prediction_horizon_windows: int = 3,
) -> pd.DataFrame:
    """
    Generate next-target prediction labels for each device in each time window.

    For each time window T and each device D:
        target = 1 if D appears as an attack DESTINATION in any of the
                    next `prediction_horizon_windows` windows [T+1, T+H]
        target = 0 otherwise

    Args:
        df: Preprocessed DataFrame with columns:
            - window_id, dst_device, is_attack
        prediction_horizon_windows: Number of future windows to check.
            Default 3 = 15 minutes (3 x 5-min windows).

    Returns:
        DataFrame with columns:
            window_id, device_id, is_future_target, target_attack_count,
            target_attack_types, earliest_target_window
    """
    print("\n" + "=" * 70)
    print("TARGET LABEL GENERATION")
    print("=" * 70)
    print(f"  Prediction horizon: {prediction_horizon_windows} windows "
          f"({prediction_horizon_windows * TIME_CONFIG['window_size_minutes']} minutes)")

    required_cols = {"window_id", "dst_device", "is_attack"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Step 1: Get all unique devices and windows ──
    all_devices = sorted(set(df["src_device"].unique()) | set(df["dst_device"].unique()))
    all_windows = sorted(df["window_id"].unique())
    n_devices = len(all_devices)
    n_windows = len(all_windows)

    print(f"  Devices: {n_devices}")
    print(f"  Windows: {n_windows}")
    print(f"  Total device-window pairs: {n_devices * n_windows:,}")

    # ── Step 2: For each window, find which devices are attack destinations ──
    # Group attack flows by window_id and destination device
    attack_flows = df[df["is_attack"] == 1]

    # Create a dict: window_id -> set of attacked destination devices
    window_attack_destinations = {}
    # Also track attack types per window per device
    window_attack_types = {}

    for window_id in all_windows:
        window_attacks = attack_flows[attack_flows["window_id"] == window_id]
        if len(window_attacks) > 0:
            destinations = set(window_attacks["dst_device"].unique())
            window_attack_destinations[window_id] = destinations

            # Track attack types
            if "label" in df.columns:
                for dev in destinations:
                    dev_attacks = window_attacks[window_attacks["dst_device"] == dev]
                    types = set(dev_attacks["label"].unique()) - {"BENIGN"}
                    window_attack_types[(window_id, dev)] = types
        else:
            window_attack_destinations[window_id] = set()

    attacked_windows = sum(1 for w in window_attack_destinations.values() if len(w) > 0)
    print(f"  Windows with attack destinations: {attacked_windows}/{n_windows}")

    # ── Step 3: Generate target labels ──
    # For each (window_id, device) pair, check if device is attacked in future windows
    records = []

    for window_id in all_windows:
        # Future windows to check
        future_windows = [
            window_id + offset
            for offset in range(1, prediction_horizon_windows + 1)
            if (window_id + offset) in window_attack_destinations
        ]

        for device in all_devices:
            is_target = False
            attack_count = 0
            attack_types = set()
            earliest_window = None

            for fw in future_windows:
                if device in window_attack_destinations.get(fw, set()):
                    is_target = True
                    attack_count += 1
                    if earliest_window is None:
                        earliest_window = fw
                    if (fw, device) in window_attack_types:
                        attack_types |= window_attack_types[(fw, device)]

            records.append({
                "window_id": window_id,
                "device_id": device,
                "is_future_target": int(is_target),
                "target_attack_count": attack_count,
                "target_attack_types": "|".join(sorted(attack_types)) if attack_types else "",
                "earliest_target_window": earliest_window,
            })

    target_df = pd.DataFrame(records)

    # ── Step 4: Statistics ──
    n_positive = target_df["is_future_target"].sum()
    n_total = len(target_df)
    imbalance = (n_total - n_positive) / max(n_positive, 1)

    print(f"\n  --- Target Label Statistics ---")
    print(f"  Total samples (device x window): {n_total:,}")
    print(f"  Positive (future target):        {n_positive:,} ({n_positive/n_total*100:.2f}%)")
    print(f"  Negative (not target):           {n_total - n_positive:,} ({(n_total-n_positive)/n_total*100:.2f}%)")
    print(f"  Class imbalance ratio:           {imbalance:.1f}:1")

    # Which devices become targets most often?
    device_target_counts = target_df.groupby("device_id")["is_future_target"].sum()
    device_target_counts = device_target_counts.sort_values(ascending=False)
    print(f"\n  --- Devices Most Often Targeted ---")
    for dev, count in device_target_counts.head(10).items():
        total_windows = len(all_windows)
        pct = count / total_windows * 100
        print(f"    {dev:<25s} target in {count:>4} / {total_windows} windows ({pct:.1f}%)")

    # Which devices are NEVER targets?
    never_targeted = device_target_counts[device_target_counts == 0]
    print(f"\n  Devices never targeted: {len(never_targeted)}")
    for dev in never_targeted.index:
        print(f"    {dev}")

    print("=" * 70)

    return target_df


def generate_device_features_per_window(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate flow-level features to device-level per time window.

    For each (window_id, device) pair, compute:
    - Traffic statistics as source (outbound)
    - Traffic statistics as destination (inbound)
    - Combined features

    Args:
        df: Preprocessed flow-level DataFrame.
        feature_cols: List of numeric feature columns to aggregate.
            If None, uses a default set of important features.

    Returns:
        DataFrame with one row per (window_id, device_id) with aggregated features.
    """
    print("\n[features] Aggregating flow features to device-level per window...")

    if feature_cols is None:
        # Default important features for aggregation
        feature_cols = [
            "flow_duration", "total_fwd_packets", "total_bwd_packets",
            "total_len_fwd_packets", "total_len_bwd_packets",
            "fwd_pkt_len_mean", "bwd_pkt_len_mean",
            "flow_bytes_per_s", "flow_packets_per_s",
            "flow_iat_mean", "fwd_iat_mean", "bwd_iat_mean",
            "pkt_len_mean", "pkt_len_std",
            "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
            "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt",
            "avg_pkt_size", "init_win_bytes_fwd", "init_win_bytes_bwd",
            "active_mean", "idle_mean",
        ]
        # Keep only features that exist in the DataFrame
        feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"[features] Using {len(feature_cols)} features for aggregation")

    # All unique devices
    all_devices = sorted(set(df["src_device"].unique()) | set(df["dst_device"].unique()))
    all_windows = sorted(df["window_id"].unique())

    # ── Aggregate as DESTINATION (inbound traffic to device) ──
    dst_agg = df.groupby(["window_id", "dst_device"])[feature_cols].agg(["mean", "sum", "count"])
    dst_agg.columns = [f"dst_{col}_{stat}" for col, stat in dst_agg.columns]
    dst_agg = dst_agg.reset_index()
    dst_agg = dst_agg.rename(columns={"dst_device": "device_id"})

    # Add destination flow count and unique source count
    dst_counts = df.groupby(["window_id", "dst_device"]).agg(
        dst_flow_count=("is_attack", "size"),
        dst_attack_count=("is_attack", "sum"),
        dst_unique_sources=("src_device", "nunique"),
    ).reset_index().rename(columns={"dst_device": "device_id"})

    # ── Aggregate as SOURCE (outbound traffic from device) ──
    src_agg = df.groupby(["window_id", "src_device"])[feature_cols].agg(["mean", "sum"]).head(0)
    # Simplified: just get flow counts as source
    src_counts = df.groupby(["window_id", "src_device"]).agg(
        src_flow_count=("is_attack", "size"),
        src_attack_count=("is_attack", "sum"),
        src_unique_destinations=("dst_device", "nunique"),
    ).reset_index().rename(columns={"src_device": "device_id"})

    # ── Create full device-window grid ──
    grid = pd.MultiIndex.from_product(
        [all_windows, all_devices],
        names=["window_id", "device_id"]
    )
    device_features = pd.DataFrame(index=grid).reset_index()

    # Merge destination features
    device_features = device_features.merge(
        dst_agg, on=["window_id", "device_id"], how="left"
    )
    device_features = device_features.merge(
        dst_counts, on=["window_id", "device_id"], how="left"
    )

    # Merge source features
    device_features = device_features.merge(
        src_counts, on=["window_id", "device_id"], how="left"
    )

    # Fill NaN with 0 (device had no traffic in this window)
    numeric_cols = device_features.select_dtypes(include=[np.number]).columns
    device_features[numeric_cols] = device_features[numeric_cols].fillna(0)

    # ── Add derived features ──
    device_features["total_flow_count"] = (
        device_features["dst_flow_count"] + device_features["src_flow_count"]
    )
    device_features["total_attack_count"] = (
        device_features["dst_attack_count"] + device_features["src_attack_count"]
    )
    device_features["is_currently_attacked"] = (device_features["dst_attack_count"] > 0).astype(int)

    print(f"[features] Generated {len(device_features):,} device-window feature rows")
    print(f"[features] Feature columns: {len(device_features.columns) - 2}")  # minus window_id, device_id

    return device_features


def create_ml_dataset(
    df_processed: pd.DataFrame,
    prediction_horizon_windows: int = 3,
) -> pd.DataFrame:
    """
    Create the complete ML-ready dataset by combining:
    1. Device-level aggregated features (per window)
    2. Target labels (per window)

    This is the final output that goes into XGBoost / GNN training.

    Args:
        df_processed: Fully preprocessed flow-level DataFrame.
        prediction_horizon_windows: Future windows to predict into.

    Returns:
        DataFrame with one row per (window_id, device_id) containing:
        - Aggregated traffic features
        - Target label (is_future_target)
        - Device metadata
    """
    print("\n" + "=" * 70)
    print("CREATING ML DATASET")
    print("=" * 70)

    # Step 1: Generate target labels
    target_df = generate_target_labels(df_processed, prediction_horizon_windows)

    # Step 2: Generate device features per window
    device_features = generate_device_features_per_window(df_processed)

    # Step 3: Merge targets with features
    ml_dataset = device_features.merge(
        target_df[["window_id", "device_id", "is_future_target",
                    "target_attack_count", "target_attack_types"]],
        on=["window_id", "device_id"],
        how="left",
    )

    # Fill any remaining NaN targets (e.g., last windows with no future data)
    ml_dataset["is_future_target"] = ml_dataset["is_future_target"].fillna(0).astype(int)

    # Step 4: Add device metadata from campus topology
    ml_dataset = _add_device_metadata(ml_dataset)

    # Step 5: Summary
    n_positive = ml_dataset["is_future_target"].sum()
    n_total = len(ml_dataset)

    print(f"\n  --- ML Dataset Summary ---")
    print(f"  Total samples:     {n_total:,}")
    print(f"  Positive targets:  {n_positive:,} ({n_positive/n_total*100:.2f}%)")
    print(f"  Feature columns:   {len(ml_dataset.columns) - 5}")  # approx
    print(f"  Time windows:      {ml_dataset['window_id'].nunique()}")
    print(f"  Devices:           {ml_dataset['device_id'].nunique()}")
    print("=" * 70)

    return ml_dataset


def _add_device_metadata(ml_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add device criticality and vulnerability from campus topology.
    """
    import json

    topology_path = PATHS["campus_topology"]
    if not topology_path.exists():
        print("[features] WARNING: Campus topology not found, skipping device metadata")
        ml_dataset["device_criticality"] = 0.5
        ml_dataset["device_vulnerability"] = 0.5
        return ml_dataset

    with open(topology_path) as f:
        topology = json.load(f)

    # Build device metadata lookup
    device_meta = {}
    for device in topology["devices"]:
        device_meta[device["id"]] = {
            "device_criticality": device.get("criticality", 0.5),
            "device_vulnerability": device.get("vulnerability", 0.5),
            "device_type": device.get("type", "unknown"),
        }

    # Map to ML dataset
    ml_dataset["device_criticality"] = ml_dataset["device_id"].map(
        lambda d: device_meta.get(d, {}).get("device_criticality", 0.3)
    )
    ml_dataset["device_vulnerability"] = ml_dataset["device_id"].map(
        lambda d: device_meta.get(d, {}).get("device_vulnerability", 0.3)
    )

    print(f"[features] Added device metadata for {len(device_meta)} topology devices")

    return ml_dataset


def create_temporal_split(
    ml_dataset: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the ML dataset temporally (no shuffling — prevents data leakage).

    Earlier windows -> train, middle -> validation, later -> test.

    Args:
        ml_dataset: Complete ML dataset with window_id.
        train_ratio: Fraction of windows for training.
        val_ratio: Fraction of windows for validation.

    Returns:
        (train_df, val_df, test_df)
    """
    all_windows = sorted(ml_dataset["window_id"].unique())
    n_windows = len(all_windows)

    train_end = int(n_windows * train_ratio)
    val_end = int(n_windows * (train_ratio + val_ratio))

    train_windows = set(all_windows[:train_end])
    val_windows = set(all_windows[train_end:val_end])
    test_windows = set(all_windows[val_end:])

    train_df = ml_dataset[ml_dataset["window_id"].isin(train_windows)].copy()
    val_df = ml_dataset[ml_dataset["window_id"].isin(val_windows)].copy()
    test_df = ml_dataset[ml_dataset["window_id"].isin(test_windows)].copy()

    print(f"\n[split] Temporal Train/Val/Test Split:")
    print(f"  Train: {len(train_df):,} samples, windows {min(train_windows)}-{max(train_windows)} "
          f"({len(train_windows)} windows)")
    print(f"  Val:   {len(val_df):,} samples, windows {min(val_windows)}-{max(val_windows)} "
          f"({len(val_windows)} windows)")
    print(f"  Test:  {len(test_df):,} samples, windows {min(test_windows)}-{max(test_windows)} "
          f"({len(test_windows)} windows)")

    # Target distribution per split
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = split_df["is_future_target"].sum()
        total = len(split_df)
        print(f"  {name} target rate: {pos}/{total} ({pos/max(total,1)*100:.2f}%)")

    return train_df, val_df, test_df


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testing target generator...")
    print("Loading processed data...")

    processed_path = PATHS["data_processed"] / "cicids2017_wednesday_processed.csv"
    if not processed_path.exists():
        print(f"[ERROR] Processed data not found: {processed_path}")
        print("Run the preprocessing pipeline first:")
        print("  python -m ml.preprocessing.pipeline --day wednesday")
        sys.exit(1)

    # Load processed data
    df = pd.read_csv(processed_path, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Loaded {len(df):,} rows")

    # Create ML dataset
    ml_dataset = create_ml_dataset(df, prediction_horizon_windows=3)

    # Temporal split
    train_df, val_df, test_df = create_temporal_split(ml_dataset)

    # Save ML dataset
    output_path = PATHS["data_processed"] / "ml_dataset_wednesday.csv"
    ml_dataset.to_csv(output_path, index=False)
    print(f"\n[OK] ML dataset saved: {output_path} ({output_path.stat().st_size/(1024*1024):.1f} MB)")
