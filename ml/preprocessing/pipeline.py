"""
Pipeline Orchestrator — Run the full preprocessing pipeline.
=============================================================

Orchestrates:
1. Load raw data
2. Detect dataset version (full vs ml_only)
3. Enrich if needed (add IPs, timestamps, devices)
4. Clean data
5. Create time windows
6. Save processed output

Usage:
    from ml.preprocessing.pipeline import run_pipeline
    df = run_pipeline(day="wednesday")

    # Or from command line:
    python -m ml.preprocessing.pipeline --day wednesday
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import PATHS, TIME_CONFIG, RANDOM_SEED, DATASET_CONFIG
from ml.preprocessing.loader import load_cicids2017, inspect_dataset
from ml.preprocessing.cleaner import clean_dataframe
from ml.preprocessing.synthetic_enrichment import enrich_ml_only_dataset


def create_time_windows(df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """
    Assign each flow to a time window based on its timestamp.

    Creates:
    - window_id: Integer index for each time window
    - window_start: Start time of the window
    - window_end: End time of the window

    Args:
        df: Cleaned DataFrame with parsed 'timestamp' column.
        window_minutes: Size of each time window in minutes.

    Returns:
        DataFrame with window columns added.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column.")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise ValueError("'timestamp' column must be datetime type.")

    # Get the global start time (rounded down to nearest window boundary)
    global_start = df["timestamp"].min().floor(f"{window_minutes}min")

    # Calculate window_id for each row
    time_deltas = (df["timestamp"] - global_start).dt.total_seconds()
    window_seconds = window_minutes * 60
    df["window_id"] = (time_deltas // window_seconds).astype(int)

    # Calculate window boundaries
    df["window_start"] = global_start + pd.to_timedelta(df["window_id"] * window_seconds, unit="s")
    df["window_end"] = df["window_start"] + pd.Timedelta(minutes=window_minutes)

    n_windows = df["window_id"].nunique()
    total_duration = df["timestamp"].max() - df["timestamp"].min()

    print(f"\n[pipeline] Time Windows:")
    print(f"  Window size: {window_minutes} minutes")
    print(f"  Total windows: {n_windows}")
    print(f"  Total duration: {total_duration}")
    print(f"  First window: {df['window_start'].min()}")
    print(f"  Last window:  {df['window_end'].max()}")

    # Show window statistics
    window_stats = df.groupby("window_id").agg(
        flow_count=("timestamp", "size"),
        attack_count=("is_attack", "sum"),
    )
    print(f"\n  Flows per window: min={window_stats['flow_count'].min():,}, "
          f"max={window_stats['flow_count'].max():,}, "
          f"mean={window_stats['flow_count'].mean():,.0f}")

    attack_windows = (window_stats["attack_count"] > 0).sum()
    print(f"  Windows with attacks: {attack_windows} / {n_windows} "
          f"({attack_windows/n_windows*100:.1f}%)")

    return df


def select_key_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Identify and organize columns into categories for downstream use.

    Returns:
        Tuple of (DataFrame, list of feature column names).
    """
    # Core identification columns (not features)
    id_cols = {
        "timestamp", "src_ip", "dst_ip", "src_device", "dst_device",
        "src_port", "dst_port", "protocol", "protocol_name", "service",
        "flow_id", "label", "is_attack", "source_file",
        "window_id", "window_start", "window_end",
        "is_synthetic_metadata",
    }

    # Feature columns (everything else that's numeric)
    feature_cols = [
        col for col in df.columns
        if col not in id_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    # Keep only id cols that exist
    existing_id_cols = [c for c in id_cols if c in df.columns]

    print(f"\n[pipeline] Column Categories:")
    print(f"  ID/metadata columns:  {len(existing_id_cols)}")
    print(f"  Feature columns:      {len(feature_cols)}")
    print(f"  Total columns:        {len(df.columns)}")

    return df, feature_cols


def run_pipeline(
    day: str = "wednesday",
    nrows: int | None = None,
    window_minutes: int | None = None,
    nan_strategy: str = "drop_rows",
    save: bool = True,
) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Handles both:
    - 'full' version (with IPs/timestamps) → clean directly
    - 'ml_only' version (MachineLearningCVE) → enrich first, then clean

    Args:
        day: Which CICIDS2017 day to process.
        nrows: Limit rows for testing (None = all).
        window_minutes: Override window size from config.
        nan_strategy: How to handle NaN values.
        save: Whether to save processed output to CSV.

    Returns:
        Fully preprocessed DataFrame.
    """
    print("\n" + "=" * 70)
    print(f"PREPROCESSING PIPELINE — {day.upper()}")
    print("=" * 70)

    # Use config defaults if not overridden
    if window_minutes is None:
        window_minutes = TIME_CONFIG["window_size_minutes"]

    # ── Step 1: Load raw data ──
    df = load_cicids2017(day, nrows=nrows)
    version = df.attrs.get("dataset_version", "unknown")

    # ── Step 2: Inspect raw data ──
    print("\n--- Raw Data Quick Stats ---")
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
    label_col = "Label" if "Label" in df.columns else "label"
    if label_col in df.columns:
        attack_count = (df[label_col] != DATASET_CONFIG["benign_label"]).sum()
        print(f"  Attack flows: {attack_count:,} ({attack_count/len(df)*100:.2f}%)")

    # ── Step 3: Enrich if ML-only version ──
    if version == "ml_only":
        print(f"\n[pipeline] Detected ML-only version — applying synthetic enrichment")
        df = enrich_ml_only_dataset(df, day=day)
    else:
        print(f"\n[pipeline] Full version detected — using native IPs/timestamps")

    # ── Step 4: Clean ──
    df = clean_dataframe(df, nan_strategy=nan_strategy)

    # ── Step 5: Time windows ──
    df = create_time_windows(df, window_minutes=window_minutes)

    # ── Step 6: Organize columns ──
    df, feature_cols = select_key_columns(df)

    # ── Step 7: Save ──
    if save:
        output_path = PATHS["data_processed"] / f"cicids2017_{day}_processed.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[pipeline] Saved: {output_path} ({size_mb:.1f} MB)")

        # Save feature column list
        feature_path = PATHS["data_processed"] / f"cicids2017_{day}_feature_cols.txt"
        with open(feature_path, "w") as f:
            f.write("\n".join(feature_cols))
        print(f"[pipeline] Saved feature list: {feature_path} ({len(feature_cols)} features)")

    # ── Final summary ──
    print(f"\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — {day.upper()}")
    print(f"  Rows:           {len(df):,}")
    print(f"  Columns:        {len(df.columns)}")
    print(f"  Features:       {len(feature_cols)}")
    print(f"  Time windows:   {df['window_id'].nunique()}")
    unique_devices = set(df["src_device"].unique()) | set(df["dst_device"].unique())
    print(f"  Unique devices: {len(unique_devices)}")
    print(f"  Attack ratio:   {df['is_attack'].mean()*100:.2f}%")
    if version == "ml_only":
        print(f"  [!] Note: IPs/timestamps are SYNTHETIC (MachineLearningCVE version)")
    print(f"=" * 70)

    return df


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CICIDS2017 preprocessing pipeline")
    parser.add_argument("--day", default="wednesday", help="Day to process")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows (for testing)")
    parser.add_argument("--window", type=int, default=None, help="Window size in minutes")
    parser.add_argument("--nan-strategy", default="drop_rows",
                        choices=["drop_rows", "fill_zero", "fill_median"])
    parser.add_argument("--no-save", action="store_true", help="Don't save output")
    args = parser.parse_args()

    df = run_pipeline(
        day=args.day,
        nrows=args.nrows,
        window_minutes=args.window,
        nan_strategy=args.nan_strategy,
        save=not args.no_save,
    )
