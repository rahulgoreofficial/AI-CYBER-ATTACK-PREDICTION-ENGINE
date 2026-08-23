"""
Data Loader — Load and inspect CICIDS2017 CSV files.
=====================================================

Handles:
- Loading raw CSV files with proper encoding
- Column name normalization (strip whitespace)
- Basic schema inspection
- Memory-efficient loading for large files

Supports both:
- "MachineLearningCVE" version (79 cols, no IPs/timestamps) ← common download
- "Full" version (84+ cols, with IPs/timestamps) ← if available

Usage:
    from ml.preprocessing.loader import load_cicids2017, inspect_dataset
    df = load_cicids2017("wednesday")
    inspect_dataset(df)
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import PATHS, DATASET_CONFIG


# ==============================================================================
# CICIDS2017 FILE REGISTRY
# ==============================================================================
# Maps short names to actual filenames (handles inconsistent casing in the dataset)
CICIDS2017_FILES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_morning": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_afternoon": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_afternoon_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "friday_afternoon_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}

# Columns that indicate the "full" version (with IPs/timestamps)
FULL_VERSION_MARKERS = {"Flow ID", "Source IP", "Destination IP", "Timestamp"}


def detect_dataset_version(df: pd.DataFrame) -> str:
    """
    Detect whether the loaded CSV is the 'full' or 'ml_only' version.

    Returns:
        'full' — contains IP addresses, timestamps, Flow ID
        'ml_only' — MachineLearningCVE version (79 cols, no IPs/timestamps)
    """
    cols_stripped = {c.strip() for c in df.columns}
    if FULL_VERSION_MARKERS.issubset(cols_stripped):
        return "full"
    return "ml_only"


def load_cicids2017(day: str = "wednesday", nrows: int | None = None) -> pd.DataFrame:
    """
    Load a CICIDS2017 CSV file by day name.

    Args:
        day: Short name of the day file (e.g., "wednesday", "tuesday",
             "thursday_morning"). See CICIDS2017_FILES for all options.
        nrows: If set, only load this many rows (for quick inspection).

    Returns:
        DataFrame with normalized column names (stripped whitespace).

    Raises:
        FileNotFoundError: If the CSV file doesn't exist at the expected path.
        ValueError: If the day name is not recognized.
    """
    if day not in CICIDS2017_FILES:
        available = ", ".join(sorted(CICIDS2017_FILES.keys()))
        raise ValueError(
            f"Unknown day '{day}'. Available: {available}"
        )

    filename = CICIDS2017_FILES[day]
    filepath = PATHS["cicids2017_dir"] / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            f"Please download CICIDS2017 and place CSV files in: {PATHS['cicids2017_dir']}\n"
            f"See docs/dataset_notes.md for download instructions."
        )

    print(f"[loader] Loading {filename}...")
    print(f"[loader] File size: {filepath.stat().st_size / (1024*1024):.1f} MB")

    # Load CSV with latin-1 encoding (handles special characters in CICIDS2017)
    df = pd.read_csv(
        filepath,
        encoding="latin-1",
        nrows=nrows,
        low_memory=False,  # Avoid mixed-type warnings
    )

    # Normalize column names: strip whitespace (CICIDS2017 has leading spaces)
    df.columns = df.columns.str.strip()

    # Detect dataset version
    version = detect_dataset_version(df)
    print(f"[loader] Dataset version: {version}")
    print(f"[loader] Loaded {len(df):,} rows x {len(df.columns)} columns")

    # Store version as an attribute for downstream use
    df.attrs["dataset_version"] = version
    df.attrs["source_day"] = day

    return df


def load_multiple_days(days: list[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple CICIDS2017 day files.

    Args:
        days: List of day names (e.g., ["tuesday", "wednesday"]).

    Returns:
        Concatenated DataFrame with a 'source_file' column added.
    """
    dfs = []
    for day in days:
        df = load_cicids2017(day)
        df["source_file"] = day
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n[loader] Combined: {len(combined):,} rows from {len(days)} files")
    return combined


def inspect_dataset(df: pd.DataFrame, show_samples: int = 3) -> dict:
    """
    Inspect a loaded dataset and print comprehensive statistics.

    Args:
        df: Loaded DataFrame.
        show_samples: Number of sample rows to display.

    Returns:
        Dictionary with inspection results.
    """
    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    # Version info
    version = df.attrs.get("dataset_version", "unknown")
    print(f"\nDataset version: {version}")

    # Shape
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.1f} MB")

    # Columns
    print(f"\n--- Columns ({len(df.columns)}) ---")
    for i, col in enumerate(df.columns):
        dtype_str = str(df[col].dtype)
        print(f"  {i+1:3d}. {col:<45s} {dtype_str:<15s}")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"\n--- Missing Values ---")
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"  {col:<45s} {count:>8,} ({pct:.2f}%)")
    else:
        print("  No missing values found.")

    # Infinity values (common in CICIDS2017)
    print(f"\n--- Infinite Values ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
            print(f"  {col:<45s} {inf_count:>8,}")
    if not inf_counts:
        print("  No infinite values found.")

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n--- Duplicates ---")
    print(f"  Duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")

    # Label distribution (attack types)
    if "Label" in df.columns:
        print(f"\n--- Label Distribution ---")
        label_counts = df["Label"].value_counts()
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"  {label:<40s} {count:>10,} ({pct:.2f}%)")

    # IP addresses (only in full version)
    if "Source IP" in df.columns:
        unique_src = df["Source IP"].nunique()
        unique_dst = df["Destination IP"].nunique()
        print(f"\n--- IP Addresses ---")
        print(f"  Unique source IPs: {unique_src}")
        print(f"  Unique destination IPs: {unique_dst}")
        print(f"  Top 5 source IPs:")
        for ip, count in df["Source IP"].value_counts().head(5).items():
            print(f"    {ip:<20s} {count:>10,}")
    else:
        print(f"\n--- IP Addresses ---")
        print(f"  Not available (MachineLearningCVE version)")

    # Destination Port distribution
    if "Destination Port" in df.columns:
        print(f"\n--- Top Destination Ports ---")
        for port, count in df["Destination Port"].value_counts().head(10).items():
            print(f"    Port {port:<10} {count:>10,}")

    # Sample rows
    if show_samples > 0:
        print(f"\n--- Sample Rows ({show_samples}) ---")
        print(df.head(show_samples).to_string())

    print("\n" + "=" * 70)

    results = {
        "version": version,
        "shape": df.shape,
        "missing_columns": dict(missing_cols) if len(missing_cols) > 0 else {},
        "inf_columns": inf_counts,
        "duplicate_count": dup_count,
        "label_distribution": dict(df["Label"].value_counts()) if "Label" in df.columns else {},
    }
    return results


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testing data loader...")
    print(f"Available days: {list(CICIDS2017_FILES.keys())}")
    print(f"Dataset directory: {PATHS['cicids2017_dir']}")

    # Quick load test (first 100 rows)
    try:
        df = load_cicids2017("wednesday", nrows=100)
        results = inspect_dataset(df, show_samples=2)
        print("\n[OK] Loader works correctly.")
    except FileNotFoundError as e:
        print(f"\n[WARN] {e}")
        print("Download the dataset first. See docs/dataset_notes.md")
