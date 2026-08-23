"""
Data Cleaner — Clean and validate CICIDS2017 data.
====================================================

Handles:
- Remove duplicate rows
- Fix data types (numeric columns stored as objects)
- Handle NaN values
- Handle Infinity values
- Remove invalid/corrupt rows
- Normalize column names to snake_case
- Parse and validate timestamps
- Sort by timestamp

Usage:
    from ml.preprocessing.cleaner import clean_dataframe
    df_clean = clean_dataframe(df_raw)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import DATASET_CONFIG


# ==============================================================================
# COLUMN NAME STANDARDIZATION
# ==============================================================================
# Map raw CICIDS2017 column names (after stripping whitespace) to clean snake_case
COLUMN_MAP = {
    "Flow ID": "flow_id",
    "Source IP": "src_ip",
    "Source Port": "src_port",
    "Destination IP": "dst_ip",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
    "Timestamp": "timestamp",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_len_fwd_packets",
    "Total Length of Bwd Packets": "total_len_bwd_packets",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_total",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_total",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Bwd PSH Flags": "bwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Bwd URG Flags": "bwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_packets_per_s",
    "Bwd Packets/s": "bwd_packets_per_s",
    "Min Packet Length": "min_pkt_len",
    "Max Packet Length": "max_pkt_len",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwe_flag_cnt",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "avg_pkt_size",
    "Avg Fwd Segment Size": "avg_fwd_seg_size",
    "Avg Bwd Segment Size": "avg_bwd_seg_size",
    "Fwd Header Length.1": "fwd_header_len_1",
    "Fwd Avg Bytes/Bulk": "fwd_avg_bytes_bulk",
    "Fwd Avg Packets/Bulk": "fwd_avg_pkts_bulk",
    "Fwd Avg Bulk Rate": "fwd_avg_bulk_rate",
    "Bwd Avg Bytes/Bulk": "bwd_avg_bytes_bulk",
    "Bwd Avg Packets/Bulk": "bwd_avg_pkts_bulk",
    "Bwd Avg Bulk Rate": "bwd_avg_bulk_rate",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_bytes",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_bytes",
    "Init_Win_bytes_forward": "init_win_bytes_fwd",
    "Init_Win_bytes_backward": "init_win_bytes_bwd",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_fwd",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
    "Label": "label",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename CICIDS2017 columns to standardized snake_case names.
    Columns not in the mapping are converted to snake_case automatically.
    """
    # First strip whitespace (already done in loader, but safety net)
    df.columns = df.columns.str.strip()

    # Apply known mapping
    renamed = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            renamed[col] = COLUMN_MAP[col]
        else:
            # Auto-convert unknown columns to snake_case
            clean = col.lower().replace(" ", "_").replace("/", "_per_").replace(".", "_")
            renamed[col] = clean

    df = df.rename(columns=renamed)
    print(f"[cleaner] Renamed {len(renamed)} columns to snake_case")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    print(f"[cleaner] Removed {removed:,} duplicate rows ({removed/before*100:.2f}%)")
    return df.reset_index(drop=True)


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to proper numeric types.
    CICIDS2017 sometimes stores numeric values as strings.
    """
    # Identify columns that should be numeric
    exclude_cols = {"flow_id", "src_ip", "dst_ip", "timestamp", "label", "source_file"}
    numeric_candidates = [c for c in df.columns if c not in exclude_cols]

    converted = 0
    for col in numeric_candidates:
        if df[col].dtype == object:
            # Try converting to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted += 1

    if converted > 0:
        print(f"[cleaner] Converted {converted} columns to numeric types")

    # Ensure port columns are integer-compatible
    for port_col in ["src_port", "dst_port"]:
        if port_col in df.columns:
            df[port_col] = pd.to_numeric(df[port_col], errors="coerce")

    # Ensure protocol is integer
    if "protocol" in df.columns:
        df["protocol"] = pd.to_numeric(df["protocol"], errors="coerce")

    return df


def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values with NaN, then handle them.
    CICIDS2017 has Inf values in flow_bytes_per_s and flow_packets_per_s.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count_before = 0

    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        inf_count = inf_mask.sum()
        if inf_count > 0:
            inf_count_before += inf_count
            # Replace Inf with NaN (will be handled in handle_missing_values)
            df.loc[inf_mask, col] = np.nan

    if inf_count_before > 0:
        print(f"[cleaner] Replaced {inf_count_before:,} infinite values with NaN")
    else:
        print(f"[cleaner] No infinite values found")

    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop_rows") -> pd.DataFrame:
    """
    Handle missing (NaN) values.

    Args:
        df: DataFrame with potential NaN values.
        strategy: How to handle NaN:
            - "drop_rows": Drop rows with any NaN in numeric columns
            - "fill_zero": Fill NaN with 0
            - "fill_median": Fill NaN with column median

    Returns:
        DataFrame with NaN handled.
    """
    missing_before = df.isnull().sum().sum()

    if missing_before == 0:
        print(f"[cleaner] No missing values to handle")
        return df

    print(f"[cleaner] Total missing values: {missing_before:,}")

    if strategy == "drop_rows":
        before = len(df)
        # Only drop based on numeric columns having NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df.dropna(subset=numeric_cols)
        dropped = before - len(df)
        print(f"[cleaner] Dropped {dropped:,} rows with NaN values ({dropped/before*100:.2f}%)")
        df = df.reset_index(drop=True)

    elif strategy == "fill_zero":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        print(f"[cleaner] Filled NaN with 0")

    elif strategy == "fill_median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        print(f"[cleaner] Filled NaN with column medians")

    return df


def handle_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle negative values in columns that should be non-negative.
    Flow durations, packet counts, byte counts should not be negative.
    """
    non_negative_cols = [
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "total_len_fwd_packets", "total_len_bwd_packets",
    ]

    fixed = 0
    for col in non_negative_cols:
        if col in df.columns:
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                df.loc[neg_mask, col] = 0
                fixed += neg_count

    if fixed > 0:
        print(f"[cleaner] Fixed {fixed:,} negative values (set to 0)")

    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the timestamp column into proper datetime objects.
    CICIDS2017 uses various date formats — try multiple parsers.
    """
    if "timestamp" not in df.columns:
        print("[cleaner] WARNING: No 'timestamp' column found")
        return df

    # Try parsing with common formats
    # CICIDS2017 typically uses: "dd/mm/yyyy hh:mm" or "dd/mm/yyyy hh:mm:ss"
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, format="mixed")
        valid = df["timestamp"].notna().sum()
        print(f"[cleaner] Parsed {valid:,}/{len(df):,} timestamps successfully")
    except Exception as e:
        print(f"[cleaner] WARNING: Timestamp parsing failed: {e}")
        # Fallback: try without format specification
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
        valid = df["timestamp"].notna().sum()
        invalid = len(df) - valid
        print(f"[cleaner] Parsed {valid:,} timestamps, {invalid:,} failed (coerced to NaT)")

    # Drop rows where timestamp couldn't be parsed
    nat_count = df["timestamp"].isna().sum()
    if nat_count > 0:
        df = df.dropna(subset=["timestamp"])
        print(f"[cleaner] Dropped {nat_count:,} rows with unparseable timestamps")
        df = df.reset_index(drop=True)

    return df


def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the DataFrame by timestamp (chronological order)."""
    if "timestamp" not in df.columns:
        print("[cleaner] WARNING: Cannot sort — no 'timestamp' column")
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    time_range = df["timestamp"].max() - df["timestamp"].min()
    print(f"[cleaner] Sorted by timestamp")
    print(f"[cleaner] Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"[cleaner] Duration: {time_range}")
    return df


def create_attack_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary 'is_attack' column from the label column.
    BENIGN = 0, any attack = 1.
    """
    if "label" not in df.columns:
        print("[cleaner] WARNING: No 'label' column found")
        return df

    benign_label = DATASET_CONFIG["benign_label"]
    df["is_attack"] = (df["label"] != benign_label).astype(int)

    attack_count = df["is_attack"].sum()
    benign_count = len(df) - attack_count
    ratio = benign_count / max(attack_count, 1)
    print(f"[cleaner] Attack flag: {attack_count:,} attacks, {benign_count:,} benign (ratio {ratio:.1f}:1)")

    return df


def clean_dataframe(
    df: pd.DataFrame,
    nan_strategy: str = "drop_rows"
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a raw CICIDS2017 DataFrame.

    Pipeline steps (in order):
    1. Rename columns to snake_case
    2. Remove duplicates
    3. Fix data types
    4. Handle infinity values
    5. Handle missing values
    6. Handle negative values
    7. Parse timestamps
    8. Sort by timestamp
    9. Create attack flag

    Args:
        df: Raw DataFrame from loader.
        nan_strategy: How to handle NaN ("drop_rows", "fill_zero", "fill_median").

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    print("\n" + "=" * 70)
    print("CLEANING PIPELINE")
    print("=" * 70)

    initial_rows = len(df)

    # Step 1: Rename columns
    df = rename_columns(df)

    # Step 2: Remove duplicates
    df = remove_duplicates(df)

    # Step 3: Fix data types
    df = fix_data_types(df)

    # Step 4: Handle infinities
    df = handle_infinities(df)

    # Step 5: Handle missing values
    df = handle_missing_values(df, strategy=nan_strategy)

    # Step 6: Handle negative values
    df = handle_negative_values(df)

    # Step 7: Parse timestamps
    df = parse_timestamps(df)

    # Step 8: Sort by timestamp
    df = sort_by_timestamp(df)

    # Step 9: Create attack flag
    df = create_attack_flag(df)

    # Summary
    final_rows = len(df)
    removed = initial_rows - final_rows
    print(f"\n--- Cleaning Summary ---")
    print(f"  Initial rows:  {initial_rows:,}")
    print(f"  Final rows:    {final_rows:,}")
    print(f"  Rows removed:  {removed:,} ({removed/initial_rows*100:.2f}%)")
    print(f"  Columns:       {len(df.columns)}")
    print("=" * 70)

    return df


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    from ml.preprocessing.loader import load_cicids2017

    print("Testing cleaner with Wednesday data (first 1000 rows)...")
    df_raw = load_cicids2017("wednesday", nrows=1000)
    df_clean = clean_dataframe(df_raw)

    print(f"\nCleaned columns: {list(df_clean.columns)}")
    print(f"Dtypes:\n{df_clean.dtypes}")
    print(f"\n[OK] Cleaner works correctly.")
