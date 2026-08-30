"""
Generalization Test — CSE-CIC-IDS2018 Dataset
===============================================
M9.4: Apply the CICIDS2017-Wednesday-trained XGBoost model to
CSE-CIC-IDS2018 data (zero-shot generalization test).

Pipeline:
1. Load 02-14-2018.csv (smallest file, ~342MB)
2. Normalize column names to match CICIDS2017 feature format
3. Assign synthetic device IDs and time windows
4. Build feature matrix with the same 120 features
5. Apply trained XGBoost model (no retraining)
6. Evaluate Top-K, MRR, PR-AUC

Results saved to: experiments/generalization_2018.json

Usage:
    python -m ml.evaluation.generalization_test
    python -m ml.evaluation.generalization_test --file 02-14-2018.csv
"""

from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import PATHS, RANDOM_SEED, TIME_CONFIG

SEPARATOR = "=" * 72

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPING: IDS2018 → CICIDS2017 feature names
# ──────────────────────────────────────────────────────────────────────────────

IDS2018_COL_MAP = {
    "Dst Port":           "dst_port",
    "Protocol":           "protocol",
    "Timestamp":          "timestamp",
    "Flow Duration":      "flow_duration",
    "Tot Fwd Pkts":       "total_fwd_packets",
    "Tot Bwd Pkts":       "total_bwd_packets",
    "TotLen Fwd Pkts":    "total_len_fwd_packets",
    "TotLen Bwd Pkts":    "total_len_bwd_packets",
    "Fwd Pkt Len Max":    "fwd_pkt_len_max",
    "Fwd Pkt Len Min":    "fwd_pkt_len_min",
    "Fwd Pkt Len Mean":   "fwd_pkt_len_mean",
    "Fwd Pkt Len Std":    "fwd_pkt_len_std",
    "Bwd Pkt Len Max":    "bwd_pkt_len_max",
    "Bwd Pkt Len Min":    "bwd_pkt_len_min",
    "Bwd Pkt Len Mean":   "bwd_pkt_len_mean",
    "Bwd Pkt Len Std":    "bwd_pkt_len_std",
    "Flow Byts/s":        "flow_bytes_per_s",
    "Flow Pkts/s":        "flow_packets_per_s",
    "Flow IAT Mean":      "flow_iat_mean",
    "Flow IAT Std":       "flow_iat_std",
    "Flow IAT Max":       "flow_iat_max",
    "Flow IAT Min":       "flow_iat_min",
    "Fwd IAT Tot":        "fwd_iat_total",
    "Fwd IAT Mean":       "fwd_iat_mean",
    "Fwd IAT Std":        "fwd_iat_std",
    "Fwd IAT Max":        "fwd_iat_max",
    "Fwd IAT Min":        "fwd_iat_min",
    "Bwd IAT Tot":        "bwd_iat_total",
    "Bwd IAT Mean":       "bwd_iat_mean",
    "Bwd IAT Std":        "bwd_iat_std",
    "Bwd IAT Max":        "bwd_iat_max",
    "Bwd IAT Min":        "bwd_iat_min",
    "Fwd PSH Flags":      "fwd_psh_flags",
    "Bwd PSH Flags":      "bwd_psh_flags",
    "Fwd URG Flags":      "fwd_urg_flags",
    "Bwd URG Flags":      "bwd_urg_flags",
    "Fwd Header Len":     "fwd_header_len",
    "Bwd Header Len":     "bwd_header_len",
    "Fwd Pkts/s":         "fwd_packets_per_s",
    "Bwd Pkts/s":         "bwd_packets_per_s",
    "Pkt Len Min":        "min_pkt_len",
    "Pkt Len Max":        "max_pkt_len",
    "Pkt Len Mean":       "pkt_len_mean",
    "Pkt Len Std":        "pkt_len_std",
    "Pkt Len Var":        "pkt_len_var",
    "FIN Flag Cnt":       "fin_flag_cnt",
    "SYN Flag Cnt":       "syn_flag_cnt",
    "RST Flag Cnt":       "rst_flag_cnt",
    "PSH Flag Cnt":       "psh_flag_cnt",
    "ACK Flag Cnt":       "ack_flag_cnt",
    "URG Flag Cnt":       "urg_flag_cnt",
    "CWE Flag Count":     "cwe_flag_cnt",
    "ECE Flag Cnt":       "ece_flag_cnt",
    "Down/Up Ratio":      "down_up_ratio",
    "Pkt Size Avg":       "avg_pkt_size",
    "Fwd Seg Size Avg":   "avg_fwd_seg_size",
    "Bwd Seg Size Avg":   "avg_bwd_seg_size",
    "Fwd Byts/b Avg":     "fwd_avg_bytes_bulk",
    "Fwd Pkts/b Avg":     "fwd_avg_pkts_bulk",
    "Fwd Blk Rate Avg":   "fwd_avg_bulk_rate",
    "Bwd Byts/b Avg":     "bwd_avg_bytes_bulk",
    "Bwd Pkts/b Avg":     "bwd_avg_pkts_bulk",
    "Bwd Blk Rate Avg":   "bwd_avg_bulk_rate",
    "Subflow Fwd Pkts":   "subflow_fwd_pkts",
    "Subflow Fwd Byts":   "subflow_fwd_bytes",
    "Subflow Bwd Pkts":   "subflow_bwd_pkts",
    "Subflow Bwd Byts":   "subflow_bwd_bytes",
    "Init Fwd Win Byts":  "init_win_bytes_fwd",
    "Init Bwd Win Byts":  "init_win_bytes_bwd",
    "Fwd Act Data Pkts":  "act_data_pkt_fwd",
    "Fwd Seg Size Min":   "min_seg_size_fwd",
    "Active Mean":        "active_mean",
    "Active Std":         "active_std",
    "Active Max":         "active_max",
    "Active Min":         "active_min",
    "Idle Mean":          "idle_mean",
    "Idle Std":           "idle_std",
    "Idle Max":           "idle_max",
    "Idle Min":           "idle_min",
    "Label":              "label",
}

# Benign label in IDS2018
IDS2018_BENIGN = "Benign"

# Synthetic device pool (mimicking the Wednesday topology)
SYNTHETIC_DEVICES = [
    "PC-01", "PC-02", "PC-03", "PC-04", "PC-05",
    "PC-06", "PC-07", "PC-08", "PC-09", "PC-10",
    "WEB-SERVER-01", "FILE-SERVER-01", "DB-SERVER-01",
    "MAIL-SERVER-01", "ROUTER-01", "SWITCH-01",
    "GATEWAY-01", "DNS-SERVER-01", "MGMT-01",
    "FACULTY-WS-01", "ADMIN-WS-01",
]
DEVICE_CRITICALITY = {
    "PC-01": 0.3, "PC-02": 0.3, "PC-03": 0.3, "PC-04": 0.3, "PC-05": 0.3,
    "PC-06": 0.3, "PC-07": 0.3, "PC-08": 0.3, "PC-09": 0.3, "PC-10": 0.3,
    "WEB-SERVER-01": 0.9, "FILE-SERVER-01": 0.85, "DB-SERVER-01": 0.95,
    "MAIL-SERVER-01": 0.7, "ROUTER-01": 0.8, "SWITCH-01": 0.6,
    "GATEWAY-01": 0.85, "DNS-SERVER-01": 0.75, "MGMT-01": 0.9,
    "FACULTY-WS-01": 0.5, "ADMIN-WS-01": 0.65,
}
DEVICE_VULNERABILITY = {d: np.random.RandomState(i).uniform(0.2, 0.9)
                        for i, d in enumerate(SYNTHETIC_DEVICES)}


# ──────────────────────────────────────────────────────────────────────────────
# LOAD + PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────

def load_ids2018(filepath: Path, max_rows: int = 300_000) -> pd.DataFrame:
    """Load and normalize CSE-CIC-IDS2018 CSV."""
    print(f"  Loading: {filepath.name} (up to {max_rows:,} rows)...")
    df = pd.read_csv(filepath, nrows=max_rows, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Rename columns
    df.rename(columns=IDS2018_COL_MAP, inplace=True)

    # Drop unmapped columns
    mapped_cols = list(IDS2018_COL_MAP.values())
    df = df[[c for c in df.columns if c in mapped_cols]]

    print(f"  After rename: {df.shape[1]} columns retained")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    return df


def create_attack_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary 'is_attack' column."""
    df = df.copy()
    df["is_attack"] = (df["label"].str.lower() != "benign").astype(int)
    print(f"  Attack flows: {df['is_attack'].sum():,} / {len(df):,} "
          f"({df['is_attack'].mean() * 100:.1f}%)")
    return df


def assign_windows(df: pd.DataFrame, window_size_minutes: int = 5) -> pd.DataFrame:
    """Assign time windows. Parse timestamp or use row order."""
    df = df.copy()

    # Try to parse timestamp
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True,
                                          errors="coerce")
        valid_ts = df["timestamp"].notna().sum()
        print(f"  Valid timestamps: {valid_ts:,}/{len(df):,}")

        if valid_ts > len(df) * 0.5:
            df.sort_values("timestamp", inplace=True)
            t_min = df["timestamp"].min()
            df["minutes_elapsed"] = (df["timestamp"] - t_min).dt.total_seconds() / 60
            df["window_id"] = (df["minutes_elapsed"] // window_size_minutes).astype(int)
        else:
            raise ValueError("Not enough valid timestamps")
    except Exception:
        # Fallback: assign windows by row order
        print("  [WARN] Timestamp parse failed — assigning windows by row order")
        rows_per_window = max(len(df) // 50, 100)
        df["window_id"] = np.arange(len(df)) // rows_per_window

    n_windows = df["window_id"].nunique()
    print(f"  Windows: {n_windows}")
    return df


def assign_devices(df: pd.DataFrame) -> pd.DataFrame:
    """Assign synthetic device IDs round-robin per window."""
    df = df.copy()
    rng = np.random.RandomState(RANDOM_SEED)

    # Assign devices based on destination port ranges (realistic heuristic)
    def port_to_device(row):
        port = row.get("dst_port", 0)
        is_atk = row.get("is_attack", 0)
        if port in range(80, 91) or port == 443:
            return "WEB-SERVER-01"
        elif port in (21, 22):
            return "FILE-SERVER-01" if is_atk else rng.choice(["PC-01", "PC-02"])
        elif port in (3306, 5432, 1433):
            return "DB-SERVER-01"
        elif port == 25:
            return "MAIL-SERVER-01"
        elif port in (53,):
            return "DNS-SERVER-01"
        elif port < 1024:
            return rng.choice(["WEB-SERVER-01", "FILE-SERVER-01", "ROUTER-01"])
        else:
            return rng.choice(SYNTHETIC_DEVICES)

    # For speed on large DFs, vectorize via port buckets
    dst_port = df.get("dst_port", pd.Series(np.zeros(len(df)))).fillna(0).astype(int)
    is_atk = df.get("is_attack", pd.Series(np.zeros(len(df)))).fillna(0).astype(int)

    devices = []
    for p, a in zip(dst_port.values, is_atk.values):
        if p in range(80, 91) or p == 443:
            devices.append("WEB-SERVER-01")
        elif p in (21, 22) and a:
            devices.append("FILE-SERVER-01")
        elif p in (3306, 5432, 1433):
            devices.append("DB-SERVER-01")
        elif p == 25:
            devices.append("MAIL-SERVER-01")
        elif p == 53:
            devices.append("DNS-SERVER-01")
        else:
            devices.append(SYNTHETIC_DEVICES[rng.randint(0, len(SYNTHETIC_DEVICES))])

    df["device_id"] = devices
    print(f"  Device distribution:\n    {pd.Series(devices).value_counts().head(8).to_dict()}")
    return df


def add_device_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add criticality, vulnerability, VLAN placeholders."""
    df = df.copy()
    df["device_criticality"]  = df["device_id"].map(DEVICE_CRITICALITY).fillna(0.5)
    df["device_vulnerability"] = df["device_id"].map(DEVICE_VULNERABILITY).fillna(0.5)
    df["vlan_dmz"]     = df["device_id"].str.contains("SERVER|GATEWAY|ROUTER").astype(int)
    df["vlan_student"] = df["device_id"].str.startswith("PC").astype(int)
    df["vlan_admin"]   = df["device_id"].str.contains("ADMIN|MGMT").astype(int)
    df["vlan_faculty"] = df["device_id"].str.contains("FACULTY").astype(int)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE AGGREGATION (mirror the Wednesday feature combiner)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_per_window_device(df: pd.DataFrame, traffic_cols: list[str]) -> pd.DataFrame:
    """Aggregate raw flows into per-window-device feature vectors."""
    agg_funcs = {c: ["sum", "mean", "std", "max", "min", "count"]
                 for c in traffic_cols if c in df.columns}

    if not agg_funcs:
        raise ValueError("No traffic columns found for aggregation.")

    agg_df = (
        df.groupby(["window_id", "device_id"])
        .agg(agg_funcs)
        .reset_index()
    )

    # Flatten multi-level columns
    agg_df.columns = [
        f"{col}_{stat}" if stat else col
        for col, stat in agg_df.columns
    ]

    # Add attack flag aggregation (is_future_target)
    attack_agg = (
        df.groupby(["window_id", "device_id"])["is_attack"]
        .agg(["sum", "max"])
        .reset_index()
    )
    attack_agg.columns = ["window_id", "device_id", "attack_count", "is_attacked"]

    agg_df = agg_df.merge(attack_agg, on=["window_id", "device_id"], how="left")

    # Build future target label: device is attacked in this OR next window
    agg_df = agg_df.sort_values(["window_id", "device_id"]).reset_index(drop=True)
    all_windows = sorted(agg_df["window_id"].unique())

    # Create a lookup: (window_id, device_id) → is_attacked
    attacked_lookup = {
        (r["window_id"], r["device_id"]): r["is_attacked"]
        for _, r in agg_df.iterrows()
    }

    horizon_windows = max(1, TIME_CONFIG["prediction_horizon_minutes"] //
                          TIME_CONFIG["window_size_minutes"])

    future_targets = []
    for _, row in agg_df.iterrows():
        w = row["window_id"]
        dev = row["device_id"]
        is_target = 0
        for dw in range(1, horizon_windows + 1):
            if attacked_lookup.get((w + dw, dev), 0):
                is_target = 1
                break
        future_targets.append(is_target)

    agg_df["is_future_target"] = future_targets

    print(f"  Aggregated: {agg_df.shape[0]} window-device pairs, "
          f"{agg_df['is_future_target'].sum()} future targets")

    return agg_df


def add_graph_features(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic graph features (since we don't have NetworkX graph for IDS2018)."""
    df = agg_df.copy()

    # Degree: proxy via count of flows
    count_col = [c for c in df.columns if c.endswith("_count")]
    if count_col:
        count_vals = df[count_col[0]].fillna(0)
        df["degree"] = (count_vals / count_vals.max().clip(1)).clip(0, 1)
    else:
        df["degree"] = 0.5

    # Betweenness: proxy via server/gateway devices
    df["betweenness_centrality"] = df["device_id"].isin(
        ["WEB-SERVER-01", "FILE-SERVER-01", "ROUTER-01", "GATEWAY-01", "DB-SERVER-01"]
    ).astype(float) * 0.8

    df["closeness_centrality"] = df["device_criticality"] * 0.7
    df["pagerank"] = df["device_criticality"] * 0.6
    df["in_degree"] = df["degree"] * 0.5
    df["out_degree"] = df["degree"] * 0.5

    # Neighbor attack count (proxy: devices in same VLAN that are attacked)
    df["neighbor_attack_count"] = 0.0
    df["attack_neighbor_count"] = 0.0

    return df


def build_feature_matrix_2018(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix builder for IDS2018 data."""
    # Traffic feature columns (numeric, not label/metadata)
    exclude = {"timestamp", "label", "is_attack", "window_id", "device_id",
               "minutes_elapsed", "flow_id"}
    traffic_cols = [
        c for c in df_raw.columns
        if c not in exclude and df_raw[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    print(f"  Traffic cols for aggregation: {len(traffic_cols)}")

    # Aggregate
    agg_df = aggregate_per_window_device(df_raw, traffic_cols)

    # Add device metadata
    agg_df["device_criticality"] = agg_df["device_id"].map(DEVICE_CRITICALITY).fillna(0.5)
    agg_df["device_vulnerability"] = agg_df["device_id"].map(DEVICE_VULNERABILITY).fillna(0.5)
    agg_df["vlan_dmz"]     = agg_df["device_id"].str.contains("SERVER|GATEWAY|ROUTER").astype(int)
    agg_df["vlan_student"] = agg_df["device_id"].str.startswith("PC").astype(int)

    # Add graph features
    agg_df = add_graph_features(agg_df)

    # Fill NaN
    numeric_cols = agg_df.select_dtypes(include=[np.number]).columns
    agg_df[numeric_cols] = agg_df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

    return agg_df


# ──────────────────────────────────────────────────────────────────────────────
# ZERO-SHOT PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_zero_shot(
    feat_df: pd.DataFrame,
    model_feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply trained XGBoost model to the IDS2018 feature matrix.

    Handles feature mismatch by:
    - Adding missing features as 0
    - Dropping extra features
    """
    xgb_model = joblib.load(PATHS["models"] / "xgboost_baseline.pkl")
    print(f"  XGBoost expects {len(model_feature_cols)} features")

    # Align features
    available = set(feat_df.columns)
    missing = [c for c in model_feature_cols if c not in available]
    extra   = [c for c in feat_df.columns if c not in model_feature_cols
               and c not in {"window_id", "device_id", "is_future_target",
                              "attack_count", "is_attacked"}]

    print(f"  Features matched : {len(model_feature_cols) - len(missing)}")
    print(f"  Features missing : {len(missing)} (will fill with 0)")
    print(f"  Features extra   : {len(extra)} (will ignore)")

    # Build aligned matrix
    X = pd.DataFrame(index=feat_df.index)
    for col in model_feature_cols:
        if col in feat_df.columns:
            X[col] = feat_df[col].values
        else:
            X[col] = 0.0

    X = X.fillna(0).replace([np.inf, -np.inf], 0).values

    probs = xgb_model.predict_proba(X)[:, 1]
    y_true = feat_df["is_future_target"].values
    device_ids = feat_df["device_id"].values
    window_ids = feat_df["window_id"].values

    return probs, y_true, device_ids, window_ids


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_generalization_test(filename: str = "02-14-2018.csv") -> dict:
    print(f"\n{SEPARATOR}")
    print("  GENERALIZATION TEST — CSE-CIC-IDS2018")
    print(f"  File: {filename}")
    print(SEPARATOR)

    filepath = PATHS["data_raw"] / "CSE-CIC-IDS2018" / filename
    if not filepath.exists():
        # Try alternate casing
        alt = PATHS["data_raw"] / "cse-cic-ids2018" / filename
        if alt.exists():
            filepath = alt
        else:
            raise FileNotFoundError(f"IDS2018 file not found: {filepath}")

    print("\n[1/5] Loading IDS2018 data...")
    df = load_ids2018(filepath)
    df = create_attack_label(df)
    df = assign_windows(df, window_size_minutes=TIME_CONFIG["window_size_minutes"])
    df = assign_devices(df)

    print("\n[2/5] Building feature matrix...")
    feat_df = build_feature_matrix_2018(df)
    print(f"  Feature matrix: {feat_df.shape}")

    print("\n[3/5] Loading trained model feature list...")
    feat_cols_path = PATHS["data_processed"] / "cicids2017_wednesday_feature_cols.txt"
    if feat_cols_path.exists():
        with open(feat_cols_path) as f:
            base_traffic_cols = [line.strip() for line in f if line.strip()]
    else:
        base_traffic_cols = []

    # Load the full feature column list from the Wednesday feature matrix
    wed_fm_path = PATHS["data_processed"] / "ml_feature_matrix_wednesday.csv"
    if wed_fm_path.exists():
        wed_df_sample = pd.read_csv(wed_fm_path, nrows=5)
        exclude_cols = {
            "window_id", "device_id", "is_future_target",
            "target_attack_count", "target_attack_types", "earliest_target_window",
        }
        model_feature_cols = [
            c for c in wed_df_sample.columns
            if c not in exclude_cols
            and wed_df_sample[c].dtype in [np.float64, np.int64, np.float32, np.int32,
                                            np.uint8, bool]
        ]
        print(f"  Wednesday model uses {len(model_feature_cols)} features")
    else:
        raise FileNotFoundError("Wednesday feature matrix not found — run Phase 2–3 first.")

    print("\n[4/5] Running zero-shot prediction...")
    probs, y_true, device_ids, window_ids = predict_zero_shot(feat_df, model_feature_cols)

    print(f"\n  Test set: {len(y_true):,} samples, {y_true.sum():.0f} positive "
          f"({y_true.mean()*100:.1f}% attack targets)")

    print("\n[5/5] Evaluating...")
    from ml.evaluation.metrics import evaluate_predictions
    results = evaluate_predictions(
        y_true, probs, device_ids, window_ids,
        model_name="XGBoost [CICIDS2017→IDS2018 zero-shot]",
        verbose=True,
    )

    # Attack label distribution
    label_counts = df["label"].value_counts().to_dict()

    # Package output
    output = {
        "dataset": "CSE-CIC-IDS2018",
        "source_file": filename,
        "rows_loaded": int(len(df)),
        "windows": int(feat_df["window_id"].nunique()),
        "device_count": int(feat_df["device_id"].nunique()),
        "attack_types": label_counts,
        "positive_rate": round(float(y_true.mean()), 4),
        "model": "XGBoost trained on CICIDS2017 Wednesday (zero-shot)",
        "features_matched": sum(1 for c in model_feature_cols if c in feat_df.columns),
        "features_total": len(model_feature_cols),
        "results": {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in results.items() if k != "model"},
        "notes": (
            "Zero-shot generalization: model trained on CICIDS2017 Wednesday, "
            "applied without retraining to CSE-CIC-IDS2018. "
            "Device IDs are synthetic (assigned by destination port heuristic). "
            "Missing features filled with 0."
        ),
    }

    # Save
    out_path = PATHS["experiments"] / "generalization_2018.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS2018 generalization test")
    parser.add_argument("--file", default="02-14-2018.csv",
                        help="IDS2018 CSV filename (default: 02-14-2018.csv)")
    args = parser.parse_args()

    result = run_generalization_test(filename=args.file)

    print(f"\n{SEPARATOR}")
    print("  GENERALIZATION TEST COMPLETE")
    print(f"  Top-1 Accuracy : {result['results'].get('top_1_hit_rate', 0):.3f}")
    print(f"  PR-AUC         : {result['results'].get('pr_auc', 0):.3f}")
    print(f"  F1             : {result['results'].get('f1', 0):.3f}")
    print(SEPARATOR)
