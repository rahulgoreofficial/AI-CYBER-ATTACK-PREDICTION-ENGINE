"""
AI Cyber Attack Prediction Engine — Central Configuration
=========================================================
All project-wide constants, paths, hyperparameters, and settings.
Import this module wherever you need configuration values.

Usage:
    from ml.config import CONFIG, PATHS, FEATURE_COLS
"""

import os
from pathlib import Path

# ==============================================================================
# PROJECT ROOT
# ==============================================================================
# Resolve project root relative to this file's location (ml/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================
PATHS = {
    # Data directories
    "data_raw": PROJECT_ROOT / "data" / "raw",
    "data_processed": PROJECT_ROOT / "data" / "processed",
    "data_synthetic": PROJECT_ROOT / "data" / "synthetic",

    # ML artifacts
    "models": PROJECT_ROOT / "models",
    "experiments": PROJECT_ROOT / "experiments",

    # Graph outputs
    "graph_output": PROJECT_ROOT / "graph" / "output",

    # Topology
    "campus_topology": PROJECT_ROOT / "data" / "synthetic" / "campus_topology.json",

    # Dataset files (CICIDS2017 — user must download)
    "cicids2017_dir": PROJECT_ROOT / "data" / "raw" / "cicids2017",
    "cse_cic_ids2018_dir": PROJECT_ROOT / "data" / "raw" / "cse-cic-ids2018",
}

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
DATASET_CONFIG = {
    # Which CICIDS2017 day files to process (start with Wednesday)
    "cicids2017_files": {
        "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
        "thursday": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "friday": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    },
    "primary_file": "wednesday",  # Start with this for first pipeline

    # Column name mapping (CICIDS2017 has inconsistent spacing)
    "column_rename": {
        " Destination Port": "dst_port",
        " Flow Duration": "flow_duration",
        " Total Fwd Packets": "total_fwd_packets",
        " Total Backward Packets": "total_bwd_packets",
        "Total Length of Fwd Packets": "total_len_fwd_packets",
        " Total Length of Bwd Packets": "total_len_bwd_packets",
        " Flow Bytes/s": "flow_bytes_per_s",
        " Flow Packets/s": "flow_packets_per_s",
        " Label": "label",
        " Timestamp": "timestamp",
        "Flow ID": "flow_id",
        " Source IP": "src_ip",
        " Source Port": "src_port",
        " Destination IP": "dst_ip",
        " Protocol": "protocol",
    },

    # Attack label for benign traffic
    "benign_label": "BENIGN",
}

# ==============================================================================
# TIME WINDOW CONFIGURATION
# ==============================================================================
TIME_CONFIG = {
    "window_size_minutes": 5,           # Size of each time window
    "prediction_horizon_minutes": 15,   # How far ahead to predict (3 windows)
    "lookback_windows": 3,              # How many past windows for features
}

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================
MODEL_CONFIG = {
    # XGBoost
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": 1,  # Will be adjusted for class imbalance
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    },

    # Isolation Forest
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": "auto",
        "random_state": 42,
        "n_jobs": -1,
    },

    # GNN (PyTorch Geometric)
    "gnn": {
        "hidden_channels": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 100,
        "patience": 10,  # Early stopping
    },
}

# ==============================================================================
# DYNAMIC RISK ENGINE WEIGHTS
# ==============================================================================
# Initial equal weights — NOT scientifically optimized
# These are a transparent starting point
RISK_WEIGHTS = {
    "attack_probability": 1 / 6,
    "anomaly_score": 1 / 6,
    "vulnerability_score": 1 / 6,
    "topology_exposure": 1 / 6,
    "asset_criticality": 1 / 6,
    "recency_score": 1 / 6,
}

# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================
GRAPH_CONFIG = {
    # Which graph metrics to compute per node
    "node_metrics": [
        "degree",
        "in_degree",
        "out_degree",
        "betweenness_centrality",
        "closeness_centrality",
        "pagerank",
    ],
    # Edge attributes to track
    "edge_attributes": [
        "frequency",
        "recency",
        "total_bytes",
        "protocol",
    ],
}

# ==============================================================================
# EVALUATION
# ==============================================================================
EVAL_CONFIG = {
    "top_k_values": [1, 3, 5],    # Top-K hit rate evaluation
    "random_seed": 42,
}

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED = 42

# ==============================================================================
# UTILITY: Ensure directories exist
# ==============================================================================
def ensure_directories():
    """Create all required directories if they don't exist."""
    dirs_to_create = [
        PATHS["data_raw"],
        PATHS["data_processed"],
        PATHS["data_synthetic"],
        PATHS["models"],
        PATHS["experiments"],
        PATHS["cicids2017_dir"],
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[config] All directories verified under: {PROJECT_ROOT}")


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AI Cyber Attack Prediction Engine — Configuration")
    print("=" * 60)
    print(f"Project Root:          {PROJECT_ROOT}")
    print(f"Data Raw:              {PATHS['data_raw']}")
    print(f"Data Processed:        {PATHS['data_processed']}")
    print(f"Synthetic Topology:    {PATHS['campus_topology']}")
    print(f"Models:                {PATHS['models']}")
    print(f"Window Size:           {TIME_CONFIG['window_size_minutes']} minutes")
    print(f"Prediction Horizon:    {TIME_CONFIG['prediction_horizon_minutes']} minutes")
    print(f"Lookback Windows:      {TIME_CONFIG['lookback_windows']}")
    print(f"Random Seed:           {RANDOM_SEED}")
    print(f"Risk Weights:          {RISK_WEIGHTS}")
    print("=" * 60)

    ensure_directories()
    print("\n[OK] Configuration loaded successfully.")
