"""
Data Loader Service — Central data and model loading singleton
==============================================================

Loads all pre-computed data and trained models at startup.
Provides a single access point for all API services.

Usage:
    from backend.app.services.data_loader import get_data_store, DataStore

    store = get_data_store()
    topology = store.topology
    risk_df = store.risk_scores
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Resolve project root
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent.parent  # backend/
PROJECT_ROOT = _BACKEND_DIR.parent       # project root

logger = logging.getLogger("backend.data_loader")


class DataStore:
    """
    Singleton data store that caches all project assets in memory.

    Loaded assets:
    - Campus topology (devices, connections, IP mapping)
    - Risk scores (per device per window)
    - Local SHAP explanations (per device per window)
    - Global SHAP feature importance
    - Model comparison metrics
    - Feature matrix metadata (columns, window/device index)
    - Trained XGBoost model (for on-demand inference)
    """

    def __init__(self):
        self._loaded = False

        # Topology
        self.topology: dict = {}
        self.devices: list[dict] = []
        self.connections: list[dict] = []
        self.device_map: dict[str, dict] = {}  # device_id → device metadata
        self.adjacency: dict[str, list[str]] = {}
        self.topology_exposure_map: dict[str, float] = {}

        # Risk scores
        self.risk_scores: list[dict] = []
        self.risk_by_window: dict[int, list[dict]] = {}  # window_id → [risk entries]

        # Explanations
        self.local_explanations: list[dict] = []
        self.explanations_by_device: dict[str, list[dict]] = {}  # device_id → [explanations]
        self.global_importance: list[dict] = []

        # Model comparison
        self.model_comparison: list[dict] = []

        # Feature matrix metadata & DataFrame
        self.feature_df: Any = None
        self.feature_columns: list[str] = []
        self.feat_cols_120: list[str] = []
        self.window_ids: list[int] = []

        # Live ML models
        self.xgboost_model: Any = None
        self.xgboost_if_model: Any = None
        self.isolation_forest_model: Any = None
        self.gnn_model: Any = None
        self.lstm_model: Any = None
        self.models_loaded: bool = False

    def load_all(self) -> None:
        """Load all data assets and ML models from disk into memory."""
        if self._loaded:
            logger.info("Data store already loaded, skipping.")
            return

        logger.info("Loading all data assets and ML models into memory...")

        self._load_topology()
        self._load_risk_scores()
        self._load_explanations()
        self._load_model_comparison()
        self._load_feature_columns()
        self._load_ml_models()

        self._loaded = True
        logger.info(
            f"Data store loaded: {len(self.devices)} devices, "
            f"{len(self.connections)} connections, "
            f"{len(self.risk_scores)} risk entries, "
            f"{len(self.local_explanations)} explanations, "
            f"{len(self.window_ids)} windows"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ──────────────────────────────────────────────────────────────────────
    # TOPOLOGY
    # ──────────────────────────────────────────────────────────────────────

    def _load_topology(self) -> None:
        """Load campus_topology.json and compute network graph metrics."""
        topo_path = PROJECT_ROOT / "data" / "synthetic" / "campus_topology.json"
        if not topo_path.exists():
            logger.warning(f"Topology file not found: {topo_path}")
            return

        with open(topo_path, "r", encoding="utf-8") as f:
            self.topology = json.load(f)

        self.devices = self.topology.get("devices", [])
        self.connections = self.topology.get("connections", [])
        self.device_map = {d["id"]: d for d in self.devices}

        # Build adjacency graph
        self.adjacency = {}
        degree_map: dict[str, int] = {}
        for conn in self.connections:
            src, dst = conn["from"], conn["to"]
            self.adjacency.setdefault(src, []).append(dst)
            self.adjacency.setdefault(dst, []).append(src)
            degree_map[src] = degree_map.get(src, 0) + 1
            degree_map[dst] = degree_map.get(dst, 0) + 1

        # Normalize degree exposure [0, 1] for all devices
        max_deg = max(degree_map.values()) if degree_map else 1
        self.topology_exposure_map = {
            dev_id: round(degree_map.get(dev_id, 1) / max_deg, 3)
            for dev_id in self.device_map.keys()
        }

        logger.info(f"Loaded topology: {len(self.devices)} devices, {len(self.connections)} connections")

    # ──────────────────────────────────────────────────────────────────────
    # RISK SCORES
    # ──────────────────────────────────────────────────────────────────────

    def _load_risk_scores(self) -> None:
        """Load risk_scores_wednesday.csv."""
        risk_path = PROJECT_ROOT / "experiments" / "risk_scores_wednesday.csv"
        if not risk_path.exists():
            logger.warning(f"Risk scores file not found: {risk_path}")
            return

        with open(risk_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.risk_scores = []
            for row in reader:
                entry = {
                    "window_id": int(row["window_id"]),
                    "device_id": row["device_id"],
                    "attack_probability": float(row["attack_probability"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "vulnerability_score": float(row["vulnerability_score"]),
                    "topology_exposure": float(row["topology_exposure"]),
                    "asset_criticality": float(row["asset_criticality"]),
                    "recency_score": float(row["recency_score"]),
                    "dynamic_risk_score": float(row["dynamic_risk_score"]),
                    "risk_rank": int(row["risk_rank"]),
                }
                self.risk_scores.append(entry)

        # Index by window_id
        self.risk_by_window = {}
        for entry in self.risk_scores:
            wid = entry["window_id"]
            if wid not in self.risk_by_window:
                self.risk_by_window[wid] = []
            self.risk_by_window[wid].append(entry)

        # Collect all unique window IDs
        self.window_ids = sorted(self.risk_by_window.keys())

        logger.info(f"Loaded risk scores: {len(self.risk_scores)} entries, {len(self.window_ids)} windows")

    # ──────────────────────────────────────────────────────────────────────
    # EXPLANATIONS
    # ──────────────────────────────────────────────────────────────────────

    def _load_explanations(self) -> None:
        """Load SHAP explanations (local + global)."""
        # Local explanations
        local_path = PROJECT_ROOT / "experiments" / "local_explanations_wednesday.json"
        if local_path.exists():
            with open(local_path, "r", encoding="utf-8") as f:
                self.local_explanations = json.load(f)

            # Index by device_id
            self.explanations_by_device = {}
            for entry in self.local_explanations:
                did = entry.get("device_id", "")
                if did not in self.explanations_by_device:
                    self.explanations_by_device[did] = []
                self.explanations_by_device[did].append(entry)

            logger.info(f"Loaded local explanations: {len(self.local_explanations)} entries")
        else:
            logger.warning(f"Local explanations not found: {local_path}")

        # Global feature importance
        global_path = PROJECT_ROOT / "experiments" / "shap_analysis_wednesday.json"
        if global_path.exists():
            with open(global_path, "r", encoding="utf-8") as f:
                shap_data = json.load(f)
                self.global_importance = shap_data.get("global_feature_importance", [])
            logger.info(f"Loaded global importance: {len(self.global_importance)} features")
        else:
            logger.warning(f"Global SHAP analysis not found: {global_path}")

    # ──────────────────────────────────────────────────────────────────────
    # MODEL COMPARISON
    # ──────────────────────────────────────────────────────────────────────

    def _load_model_comparison(self) -> None:
        """Load model_comparison_wednesday.json."""
        comp_path = PROJECT_ROOT / "experiments" / "model_comparison_wednesday.json"
        if not comp_path.exists():
            logger.warning(f"Model comparison file not found: {comp_path}")
            return

        with open(comp_path, "r", encoding="utf-8") as f:
            self.model_comparison = json.load(f)

        logger.info(f"Loaded model comparison: {len(self.model_comparison)} models")

    # ──────────────────────────────────────────────────────────────────────
    # FEATURE COLUMNS
    # ──────────────────────────────────────────────────────────────────────

    def _load_feature_columns(self) -> None:
        """Load feature columns and feature matrix DataFrame into memory."""
        cols_path = PROJECT_ROOT / "data" / "processed" / "cicids2017_wednesday_feature_cols.txt"
        if cols_path.exists():
            with open(cols_path, "r", encoding="utf-8") as f:
                self.feature_columns = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded feature columns: {len(self.feature_columns)}")
        else:
            logger.warning(f"Feature columns file not found: {cols_path}")

        fm_path = PROJECT_ROOT / "data" / "processed" / "ml_feature_matrix_wednesday.csv"
        if fm_path.exists():
            try:
                import pandas as pd
                import numpy as np
                self.feature_df = pd.read_csv(fm_path, low_memory=False)
                exclude = {
                    "window_id", "device_id", "is_future_target",
                    "target_attack_count", "target_attack_types", "earliest_target_window",
                }
                self.feat_cols_120 = [
                    c for c in self.feature_df.columns
                    if c not in exclude
                    and self.feature_df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]
                ]
                logger.info(f"Loaded feature matrix DataFrame: {len(self.feature_df)} rows x {len(self.feat_cols_120)} features")
            except Exception as e:
                logger.warning(f"Failed to load feature matrix DataFrame: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # ML MODELS
    # ──────────────────────────────────────────────────────────────────────

    def _load_ml_models(self) -> None:
        """Load all trained ML models: XGBoost, Isolation Forest, GNN, and Temporal LSTM."""
        models_dir = PROJECT_ROOT / "models"

        # XGBoost baseline
        xgb_path = models_dir / "xgboost_baseline.pkl"
        if xgb_path.exists():
            try:
                import joblib
                self.xgboost_model = joblib.load(xgb_path)
                logger.info("Loaded XGBoost baseline model")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost model: {e}")

        # XGBoost + IF
        xgb_if_path = models_dir / "xgboost_with_if.pkl"
        if xgb_if_path.exists():
            try:
                import joblib
                self.xgboost_if_model = joblib.load(xgb_if_path)
                logger.info("Loaded XGBoost + IF model")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost + IF model: {e}")

        # Isolation Forest
        if_path = models_dir / "isolation_forest.pkl"
        if if_path.exists():
            try:
                import joblib
                self.isolation_forest_model = joblib.load(if_path)
                logger.info("Loaded Isolation Forest model")
            except Exception as e:
                logger.warning(f"Failed to load Isolation Forest model: {e}")

        # GNN (GraphSAGE)
        gnn_path = models_dir / "gnn_model.pt"
        if gnn_path.exists():
            try:
                import torch
                from ml.gnn.model import NextTargetGNN
                gnn_ckpt = torch.load(gnn_path, map_location="cpu", weights_only=False)
                self.gnn_model = NextTargetGNN(**gnn_ckpt["config"])
                self.gnn_model.load_state_dict(gnn_ckpt["model_state_dict"])
                self.gnn_model.eval()
                logger.info("Loaded GraphSAGE GNN model")
            except Exception as e:
                logger.warning(f"Failed to load GNN model: {e}")

        # Temporal LSTM
        lstm_path = models_dir / "temporal_lstm.pt"
        if lstm_path.exists():
            try:
                import torch
                from ml.temporal.temporal_lstm import TemporalLSTM
                lstm_ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
                cfg = lstm_ckpt["config"]
                self.lstm_model = TemporalLSTM(
                    input_dim=cfg["input_dim"],
                    hidden_dim=cfg["hidden_dim"],
                    num_layers=cfg["num_layers"],
                    dropout=cfg.get("dropout", 0.3),
                )
                self.lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
                self.lstm_model.eval()
                logger.info("Loaded Temporal LSTM model")
            except Exception as e:
                logger.warning(f"Failed to load Temporal LSTM model: {e}")

        self.models_loaded = any([
            self.xgboost_model is not None,
            self.xgboost_if_model is not None,
            self.isolation_forest_model is not None,
            self.gnn_model is not None,
            self.lstm_model is not None,
        ])

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def get_risk_level(self, risk_score: float) -> str:
        """Convert numeric risk score to risk level string."""
        if risk_score >= 0.75:
            return "critical"
        elif risk_score >= 0.5:
            return "high"
        elif risk_score >= 0.25:
            return "medium"
        else:
            return "low"

    def get_device_info(self, device_id: str) -> Optional[dict]:
        """Get device metadata by ID."""
        return self.device_map.get(device_id)

    def get_topology_exposure(self, device_id: str) -> float:
        """Get precomputed normalized degree exposure for a device."""
        return self.topology_exposure_map.get(device_id, 0.3)

    def get_latest_window_id(self) -> int:
        """Get the most recent window ID."""
        return self.window_ids[-1] if self.window_ids else 0

    def get_device_features(self, window_id: int, device_id: str):
        """Get raw 120-feature numpy vector for a (window, device) pair, or zero-vector fallback."""
        import numpy as np
        if self.feature_df is not None and len(self.feat_cols_120) > 0:
            match = self.feature_df[
                (self.feature_df["window_id"] == window_id) &
                (self.feature_df["device_id"] == device_id)
            ]
            if len(match) > 0:
                vec = match[self.feat_cols_120].values[0].astype(np.float32)
                return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Baseline fallback for devices without raw flows in this window
        n_feats = len(self.feat_cols_120) if len(self.feat_cols_120) > 0 else 120
        return np.zeros(n_feats, dtype=np.float32)


# ==============================================================================
# SINGLETON ACCESS
# ==============================================================================

_data_store: Optional[DataStore] = None


def get_data_store() -> DataStore:
    """Get or create the singleton DataStore instance."""
    global _data_store
    if _data_store is None:
        _data_store = DataStore()
    return _data_store
