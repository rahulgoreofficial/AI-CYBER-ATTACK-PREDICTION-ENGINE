"""
Ablation Study — AI Cyber Attack Prediction Engine
====================================================
M9.2: Comprehensive ablation study comparing:
  - Feature group ablations (remove traffic / graph / asset features)
  - Model progression (Heuristic → XGBoost → XGBoost+IF → Dynamic Risk)

Results saved to: experiments/ablation_study_wednesday.json
                  experiments/ablation_study_wednesday.csv

Usage:
    python -m ml.evaluation.ablation_study
    python -m ml.evaluation.ablation_study --day wednesday
"""

from __future__ import annotations

import sys
import json
import argparse
import warnings
from pathlib import Path
from copy import deepcopy

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import PATHS, MODEL_CONFIG, RANDOM_SEED, RISK_WEIGHTS
from ml.evaluation.metrics import evaluate_predictions, print_comparison_table

SEPARATOR = "=" * 72


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE GROUPS
# ──────────────────────────────────────────────────────────────────────────────

# Prefixes / keywords used to identify each feature group
FEATURE_GROUPS = {
    "traffic": [
        # Flow statistics from CICIDS2017
        "flow_duration", "total_fwd_packets", "total_bwd_packets",
        "total_len_fwd", "total_len_bwd", "fwd_pkt_len", "bwd_pkt_len",
        "flow_bytes_per_s", "flow_packets_per_s", "flow_iat", "fwd_iat",
        "bwd_iat", "psh_flags", "urg_flags", "header_len", "packets_per_s",
        "pkt_len", "flag_cnt", "down_up_ratio", "avg_pkt", "avg_fwd", "avg_bwd",
        "subflow", "init_win", "act_data", "min_seg", "active_", "idle_",
        "bulk_rate", "bulk_bytes", "bulk_pkts",
        # Aggregated per window-device
        "_sum", "_mean", "_std", "_max", "_min", "_count",
    ],
    "graph": [
        "degree", "betweenness", "closeness", "pagerank",
        "in_degree", "out_degree", "centrality", "neighbor_attack",
        "attack_neighbor",
    ],
    "asset": [
        "device_criticality", "device_vulnerability", "vlan_",
        "device_type_", "is_server", "is_gateway",
    ],
}


def classify_feature(col: str) -> str:
    """
    Assign a feature column to one of: traffic, graph, asset, other.
    """
    col_lower = col.lower()

    for group, keywords in FEATURE_GROUPS.items():
        for kw in keywords:
            if kw.lower() in col_lower:
                return group
    return "other"


def get_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Split feature columns into groups.

    Returns:
        Dict mapping group name → list of column names.
    """
    groups: dict[str, list[str]] = {
        "traffic": [],
        "graph": [],
        "asset": [],
        "other": [],
    }
    for col in feature_cols:
        group = classify_feature(col)
        groups[group].append(col)

    for name, cols in groups.items():
        print(f"  [{name}] {len(cols)} features")

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_and_split(day: str = "wednesday") -> tuple:
    """Load feature matrix and create temporal train/val/test splits."""
    fm_path = PATHS["data_processed"] / f"ml_feature_matrix_{day}.csv"
    df = pd.read_csv(fm_path, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    exclude_cols = {
        "window_id", "device_id", "is_future_target",
        "target_attack_count", "target_attack_types", "earliest_target_window",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]
    ]

    # Temporal split (same as Phase 4)
    all_windows = sorted(df["window_id"].unique())
    n = len(all_windows)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train_w = set(all_windows[:train_end])
    val_w   = set(all_windows[train_end:val_end])
    test_w  = set(all_windows[val_end:])

    train_df = df[df["window_id"].isin(train_w)].copy()
    val_df   = df[df["window_id"].isin(val_w)].copy()
    test_df  = df[df["window_id"].isin(test_w)].copy()

    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    return df, train_df, val_df, test_df, feature_cols


# ──────────────────────────────────────────────────────────────────────────────
# XGBOOST RETRAINING (for ablation)
# ──────────────────────────────────────────────────────────────────────────────

def quick_train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> xgb.XGBClassifier:
    """Train a quick XGBoost model for ablation (fewer estimators for speed)."""
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    spw = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=100,       # Fewer for ablation speed
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=10,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# HEURISTIC BASELINE
# ──────────────────────────────────────────────────────────────────────────────

def heuristic_scores(df: pd.DataFrame) -> np.ndarray:
    """Degree × criticality × (1 + neighbor_attack_count) heuristic."""
    degree = df["degree"].values if "degree" in df.columns else np.ones(len(df))
    crit   = df["device_criticality"].values if "device_criticality" in df.columns else np.ones(len(df))
    nac    = df["neighbor_attack_count"].values if "neighbor_attack_count" in df.columns else np.zeros(len(df))
    degree_n = degree / max(degree.max(), 1)
    crit_n   = crit
    nac_n    = nac / max(nac.max(), 1)
    scores = degree_n * crit_n * (1 + nac_n)
    s_max = scores.max()
    return scores / s_max if s_max > 0 else scores


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ABLATION STUDY
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation_study(day: str = "wednesday") -> list[dict]:
    """
    Run the full ablation study.

    Ablations:
    A) Model progression:
       1. Heuristic baseline
       2. XGBoost (all features)
       3. XGBoost + Isolation Forest
       4. Dynamic Risk Engine

    B) Feature group ablations (XGBoost only):
       5. XGBoost — No Traffic Features
       6. XGBoost — No Graph Features
       7. XGBoost — No Asset Features
       8. XGBoost — Graph + Asset Only (no traffic)
       9. XGBoost — Traffic + Asset Only (no graph)
      10. XGBoost — Traffic + Graph Only (no asset)

    Returns:
        List of result dicts for each ablation variant.
    """
    print(f"\n{SEPARATOR}")
    print("  ABLATION STUDY — AI Cyber Attack Prediction Engine")
    print(f"  Day: {day.upper()}")
    print(SEPARATOR)

    print("\n[1/3] Loading data...")
    df, train_df, val_df, test_df, feature_cols = load_and_split(day)

    print("\n[2/3] Classifying feature groups...")
    groups = get_feature_groups(feature_cols)
    traffic_cols = groups["traffic"]
    graph_cols   = groups["graph"]
    asset_cols   = groups["asset"]
    other_cols   = groups["other"]

    print(f"\n  Traffic: {len(traffic_cols)} | Graph: {len(graph_cols)} | "
          f"Asset: {len(asset_cols)} | Other: {len(other_cols)}")

    all_results = []

    def run_variant(name: str, cols: list[str], use_if: bool = False,
                    use_risk: bool = False) -> dict | None:
        """Train and evaluate a single ablation variant."""
        if len(cols) == 0:
            print(f"  [{name}] SKIPPED — no features remaining")
            return None

        X_tr = train_df[cols].values
        y_tr = train_df["is_future_target"].values
        X_vl = val_df[cols].values
        y_vl = val_df["is_future_target"].values
        X_te = test_df[cols].values
        y_te = test_df["is_future_target"].values

        try:
            model = quick_train_xgboost(X_tr, y_tr, X_vl, y_vl)

            if use_if:
                # Augment with anomaly score
                if_m = IsolationForest(n_estimators=100, contamination="auto",
                                        random_state=RANDOM_SEED, n_jobs=-1)
                if_m.fit(X_tr)

                def anomaly(X):
                    s = -if_m.decision_function(X)
                    s_min, s_max = s.min(), s.max()
                    return (s - s_min) / (s_max - s_min) if s_max > s_min else np.zeros(len(X))

                X_tr_aug = np.column_stack([X_tr, anomaly(X_tr)])
                X_vl_aug = np.column_stack([X_vl, anomaly(X_vl)])
                X_te_aug = np.column_stack([X_te, anomaly(X_te)])
                model2 = quick_train_xgboost(X_tr_aug, y_tr, X_vl_aug, y_vl)
                probs = model2.predict_proba(X_te_aug)[:, 1]
                anomaly_te = anomaly(X_te)
            else:
                probs = model.predict_proba(X_te)[:, 1]
                anomaly_te = np.zeros(len(X_te))

            if use_risk:
                # Apply dynamic risk engine on top of XGBoost probs
                n = len(test_df)
                vuln   = test_df["device_vulnerability"].values if "device_vulnerability" in test_df.columns else np.full(n, 0.5)
                bc     = test_df["betweenness_centrality"].values if "betweenness_centrality" in test_df.columns else np.full(n, 0.5)
                bc_n   = bc / max(bc.max(), 1e-10)
                crit   = test_df["device_criticality"].values if "device_criticality" in test_df.columns else np.full(n, 0.5)
                nac    = test_df["neighbor_attack_count"].values if "neighbor_attack_count" in test_df.columns else np.zeros(n)
                nac_n  = nac / max(nac.max(), 1e-10)
                w = RISK_WEIGHTS
                probs = (
                    w["attack_probability"] * probs
                    + w["anomaly_score"]       * anomaly_te
                    + w["vulnerability_score"] * vuln
                    + w["topology_exposure"]   * bc_n
                    + w["asset_criticality"]   * crit
                    + w["recency_score"]       * nac_n
                )

            results = evaluate_predictions(
                y_te, probs,
                test_df["device_id"].values,
                test_df["window_id"].values,
                model_name=name,
                verbose=False,
            )
            results["n_features"] = len(cols)
            results["feature_groups_removed"] = ""
            print(f"  [{name:<52}] Top-1={results['top_1_hit_rate']:.2f}  "
                  f"F1={results['f1']:.2f}  PR-AUC={results['pr_auc']:.2f}")
            return results

        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
            return None

    print(f"\n[3/3] Running ablation variants...\n")

    # ─── A. Model Progression ────────────────────────────────────────────────

    print("  --- Part A: Model Progression ---")

    # A1: Heuristic baseline
    h_scores = heuristic_scores(test_df)
    y_te = test_df["is_future_target"].values
    h_results = evaluate_predictions(
        y_te, h_scores,
        test_df["device_id"].values, test_df["window_id"].values,
        model_name="A1: Heuristic Baseline",
        verbose=False,
    )
    h_results["n_features"] = 3
    h_results["feature_groups_removed"] = "none (heuristic)"
    print(f"  [A1: Heuristic Baseline                              ] "
          f"Top-1={h_results['top_1_hit_rate']:.2f}  F1={h_results['f1']:.2f}  "
          f"PR-AUC={h_results['pr_auc']:.2f}")
    all_results.append(h_results)

    # A2: XGBoost (all features)
    r = run_variant("A2: XGBoost (all features)", feature_cols)
    if r:
        r["feature_groups_removed"] = "none"
        all_results.append(r)

    # A3: XGBoost + Isolation Forest (all features)
    r = run_variant("A3: XGBoost + Isolation Forest", feature_cols, use_if=True)
    if r:
        r["feature_groups_removed"] = "none (+ anomaly score)"
        all_results.append(r)

    # A4: Dynamic Risk Engine
    r = run_variant("A4: Dynamic Risk Engine", feature_cols, use_if=True, use_risk=True)
    if r:
        r["feature_groups_removed"] = "none (full risk formula)"
        all_results.append(r)

    # ─── B. Feature Group Ablations ──────────────────────────────────────────

    print("\n  --- Part B: Feature Group Ablations (XGBoost only) ---")

    # B1: Remove traffic features
    no_traffic = [c for c in feature_cols if c not in traffic_cols]
    r = run_variant("B1: XGBoost — No Traffic", no_traffic)
    if r:
        r["feature_groups_removed"] = "traffic"
        all_results.append(r)

    # B2: Remove graph features
    no_graph = [c for c in feature_cols if c not in graph_cols]
    r = run_variant("B2: XGBoost — No Graph", no_graph)
    if r:
        r["feature_groups_removed"] = "graph"
        all_results.append(r)

    # B3: Remove asset features
    no_asset = [c for c in feature_cols if c not in asset_cols]
    r = run_variant("B3: XGBoost — No Asset", no_asset)
    if r:
        r["feature_groups_removed"] = "asset"
        all_results.append(r)

    # B4: Graph + Asset only (remove traffic)
    graph_asset = [c for c in feature_cols if c in graph_cols or c in asset_cols or c in other_cols]
    r = run_variant("B4: XGBoost — Graph + Asset only", graph_asset)
    if r:
        r["feature_groups_removed"] = "traffic"
        all_results.append(r)

    # B5: Traffic + Asset only (remove graph)
    traffic_asset = [c for c in feature_cols if c in traffic_cols or c in asset_cols or c in other_cols]
    r = run_variant("B5: XGBoost — Traffic + Asset only", traffic_asset)
    if r:
        r["feature_groups_removed"] = "graph"
        all_results.append(r)

    # B6: Traffic + Graph only (remove asset)
    traffic_graph = [c for c in feature_cols if c in traffic_cols or c in graph_cols or c in other_cols]
    r = run_variant("B6: XGBoost — Traffic + Graph only", traffic_graph)
    if r:
        r["feature_groups_removed"] = "asset"
        all_results.append(r)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# SAVE & REPORT
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], day: str = "wednesday") -> None:
    """Save ablation results to JSON and CSV."""
    out_dir = PATHS["experiments"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / f"ablation_study_{day}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # CSV
    csv_path = out_dir / f"ablation_study_{day}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def print_ablation_table(results: list[dict]) -> None:
    """Print a clean comparison table of all ablation variants."""
    print(f"\n{SEPARATOR}")
    print("  ABLATION STUDY RESULTS")
    print(SEPARATOR)
    print(f"\n  {'Variant':<52} {'#Feat':>5} {'Top-1':>6} {'Top-3':>6} "
          f"{'MRR':>6} {'PR-AUC':>7} {'F1':>6}")
    print(f"  {'-' * 90}")
    for r in results:
        print(
            f"  {r['model']:<52} "
            f"{r.get('n_features', '?'):>5} "
            f"{r.get('top_1_hit_rate', 0):.3f}  "
            f"{r.get('top_3_hit_rate', 0):.3f}  "
            f"{r.get('mrr', 0):.3f}  "
            f"{r.get('pr_auc', 0):.4f}  "
            f"{r.get('f1', 0):.3f}"
        )
    print(f"\n{SEPARATOR}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for the prediction engine")
    parser.add_argument("--day", default="wednesday", help="Dataset day (default: wednesday)")
    args = parser.parse_args()

    results = run_ablation_study(day=args.day)
    print_ablation_table(results)
    save_results(results, day=args.day)

    print("\n  [DONE] Ablation study complete.")
