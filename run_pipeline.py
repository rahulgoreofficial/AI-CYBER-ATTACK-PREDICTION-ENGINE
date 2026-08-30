# -*- coding: utf-8 -*-
"""
End-to-End Pipeline Demo -- AI Cyber Attack Prediction Engine
=============================================================
M9.1: Full pipeline orchestration — load data, run prediction,
display ranked risk, SHAP explanations, and recommendations.

Usage:
    python run_pipeline.py
    python run_pipeline.py --window 8
    python run_pipeline.py --topk 5 --window 10
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml.config import PATHS, RANDOM_SEED

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 72
SEP_THIN  = "-" * 72


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def check_artifacts() -> bool:
    """Verify all required files are present before running."""
    required = {
        "Feature matrix": PATHS["data_processed"] / "ml_feature_matrix_wednesday.csv",
        "XGBoost model":  PATHS["models"] / "xgboost_baseline.pkl",
        "Isolation Forest": PATHS["models"] / "isolation_forest.pkl",
        "XGBoost+IF model": PATHS["models"] / "xgboost_with_if.pkl",
        "SHAP analysis":  PATHS["experiments"] / "shap_analysis_wednesday.json",
        "Risk scores":    PATHS["experiments"] / "risk_scores_wednesday.csv",
        "Model comparison": PATHS["experiments"] / "model_comparison_wednesday.json",
        "Campus topology": PATHS["campus_topology"],
    }
    ok = True
    for name, path in required.items():
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {name}")
        if not path.exists():
            ok = False
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(window_id: int | None = None, top_k: int = 5) -> dict:
    """
    Execute the complete end-to-end prediction pipeline.

    1. Load feature matrix + models
    2. Select a time window
    3. Run XGBoost + IsolationForest prediction
    4. Apply dynamic risk scoring
    5. Display ranked predictions with SHAP feature explanations
    6. Show defensive recommendations

    Returns:
        Dictionary with all pipeline results.
    """
    section("STEP 1 — ARTIFACT CHECK")
    if not check_artifacts():
        print("\n[ERROR] Missing artifacts. Run the full ML pipeline first.")
        sys.exit(1)
    print("\n  All artifacts present.")

    # ──────────────────────────────────────────────────────────────────────────
    # Load feature matrix
    # ──────────────────────────────────────────────────────────────────────────
    section("STEP 2 — LOAD DATA")
    fm_path = PATHS["data_processed"] / "ml_feature_matrix_wednesday.csv"
    df = pd.read_csv(fm_path, low_memory=False)
    print(f"  Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Metadata / excluded columns
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
    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  Time windows    : {df['window_id'].nunique()}")
    print(f"  Unique devices  : {df['device_id'].nunique()}")
    print(f"  Attack targets  : {int(df['is_future_target'].sum())} positive samples")

    # ──────────────────────────────────────────────────────────────────────────
    # Select window
    # ──────────────────────────────────────────────────────────────────────────
    all_windows = sorted(df["window_id"].unique())
    if window_id is None:
        # Pick a window from the test split (last 15% of windows)
        test_start = int(len(all_windows) * 0.85)
        window_id = all_windows[test_start]
        print(f"\n  Auto-selected test window: {window_id}")
    elif window_id not in all_windows:
        print(f"\n  [WARN] Window {window_id} not found. Available: {all_windows[:5]}... "
              f"Using first window.")
        window_id = all_windows[0]

    window_df = df[df["window_id"] == window_id].copy()
    print(f"\n  Selected window  : {window_id}")
    print(f"  Devices in window: {len(window_df)}")
    print(f"  Actual targets   : {int(window_df['is_future_target'].sum())}")

    section("STEP 3 — RUN ML MODELS")

    # ── Load models ──
    xgb_model  = joblib.load(PATHS["models"] / "xgboost_baseline.pkl")
    xgbif_model = joblib.load(PATHS["models"] / "xgboost_with_if.pkl")
    if_model   = joblib.load(PATHS["models"] / "isolation_forest.pkl")
    print("  Loaded: XGBoost, XGBoost+IF, Isolation Forest")

    X_window = window_df[feature_cols].values

    # ── XGBoost base probabilities ──
    xgb_probs = xgb_model.predict_proba(X_window)[:, 1]

    # ── Anomaly scores ──
    raw_anomaly = if_model.decision_function(X_window)
    anomaly_scores = -raw_anomaly
    a_min, a_max = anomaly_scores.min(), anomaly_scores.max()
    if a_max > a_min:
        anomaly_scores = (anomaly_scores - a_min) / (a_max - a_min)
    else:
        anomaly_scores = np.zeros_like(anomaly_scores)

    # ── XGBoost+IF probabilities ──
    # The augmented model was trained on features + anomaly score
    # We need to match the feature count it was trained on
    try:
        n_features_xgbif = xgbif_model.n_features_in_
        if n_features_xgbif == len(feature_cols) + 1:
            X_window_aug = np.column_stack([X_window, anomaly_scores])
        else:
            X_window_aug = X_window
        xgbif_probs = xgbif_model.predict_proba(X_window_aug)[:, 1]
    except Exception:
        xgbif_probs = xgb_probs  # fallback

    print(f"  XGBoost max prob   : {xgb_probs.max():.4f}")
    print(f"  XGBoost+IF max prob: {xgbif_probs.max():.4f}")
    print(f"  Max anomaly score  : {anomaly_scores.max():.4f}")

    section("STEP 4 — DYNAMIC RISK SCORING")

    # Risk = 0.40*attack_prob + 0.15*anomaly + 0.15*vulnerability
    #      + 0.10*topology + 0.15*criticality + 0.05*recency
    from ml.config import RISK_WEIGHTS
    weights = RISK_WEIGHTS

    n = len(window_df)
    c_attack = xgb_probs
    c_anomaly = anomaly_scores
    c_vuln  = window_df["device_vulnerability"].values if "device_vulnerability" in window_df.columns else np.full(n, 0.5)
    c_topo  = window_df["betweenness_centrality"].values if "betweenness_centrality" in window_df.columns else np.full(n, 0.5)
    if c_topo.max() > 0:
        c_topo = c_topo / c_topo.max()
    c_crit  = window_df["device_criticality"].values if "device_criticality" in window_df.columns else np.full(n, 0.5)
    nac = window_df["neighbor_attack_count"].values if "neighbor_attack_count" in window_df.columns else np.zeros(n)
    c_recency = nac / max(nac.max(), 1e-10)

    risk = (
        weights["attack_probability"] * c_attack
        + weights["anomaly_score"]    * c_anomaly
        + weights["vulnerability_score"] * c_vuln
        + weights["topology_exposure"] * c_topo
        + weights["asset_criticality"] * c_crit
        + weights["recency_score"]    * c_recency
    )

    result_df = window_df[["device_id", "is_future_target"]].copy()
    result_df["attack_prob"] = c_attack
    result_df["anomaly_score"] = c_anomaly
    result_df["dynamic_risk"] = risk
    result_df = result_df.sort_values("dynamic_risk", ascending=False).reset_index(drop=True)
    result_df["rank"] = range(1, len(result_df) + 1)

    section("STEP 5 — RANKED PREDICTIONS")

    print(f"\n  Top-{top_k} Predicted Targets for Window {window_id}:\n")
    print(f"  {'Rank':<5} {'Device':<25} {'Attack Prob':>12} {'Anomaly':>10} "
          f"{'Risk Score':>12} {'Actual Target':>14}")
    print(f"  {SEP_THIN}")

    for _, row in result_df.head(top_k).iterrows():
        actual = "[TARGET]" if row["is_future_target"] else ""
        print(
            f"  {int(row['rank']):<5} "
            f"{row['device_id']:<25} "
            f"{row['attack_prob']:>12.4f} "
            f"{row['anomaly_score']:>10.4f} "
            f"{row['dynamic_risk']:>12.4f} "
            f"{actual:>14}"
        )

    # Compute Top-1 hit
    top1_hit = int(result_df.iloc[0]["is_future_target"])
    top3 = result_df.head(3)
    top3_hit = int(top3["is_future_target"].sum() > 0)
    actual_targets = result_df[result_df["is_future_target"] == 1]["device_id"].tolist()

    print(f"\n  Top-1 correct : {'YES [OK]' if top1_hit else 'NO [X]'}")
    print(f"  Top-3 correct : {'YES [OK]' if top3_hit else 'NO [X]'}")
    print(f"  Actual targets: {actual_targets if actual_targets else 'none in this window'}")

    section("STEP 6 — SHAP FEATURE EXPLANATION")

    # Load global SHAP importance
    shap_path = PATHS["experiments"] / "shap_analysis_wednesday.json"
    with open(shap_path) as f:
        shap_data = json.load(f)

    global_importance = shap_data.get("global_importance", {})
    if global_importance:
        top_features = sorted(
            global_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        print(f"\n  Top-10 Most Important Features (Global SHAP):\n")
        for i, (feat, imp) in enumerate(top_features, 1):
            bar = "█" * int(imp * 50)
            print(f"  {i:>3}. {feat:<42} {imp:.4f} {bar}")
    else:
        print("  SHAP global importance not available in current JSON format.")

    # Per-device explanation for top predicted device
    top_device = result_df.iloc[0]["device_id"]
    local_explanations_path = PATHS["experiments"] / "local_explanations_wednesday.json"
    if local_explanations_path.exists():
        with open(local_explanations_path) as f:
            local_data = json.load(f)

        # Find explanation for the top predicted device in this window
        matching = [
            e for e in local_data
            if e.get("device_id") == top_device
        ]
        if matching:
            # Take the most recent explanation
            explanation = matching[-1]
            top_contribs = sorted(
                explanation.get("feature_contributions", {}).items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            print(f"\n  Top-5 SHAP contributions for {top_device} (most recent window):\n")
            for feat, contrib in top_contribs:
                direction = "[+]" if contrib > 0 else "[-]"
                print(f"    {direction} {feat:<42} {contrib:+.4f}")

    section("STEP 7 — RECOMMENDATIONS")

    # Load recommendations from backend engine (rule-based)
    print(f"\n  Defensive actions for top predicted target: {top_device}\n")
    try:
        from backend.app.recommendations.engine import RecommendationEngine
        engine = RecommendationEngine()

        # Build a minimal context for the recommendation engine
        top_row = result_df.iloc[0]
        context = {
            "device_id": top_device,
            "attack_probability": float(top_row["attack_prob"]),
            "risk_score": float(top_row["dynamic_risk"]),
            "anomaly_score": float(top_row["anomaly_score"]),
        }
        recommendations = engine.get_recommendations(top_device, context)

        for i, rec in enumerate(recommendations[:5], 1):
            priority = rec.get("priority", "medium").upper()
            action = rec.get("action", "")
            category = rec.get("category", "")
            print(f"  {i}. [{priority:<8}] [{category}] {action}")
    except Exception as e:
        print(f"  (Recommendation engine not available in headless mode: {e})")
        # Fallback: static recommendations based on risk level
        print("  1. [CRITICAL ] Isolate or segment the predicted target device")
        print("  2. [HIGH     ] Increase monitoring/logging for network traffic")
        print("  3. [HIGH     ] Apply emergency patching if vulnerable CVEs known")
        print("  4. [MEDIUM   ] Alert SOC team with attack probability and device")
        print("  5. [MEDIUM   ] Review access control lists for adjacent devices")

    section("STEP 8 — SUMMARY")

    print("\n  Pipeline completed successfully.\n")
    print(f"  Dataset            : CICIDS2017 Wednesday")
    print(f"  Time Window        : {window_id} of {max(all_windows)}")
    print(f"  Devices evaluated  : {len(window_df)}")
    print(f"  Top predicted      : {result_df.iloc[0]['device_id']}")
    print(f"  Top-1 accuracy     : {'CORRECT [OK]' if top1_hit else 'INCORRECT [X]'}")
    print(f"  Top-3 accuracy     : {'CORRECT ✓' if top3_hit else 'INCORRECT ✗'}")
    print(f"  Max risk score     : {result_df.iloc[0]['dynamic_risk']:.4f}")

    results = {
        "window_id": window_id,
        "top_k": top_k,
        "devices_evaluated": len(window_df),
        "top_predicted_device": result_df.iloc[0]["device_id"],
        "top_1_correct": bool(top1_hit),
        "top_3_correct": bool(top3_hit),
        "actual_targets": actual_targets,
        "ranked_predictions": result_df[
            ["device_id", "attack_prob", "anomaly_score", "dynamic_risk", "is_future_target", "rank"]
        ].head(top_k).to_dict(orient="records"),
    }

    print(f"\n{SEPARATOR}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end AI Cyber Attack Prediction Engine pipeline demo"
    )
    parser.add_argument("--window", type=int, default=None,
                        help="Time window ID to predict for (default: auto-select test window)")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of top-K predictions to display (default: 5)")
    args = parser.parse_args()

    print("\n" + SEPARATOR)
    print("  AI CYBER ATTACK PREDICTION ENGINE -- END-TO-END PIPELINE DEMO")
    print("  M9.1 -- Integration Verification")
    print(SEPARATOR)

    results = run_pipeline(window_id=args.window, top_k=args.topk)
    sys.exit(0)
