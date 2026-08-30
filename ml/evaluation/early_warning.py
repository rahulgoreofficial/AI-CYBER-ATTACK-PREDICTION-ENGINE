"""
Early-Warning Time Evaluation — AI Cyber Attack Prediction Engine
=================================================================
M9.3: Compute how many minutes AHEAD of actual attacks our model
correctly predicts the next target.

For each test window where the model is correct (Top-1 hit):
  lead_time = (attack_window_start - prediction_window_end)

This is the "early warning" — how much time defenders gain to respond.

Results saved to: experiments/early_warning_wednesday.json

Usage:
    python -m ml.evaluation.early_warning
    python -m ml.evaluation.early_warning --day wednesday
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
from ml.config import PATHS, RANDOM_SEED, TIME_CONFIG, RISK_WEIGHTS

SEPARATOR = "=" * 72


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_feature_matrix(day: str) -> pd.DataFrame:
    path = PATHS["data_processed"] / f"ml_feature_matrix_{day}.csv"
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def load_processed_events(day: str) -> pd.DataFrame | None:
    """Load the preprocessed event table that has actual timestamps per window."""
    path = PATHS["data_processed"] / f"cicids2017_{day}_processed.csv"
    if not path.exists():
        print(f"  [WARN] Processed events not found at {path}")
        return None
    # Read only necessary columns to keep memory low
    try:
        needed = ["device_id", "window_id", "timestamp", "label"]
        df = pd.read_csv(
            path, usecols=lambda c: any(n in c.lower() for n in
                                        ["window_id", "timestamp", "label", "device_id",
                                         "dst_device", "src_device"]),
            low_memory=False,
            nrows=500_000,  # cap for speed
        )
        print(f"  Loaded events: {df.shape[0]:,} rows, cols: {list(df.columns[:8])}")
        return df
    except Exception as e:
        print(f"  [WARN] Could not load processed events: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def predict_all_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_windows: set,
) -> pd.DataFrame:
    """
    Run XGBoost + Dynamic Risk on all test windows.

    Returns per-row DataFrame with:
      window_id, device_id, pred_score, is_future_target, rank_in_window
    """
    xgb_model = joblib.load(PATHS["models"] / "xgboost_baseline.pkl")
    if_model  = joblib.load(PATHS["models"] / "isolation_forest.pkl")

    test_df = df[df["window_id"].isin(test_windows)].copy()

    X = test_df[feature_cols].values

    # XGBoost probabilities
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    # Anomaly scores
    raw_a = -if_model.decision_function(X)
    a_min, a_max = raw_a.min(), raw_a.max()
    anomaly = (raw_a - a_min) / (a_max - a_min) if a_max > a_min else np.zeros(len(X))

    # Dynamic risk
    n = len(test_df)
    vuln  = test_df["device_vulnerability"].values if "device_vulnerability" in test_df.columns else np.full(n, 0.5)
    bc    = test_df["betweenness_centrality"].values if "betweenness_centrality" in test_df.columns else np.full(n, 0.5)
    bc_n  = bc / max(bc.max(), 1e-10)
    crit  = test_df["device_criticality"].values if "device_criticality" in test_df.columns else np.full(n, 0.5)
    nac   = test_df["neighbor_attack_count"].values if "neighbor_attack_count" in test_df.columns else np.zeros(n)
    nac_n = nac / max(nac.max(), 1e-10)

    w = RISK_WEIGHTS
    risk = (
        w["attack_probability"]   * xgb_probs
        + w["anomaly_score"]      * anomaly
        + w["vulnerability_score"]* vuln
        + w["topology_exposure"]  * bc_n
        + w["asset_criticality"]  * crit
        + w["recency_score"]      * nac_n
    )

    result = test_df[["window_id", "device_id", "is_future_target"]].copy()
    result["pred_score"] = risk
    result["attack_prob"] = xgb_probs

    # Per-window rank
    result["rank"] = result.groupby("window_id")["pred_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# EARLY-WARNING CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_early_warning(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    window_size_minutes: int = 5,
    horizon_minutes: int = 15,
) -> dict:
    """
    Compute early-warning lead times for correct predictions.

    For each test window where Top-1 rank is an actual target:
      lead_time_minutes = horizon_minutes
        (because we predict X minutes into the future by design,
         so a correct prediction at window t means the attack
         happens within [t+1, t+horizon/window_size] windows)

    We also compute the window-level lead time distribution
    by finding the earliest actual attack window for the top-predicted
    device vs the prediction window.

    Returns:
        Dictionary with early-warning statistics.
    """
    window_size = window_size_minutes

    # -- Approach 1: Design-level lead time --------------------------------
    # By construction, our target label is "is this device attacked within
    # the next `horizon` minutes?" So a correct prediction ALWAYS implies
    # we predicted at least 1 window ahead.
    # The precise lead time depends on WHEN the attack actually starts.

    # Find correct Top-1 predictions
    top1 = predictions[predictions["rank"] == 1].copy()
    correct_top1 = top1[top1["is_future_target"] == 1]
    incorrect_top1 = top1[top1["is_future_target"] == 0]

    total_windows_with_targets = predictions[
        predictions["is_future_target"] == 1
    ]["window_id"].nunique()
    correct_count = len(correct_top1)
    incorrect_count = len(incorrect_top1)

    print(f"\n  Test windows with actual targets : {total_windows_with_targets}")
    print(f"  Correct Top-1 predictions        : {correct_count}")
    print(f"  Incorrect Top-1 predictions      : {incorrect_count}")

    # -- Approach 2: Window-gap lead time ----------------------------------
    # For each window_id where we're correct, find how many windows
    # until the target actually gets attacked.
    # Since "earliest_target_window" is in the feature matrix:
    lead_times_minutes = []
    lead_times_windows = []
    per_window_details = []

    if "earliest_target_window" in df.columns:
        for _, row in correct_top1.iterrows():
            w = row["window_id"]
            dev = row["device_id"]

            # Look up this device in this window
            entry = df[(df["window_id"] == w) & (df["device_id"] == dev)]
            if entry.empty:
                continue

            earliest_target = entry["earliest_target_window"].values[0]

            if pd.isna(earliest_target) or earliest_target <= 0:
                # Attack happens within the prediction horizon,
                # so lead time is ≥ 1 window
                lead_windows = 1
            else:
                lead_windows = max(int(earliest_target) - w, 1)

            lead_minutes = lead_windows * window_size
            lead_times_windows.append(lead_windows)
            lead_times_minutes.append(lead_minutes)

            per_window_details.append({
                "prediction_window": int(w),
                "predicted_device": dev,
                "earliest_attack_window": int(earliest_target) if not pd.isna(earliest_target) else w + 1,
                "lead_time_windows": lead_windows,
                "lead_time_minutes": lead_minutes,
                "pred_score": float(row["pred_score"]),
            })
    else:
        # Fallback: use design-level lead time (minimum 1 window = window_size minutes)
        print("  [INFO] 'earliest_target_window' column not found; "
              "using design-level lead time (≥1 window).")
        for _, row in correct_top1.iterrows():
            lead_times_windows.append(1)
            lead_times_minutes.append(window_size)
            per_window_details.append({
                "prediction_window": int(row["window_id"]),
                "predicted_device": row["device_id"],
                "earliest_attack_window": int(row["window_id"]) + 1,
                "lead_time_windows": 1,
                "lead_time_minutes": window_size,
                "pred_score": float(row["pred_score"]),
            })

    if not lead_times_minutes:
        print("  [WARN] No correct Top-1 predictions found for early-warning calculation.")
        return {"error": "No correct predictions in test split"}

    arr = np.array(lead_times_minutes)

    stats = {
        "dataset": "CICIDS2017 Wednesday",
        "window_size_minutes": window_size,
        "prediction_horizon_minutes": horizon_minutes,
        "total_test_windows_with_targets": int(total_windows_with_targets),
        "correct_top1_predictions": int(correct_count),
        "incorrect_top1_predictions": int(incorrect_count),
        "top1_accuracy": round(correct_count / max(total_windows_with_targets, 1), 4),
        "early_warning": {
            "mean_lead_time_minutes": round(float(arr.mean()), 2),
            "min_lead_time_minutes": round(float(arr.min()), 2),
            "max_lead_time_minutes": round(float(arr.max()), 2),
            "median_lead_time_minutes": round(float(np.median(arr)), 2),
            "std_lead_time_minutes": round(float(arr.std()), 2),
            "mean_lead_time_windows": round(float(np.mean(lead_times_windows)), 2),
        },
        "per_window_details": per_window_details,
        "interpretation": (
            f"On average, the model correctly identifies the next attack target "
            f"{arr.mean():.1f} minutes before the attack begins, "
            f"giving defenders {arr.mean():.0f} minutes to respond."
        ),
    }

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run_early_warning(day: str = "wednesday") -> dict:
    print(f"\n{SEPARATOR}")
    print("  EARLY-WARNING TIME EVALUATION")
    print(f"  Dataset: CICIDS2017 {day.capitalize()}")
    print(SEPARATOR)

    # Load feature matrix
    df = load_feature_matrix(day)

    # Identify feature columns
    exclude_cols = {
        "window_id", "device_id", "is_future_target",
        "target_attack_count", "target_attack_types", "earliest_target_window",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]
    ]
    print(f"  Feature columns: {len(feature_cols)}")

    # Temporal split — same as training
    all_windows = sorted(df["window_id"].unique())
    n = len(all_windows)
    test_start = int(n * 0.85)
    test_windows = set(all_windows[test_start:])
    print(f"  Test windows: {len(test_windows)} (windows {min(test_windows)}–{max(test_windows)})")

    # Generate predictions on test split
    print("\n  Running predictions on test split...")
    predictions = predict_all_windows(df, feature_cols, test_windows)

    # Compute early-warning statistics
    print("\n  Computing early-warning lead times...")
    stats = compute_early_warning(
        predictions, df,
        window_size_minutes=TIME_CONFIG["window_size_minutes"],
        horizon_minutes=TIME_CONFIG["prediction_horizon_minutes"],
    )

    # Print summary
    if "error" not in stats:
        ew = stats["early_warning"]
        print(f"\n  {'─'*60}")
        print(f"  Early-Warning Summary:")
        print(f"  {'─'*60}")
        print(f"  Top-1 Accuracy   : {stats['top1_accuracy']:.1%}")
        print(f"  Mean lead time   : {ew['mean_lead_time_minutes']:.1f} minutes")
        print(f"  Min lead time    : {ew['min_lead_time_minutes']:.1f} minutes")
        print(f"  Max lead time    : {ew['max_lead_time_minutes']:.1f} minutes")
        print(f"  Median lead time : {ew['median_lead_time_minutes']:.1f} minutes")
        print(f"\n  Interpretation:")
        print(f"  {stats['interpretation']}")

    # Save
    out_path = PATHS["experiments"] / f"early_warning_{day}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early-warning time evaluation")
    parser.add_argument("--day", default="wednesday", help="Dataset day")
    args = parser.parse_args()

    result = run_early_warning(day=args.day)
    print(f"\n{SEPARATOR}")
    print("  [DONE] Early-warning evaluation complete.")
    print(SEPARATOR)
