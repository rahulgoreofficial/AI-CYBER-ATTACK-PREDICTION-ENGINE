"""
Local Explanations Utility Module
=================================

Provides helper functions for local (per-instance, per-window, per-device)
explanations using SHAPExplainer.

Functions:
    - analyze_top_predictions: Evaluates and generates local explanations for the top predicted targets.
    - explain_top_predictions_for_day: Generates per-window top-K device prediction explanations.
    - save_per_device_explanations: Extracts and saves detailed explanation records per device to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from ml.config import PATHS
from ml.explainability.shap_explainer import SHAPExplainer


def analyze_top_predictions(
    day: str = "wednesday",
    top_k: int = 5,
    model_name: str = "xgboost_baseline",
) -> list[dict[str, Any]]:
    """
    Analyze and explain the top-K highest risk/attack probability devices
    across all windows for a given day.
    """
    explainer = SHAPExplainer.from_saved_model(day=day, model_name=model_name)
    fm = explainer.feature_matrix

    # Find highest probability or actual target records
    if "is_future_target" in fm.columns:
        top_records = fm[fm["is_future_target"] == 1]
        if len(top_records) == 0:
            top_records = fm.head(top_k)
        else:
            top_records = top_records.head(top_k)
    else:
        top_records = fm.head(top_k)

    explanations = []
    for _, row in top_records.iterrows():
        device_id = str(row["device_id"])
        window_id = int(row["window_id"])
        exp = explainer.explain_prediction(device_id=device_id, window_id=window_id, top_n=10)
        explanations.append(exp)

    return explanations


def explain_top_predictions_for_day(
    day: str = "wednesday",
    window_id: int | None = None,
    top_k: int = 5,
    model_name: str = "xgboost_baseline",
) -> list[dict[str, Any]]:
    """
    Generate SHAP explanations for top-K predictions in a specific window,
    or for each window if window_id is None.
    """
    explainer = SHAPExplainer.from_saved_model(day=day, model_name=model_name)
    fm = explainer.feature_matrix

    if window_id is not None:
        window_ids = [window_id]
    else:
        window_ids = sorted(fm["window_id"].unique().tolist())[:5]

    results = []
    for wid in window_ids:
        w_df = fm[fm["window_id"] == wid]
        for _, row in w_df.head(top_k).iterrows():
            exp = explainer.explain_prediction(
                device_id=str(row["device_id"]),
                window_id=int(wid),
                top_n=5,
            )
            results.append(exp)

    return results


def save_per_device_explanations(
    day: str = "wednesday",
    output_path: Path | str | None = None,
    model_name: str = "xgboost_baseline",
    top_n_features: int = 5,
) -> Path:
    """
    Compute and save local explanations for key devices to disk for fast API lookup.
    """
    explainer = SHAPExplainer.from_saved_model(day=day, model_name=model_name)
    explanations = explainer.explain_all_predictions(top_n=top_n_features)

    if output_path is None:
        output_path = PATHS["experiments"] / f"local_explanations_{day}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)

    print(f"[shap] Saved {len(explanations)} per-device local explanations to {output_path}")
    return output_path
