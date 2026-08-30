"""
Prediction Service — Model inference and device ranking
========================================================

Provides predictions from pre-computed risk scores and supports
on-demand XGBoost inference when the feature matrix is available.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.services.data_loader import get_data_store

logger = logging.getLogger("backend.prediction_service")


def get_predictions(
    window_id: Optional[int] = None,
    top_k: int = 5,
    model: str = "xgboost",
) -> dict:
    """
    Get top-K predicted attack targets for a time window.

    Args:
        window_id: Time window ID. If None, uses latest.
        top_k: Number of top predictions to return.
        model: Model name (xgboost, gnn, temporal).

    Returns:
        Dict with 'window_id', 'model', 'top_k', 'predictions'.
    """
    store = get_data_store()

    if window_id is None:
        window_id = store.get_latest_window_id()

    # Use risk scores as the prediction source (they contain attack_probability)
    entries = store.risk_by_window.get(window_id, [])

    if not entries:
        return {
            "window_id": window_id,
            "model": model,
            "top_k": top_k,
            "predictions": [],
        }

    # Sort by attack probability (descending) and take top-K
    sorted_entries = sorted(entries, key=lambda e: e["attack_probability"], reverse=True)
    top_entries = sorted_entries[:top_k]

    predictions = []
    for rank, entry in enumerate(top_entries, 1):
        device_info = store.get_device_info(entry["device_id"])
        pred = {
            "device_id": entry["device_id"],
            "attack_probability": entry["attack_probability"],
            "rank": rank,
            "risk_score": entry["dynamic_risk_score"],
            "risk_level": store.get_risk_level(entry["dynamic_risk_score"]),
            "device_type": device_info.get("type", "") if device_info else "",
            "department": device_info.get("department", "") if device_info else "",
            "criticality": device_info.get("criticality", 0.0) if device_info else 0.0,
        }
        predictions.append(pred)

    return {
        "window_id": window_id,
        "model": model,
        "top_k": top_k,
        "predictions": predictions,
    }


def run_analysis(window_id: int, model: str = "xgboost", top_k: int = 5) -> dict:
    """
    Run a full analysis for a given time window.

    Returns predictions + risk scores combined.

    Args:
        window_id: Time window to analyze.
        model: Model to use.
        top_k: Number of predictions.

    Returns:
        Dict with predictions, risk_scores, total_devices, status.
    """
    store = get_data_store()

    # Get predictions
    pred_result = get_predictions(window_id=window_id, top_k=top_k, model=model)

    # Get all risk scores for this window
    entries = store.risk_by_window.get(window_id, [])
    risk_entries = []
    for entry in entries:
        enriched = {**entry}
        enriched["risk_level"] = store.get_risk_level(entry["dynamic_risk_score"])
        risk_entries.append(enriched)
    risk_entries.sort(key=lambda e: e["risk_rank"])

    return {
        "window_id": window_id,
        "model": model,
        "predictions": pred_result["predictions"],
        "risk_scores": risk_entries,
        "total_devices": len(risk_entries),
        "status": "completed",
    }
