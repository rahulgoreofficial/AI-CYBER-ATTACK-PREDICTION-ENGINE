"""
Risk Service — Dynamic risk score computation and ranking
==========================================================

Provides dynamic risk scores across all network devices in a time window,
supporting on-the-fly re-computation with custom risk weights.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.services.data_loader import get_data_store
from backend.app.services.prediction_service import (
    calculate_device_dynamic_risk,
    predict_window_probabilities,
)

logger = logging.getLogger("backend.risk_service")


def get_risk_scores(
    window_id: Optional[int] = None,
    weights: Optional[dict[str, float]] = None,
    model: str = "xgboost",
) -> dict:
    """
    Get dynamic risk scores for all devices in a time window.

    Args:
        window_id: Time window ID. If None, uses the latest window.
        weights: Optional dictionary of risk component weights.
        model: Model used for attack probabilities (xgboost, gnn, temporal).

    Returns:
        Dict with 'window_id', 'entries' (sorted by risk_rank), 'total_devices'.
    """
    store = get_data_store()

    if window_id is None:
        window_id = store.get_latest_window_id()

    # Get live probabilities and anomalies
    prob_map, anom_map, _ = predict_window_probabilities(window_id, model_name=model)

    result_entries = []
    # Evaluate every device in the network
    for dev_id in store.device_map.keys():
        prob = prob_map.get(dev_id, 0.05)
        anom = anom_map.get(dev_id, 0.05)

        eval_result = calculate_device_dynamic_risk(dev_id, prob, anom, weights=weights)
        eval_result["window_id"] = window_id
        result_entries.append(eval_result)

    # Sort descending by dynamic_risk_score
    result_entries.sort(key=lambda e: e["dynamic_risk_score"], reverse=True)
    for rank, entry in enumerate(result_entries, 1):
        entry["risk_rank"] = rank

    return {
        "window_id": window_id,
        "entries": result_entries,
        "total_devices": len(result_entries),
    }


def get_device_risk(device_id: str, window_id: Optional[int] = None) -> Optional[dict]:
    """
    Get risk data for a specific device in a window.

    Args:
        device_id: Device identifier.
        window_id: Time window ID. If None, uses the latest window.

    Returns:
        Risk entry dict or None if device not in topology.
    """
    store = get_data_store()
    if device_id not in store.device_map:
        return None

    if window_id is None:
        window_id = store.get_latest_window_id()

    # Get probabilities for window
    prob_map, anom_map, _ = predict_window_probabilities(window_id, model_name="xgboost")
    prob = prob_map.get(device_id, 0.05)
    anom = anom_map.get(device_id, 0.05)

    risk_eval = calculate_device_dynamic_risk(device_id, prob, anom)
    risk_eval["window_id"] = window_id
    risk_eval["risk_rank"] = 1
    return risk_eval


def get_available_windows() -> list[dict]:
    """
    Get all available time windows with summary info.

    Returns:
        List of window info dicts with device_count, has_attack, etc.
    """
    store = get_data_store()

    windows = []
    for wid in store.window_ids:
        entries = store.risk_by_window.get(wid, [])
        attack_devices = [e for e in entries if e.get("attack_probability", 0) > 0.5]
        windows.append({
            "window_id": wid,
            "device_count": len(store.device_map),
            "has_attack": len(attack_devices) > 0,
            "attack_device_count": len(attack_devices),
        })

    return windows
