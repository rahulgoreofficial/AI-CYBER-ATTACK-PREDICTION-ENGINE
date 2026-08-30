"""
Risk Service — Dynamic risk score computation and ranking
==========================================================

Provides risk scores from pre-computed data and supports
re-computation with adjustable weights.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.app.services.data_loader import get_data_store

logger = logging.getLogger("backend.risk_service")


def get_risk_scores(window_id: Optional[int] = None) -> dict:
    """
    Get risk scores for all devices in a time window.

    Args:
        window_id: Time window ID. If None, uses the latest window.

    Returns:
        Dict with 'window_id', 'entries' (sorted by risk_rank), 'total_devices'.
    """
    store = get_data_store()

    if window_id is None:
        window_id = store.get_latest_window_id()

    entries = store.risk_by_window.get(window_id, [])

    # Add risk_level and sort by risk_rank
    result_entries = []
    for entry in entries:
        enriched = {**entry}
        enriched["risk_level"] = store.get_risk_level(entry["dynamic_risk_score"])
        result_entries.append(enriched)

    result_entries.sort(key=lambda e: e["risk_rank"])

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
        Risk entry dict or None if not found.
    """
    store = get_data_store()

    if window_id is None:
        window_id = store.get_latest_window_id()

    entries = store.risk_by_window.get(window_id, [])
    for entry in entries:
        if entry["device_id"] == device_id:
            result = {**entry}
            result["risk_level"] = store.get_risk_level(entry["dynamic_risk_score"])
            return result

    return None


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
        # A window has attacks if any device has attack_probability > 0.5
        attack_devices = [e for e in entries if e.get("attack_probability", 0) > 0.5]
        windows.append({
            "window_id": wid,
            "device_count": len(entries),
            "has_attack": len(attack_devices) > 0,
            "attack_device_count": len(attack_devices),
        })

    return windows
