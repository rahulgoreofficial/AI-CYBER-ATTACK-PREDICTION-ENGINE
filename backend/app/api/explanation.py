"""
Explanation API — GET /api/explanation/{device_id}
===================================================

Returns SHAP-based prediction explanations for a device.
"""

from fastapi import APIRouter, Path, Query, HTTPException
from typing import Optional

from backend.app.models.schemas import ExplanationResponse
from backend.app.services.data_loader import get_data_store

router = APIRouter(prefix="/api", tags=["Explanations"])


@router.get("/explanation/{device_id}", response_model=ExplanationResponse)
async def get_explanation(
    device_id: str = Path(..., description="Device ID, e.g. WEB-SERVER-01"),
    window_id: Optional[int] = Query(None, description="Specific window ID (default: all)"),
):
    """
    Get SHAP-based prediction explanation for a device.

    Returns the top contributing features (with SHAP values, direction,
    and contribution percentage) for the specified device. Also includes
    global feature importance rankings.
    """
    store = get_data_store()

    # Check if this is a real physical LAN device
    if any(k in device_id for k in ["192.168", "HOST", "LAN", "Router", "Gateway", "Phone", "Smart", "Peer", "Client"]) or device_id not in store.device_map:
        from backend.app.services.lan_service import get_lan_explanation
        return get_lan_explanation(device_id)

    if window_id is None:
        window_id = store.get_latest_window_id()

    # Get local explanations for this device at the current live window
    device_explanations = [
        e for e in store.explanations_by_device.get(device_id, [])
        if e.get("window_id") == window_id
    ]

    # If no precomputed explanation exists for this window, compute live on-demand Tree SHAP!
    if not device_explanations:
        from backend.app.services.prediction_service import get_live_device_explanation
        return get_live_device_explanation(device_id=device_id, window_id=window_id)

    return {
        "device_id": device_id,
        "explanations": device_explanations,
        "global_importance": store.global_importance,
    }
