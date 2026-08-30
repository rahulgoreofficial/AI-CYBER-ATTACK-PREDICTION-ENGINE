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

    # Validate device exists
    if device_id not in store.device_map:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{device_id}' not found. "
                   f"Available devices: {list(store.device_map.keys())}",
        )

    # Get local explanations for this device
    device_explanations = store.explanations_by_device.get(device_id, [])

    if window_id is not None:
        device_explanations = [
            e for e in device_explanations
            if e.get("window_id") == window_id
        ]

    return {
        "device_id": device_id,
        "explanations": device_explanations,
        "global_importance": store.global_importance,
    }
