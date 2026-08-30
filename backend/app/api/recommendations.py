"""
Recommendations API — GET /api/recommendations/{device_id}
============================================================

Returns defensive action recommendations for a device.
"""

from fastapi import APIRouter, Path, Query, HTTPException
from typing import Optional

from backend.app.models.schemas import RecommendationResponse
from backend.app.services.data_loader import get_data_store
from backend.app.services.risk_service import get_device_risk
from backend.app.recommendations.engine import RecommendationEngine

router = APIRouter(prefix="/api", tags=["Recommendations"])

# Instantiate the recommendation engine (from M6.2)
_engine = RecommendationEngine()


@router.get("/recommendations/{device_id}", response_model=RecommendationResponse)
async def get_recommendations(
    device_id: str = Path(..., description="Device ID, e.g. WEB-SERVER-01"),
    window_id: Optional[int] = Query(None, description="Time window ID (default: latest)"),
):
    """
    Get defensive action recommendations for a device.

    Uses the rule-based recommendation engine (10 rules across 7 categories)
    to generate prioritized defensive actions based on the device's risk profile,
    SHAP feature explanations, and device metadata.
    """
    store = get_data_store()

    # Validate device exists
    device_info = store.get_device_info(device_id)
    if device_info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{device_id}' not found. "
                   f"Available devices: {list(store.device_map.keys())}",
        )

    # Get risk data for this device
    risk_data = get_device_risk(device_id, window_id=window_id)

    # Get SHAP top features for this device
    device_explanations = store.explanations_by_device.get(device_id, [])
    top_features = []
    if device_explanations:
        # Use the first (or matching window) explanation
        target_expl = device_explanations[0]
        if window_id is not None:
            for expl in device_explanations:
                if expl.get("window_id") == window_id:
                    target_expl = expl
                    break
        top_features = [f["name"] for f in target_expl.get("top_features", [])]

    # Build context for the recommendation engine
    context = {
        "device_id": device_id,
        "attack_probability": risk_data["attack_probability"] if risk_data else 0.0,
        "risk_score": risk_data["dynamic_risk_score"] if risk_data else 0.0,
        "anomaly_score": risk_data["anomaly_score"] if risk_data else 0.0,
        "vulnerability_score": device_info.get("vulnerability", 0.0),
        "criticality": device_info.get("criticality", 0.0),
        "device_type": device_info.get("type", ""),
        "department": device_info.get("department", ""),
        "topology_exposure": risk_data["topology_exposure"] if risk_data else 0.0,
        "top_features": top_features,
    }

    # Generate recommendations
    recommendations = _engine.generate_recommendations(context)

    return {
        "device_id": device_id,
        "recommendations": recommendations,
        "total": len(recommendations),
    }
