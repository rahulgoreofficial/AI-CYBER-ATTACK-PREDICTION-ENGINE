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

    # Check if this is a real physical LAN device
    if any(k in device_id for k in ["192.168", "HOST", "LAN", "Router", "Gateway", "Phone", "Smart", "Peer", "Client"]) or device_info is None:
        from backend.app.services.lan_service import get_lan_recommendations
        return get_lan_recommendations(device_id)

    if window_id is None:
        window_id = store.get_latest_window_id()

    # Get risk data for this device
    risk_data = get_device_risk(device_id, window_id=window_id)

    # Get SHAP top features for this device (or compute live!)
    device_explanations = store.explanations_by_device.get(device_id, [])
    top_features = []
    if device_explanations:
        target_expl = device_explanations[0]
        if window_id is not None:
            for expl in device_explanations:
                if expl.get("window_id") == window_id:
                    target_expl = expl
                    break
        top_features = [f["name"] for f in target_expl.get("top_features", [])]
    else:
        # Live SHAP calculation
        from backend.app.services.prediction_service import get_live_device_explanation
        live_expl = get_live_device_explanation(device_id, window_id)
        if live_expl.get("explanations"):
            top_features = [f["name"] for f in live_expl["explanations"][0].get("top_features", [])]

    # Build context for the recommendation engine
    context = {
        "device_id": device_id,
        "attack_probability": risk_data["attack_probability"] if risk_data else 0.05,
        "risk_score": risk_data["dynamic_risk_score"] if risk_data else 0.15,
        "anomaly_score": risk_data["anomaly_score"] if risk_data else 0.05,
        "vulnerability_score": device_info.get("vulnerability", 0.2),
        "criticality": device_info.get("criticality", 0.5),
        "device_type": device_info.get("type", "workstation"),
        "department": device_info.get("department", "general"),
        "topology_exposure": risk_data["topology_exposure"] if risk_data else store.get_topology_exposure(device_id),
        "top_features": top_features,
    }

    # Generate recommendations
    recommendations = _engine.generate_recommendations(context)

    # If no high-priority alert rules triggered, supply proactive baseline defensive hygiene
    if not recommendations:
        dev_type = device_info.get("type", "device")
        crit = device_info.get("criticality", 0.5)
        prio = "high" if crit >= 0.7 else "medium" if crit >= 0.4 else "low"

        recommendations = [
            {
                "action": f"Verify 802.1X network access control & VLAN segmentation on {device_id}",
                "priority": prio,
                "reason": f"Device is a critical {dev_type} in {device_info.get('department', 'campus')} subnet with normalized exposure {context['topology_exposure']:.2f}.",
                "category": "access_control",
                "rule_name": "baseline_segmentation",
            },
            {
                "action": f"Enable continuous NetFlow anomaly telemetry and syslog forwarding for {device_id}",
                "priority": "low",
                "reason": "Ensure behavioral baselining and rapid lateral movement detection.",
                "category": "monitoring",
                "rule_name": "baseline_telemetry",
            },
            {
                "action": f"Review open ports ({', '.join(str(p) for p in device_info.get('open_ports', [80, 443]))}) and enforce least-privilege ACLs",
                "priority": "low",
                "reason": "Prevent unauthorized lateral hopping from potentially compromised adjacent nodes.",
                "category": "patching",
                "rule_name": "least_privilege_audit",
            },
        ]

    return {
        "device_id": device_id,
        "recommendations": recommendations,
        "total": len(recommendations),
    }
