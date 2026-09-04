"""
Risk API — GET /api/risk
=========================

Returns ranked dynamic risk scores for all devices.
"""

from fastapi import APIRouter, Query
from typing import Optional

from backend.app.models.schemas import RiskResponse
from backend.app.services.risk_service import get_risk_scores

router = APIRouter(prefix="/api", tags=["Risk"])


@router.get("/risk", response_model=RiskResponse)
async def get_risk(
    window_id: Optional[int] = Query(None, description="Time window ID (default: latest)"),
    source: Optional[str] = Query("lan", description="'lan' for real network devices, 'campus' for benchmark"),
):
    """
    Get dynamic risk scores for all devices.
    Defaults to real physical LAN devices, or enterprise campus simulation.
    """
    if source == "lan":
        from backend.app.services.lan_service import get_lan_risk_scores
        return get_lan_risk_scores()

    result = get_risk_scores(window_id=window_id)
    return result
