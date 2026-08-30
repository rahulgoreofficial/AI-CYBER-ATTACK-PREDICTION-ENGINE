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
):
    """
    Get dynamic risk scores for all devices in a time window.

    Returns devices ranked by dynamic risk score, which combines
    attack probability, anomaly score, vulnerability, topology exposure,
    criticality, and recency.
    """
    result = get_risk_scores(window_id=window_id)
    return result
