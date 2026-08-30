"""
Network API — GET /api/network
================================

Returns the campus network topology with risk overlays.
"""

from fastapi import APIRouter, Query
from typing import Optional

from backend.app.models.schemas import NetworkResponse
from backend.app.services.graph_service import get_network_topology

router = APIRouter(prefix="/api", tags=["Network"])


@router.get("/network", response_model=NetworkResponse)
async def get_network(
    window_id: Optional[int] = Query(None, description="Time window ID for risk overlay"),
):
    """
    Get the campus network topology.

    Returns all devices (nodes) and connections (edges) with optional
    risk score overlays from a specific time window. If no window_id
    is provided, uses the latest available window.
    """
    result = get_network_topology(window_id=window_id)
    return result
