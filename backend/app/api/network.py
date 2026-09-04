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
    source: Optional[str] = Query("lan", description="'lan' for real physical network devices, 'campus' for 21-node benchmark"),
):
    """
    Get network topology with risk overlays.
    Defaults to 'lan' (real connected devices on local Wi-Fi/LAN), or 'campus' (benchmark model).
    """
    if source == "lan":
        from backend.app.services.lan_service import get_lan_network_topology
        return get_lan_network_topology()

    result = get_network_topology(window_id=window_id)
    return result
