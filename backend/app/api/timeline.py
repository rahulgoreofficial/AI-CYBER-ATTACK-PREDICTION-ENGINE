"""
Timeline API — GET /api/timeline
==================================

Returns available time windows for analysis.
"""

from fastapi import APIRouter

from backend.app.models.schemas import TimelineResponse
from backend.app.services.risk_service import get_available_windows

router = APIRouter(prefix="/api", tags=["Timeline"])


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline():
    """
    Get all available time windows for analysis.

    Returns a list of time windows with device counts and attack indicators.
    The frontend can use this to populate the timeline selector.
    """
    windows = get_available_windows()
    return {
        "total_windows": len(windows),
        "windows": windows,
    }
