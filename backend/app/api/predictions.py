"""
Predictions API — GET /api/predictions
========================================

Returns top-K predicted future attack targets.
"""

from fastapi import APIRouter, Query
from typing import Optional

from backend.app.models.schemas import PredictionResponse
from backend.app.services.prediction_service import get_predictions

router = APIRouter(prefix="/api", tags=["Predictions"])


@router.get("/predictions", response_model=PredictionResponse)
async def get_prediction_list(
    window_id: Optional[int] = Query(None, description="Time window ID (default: latest)"),
    top_k: int = Query(5, ge=1, le=21, description="Number of top predictions"),
    model: str = Query("xgboost", description="Model: xgboost, gnn, temporal"),
    source: Optional[str] = Query("lan", description="'lan' for real network devices, 'campus' for benchmark"),
):
    """
    Get top-K predicted future attack targets.
    Defaults to real physical LAN devices, or enterprise campus simulation.
    """
    if source == "lan":
        from backend.app.services.lan_service import get_lan_predictions
        return get_lan_predictions(top_k=top_k, model=model)

    result = get_predictions(window_id=window_id, top_k=top_k, model=model)
    return result
