"""
Analyze API — POST /api/analyze
=================================

Triggers analysis for a selected time window.
"""

from fastapi import APIRouter

from backend.app.models.schemas import AnalyzeRequest, AnalyzeResponse
from backend.app.services.prediction_service import run_analysis

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def trigger_analysis(request: AnalyzeRequest):
    """
    Trigger a full analysis for a specific time window.

    Runs live model prediction (XGBoost, GNN, or Temporal LSTM) and dynamic
    risk scoring for the selected window with custom weights, returning top-K
    predictions and all device risk scores.
    """
    weights = {
        "w_prob": request.w_prob,
        "w_anom": request.w_anom,
        "w_crit": request.w_crit,
        "w_expo": request.w_expo,
        "w_vuln": request.w_vuln,
    }
    result = run_analysis(
        window_id=request.window_id,
        model=request.model,
        top_k=request.top_k,
        weights=weights,
    )
    return result
