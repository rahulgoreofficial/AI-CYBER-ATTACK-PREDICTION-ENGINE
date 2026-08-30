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

    Runs prediction and risk scoring for the selected window and
    returns the complete analysis results including top-K predictions
    and all device risk scores.
    """
    result = run_analysis(
        window_id=request.window_id,
        model=request.model,
        top_k=request.top_k,
    )
    return result
