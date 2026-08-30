"""
Evaluation API — GET /api/evaluation
======================================

Returns model performance metrics (comparison table).
"""

from fastapi import APIRouter

from backend.app.models.schemas import EvaluationResponse
from backend.app.services.data_loader import get_data_store

router = APIRouter(prefix="/api", tags=["Evaluation"])


@router.get("/evaluation", response_model=EvaluationResponse)
async def get_evaluation():
    """
    Get model comparison evaluation metrics.

    Returns performance metrics (Top-K, MRR, PR-AUC, F1, etc.) for all
    trained models: Heuristic, XGBoost, XGBoost + IF, Dynamic Risk,
    GNN (GraphSAGE), and Temporal LSTM.
    """
    store = get_data_store()

    models = store.model_comparison

    # Find best model by F1
    best_model = ""
    best_f1 = 0.0
    for m in models:
        if m.get("f1", 0) > best_f1:
            best_f1 = m["f1"]
            best_model = m["model"]

    return {
        "dataset": "CICIDS2017 Wednesday",
        "models": models,
        "best_model": best_model,
        "best_f1": best_f1,
    }
