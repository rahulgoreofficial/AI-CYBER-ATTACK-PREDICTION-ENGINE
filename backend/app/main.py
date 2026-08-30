"""
FastAPI Application Entry Point — AI Cyber Attack Prediction Engine
====================================================================

Main application that:
- Initializes the FastAPI app with metadata
- Configures CORS for the React frontend
- Loads all data/models at startup
- Registers all API routers
- Provides a health check endpoint

Run:
    cd c:\\EDI\\Sem 3\\antitry1
    python -m uvicorn backend.app.main:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs

ReDoc:
    http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models.schemas import HealthResponse
from backend.app.services.data_loader import get_data_store

# ──────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend.main")


# ──────────────────────────────────────────────────────────────────────────
# LIFESPAN (startup/shutdown)
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data at startup, cleanup at shutdown."""
    logger.info("=" * 60)
    logger.info("AI Cyber Attack Prediction Engine — Backend Starting")
    logger.info("=" * 60)

    # Load all data and models into memory
    store = get_data_store()
    store.load_all()

    logger.info("=" * 60)
    logger.info("Backend ready — all data loaded")
    logger.info("=" * 60)

    yield  # Application runs here

    logger.info("Backend shutting down...")


# ──────────────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Cyber Attack Prediction Engine",
    description=(
        "Proactive cybersecurity decision-support API combining temporal attack "
        "propagation analysis, network graph topology, anomaly detection, and "
        "explainable AI to predict future attack targets and prioritize defensive responses."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ──────────────────────────────────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # React dev server (CRA)
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────
# REGISTER ROUTERS
# ──────────────────────────────────────────────────────────────────────────

from backend.app.api.network import router as network_router
from backend.app.api.risk import router as risk_router
from backend.app.api.predictions import router as predictions_router
from backend.app.api.timeline import router as timeline_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.analyze import router as analyze_router
from backend.app.api.explanation import router as explanation_router
from backend.app.api.recommendations import router as recommendations_router
from backend.app.api.attack_path import router as attack_path_router

app.include_router(network_router)
app.include_router(risk_router)
app.include_router(predictions_router)
app.include_router(timeline_router)
app.include_router(evaluation_router)
app.include_router(analyze_router)
app.include_router(explanation_router)
app.include_router(recommendations_router)
app.include_router(attack_path_router)


# ──────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the server status, whether data and models are loaded,
    and summary counts.
    """
    store = get_data_store()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": store.models_loaded,
        "data_loaded": store.is_loaded,
        "device_count": len(store.devices),
        "window_count": len(store.window_ids),
    }


# ──────────────────────────────────────────────────────────────────────────
# ROOT
# ──────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "message": "AI Cyber Attack Prediction Engine API",
        "docs": "/docs",
        "health": "/health",
    }
