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

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models.schemas import HealthResponse
from backend.app.services.data_loader import get_data_store
from backend.app.services.scanner import get_scanner

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

# ──────────────────────────────────────────────────────────────────────────
# WEBSOCKET MANAGER — Real-time device event broadcasting
# ──────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages WebSocket connections for real-time network event broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected ({len(self.active_connections)} clients)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected ({len(self.active_connections)} clients)")

    async def broadcast(self, message: dict[str, Any]):
        """Broadcast a JSON message to all connected WebSocket clients."""
        if not self.active_connections:
            return
        text = json.dumps(message)
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_text(text)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data at startup, start network scanner, cleanup at shutdown."""
    logger.info("=" * 60)
    logger.info("AI Cyber Attack Prediction Engine — Backend Starting")
    logger.info("=" * 60)

    # Load all data and models into memory
    store = get_data_store()
    store.load_all()

    # Start background network scanner
    scanner = get_scanner()
    main_loop = asyncio.get_running_loop()

    def on_network_change(event: dict):
        """Called by the scanner when a network change is detected."""
        try:
            if main_loop.is_running():
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast(event), main_loop)
        except Exception as e:
            logger.debug(f"Broadcast skipped: {e}")

    scanner.add_change_callback(on_network_change)
    scanner_task = asyncio.create_task(scanner.run_loop(interval=8))

    logger.info("=" * 60)
    logger.info("Backend ready — all data loaded, network scanner active")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    scanner.stop()
    scanner_task.cancel()
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
    allow_origins=["*"],  # Allow localhost and any device on the same LAN/Wi-Fi subnet
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────
# REGISTER ROUTERS
# ──────────────────────────────────────────────────────────────────

from backend.app.api.network import router as network_router
from backend.app.api.risk import router as risk_router
from backend.app.api.predictions import router as predictions_router
from backend.app.api.timeline import router as timeline_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.analyze import router as analyze_router
from backend.app.api.explanation import router as explanation_router
from backend.app.api.recommendations import router as recommendations_router
from backend.app.api.attack_path import router as attack_path_router
from backend.app.api.lan import router as lan_router

app.include_router(network_router)
app.include_router(risk_router)
app.include_router(predictions_router)
app.include_router(timeline_router)
app.include_router(evaluation_router)
app.include_router(analyze_router)
app.include_router(explanation_router)
app.include_router(recommendations_router)
app.include_router(attack_path_router)
app.include_router(lan_router)


# ──────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT — Real-time network events
# ──────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/network")
async def websocket_network(websocket: WebSocket):
    """
    WebSocket endpoint for real-time network device events.
    Clients receive JSON messages with types: connect, disconnect, update, full_scan.
    """
    await ws_manager.connect(websocket)

    # Send initial state immediately
    scanner = get_scanner()
    initial_devices = scanner.get_all_devices()
    try:
        await websocket.send_text(json.dumps({
            "type": "full_scan",
            "device": {},
            "total_devices": len(initial_devices),
            "all_devices": initial_devices,
            "timestamp": None,
            "scan_cycle": scanner._scan_count,
        }))
    except Exception:
        pass

    try:
        while True:
            # Keep the connection alive by waiting for client messages
            # (client can send pings or config changes)
            data = await websocket.receive_text()
            # Echo back acknowledgment
            await websocket.send_text(json.dumps({"type": "ack", "message": data}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


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
    scanner = get_scanner()
    scanner_status = scanner.get_status()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": store.models_loaded,
        "data_loaded": store.is_loaded,
        "device_count": scanner_status["online_devices"] or len(store.devices),
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
