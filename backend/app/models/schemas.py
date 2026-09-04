"""
Pydantic v2 Schemas — Request/Response Models for All API Endpoints
===================================================================

Defines typed models for the FastAPI backend. Every API response
is validated through these schemas, ensuring consistent JSON structure
for the React frontend.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Any, Union


# ==============================================================================
# NETWORK / TOPOLOGY
# ==============================================================================

class DeviceNode(BaseModel):
    """A single device in the campus network topology."""
    id: str = Field(..., description="Device identifier, e.g. WEB-SERVER-01")
    name: str = Field("", description="Human-readable device name")
    type: str = Field(..., description="Device type: server, workstation, router, etc.")
    department: str = Field("", description="Department: student, faculty, admin, IT Infrastructure")
    vlan: str = Field("", description="VLAN segment: student, faculty, admin, core, dmz, server")
    os: str = Field("", description="Operating system")
    criticality: float = Field(0.0, ge=0.0, le=1.0, description="Asset criticality [0, 1]")
    vulnerability: float = Field(0.0, ge=0.0, le=1.0, description="Vulnerability score [0, 1]")
    open_ports: list[int] = Field(default_factory=list, description="List of open ports")
    description: str = Field("", description="Device description")
    label: Optional[str] = Field(None, description="Custom display label for Cytoscape")
    # Risk overlay (populated when risk data is available)
    risk_score: Optional[float] = Field(None, description="Dynamic risk score [0, 1]")
    risk_level: Optional[str] = Field(None, description="Risk level: critical, high, medium, low")
    attack_probability: Optional[float] = Field(None, description="Predicted attack probability [0, 1]")


class NetworkEdge(BaseModel):
    """A connection between two devices."""
    source: str = Field(..., description="Source device ID")
    target: str = Field(..., description="Target device ID")
    connection_type: str = Field("", description="Connection type: trunk, ethernet, wifi, fiber")
    bandwidth: str = Field("", description="Bandwidth, e.g. 1Gbps")


class NetworkResponse(BaseModel):
    """Full network topology response."""
    nodes: list[DeviceNode]
    edges: list[NetworkEdge]
    total_nodes: int
    total_edges: int


# ==============================================================================
# RISK
# ==============================================================================

class RiskEntry(BaseModel):
    """Risk score for a single device in a time window."""
    device_id: str
    window_id: Optional[int] = Field(default=None)
    attack_probability: float
    anomaly_score: float
    vulnerability_score: float
    topology_exposure: float
    asset_criticality: float
    recency_score: Optional[float] = Field(default=1.0)
    dynamic_risk_score: float
    risk_rank: int
    risk_level: str = Field("", description="Derived: critical/high/medium/low")


class RiskResponse(BaseModel):
    """Ranked risk scores response."""
    window_id: Optional[int] = Field(default=None)
    entries: list[RiskEntry]
    total_devices: int


# ==============================================================================
# PREDICTIONS
# ==============================================================================

class PredictionEntry(BaseModel):
    """A single device prediction."""
    device_id: str
    attack_probability: float
    rank: int
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    device_type: Optional[str] = None
    department: Optional[str] = None
    criticality: Optional[float] = None


class PredictionResponse(BaseModel):
    """Top-K predictions response."""
    window_id: Optional[int] = Field(default=None)
    model: str
    top_k: int
    predictions: list[PredictionEntry]


# ==============================================================================
# TIMELINE
# ==============================================================================

class TimelineEntry(BaseModel):
    """A single time window entry."""
    window_id: int
    device_count: int
    has_attack: bool
    attack_device_count: int = 0


class TimelineResponse(BaseModel):
    """Available time windows for analysis."""
    total_windows: int
    windows: list[TimelineEntry]


# ==============================================================================
# EVALUATION / MODEL COMPARISON
# ==============================================================================

class ModelMetrics(BaseModel):
    """Performance metrics for a single model."""
    model: str
    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    mrr: float
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float


class EvaluationResponse(BaseModel):
    """Model comparison evaluation response."""
    dataset: str
    models: list[ModelMetrics]
    best_model: str
    best_f1: float


# ==============================================================================
# ANALYSIS TRIGGER
# ==============================================================================

class AnalyzeRequest(BaseModel):
    """Request to trigger analysis on the real-time live telemetry stream."""
    window_id: Optional[int] = Field(default=None, description="Time window ID to analyze (defaults to real-time live stream)")
    model: str = Field("xgboost", description="Model to use: xgboost, gnn, temporal")
    top_k: int = Field(5, ge=1, le=21, description="Number of top predictions to return")
    # Optional dynamic risk weights
    w_prob: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Attack probability weight")
    w_anom: Optional[float] = Field(default=0.15, ge=0.0, le=1.0, description="Anomaly score weight")
    w_crit: Optional[float] = Field(default=0.25, ge=0.0, le=1.0, description="Asset criticality weight")
    w_expo: Optional[float] = Field(default=0.15, ge=0.0, le=1.0, description="Topology exposure weight")
    w_vuln: Optional[float] = Field(default=0.10, ge=0.0, le=1.0, description="Vulnerability score weight")


class AnalyzeResponse(BaseModel):
    """Analysis result for a time window."""
    window_id: Optional[int] = Field(default=None)
    model: str
    predictions: list[PredictionEntry]
    risk_scores: list[RiskEntry]
    total_devices: int
    status: str = "completed"
    inference_ms: Optional[float] = Field(None, description="Actual model inference latency in ms")
    is_live_inference: bool = Field(True, description="Whether predictions were generated live by the model")


# ==============================================================================
# EXPLANATIONS (SHAP)
# ==============================================================================

class ExplanationFeature(BaseModel):
    """A single SHAP feature contribution."""
    name: str
    value: Optional[float] = Field(default=1.0)
    shap_value: float
    direction: str = Field("", description="increases_risk or decreases_risk")
    contribution_pct: float = Field(0.0, description="Percentage contribution")
    importance: Optional[float] = Field(default=0.0)


class DeviceExplanation(BaseModel):
    """SHAP explanation for a single device."""
    device_id: Optional[str] = None
    window_id: Optional[int] = Field(default=None)
    attack_probability: float
    base_value: float
    top_features: list[ExplanationFeature]


class GlobalFeatureImportance(BaseModel):
    """Global SHAP feature importance entry."""
    rank: Optional[int] = Field(default=1)
    feature: str
    mean_abs_shap: Optional[float] = Field(default=0.0)
    importance: Optional[float] = Field(default=0.0)


class ExplanationResponse(BaseModel):
    """Full explanation response for a device."""
    device_id: str
    explanations: list[DeviceExplanation]
    global_importance: list[GlobalFeatureImportance]


# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================

class RecommendationEntry(BaseModel):
    """A single defensive recommendation."""
    action: Optional[str] = Field(default="")
    title: Optional[str] = Field(default="")
    description: Optional[str] = Field(default="")
    mitre_id: Optional[str] = Field(default="")
    mitre_tactic: Optional[str] = Field(default="")
    priority: Any = Field(default="high")
    urgency: Optional[str] = Field(default="high")
    reason: Optional[str] = Field(default="")
    category: Optional[str] = Field(default="")
    rule_name: Optional[str] = Field(default="")


class RecommendationResponse(BaseModel):
    """Recommendations for a device."""
    device_id: str
    recommendations: list[RecommendationEntry]
    total: Optional[int] = Field(default=0)


# ==============================================================================
# ATTACK PATH
# ==============================================================================

class AttackPathNode(BaseModel):
    """A node in the attack propagation path."""
    device_id: str
    device_type: str = ""
    attack_probability: float = 0.0
    risk_score: float = 0.0
    step: int = 0


class AttackPathResponse(BaseModel):
    """Attack propagation path for a device."""
    device_id: str
    path: list[AttackPathNode]
    total_steps: int
    description: str = ""


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: bool = False
    data_loaded: bool = False
    device_count: int = 0
    window_count: int = 0
