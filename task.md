# Task Tracker — AI Cyber Attack Prediction Engine

## Phase 1 — Foundation (Milestones M1.1–M1.4)

### M1.1 — Project Setup

- [x] Create directory scaffold (all folders)
- [x] Create `README.md`
- [x] Create `.gitignore`
- [x] Create `requirements.txt`
- [x] Create `ml/config.py`
- [x] Initialize git repository ✓

### M1.2 — Synthetic Topology

- [x] Create `data/synthetic/campus_topology.json` (21 devices, 30 connections, 7 segments)
- [x] Verify topology loads correctly ✓

### M1.3 — Dataset Acquisition

- [x] Create `docs/dataset_notes.md` (download instructions, known issues, day-by-day content)
- [x] Create `docs/target_definition.md` (temporal target definition, anti-leakage rules)
- [x] Download CICIDS2017 dataset into `data/raw/cicids2017/` ✓

### M1.4 — Data Exploration

- [x] Dataset schema, file inspection, class distribution & version detection (`ml_only` vs `full`) ✓

---

## Phase 2 — Preprocessing Pipeline (M2.1–M2.5)

- [x] M2.1 — Cleaning pipeline (`loader.py`, `cleaner.py`) ✓
- [x] M2.2 — Encoding + scaling (`encoder.py`, `scaler.py` incorporated / handled) ✓
- [x] M2.3 — IP-to-device mapping & synthetic enrichment (`synthetic_enrichment.py`) ✓
- [x] M2.4 — Time windowing & Pipeline Execution on full Wednesday dataset (`pipeline.py`) ✓
- [x] M2.5 — Target label generation (`target_generator.py`) ✓

---

## Phase 3 — Graph & Features (M3.1–M3.4)

- [x] M3.1 — Graph construction per time window (`graph/construction.py`) — 97 graphs, avg 24.6 nodes, 149.5 edges ✓
- [x] M3.2 — Graph feature extraction (degree, betweenness, closeness, PageRank in `graph/features.py`) — 12 features, 2,388 rows ✓
- [x] M3.3 — Attack sequence & propagation chain extraction (`graph/attack_chains.py`) — 0 chains (Wed attacks are single-hop DoS) ✓
- [x] M3.4 — Combined feature matrix generation (`ml/feature_engineering/feature_combiner.py`) — 1,649 samples × 120 features, saved to CSV ✓

## Phase 4 -- Baseline Models (M4.1--M4.4)

- [x] M4.1 -- Heuristic baseline model (degree x criticality x neighbor_attacks) -- Top-1=1.0, F1=0.40
- [x] M4.2 -- XGBoost next-target ranking model -- Top-1=1.0, F1=0.97, PR-AUC=1.0
- [x] M4.3 -- Isolation Forest anomaly detection + XGBoost -- anomaly_score_if ranked 3rd most important feature
- [x] M4.4 -- Dynamic risk engine formulation -- F1=1.0, risk rankings show WEB-SERVER-01 consistently #1

## Phase 5 — GNN (M5.1–M5.3)

- [x] M5.1 — PyTorch Geometric dataset converter (`ml/gnn/dataset.py`) — 97 graphs, 22 nodes avg, 120 features, 4 edge features ✓
- [x] M5.2 — GNN model architecture (`ml/gnn/model.py`) — 2-layer GraphSAGE, 24,001 params ✓
- [x] M5.3 — GNN training & evaluation vs. XGBoost baseline — Top-1=0.86–1.0, Top-3=0.93–1.0, MRR=0.90–1.0, ROC-AUC=0.938, CUDA accelerated on RTX 5060 ✓

## Phase 6 — Explainability & Recommendations (M6.1–M6.3)

- [x] M6.1 — SHAP explainer integration (`ml/explainability/shap_explainer.py`, `local_explanations.py`) — TreeExplainer for XGBoost, global feature importance (vlan_dmz, dst_total_fwd_packets_sum #1 & #2), per-prediction contribution breakdowns, saved to `experiments/shap_analysis_wednesday.json` & `local_explanations_wednesday.json` ✓
- [x] M6.2 — Rule-based recommendation engine (`backend/app/recommendations/engine.py`) — 10 prioritized rules across 7 categories (incident response, isolation, access control, monitoring, patching, micro-segmentation, data protection) with unit tests passing ✓
- [x] M6.3 — Temporal model experimentation (`ml/temporal/temporal_lstm.py`) — Lookback sequence dataset (1,615 sequences), 1-layer LSTM (47,681 params), Top-1=0.57, Top-3=0.86, MRR=0.71, PR-AUC=0.59, ROC-AUC=0.92, model saved to `models/temporal_lstm.pt` ✓

---

## Phase 7 — Backend (M7.1–M7.4) ✅ COMPLETE

### M7.1 — FastAPI App Scaffolding
- [x] Create `backend/app/main.py` — FastAPI entry point with CORS, lifespan startup, health check ✓
- [x] Create `backend/app/models/schemas.py` — 18 Pydantic v2 models for all endpoints ✓
- [x] Create `backend/app/services/data_loader.py` — Singleton loading 21 devices, 255 risk entries, 1,649 explanations, 3 ML models ✓
- [x] Verify server starts and `/health` returns 200 ✓

### M7.2 — Network, Risk & Prediction API Endpoints
- [x] Create `backend/app/api/network.py` — `GET /api/network` (21 nodes, 30 edges, risk overlay) ✓
- [x] Create `backend/app/api/risk.py` — `GET /api/risk` (ranked risk scores, WEB-SERVER-01 #1) ✓
- [x] Create `backend/app/api/predictions.py` — `GET /api/predictions` (top-K targets with probabilities) ✓
- [x] Create `backend/app/api/timeline.py` — `GET /api/timeline` (15 time windows) ✓
- [x] Create `backend/app/api/evaluation.py` — `GET /api/evaluation` (6 models, best=Dynamic Risk F1=1.0) ✓
- [x] Create `backend/app/services/prediction_service.py` — Prediction inference + analysis logic ✓
- [x] Create `backend/app/services/risk_service.py` — Risk score lookups + window listing ✓
- [x] Create `backend/app/services/graph_service.py` — Topology formatting with risk overlays ✓

### M7.3 — Analysis Trigger Endpoint
- [x] Create `backend/app/api/analyze.py` — `POST /api/analyze` (returns predictions + risk scores) ✓

### M7.4 — Explanation & Recommendation Endpoints
- [x] Create `backend/app/api/explanation.py` — `GET /api/explanation/{device_id}` (97 SHAP explanations for WEB-SERVER-01) ✓
- [x] Create `backend/app/api/recommendations.py` — `GET /api/recommendations/{device_id}` (SOC escalation, monitoring, segmentation) ✓
- [x] Create `backend/app/api/attack_path.py` — `GET /api/attack-path/{device_id}` (5-hop path: PC-08→...→WEB-SERVER-01) ✓
- [x] End-to-end verification: **11/11 endpoints return correct JSON, all HTTP 200** ✓

---

## Phase 8 — Frontend Dashboard (M8.1–M8.5) ✅ COMPLETE

- [x] M8.1 — React + Vite cyber-theme scaffold (`src/index.css`, `App.jsx`, `Sidebar.jsx`, `Header.jsx`) ✓
- [x] M8.2 — Cytoscape.js Network Graph visualization (dagre layout, 21 nodes, 30 edges, risk overlays, legend) ✓
- [x] M8.3 — Ranked prediction list & dynamic risk dashboard (`PredictionPanel.jsx`, `RiskTable.jsx`, `TimelineSelector.jsx`, `MetricsPanel.jsx`) ✓
- [x] M8.4 — SHAP explanation panel & defensive actions cards (`ExplanationPanel.jsx`, `RecommendationPanel.jsx`) ✓
- [x] M8.5 — Attack propagation path animation (`AttackPath.jsx`, 5-hop path trace & graph highlight) ✓
- [x] Deliverable: Technical Analysis & Quickstart Guide PDF (`AI_Cyber_Attack_Prediction_Engine_Analysis_and_Tutorial.pdf`) ✓
- [x] Deliverable: End-to-end automated verification script passing for all ML, Backend, and Frontend layers ✓

## Phase 9 — Integration & Evaluation (M9.1–M9.5) ✅ COMPLETE

- [x] M9.1 — End-to-end pipeline integration (`run_pipeline.py`) — Top-1=1.0, WEB-SERVER-01 ranked #1 with risk=0.8527 ✓
- [x] M9.2 — Comprehensive ablation study (`ml/evaluation/ablation_study.py`) — 10 variants; all maintain Top-1=1.0; asset features contribute most to F1 ✓
- [x] M9.3 — Early-warning time evaluation (`ml/evaluation/early_warning.py`) — 100% Top-1 on test windows, 5-min avg lead time ✓
- [x] M9.4 — Multi-dataset generalization test (`ml/evaluation/generalization_test.py`) — IDS2018 zero-shot: Top-3=1.0, ROC-AUC=0.856 ✓
- [x] M9.5 — Final technical report (`docs/evaluation_results.md`) — model comparison, ablation, early-warning, generalization tables ✓
