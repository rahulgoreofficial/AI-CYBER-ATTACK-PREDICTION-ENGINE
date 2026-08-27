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

- [ ] M6.1 — SHAP explainer integration (`ml/explainability/`)
- [ ] M6.2 — Rule-based recommendation engine (`backend/app/recommendations/`)
- [ ] M6.3 — Temporal model experimentation (optional/advanced)

## Phase 7 — Backend (M7.1–M7.4)

- [ ] M7.1 — FastAPI app scaffolding
- [ ] M7.2 — Network, Risk, & Prediction API endpoints
- [ ] M7.3 — Analysis trigger endpoint
- [ ] M7.4 — Explanation & Recommendation endpoints

## Phase 8 — Frontend Dashboard (M8.1–M8.5)

- [ ] M8.1 — React + Tailwind CSS scaffold
- [ ] M8.2 — Cytoscape.js Network Graph visualization
- [ ] M8.3 — Ranked prediction list & dynamic risk dashboard
- [ ] M8.4 — SHAP explanation panel & defensive actions cards
- [ ] M8.5 — Attack propagation path animation

## Phase 9 — Integration & Evaluation (M9.1–M9.5)

- [ ] M9.1 — End-to-end pipeline & UI integration
- [ ] M9.2 — Comprehensive ablation study
- [ ] M9.3 — Early-warning time evaluation
- [ ] M9.4 — Multi-dataset generalization test
- [ ] M9.5 — Final technical report, paper, & presentation
