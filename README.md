# AI Cyber Attack Prediction Engine

## Using Graph Intelligence and Machine Learning

> **Predict where an attack is likely to propagate next and help defenders decide what to protect first.**

A proactive cybersecurity decision-support platform that combines temporal attack propagation analysis, network graph topology, anomaly detection, and explainable AI to predict future attack targets and prioritize defensive responses.

---

## Project Overview

| Aspect | Detail |
|---|---|
| **Domain** | AI, Machine Learning, Graph ML, Cybersecurity |
| **Primary Use Case** | University/Campus Network Defense |
| **Core Output** | Next-target prediction + risk prioritization + explanation |
| **Datasets** | CICIDS2017, CSE-CIC-IDS2018, Synthetic Campus Topology |
| **ML Models** | XGBoost, Isolation Forest, GNN (PyTorch Geometric) |
| **Backend** | FastAPI + PostgreSQL |
| **Frontend** | React + Tailwind CSS + Cytoscape.js |

## System Architecture

```
Network Traffic / Alerts / Host Logs
        ↓
  Data Preprocessing (Cleaning / Encoding / Scaling)
        ↓
  Temporal Event Table
        ↓
  Network Topology + Assets (Graph Construction)
        ↓
  Feature Engineering (Traffic + Anomaly + Graph + Temporal)
        ↓
  ┌──────────────────────────┐
  │   Parallel AI Models     │
  │  • Isolation Forest      │
  │  • XGBoost               │
  │  • GNN                   │
  │  • Temporal Model        │
  └──────────────────────────┘
        ↓
  Next-Target Ranking
        ↓
  Dynamic Risk Engine (Probability + Criticality + Exposure)
        ↓
  Explainable AI / SHAP
        ↓
  Defensive Recommendations
        ↓
  FastAPI Backend → React Dashboard
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 15+ (for backend phase)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ai-cyber-attack-prediction-engine

# Create Python virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Download CICIDS2017 dataset
# See docs/dataset_notes.md for instructions
```

### Project Structure

```
├── backend/          # FastAPI backend + APIs
├── ml/               # ML pipeline (preprocessing, models, evaluation)
├── graph/            # Graph construction + features
├── data/             # Raw, processed, and synthetic data
├── frontend/         # React dashboard
├── notebooks/        # Jupyter exploration notebooks
├── models/           # Saved trained models
├── experiments/      # Experiment logs and results
└── docs/             # Documentation
```

## Development Phases

1. **Foundation** — Project setup, topology, dataset acquisition
2. **Preprocessing** — Cleaning, encoding, time windowing, target labels
3. **Graph & Features** — Graph construction, centrality, attack sequences
4. **Baseline Models** — Heuristic, XGBoost, Isolation Forest, risk engine
5. **GNN** — PyTorch Geometric graph neural network
6. **Explainability** — SHAP integration, recommendations
7. **Backend** — FastAPI APIs
8. **Frontend** — React + Cytoscape.js dashboard
9. **Evaluation** — Ablation study, comparison, documentation

## License

This project is developed for academic purposes at VIT as a 3rd-semester Computer Engineering project.

---

> **Note**: This is a defensive cybersecurity research prototype. All experimentation uses public datasets and synthetic networks only. No offensive exploitation is involved.
