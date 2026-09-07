# 🛡️ AI Cyber Attack Prediction Engine

### *Proactive Threat Anticipation, Graph Intelligence & Dynamic Risk Prioritization*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-eb4224)](https://xgboost.readthedocs.io/)
[![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-3.28-ea580c)](https://js.cytoscape.org/)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20MIT-green)](#license)

> **"Predict where an attack is likely to propagate next and help defenders decide what to protect first."**

The **AI Cyber Attack Prediction Engine** is a proactive cybersecurity decision-support platform designed to shift incident response from reactive alert fatigue to anticipatory defense. By combining **temporal attack propagation analysis**, **network graph topology**, **unsupervised anomaly detection**, **graph neural networks (GNN)**, and **explainable AI (Tree SHAP)**, the system forecasts future lateral movement targets, computes multi-factor dynamic risk scores, and prescribes prioritized MITRE ATT&CK mitigation playbooks in real time.

---

## 🌟 Key Highlights & Innovations

- **Dual-Domain Execution**:
  - **Live Physical Wi-Fi / LAN Network**: Zero-hardcoded Layer-2 active discovery of real connected devices (smartphones, IoT devices, smart TVs, routers, and workstations).
  - **21-Node Campus Benchmark Network**: Evaluated across 15 temporal attack windows using CICIDS2017 and CSE-CIC-IDS2018 datasets.
- **Hardware Layer-2 Discovery Engine**: High-speed parallel Win32 `SendARP` sweeper (128 concurrent workers) bypassing ICMP firewall blocks to identify stealthy endpoints in under 4 seconds.
- **100% Dynamic Network Detection**: Dynamically discovers host machine BIOS, default gateway IP from OS routing tables, and active `/24` subnets, with automatic network-switch handling when transitioning between Wi-Fi, Ethernet, or mobile hotspots.
- **Full-Duplex WebSocket Push**: Real-time event stream (`/ws/network`) streaming `connect`, `disconnect`, `update`, `network_switch`, and `full_scan` events with instant UI synchronization.
- **Multi-Model AI Zoo**: Side-by-side inference with XGBoost, Isolation Forest, Hybrid XGBoost+IF, GraphSAGE GNN, and 2-layer Temporal LSTM.
- **Explainable AI (XAI)**: Live Tree SHAP waterfall feature attributions revealing *why* specific endpoints are prioritized for defense.
- **MITRE ATT&CK Defensive Playbooks**: Context-aware mitigation strategies tailored to discovered vulnerabilities, open ports, and topology centrality.

---

## 🏛️ System Architecture

```
                    [ Live Subnet / Campus Benchmark Traffic ]
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
  [ Active LAN Scanner ]                                [ Preprocessed Dataset ]
  • Win32 SendARP Layer-2                               • CICIDS2017 & 2018 Data
  • Dynamic Route / BIOS Lookup                         • 15 Temporal Windows
  • OUI + Reverse DNS Brand Rec                         • Graph Topology & Assets
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      ▼
                        [ Feature Engineering Pipeline ]
                        • Network Traffic Flow Vectors
                        • Isolation Forest Anomaly Scores
                        • Graph Centrality (Degree, Betweenness)
                        • Temporal Attack Propagation History
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    Multi-Model AI Zoo     │
                        │  • XGBoost Supervised     │
                        │  • Isolation Forest       │
                        │  • Hybrid XGBoost + IF    │
                        │  • GraphSAGE (PyG GNN)    │
                        │  • Temporal LSTM          │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        [ Dynamic Risk Scoring Engine ]
              Risk = 0.40(Prob) + 0.35(Crit) + 0.15(Vuln) + 0.05
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [ Explainable AI (SHAP) ]            [ MITRE ATT&CK Mitigation ]
        • Feature Importance                 • Prioritized Action Plans
        • Waterfall Attributions             • Lateral Path Traversal
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                            [ FastAPI Backend ]
                            • REST API Endpoints
                            • WebSocket /ws/network
                                      │
                                      ▼
                      [ React SOC Dark Glassmorphism UI ]
                      • Interactive Cytoscape Topology
                      • Real-Time Target Ranking & SHAP
                      • Live Connect/Disconnect Toasts
```

---

## 🤖 AI & Machine Learning Model Zoo

The engine evaluates and runs multiple complementary machine learning architectures:

| Model Architecture | Implementation | Purpose & Strength |
| :--- | :--- | :--- |
| **XGBoost Classifier** | `ml/models/xgboost_model.py` | High-accuracy tabular target classification using gradient-boosted decision trees. |
| **Isolation Forest** | `ml/models/isolation_forest_model.py` | Unsupervised anomaly scoring detecting out-of-distribution traffic patterns. |
| **Hybrid XGBoost + IF** | `ml/models/hybrid_model.py` | Combines supervised loss with unsupervised anomaly boost features. |
| **GraphSAGE GNN** | `ml/models/gnn_model.py` | PyTorch Geometric 2-layer Graph Convolution capturing neighborhood topology and edge centrality. |
| **Temporal LSTM** | `ml/models/temporal_model.py` | 2-layer PyTorch sequence model forecasting multi-step lateral movement trajectories over time. |

### Empirical Model Performance Comparison

Evaluated on the benchmark dataset across 15 attack progression windows:

| Model | Top-1 Hit Rate | Top-3 Hit Rate | Top-5 Hit Rate | MRR | PR-AUC | ROC-AUC | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dynamic Risk Engine** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **XGBoost Supervised** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.933 | **1.000** | **0.966** |
| **XGBoost + IF Hybrid** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.933 | **1.000** | **0.966** |
| **GraphSAGE GNN** | 0.857 | 0.929 | 0.929 | 0.903 | 0.432 | 0.938 | **1.000** | 0.143 | 0.250 |
| **Temporal LSTM** | 0.571 | 0.857 | 0.857 | 0.713 | 0.593 | 0.921 | 0.400 | 0.571 | 0.471 |
| **Heuristic Baseline** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 1.000 | 0.400 |

*Note: The Dynamic Risk Engine aggregates model probability with asset criticality, topology exposure, and vulnerability penalty, yielding perfect rank ordering of the primary threat vector.*

---

## ⚡ Real-Time Network Discovery & WebSocket Streaming

### 1. Hardware-Aware Layer-2 Scanning

Traditional tools rely on passive OS ARP caches or sequential ICMP pings. Because mobile devices (Android/iOS) and Windows firewalls frequently drop ICMP echo requests, active endpoints often remain invisible.

- **Win32 `SendARP` Execution**: The engine uses `ctypes.windll.iphlpapi.SendARP` in parallel worker pools to query all 254 subnet addresses at Layer 2 in ~4 seconds.
- **ICMP Firewall Bypass**: Devices connected to Wi-Fi/Ethernet *must* respond to Layer-2 ARP in order to maintain an IP link, guaranteeing discovery of silent smartphones, smart TVs, and IoT hardware.

### 2. Zero Hardcoding Architecture

- **Dynamic Gateway Discovery**: Detects the true default gateway router IP via OS routing table introspection (`route print 0.0.0.0`), whether on home Wi-Fi (`192.168.31.1`), corporate subnets (`10.x.x.x`), or mobile hotspots (`172.20.10.1`).
- **Dynamic BIOS Manufacturer Query**: Queries the true hardware motherboard manufacturer (`HARDWARE\DESCRIPTION\System\BIOS\SystemManufacturer`), automatically reflecting Asus, Dell, Lenovo, HP, or Apple hardware.
- **Network Switch Detection**: Automatically senses subnet transitions when moving between networks, clears stale registries, and broadcasts instant alerts.

### 3. Full-Duplex WebSocket Protocol (`/ws/network`)

Clients connect via WebSocket to receive real-time JSON payloads:

```json
{
  "type": "connect",
  "device": {
    "device_id": "OnePlus Smartphone (192.168.31.172)",
    "ip_address": "192.168.31.172",
    "mac_address": "26-B1-DE-57-E5-D6",
    "manufacturer": "OnePlus",
    "device_type": "mobile",
    "dynamic_risk_score": 0.357,
    "risk_level": "medium",
    "attack_probability": 0.325
  },
  "total_devices": 5,
  "scan_cycle": 28,
  "subnet": "192.168.31.0/24"
}
```

---

## ⚖️ Dynamic Multi-Factor Risk Engine

The engine calculates risk dynamically rather than relying on static CVSS scores:

$$\text{Risk Score} = \left( w_{\text{prob}} \cdot P_{\text{attack}} \right) + \left( w_{\text{crit}} \cdot C_{\text{asset}} \right) + \left( w_{\text{vuln}} \cdot V_{\text{vuln}} \right) + \left( w_{\text{expo}} \cdot E_{\text{topo}} \right) + \left( w_{\text{anom}} \cdot A_{\text{anom}} \right)$$

Default Calibration:

- **Predicted Attack Probability ($w_{\text{prob}} = 0.40$)**: Multi-model inference likelihood.
- **Asset Criticality ($w_{\text{crit}} = 0.35$)**: Host machines (0.95), Gateway Routers (0.90), Servers (0.80), IoT (0.50), Mobile (0.40).
- **Vulnerability Exposure ($w_{\text{vuln}} = 0.15$)**: Base vulnerability boosted by dangerous open ports (445, 3389, 22, 23).
- **Base Offset ($+0.05$)**: Prevents zero-risk false confidence.

| Risk Score Range | Severity Band | UI Visual Indicator |
| :---: | :---: | :---: |
| **0.80 – 1.00** | **Critical** | Pulsing Red Glow (`#ef4444`) |
| **0.60 – 0.79** | **High** | Vivid Orange Glow (`#f97316`) |
| **0.35 – 0.59** | **Medium** | Warning Amber (`#eab308`) |
| **0.00 – 0.34** | **Low** | Secure Emerald Green (`#22c55e`) |

---

## 🔍 Explainable AI (Tree SHAP) & Decision Support

Defenders cannot act on opaque probability numbers. The engine integrates **Tree SHAP (SHapley Additive exPlanations)** to compute exact feature attributions for every prediction:

```
Attack Probability: 32.5% (Base value: 15.0%)
  ┌────────────────────────────────────────────────────────┐
  │ +0.1900 (+30.0%) Wi-Fi Lateral Propagation Exposure   ▲│
  │ +0.1500 (+25.0%) Mobile Device Classification          ▲│
  │ +0.1200 (+18.0%) Dynamic DHCP Lease Allocation         ▲│
  │ -0.1800 (-27.0%) Mobile OS Security Patch Baseline     ▼│
  └────────────────────────────────────────────────────────┘
```

### MITRE ATT&CK Defensive Playbooks

Automatically prescribes prioritized remediation actions:

- **M1038 (Execution Prevention)**: Disable HTTP/SSH admin interfaces on wireless client stations.
- **M1030 (Network Segmentation)**: Enforce Access Point Client Isolation to prevent lateral peer-to-peer scanning.
- **M1041 (Encryption Enforcement)**: Enforce WPA3-Personal with 802.11w Protected Management Frames.
- **M1042 (Account Protection)**: Change default manufacturer credentials and isolate IoT endpoints on dedicated guest VLANs.

---

## 🖥️ Modern Dark SOC Glassmorphism Dashboard

Built with React 18 and Cytoscape.js, featuring a curated dark glassmorphism aesthetic:

- **Interactive Topology Graph**:
  - Distinct SVG geometric shapes for device categories: **Servers** (`round-rectangle`), **Routers** (`diamond`), **Switches** (`hexagon`), **Mobile Devices** (`round-pentagon`), and **IoT** (`round-octagon`).
  - Animated pulsing glow on critical and newly connected nodes.
- **Dual Network Source Switcher**: Toggle instantly between **Live Real LAN** and **21-Node Campus Benchmark**.
- **Predicted Targets List**: Top-K rankings with real-time probability progress bars.
- **Risk Prioritization Matrix**: Sortable data table with anomaly scores, exposure ratings, and vulnerability indices.
- **Subnet Details Modal**: Full peer inspection displaying MAC addresses, hardware vendors, open ports, and live latency.
- **Live Event Toast Notifications**: Automatic toast banners for `🟢 Connected` and `🔴 Disconnected` devices.

---

## 📡 API Reference & Endpoints

Interactive Swagger UI documentation is available at `http://localhost:8000/docs`.

| Endpoint | Method | Params | Description |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | — | Health check, model initialization status, and active device counts |
| `/ws/network` | `WS` | — | Full-duplex WebSocket stream for real-time device change events |
| `/api/network` | `GET` | `source=lan\|campus` | Network topology nodes, edges, and exposure metrics |
| `/api/network/lan-devices` | `GET` | — | List of all discovered physical LAN devices and host telemetry |
| `/api/network/lan-devices/stream-status` | `GET` | — | Status of background scanner (cycle count, latency, subnet) |
| `/api/risk` | `GET` | `source=lan\|campus` | Ranked multi-factor dynamic risk scores across all devices |
| `/api/predictions` | `GET` | `model`, `top_k`, `source` | Top-K predicted attack targets using specified model |
| `/api/explanation` | `GET` | `device_id` | Live Tree SHAP feature attribution waterfall values |
| `/api/recommendations` | `GET` | `device_id` | Prioritized MITRE ATT&CK mitigation recommendations |
| `/api/attack-path` | `GET` | `device_id` | Lateral movement propagation trajectory and bottleneck analysis |
| `/api/timeline` | `GET` | — | Multi-window temporal evolution across attack phases |
| `/api/evaluation` | `GET` | — | Comparative benchmark metrics (AUC, F1, Hit Rate, MRR) |
| `/api/analyze` | `POST` | JSON payload | Ad-hoc traffic vector inference across all models |

---

## 🚀 Quick Start & Installation

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**
- Windows OS recommended for native Win32 `SendARP` Layer-2 scanning (Linux/macOS fallback included)

### 1. Clone the Repository

```bash
git clone https://github.com/rahulgoreofficial/AI-CYBER-ATTACK-PREDICTION-ENGINE.git
cd AI-CYBER-ATTACK-PREDICTION-ENGINE
```

### 2. Backend Setup

```bash
# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate          # Windows PowerShell / CMD
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now live at [http://localhost:8000](http://localhost:8000).  
Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Frontend Setup

```bash
# In a new terminal, navigate to frontend
cd frontend

# Install Node modules
npm install

# Start the Vite development server
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.  
*(To access from any mobile phone on the same Wi-Fi, open `http://<your-host-ip>:5173`)*.

### 4. Cloud / Hugging Face Space Deployment

The project includes a standalone deployment configuration under `hf_space/`:

```bash
cd hf_space
pip install -r requirements.txt
python app.py
```

---

## 📂 Project Directory Structure

```
ai-cyber-attack-prediction-engine/
├── backend/
│   └── app/
│       ├── api/               # API Routers (predictions, risk, network, lan, etc.)
│       ├── models/            # Pydantic v2 validation schemas
│       ├── recommendations/   # MITRE ATT&CK mitigation rule engine
│       ├── services/          # Core services (scanner, lan_service, prediction_service)
│       └── main.py            # FastAPI entry point, lifespan, & WebSocket manager
├── frontend/
│   ├── src/
│   │   ├── components/        # NetworkGraph, RiskTable, PredictionPanel, SHAP, etc.
│   │   ├── pages/             # Dashboard.jsx (Real-time live SOC dashboard)
│   │   ├── services/          # api.js (Axios HTTP client + WebSocket manager)
│   │   └── index.css          # Dark glassmorphism design system
│   ├── package.json
│   └── vite.config.js
├── ml/
│   ├── dataset/               # Data ingestion, cleaning, time windowing
│   ├── features/              # Flow, anomaly, graph, and temporal feature extraction
│   └── models/                # XGBoost, Isolation Forest, GraphSAGE, LSTM implementations
├── models/                    # Serialized trained model weights (.pkl, .pt)
├── experiments/               # Empirical evaluation logs, ablation studies, SHAP CSVs
├── data/                      # Raw, processed, and synthetic network data
├── hf_space/                  # Hugging Face Space standalone deployment bundle
├── run_pipeline.py            # End-to-end training and feature generation pipeline
├── verify_all.py              # System validation & sanity test suite
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation
```

---

## 🔬 Academic & Research Context

This project was developed at the **Vishwakarma Institute of Technology (VIT)** as an engineering design and innovation research project.

- **Primary Benchmark Datasets**:
  - [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity)
  - [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) (AWS Attack Simulation)
- **Problem Statement**: Traditional Intrusion Detection Systems (IDS) trigger thousands of disconnected alerts daily without contextualizing topological progression. This research models cybersecurity defense as a **dynamic graph prediction challenge**, identifying the most probable subsequent attack hops before lateral exploitation occurs.

---

## 📜 License & Ethical Use

Distributed under the **MIT License**. See `LICENSE` for more information.

> **Defensive Cybersecurity Notice**: This project is exclusively developed as a defensive cybersecurity research tool and decision-support system. All experimental validation was conducted on public academic benchmark datasets and authorized local testbed networks. It contains no offensive exploitation capabilities or payload delivery systems.
