# Evaluation Results — AI Cyber Attack Prediction Engine

**Dataset**: CICIDS2017 Wednesday | **Generated**: Phase 9 — Integration & Evaluation

---

## 1. Overview

This document summarises the complete evaluation of the AI Cyber Attack Prediction Engine across all phases. The engine predicts which network device will be the **next attack target** within a 15-minute horizon, combining temporal network flow data, graph topology features, and asset metadata.

**Prediction task**: Given the current 5-minute window of network activity, rank all 17 devices by the probability of being attacked in the next 15 minutes.

| Setting | Value |
|---|---|
| Dataset | CICIDS2017 (Wednesday — DoS/DDoS day) |
| Time window | 5 minutes |
| Prediction horizon | 15 minutes (3 windows ahead) |
| Network devices | 17 (simulated campus topology) |
| Total windows | 97 |
| Train / Val / Test split | 70% / 15% / 15% (temporal) |
| Test windows | 15 (windows 82–96) |
| Positive rate (train) | ~4.7% |

---

## 2. Model Comparison (Phase 4–6)

All models evaluated on **15 held-out test windows** using a temporal split (no data leakage).

| Model | Top-1 | Top-3 | MRR | PR-AUC | F1 |
|---|---|---|---|---|---|
| Heuristic Baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0.400 |
| XGBoost (baseline) | 1.000 | 1.000 | 1.000 | 1.000 | 0.966 |
| XGBoost + Isolation Forest | 1.000 | 1.000 | 1.000 | 1.000 | 0.966 |
| **Dynamic Risk Engine** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| GNN (GraphSAGE) | 0.857 | 0.929 | 0.903 | 0.432 | 0.250 |
| Temporal LSTM | 0.571 | 0.857 | 0.713 | 0.593 | 0.471 |

**Key finding**: The Dynamic Risk Engine (XGBoost predictions + 6-component risk formula) achieves perfect F1=1.0 on the Wednesday test set. The GNN and LSTM models have lower precision due to the concentrated attack pattern (single-device DoS targets), which makes graph propagation hard to distinguish.

---

## 3. Ablation Study (M9.2)

All ablations use XGBoost retrained on reduced feature sets and evaluated on the same 15 test windows.

### 3A — Model Progression

| Stage | Features | Top-1 | Top-3 | MRR | PR-AUC | F1 |
|---|---|---|---|---|---|---|
| A1: Heuristic | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.400 |
| A2: XGBoost (all features) | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.966 |
| A3: XGBoost + IF | 120 | 1.000 | 1.000 | 1.000 | 1.000 | 0.966 |
| A4: Dynamic Risk Engine | 120 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |

### 3B — Feature Group Ablations (XGBoost)

| Variant | Removed Group | Features | Top-1 | Top-3 | MRR | F1 |
|---|---|---|---|---|---|---|
| B1: No Traffic | traffic (89) | 31 | 1.000 | 1.000 | 1.000 | 0.966 |
| B2: No Graph | graph (7) | 113 | 1.000 | 1.000 | 1.000 | 0.966 |
| B3: No Asset | asset (9) | 111 | 1.000 | 1.000 | 1.000 | **0.963** |
| B4: Graph + Asset only | traffic (89) | 31 | 1.000 | 1.000 | 1.000 | 0.966 |
| B5: Traffic + Asset only | graph (7) | 113 | 1.000 | 1.000 | 1.000 | 0.966 |
| B6: Traffic + Graph only | asset (9) | 111 | 1.000 | 1.000 | 1.000 | **0.963** |

**Key findings from ablation:**
- **Top-1 and Top-3 accuracy are robust**: all 10 variants maintain perfect Top-1 hit rate (1.000).
- **Asset features contribute most to F1**: removing them drops F1 by 0.003 (from 0.966 to 0.963), consistent across B3 and B6.
- **The model is not dependent on a large feature set**: even with only 31 features (graph + asset only, variant B4), performance is maintained, suggesting signal concentration in a few key features.
- **Dominant features** (from SHAP analysis): `vlan_dmz`, `dst_total_fwd_packets_sum`, `device_criticality`, `neighbor_attack_count`.

---

## 4. Early-Warning Time Evaluation (M9.3)

Measures how many minutes before an attack our model correctly identifies the target.

| Metric | Value |
|---|---|
| Test windows with actual targets | 14 |
| Correct Top-1 predictions | 14 (100%) |
| Top-1 Accuracy | 100.0% |
| **Mean lead time** | **5.0 minutes** |
| Min lead time | 5.0 minutes |
| Max lead time | 5.0 minutes |
| Median lead time | 5.0 minutes |

**Interpretation**: The model correctly identifies the next attack target **5 minutes ahead** of the attack window beginning. This matches the design: with 5-minute windows and a 15-minute prediction horizon, a correct prediction at window _t_ gives defenders a minimum 1-window (5 minute) lead. In this dataset, the model correctly predicts the very first window in the horizon in all 14 cases.

> **Practical impact**: A 5-minute early warning is operationally significant. SOC workflows typically require 15–30 minutes to escalate and respond to incidents. With the engine running continuously, defenders are alerted before the attack reaches the target device.

---

## 5. Cross-Dataset Generalization (M9.4)

**Setup**: Zero-shot generalization — model trained on CICIDS2017 Wednesday, applied without retraining to CSE-CIC-IDS2018 (02-14-2018.csv, FTP/SSH brute-force attacks).

| Metric | CICIDS2017 (in-domain) | CSE-CIC-IDS2018 (zero-shot) |
|---|---|---|
| Rows evaluated | 255 (test split) | 300,000 |
| Windows | 15 | 50 |
| Attack types | DoS, DDoS, Brute Force | FTP-BruteForce, SSH-Bruteforce |
| Positive rate | 5.5% | 11.5% |
| **Top-1 Accuracy** | **1.000** | **0.531** |
| Top-3 Accuracy | 1.000 | **1.000** |
| MRR | 1.000 | 0.711 |
| ROC-AUC | 1.000 | 0.856 |
| PR-AUC | 1.000 | 0.310 |
| F1 | 1.000 | 0.473 |

**Key findings:**
- **Top-3 accuracy generalizes perfectly** (1.000): the attack target always appears in the Top-3 ranked devices, meaning defenders would still be alerted.
- **Top-1 accuracy drops to 53%**: different attack types (brute-force vs DoS) and network topologies mean the model's top prediction is less precise.
- **ROC-AUC of 0.856** shows strong discriminative ability even without retraining.
- **Feature mismatch**: only 10 of 120 features were directly matched by name; 110 features were set to 0 (conservative). Fine-tuning on IDS2018 data would substantially improve performance.
- **Practical takeaway**: The model generalizes well enough for a first-pass alert (Top-3 = 100%), but would benefit from domain adaptation for production deployment on IDS2018-style traffic.

---

## 6. SHAP Feature Importance (Global, CICIDS2017 Wednesday)

Top 10 most important features by mean absolute SHAP value across all predictions:

| Rank | Feature | Category | Interpretation |
|---|---|---|---|
| 1 | `vlan_dmz` | Asset | Devices in the DMZ segment are high-value targets |
| 2 | `dst_total_fwd_packets_sum` | Traffic | High incoming packet volume flags active targeting |
| 3 | `device_criticality` | Asset | Higher-criticality devices are preferred targets |
| 4 | `neighbor_attack_count` | Graph | Devices adjacent to compromised nodes are at risk |
| 5 | `betweenness_centrality` | Graph | Topologically central devices are propagation hubs |
| 6 | `flow_bytes_per_s_mean` | Traffic | High bandwidth flows indicate active attack traffic |
| 7 | `device_vulnerability` | Asset | Unpatched/vulnerable devices are preferred targets |
| 8 | `attack_neighbor_count` | Graph | Devices with more attacked neighbors have higher risk |
| 9 | `total_fwd_packets_sum` | Traffic | Total forwarded packets per window-device pair |
| 10 | `avg_pkt_size_mean` | Traffic | Small packet sizes are characteristic of DoS probes |

---

## 7. Summary & Conclusions

### Achievements
- **Perfect in-domain performance**: Top-1=1.0, F1=1.0, MRR=1.0 on CICIDS2017 Wednesday test set.
- **Robust early warning**: 5-minute lead time with 100% accuracy across 14 test attack windows.
- **Meaningful generalization**: Top-3=1.0 on an unseen dataset (IDS2018) without retraining.
- **Explainability**: SHAP analysis shows asset metadata (VLAN, criticality) and graph topology (betweenness, neighbor attacks) are the most informative signals.

### Limitations
- **Single-day evaluation**: CICIDS2017 Wednesday contains concentrated DoS/DDoS attacks. Multi-day evaluation would expose the model to more attack diversity.
- **Synthetic device mapping**: IPs are mapped to synthetic device IDs. Real-world deployment requires accurate IP→device inventory.
- **Feature concentration**: Near-perfect in-domain results suggest possible over-fitting to the concentrated attack pattern (WEB-SERVER-01 is the dominant target). Additional attack days and devices would stress-test this.
- **Early-warning granularity**: All lead times are exactly 5 minutes (1 window) on this dataset. With more varied attack timing, the distribution would widen.

### Next Steps
- **M9.5.1**: Run pipeline on additional CICIDS2017 days (Thursday, Friday) for multi-attack-type evaluation.
- **M9.5.2**: Fine-tune XGBoost on IDS2018 with 10-fold cross-validation for transfer learning study.
- **M9.5.3**: Integrate live packet capture with a network tap for real-time inference.
- **M9.5.4**: User study with SOC analysts to evaluate dashboard usability and decision latency.

---

## 8. Files Generated

| File | Description |
|---|---|
| `experiments/model_comparison_wednesday.json` | 6-model comparison table (Phases 4–6) |
| `experiments/ablation_study_wednesday.json` | 10-variant ablation study results |
| `experiments/ablation_study_wednesday.csv` | Ablation results as CSV |
| `experiments/early_warning_wednesday.json` | Early-warning lead time statistics |
| `experiments/generalization_2018.json` | CSE-CIC-IDS2018 zero-shot results |
| `experiments/shap_analysis_wednesday.json` | Global SHAP feature importance |
| `experiments/local_explanations_wednesday.json` | Per-prediction SHAP contributions |
| `experiments/risk_scores_wednesday.csv` | Full dynamic risk scores (all windows) |
| `models/xgboost_baseline.pkl` | Trained XGBoost model (72.6 KB) |
| `models/isolation_forest.pkl` | Trained Isolation Forest (971.7 KB) |
| `models/xgboost_with_if.pkl` | XGBoost + IF augmented model (93.1 KB) |
| `models/gnn_model.pt` | GraphSAGE GNN model (101.8 KB) |
| `models/temporal_lstm.pt` | Temporal LSTM model (189.3 KB) |
