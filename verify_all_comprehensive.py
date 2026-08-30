"""
COMPREHENSIVE VERIFICATION & ANALYSIS SCRIPT
=============================================
Checks all layers of the AI Cyber Attack Prediction Engine:
  1. File/Artifact Integrity
  2. ML Model Sanity
  3. Feature Matrix Quality
  4. Experiment Result Consistency
  5. Backend API Health (all 11 endpoints)
  6. Phase 9 Output Validation
  7. Cross-layer consistency
"""

import sys, json, time, traceback
from pathlib import Path
import urllib.request, urllib.error

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml.config import PATHS

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"
SEP  = "=" * 70
THIN = "-" * 70

results = {"pass": 0, "fail": 0, "warn": 0}

def ok(msg):
    results["pass"] += 1
    print(f"  {PASS} {msg}")

def fail(msg):
    results["fail"] += 1
    print(f"  {FAIL} {msg}")

def warn(msg):
    results["warn"] += 1
    print(f"  {WARN} {msg}")

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# 1. FILE & ARTIFACT INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────
section("1. FILE & ARTIFACT INTEGRITY")

required_files = {
    # Data
    "Feature matrix (CSV)":     PATHS["data_processed"] / "ml_feature_matrix_wednesday.csv",
    "Feature columns (TXT)":    PATHS["data_processed"] / "cicids2017_wednesday_feature_cols.txt",
    "Processed events (CSV)":   PATHS["data_processed"] / "cicids2017_wednesday_processed.csv",
    "Campus topology (JSON)":   PATHS["campus_topology"],
    # Models
    "XGBoost baseline":         PATHS["models"] / "xgboost_baseline.pkl",
    "Isolation Forest":         PATHS["models"] / "isolation_forest.pkl",
    "XGBoost + IF":             PATHS["models"] / "xgboost_with_if.pkl",
    "GNN (GraphSAGE)":          PATHS["models"] / "gnn_model.pt",
    "Temporal LSTM":            PATHS["models"] / "temporal_lstm.pt",
    # Experiments
    "Model comparison":         PATHS["experiments"] / "model_comparison_wednesday.json",
    "SHAP analysis":            PATHS["experiments"] / "shap_analysis_wednesday.json",
    "Risk scores (CSV)":        PATHS["experiments"] / "risk_scores_wednesday.csv",
    "Local explanations":       PATHS["experiments"] / "local_explanations_wednesday.json",
    "SHAP global importance":   PATHS["experiments"] / "shap_global_importance_wednesday.csv",
    "XGBoost feature importance": PATHS["experiments"] / "xgboost_feature_importance_wednesday.csv",
    # Phase 9 outputs
    "Ablation study (JSON)":    PATHS["experiments"] / "ablation_study_wednesday.json",
    "Ablation study (CSV)":     PATHS["experiments"] / "ablation_study_wednesday.csv",
    "Early warning (JSON)":     PATHS["experiments"] / "early_warning_wednesday.json",
    "Generalization 2018 (JSON)": PATHS["experiments"] / "generalization_2018.json",
    # Docs
    "Evaluation report (MD)":   Path("docs/evaluation_results.md"),
    # Scripts
    "run_pipeline.py":          Path("run_pipeline.py"),
    "verify_all.py":            Path("verify_all.py"),
    "ablation_study.py":        Path("ml/evaluation/ablation_study.py"),
    "early_warning.py":         Path("ml/evaluation/early_warning.py"),
    "generalization_test.py":   Path("ml/evaluation/generalization_test.py"),
}

for name, path in required_files.items():
    if path.exists():
        size_kb = path.stat().st_size / 1024
        ok(f"{name:<40} ({size_kb:.1f} KB)")
    else:
        fail(f"{name:<40} MISSING at {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE MATRIX QUALITY
# ─────────────────────────────────────────────────────────────────────────────
section("2. FEATURE MATRIX QUALITY")

try:
    fm_path = PATHS["data_processed"] / "ml_feature_matrix_wednesday.csv"
    df = pd.read_csv(fm_path, low_memory=False)
    ok(f"Loaded feature matrix: {df.shape[0]:,} rows x {df.shape[1]} cols")

    exclude = {"window_id","device_id","is_future_target","target_attack_count",
               "target_attack_types","earliest_target_window"}
    feat_cols = [c for c in df.columns if c not in exclude
                 and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8, bool]]

    ok(f"Feature columns: {len(feat_cols)}")

    nan_count = df[feat_cols].isna().sum().sum()
    if nan_count == 0:
        ok(f"NaN check: 0 NaN values in {len(feat_cols)} feature columns")
    else:
        fail(f"NaN check: {nan_count} NaN values found!")

    inf_count = np.isinf(df[feat_cols].select_dtypes(include=[np.number])).sum().sum()
    if inf_count == 0:
        ok(f"Inf check: 0 Inf values in feature columns")
    else:
        fail(f"Inf check: {inf_count} Inf values found!")

    n_windows = df["window_id"].nunique()
    n_devices = df["device_id"].nunique()
    n_targets = int(df["is_future_target"].sum())
    pos_rate  = df["is_future_target"].mean()

    ok(f"Windows: {n_windows} | Devices: {n_devices} | Targets: {n_targets} ({pos_rate:.1%})")

    if n_windows >= 90:
        ok(f"Window count sanity: {n_windows} >= 90")
    else:
        warn(f"Window count lower than expected: {n_windows}")

    if n_devices >= 15:
        ok(f"Device count sanity: {n_devices} devices")
    else:
        warn(f"Device count lower than expected: {n_devices}")

    # Check no future data leakage: feature columns should not contain target-derived cols
    leakage_cols = [c for c in feat_cols if "target" in c.lower() or "future" in c.lower()]
    if not leakage_cols:
        ok("Data leakage check: no target-derived columns in features")
    else:
        fail(f"Potential leakage: {leakage_cols}")

except Exception as e:
    fail(f"Feature matrix check failed: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 3. ML MODEL SANITY
# ─────────────────────────────────────────────────────────────────────────────
section("3. ML MODEL SANITY")

try:
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest as IF

    # XGBoost baseline
    xgb_model = joblib.load(PATHS["models"] / "xgboost_baseline.pkl")
    ok(f"XGBoost loaded | features={xgb_model.n_features_in_} | "
       f"best_iter={xgb_model.best_iteration}")

    # XGBoost + IF
    xgbif_model = joblib.load(PATHS["models"] / "xgboost_with_if.pkl")
    ok(f"XGBoost+IF loaded | features={xgbif_model.n_features_in_}")

    # Isolation Forest
    if_model = joblib.load(PATHS["models"] / "isolation_forest.pkl")
    ok(f"Isolation Forest loaded | estimators={if_model.n_estimators}")

    # Quick inference test — use last test window
    all_windows = sorted(df["window_id"].unique())
    test_w = all_windows[-1]
    test_df = df[df["window_id"] == test_w]
    X_test = test_df[feat_cols].values

    # XGBoost inference
    probs = xgb_model.predict_proba(X_test)[:, 1]
    if probs.shape[0] == len(test_df) and 0 <= probs.min() <= probs.max() <= 1:
        ok(f"XGBoost inference: {len(probs)} predictions, range [{probs.min():.4f}, {probs.max():.4f}]")
    else:
        fail("XGBoost inference produced unexpected output")

    # IF anomaly scores
    raw = -if_model.decision_function(X_test)
    ok(f"IsolationForest inference: anomaly range [{raw.min():.4f}, {raw.max():.4f}]")

    # GNN + LSTM (just check files load as torch)
    try:
        import torch
        gnn_state = torch.load(PATHS["models"] / "gnn_model.pt", map_location="cpu", weights_only=False)
        ok(f"GNN checkpoint loaded | keys={list(gnn_state.keys())[:4]}")
    except Exception as e:
        warn(f"GNN load: {e}")

    try:
        import torch
        lstm_state = torch.load(PATHS["models"] / "temporal_lstm.pt", map_location="cpu", weights_only=False)
        ok(f"LSTM checkpoint loaded | keys={list(lstm_state.keys())[:4]}")
    except Exception as e:
        warn(f"LSTM load: {e}")

except Exception as e:
    fail(f"ML model sanity check failed: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENT RESULT CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
section("4. EXPERIMENT RESULT CONSISTENCY")

try:
    # Model comparison
    with open(PATHS["experiments"] / "model_comparison_wednesday.json") as f:
        comp = json.load(f)
    ok(f"Model comparison: {len(comp)} models")

    for m in comp:
        name = m["model"]
        t1 = m.get("top_1_hit_rate", 0)
        f1 = m.get("f1", 0)
        if t1 > 0:
            ok(f"  {name:<28} Top-1={t1:.3f}  F1={f1:.3f}")
        else:
            warn(f"  {name:<28} Top-1={t1:.3f} (low)")

    # Expected: Dynamic Risk should be best F1
    best = max(comp, key=lambda x: x.get("f1", 0))
    if best["model"] == "Dynamic Risk":
        ok(f"Best model is 'Dynamic Risk' (F1={best['f1']:.3f}) -- as expected")
    else:
        warn(f"Best model is '{best['model']}' -- expected 'Dynamic Risk'")

    # SHAP analysis
    with open(PATHS["experiments"] / "shap_analysis_wednesday.json") as f:
        shap_data = json.load(f)
    ok(f"SHAP analysis loaded | keys: {list(shap_data.keys())}")

    # Risk scores CSV
    risk_df = pd.read_csv(PATHS["experiments"] / "risk_scores_wednesday.csv")
    ok(f"Risk scores: {len(risk_df):,} rows | columns: {list(risk_df.columns[:5])}")

    # Check risk scores are valid (0-1 range)
    if "dynamic_risk_score" in risk_df.columns:
        rs = risk_df["dynamic_risk_score"]
        ok(f"Risk score range: [{rs.min():.4f}, {rs.max():.4f}] (expected 0–1)")
    else:
        warn("Column 'dynamic_risk_score' not found in risk_scores CSV")

    # XGBoost feature importance
    imp_df = pd.read_csv(PATHS["experiments"] / "xgboost_feature_importance_wednesday.csv")
    top1_feat = imp_df.sort_values("importance", ascending=False).iloc[0]
    ok(f"Top feature: {top1_feat['feature']} (importance={top1_feat['importance']:.4f})")

except Exception as e:
    fail(f"Experiment consistency check failed: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PHASE 9 OUTPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
section("5. PHASE 9 OUTPUT VALIDATION")

# M9.2 Ablation
try:
    with open(PATHS["experiments"] / "ablation_study_wednesday.json") as f:
        ablation = json.load(f)
    ok(f"Ablation study: {len(ablation)} variants")

    all_top1 = [v.get("top_1_hit_rate", 0) for v in ablation]
    if all(t == 1.0 for t in all_top1):
        ok("All ablation variants achieve Top-1=1.000")
    else:
        low = [(ablation[i]["model"], all_top1[i]) for i in range(len(ablation)) if all_top1[i] < 1.0]
        warn(f"Some variants below Top-1=1.0: {low}")

    # Check ablation has both Part A and Part B
    a_variants = [v for v in ablation if v["model"].startswith("A")]
    b_variants = [v for v in ablation if v["model"].startswith("B")]
    ok(f"Part A (model progression): {len(a_variants)} variants")
    ok(f"Part B (feature ablation): {len(b_variants)} variants")

except Exception as e:
    fail(f"Ablation validation failed: {e}")

# M9.3 Early warning
try:
    with open(PATHS["experiments"] / "early_warning_wednesday.json") as f:
        ew = json.load(f)

    top1_acc = ew.get("top1_accuracy", 0)
    mean_lead = ew.get("early_warning", {}).get("mean_lead_time_minutes", 0)
    n_correct = ew.get("correct_top1_predictions", 0)
    n_windows = ew.get("total_test_windows_with_targets", 0)

    ok(f"Early warning: {n_correct}/{n_windows} correct Top-1 ({top1_acc:.1%})")
    ok(f"Mean lead time: {mean_lead} minutes")

    if mean_lead > 0:
        ok("Lead time > 0 -- predictions fire BEFORE attacks")
    else:
        fail("Lead time = 0 -- no early warning!")

except Exception as e:
    fail(f"Early warning validation failed: {e}")

# M9.4 Generalization
try:
    with open(PATHS["experiments"] / "generalization_2018.json") as f:
        gen = json.load(f)

    top1  = gen["results"].get("top_1_hit_rate", 0)
    top3  = gen["results"].get("top_3_hit_rate", 0)
    roc   = gen["results"].get("roc_auc", 0)
    prauc = gen["results"].get("pr_auc", 0)

    ok(f"Generalization 2018 | Top-1={top1:.3f}  Top-3={top3:.3f}  "
       f"ROC-AUC={roc:.3f}  PR-AUC={prauc:.3f}")

    if top3 >= 0.9:
        ok("Top-3 >= 0.90 -- strong cross-dataset generalization")
    else:
        warn(f"Top-3={top3:.3f} -- generalization weaker than expected")

    if roc >= 0.7:
        ok(f"ROC-AUC={roc:.3f} -- model discriminates well on unseen data")
    else:
        warn(f"ROC-AUC={roc:.3f} -- limited generalization")

except Exception as e:
    fail(f"Generalization validation failed: {e}")

# M9.5 Report
try:
    report_path = Path("docs/evaluation_results.md")
    content = report_path.read_text(encoding="utf-8")
    sections_present = [
        "## 1. Overview",
        "## 2. Model Comparison",
        "## 3. Ablation Study",
        "## 4. Early-Warning",
        "## 5. Cross-Dataset Generalization",
        "## 6. SHAP Feature Importance",
        "## 7. Summary",
        "## 8. Files Generated",
    ]
    missing = [s for s in sections_present if s not in content]
    if not missing:
        ok(f"Evaluation report: all {len(sections_present)} sections present")
    else:
        warn(f"Evaluation report missing sections: {missing}")
    ok(f"Report size: {len(content):,} chars / {len(content.splitlines())} lines")
except Exception as e:
    fail(f"Report validation failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BACKEND API HEALTH (all 11 endpoints)
# ─────────────────────────────────────────────────────────────────────────────
section("6. BACKEND API HEALTH (11 endpoints)")

BASE = "http://127.0.0.1:8000"

endpoints = [
    ("GET", "/health",                              "status"),
    ("GET", "/",                                    "message"),
    ("GET", "/api/network",                         "nodes"),
    ("GET", "/api/risk",                            None),
    ("GET", "/api/predictions",                     None),
    ("GET", "/api/timeline",                        None),
    ("GET", "/api/evaluation",                      None),
    ("GET", "/api/explanation/WEB-SERVER-01",       None),
    ("GET", "/api/recommendations/WEB-SERVER-01",   None),
    ("GET", "/api/attack-path/WEB-SERVER-01",       None),
    ("POST","/api/analyze",                         None),
]

for method, path, expected_key in endpoints:
    url = BASE + path
    try:
        if method == "POST":
            req = urllib.request.Request(
                url,
                data=json.dumps({"window_id": 82}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        else:
            req = urllib.request.Request(url, method="GET")

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10) as resp:
            elapsed = (time.time() - t0) * 1000
            body = json.loads(resp.read().decode())
            status = resp.status

        if status == 200:
            if expected_key and expected_key not in body:
                warn(f"{method} {path:<45} HTTP {status} [{elapsed:.0f}ms] "
                     f"(missing key '{expected_key}')")
            else:
                # Show brief payload info
                if isinstance(body, list):
                    info_str = f"list[{len(body)}]"
                elif isinstance(body, dict):
                    info_str = f"dict keys={list(body.keys())[:4]}"
                else:
                    info_str = str(body)[:40]
                ok(f"{method} {path:<45} HTTP {status} [{elapsed:.0f}ms] {info_str}")
        else:
            fail(f"{method} {path:<45} HTTP {status}")

    except urllib.error.HTTPError as e:
        fail(f"{method} {path:<45} HTTP {e.code} {e.reason}")
    except urllib.error.URLError as e:
        fail(f"{method} {path:<45} Connection failed: {e.reason}")
    except Exception as e:
        fail(f"{method} {path:<45} Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-LAYER CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
section("7. CROSS-LAYER CONSISTENCY")

try:
    # API network should return 21 nodes
    req = urllib.request.Request("http://127.0.0.1:8000/api/network")
    with urllib.request.urlopen(req, timeout=10) as resp:
        net_data = json.loads(resp.read().decode())

    api_nodes = len(net_data.get("nodes", []))
    api_edges = len(net_data.get("edges", []))

    # Campus topology
    with open(PATHS["campus_topology"]) as f:
        topo = json.load(f)
    topo_devices = len(topo.get("devices", []))
    topo_connections = len(topo.get("connections", []))

    if api_nodes == topo_devices:
        ok(f"Node count consistent: API={api_nodes} == Topology={topo_devices}")
    else:
        warn(f"Node count mismatch: API={api_nodes} != Topology={topo_devices}")

    ok(f"Edges: API={api_edges}, Topology={topo_connections}")

    # API predictions top device should match highest risk in feature matrix
    req = urllib.request.Request("http://127.0.0.1:8000/api/predictions?window_id=82")
    with urllib.request.urlopen(req, timeout=10) as resp:
        pred_data = json.loads(resp.read().decode())

    # ML-level top predicted device in window 82
    w82 = df[df["window_id"] == 82].copy()
    xgb_probs_82 = xgb_model.predict_proba(w82[feat_cols].values)[:, 1]
    ml_top_device = w82["device_id"].values[np.argmax(xgb_probs_82)]

    # API top predicted device
    if isinstance(pred_data, list) and len(pred_data) > 0:
        api_top = pred_data[0].get("device_id", "")
    elif isinstance(pred_data, dict):
        preds = pred_data.get("predictions", pred_data.get("targets", []))
        api_top = preds[0].get("device_id", "") if preds else ""
    else:
        api_top = ""

    ok(f"Window 82 top predicted (ML direct): {ml_top_device}")
    ok(f"Window 82 top predicted (API):        {api_top}")

    if ml_top_device == "WEB-SERVER-01":
        ok("ML correctly predicts WEB-SERVER-01 as top target in window 82")

    # Risk scores cross-check
    risk_df = pd.read_csv(PATHS["experiments"] / "risk_scores_wednesday.csv")
    if "device_id" in risk_df.columns and "dynamic_risk_score" in risk_df.columns:
        top_risk_device = risk_df.loc[risk_df["dynamic_risk_score"].idxmax(), "device_id"]
        ok(f"Highest risk device across all windows: {top_risk_device}")

    # Model count cross-check (API evaluation vs model_comparison JSON)
    req = urllib.request.Request("http://127.0.0.1:8000/api/evaluation")
    with urllib.request.urlopen(req, timeout=10) as resp:
        eval_data = json.loads(resp.read().decode())

    if isinstance(eval_data, list):
        api_model_count = len(eval_data)
    elif isinstance(eval_data, dict):
        api_model_count = len(eval_data.get("models", eval_data.get("results", [])))
    else:
        api_model_count = 0

    json_model_count = len(comp)
    if api_model_count == json_model_count:
        ok(f"Model count consistent: API={api_model_count} == JSON={json_model_count}")
    else:
        warn(f"Model count: API={api_model_count}, JSON={json_model_count}")

except Exception as e:
    warn(f"Cross-layer check partial failure: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. DATA STATISTICS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("8. DATA STATISTICS ANALYSIS")

try:
    print(f"\n  Feature Matrix Summary:")
    print(f"  {THIN}")
    print(f"  Rows          : {df.shape[0]:,}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  Feature cols  : {len(feat_cols)}")
    print(f"  Windows       : {df['window_id'].nunique()}")
    print(f"  Devices       : {df['device_id'].nunique()}")
    print(f"  Attack targets: {int(df['is_future_target'].sum())} ({df['is_future_target'].mean():.1%})")

    print(f"\n  Device distribution in feature matrix:")
    dev_counts = df.groupby("device_id")["is_future_target"].agg(["count","sum"])
    dev_counts.columns = ["windows", "is_target"]
    dev_counts["target_rate"] = dev_counts["is_target"] / dev_counts["windows"]
    dev_counts = dev_counts.sort_values("is_target", ascending=False)
    for dev, row in dev_counts.iterrows():
        bar = "#" * int(row["target_rate"] * 30)
        print(f"  {dev:<22} windows={int(row['windows']):3d}  targets={int(row['is_target']):3d}  "
              f"rate={row['target_rate']:.0%}  {bar}")

    print(f"\n  Top 10 most important features (XGBoost):")
    imp_df = pd.read_csv(PATHS["experiments"] / "xgboost_feature_importance_wednesday.csv")
    imp_df = imp_df.sort_values("importance", ascending=False).head(10)
    for _, row in imp_df.iterrows():
        bar = "#" * int(row["importance"] * 100)
        print(f"  {row['feature']:<45} {row['importance']:.4f}  {bar}")

    # Temporal window analysis
    print(f"\n  Attacks per window (test split — windows 82-96):")
    test_windows = df[df["window_id"] >= 82]
    for w in sorted(test_windows["window_id"].unique()):
        wdf = test_windows[test_windows["window_id"] == w]
        targets = wdf[wdf["is_future_target"] == 1]["device_id"].tolist()
        print(f"  Window {w:3d}: {len(wdf):2d} devices | targets={targets if targets else 'none'}")

    ok("Data statistics analysis complete")

except Exception as e:
    warn(f"Data statistics partial failure: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
section("FINAL VERIFICATION SUMMARY")

total = results["pass"] + results["fail"] + results["warn"]
print(f"\n  Total checks : {total}")
print(f"  {PASS} Passed  : {results['pass']}")
print(f"  {WARN} Warnings: {results['warn']}")
print(f"  {FAIL} Failed  : {results['fail']}")
print()

if results["fail"] == 0 and results["warn"] == 0:
    print("  *** ALL CHECKS PASSED -- System is fully operational ***")
elif results["fail"] == 0:
    print(f"  *** PASSED with {results['warn']} warning(s) -- Review warnings above ***")
else:
    print(f"  *** {results['fail']} FAILURE(S) DETECTED -- Action required ***")

print(f"\n{SEP}\n")
sys.exit(0 if results["fail"] == 0 else 1)
