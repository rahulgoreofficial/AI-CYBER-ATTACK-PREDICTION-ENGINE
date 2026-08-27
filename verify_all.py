import json
from pathlib import Path
import pandas as pd
from ml.config import PATHS

print("=== VERIFYING ARTIFACTS ===")

# 1. Check feature matrix
fm_path = PATHS["data_processed"] / "ml_feature_matrix_wednesday.csv"
assert fm_path.exists(), "Feature matrix missing"
df_fm = pd.read_csv(fm_path)
print(f"Feature Matrix: {df_fm.shape} (Rows x Cols)")
exclude_cols = {"window_id", "device_id", "is_future_target", "target_attack_count", "target_attack_types", "earliest_target_window"}
feature_cols = [c for c in df_fm.columns if c not in exclude_cols]
assert df_fm[feature_cols].isna().sum().sum() == 0, f"Found NaNs in {len(feature_cols)} feature columns!"
print(f"Verified: All {len(feature_cols)} feature columns have 0 NaNs.")

# 2. Check models
for m in ["xgboost_baseline.pkl", "isolation_forest.pkl", "xgboost_with_if.pkl", "gnn_model.pt"]:
    p = PATHS["models"] / m
    assert p.exists(), f"Model {m} missing"
    print(f"Model found: {m} ({p.stat().st_size / 1024:.1f} KB)")

# 3. Check experiment outputs
comp_path = PATHS["experiments"] / "model_comparison_wednesday.json"
assert comp_path.exists(), "Model comparison missing"
with open(comp_path) as f:
    comp = json.load(f)

print(f"\nExperiment results logged: {len(comp)} models evaluated")
for c in comp:
    print(f"  - {c['model']}: Top-1={c.get('top_1_hit_rate', 0):.2f}, MRR={c.get('mrr', 0):.2f}, PR-AUC={c.get('pr_auc', 0):.2f}, F1={c.get('f1', 0):.2f}")

print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
