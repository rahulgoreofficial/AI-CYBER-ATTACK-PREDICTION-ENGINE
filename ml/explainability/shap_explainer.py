"""
SHAP Explainability Module — M6.1
====================================

Provides prediction-level and global explainability for the XGBoost
next-target prediction model using SHAP (SHapley Additive exPlanations).

Key capabilities:
    1. Per-device, per-window explanations (top-N features driving the prediction)
    2. Global feature importance ranking (aggregated SHAP values)
    3. Direction indicators (feature increases or decreases attack risk)
    4. Cached computation for fast API serving

Uses ``shap.TreeExplainer`` — exact SHAP values for tree-based models
(no sampling/approximation needed).

Usage:
    from ml.explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer.from_saved_model(day="wednesday")
    explanation = explainer.explain_prediction("WEB-SERVER-01", window_id=85)

    # Or CLI:
    python -m ml.explainability.shap_explainer --day wednesday
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP is required for the explainability module.\n"
        "Install: pip install shap>=0.43.0"
    )

from ml.config import PATHS, RANDOM_SEED


# ==============================================================================
# SHAP EXPLAINER CLASS
# ==============================================================================

class SHAPExplainer:
    """
    SHAP-based explainability for the XGBoost next-target prediction model.

    Wraps ``shap.TreeExplainer`` to provide:
    - Per-prediction explanations (which features matter for a specific device)
    - Global feature importance (which features matter overall)
    - Human-readable direction indicators

    Args:
        model: Trained XGBoost classifier.
        feature_matrix: Full feature matrix DataFrame.
        feature_cols: List of feature column names.
    """

    def __init__(
        self,
        model,
        feature_matrix: pd.DataFrame,
        feature_cols: list[str],
    ):
        self.model = model
        self.feature_matrix = feature_matrix
        self.feature_cols = feature_cols

        # Build SHAP TreeExplainer (exact for tree models)
        print("[shap] Initializing TreeExplainer...")
        self.explainer = shap.TreeExplainer(model)

        # Cache for computed SHAP values
        self._shap_values_cache = None
        self._shap_base_value = None

    @classmethod
    def from_saved_model(
        cls,
        day: str = "wednesday",
        model_name: str = "xgboost_baseline",
    ) -> "SHAPExplainer":
        """
        Create a SHAPExplainer from saved model and feature matrix.

        Args:
            day: CICIDS2017 day identifier.
            model_name: Which XGBoost model to load
                        ('xgboost_baseline' or 'xgboost_with_if').

        Returns:
            Initialized SHAPExplainer.
        """
        # Load model
        model_path = PATHS["models"] / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run Phase 4 first: python -m ml.xgboost_model.train --day {day}"
            )
        model = joblib.load(model_path)
        print(f"[shap] Loaded model: {model_path.name}")

        # Load feature matrix
        fm_path = PATHS["data_processed"] / f"ml_feature_matrix_{day}.csv"
        if not fm_path.exists():
            raise FileNotFoundError(
                f"Feature matrix not found: {fm_path}\n"
                f"Run: python -m ml.feature_engineering.feature_combiner --day {day}"
            )
        feature_matrix = pd.read_csv(fm_path, low_memory=False)
        print(f"[shap] Loaded feature matrix: {feature_matrix.shape[0]:,} rows × "
              f"{feature_matrix.shape[1]} cols")

        # Identify feature columns (same logic as xgboost_model/train.py)
        exclude_cols = {
            "window_id", "device_id",
            "is_future_target", "target_attack_count",
            "target_attack_types", "earliest_target_window",
        }
        feature_cols = [
            c for c in feature_matrix.columns
            if c not in exclude_cols
            and feature_matrix[c].dtype in [
                np.float64, np.int64, np.float32, np.int32, np.uint8, bool
            ]
        ]
        print(f"[shap] Feature columns: {len(feature_cols)}")

        return cls(model, feature_matrix, feature_cols)

    # ==========================================================================
    # COMPUTE SHAP VALUES
    # ==========================================================================

    def compute_shap_values(
        self,
        X: np.ndarray | None = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Compute SHAP values for the given feature matrix.

        If X is None, computes for the entire feature matrix (and caches).

        Returns:
            (shap_values, base_value) where:
            - shap_values: [n_samples, n_features] array
            - base_value: Expected model output (base prediction)
        """
        if X is None and use_cache and self._shap_values_cache is not None:
            return self._shap_values_cache, self._shap_base_value

        if X is None:
            X = self.feature_matrix[self.feature_cols].values

        print(f"[shap] Computing SHAP values for {X.shape[0]:,} samples...")
        shap_values = self.explainer.shap_values(X)

        # TreeExplainer.shap_values() for binary classification may return
        # a list [class_0_shap, class_1_shap] or a single array.
        # We want class 1 (attack probability) SHAP values.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (positive = attack target)

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])  # Class 1 base value
        else:
            base_value = float(base_value)

        # Cache if full matrix
        if X is None or X.shape[0] == len(self.feature_matrix):
            self._shap_values_cache = shap_values
            self._shap_base_value = base_value

        print(f"[shap] SHAP values shape: {shap_values.shape}, "
              f"base_value: {base_value:.6f}")

        return shap_values, base_value

    # ==========================================================================
    # PER-PREDICTION EXPLANATION
    # ==========================================================================

    def explain_prediction(
        self,
        device_id: str,
        window_id: int,
        top_n: int = 10,
    ) -> dict:
        """
        Explain a single prediction: why was this device flagged (or not)?

        Args:
            device_id: Device identifier (e.g., 'WEB-SERVER-01').
            window_id: Time window identifier.
            top_n: Number of top contributing features to return.

        Returns:
            Dictionary with:
            - device_id, window_id
            - attack_probability (model output)
            - base_value (expected model output before features)
            - features: [{name, value, shap_value, direction, contribution_pct}, ...]
              sorted by |shap_value| descending
        """
        # Find the row in the feature matrix
        mask = (
            (self.feature_matrix["device_id"] == device_id) &
            (self.feature_matrix["window_id"] == window_id)
        )
        matching = self.feature_matrix[mask]

        if len(matching) == 0:
            return {
                "device_id": device_id,
                "window_id": window_id,
                "error": f"No data found for device '{device_id}' in window {window_id}",
                "features": [],
            }

        row_idx = matching.index[0]
        X_single = matching[self.feature_cols].values

        # Get model prediction
        try:
            attack_prob = float(self.model.predict_proba(X_single)[0, 1])
        except Exception:
            attack_prob = float(self.model.predict_proba(X_single)[0])

        # Compute SHAP values for this single sample
        shap_vals_single = self.explainer.shap_values(X_single)
        if isinstance(shap_vals_single, list):
            shap_vals_single = shap_vals_single[1]
        shap_vals_single = shap_vals_single.flatten()

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])
        else:
            base_value = float(base_value)

        # Build feature explanations
        feature_values = X_single.flatten()
        abs_shap = np.abs(shap_vals_single)
        total_abs_shap = abs_shap.sum()

        # Sort by absolute SHAP value (most important first)
        sorted_idx = np.argsort(abs_shap)[::-1][:top_n]

        features = []
        for idx in sorted_idx:
            sv = float(shap_vals_single[idx])
            fv = float(feature_values[idx])
            contribution_pct = (abs_shap[idx] / total_abs_shap * 100) if total_abs_shap > 0 else 0

            features.append({
                "name": self.feature_cols[idx],
                "value": round(fv, 6),
                "shap_value": round(sv, 6),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
                "contribution_pct": round(float(contribution_pct), 2),
            })

        return {
            "device_id": device_id,
            "window_id": int(window_id),
            "attack_probability": round(attack_prob, 6),
            "base_value": round(base_value, 6),
            "top_features": features,
        }

    # ==========================================================================
    # GLOBAL FEATURE IMPORTANCE
    # ==========================================================================

    def global_feature_importance(
        self,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Compute global feature importance: mean |SHAP value| across all samples.

        Returns:
            DataFrame with columns: feature, mean_abs_shap, rank
        """
        shap_values, _ = self.compute_shap_values()

        mean_abs = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        imp_df["rank"] = range(1, len(imp_df) + 1)

        print(f"\n  --- Global SHAP Feature Importance (Top {top_n}) ---")
        for _, row in imp_df.head(top_n).iterrows():
            bar = "#" * int(row["mean_abs_shap"] / imp_df["mean_abs_shap"].max() * 40)
            print(f"  {row['rank']:3d}. {row['feature']:<40s} "
                  f"{row['mean_abs_shap']:.6f}  {bar}")

        return imp_df

    # ==========================================================================
    # BATCH EXPLANATIONS (for API precomputation)
    # ==========================================================================

    def explain_all_predictions(
        self,
        window_ids: list[int] | None = None,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Generate explanations for all device-window pairs (or a subset).

        Useful for pre-computing explanations to serve from the API.

        Args:
            window_ids: Specific windows to explain (None = all).
            top_n: Top features per prediction.

        Returns:
            List of explanation dicts.
        """
        fm = self.feature_matrix
        if window_ids is not None:
            fm = fm[fm["window_id"].isin(window_ids)]

        # Compute SHAP values in batch (much faster than per-sample)
        X = fm[self.feature_cols].values
        shap_values, base_value = self.compute_shap_values(X, use_cache=False)

        # Get model predictions
        try:
            probs = self.model.predict_proba(X)[:, 1]
        except Exception:
            probs = self.model.predict_proba(X)

        explanations = []
        for i in range(len(fm)):
            row = fm.iloc[i]
            sv = shap_values[i]
            abs_sv = np.abs(sv)
            total_abs = abs_sv.sum()
            sorted_idx = np.argsort(abs_sv)[::-1][:top_n]

            features = []
            for idx in sorted_idx:
                features.append({
                    "name": self.feature_cols[idx],
                    "value": round(float(X[i, idx]), 6),
                    "shap_value": round(float(sv[idx]), 6),
                    "direction": "increases_risk" if sv[idx] > 0 else "decreases_risk",
                    "contribution_pct": round(
                        float(abs_sv[idx] / total_abs * 100) if total_abs > 0 else 0, 2
                    ),
                })

            explanations.append({
                "device_id": row["device_id"],
                "window_id": int(row["window_id"]),
                "attack_probability": round(float(probs[i]), 6),
                "base_value": round(base_value, 6),
                "top_features": features,
            })

        print(f"[shap] Generated {len(explanations)} explanations")
        return explanations

    # ==========================================================================
    # SAVE ANALYSIS
    # ==========================================================================

    def save_analysis(
        self,
        day: str = "wednesday",
        top_n_global: int = 20,
        sample_explanations: int = 5,
    ) -> Path:
        """
        Run full SHAP analysis and save results to experiments/.

        Saves:
        - Global feature importance
        - Sample per-device explanations
        - Summary statistics

        Returns:
            Path to saved JSON file.
        """
        print(f"\n{'='*60}")
        print(f"SHAP ANALYSIS — {day.upper()}")
        print(f"{'='*60}")

        # Global importance
        global_imp = self.global_feature_importance(top_n=top_n_global)

        # Sample explanations (pick interesting devices)
        # Find devices with highest attack probability
        fm = self.feature_matrix
        if "is_future_target" in fm.columns:
            target_devices = fm[fm["is_future_target"] == 1]
            if len(target_devices) > 0:
                sample_rows = target_devices.head(sample_explanations)
            else:
                sample_rows = fm.head(sample_explanations)
        else:
            sample_rows = fm.head(sample_explanations)

        sample_explanations_list = []
        for _, row in sample_rows.iterrows():
            expl = self.explain_prediction(
                row["device_id"], row["window_id"], top_n=10
            )
            sample_explanations_list.append(expl)

            # Print a sample
            if len(sample_explanations_list) <= 3:
                print(f"\n  --- Explanation: {expl['device_id']} "
                      f"(window {expl['window_id']}) ---")
                print(f"  Attack Probability: {expl['attack_probability']:.4f}")
                for f in expl["top_features"][:5]:
                    arrow = "[+]" if f["direction"] == "increases_risk" else "[-]"
                    print(f"    {arrow} {f['name']:<35s} "
                          f"val={f['value']:.4f}  "
                          f"SHAP={f['shap_value']:+.4f}  "
                          f"({f['contribution_pct']:.1f}%)")

        # Build output
        analysis = {
            "day": day,
            "num_samples": len(fm),
            "num_features": len(self.feature_cols),
            "global_feature_importance": [
                {
                    "rank": int(row["rank"]),
                    "feature": row["feature"],
                    "mean_abs_shap": round(float(row["mean_abs_shap"]), 8),
                }
                for _, row in global_imp.head(top_n_global).iterrows()
            ],
            "sample_explanations": sample_explanations_list,
        }

        # Save
        out_path = PATHS["experiments"] / f"shap_analysis_{day}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n[shap] Analysis saved: {out_path}")

        # Also save global importance as CSV
        csv_path = PATHS["experiments"] / f"shap_global_importance_{day}.csv"
        global_imp.to_csv(csv_path, index=False)
        print(f"[shap] Global importance CSV: {csv_path}")

        return out_path


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M6.1 — SHAP Explainability Analysis"
    )
    parser.add_argument("--day", default="wednesday", help="CICIDS2017 day")
    parser.add_argument(
        "--model", default="xgboost_baseline",
        help="Model to explain: 'xgboost_baseline' or 'xgboost_with_if'"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top features for global importance"
    )
    args = parser.parse_args()

    # Build explainer
    explainer = SHAPExplainer.from_saved_model(
        day=args.day, model_name=args.model
    )

    # Run full analysis
    explainer.save_analysis(day=args.day, top_n_global=args.top_n)

    print("\n[OK] M6.1 — SHAP explainability complete.")
