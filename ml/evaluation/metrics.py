"""
Evaluation Metrics — Top-K hit rate, MRR, PR-AUC, and early warning.
=====================================================================

Ranking-based evaluation metrics for next-target prediction.
Unlike standard classification, we care about RANKED lists:
    "Did the actual next target appear in the model's Top-K predictions?"

Usage:
    from ml.evaluation.metrics import evaluate_predictions
    results = evaluate_predictions(y_true, y_scores, device_ids, window_ids)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.config import EVAL_CONFIG


def top_k_hit_rate(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    device_ids: np.ndarray,
    window_ids: np.ndarray,
    k: int = 3,
) -> float:
    """
    Compute Top-K hit rate: fraction of windows where at least one actual
    target appears in the model's Top-K ranked predictions.

    For each window:
    1. Rank all devices by predicted score (descending)
    2. Take the top-K devices
    3. Check if ANY actual target is in the top-K

    Args:
        y_true: Binary ground truth (1 = actual future target).
        y_scores: Predicted probabilities/scores.
        device_ids: Device identifiers for each sample.
        window_ids: Window identifiers for each sample.
        k: Number of top predictions to consider.

    Returns:
        Hit rate in [0, 1].
    """
    df = pd.DataFrame({
        "window_id": window_ids,
        "device_id": device_ids,
        "y_true": y_true,
        "y_score": y_scores,
    })

    hits = 0
    windows_with_targets = 0

    for window_id, group in df.groupby("window_id"):
        # Only evaluate windows that have at least one actual target
        if group["y_true"].sum() == 0:
            continue

        windows_with_targets += 1

        # Rank by predicted score, take top-K
        top_k = group.nlargest(k, "y_score")
        if top_k["y_true"].sum() > 0:
            hits += 1

    if windows_with_targets == 0:
        return 0.0

    return hits / windows_with_targets


def mean_reciprocal_rank(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    device_ids: np.ndarray,
    window_ids: np.ndarray,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    For each window with actual targets:
    1. Rank devices by predicted score (descending)
    2. Find the rank of the first actual target
    3. Reciprocal rank = 1 / rank_of_first_target

    MRR = mean of reciprocal ranks across all windows with targets.

    Returns:
        MRR in [0, 1]. Higher is better.
    """
    df = pd.DataFrame({
        "window_id": window_ids,
        "device_id": device_ids,
        "y_true": y_true,
        "y_score": y_scores,
    })

    reciprocal_ranks = []

    for window_id, group in df.groupby("window_id"):
        if group["y_true"].sum() == 0:
            continue

        # Rank by predicted score (descending)
        ranked = group.sort_values("y_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)

        # Find the rank of the first actual target
        target_ranks = ranked[ranked["y_true"] == 1]["rank"]
        if len(target_ranks) > 0:
            first_target_rank = target_ranks.min()
            reciprocal_ranks.append(1.0 / first_target_rank)
        else:
            reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0

    return np.mean(reciprocal_ranks)


def pr_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Precision-Recall AUC (Area Under the PR Curve).

    More informative than ROC-AUC for imbalanced datasets.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_scores)


def evaluate_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    device_ids: np.ndarray,
    window_ids: np.ndarray,
    top_k_values: list[int] | None = None,
    threshold: float = 0.5,
    model_name: str = "Model",
    verbose: bool = True,
) -> dict:
    """
    Comprehensive evaluation of next-target predictions.

    Computes:
    - Top-K hit rates (K=1, 3, 5)
    - MRR (Mean Reciprocal Rank)
    - PR-AUC
    - ROC-AUC
    - Precision, Recall, F1 at threshold

    Args:
        y_true: Binary ground truth.
        y_scores: Predicted probabilities.
        device_ids: Device IDs per sample.
        window_ids: Window IDs per sample.
        top_k_values: K values for Top-K hit rate.
        threshold: Classification threshold for P/R/F1.
        model_name: Name for display.
        verbose: Print results.

    Returns:
        Dictionary with all metrics.
    """
    if top_k_values is None:
        top_k_values = EVAL_CONFIG.get("top_k_values", [1, 3, 5])

    results = {"model": model_name}

    # ── Ranking Metrics ──
    for k in top_k_values:
        hit_rate = top_k_hit_rate(y_true, y_scores, device_ids, window_ids, k=k)
        results[f"top_{k}_hit_rate"] = hit_rate

    results["mrr"] = mean_reciprocal_rank(y_true, y_scores, device_ids, window_ids)

    # ── Global Metrics ──
    results["pr_auc"] = pr_auc_score(y_true, y_scores)

    if len(np.unique(y_true)) >= 2:
        results["roc_auc"] = roc_auc_score(y_true, y_scores)
    else:
        results["roc_auc"] = 0.0

    # ── Threshold-based Metrics ──
    y_pred = (y_scores >= threshold).astype(int)
    results["precision"] = precision_score(y_true, y_pred, zero_division=0)
    results["recall"] = recall_score(y_true, y_pred, zero_division=0)
    results["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # ── Print Results ──
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS — {model_name}")
        print(f"{'='*60}")
        print(f"  Samples:    {len(y_true):,} "
              f"(pos={y_true.sum():.0f}, neg={len(y_true) - y_true.sum():.0f})")

        print(f"\n  --- Ranking Metrics ---")
        for k in top_k_values:
            print(f"  Top-{k} Hit Rate:  {results[f'top_{k}_hit_rate']:.4f}")
        print(f"  MRR:             {results['mrr']:.4f}")

        print(f"\n  --- Global Metrics ---")
        print(f"  PR-AUC:          {results['pr_auc']:.4f}")
        print(f"  ROC-AUC:         {results['roc_auc']:.4f}")

        print(f"\n  --- Threshold={threshold} ---")
        print(f"  Precision:       {results['precision']:.4f}")
        print(f"  Recall:          {results['recall']:.4f}")
        print(f"  F1:              {results['f1']:.4f}")
        print(f"{'='*60}")

    return results


def print_comparison_table(all_results: list[dict]) -> None:
    """
    Print a formatted comparison table of multiple model results.

    Args:
        all_results: List of result dicts from evaluate_predictions().
    """
    if not all_results:
        return

    print(f"\n{'='*90}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*90}")

    # Header
    header = f"{'Model':<25s} | {'Top-1':>6s} | {'Top-3':>6s} | {'Top-5':>6s} | {'MRR':>6s} | {'PR-AUC':>7s} | {'F1':>6s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        row = (
            f"{r.get('model', '?'):<25s} | "
            f"{r.get('top_1_hit_rate', 0):.4f} | "
            f"{r.get('top_3_hit_rate', 0):.4f} | "
            f"{r.get('top_5_hit_rate', 0):.4f} | "
            f"{r.get('mrr', 0):.4f} | "
            f"{r.get('pr_auc', 0):.5f} | "
            f"{r.get('f1', 0):.4f}"
        )
        print(row)

    print(f"{'='*90}")


# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("Testing evaluation metrics with synthetic data...")

    np.random.seed(42)

    # Simulate 5 windows × 10 devices
    n_windows = 5
    n_devices = 10
    device_ids = np.array([f"DEV-{i:02d}" for i in range(n_devices)] * n_windows)
    window_ids = np.repeat(range(n_windows), n_devices)

    # Synthetic ground truth: device 0 is the target in 3/5 windows
    y_true = np.zeros(n_windows * n_devices)
    for w in [0, 2, 4]:
        y_true[w * n_devices + 0] = 1  # DEV-00 is target

    # Synthetic scores: model gives DEV-00 high scores
    y_scores = np.random.uniform(0, 0.3, n_windows * n_devices)
    for w in range(n_windows):
        y_scores[w * n_devices + 0] = np.random.uniform(0.6, 0.9)

    results = evaluate_predictions(
        y_true, y_scores, device_ids, window_ids,
        model_name="Synthetic Test"
    )

    assert results["top_1_hit_rate"] > 0, "Top-1 should be > 0"
    assert results["mrr"] > 0, "MRR should be > 0"
    print("\n[OK] Evaluation metrics test passed.")
