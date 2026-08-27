"""
GNN Training & Evaluation — M5.3
===================================

Complete training pipeline for the GraphSAGE next-target prediction model.

Workflow:
1. Build PyG dataset (reuses dataset.py)
2. Train with BCEWithLogitsLoss + class imbalance weighting
3. Early stopping on validation loss
4. Evaluate on test set with same metrics as Phase 4 (Top-K, MRR, PR-AUC, F1)
5. Save model + update comparison table
6. Print unified model comparison: Heuristic vs XGBoost vs GNN

Usage:
    python -m ml.gnn.train --day wednesday
    python -m ml.gnn.train --day wednesday --epochs 200 --lr 0.001
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ml.gnn.dataset import build_pyg_dataset
from ml.gnn.model import create_model
from ml.config import PATHS, MODEL_CONFIG, RANDOM_SEED
from ml.evaluation.metrics import evaluate_predictions, print_comparison_table


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataset: list,
    train_indices: list[int],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train for one epoch over all training graphs.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for idx in train_indices:
        data = dataset[idx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        optimizer.zero_grad()
        logits = model(x, edge_index).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.shape[0]
        total_samples += y.shape[0]

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataset: list,
    indices: list[int],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on a set of graphs.

    Returns:
        (avg_loss, all_y_true, all_y_scores, all_device_ids, all_window_ids)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_y_true = []
    all_y_scores = []
    all_device_ids = []
    all_window_ids = []

    for idx in indices:
        data = dataset[idx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        logits = model(x, edge_index).squeeze(-1)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.shape[0]
        total_samples += y.shape[0]

        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.cpu().numpy()

        all_y_true.append(y_np)
        all_y_scores.append(probs)
        all_device_ids.extend(data.device_ids)
        all_window_ids.extend([data.window_id] * len(data.device_ids))

    avg_loss = total_loss / max(total_samples, 1)
    all_y_true = np.concatenate(all_y_true)
    all_y_scores = np.concatenate(all_y_scores)
    all_device_ids = np.array(all_device_ids)
    all_window_ids = np.array(all_window_ids)

    return avg_loss, all_y_true, all_y_scores, all_device_ids, all_window_ids


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

def train_and_evaluate(
    day: str = "wednesday",
    epochs: int | None = None,
    lr: float | None = None,
    hidden_channels: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    patience: int | None = None,
    save_model: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full Phase 5 pipeline: train GNN and evaluate against Phase 4 baselines.

    Args:
        day: CICIDS2017 day identifier.
        epochs: Training epochs (default from config).
        lr: Learning rate (default from config).
        hidden_channels: Hidden dimension (default from config).
        num_layers: Number of GNN layers (default from config).
        dropout: Dropout rate (default from config).
        patience: Early stopping patience (default from config).
        save_model: Whether to save the trained model.
        verbose: Print progress.

    Returns:
        Dictionary with training history and evaluation results.
    """
    set_seed()

    gnn_config = MODEL_CONFIG["gnn"]
    epochs = epochs or gnn_config.get("epochs", 100)
    lr = lr or gnn_config.get("learning_rate", 0.001)
    hidden_channels = hidden_channels or gnn_config.get("hidden_channels", 64)
    num_layers = num_layers or gnn_config.get("num_layers", 2)
    dropout = dropout or gnn_config.get("dropout", 0.3)
    patience = patience or gnn_config.get("patience", 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"PHASE 5 — GNN TRAINING ({day.upper()})")
    print(f"{'='*70}")
    print(f"  Device:          {device}")
    print(f"  Epochs:          {epochs}")
    print(f"  Learning Rate:   {lr}")
    print(f"  Hidden Channels: {hidden_channels}")
    print(f"  Num Layers:      {num_layers}")
    print(f"  Dropout:         {dropout}")
    print(f"  Patience:        {patience}")

    # ── Build dataset ──
    dataset, splits = build_pyg_dataset(day=day, verbose=verbose)

    if len(dataset) == 0:
        print("[ERROR] No data objects created. Aborting.")
        return {"error": "empty_dataset"}

    in_channels = dataset[0].x.shape[1]
    print(f"\n[gnn] Input features: {in_channels}")

    # ── Create model ──
    model = create_model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(device)

    # ── Loss function with class imbalance handling ──
    # Compute pos_weight from training data
    train_pos = sum(dataset[i].y.sum().item() for i in splits["train"])
    train_neg = sum((dataset[i].y.shape[0] - dataset[i].y.sum()).item() for i in splits["train"])
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32).to(device)
    print(f"[gnn] pos_weight: {pos_weight.item():.2f} (neg/pos ratio in train)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ── Training loop ──
    print(f"\n{'-'*50}")
    print("Training...")
    print(f"{'-'*50}")

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, dataset, splits["train"], optimizer, criterion, device
        )
        val_loss, _, _, _, _ = evaluate_epoch(
            model, dataset, splits["val"], criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Best: {best_epoch}")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch}, val_loss: {best_val_loss:.6f})")
            break

    # ── Load best model ──
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\n[gnn] Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.6f}")

    # ── Evaluate on test set ──
    print(f"\n{'-'*50}")
    print("Evaluating on test set...")
    print(f"{'-'*50}")

    test_loss, y_true, y_scores, device_ids, window_ids = evaluate_epoch(
        model, dataset, splits["test"], criterion, device
    )

    gnn_results = evaluate_predictions(
        y_true, y_scores,
        device_ids, window_ids,
        model_name="GNN (GraphSAGE)",
        verbose=verbose,
    )

    # ── Save model ──
    if save_model:
        model_path = PATHS["models"] / "gnn_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history,
        }, model_path)
        print(f"[gnn] Model saved: {model_path}")

    # ── Load Phase 4 results and build comparison ──
    comp_path = PATHS["experiments"] / f"model_comparison_{day}.json"
    all_results = []
    if comp_path.exists():
        with open(comp_path) as f:
            all_results = json.load(f)
        print(f"[gnn] Loaded {len(all_results)} previous model results")

    # Remove any existing GNN entry and add new one
    all_results = [r for r in all_results if "GNN" not in r.get("model", "")]
    all_results.append(gnn_results)

    # Save updated comparison
    if save_model:
        with open(comp_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[gnn] Updated comparison: {comp_path}")

    # ── Print comparison table ──
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # ── Sample predictions ──
    if verbose and len(splits["test"]) > 0:
        print(f"\n  --- Sample Test Predictions ---")
        sample_idx = splits["test"][0]
        sample_data = dataset[sample_idx]
        sample_probs = model.predict_proba(
            sample_data.x.to(device),
            sample_data.edge_index.to(device),
        ).cpu().numpy()

        # Rank devices
        ranked_idx = np.argsort(sample_probs)[::-1]
        print(f"  Window {sample_data.window_id}:")
        for rank, idx in enumerate(ranked_idx[:5]):
            dev = sample_data.device_ids[idx]
            prob = sample_probs[idx]
            actual = " [TARGET]" if sample_data.y[idx].item() == 1 else ""
            print(f"    Rank {rank+1}: {dev:<20s} prob={prob:.4f} {actual}")

    return {
        "gnn_results": gnn_results,
        "all_results": all_results,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 — Train and evaluate GNN (GraphSAGE) model"
    )
    parser.add_argument("--day", default="wednesday", help="CICIDS2017 day")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=None, help="Hidden channels")
    parser.add_argument("--layers", type=int, default=None, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    args = parser.parse_args()

    results = train_and_evaluate(
        day=args.day,
        epochs=args.epochs,
        lr=args.lr,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        patience=args.patience,
        save_model=not args.no_save,
    )
