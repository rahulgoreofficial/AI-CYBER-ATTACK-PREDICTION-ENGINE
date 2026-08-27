"""
Temporal LSTM Model — M6.3
============================

Per-device temporal sequence model for next-target prediction.

Instead of treating each (device, window) independently (like XGBoost),
this model captures how a device's feature profile *evolves* over
consecutive time windows using an LSTM.

Architecture:
    For each device, build a sequence of feature vectors across
    ``lookback`` consecutive windows:
        [x_{t-k}, x_{t-k+1}, ..., x_t] → LSTM → Linear → Sigmoid

    The LSTM learns temporal patterns (e.g., gradually increasing
    traffic anomalies before an attack) that per-window models miss.

Note:
    CICIDS2017 Wednesday has mostly single-hop DoS attacks (0 propagation
    chains in M3.3), so temporal ordering may not add much signal.
    This module will benchmark against baselines and report findings.

Usage:
    python -m ml.temporal.temporal_lstm --day wednesday
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
from torch.utils.data import Dataset, DataLoader

from ml.config import PATHS, MODEL_CONFIG, TIME_CONFIG, RANDOM_SEED
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
# SEQUENCE DATASET
# ==============================================================================

class TemporalSequenceDataset(Dataset):
    """
    Per-device temporal sequence dataset.

    For each device, builds sequences of ``lookback`` consecutive window
    feature vectors. The target is whether the device becomes an attack
    target in the window immediately after the sequence.

    Args:
        sequences: List of (X_seq, y) tuples where X_seq is [lookback, n_features].
        device_ids: Corresponding device IDs.
        window_ids: Corresponding (final) window IDs.
    """

    def __init__(
        self,
        sequences: list[tuple[np.ndarray, float]],
        device_ids: list[str],
        window_ids: list[int],
    ):
        self.sequences = sequences
        self.device_ids = device_ids
        self.window_ids = window_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x_seq, y = self.sequences[idx]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def build_temporal_sequences(
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    lookback: int = 3,
    verbose: bool = True,
) -> tuple[list[tuple[np.ndarray, float]], list[str], list[int]]:
    """
    Build temporal sequences from the feature matrix.

    For each device and each window W (where the device has data in at least
    ``lookback`` preceding windows), create:
        X_seq = [features at W-lookback+1, ..., features at W]
        y = is_future_target at window W

    Args:
        feature_matrix: Full feature matrix DataFrame.
        feature_cols: List of feature column names.
        lookback: Number of consecutive windows per sequence.
        verbose: Print progress.

    Returns:
        (sequences, device_ids, window_ids)
    """
    fm = feature_matrix.copy()
    all_windows = sorted(fm["window_id"].unique())
    all_devices = sorted(fm["device_id"].unique())
    n_features = len(feature_cols)

    if verbose:
        print(f"[temporal] Building sequences: {len(all_devices)} devices, "
              f"{len(all_windows)} windows, lookback={lookback}")

    # Index feature matrix for fast lookup
    # Build a dict: (device_id, window_id) -> feature_vector, target
    fm_index = {}
    for _, row in fm.iterrows():
        key = (row["device_id"], row["window_id"])
        features = row[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        target = float(row.get("is_future_target", 0))
        fm_index[key] = (features, target)

    sequences = []
    device_ids = []
    window_ids = []

    for device in all_devices:
        # Get windows where this device has data
        device_windows = sorted([
            w for w in all_windows if (device, w) in fm_index
        ])

        if len(device_windows) < lookback:
            continue

        for i in range(lookback - 1, len(device_windows)):
            # Check if we have consecutive windows
            seq_windows = device_windows[i - lookback + 1: i + 1]

            # Build sequence
            seq = np.zeros((lookback, n_features), dtype=np.float32)
            valid = True
            for j, w in enumerate(seq_windows):
                if (device, w) in fm_index:
                    seq[j] = fm_index[(device, w)][0]
                else:
                    valid = False
                    break

            if not valid:
                continue

            # Target: is the device a future target at the last window?
            _, target = fm_index[(device, seq_windows[-1])]

            sequences.append((seq, target))
            device_ids.append(device)
            window_ids.append(seq_windows[-1])

    if verbose:
        n_pos = sum(1 for _, y in sequences if y == 1)
        print(f"[temporal] Built {len(sequences)} sequences "
              f"({n_pos} positive, {len(sequences) - n_pos} negative)")

    return sequences, device_ids, window_ids


# ==============================================================================
# TEMPORAL LSTM MODEL
# ==============================================================================

class TemporalLSTM(nn.Module):
    """
    LSTM-based temporal model for next-target prediction.

    Takes a sequence of per-device feature vectors across consecutive
    time windows and predicts whether the device will be attacked next.

    Architecture:
        Input [batch, lookback, n_features]
          → LSTM(n_features, hidden_dim) [1 layer]
          → Last hidden state [batch, hidden_dim]
          → Linear(hidden_dim, 1)
          → Sigmoid (during inference)

    Args:
        input_dim: Number of features per time step.
        hidden_dim: LSTM hidden dimension (default: 64).
        num_layers: Number of LSTM layers (default: 1).
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

        # Initialize
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch, lookback, input_dim].

        Returns:
            logits: [batch, 1]
        """
        # LSTM forward: output is [batch, seq_len, hidden_dim]
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        last_hidden = self.dropout(last_hidden)

        logits = self.classifier(last_hidden)  # [batch, 1]
        return logits

    def predict_proba(self, x):
        """Get probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs


# ==============================================================================
# TRAINING
# ==============================================================================

def train_and_evaluate(
    day: str = "wednesday",
    lookback: int | None = None,
    hidden_dim: int = 64,
    epochs: int = 80,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 10,
    save_model: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full M6.3 pipeline: build sequences, train LSTM, evaluate, compare.

    Args:
        day: CICIDS2017 day identifier.
        lookback: Sequence length (default: from TIME_CONFIG).
        hidden_dim: LSTM hidden dimension.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Training batch size.
        patience: Early stopping patience.
        save_model: Whether to save the model.
        verbose: Print progress.

    Returns:
        Dictionary with results.
    """
    set_seed()

    if lookback is None:
        lookback = TIME_CONFIG.get("lookback_windows", 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"PHASE 6.3 — TEMPORAL LSTM ({day.upper()})")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Lookback:   {lookback} windows")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Epochs:     {epochs}")
    print(f"  LR:         {lr}")
    print(f"  Batch Size: {batch_size}")

    # ── Load feature matrix ──
    fm_path = PATHS["data_processed"] / f"ml_feature_matrix_{day}.csv"
    if not fm_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {fm_path}")

    feature_matrix = pd.read_csv(fm_path, low_memory=False)
    print(f"[temporal] Loaded feature matrix: {feature_matrix.shape}")

    # Feature columns
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
    n_features = len(feature_cols)
    print(f"[temporal] Features: {n_features}")

    # ── Build sequences ──
    sequences, dev_ids, win_ids = build_temporal_sequences(
        feature_matrix, feature_cols, lookback=lookback, verbose=verbose
    )

    if len(sequences) == 0:
        print("[ERROR] No sequences built. Need more consecutive windows per device.")
        return {"error": "no_sequences"}

    # ── Temporal split ──
    # Sort by window_id, then split
    all_window_ids = sorted(set(win_ids))
    n_win = len(all_window_ids)
    train_end = int(n_win * 0.7)
    val_end = int(n_win * 0.85)

    train_windows = set(all_window_ids[:train_end])
    val_windows = set(all_window_ids[train_end:val_end])
    test_windows = set(all_window_ids[val_end:])

    train_idx = [i for i, w in enumerate(win_ids) if w in train_windows]
    val_idx = [i for i, w in enumerate(win_ids) if w in val_windows]
    test_idx = [i for i, w in enumerate(win_ids) if w in test_windows]

    def make_dataset(indices):
        seqs = [sequences[i] for i in indices]
        dids = [dev_ids[i] for i in indices]
        wids = [win_ids[i] for i in indices]
        return TemporalSequenceDataset(seqs, dids, wids)

    train_ds = make_dataset(train_idx)
    val_ds = make_dataset(val_idx)
    test_ds = make_dataset(test_idx)

    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        n_pos = sum(1 for _, y in ds.sequences if y == 1)
        print(f"[temporal] {name}: {len(ds)} sequences, {n_pos} positive")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Create model ──
    model = TemporalLSTM(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=1,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[temporal] Model: {total_params:,} parameters")

    # ── Loss with class imbalance ──
    n_pos_train = sum(1 for _, y in train_ds.sequences if y == 1)
    n_neg_train = len(train_ds) - n_pos_train
    pos_weight = torch.tensor(
        [n_neg_train / max(n_pos_train, 1)], dtype=torch.float32
    ).to(device)
    print(f"[temporal] pos_weight: {pos_weight.item():.2f}")

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
        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch).squeeze(-1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_batch.shape[0]
            train_samples += y_batch.shape[0]
        train_loss /= max(train_samples, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * y_batch.shape[0]
                val_samples += y_batch.shape[0]
        val_loss /= max(val_samples, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Best: {best_epoch}")

        if epoch - best_epoch >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best: {best_epoch}, val_loss: {best_val_loss:.6f})")
            break

    # ── Load best model ──
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    print(f"\n[temporal] Best epoch: {best_epoch}, val_loss: {best_val_loss:.6f}")

    # ── Evaluate on test set ──
    print(f"\n{'-'*50}")
    print("Evaluating on test set...")
    print(f"{'-'*50}")

    model.eval()
    all_y_true = []
    all_y_scores = []
    all_device_ids = []
    all_window_ids = []

    with torch.no_grad():
        test_batch_start = 0
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            probs = torch.sigmoid(model(x_batch).squeeze(-1)).cpu().numpy()
            y_np = y_batch.numpy()

            batch_size_actual = len(y_np)
            batch_indices = range(test_batch_start, test_batch_start + batch_size_actual)
            test_batch_start += batch_size_actual

            all_y_true.append(y_np)
            all_y_scores.append(probs)
            for i in batch_indices:
                if i < len(test_ds.device_ids):
                    all_device_ids.append(test_ds.device_ids[i])
                    all_window_ids.append(test_ds.window_ids[i])

    all_y_true = np.concatenate(all_y_true)
    all_y_scores = np.concatenate(all_y_scores)
    all_device_ids = np.array(all_device_ids)
    all_window_ids = np.array(all_window_ids)

    lstm_results = evaluate_predictions(
        all_y_true, all_y_scores,
        all_device_ids, all_window_ids,
        model_name="Temporal LSTM",
        verbose=verbose,
    )

    # ── Save model ──
    if save_model:
        model_path = PATHS["models"] / "temporal_lstm.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": n_features,
                "hidden_dim": hidden_dim,
                "num_layers": 1,
                "dropout": 0.3,
                "lookback": lookback,
            },
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history,
        }, model_path)
        print(f"[temporal] Model saved: {model_path}")

    # ── Load comparison and update ──
    comp_path = PATHS["experiments"] / f"model_comparison_{day}.json"
    all_results = []
    if comp_path.exists():
        with open(comp_path) as f:
            all_results = json.load(f)
        print(f"[temporal] Loaded {len(all_results)} previous model results")

    # Remove existing Temporal entry and add new
    all_results = [r for r in all_results if "Temporal" not in r.get("model", "")]
    all_results.append(lstm_results)

    if save_model:
        with open(comp_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[temporal] Updated comparison: {comp_path}")

    # ── Print comparison ──
    if len(all_results) > 1:
        print_comparison_table(all_results)

    return {
        "lstm_results": lstm_results,
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
        description="M6.3 — Temporal LSTM Model (optional/advanced)"
    )
    parser.add_argument("--day", default="wednesday", help="CICIDS2017 day")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Sequence length (windows)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="LSTM hidden dimension")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save model")
    args = parser.parse_args()

    results = train_and_evaluate(
        day=args.day,
        lookback=args.lookback,
        hidden_dim=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_model=not args.no_save,
    )
