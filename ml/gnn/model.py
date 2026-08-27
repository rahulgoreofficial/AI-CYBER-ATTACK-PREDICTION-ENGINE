"""
GNN Model Architecture — M5.2
================================

A 2-layer GraphSAGE model for node-level binary classification
(next-target prediction).

Architecture:
    Input (d_in features)
      → SAGEConv(d_in, hidden)  + BatchNorm + ReLU + Dropout
      → SAGEConv(hidden, hidden) + BatchNorm + ReLU + Dropout
      → Linear(hidden, 1)
      → Sigmoid (during inference)

Why GraphSAGE:
    - Aggregates from local neighborhoods (learns topology influence)
    - Works well for inductive settings (handles varying graph sizes)
    - More scalable than full-graph GCN approaches
    - Natural fit for "neighborhood influence propagation" in attack prediction

Usage:
    from ml.gnn.model import NextTargetGNN
    model = NextTargetGNN(in_channels=120, hidden_channels=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, global_mean_pool


class NextTargetGNN(nn.Module):
    """
    GraphSAGE-based GNN for predicting which network device
    will be the next attack target.

    Node-level binary classification: for each node in the graph,
    output a probability of being targeted in the next time horizon.

    Args:
        in_channels: Number of input node features.
        hidden_channels: Hidden layer dimension (default: 64).
        num_layers: Number of GraphSAGE layers (default: 2).
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Build SAGE convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer: input → hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Intermediate layers: hidden → hidden
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Classification head: hidden → 1 (binary output)
        self.classifier = nn.Linear(hidden_channels, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for the classifier head."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.

        Args:
            x: Node feature matrix [num_nodes, in_channels].
            edge_index: COO edge index [2, num_edges].
            edge_attr: Edge features (not used by SAGEConv, but kept for API compat).

        Returns:
            logits: Raw predictions [num_nodes, 1] (apply sigmoid for probs).
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Classification head
        logits = self.classifier(x)  # [num_nodes, 1]

        return logits

    def predict_proba(self, x, edge_index, edge_attr=None):
        """
        Get probabilities (sigmoid applied to logits).

        Returns:
            probabilities: [num_nodes] tensor of attack probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_attr)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs

    def __repr__(self):
        return (
            f"NextTargetGNN(\n"
            f"  layers={self.num_layers},\n"
            f"  hidden={self.convs[0].out_channels if self.convs else '?'},\n"
            f"  dropout={self.dropout},\n"
            f"  classifier={self.classifier}\n"
            f")"
        )


# ==============================================================================
# MODEL FACTORY
# ==============================================================================

def create_model(
    in_channels: int,
    hidden_channels: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> NextTargetGNN:
    """
    Create a NextTargetGNN model with the given configuration.

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Hidden dimension (default: 64).
        num_layers: Number of SAGE layers (default: 2).
        dropout: Dropout rate (default: 0.3).

    Returns:
        Initialized NextTargetGNN model.
    """
    model = NextTargetGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[gnn] Model created: {total_params:,} total params, "
          f"{trainable_params:,} trainable")
    return model


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    print("Testing NextTargetGNN model...")

    # Simulate a small graph: 10 nodes, 120 features, 15 edges
    num_nodes = 10
    num_features = 120
    num_edges = 15

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.zeros(num_nodes)
    y[0] = 1.0  # One target

    # Create model
    model = create_model(in_channels=num_features)
    print(model)

    # Forward pass
    logits = model(x, edge_index)
    assert logits.shape == (num_nodes, 1), f"Bad logits shape: {logits.shape}"
    print(f"Logits shape: {logits.shape}")

    # Probabilities
    probs = model.predict_proba(x, edge_index)
    assert probs.shape == (num_nodes,), f"Bad probs shape: {probs.shape}"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probs out of [0,1]!"
    print(f"Probs shape: {probs.shape}, range: [{probs.min():.4f}, {probs.max():.4f}]")

    # Loss computation
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits.squeeze(), y)
    print(f"Sample loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("Backward pass: OK")

    print("\n[OK] Model test passed.")
