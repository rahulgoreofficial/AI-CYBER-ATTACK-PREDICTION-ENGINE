"""
Graph Neural Network — PyTorch Geometric models for graph-based prediction.

Modules:
    dataset.py  — M5.1: PyG dataset converter (NetworkX → PyG Data)
    model.py    — M5.2: GraphSAGE model architecture
    train.py    — M5.3: Training, evaluation, and comparison pipeline
"""

from ml.gnn.model import NextTargetGNN, create_model
from ml.gnn.dataset import build_pyg_dataset
