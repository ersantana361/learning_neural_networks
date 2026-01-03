"""
Makemore: WaveNet-style Hierarchical Model
Based on Lecture 6 of Andrej Karpathy's Neural Networks: Zero to Hero

Video: https://youtu.be/t3YJ5hKiMQ0

This module implements:
- Hierarchical/tree-like architecture
- Dilated convolutions (conceptually)
- torch.nn.Module classes
- torch.nn.Sequential
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BLOCK_SIZE = 8        # Context length (should be power of 2 for tree structure)
EMBEDDING_DIM = 24    # Dimension of character embeddings
HIDDEN_SIZE = 128     # Number of neurons in hidden layers


class Linear(nn.Module):
    """Custom Linear layer (for understanding)."""

    def __init__(self, fan_in, fan_out, bias=True):
        super().__init__()
        # TODO: Initialize weight and bias
        # Use Kaiming initialization
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


class BatchNorm1d(nn.Module):
    """Custom BatchNorm layer (for understanding)."""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # TODO: Initialize gamma, beta, running_mean, running_var
        pass

    def forward(self, x):
        # TODO: Implement batch normalization
        # During training: use batch statistics
        # During inference: use running statistics
        pass


class Tanh(nn.Module):
    """Tanh activation (for understanding torch.nn structure)."""

    def forward(self, x):
        return torch.tanh(x)


class Embedding(nn.Module):
    """Custom Embedding layer (for understanding)."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # TODO: Initialize embedding weight matrix
        pass

    def forward(self, x):
        # TODO: Look up embeddings for input indices
        pass


class FlattenConsecutive(nn.Module):
    """Flatten consecutive elements (for tree-like architecture)."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        # TODO: Reshape to group n consecutive elements
        # e.g., (B, 8, C) with n=2 -> (B, 4, 2*C)
        pass


class Sequential(nn.Module):
    """Custom Sequential container (for understanding)."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class WaveNetModel(nn.Module):
    """
    WaveNet-style hierarchical character-level language model.

    Architecture (for block_size=8):
    - Embed characters: (B, 8) -> (B, 8, emb_dim)
    - Flatten pairs: (B, 8, emb_dim) -> (B, 4, 2*emb_dim)
    - Linear + BN + Tanh: (B, 4, 2*emb_dim) -> (B, 4, hidden)
    - Flatten pairs: (B, 4, hidden) -> (B, 2, 2*hidden)
    - Linear + BN + Tanh: (B, 2, 2*hidden) -> (B, 2, hidden)
    - Flatten pairs: (B, 2, hidden) -> (B, 1, 2*hidden)
    - Linear: (B, 1, 2*hidden) -> (B, vocab_size)
    """

    def __init__(self, vocab_size, block_size=BLOCK_SIZE,
                 embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE):
        super().__init__()

        # TODO: Build the hierarchical model
        # Hint: Use FlattenConsecutive to merge pairs at each level
        pass

    def forward(self, x):
        # TODO: Forward pass through all layers
        pass


def load_and_prepare_data(filepath='../../data/names.txt'):
    """Load data and create train/dev/test splits."""
    # TODO: Load names, build vocabulary, create datasets
    pass


def train(model, train_loader, dev_loader, epochs=10000, lr=0.1):
    """Train the WaveNet model."""
    # TODO: Training loop with mini-batches
    pass


if __name__ == "__main__":
    print("WaveNet-style Character-Level Language Model")
    print("=" * 50)

    # TODO: Load and prepare data

    # TODO: Create model

    # TODO: Train model

    # TODO: Sample from model

    print("\nRun the notebook for interactive exploration!")
