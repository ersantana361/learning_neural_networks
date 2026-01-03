"""
nanoGPT: A minimal GPT implementation
Based on Lecture 7 of Andrej Karpathy's Neural Networks: Zero to Hero

Video: https://youtu.be/kCc8FmEb1nY

This module implements:
- Self-attention mechanism
- Multi-head attention
- Transformer blocks
- Positional encodings
- Causal masking
- Full GPT model from "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 64       # Number of independent sequences per batch
BLOCK_SIZE = 256      # Maximum context length
EMBEDDING_DIM = 384   # Embedding dimension
NUM_HEADS = 6         # Number of attention heads
NUM_LAYERS = 6        # Number of transformer blocks
DROPOUT = 0.2         # Dropout rate
LEARNING_RATE = 3e-4  # Learning rate


class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size, embedding_dim=EMBEDDING_DIM,
                 block_size=BLOCK_SIZE, dropout=DROPOUT):
        super().__init__()
        # TODO: Initialize key, query, value projections
        # TODO: Register causal mask (lower triangular)
        # TODO: Dropout layer
        pass

    def forward(self, x):
        """
        Apply self-attention.

        Args:
            x: (B, T, C) input tensor

        Returns:
            (B, T, head_size) attention output
        """
        # TODO: Compute queries, keys, values
        # TODO: Compute attention scores (Q @ K^T / sqrt(d_k))
        # TODO: Apply causal mask
        # TODO: Softmax
        # TODO: Apply dropout
        # TODO: Weighted sum of values
        pass


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size, embedding_dim=EMBEDDING_DIM,
                 dropout=DROPOUT):
        super().__init__()
        # TODO: Create multiple attention heads
        # TODO: Projection layer
        # TODO: Dropout
        pass

    def forward(self, x):
        # TODO: Run all heads in parallel and concatenate
        # TODO: Project back to embedding dimension
        pass


class FeedForward(nn.Module):
    """Feed-forward network (per-position)."""

    def __init__(self, embedding_dim=EMBEDDING_DIM, dropout=DROPOUT):
        super().__init__()
        # TODO: Two linear layers with ReLU in between
        # Hidden dimension is typically 4x embedding_dim
        pass

    def forward(self, x):
        # TODO: Linear -> ReLU -> Linear -> Dropout
        pass


class Block(nn.Module):
    """Transformer block: communication (attention) + computation (FFN)."""

    def __init__(self, embedding_dim=EMBEDDING_DIM, num_heads=NUM_HEADS,
                 dropout=DROPOUT):
        super().__init__()
        head_size = embedding_dim // num_heads
        # TODO: Multi-head attention
        # TODO: Feed-forward network
        # TODO: Layer norms
        pass

    def forward(self, x):
        # TODO: Attention with residual connection and layer norm
        # TODO: FFN with residual connection and layer norm
        pass


class GPT(nn.Module):
    """GPT Language Model."""

    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM,
                 block_size=BLOCK_SIZE, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.block_size = block_size

        # TODO: Token embedding table
        # TODO: Position embedding table
        # TODO: Transformer blocks
        # TODO: Final layer norm
        # TODO: Language model head (projects to vocab)
        pass

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: (B, T) tensor of token indices
            targets: (B, T) tensor of target indices (optional)

        Returns:
            logits: (B, T, vocab_size) predictions
            loss: scalar loss if targets provided, else None
        """
        # TODO: Get token embeddings
        # TODO: Add position embeddings
        # TODO: Pass through transformer blocks
        # TODO: Final layer norm
        # TODO: Project to vocabulary
        # TODO: Compute loss if targets provided
        pass

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.

        Args:
            idx: (B, T) current context
            max_new_tokens: number of tokens to generate

        Returns:
            (B, T + max_new_tokens) extended sequence
        """
        # TODO: Loop to generate tokens one at a time
        # - Crop context to block_size
        # - Get predictions
        # - Sample from distribution
        # - Append to sequence
        pass


def load_shakespeare(filepath='../../data/shakespeare.txt'):
    """Load and prepare Shakespeare dataset."""
    # TODO: Read text file
    # TODO: Build vocabulary (character-level)
    # TODO: Encode text to integers
    pass


def get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """Generate a batch of training data."""
    # TODO: Randomly sample starting positions
    # TODO: Extract context and target sequences
    pass


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters=200):
    """Estimate loss on train and validation sets."""
    # TODO: Evaluate on multiple batches and average
    pass


def train(model, train_data, val_data, epochs=5000, lr=LEARNING_RATE):
    """Train the GPT model."""
    # TODO: Training loop
    # - Sample batch
    # - Forward pass
    # - Compute loss
    # - Backward pass
    # - Update parameters
    # - Periodically evaluate
    pass


if __name__ == "__main__":
    print("nanoGPT: Minimal GPT Implementation")
    print("=" * 40)

    # TODO: Load Shakespeare data

    # TODO: Create model

    # TODO: Train model

    # TODO: Generate text

    print("\nRun the notebook for interactive exploration!")
