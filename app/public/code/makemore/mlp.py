"""
Makemore: MLP Character-Level Language Model
Based on Lecture 3 of Andrej Karpathy's Neural Networks: Zero to Hero

Video: https://youtu.be/TCH_1BHY58I

This module implements:
- Multi-layer perceptron for character-level language modeling
- Embedding layer
- Train/dev/test splits
- Learning rate schedules
- Hyperparameter experimentation
"""

import torch
import torch.nn.functional as F
import random

# Hyperparameters
BLOCK_SIZE = 3        # Context length: how many characters to use to predict the next
EMBEDDING_DIM = 10    # Dimension of character embeddings
HIDDEN_SIZE = 200     # Number of neurons in hidden layer
BATCH_SIZE = 32       # Mini-batch size
LEARNING_RATE = 0.1   # Initial learning rate
EPOCHS = 200000       # Number of training iterations


def load_names(filepath='../../data/names.txt'):
    """Load names dataset."""
    # TODO: Load and return list of names
    pass


def build_vocabulary(names):
    """Build character vocabulary."""
    # TODO: Create stoi and itos mappings
    pass


def build_dataset(names, stoi, block_size=BLOCK_SIZE):
    """Build dataset of (context, target) pairs."""
    # TODO: Create X (contexts) and Y (targets) tensors
    # Each training example: [c1, c2, c3] -> c4
    pass


def split_data(names, train_ratio=0.8, dev_ratio=0.1):
    """Split data into train/dev/test sets."""
    # TODO: Randomly shuffle and split
    pass


class MLP:
    """Multi-layer perceptron character-level language model."""

    def __init__(self, vocab_size, block_size=BLOCK_SIZE,
                 embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE):
        self.block_size = block_size

        # TODO: Initialize parameters
        # C: embedding lookup table (vocab_size, embedding_dim)
        # W1: first layer weights (block_size * embedding_dim, hidden_size)
        # b1: first layer biases (hidden_size,)
        # W2: second layer weights (hidden_size, vocab_size)
        # b2: second layer biases (vocab_size,)
        pass

    def forward(self, X):
        """Forward pass."""
        # TODO: Implement forward pass
        # 1. Embed the input characters
        # 2. Concatenate embeddings
        # 3. Hidden layer with tanh
        # 4. Output layer
        # 5. Return logits (pre-softmax)
        pass

    def loss(self, X, Y):
        """Compute cross-entropy loss."""
        # TODO: Forward pass and compute loss
        pass

    def parameters(self):
        """Return list of all parameters."""
        # TODO: Return [C, W1, b1, W2, b2]
        pass

    def sample(self, itos, num_samples=10):
        """Sample names from the model."""
        # TODO: Generate names by sampling
        pass


def train(model, Xtr, Ytr, Xdev, Ydev, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    """Train the model."""
    # TODO: Training loop
    # - Sample mini-batch
    # - Forward pass
    # - Compute loss
    # - Backward pass
    # - Update parameters
    # - Optionally decay learning rate
    # - Track train/dev loss
    pass


if __name__ == "__main__":
    print("MLP Character-Level Language Model")
    print("=" * 40)

    # TODO: Load and prepare data

    # TODO: Build vocabulary

    # TODO: Create train/dev/test splits

    # TODO: Build datasets

    # TODO: Initialize and train model

    # TODO: Sample from trained model

    print("\nRun the notebook for interactive exploration!")
