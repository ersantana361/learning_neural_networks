"""
Makemore: Bigram Character-Level Language Model
Based on Lecture 2 of Andrej Karpathy's Neural Networks: Zero to Hero

Video: https://youtu.be/PaCmpygFfXo

This module implements:
- Character-level bigram language model
- torch.Tensor basics
- Negative log likelihood loss
- Sampling from the model
"""

import torch

# Hyperparameters
# TODO: Define any hyperparameters


def load_names(filepath='../../data/names.txt'):
    """Load names dataset."""
    # TODO: Load and return list of names
    pass


def build_vocabulary(names):
    """Build character vocabulary from names."""
    # TODO: Create mappings from characters to indices (stoi) and vice versa (itos)
    # Remember to include special start/end token
    pass


def create_bigram_counts(names, stoi):
    """Create bigram count matrix."""
    # TODO: Count occurrences of character pairs
    # Return a 2D tensor of counts
    pass


def create_probability_matrix(counts):
    """Convert counts to probabilities (row-normalized)."""
    # TODO: Normalize each row to sum to 1
    pass


def compute_nll(names, P, stoi):
    """Compute negative log likelihood of the dataset."""
    # TODO: Calculate NLL given probability matrix
    pass


def sample(P, itos, num_samples=10):
    """Sample names from the bigram model."""
    # TODO: Generate new names by sampling from probability distribution
    pass


# Neural Network Approach
def create_training_data(names, stoi):
    """Create training data (xs, ys) for neural network."""
    # TODO: Create input-output pairs for bigram prediction
    pass


class BigramNN:
    """Simple bigram model using neural network (single layer)."""

    def __init__(self, vocab_size):
        # TODO: Initialize weights
        # Single matrix W of shape (vocab_size, vocab_size)
        pass

    def forward(self, x):
        """Forward pass: one-hot encode, multiply by W, softmax."""
        # TODO: Implement forward pass
        pass

    def loss(self, x, y):
        """Compute cross-entropy loss."""
        # TODO: Implement loss computation
        pass


def train_nn(model, xs, ys, epochs=100, lr=1.0):
    """Train the neural network bigram model."""
    # TODO: Training loop with gradient descent
    pass


if __name__ == "__main__":
    print("Bigram Character-Level Language Model")
    print("=" * 40)

    # TODO: Load data
    # names = load_names()

    # TODO: Build vocabulary
    # stoi, itos = build_vocabulary(names)

    # TODO: Create and train model

    # TODO: Sample from model

    print("\nRun the notebook for interactive exploration!")
