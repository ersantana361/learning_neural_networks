"""
Micrograd: A tiny autograd engine
Based on Lecture 1 of Andrej Karpathy's Neural Networks: Zero to Hero

Video: https://youtu.be/VMj-3S1tku0

This module implements:
- Value class with automatic differentiation
- Computational graph construction
- Backpropagation via reverse-mode autodiff
- Simple neural network layers (Neuron, Layer, MLP)
"""


class Value:
    """
    A scalar value that tracks its computational graph for automatic differentiation.

    Example:
        a = Value(2.0)
        b = Value(-3.0)
        c = a * b
        c.backward()
        print(a.grad)  # dc/da
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # TODO: Implement addition
        pass

    def __mul__(self, other):
        # TODO: Implement multiplication
        pass

    def __pow__(self, other):
        # TODO: Implement power
        pass

    def tanh(self):
        # TODO: Implement tanh activation
        pass

    def exp(self):
        # TODO: Implement exponential
        pass

    def backward(self):
        # TODO: Implement backpropagation
        # Build topological order of all nodes
        # Call _backward() in reverse order
        pass

    # Reverse operations for commutativity
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)


class Neuron:
    """A single neuron with weights, bias, and nonlinearity."""

    def __init__(self, nin):
        # TODO: Initialize weights and bias
        pass

    def __call__(self, x):
        # TODO: Implement forward pass (w*x + b, then activation)
        pass

    def parameters(self):
        # TODO: Return list of parameters
        pass


class Layer:
    """A layer of neurons."""

    def __init__(self, nin, nout):
        # TODO: Create nout neurons, each with nin inputs
        pass

    def __call__(self, x):
        # TODO: Forward pass through all neurons
        pass

    def parameters(self):
        # TODO: Return all parameters from all neurons
        pass


class MLP:
    """Multi-Layer Perceptron."""

    def __init__(self, nin, nouts):
        # TODO: Build layers based on architecture
        # nouts is a list like [4, 4, 1] for hidden, hidden, output
        pass

    def __call__(self, x):
        # TODO: Forward pass through all layers
        pass

    def parameters(self):
        # TODO: Return all parameters from all layers
        pass


if __name__ == "__main__":
    # Test your implementation
    print("Testing Value class...")

    # Simple test
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')

    print(f"a = {a}")
    print(f"b = {b}")

    # TODO: Add more tests as you implement operations
