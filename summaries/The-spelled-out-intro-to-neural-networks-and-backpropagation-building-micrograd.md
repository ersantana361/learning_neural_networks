---
title: The spelled-out intro to neural networks and backpropagation: building micrograd
tags:
  - automatic differentiation
  - backpropagation
  - autograd engine
  - computational graph
  - chain rule
  - gradient descent
  - neural network training
  - Micrograd
  - PyTorch fundamentals
  - scalar autograd
  - manual gradient calculation
  - topological sort
  - loss function
  - parameter update
  - zeroing gradients
  - pedagogical implementation
  - neural network from scratch
  - directed acyclic graph (DAG)
  - local gradient
  - forward pass
  - backward pass
---

## Introduction

**Key objectives of the video**: To provide an intuitive, ground-up understanding of neural network training by building an autograd engine (Micrograd) and a simple neural network library from scratch.

**Core themes and methodologies**: Implementing automatic differentiation (backpropagation) for scalar values, constructing neural network components (neurons, layers, MLPs), and demonstrating gradient descent optimization.

**Target audience**: Individuals seeking a foundational, code-first understanding of the mechanics behind deep learning frameworks like PyTorch.

---

## Detailed Analysis

### Segment: Introduction to Micrograd and Backpropagation (0:00 - 1:57)

**Key verbatim excerpts**:
- "backpropagation is this algorithm that allows you to efficiently evaluate the gradient"
- "micrograd is basically an autograd engine... it implements backpropagation"

**Technical analysis**: Establishes backpropagation as the core mathematical algorithm for training neural networks by computing gradients of a loss function with respect to network weights. Introduces Micrograd as a minimal, pedagogical implementation to demystify this process.

#### 💡 Intuitive Understanding: What is Backpropagation?

**Analogy - The Blame Game**: Imagine you're the manager of a factory producing widgets. The final product has a defect (the "loss"). Backpropagation is like tracing back through the assembly line asking: "How much did each worker contribute to this defect?" Workers who contributed more get more "blame" (larger gradients) and need to adjust their technique more.

**Mental Model**: Picture a river flowing downstream (the forward pass). Backpropagation is like sending a message upstream: "Hey, the water quality at the end is bad. Each tributary needs to know how much it contributed to the problem."

**Why It Matters**: Without backpropagation, we'd have to randomly guess how to improve our neural network. With it, we know *exactly* which direction to adjust each parameter to reduce errors. This transforms learning from random search into efficient optimization.

---

### Segment: Demonstrating Micrograd's Functionality (1:58 - 4:31)

**Key verbatim excerpts**:
- "micrograd will in the background build out this entire mathematical expression"
- "we can call... backward and this will basically... initialize back propagation"

**Technical analysis**: Shows how Micrograd constructs a computational graph from operations on `Value` objects. The `.backward()` call triggers the recursive application of the chain rule, calculating derivatives (`.grad`) for all inputs. This is the fundamental mechanism that powers gradient-based optimization.

#### 💡 Intuitive Understanding: The Computational Graph

**Analogy - A Recipe**: A computational graph is like a detailed recipe. Each step (node) produces an intermediate result. If the final dish tastes wrong, you can trace back through the recipe to find which step caused the problem and how sensitive the final taste is to each ingredient amount.

**Mental Model**: Visualize a flowchart where:
- Each box represents a value or operation
- Arrows show data dependencies
- Forward pass: Follow arrows forward to compute the output
- Backward pass: Follow arrows backward to compute gradients

```
     a ──┐
         ├──► [×] ──► c ──┐
     b ──┘                 ├──► [+] ──► e (output)
                      d ──┘
```

**Why It Matters**: The graph structure makes backpropagation efficient. Instead of computing each gradient independently, we reuse intermediate results as we traverse backward. This is why neural networks with millions of parameters can be trained in reasonable time.

---

### Segment: Pedagogical Rationale for Scalar Engine (4:32 - 6:48)

**Key verbatim excerpts**:
- "micrograd is a scalar valued autograd engine"
- "this is really done so that you understand... back propagation and chain rule"

**Technical analysis**: Justifies the scalar-based design as a pedagogical tool to isolate the core calculus (chain rule) from the complexities of tensor operations and parallel efficiency found in production libraries (PyTorch, JAX). The underlying math is identical.

#### 💡 Intuitive Understanding: Scalars vs Tensors

**Analogy - Learning to Drive**: Learning backpropagation with scalars is like learning to drive in an empty parking lot before hitting the highway. The principles are identical, but the simpler environment lets you focus on fundamentals.

**Mental Model**: Think of tensors as "batches" of scalars processed in parallel. The chain rule works the same way, just applied element-wise. Once you understand scalar backprop, tensor backprop is just "do it for many numbers at once."

**Why It Matters**: PyTorch, TensorFlow, and JAX all use the same principles under the hood—they just operate on tensors for efficiency. Master scalar backprop, and you've mastered the conceptual foundation of all deep learning frameworks.

---

### Segment: Intuition of Derivatives and Numerical Approximation (8:20 - 14:11)

**Key verbatim excerpts**:
- "what is the derivative... it is the limit as h goes to zero"
- "we can basically evaluate the derivative here numerically"

**Technical analysis**: Reviews the foundational concept of a derivative as a local sensitivity measure. Uses finite differences (`(f(x+h) - f(x))/h`) to numerically approximate gradients, establishing a baseline for verifying the correctness of the automatic differentiation system to be built.

#### 💡 Intuitive Understanding: What is a Derivative?

**Analogy - The Speedometer**: If position is a function of time, the derivative is your speedometer reading—it tells you how fast your position is changing *right now*. For neural networks, the derivative tells you how fast the loss changes when you wiggle a particular weight.

**Mental Model**: The derivative answers: "If I nudge this input by a tiny amount ε, how much does the output change?" A derivative of 3 means the output changes roughly 3× as much as the input nudge.

```
derivative = lim(h→0) [f(x + h) - f(x)] / h
           ≈ "output change" / "input change"
           ≈ "sensitivity of output to input"
```

**Why It Matters**: Derivatives are the language of optimization. Positive derivative means "increase input → increase output." Negative means the opposite. Zero means "we're at a local maximum or minimum." This is the foundation of gradient descent.

---

### Segment: Building the `Value` Object and Computational Graph (19:20 - 25:03)

**Key verbatim excerpts**:
- "we're going to build out this value object"
- "we need to know... keep pointers about what values produce what other values"

**Technical analysis**: Implements the core data structure (`Value`) that wraps a scalar, its gradient, and crucially, stores references to its child `Value` objects and the operation that created it. This builds the directed acyclic graph (DAG) necessary for traversing the computation during backpropagation.

#### 💡 Intuitive Understanding: The Value Object

**Analogy - A Package with a Tracking Number**: Each `Value` is like a shipped package. It contains:
- The actual item (`.data` - the number)
- Shipping history (`.prev` - what values created it)
- A "damage report" slot (`.grad` - to be filled during backward pass)
- Instructions for damage attribution (`.backward` - how to distribute blame)

**Mental Model**: The `Value` class has four key attributes:
```python
class Value:
    data: float      # The actual number
    grad: float      # How much this affects the final loss (filled during backward)
    _prev: set       # Parent nodes that created this value
    _backward: func  # How to propagate gradients to parents
```

**Why It Matters**: This simple data structure is the building block of every modern autograd system. PyTorch's `Tensor` class is essentially a sophisticated version of this, with the same core concept of tracking computational history.

---

### Segment: Manual Backpropagation and Chain Rule Application (29:33 - 40:33)

**Key verbatim excerpts**:
- "we are going to compute the derivative of that node with respect to L"
- "the chain rule is telling us that... you take... and you multiply it by"

**Technical analysis**: Walks through manually calculating gradients for a simple graph. This critical segment visually demonstrates how gradients flow backward from the output. The chain rule is shown as the mechanism for combining a "global" gradient from upstream with a "local" gradient of an operation.

#### 💡 Intuitive Understanding: The Chain Rule

**Analogy - The Telephone Game**: Imagine you whisper a message through a chain of people. The chain rule tells you how the final message relates to the original. Each person modifies the message by some factor. The total modification is the product of all individual modifications.

**Mathematical Form**:
```
If: y = f(g(x))
Then: dy/dx = df/dg × dg/dx
       ↑         ↑        ↑
    "total"  "outer"  "inner"
    effect   effect   effect
```

**Mental Model - Local × Global**: At each node during backpropagation:
1. **Global gradient**: "How much does the final loss change when I change?" (received from downstream)
2. **Local gradient**: "How much does my output change when my input changes?" (computed from the operation)
3. **Propagated gradient**: Global × Local (sent upstream)

```
      a ────► [f] ────► b ────► [g] ────► c (loss)

Backward:
      ∂c/∂a ◄── [∂b/∂a] ◄── ∂c/∂b ◄── [∂c/∂b] ◄── 1
              (local)              (local)
```

**Why It Matters**: The chain rule is why backpropagation works. It lets us decompose the problem of "how does the loss depend on this weight deep in the network?" into a series of simple, local derivative calculations chained together.

---

### Segment: Automating Backpropagation with Backward Functions (69:27 - 77:23)

**Key verbatim excerpts**:
- "we're going to store a special... backward... function"
- "our job is to take out's grad and propagate it into self's grad and other grad"

**Technical analysis**: Encodes the manual chain rule logic into *local backward functions* attached to each operation (`+`, `*`, `tanh`). Each function knows how to distribute the incoming gradient to its inputs using its local derivatives. This modularizes the backpropagation algorithm.

#### 💡 Intuitive Understanding: Local Backward Functions

**Analogy - Assembly Line Workers**: Each operation is like a worker who knows exactly two things:
1. How to do their job (forward pass)
2. How to pass blame backward (backward pass)

When the quality inspector (loss function) finds a problem, each worker knows how to distribute responsibility to their upstream suppliers.

**Mental Model - Operation-Specific Rules**:

| Operation | Forward | Backward (local derivative) |
|-----------|---------|---------------------------|
| c = a + b | c = a + b | ∂c/∂a = 1, ∂c/∂b = 1 |
| c = a × b | c = a × b | ∂c/∂a = b, ∂c/∂b = a |
| c = tanh(a) | c = tanh(a) | ∂c/∂a = 1 - tanh²(a) |
| c = a ** n | c = aⁿ | ∂c/∂a = n × a^(n-1) |

**Why It Matters**: This modular design means you can add new operations to an autograd system just by defining their forward and backward functions. The system automatically chains them together. This is how PyTorch supports thousands of different operations.

---

### Segment: Topological Sort and Full Backward Pass (77:24 - 82:26)

**Key verbatim excerpts**:
- "this ordering of graphs can be achieved using something called topological sort"
- "we're just calling dot underscore backward on all of the nodes"

**Technical analysis**: Implements a topological sort to order all nodes in the graph from inputs to output. The backward pass then processes nodes in *reverse* topological order, ensuring that a node's gradient is fully calculated before propagating it to its children. This is the complete backpropagation algorithm.

#### 💡 Intuitive Understanding: Why Order Matters

**Analogy - Paying Debts**: Imagine a chain of people who owe each other money: A owes B, B owes C, C owes D. To figure out everyone's final balance, you must settle debts in the right order. If D needs money from C to pay their bills, C must be paid first.

**Mental Model**: In backpropagation, gradients "accumulate" as they flow backward. A node might receive gradients from multiple downstream nodes. Topological sort ensures:
1. All downstream nodes are processed first
2. When we process a node, its gradient is complete (all contributions received)
3. Only then do we propagate to upstream nodes

```
Forward order:  a → b → c → d → e (topological)
Backward order: e → d → c → b → a (reverse topological)
```

**Why It Matters**: Without proper ordering, you'd propagate incomplete gradients, leading to wrong results. Topological sort guarantees correctness regardless of graph complexity. This is why PyTorch can handle arbitrary computational graphs automatically.

---

### Segment: Implementing Neural Network Components (103:57 - 110:05)

**Key verbatim excerpts**:
- "neural nets are just a specific class of mathematical expressions"
- "let's start with a single individual neuron"

**Technical analysis**: Builds neural network abstractions (`Neuron`, `Layer`, `MLP`) on top of the autograd engine. Shows that a neuron is simply a mathematical expression (weighted sum + non-linearity), and networks are compositions of these layers. This bridges the low-level calculus with high-level deep learning concepts.

#### 💡 Intuitive Understanding: What is a Neuron?

**Analogy - A Voting Committee Member**: Each neuron is like a committee member who:
1. Listens to multiple inputs (previous neurons or raw features)
2. Weighs each input by importance (learned weights)
3. Adds a personal bias (learned bias term)
4. Makes a decision through a non-linear "voting function" (activation like tanh or ReLU)

**Mental Model**:
```
Neuron computation:
                    w1
    x1 ──────────────┐
                     │
                w2   ▼
    x2 ──────────► [Σ + b] ──► [tanh] ──► output
                     ▲
                w3   │
    x3 ──────────────┘

output = tanh(w1×x1 + w2×x2 + w3×x3 + b)
```

**Why It Matters**: Understanding that a neuron is just a parameterized function demystifies neural networks. A network is just many simple functions composed together. There's no magic—just calculus and optimization applied to a clever function architecture.

---

### Segment: Loss Function and Gradient Descent Training Loop (111:08 - 130:24)

**Key verbatim excerpts**:
- "the loss... is a single number that... measures how well the neural net is performing"
- "we want to minimize the loss... we actually want to go in the opposite direction"

**Technical analysis**: Integrates all components into a training loop. Defines a loss (Mean Squared Error) quantifying network error. Demonstrates gradient descent: forward pass, backward pass to get gradients, then updating each parameter with `p.data += -step_size * p.grad`. This is the essence of how neural networks learn from data.

#### 💡 Intuitive Understanding: The Training Loop

**Analogy - Adjusting a Recipe**: Training a neural network is like perfecting a recipe:
1. **Forward pass**: Cook the dish (make predictions)
2. **Compute loss**: Taste it and score how bad it is (compare to target)
3. **Backward pass**: Figure out which ingredients caused the bad taste (compute gradients)
4. **Update**: Adjust ingredient amounts in the direction that improves taste (gradient descent)
5. **Repeat**: Keep iterating until the dish is perfect

**Mental Model - The Core Training Loop**:
```python
for step in range(num_steps):
    # 1. Forward pass: compute predictions
    predictions = model(inputs)

    # 2. Compute loss: how wrong are we?
    loss = mean_squared_error(predictions, targets)

    # 3. Backward pass: who's responsible?
    loss.backward()

    # 4. Update: adjust parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # 5. Zero gradients for next iteration
    model.zero_grad()
```

**Why It Matters**: This loop is the heartbeat of all neural network training—from tiny demos to GPT-4. Understanding it lets you debug training issues, design custom training procedures, and reason about why networks learn (or fail to learn).

---

### Segment: Critical Bug - Forgetting to Zero Gradients (130:25 - 134:04)

**Key verbatim excerpts**:
- "we actually have a really terrible bug... you forgot to zero grad"
- "all these backward operations do a plus equals on the grad"

**Technical analysis**: Highlights a common, subtle bug where gradients accumulate across training steps because `backward` uses `+=`. The fix is to explicitly set all `.grad` attributes to zero before each `.backward()` call. This is a crucial practical detail for correct optimization.

#### 💡 Intuitive Understanding: Gradient Accumulation Bug

**Analogy - A Running Tab**: Imagine a restaurant bill that never gets cleared. Each meal adds to the total, but you keep paying based on the accumulated total instead of just the current meal. Soon you're paying for hundreds of meals you already paid for!

**Mental Model**: Gradients use `+=` because a variable might be used multiple times in a computation (its gradient is the *sum* of contributions from each use). But between training steps, we want fresh gradients:

```python
# BUG: gradients accumulate across steps!
for step in range(100):
    loss = compute_loss()
    loss.backward()  # grad += new_gradient (ACCUMULATES!)
    update_params()

# FIX: zero gradients before each backward pass
for step in range(100):
    zero_all_gradients()  # grad = 0
    loss = compute_loss()
    loss.backward()       # grad = new_gradient (FRESH!)
    update_params()
```

**Why It Matters**: This bug is incredibly common, especially for beginners. It causes training to behave erratically or fail to converge. Always remember: **zero gradients before backward, every single time**. This is why PyTorch has `optimizer.zero_grad()`.

---

## Conclusion

**Summary of key technical takeaways**: Neural network training is the iterative application of:
1. **Forward pass**: Expression evaluation (compute predictions)
2. **Backward pass**: Gradient calculation via chain rule on a computational graph
3. **Parameter update**: Gradient descent (move opposite to gradients)

Autograd engines automate step 2, making gradient computation automatic regardless of network architecture.

**Practical applications**: The principles demonstrated with Micrograd scale directly to production libraries like PyTorch, which add tensor operations, GPU acceleration, and more advanced optimizers but retain the same core autograd and module-based design.

**Long-term recommendations**: Use this foundational understanding to debug training issues, reason about model behavior, and contribute to or extend advanced frameworks. The conceptual model of computational graphs and gradient flow is universal in modern deep learning.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Explain in your own words**: Why does backpropagation use the chain rule? What would happen if we tried to compute gradients without it?

2. **Debugging scenario**: Your neural network's loss is increasing instead of decreasing during training. List three possible causes related to concepts from this lecture.

3. **Compare and contrast**: What's the difference between a "local gradient" and a "global gradient" in the context of backpropagation? Give an example.

4. **Reasoning about flow**: If a node in the computational graph has a gradient of zero, what does this mean for all nodes upstream of it? Why?

5. **Efficiency insight**: Why is it important that computational graphs are DAGs (Directed Acyclic Graphs) and not general graphs with cycles?

### Coding Challenges

1. **Implement a new operation**: Extend the `Value` class to support the `exp()` operation (e^x). Implement both the forward pass and the backward function. Test it using numerical gradient checking.

   ```python
   # Hint: d/dx(e^x) = e^x
   # Your implementation should pass:
   a = Value(2.0)
   b = a.exp()
   b.backward()
   assert abs(a.grad - math.exp(2.0)) < 1e-6
   ```

2. **Implement ReLU**: Add the ReLU activation function (max(0, x)) to the `Value` class. Be careful with the gradient when x = 0 (it's undefined, but conventionally set to 0).

   ```python
   # Hint: d/dx(ReLU(x)) = 1 if x > 0, else 0
   ```

3. **Gradient accumulation**: Create a computational graph where one variable is used twice (e.g., `y = a * a`). Manually trace through the backward pass to verify that gradients accumulate correctly. Why is the gradient of `a` equal to `2*a`?

4. **Visualize the graph**: Write a function that takes a `Value` and prints out the computational graph in a readable format. Include the operation type and current gradient at each node.

5. **Build a classifier**: Using your Micrograd implementation, build a simple binary classifier for the XOR problem. Train it and plot the loss over time. How many neurons do you need in the hidden layer?

### Reflection

- **Connect the dots**: How does understanding scalar backpropagation help you debug issues in PyTorch, where you're working with tensors? Can you think of a specific scenario?

- **Historical perspective**: Before automatic differentiation, researchers computed gradients by hand for each new architecture. How might this have limited the development of deep learning?

- **Look ahead**: The chain rule requires knowing the derivative of each operation. What challenges might arise when trying to differentiate operations that don't have clean mathematical derivatives (e.g., if statements, loops)?
