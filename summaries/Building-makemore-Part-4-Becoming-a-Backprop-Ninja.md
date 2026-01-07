---
title: Building makemore Part 4: Becoming a Backprop Ninja
tags:
  - manual backpropagation
  - gradient computation
  - autograd replacement
  - neural network internals
  - PyTorch
  - chain rule
  - cross-entropy loss
  - softmax
  - batch normalization
  - linear layer gradients
  - tanh activation
  - debugging neural networks
  - computational graph
  - tensor operations
  - training loop
  - gradient derivation
  - numerical stability
  - vanishing gradients
  - exploding gradients
  - backpropagation mechanics
---

*Introduction*
• *Key objectives of the video*: To manually implement the backward pass of a neural network using tensor-level operations, replacing PyTorch's autograd (`loss.backward()`). The goal is to deepen understanding of backpropagation, improve debugging skills, and demystify gradient computation.
• *Core themes and methodologies*: Treating backpropagation as a "leaky abstraction" that requires internal understanding. The method involves breaking down the forward pass into intermediate tensors, then manually calculating and chaining gradients backward through each operation, including cross-entropy loss, linear layers, tanh activation, and batch normalization.
• *Target audience*: Machine learning practitioners and students with foundational knowledge of neural networks and PyTorch, who seek a deeper, hands-on understanding of gradient flow and backpropagation mechanics.

*Detailed Analysis*

• *0:00 - 3:00: Introduction and Motivation for Manual Backpropagation*
    1. *"back propagation does doesn't just make your neural networks just work magically"*
    2. *"it is a leaky abstraction in the sense that you can shoot yourself in the foot if you do not understanding its internals"*
    • *Technical analysis*: The presenter argues that relying solely on autograd can obscure critical issues like vanishing/exploding gradients, dead neurons, and implementation bugs. Understanding the tensor-level flow is essential for effective debugging and optimization.

### 💡 Intuitive Understanding

**Analogy:** Autograd is like a GPS that gives you turn-by-turn directions. It's convenient, but if you don't understand how roads connect, you can't troubleshoot when the GPS gives bad directions. Manual backpropagation is like learning to read a map—you understand *why* each turn matters.

**Mental Model:** Think of autograd as a helpful layer that hides complexity. But hidden complexity can cause hidden problems:
- Why is my model not learning? (vanishing gradients)
- Why does training explode? (exploding gradients)
- Why do some neurons never update? (dead ReLU, saturation)
- Why does my custom layer break everything? (incorrect gradient)

**Why It Matters:** The most important debugging skills in deep learning come from understanding gradient flow. When something goes wrong, autograd can't tell you *why*—you need to understand the mechanics.

---

• *3:00 - 7:04: Historical Context and Exercise Overview*
    1. *"about 10 years ago in deep learning this was fairly standard and in fact pervasive"*
    2. *"we're going to keep everything the same so we're still going to have a two layer multiplayer perceptron"*
    • *Technical analysis*: Provides historical perspective, showing that manual gradient calculation was once the norm. Outlines the structure of the upcoming exercises: 1) Backpropagate through the entire broken-down graph, 2) Analytically derive the gradient for the cross-entropy loss, 3) Analytically derive the gradient for batch normalization, 4) Assemble a full training loop with manual gradients.

### 💡 Intuitive Understanding

**Analogy:** Before automatic transmission (autograd), everyone drove stick shift (manual gradients). While you can drive without understanding transmissions, professional drivers (researchers) still learn manual because it gives finer control and deeper understanding.

**Mental Model:** The four exercises build progressively:
```
Exercise 1: Backprop through each tiny step (tedious but illuminating)
Exercise 2: Derive cross-entropy gradient as single formula (efficient)
Exercise 3: Derive BatchNorm gradient as single formula (challenging)
Exercise 4: Put it all together in a training loop (validation)
```

**Why It Matters:** This is deliberate practice. Just like musicians practice scales, ML practitioners should trace gradients by hand at least once. It builds intuition that lasts a career.

---

• *7:04 - 20:00: Backpropagation Through the Loss Function (Exercise 1 - Part 1)*
    1. *"d-lock props will hold the derivative of the loss with respect to all the elements of log props"*
    2. *"D loss by D Lock probs is negative 1 over n in all these places"*
    • *Technical analysis*: Starts backpropagation from the loss. Derives that the gradient for the log probabilities (`dlogprobs`) is `-1/n` at the positions corresponding to the correct labels (indices in `yb`) and zero elsewhere. This is implemented by creating a zero tensor and scattering the `-1/n` values.

### 💡 Intuitive Understanding

**Analogy:** Imagine a student's test with 100 multiple-choice questions. The gradient is the feedback. For each question, the feedback only goes to the answer they chose. If question 5's correct answer was B and they picked B, the feedback is "good job on B." The gradient to all other answers is zero because they weren't selected.

**Mental Model:** The structure of dlogprobs:
```
Forward: loss = -mean(logprobs[range(n), labels])
         We only look at the log prob of the correct class

Backward: dlogprobs is mostly zeros
          dlogprobs[i, labels[i]] = -1/n for each example i

Why -1/n?
- The -1 comes from the negative in the loss
- The 1/n comes from the mean (averaging over n examples)
```

**Why It Matters:** This is your first concrete example of "sparse gradients." Not all paths contribute to the loss, so not all paths get gradients. Understanding where gradients are zero is as important as where they're non-zero.

---

• *20:00 - 33:00: Backpropagation Through Softmax and Normalization*
    1. *"local derivative of log of x is just simply one over X"*
    2. *"addition is a router of gradient whatever gradient comes from above it just gets routed equally"*
    • *Technical analysis*: Works backward through the log operation (`dprobs = dlogprobs / probs`). Then backpropagates through the division for normalization, carefully handling the broadcasting of `countsum_inv`. Highlights the duality between summation in the forward pass (which creates `countsum`) and replication/broadcasting in the backward pass.

### 💡 Intuitive Understanding

**Analogy:** Think of gradients as water flowing backward through pipes. When water flows *forward* through a Y-junction (summation), it combines. When gradients flow *backward* through that same junction, they duplicate—the same gradient goes to both branches.

**Mental Model:** The fundamental duality:
```
Forward Operation | Backward Operation
------------------|-------------------
Sum (many → one)  | Broadcast (one → many copies)
Broadcast         | Sum (many → one)
Multiply          | Multiply by other input
Add               | Identity (pass through)
Log               | Divide by x
```

**Why It Matters:** This duality is the core pattern of backpropagation. Once you internalize it, you can derive gradients for any operation:
- If forward combines values → backward distributes gradients
- If forward distributes values → backward combines gradients

---

• *33:00 - 41:00: Backpropagation Through Logit Stabilization*
    1. *"the only reason we're doing this is for the numerical stability of the softmax"*
    2. *"the gradient on logic masses should be zero right"*
    • *Technical analysis*: Backpropagates through the subtraction of `logitmaxes` (the row-wise max of logits). Shows that the gradient for `logitmaxes` is negligibly small (≈1e-9), confirming that this numerical stability step does not meaningfully affect the loss, as expected.

### 💡 Intuitive Understanding

**Analogy:** Subtracting the max from logits is like translating a picture left or right—the relative positions (which is what softmax cares about) don't change. Since the output doesn't change, the gradient of "how much to shift" should be zero.

**Mental Model:** Mathematical invariance leads to zero gradients:
```
softmax([1, 2, 3]) = softmax([1-3, 2-3, 3-3]) = softmax([-2, -1, 0])

The subtraction doesn't change the output.
Therefore: d(loss)/d(logitmaxes) ≈ 0

The gradient isn't exactly zero due to floating-point, but ≈1e-9.
```

**Why It Matters:** This is a sanity check. When you know something shouldn't affect the output, verify that the gradient is ~0. If it's not, you have a bug. This kind of reasoning is essential for debugging.

---

• *41:00 - 55:00: Backpropagation Through a Linear Layer*
    1. *"the backward Paths of a matrix multiply is a matrix multiply"*
    2. *"I can never remember the formulas... the dimensions have to work out"*
    • *Technical analysis*: Derives gradients for a linear layer (`logits = h @ W2 + b2`). Uses a small 2D example to intuitively arrive at the formulas: `dh = dlogits @ W2.T`, `dW2 = h.T @ dlogits`, `db2 = dlogits.sum(0)`. Emphasizes that dimensional analysis is a reliable way to deduce the correct operations.

### 💡 Intuitive Understanding

**Analogy:** Imagine a linear layer as a switchboard connecting inputs to outputs. Each connection has a weight. The gradient tells you: for each connection, how much did changing that weight affect the loss? The answer depends on how much signal passed through that connection (the activation) and how much the output mattered (the gradient).

**Mental Model:** The matrix calculus cheat sheet:
```
Forward: Y = X @ W + b
         Shape: (N, out) = (N, in) @ (in, out) + (out,)

Backward:
  dX = dY @ W.T        # Shape: (N, in) = (N, out) @ (out, in)
  dW = X.T @ dY        # Shape: (in, out) = (in, N) @ (N, out)
  db = dY.sum(axis=0)  # Shape: (out,) = sum over batch

The key insight: dimensions must match. Use this to derive formulas!
```

**Why It Matters:** Linear layers are everywhere. This gradient pattern is the building block. Once you can derive this without thinking, you can derive gradients for any neural network architecture.

---

• *55:00 - 64:00: Backpropagation Through Tanh and BatchNorm Scaling*
    1. *"d a by DZ ... is just one minus a square"*
    2. *"the correct thing to do is to sum because it's being replicated"*
    • *Technical analysis*: Applies the derivative of tanh: `dhpreact = (1 - h**2) * dh`. Then backpropagates into the BatchNorm gain and bias (`hpreact = bnraw * bngain + bnbias`). Correctly sums gradients over the batch dimension for `bngain` and `bnbias` due to broadcasting in the forward pass.

### 💡 Intuitive Understanding

**Analogy:** The tanh gradient `(1 - h²)` is like a gate that's open in the middle and closed at the edges. When h is near 0, the gate is wide open (gradient ≈ 1). When h is near ±1 (saturated), the gate is almost closed (gradient ≈ 0). Gradients can't flow through closed gates.

**Mental Model:** Tanh saturation visualized:
```
h value:  -1.0   -0.5    0.0    0.5    1.0
(1-h²):    0.0    0.75   1.0    0.75   0.0
           ^               ^              ^
         Closed          Open          Closed
```

For the BatchNorm gain/bias:
```
Forward: hpreact = bnraw * bngain + bnbias
         bngain and bnbias are broadcast across batch

Backward: dbngain = (dpreact * bnraw).sum(batch_dim)
          dbnbias = dpreact.sum(batch_dim)

Broadcasting forward → Summing backward
```

**Why It Matters:** This is where you see why saturated activations kill learning. And why the broadcast/sum duality matters: every shared parameter (like bngain) accumulates gradients from all examples that used it.

---

• *64:00 - 75:00: Backpropagation Through BatchNorm Standardization*
    1. *"anytime you have a sum in the forward pass that turns into a replication or broadcasting in the backward pass"*
    2. *"I'm using the bezels correction dividing by n minus 1 instead of dividing by n"*
    • *Technical analysis*: Manually backpropagates through the batch normalization steps: calculating mean, variance, and the standardized output `bnraw`. Discusses the Bessel's correction (using `n-1` for unbiased variance estimate) and criticizes the train/test mismatch in the original BatchNorm paper.

### 💡 Intuitive Understanding

**Analogy:** BatchNorm's forward pass is like standardizing test scores: subtract the class average, divide by the spread. The backward pass is trickier—you need to figure out: if I want to change the output, how should I change each individual score, given that the average and spread depend on everyone's scores?

**Mental Model:** The dependencies in BatchNorm:
```
Forward:
  mean = x.mean()           # Depends on all x
  var = ((x - mean)**2).mean()  # Depends on all x
  bnraw = (x - mean) / sqrt(var + eps)  # Each output depends on all inputs!

This creates coupling: each input affects the mean and variance,
which affects all outputs. Backprop must account for this.
```

**Why It Matters:** BatchNorm's backward pass is notoriously tricky because of these dependencies. Getting it wrong causes subtle bugs. Deriving it manually is the ultimate test of understanding.

---

• *75:00 - 86:30: Completing Backpropagation Through the Network*
    1. *"we just need to re-represent the shape of those derivatives"*
    2. *"we just need to undo the indexing... gradients that arrive there have to add"*
    • *Technical analysis*: Completes the backward pass through the first linear layer and the embedding lookup. For the embedding, gradients are routed back to the correct rows of the embedding table (`C`) by summing gradients from all batch positions where that embedding was used, implemented via a for-loop.

### 💡 Intuitive Understanding

**Analogy:** The embedding lookup is like students checking out books from a library. Each student (input token) checks out their assigned book (embedding row). During backprop, the feedback goes back to the books—but if the same book was checked out by multiple students, all their feedback gets combined.

**Mental Model:** Embedding backward pass:
```
Forward: emb = C[X]  # Look up rows of C
         If X = [2, 5, 2], we look up C[2], C[5], C[2]

Backward: dC accumulates gradients
          dC[2] += demb[0]  # From first position
          dC[5] += demb[1]  # From second position
          dC[2] += demb[2]  # From third position (adds to existing!)

Key insight: The same row can be looked up multiple times,
so gradients accumulate, they don't replace.
```

**Why It Matters:** Embeddings are fundamental to NLP. Understanding their gradient flow explains why frequently-used words get more gradient updates (more examples contribute), and why rare words learn slower.

---

• *86:30 - 96:30: Analytical Gradient for Cross-Entropy Loss (Exercise 2)*
    1. *"the expression simplify quite a bit... we either end up with... pirai... or P at I minus 1"*
    2. *"the amount to which your prediction is incorrect is exactly the amount by which you're going to get a pull or a push"*
    • *Technical analysis*: Derives the analytical gradient for softmax cross-entropy loss: `dlogits = softmax(logits)`. Then, at the correct class indices, subtracts 1. This is significantly more efficient than backpropagating through each atomic operation. Provides an intuitive interpretation of the gradient as "forces" that pull up the correct probability and pull down incorrect ones proportionally to their current probability.

### 💡 Intuitive Understanding

**Analogy:** Imagine a tug-of-war. The correct answer is pulling with a force of "1 - its_current_probability". All wrong answers are pushing with a force equal to their current probabilities. If the model is already confident and correct, the tug is gentle. If it's confident and wrong, the tug is massive.

**Mental Model:** The elegant result:
```
Forward: loss = -log(softmax(logits)[correct_class])

Backward: dlogits = softmax(logits)  # Start with probabilities
          dlogits[correct_class] -= 1  # Subtract 1 at correct position

Example: True class = 2, probs = [0.1, 0.2, 0.7]
         dlogits = [0.1, 0.2, 0.7 - 1] = [0.1, 0.2, -0.3]

         The -0.3 says "push logits[2] UP" (it's the correct class)
         The 0.1 and 0.2 say "push those DOWN" (they're wrong)
```

**Why It Matters:** This is one of the most beautiful results in deep learning. The gradient has an intuitive interpretation: "increase the correct class, decrease wrong classes, proportionally to current errors." And it's implemented in one line instead of 20.

---

• *96:30 - 110:00: Analytical Gradient for BatchNorm (Exercise 3)*
    1. *"we are going to consider it as a glued single mathematical expression and back propagate through it in a very efficient manner"*
    2. *"this is a whole exercise by itself because you have to consider the fact that this formula here is just for a single neuron"*
    • *Technical analysis*: Presents the complex, multi-step derivation of the BatchNorm backward pass using calculus on paper. The final vectorized implementation condenses the operation into a single, efficient line of code that handles broadcasting across all features and examples in the batch. The result matches PyTorch's gradient.

### 💡 Intuitive Understanding

**Analogy:** Deriving the BatchNorm gradient is like solving a complicated mechanical puzzle. Each piece connects to others. But once solved, the answer is surprisingly compact—like a folded origami that started as a full sheet of paper.

**Mental Model:** The BatchNorm backward formula (simplified):
```
Given: y = (x - μ) / σ
Where μ and σ depend on x (batch statistics)

The gradient involves:
1. Direct effect: d(loss)/dy * dy/dx (treating μ, σ as constant)
2. Indirect effect through μ: d(loss)/dy * dy/dμ * dμ/dx
3. Indirect effect through σ: d(loss)/dy * dy/dσ * dσ/dx

Final formula (vectorized):
dx = (1/σ) * (dy - dy.mean() - y * (dy*y).mean())

This one line replaces ~10 lines of step-by-step backprop!
```

**Why It Matters:** Fused kernels like this are how production code runs fast. Understanding the derivation lets you create efficient custom operations and verify that frameworks are correct.

---

• *110:00 - 115:30: Full Training Loop with Manual Gradients (Exercise 4)*
    1. *"we've really gotten to Lost That backward and we've pulled out all the code and inserted it here"*
    2. *"you can count yourself as one of these buff doji's on the left"*
    • *Technical analysis*: Assembles the manually derived backward passes into a complete training loop, replacing `loss.backward()` and `optimizer.step()`. The model achieves identical performance to the autograd version, proving the correctness of the manual implementation. The final backward pass code is compact (~20 lines), demonstrating a clear understanding of the entire gradient flow.

### 💡 Intuitive Understanding

**Analogy:** This is the final exam—running the full training loop manually proves you understand every step. It's like a chef who can make a complex dish not just by following a recipe, but by understanding why each ingredient is added.

**Mental Model:** The complete training loop structure:
```python
for step in range(max_steps):
    # Forward pass
    emb = C[Xb]
    h = tanh(emb @ W1 + b1)  # With BatchNorm if used
    logits = h @ W2 + b2
    loss = cross_entropy(logits, Yb)

    # Backward pass (manual, no loss.backward())
    dlogits = softmax(logits)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dh = dlogits @ W2.T
    # ... continue backward through all layers

    # Update (manual, no optimizer.step())
    for p, dp in zip(parameters, gradients):
        p.data -= learning_rate * dp
```

**Why It Matters:** You've now implemented what autograd does. You can:
- Debug gradient issues at any layer
- Create custom backward passes
- Understand performance optimizations
- Contribute to deep learning frameworks

---

*Conclusion*
• *Summary of key technical takeaways*: Backpropagation is a systematic application of the chain rule at the tensor level. Key patterns include the duality between summation and broadcasting, the simplicity of linear layer gradients (`dY/dX` involves transposing the other matrix), and the efficiency of using analytically derived gradients for complex blocks like softmax-cross-entropy and BatchNorm.
• *Practical applications*: This deep understanding enables effective debugging of gradient-related issues (e.g., vanishing gradients, implementation bugs), allows for customization of gradient flow (e.g., gradient clipping, custom layers), and builds intuition for how architectural choices affect learning dynamics.
• *Long-term recommendations*: While using autograd is standard practice, practitioners should internalize the mechanics of backpropagation. This foundational knowledge is crucial for innovating new architectures, optimizing training stability, and diagnosing model failures. The exercise solidifies the transition from seeing neural networks as a black box to understanding them as a composable, differentiable computational graph.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Gradient Routing:** In the expression `y = a + b`, if `dy = 5`, what are `da` and `db`? What if `y = a * b` where `a=2, b=3`?

2. **Sparse Gradients:** Explain why `dlogprobs` is mostly zeros. What determines which positions are non-zero?

3. **The Duality:** If the forward pass computes `y = x.sum()`, what does the backward pass look like? What about `y = x.broadcast_to(shape)`?

4. **Saturation Check:** A layer has tanh activation. You observe that 90% of the outputs are in the range [-0.1, 0.1]. Is this good or bad for gradient flow? What about if 90% are in [-1.0, -0.9] ∪ [0.9, 1.0]?

5. **Cross-Entropy Beauty:** Why is the gradient `softmax(logits) - one_hot(labels)` so elegant? What does each term contribute?

6. **Embedding Accumulation:** If a token appears 10 times in a batch, how does this affect its embedding update compared to a token that appears once?

### Coding Challenges

1. **Verify Your Understanding:**
   ```python
   # Implement backward pass for this forward pass:
   def forward(x, W, b):
       z = x @ W + b       # linear
       a = torch.tanh(z)   # activation
       return a.sum()      # loss

   def backward(x, W, b, a, z):
       # Fill in:
       # dW = ?
       # db = ?
       # dx = ?
       pass
   ```
   Compare your gradients to PyTorch's autograd.

2. **Cross-Entropy Gradient:**
   ```python
   def cross_entropy_backward(logits, targets):
       """Return dlogits for cross-entropy loss."""
       # Implement the elegant one-liner
       pass
   ```

3. **BatchNorm Backward from Scratch:**
   ```python
   class BatchNorm1d:
       def forward(self, x, training=True):
           # Save what you need for backward
           pass

       def backward(self, dout):
           # Implement the full backward pass
           # Return dx, dgamma, dbeta
           pass
   ```

4. **Gradient Checker:**
   ```python
   def gradient_check(f, x, dx_analytical, eps=1e-5):
       """Compare analytical gradient to numerical gradient."""
       dx_numerical = np.zeros_like(x)
       for i in range(x.size):
           # Compute (f(x+eps) - f(x-eps)) / (2*eps)
           pass
       # Compare dx_analytical and dx_numerical
       pass
   ```
   Use this to verify all your backward implementations.

5. **Full Manual Training:**
   - Take any small network you've built
   - Replace `loss.backward()` with your own backward pass
   - Replace `optimizer.step()` with manual parameter updates
   - Verify the training curve matches the autograd version

### Reflection

- The video mentions that 10 years ago, manual backpropagation was standard. Research what changed. What was the first widely-used autograd framework? How did the field transition?

- Consider the trade-off: analytical gradients (like the cross-entropy formula) are more efficient but harder to derive. Step-by-step gradients are easier to implement but slower. When would you choose one over the other in a production system?

- The gradient for cross-entropy is `softmax(logits) - one_hot(labels)`. This means if the model is 100% confident and correct, the gradient is zero. Is this desirable? What are the implications for training dynamics?

- Implement gradient checkpointing: a technique where you recompute activations during backward instead of storing them. When is this useful? Implement it for a simple network and measure memory savings.
