---
title: Building makemore Part 4: Becoming a Backprop Ninja
tags:
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

• *3:00 - 7:04: Historical Context and Exercise Overview*
    1. *"about 10 years ago in deep learning this was fairly standard and in fact pervasive"*
    2. *"we're going to keep everything the same so we're still going to have a two layer multiplayer perceptron"*
    • *Technical analysis*: Provides historical perspective, showing that manual gradient calculation was once the norm. Outlines the structure of the upcoming exercises: 1) Backpropagate through the entire broken-down graph, 2) Analytically derive the gradient for the cross-entropy loss, 3) Analytically derive the gradient for batch normalization, 4) Assemble a full training loop with manual gradients.

• *7:04 - 20:00: Backpropagation Through the Loss Function (Exercise 1 - Part 1)*
    1. *"d-lock props will hold the derivative of the loss with respect to all the elements of log props"*
    2. *"D loss by D Lock probs is negative 1 over n in all these places"*
    • *Technical analysis*: Starts backpropagation from the loss. Derives that the gradient for the log probabilities (`dlogprobs`) is `-1/n` at the positions corresponding to the correct labels (indices in `yb`) and zero elsewhere. This is implemented by creating a zero tensor and scattering the `-1/n` values.

• *20:00 - 33:00: Backpropagation Through Softmax and Normalization*
    1. *"local derivative of log of x is just simply one over X"*
    2. *"addition is a router of gradient whatever gradient comes from above it just gets routed equally"*
    • *Technical analysis*: Works backward through the log operation (`dprobs = dlogprobs / probs`). Then backpropagates through the division for normalization, carefully handling the broadcasting of `countsum_inv`. Highlights the duality between summation in the forward pass (which creates `countsum`) and replication/broadcasting in the backward pass.

• *33:00 - 41:00: Backpropagation Through Logit Stabilization*
    1. *"the only reason we're doing this is for the numerical stability of the softmax"*
    2. *"the gradient on logic masses should be zero right"*
    • *Technical analysis*: Backpropagates through the subtraction of `logitmaxes` (the row-wise max of logits). Shows that the gradient for `logitmaxes` is negligibly small (≈1e-9), confirming that this numerical stability step does not meaningfully affect the loss, as expected.

• *41:00 - 55:00: Backpropagation Through a Linear Layer*
    1. *"the backward Paths of a matrix multiply is a matrix multiply"*
    2. *"I can never remember the formulas... the dimensions have to work out"*
    • *Technical analysis*: Derives gradients for a linear layer (`logits = h @ W2 + b2`). Uses a small 2D example to intuitively arrive at the formulas: `dh = dlogits @ W2.T`, `dW2 = h.T @ dlogits`, `db2 = dlogits.sum(0)`. Emphasizes that dimensional analysis is a reliable way to deduce the correct operations.

• *55:00 - 64:00: Backpropagation Through Tanh and BatchNorm Scaling*
    1. *"d a by DZ ... is just one minus a square"*
    2. *"the correct thing to do is to sum because it's being replicated"*
    • *Technical analysis*: Applies the derivative of tanh: `dhpreact = (1 - h**2) * dh`. Then backpropagates into the BatchNorm gain and bias (`hpreact = bnraw * bngain + bnbias`). Correctly sums gradients over the batch dimension for `bngain` and `bnbias` due to broadcasting in the forward pass.

• *64:00 - 75:00: Backpropagation Through BatchNorm Standardization*
    1. *"anytime you have a sum in the forward pass that turns into a replication or broadcasting in the backward pass"*
    2. *"I'm using the bezels correction dividing by n minus 1 instead of dividing by n"*
    • *Technical analysis*: Manually backpropagates through the batch normalization steps: calculating mean, variance, and the standardized output `bnraw`. Discusses the Bessel's correction (using `n-1` for unbiased variance estimate) and criticizes the train/test mismatch in the original BatchNorm paper.

• *75:00 - 86:30: Completing Backpropagation Through the Network*
    1. *"we just need to re-represent the shape of those derivatives"*
    2. *"we just need to undo the indexing... gradients that arrive there have to add"*
    • *Technical analysis*: Completes the backward pass through the first linear layer and the embedding lookup. For the embedding, gradients are routed back to the correct rows of the embedding table (`C`) by summing gradients from all batch positions where that embedding was used, implemented via a for-loop.

• *86:30 - 96:30: Analytical Gradient for Cross-Entropy Loss (Exercise 2)*
    1. *"the expression simplify quite a bit... we either end up with... pirai... or P at I minus 1"*
    2. *"the amount to which your prediction is incorrect is exactly the amount by which you're going to get a pull or a push"*
    • *Technical analysis*: Derives the analytical gradient for softmax cross-entropy loss: `dlogits = softmax(logits)`. Then, at the correct class indices, subtracts 1. This is significantly more efficient than backpropagating through each atomic operation. Provides an intuitive interpretation of the gradient as "forces" that pull up the correct probability and pull down incorrect ones proportionally to their current probability.

• *96:30 - 110:00: Analytical Gradient for BatchNorm (Exercise 3)*
    1. *"we are going to consider it as a glued single mathematical expression and back propagate through it in a very efficient manner"*
    2. *"this is a whole exercise by itself because you have to consider the fact that this formula here is just for a single neuron"*
    • *Technical analysis*: Presents the complex, multi-step derivation of the BatchNorm backward pass using calculus on paper. The final vectorized implementation condenses the operation into a single, efficient line of code that handles broadcasting across all features and examples in the batch. The result matches PyTorch's gradient.

• *110:00 - 115:30: Full Training Loop with Manual Gradients (Exercise 4)*
    1. *"we've really gotten to Lost That backward and we've pulled out all the code and inserted it here"*
    2. *"you can count yourself as one of these buff doji's on the left"*
    • *Technical analysis*: Assembles the manually derived backward passes into a complete training loop, replacing `loss.backward()` and `optimizer.step()`. The model achieves identical performance to the autograd version, proving the correctness of the manual implementation. The final backward pass code is compact (~20 lines), demonstrating a clear understanding of the entire gradient flow.

*Conclusion*
• *Summary of key technical takeaways*: Backpropagation is a systematic application of the chain rule at the tensor level. Key patterns include the duality between summation and broadcasting, the simplicity of linear layer gradients (`dY/dX` involves transposing the other matrix), and the efficiency of using analytically derived gradients for complex blocks like softmax-cross-entropy and BatchNorm.
• *Practical applications*: This deep understanding enables effective debugging of gradient-related issues (e.g., vanishing gradients, implementation bugs), allows for customization of gradient flow (e.g., gradient clipping, custom layers), and builds intuition for how architectural choices affect learning dynamics.
• *Long-term recommendations*: While using autograd is standard practice, practitioners should internalize the mechanics of backpropagation. This foundational knowledge is crucial for innovating new architectures, optimizing training stability, and diagnosing model failures. The exercise solidifies the transition from seeing neural networks as a black box to understanding them as a composable, differentiable computational graph.