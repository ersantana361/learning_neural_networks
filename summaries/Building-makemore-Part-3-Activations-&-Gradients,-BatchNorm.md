---
title: Building makemore Part 3: Activations & Gradients, BatchNorm
tags:
  - neural network initialization
  - activation statistics
  - gradient statistics
  - batch normalization
  - Kaiming initialization
  - vanishing gradients
  - tanh saturation
  - weight scaling
  - training stability
  - update-to-data ratio
  - deep network training
  - activation normalization
  - gradient flow
  - running statistics
  - learnable scale and shift
  - training diagnostics
  - PyTorch modules
---

*Introduction*
• *Key objectives:* To analyze neural network initialization, understand activation and gradient statistics, and introduce batch normalization as a tool for stabilizing deep network training.
• *Core themes and methodologies:* Diagnosing improper weight initialization, visualizing activation/gradient distributions, applying Kaiming initialization principles, and implementing a batch normalization layer.
• *Target audience:* Individuals with foundational knowledge of neural networks and backpropagation, looking to understand the practical challenges of training deeper models.

*Detailed Analysis*

• **0:00 - Introduction & Motivation**
    • *Excerpts:*
        • "we have to stick around the level of multilayer perceptron... to have a very good intuitive understanding of the activations... and the gradients"
        • "the key to understanding why [RNNs] are not optimizable easily is to understand the activations and the gradients"
    • *Technical analysis and implications:* Establishes that internal state analysis is critical for understanding optimization difficulties in complex architectures like RNNs. This foreshadows the need for careful initialization and normalization.

### 💡 Intuitive Understanding

**Analogy:** Think of training a neural network like passing a message through a chain of people. Each person (layer) slightly modifies the message. If everyone whispers too softly (vanishing gradients) or shouts too loudly (exploding gradients), the original meaning gets lost. Understanding activations and gradients is like checking that each person in the chain is speaking at just the right volume.

**Mental Model:** Imagine each layer as a pipe in a water system. Activations are the water flowing forward; gradients are the pressure signals flowing backward. If any pipe is clogged (saturated neurons) or too narrow (vanishing gradients), the system fails. This lecture teaches you how to be a "neural network plumber."

**Why It Matters:** Most neural network debugging comes down to understanding what's happening inside. Networks don't fail obviously—they just learn slowly or get stuck. The skills learned here are how practitioners diagnose real problems.

---

• **4:19 - Diagnosing Improper Initialization**
    • *Excerpts:*
        • "the network is very confidently wrong... record very high loss"
        • "we want the logits to be roughly zero... at initialization"
    • *Technical analysis and implications:* Shows that extreme, uncalibrated logits from the final layer lead to a high initial loss. The expected loss for a uniform distribution over 27 classes is ~3.29 (negative log of 1/27). A much higher initial loss indicates poorly scaled weights, forcing the early optimization to merely squash weights instead of learning useful features.

### 💡 Intuitive Understanding

**Analogy:** Imagine you're playing a guessing game with 27 possible answers. A smart starting strategy is to say "I have no idea—each answer is equally likely" (1/27 probability each). But if you start by confidently declaring one specific answer, and you're almost certainly wrong, you look foolish. That's what a network does with bad initialization—it starts with strong, wrong opinions instead of humble uncertainty.

**Mental Model:** The expected initial loss is like a calibration checkpoint:
- 27 classes → expected loss ≈ -log(1/27) ≈ 3.29
- If initial loss >> 3.29, the network is "overconfident and wrong"
- If initial loss ≈ 3.29, the network starts with appropriate uncertainty

**Why It Matters:** The first few steps of training shouldn't be spent "unlearning" bad initial predictions. They should be spent learning useful patterns. A "hockey stick" loss curve (high loss that drops sharply) indicates wasted computation at the start.

---

• **8:54 - Fixing Output Layer Initialization**
    • *Excerpts:*
        • "B2 is just... zero at initialization"
        • "scale down W2 by 0.1"
    • *Technical analysis and implications:* Zero-initializing the final bias and scaling the final weight matrix ensures the softmax input (logits) starts near zero. This yields the expected uniform prediction and eliminates the initial "hockey stick" loss curve, leading to more efficient training and slightly improved final loss.

### 💡 Intuitive Understanding

**Analogy:** If logits are like "confidence scores" for each class, starting them all at zero is like saying "I'm equally unsure about everything." Starting them at random large values is like randomly shouting wrong answers with high confidence.

**Mental Model:** For classification tasks:
- Logits → Softmax → Probabilities
- If all logits ≈ 0, softmax produces uniform probabilities
- The scaling factor (0.1, 0.01) compresses the logits toward zero

**Why It Matters:** This is a specific trick: always think about what your output layer should produce at initialization. For classification, uniform predictions. For regression to mean 0, zero output. Match the initialization to the expected "I don't know yet" state.

---

• **13:00 - Analyzing Hidden Layer Saturation**
    • *Excerpts:*
        • "many of the elements are one or negative one"
        • "if the outputs of your tanh are very close to 1... you're going to get a zero... killing the gradient"
    • *Technical analysis and implications:* Pre-activations that are too large cause tanh neurons to saturate, residing in flat regions where the local gradient (1 - t²) approaches zero. This impedes gradient flow during backpropagation, slowing or preventing learning in affected neurons.

### 💡 Intuitive Understanding

**Analogy:** Tanh is like a thermostat. In the middle range (around 0), small changes in input cause noticeable changes in output—the system is responsive. But at the extremes (near ±1), the thermostat is "maxed out"—pushing harder doesn't change anything. The gradient is zero because there's nowhere left to go.

**Mental Model:** Visualize the tanh curve:
```
    1 ──────────────╭─────────  ← Saturated (gradient ≈ 0)
                   ╱
    0 ────────────╱───────────  ← Active region (gradient > 0)
                 ╱
   -1 ──────────╯─────────────  ← Saturated (gradient ≈ 0)
```
The steep middle region is where learning happens. The flat tails are "dead zones."

**Why It Matters:** Saturated neurons are effectively "off" during that training step—they receive no gradient signal. If many neurons saturate, the network learns very slowly. This is the core problem that proper initialization and batch normalization solve.

---

• **24:07 - Fixing Hidden Layer Initialization**
    • *Excerpts:*
        • "H preact is too far off from zero... we want this preactivation to be closer to zero"
        • "multiply everything by 0.1"
    • *Technical analysis and implications:* Scaling down the weights (`W1`) feeding into the hidden layer reduces the magnitude of pre-activations. This prevents saturation, ensures gradients can flow effectively, and improves the final validation loss.

### 💡 Intuitive Understanding

**Analogy:** If the pre-activation is the "input dial" to tanh, and the dial is turned way past the useful range, the signal is clipped. Scaling down the weights is like installing a gentler dial that stays in the useful range.

**Mental Model:** Chain of effects:
1. Smaller weights → smaller pre-activations
2. Smaller pre-activations → tanh stays in active region
3. Active tanh → non-zero gradients
4. Non-zero gradients → neurons can learn

**Why It Matters:** This manual scaling (×0.1) works but feels arbitrary. The natural question is: "Is there a principled way to choose this scaling?" That's exactly what Kaiming initialization provides.

---

• **28:01 - Introducing Kaiming Initialization**
    • *Excerpts:*
        • "how do we scale these W's to preserve... [a] distribution to remain Gaussian?"
        • "you are supposed to divide by the square root of the fan-in"
    • *Technical analysis and implications:* For a linear layer with Gaussian input (mean 0, std 1), multiplying by weights with std `1/sqrt(fan_in)` preserves the output's standard deviation at 1. This is the core principle of variance-preserving initialization.

### 💡 Intuitive Understanding

**Analogy:** Imagine you're mixing paint from multiple cans. Each can contributes some color. If you have more cans (higher fan-in), each individual contribution should be smaller to avoid making the result too saturated. The `1/sqrt(fan_in)` rule is the mathematical formula for "how much each can should contribute."

**Mental Model:** The math behind it:
```
y = sum(w_i * x_i) for i = 1 to fan_in

If x_i ~ N(0, 1) and w_i ~ N(0, σ²)
Then y ~ N(0, fan_in * σ²)

To get y ~ N(0, 1), we need:
fan_in * σ² = 1
σ = 1/sqrt(fan_in)
```

**Why It Matters:** This transforms initialization from guesswork to mathematics. Given any layer's input size, you know exactly how to scale the weights to maintain healthy activation magnitudes throughout the network.

---

• **32:34 - Kaiming Initialization for Nonlinearities**
    • *Excerpts:*
        • "for the case of ReLU... you have to compensate... with a gain"
        • "they find that... standard deviation is sqrt(2 / fan_in)"
    • *Technical analysis and implications:* Nonlinearities like ReLU (which zeros out half the distribution) and tanh (a contracting function) distort the variance. A *gain* factor (e.g., `sqrt(2)` for ReLU, `5/3` for tanh) must be applied to the weight initialization to compensate and maintain stable activations throughout the network.

### 💡 Intuitive Understanding

**Analogy:** If the linear layer is a pipe that preserves water flow, the nonlinearity is like a valve. ReLU blocks half the flow (negative values become zero). Tanh squishes the flow (outputs are compressed to [-1,1]). The "gain" is like widening the pipe before the valve to compensate for what the valve will block.

**Mental Model:** Common gain values:
| Nonlinearity | Gain | Reasoning |
|--------------|------|-----------|
| Linear/Identity | 1.0 | No distortion |
| ReLU | √2 ≈ 1.41 | Half the values zeroed |
| Leaky ReLU | √(2/(1+α²)) | Less zeroing |
| Tanh | 5/3 ≈ 1.67 | Compression effect |
| Sigmoid | 1.0 | Usually handled specially |

**Why It Matters:** This is why PyTorch's `torch.nn.init.kaiming_normal_` takes a `nonlinearity` argument. The initialization must be matched to the activation function that follows.

---

• **40:48 - Introducing Batch Normalization**
    • *Excerpts:*
        • "why not take the hidden States and... just normalize them to be Gaussian?"
        • "it's a perfectly differentiable operation"
    • *Technical analysis and implications:* Batch Norm (BN) explicitly standardizes a layer's inputs to have zero mean and unit variance *per batch*. This forcefully creates the desired activation distribution, making the network less sensitive to initial weight scaling.

### 💡 Intuitive Understanding

**Analogy:** Proper initialization is like carefully packing luggage so nothing shifts during travel. Batch normalization is like having a luggage handler who automatically repacks everything at each checkpoint. Even if things shifted, they get corrected. It's a more robust (but more complex) solution.

**Mental Model:** Batch Norm operation:
```
Input: x (batch of activations)

1. μ = mean(x, dim=batch)     # Per-feature mean
2. σ² = var(x, dim=batch)     # Per-feature variance
3. x̂ = (x - μ) / √(σ² + ε)    # Standardize
4. y = γ * x̂ + β              # Scale and shift (learnable)

Output: y (normalized activations)
```

**Why It Matters:** Batch Norm was a breakthrough in 2015 that enabled training much deeper networks. It makes the "internal covariate shift" problem less severe—each layer sees inputs with stable statistics regardless of what previous layers are doing.

---

• **46:19 - Batch Normalization Scale and Shift**
    • *Excerpts:*
        • "we have to also introduce this additional component... scale and shift"
        • "BN gain will multiply... and the BN bias will offset"
    • *Technical analysis and implications:* The learnable parameters `gamma` (scale) and `beta` (shift) after normalization restore the network's expressive power. Without them, the normalized distribution would be fixed, limiting the model's capacity.

### 💡 Intuitive Understanding

**Analogy:** Normalization without scale/shift is like forcing everyone to speak at exactly the same volume and pitch—no expression. The learnable γ and β allow the network to learn "actually, this layer works better with slightly different statistics" while still benefiting from the standardization step.

**Mental Model:** The identity transform is recoverable:
- If γ learns to equal σ and β learns to equal μ
- Then the output equals the input: no normalization effect
- This means BN can "turn itself off" if that's optimal
- But typically, γ and β learn values close to 1 and 0

**Why It Matters:** This is a key design principle: when adding constraints, always include learnable parameters that can undo the constraint if needed. This way, the model capacity is preserved, and the constraint only helps if it helps.

---

• **52:51 - Batch Norm's Side Effects and Inference**
    • *Excerpts:*
        • "the examples in the batch are coupled mathematically"
        • "at test time we are going to fix [mean and std] and use them during inference"
    • *Technical analysis and implications:* BN introduces dependency between examples in a batch, which acts as a regularizer. For inference, running estimates of mean and variance (`running_mean`, `running_var`) are used instead of batch statistics, enabling evaluation on single examples.

### 💡 Intuitive Understanding

**Analogy:** During training, batch norm is like grading on a curve—your score depends on how others in your batch performed. This adds some randomness (regularization) since your curve-mates vary. During inference, you're graded against a fixed historical average—no dependence on who else is being evaluated.

**Mental Model:** Training vs. Inference:
```
Training:
  - Use batch statistics (μ_batch, σ_batch)
  - Update running averages: μ_running ← 0.9*μ_running + 0.1*μ_batch
  - Coupling between examples = regularization

Inference:
  - Use frozen running statistics (μ_running, σ_running)
  - Each example processed independently
  - Deterministic behavior
```

**Why It Matters:** This dual behavior is why batch norm layers have a `.train()` and `.eval()` mode. Forgetting to switch modes is a common bug: training accuracy looks fine, but test accuracy is wrong because stale running statistics are used (or vice versa).

---

• **60:56 - Practical Details of Batch Norm**
    • *Excerpts:*
        • "Epsilon is... preventing a division by zero"
        • "biases now are actually useless because... whatever bias you add here is going to get subtracted"
    • *Technical analysis and implications:* A small `epsilon` stabilizes training. When using BN after a linear/conv layer, the bias term in that preceding layer becomes redundant and should be omitted, as its effect is canceled by the subsequent subtraction of the mean.

### 💡 Intuitive Understanding

**Analogy:** The epsilon is like a safety net. If a feature has zero variance (constant value), dividing by σ=0 would explode. The epsilon (typically 1e-5) ensures you're dividing by "almost zero" instead, which is just a very large number instead of infinity.

**Mental Model:** Bias elimination:
```
Linear: y = Wx + b
BatchNorm: z = (y - mean(y)) / std(y)

The mean(y) = mean(Wx + b) = mean(Wx) + b
So: z = (Wx + b - mean(Wx) - b) / std(y)
       = (Wx - mean(Wx)) / std(y)

The b cancels out! It's absorbed into the β of batch norm.
```

**Why It Matters:** This is a practical optimization: use `Linear(in, out, bias=False)` before BatchNorm. You save parameters and computation without losing expressivity.

---

• **74:11 - Module-based Implementation & Diagnostics**
    • *Excerpts:*
        • "torchify our code... structure our code into these modules"
        • "looking at the activation statistics both in the forward pass and in the backward pass"
    • *Technical analysis and implications:* Refactoring into modular layers (Linear, BatchNorm1d, Tanh) mimics PyTorch's design, improving code clarity and reusability. Systematic plotting of activation/gradient histograms and the *update-to-data ratio* provides crucial diagnostics for training health.

### 💡 Intuitive Understanding

**Analogy:** Building with modules is like using LEGO bricks instead of sculpting from clay. Each brick (Linear, BatchNorm, Tanh) has a well-defined interface. You can swap, rearrange, and reuse pieces. The diagnostic plots are like X-rays—they let you see inside the model without opening it up.

**Mental Model:** The modular structure:
```
class Layer:
    def __init__(self, ...): # Initialize parameters
    def __call__(self, x):   # Forward pass
    def parameters(self):     # Return learnable params

class Model:
    def __init__(self):
        self.layers = [Linear(...), BatchNorm(...), Tanh(), ...]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Why It Matters:** This structure is exactly how PyTorch's `nn.Module` works. Understanding it here makes the transition to PyTorch natural, and helps you debug when things go wrong.

---

• **99:56 - The Update-to-Data Ratio**
    • *Excerpts:*
        • "what matters... is the update to the data ratio"
        • "should be roughly 1e-3... a good rough heuristic"
    • *Technical analysis and implications:* The ratio `(learning_rate * gradient_std) / weight_std` measures the relative change applied to parameters per step. A log value near -3 (~0.001) indicates a well-calibrated learning rate. Values much higher suggest unstable updates; much lower suggest slow learning.

### 💡 Intuitive Understanding

**Analogy:** Imagine weights as your position on a map, and gradients as suggested directions. The update-to-data ratio asks: "How far am I moving relative to how far I've traveled?" If each step is 1% of your total journey, that's a measured 1e-2 pace. If each step is 50% of your position, you're erratically bouncing around. If it's 0.0001%, you're barely moving.

**Mental Model:** Interpreting the ratio:
```
Ratio ≈ 1e-3 (log ≈ -3): Healthy, typical learning
Ratio ≈ 1e-1 (log ≈ -1): Too large! Unstable training
Ratio ≈ 1e-5 (log ≈ -5): Too small! Slow learning

The ratio should be consistent across layers.
If one layer has ratio << others: it's learning too slowly
If one layer has ratio >> others: it might be unstable
```

**Why It Matters:** This single metric tells you if your learning rate is appropriate. Before training for hours, check the update-to-data ratio on a few batches. It's one of the most useful diagnostics for hyperparameter tuning.

---

*Conclusion*
• *Summary of key technical takeaways:*
    • Proper initialization aims for **zero-mean, unit-variance activations** throughout the network at the start of training.
    • **Kaiming Initialization** (`std = gain / sqrt(fan_in)`) is a principled method to achieve this, with a *gain* specific to the nonlinearity (e.g., 1 for linear, `sqrt(2)` for ReLU, `5/3` for tanh).
    • **Batch Normalization** explicitly enforces this property during training, using batch statistics, learnable scale/shift parameters, and running statistics for inference. It significantly stabilizes training for deep networks.
    • Critical **diagnostics** include monitoring activation/gradient distributions and the update-to-data ratio to identify issues like saturation, vanishing/exploding gradients, and poorly scaled learning rates.

• *Practical applications:*
    • Always use a principled initialization scheme (e.g., Kaiming) for neural network weights.
    • Employ normalization layers (like BatchNorm, LayerNorm) when building deep networks to improve training stability and robustness to initialization.
    • During model development, implement visual diagnostics for activations, gradients, and parameter updates to quickly identify training issues.

• *Long-term recommendations:*
    • While Batch Normalization is historically important and effective, be aware of its drawbacks (batch dependence, complexity). For new projects, consider modern alternatives like **Layer Normalization** or **Group Normalization**, which do not couple batch examples.
    • Understanding activation/gradient flow remains essential for debugging and designing novel architectures, even with advanced optimizers and normalization techniques.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Calibration Check:** If you're building a classifier with 1000 classes, what should the initial loss approximately be? What if it's 10x higher than expected?

2. **Saturation Diagnosis:** You notice that 80% of your tanh outputs are between 0.99 and 1.0. What does this indicate? How would you fix it?

3. **Gain Intuition:** Why does ReLU need a gain of √2 while tanh needs 5/3? What property of each activation causes this?

4. **BatchNorm Modes:** You train a model with BatchNorm and it achieves 95% training accuracy. But when you run inference, accuracy drops to 60%. What likely went wrong?

5. **Bias Redundancy:** Explain why a bias term in a Linear layer followed by BatchNorm is redundant. Where does the bias "go"?

6. **Update Ratio:** Your model's update-to-data ratio is 1e-6 for all layers. What should you do? What if only one layer has this ratio while others are 1e-3?

### Coding Challenges

1. **Initialization Experiment:**
   - Create a 10-layer MLP with tanh activations
   - Compare training with (a) default random init, (b) Xavier init, (c) Kaiming init
   - Plot the activation distributions at each layer for the first batch
   - Compare training loss curves for the first 100 steps

2. **Build BatchNorm from Scratch:**
   ```python
   class BatchNorm1d:
       def __init__(self, dim, eps=1e-5, momentum=0.1):
           # Initialize gamma, beta, running_mean, running_var
           pass

       def __call__(self, x, training=True):
           # Implement the forward pass
           # Handle training vs. inference mode
           pass
   ```
   Verify your implementation matches PyTorch's `nn.BatchNorm1d` output.

3. **Gradient Flow Visualization:**
   - Build a network and capture gradients at each layer during backprop
   - Plot histograms of gradients for each layer on the same figure
   - Identify layers where gradients are vanishing or exploding
   - Add batch normalization and show how the gradient distribution changes

4. **Update-to-Data Ratio Monitor:**
   ```python
   def compute_update_ratio(model, learning_rate):
       # For each parameter, compute:
       # ratio = (learning_rate * grad.std()) / param.data.std()
       pass
   ```
   Create a training loop that logs this ratio every 100 steps and visualizes it over time.

5. **Breaking and Fixing BatchNorm:**
   - Train a model with BatchNorm but "forget" to call `model.eval()` during validation
   - Show the unstable validation loss/accuracy
   - Fix it and show the stable behavior
   - Explain why the behavior differs

### Reflection

- BatchNorm was published in 2015 and was considered a breakthrough. Yet today, Transformers often use LayerNorm instead. Research why LayerNorm became preferred for attention-based models. What properties of self-attention make batch statistics problematic?

- The update-to-data ratio heuristic of 1e-3 comes from experience. Consider: if all layers should have similar ratios, what does this imply about how the learning rate interacts with network depth? Should deeper networks use smaller learning rates?

- Consider this claim: "With BatchNorm, the actual initialization values don't matter much." Is this true? Design an experiment to test the limits of this claim. At what point does initialization become so bad that even BatchNorm can't help?
