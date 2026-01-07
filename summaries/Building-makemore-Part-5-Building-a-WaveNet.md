---
title: Building makemore Part 5: Building a WaveNet
tags:
  - hierarchical language model
  - WaveNet-inspired architecture
  - character-level modeling
  - PyTorch-like modular design
  - neural network layers
  - tensor shape manipulation
  - batch normalization debugging
  - progressive context fusion
  - sequence modeling
  - autoregressive models
  - dilated causal convolutions
  - model scaling
  - deep learning development workflow
  - loss visualization
  - learning rate decay
  - experimental harness
---

*Introduction*
• *Key objectives*: Implement a hierarchical, WaveNet-inspired character-level language model to improve upon a simple multi-layer perceptron. Transition from a custom neural network library to a PyTorch-like modular design.
• *Core themes and methodologies*: Building neural networks from modular layers (embeddings, linear, batch norm, activation, flatten). Progressively fusing character context in a tree-like structure. Debugging and refining layer implementations (especially batch norm). Analyzing tensor shapes and computational graphs.
• *Target audience*: Individuals with foundational knowledge in neural networks and PyTorch, interested in understanding low-level implementation, architecture design, and the development process of deep learning models.

*Detailed Analysis*

• **0:00 - 1:38: Introduction and Motivation**
    1. "we would like to take more characters in a sequence as an input not just three"
    2. "we're actually going to arrive at something that looks very much like a wavenet"
    *Technical analysis*: The lecturer sets the goal: scale the context window and move from a single hidden layer to a deeper, hierarchical model that fuses information progressively, mimicking the WaveNet architecture for efficiency and performance.

### 💡 Intuitive Understanding

**Analogy:** Imagine reading a word by first recognizing letter pairs ("th", "at"), then combining pairs into syllables ("that"), then syllables into words. This is hierarchical processing—you don't try to read all letters at once. WaveNet applies this same principle to sequences.

**Mental Model:** The progression:
```
MLP approach:     [c1, c2, c3, c4, c5, c6, c7, c8] → flatten → one big prediction

WaveNet approach: [c1, c2] [c3, c4] [c5, c6] [c7, c8]  Level 1: pair fusion
                     ↓        ↓        ↓        ↓
                  [p12]    [p34]    [p56]    [p78]     Level 2: quad fusion
                     ↓        ↓        ↓        ↓
                  [q1234]            [q5678]           Level 3: octet fusion
                           ↓
                      [final prediction]
```

**Why It Matters:** Hierarchical processing is computationally efficient (fewer parameters) and learns meaningful intermediate representations. It's the core insight behind many modern architectures.

---

• **1:41 - 6:53: Code Refactoring and PyTorch-ification**
    1. "we want to think of these modules as building blocks and like a Lego building block bricks"
    2. "let's create layers for these and then we can add those layers to just our list"
    *Technical analysis*: The code is refactored to increase modularity. Custom `Embedding` and `Flatten` layers are created, and a `Sequential` container is implemented to manage layers. This mirrors PyTorch's `torch.nn` API, simplifying the forward pass and improving code organization.

### 💡 Intuitive Understanding

**Analogy:** Before: writing an essay as one giant paragraph. After: organizing into sections, each with a clear purpose. The content is the same, but the structure makes it easier to read, modify, and reuse.

**Mental Model:** The modular design pattern:
```python
# Old way: everything tangled together
emb = C[X]
h = emb.view(B, -1) @ W1 + b1
h = torch.tanh(h)
logits = h @ W2 + b2

# New way: composable LEGO blocks
model = Sequential([
    Embedding(vocab_size, emb_dim),
    Flatten(),
    Linear(emb_dim * context, hidden),
    Tanh(),
    Linear(hidden, vocab_size)
])
logits = model(X)
```

**Why It Matters:** Once you have modular layers, you can:
- Swap components easily (try ReLU instead of Tanh)
- Stack layers to make deeper networks
- Debug one layer at a time
- Reuse code across projects

This is exactly how PyTorch's `nn.Module` system works.

---

• **6:58 - 9:16: Visualization and Debugging**
    1. "we need to average up some of these values to get a more sort of representative value"
    2. "we see that we basically made a lot of progress and then here this is the learning rate decay"
    *Technical analysis*: Demonstrates practical debugging and visualization. A rolling average is applied to the noisy loss curve for better interpretation, revealing the impact of learning rate decay on optimization convergence.

### 💡 Intuitive Understanding

**Analogy:** Raw loss values are like looking at individual raindrops. A rolling average is like watching the rain level in a bucket—it shows the overall trend without the noise.

**Mental Model:** Smoothing the loss curve:
```python
# Raw loss: noisy, hard to interpret
losses = [2.5, 2.3, 2.7, 2.4, 2.2, 2.6, ...]

# Rolling average: smooth, shows trends
window = 100
smooth = [mean(losses[i:i+window]) for i in range(len(losses)-window)]

# What you're looking for:
# - Overall downward trend (learning is happening)
# - Plateaus (might need learning rate decay)
# - Sudden jumps (might be bugs or learning rate too high)
```

**Why It Matters:** You can't improve what you can't see. Good visualization practices are essential for debugging and hyperparameter tuning. The loss curve tells a story about your training run.

---

• **9:19 - 17:02: Implementing Hierarchical Structure**
    1. "we don't want to Matrix multiply 80... immediately instead we want to group these"
    2. "we want this to be a 4 by 4 by 20 where basically every two consecutive characters are packed"
    *Technical analysis*: The core architectural shift. Instead of flattening all context characters at once, they are grouped (e.g., in pairs). A new `FlattenConsecutive` layer is created to reshape the tensor (e.g., `[B, T, C]` to `[B, T//n, C*n]`), allowing linear layers to fuse small groups of characters in parallel across a new "batch" dimension.

### 💡 Intuitive Understanding

**Analogy:** Imagine organizing a library. The MLP approach throws all books in one pile and tries to understand them all at once. The hierarchical approach first sorts books by shelf, then understands each shelf, then understands sections, then the whole library.

**Mental Model:** The FlattenConsecutive operation:
```
Input: [Batch, 8 chars, 10 emb_dims]
       Shape: [32, 8, 10]

FlattenConsecutive(2):
  - Group every 2 consecutive positions
  - Concatenate their embeddings

Output: [Batch, 4 groups, 20 dims]
        Shape: [32, 4, 20]

Each position in the output represents 2 characters.
Now a linear layer can learn "bigram features".
```

**Why It Matters:** This is the key architectural innovation. By fusing information gradually:
- Each layer has fewer inputs (20 instead of 80)
- Intermediate representations are meaningful (bigrams, quadgrams)
- The model can learn compositional structure

---

• **17:02 - 20:51: Baseline Performance and Bug Discovery**
    1. "simply scaling up the context length from 3 to 8 gives us a performance of 2.02"
    2. "the batch norm is not doing what we need what we wanted to do"
    *Technical analysis*: Increasing context improves performance, establishing a baseline. A critical bug is found: the custom `BatchNorm1d` layer was incorrectly handling 3D inputs by only normalizing over the first dimension, not treating the new grouping dimension as part of the batch. This led to independent statistics per position, reducing stability.

### 💡 Intuitive Understanding

**Analogy:** Imagine grading tests. The correct way: grade all students' answers to question 1 on a curve together. The bug: grade each student's question 1 separately (no curve at all). The bug means each "position" in the sequence gets its own statistics, which is noisy and wrong.

**Mental Model:** The BatchNorm bug:
```
Input shape: [32 batch, 4 positions, 20 features]

Wrong approach:
  - Compute mean/std over dim=0 only
  - Each position (4 of them) gets its own stats
  - Stats computed from only 32 samples each

Correct approach:
  - Compute mean/std over dims=(0, 1)
  - All positions pooled together
  - Stats computed from 32 * 4 = 128 samples
  - Much more stable!
```

**Why It Matters:** This is a classic deep learning bug: everything *runs* but gives suboptimal results. The only way to catch it is to think carefully about what each layer *should* be doing. Shape matching isn't enough—semantics matter.

---

• **20:51 - 24:33: Fixing BatchNorm and Final Results**
    1. "the dimension we want to reduce over is either 0 or the Tuple zero and one depending on the dimensionality"
    2. "we went from 2.029 to 2.022... we're estimating them using 32 times 4 numbers"
    *Technical analysis*: The `BatchNorm1d` is fixed to normalize over all dimensions except the channel dimension (e.g., dims=(0,1) for 3D input). This correctly pools statistics across the batch and the intra-example groups, leading to more stable estimates and a slight performance improvement.

### 💡 Intuitive Understanding

**Analogy:** The fix is like ensuring you have enough data for a reliable survey. Polling 32 people is okay, but polling 128 people (by combining responses across positions) gives much more reliable statistics.

**Mental Model:** The fixed BatchNorm:
```python
class BatchNorm1d:
    def forward(self, x):
        # Determine reduction dims based on input shape
        if x.dim() == 2:
            dim = (0,)      # [Batch, Features] → reduce over Batch
        elif x.dim() == 3:
            dim = (0, 1)    # [Batch, Seq, Features] → reduce over Batch and Seq

        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True)
        x_norm = (x - mean) / (var + eps).sqrt()
        return self.gamma * x_norm + self.beta
```

**Why It Matters:** A 0.007 improvement in loss might seem small, but in language modeling, small gains compound. More importantly, the model is now behaving as intended—stable statistics lead to stable training, especially as models get deeper.

---

• **24:33 - 28:10: Scaling Up and Concluding Insights**
    1. "we are now getting validation performance of 1.993 so we've crossed over the 2.0 territory"
    2. "the use of convolutions is strictly for efficiency it doesn't actually change the model we've implemented"
    *Technical analysis*: Scaling the model (embedding size, hidden units) yields further gains. The lecturer explains that the implemented hierarchical structure is functionally equivalent to a dilated causal convolutional network (like WaveNet); convolutions are an implementation optimization that reuses computations across the sequence.

### 💡 Intuitive Understanding

**Analogy:** Imagine you're building the same wall using two methods: (1) placing each brick by hand, or (2) using a machine that places 10 bricks at once. The wall is identical, but method 2 is faster. Convolutions are the machine—they compute the same thing as our FlattenConsecutive + Linear, but more efficiently.

**Mental Model:** Convolutions as efficient grouping:
```
Our implementation:
  x = FlattenConsecutive(x, 2)  # Group pairs
  x = Linear(x)                  # Transform each group

This works, but for each group position, we apply Linear separately.

With convolution:
  x = Conv1d(x, kernel_size=2, stride=2)

The Conv1d does the same thing, but GPUs are optimized for it.
The mathematical operation is identical—only the implementation differs.
```

**Why It Matters:** Understanding this equivalence demystifies convolutions. They're not magic—they're just efficient implementations of "apply the same transformation to local groups." WaveNet's dilated convolutions extend this to capture longer-range patterns efficiently.

---

• **28:10 - End: Development Process and Future Directions**
    1. "there's a ton of trying to make the shapes work and there's a lot of gymnastics"
    2. "I very often prototype these layers and implementations in jupyter notebooks"
    *Technical analysis*: Provides meta-commentary on the deep learning development workflow: heavy reliance on documentation (despite its flaws), meticulous shape debugging, prototyping in notebooks, and transferring to code repositories. Outlines future topics: convolutional implementation, residual/skip connections, experimental harnesses, RNNs, and Transformers.

### 💡 Intuitive Understanding

**Analogy:** Building neural networks is like woodworking. The final piece looks elegant, but the process involves lots of measuring (shape debugging), test fits (prototyping), adjustments, and occasionally starting over when something doesn't fit.

**Mental Model:** The practitioner's workflow:
```
1. Sketch the architecture on paper
   - What shapes flow where?
   - What's the receptive field?

2. Prototype in a notebook
   - Create fake input tensors
   - Check shapes at each layer
   - Verify gradients flow

3. Debug with print statements
   - print(x.shape) everywhere
   - Visualize activations/gradients

4. Move to production code
   - Clean up into modules
   - Add tests and validation
   - Build experimental harness
```

**Why It Matters:** This is the real workflow. Nobody writes neural networks correctly the first time. Knowing how to debug shapes, visualize progress, and iterate quickly is what separates productive practitioners from frustrated ones.

---

*Conclusion*
• *Summary of key technical takeaways*: Hierarchical, progressive fusion of context (via grouping and flattening) is a powerful architecture for sequence modeling. Careful tensor shape manipulation is fundamental. Batch normalization must correctly aggregate statistics over all relevant batch dimensions. Modular layer design greatly simplifies building and experimenting with complex networks.
• *Practical applications*: The principles are directly applicable to implementing and understanding modern autoregressive models (WaveNet, Transformers). The debugging process (shape checking, loss visualization, fixing layer states) is essential for real-world model development.
• *Long-term recommendations*: Build a robust experimental harness for hyperparameter tuning. Implement the discussed efficiency optimizations using causal convolutions. Explore advanced architectural components like gated activations and residual connections from the WaveNet paper. Transition to using `torch.nn` directly now that its internal workings are understood.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Hierarchical vs Flat:** Why is `[8 chars] → Linear(80 → 200) → predict` worse than `[8 chars] → group pairs → Linear(20 → 100) → group pairs → Linear(200 → 200) → predict`? Consider both parameter count and learned representations.

2. **BatchNorm Dimensions:** For an input tensor of shape `[32, 4, 20]` (batch, sequence, features), which dimensions should BatchNorm normalize over? Why?

3. **Convolution Equivalence:** Explain how `Conv1d(in=10, out=100, kernel_size=2, stride=2)` is equivalent to `FlattenConsecutive(2)` followed by `Linear(20, 100)`.

4. **Receptive Field:** In a 3-level hierarchical model where each level groups 2 consecutive positions, what is the receptive field of the final output? How does this compare to a single-layer model?

5. **Rolling Average:** Why do we use a rolling average for the loss curve instead of just plotting raw values? What window size would you choose and why?

6. **Context Length Scaling:** The model goes from context=3 to context=8 and improves. Would context=64 be even better? What are the trade-offs?

### Coding Challenges

1. **Implement FlattenConsecutive:**
   ```python
   class FlattenConsecutive:
       def __init__(self, n):
           self.n = n

       def __call__(self, x):
           # x: [B, T, C]
           # output: [B, T//n, C*n]
           pass

       def parameters(self):
           return []
   ```
   Test with various input shapes and verify the output is correct.

2. **Build a 3-Level Hierarchical Model:**
   ```python
   # Input: 8 characters
   # Level 1: Groups of 2 → 4 positions
   # Level 2: Groups of 2 → 2 positions
   # Level 3: Groups of 2 → 1 position
   # Output: prediction for next character

   model = Sequential([
       Embedding(vocab_size, emb_dim),
       # ... fill in the layers
   ])
   ```

3. **Fix the BatchNorm Bug:**
   Start with a broken BatchNorm that only normalizes over dim=0. Then fix it to handle 3D tensors correctly. Measure the performance difference.

4. **Visualize Intermediate Representations:**
   - At each level of the hierarchy, extract and visualize the activations
   - Do the learned representations make sense? (e.g., do nearby characters activate similar patterns?)

5. **Convert to Convolutions:**
   - Rewrite the FlattenConsecutive + Linear pattern using Conv1d
   - Verify that both implementations produce the same output (up to floating-point precision)
   - Compare the speed of both implementations

### Reflection

- WaveNet was originally designed for audio (generating raw waveforms). Research how the dilated convolution pattern allows WaveNet to have a receptive field of thousands of samples while remaining computationally tractable. How does dilation differ from our FlattenConsecutive approach?

- The lecture mentions that the hierarchical structure learns "meaningful intermediate representations." What might these representations correspond to for character-level language modeling? (Think: character → character pairs → short subwords → longer units)

- Modern Transformers use attention instead of hierarchical convolutions. What are the trade-offs? Why did attention "win" for NLP while WaveNet-style architectures remained popular for audio?

- Design an experiment to determine the optimal number of hierarchical levels for a given context length. What metrics would you use? How would you balance model capacity against overfitting?
