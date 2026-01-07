---
title: Let's reproduce GPT-2 (124M)
tags:
  - GPT-2 reproduction
  - model implementation from scratch
  - PyTorch training pipeline
  - performance optimization
  - mixed precision training (BF16)
  - torch.compile
  - Flash Attention
  - distributed training (DDP)
  - gradient accumulation
  - hyperparameter tuning
  - weight tying
  - data loading (FineWeb-Edu)
  - validation and evaluation
  - HellaSwag benchmark
  - model checkpointing
  - hardware utilization (A100)
  - numerical efficiency
  - cosine learning rate schedule
  - gradient clipping
---

*Introduction*
• *Key objectives*: Reproduce the GPT-2 (124M parameter) model from scratch, train it, and surpass its performance using modern techniques and hardware.
• *Core themes and methodologies*: Implementation of the GPT-2 architecture, loading pre-trained weights for verification, data loading, optimization, and performance tuning (mixed precision, compilation, distributed training).
• *Target audience*: Developers and researchers with foundational knowledge of Transformers and PyTorch, interested in the practical details of training large language models.

*Detailed Analysis*

• *0:00 - 3:36: Introduction and Project Scope*
    1. "we are going to reproduce the gpt2 model the 124 million version"
    2. "we're going to try to surpass that model"
    3. *Technical analysis*: Establishes the goal of replicating and improving upon a specific, historical model. Highlights the accessibility of such training today (~$10, 1 hour on cloud compute).

### 💡 Intuitive Understanding

**Analogy:** Reproducing GPT-2 is like building a classic car from scratch using modern parts. The original design is known, but you can use better materials and techniques to build something that performs even better.

**Mental Model:** The cost revolution:
```
2019: GPT-2 training (original)
  - Unknown cost, likely $10K-100K+
  - Specialized hardware access required
  - Months of development

2024: GPT-2 reproduction
  - ~$10 on cloud compute
  - 1 hour on 8x A100 GPUs
  - All techniques publicly known

Democratization of LLM training in action.
```

**Why It Matters:** This lecture proves that cutting-edge ML is accessible. You can train a model that matches 2019's state-of-the-art for the cost of a meal. The barriers are knowledge, not resources.

---

• *3:40 - 13:48: Loading and Inspecting the Target Model*
    1. "load the GPT to 124 M model as it was released"
    2. "these are just the parameters these are weights floats"
    3. *Technical analysis*: Demonstrates loading OpenAI's weights via Hugging Face to establish a ground-truth target. Analysis of positional embeddings shows they are learned, not fixed sinusoids, and can reveal training status (e.g., "undertrained").

### 💡 Intuitive Understanding

**Analogy:** Before building a replica car, you examine the original in detail. How are the parts connected? What do the settings look like? You're establishing what "success" means.

**Mental Model:** Model forensics:
```
Load pretrained GPT-2:
  model = GPT2LMHeadModel.from_pretrained("gpt2")

Inspect weight shapes:
  wte: [50257, 768]    # Token embeddings
  wpe: [1024, 768]     # Position embeddings
  h.0.attn.c_attn: [768, 2304]  # QKV projection

Verify behavior:
  Generate text → should be coherent
  Check loss on known data → establishes target
```

**Why It Matters:** The pretrained model is your "answer key." By loading it and generating text, you know exactly what success looks like. Any deviation in your implementation will be detectable.

---

• *13:48 - 28:01: Implementing the GPT-2 Architecture from Scratch*
    1. "we want to write this from scratch ourselves"
    2. "gpt2 is a decoder only Transformer"
    3. *Technical analysis*: Details architectural deviations from the original Transformer: removal of encoder, pre-normalization (layer norms before attention/MLP), and an added final layer norm. The implementation carefully names modules to align with Hugging Face's key names for easy weight loading.

### 💡 Intuitive Understanding

**Analogy:** Following a blueprint but in a slightly different order. The original Transformer paper put normalization after the block (post-norm). GPT-2 puts it before (pre-norm). Same ingredients, different recipe.

**Mental Model:** GPT-2 vs Original Transformer:
```
Original Transformer (2017):
  x = Attention(x) + x     # Attention, then add residual
  x = LayerNorm(x)         # Norm after
  x = FFN(x) + x
  x = LayerNorm(x)

GPT-2 (Pre-Norm):
  x = LayerNorm(x)         # Norm before
  x = Attention(x) + x     # Then attention + residual
  x = LayerNorm(x)
  x = FFN(x) + x

Pre-norm is more stable for deep networks.
Also: decoder only (no encoder, no cross-attention).
```

**Why It Matters:** These architectural details matter for training stability. Pre-norm makes gradient flow easier. Understanding why these changes were made helps you design your own architectures.

---

• *28:01 - 31:00: Verifying the Custom Implementation*
    1. "we can load the weights and the biases into our module"
    2. "they look sensible and now we want to initialize from scratch"
    3. *Technical analysis*: Successfully ports pre-trained weights into the custom `GPT` class, validating the implementation's correctness. This creates a verified baseline from which to begin training from random initialization.

### 💡 Intuitive Understanding

**Analogy:** After building your replica car, you swap in the original engine to make sure it runs. If it works with the original parts, your chassis is correct. Then you can train your own engine.

**Mental Model:** The verification strategy:
```
1. Build custom GPT class
2. Load pretrained weights from Hugging Face
3. Generate text → should match original model
4. If it works: implementation is correct
5. Now re-initialize randomly and train from scratch

This catches bugs BEFORE you waste compute on training.
```

**Why It Matters:** Debugging neural networks is hard. By loading known-good weights first, you isolate implementation bugs from training bugs. This is a professional practice that saves enormous time.

---

• *31:00 - 46:00: Setting Up Training Infrastructure*
    1. "to train the model we're going to need some data set"
    2. "we want to actually iterate these XY batches"
    3. *Technical analysis*: Implements a basic data loader for the TinyShakespeare dataset. Explains the crucial pattern of creating input/target tensors (`x` and `y`) by offsetting a 1D token sequence, then viewing it as a 2D `(batch, time)` tensor.

### 💡 Intuitive Understanding

**Analogy:** The x/y offset trick is like teaching word prediction with flashcards. Card front: "The cat sat on the". Card back: "cat sat on the mat". The input and target are the same sequence, just shifted by one.

**Mental Model:** Data loading for language models:
```
Token sequence: [A, B, C, D, E, F, G, H, I, J]

Batch with block_size=4:
  x (input):  [[A, B, C, D], [E, F, G, H]]
  y (target): [[B, C, D, E], [F, G, H, I]]

Same data, offset by 1!

One sequence = many training examples.
```

**Why It Matters:** This simple offset trick is how all language model training works. Understanding it clarifies why context windows matter and why training is so efficient (every position is a training example).

---

• *46:00 - 56:00: Initial Training Loop and Loss*
    1. "we want to calculate the loss"
    2. "we expect something around 10.82"
    3. *Technical analysis*: Adds loss calculation (cross-entropy) to the forward pass. Sanity-checks the initial loss against the theoretical value for a uniform output distribution over the vocabulary (~1/50257), confirming proper initialization.

### 💡 Intuitive Understanding

**Analogy:** Before training, the model is guessing randomly among 50,257 possible tokens. The expected loss is -log(1/50257) ≈ 10.82. If initial loss is wildly different, something is wrong with initialization.

**Mental Model:** Initial loss sanity check:
```
Vocab size: 50,257 tokens
Random guess: 1/50257 probability per token

Expected loss = -log(1/50257) = log(50257) ≈ 10.82

If initial loss ≈ 10.82: ✓ Good initialization
If initial loss >> 10.82: ✗ Model is overconfident (bad init)
If initial loss << 10.82: ✗ Something leaking (probably a bug)
```

**Why It Matters:** This simple check catches many initialization bugs. It's a sanity gate before you spend any compute on training.

---

• *56:00 - 66:00: Optimization and Hyperparameter Initialization*
    1. "we're using the AdamW optimizer"
    2. "we are overfitting a single batch"
    3. *Technical analysis*: Sets up a basic training loop with AdamW. Demonstrates successful overfitting on a single batch, a good sanity check. Introduces learning rate (3e-4) and the concept of checking loss convergence.

### 💡 Intuitive Understanding

**Analogy:** Before training on all your data, check if the car can drive at all. Can you overfit one batch? If yes, the engine works. If no, fix bugs before scaling up.

**Mental Model:** The overfitting sanity check:
```
Steps:
1. Pick one small batch
2. Train on only this batch for 50 steps
3. Check: does loss go to ~0?

If yes: model CAN learn, training loop works
If no: bug in forward/backward/optimizer

This takes 30 seconds and catches 90% of bugs.
```

**Why It Matters:** This is THE most important debugging technique. Before training for hours, spend 30 seconds verifying the training loop actually works. Never skip this step.

---

• *66:00 - 74:00: Correcting Implementation Details: Weight Tying and Initialization*
    1. "this is a common weight tying scheme"
    2. "scale the weight of residual layers by factor of one over sqrt(n)"
    3. *Technical analysis*: Correctly implements weight tying between the token embedding and the LM head, saving parameters and adding a useful inductive bias. Applies GPT-2's specific initialization scheme (std=0.02 for weights, zero for biases) and scales residual block outputs to control activation growth.

### 💡 Intuitive Understanding

**Analogy:** Weight tying is like using the same dictionary for looking up words and predicting words. If "cat" is close to "dog" in the embedding, "cat" and "dog" should also be predicted in similar contexts.

**Mental Model:** Weight tying:
```
Without tying:
  wte: [vocab, d_model]  # Token → embedding
  lm_head: [d_model, vocab]  # Embedding → prediction
  Total: 2 × vocab × d_model parameters

With tying:
  wte: [vocab, d_model]
  lm_head = wte.T  # Same weights, transposed
  Total: 1 × vocab × d_model parameters

For GPT-2: saves ~50M parameters!
Also provides useful inductive bias.
```

**Why It Matters:** Weight tying is standard in language models. It reduces parameters AND provides a useful prior: similar embeddings should predict similar tokens.

---

• *74:00 - 85:00: Hardware Utilization and Precision (TF32)*
    1. "deep learning can tolerate significantly lower precisions"
    2. "tf32 is a lower Precision format"
    3. *Technical analysis*: Motivates the use of lower precision to increase computational throughput and reduce memory bandwidth pressure. Enables TensorFloat-32 (TF32) in PyTorch, which offers ~8x potential FLOPs gain on Ampere GPUs by truncating mantissa bits inside tensor core operations.

### 💡 Intuitive Understanding

**Analogy:** Precision is like significant digits in math. For ballpark estimates, 3.14 is enough—you don't need 3.14159265358979. Neural networks are surprisingly tolerant of "rounding."

**Mental Model:** Precision formats:
```
FP32: 1 bit sign, 8 bits exponent, 23 bits mantissa
  - Full precision, slow

TF32: 1 bit sign, 8 bits exponent, 10 bits mantissa
  - Same range as FP32
  - Less precision (10 vs 23 bits)
  - ~8x faster on tensor cores

BF16: 1 bit sign, 8 bits exponent, 7 bits mantissa
  - Same range as FP32
  - Even less precision
  - ~16x faster, half memory
```

**Why It Matters:** Training speed is often bottlenecked by memory bandwidth, not compute. Lower precision = smaller numbers = faster to move around. Modern GPUs are optimized for this.

---

• *85:00 - 100:00: Mixed Precision Training (BF16)*
    1. "we are going to drop down to B float 16"
    2. "our activations have been changed to BF16"
    3. *Technical analysis*: Implements BF16 mixed precision training via `torch.autocast`. BF16 preserves the dynamic range of FP32 (unlike FP16) by keeping the same exponent bits, simplifying training by often avoiding the need for gradient scaling. Parameters remain in FP32.

### 💡 Intuitive Understanding

**Analogy:** BF16 is like shorthand for taking notes. You write faster with abbreviations, but important numbers (model weights) stay in full form. You compute in shorthand, but accumulate in full precision.

**Mental Model:** Mixed precision workflow:
```
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Forward pass in BF16 (fast, small memory)
    logits = model(x)
    loss = F.cross_entropy(logits, y)

# Backward and optimizer step in FP32
loss.backward()  # Gradients computed in mixed precision
optimizer.step()  # Weights updated in FP32

Why BF16 over FP16?
  - Same exponent → same range → no gradient scaling needed
  - Simpler code, fewer NaN issues
```

**Why It Matters:** Mixed precision is ~2x speedup for free. BF16 is the "just works" choice for modern GPUs (A100, H100). Use it.

---

• *100:00 - 115:00: Compilation with `torch.compile`*
    1. "torch. compile is really quite incredible infrastructure"
    2. "this is about 2.3x Improvement"
    3. *Technical analysis*: Uses `torch.compile` to fuse operations and reduce Python overhead. The compiler analyzes the entire model graph, enabling optimizations like kernel fusion (e.g., combining elementwise ops in GELU), which minimizes expensive round-trips to GPU memory (HBM).

### 💡 Intuitive Understanding

**Analogy:** torch.compile is like a smart assistant who watches you work, notices you're making the same trips between your desk and the printer, and suggests: "Why not print all at once?" It fuses operations to minimize data movement.

**Mental Model:** What torch.compile does:
```
Without compile:
  x = a + b     # Load a,b from memory, compute, store x
  y = x * c     # Load x,c from memory, compute, store y
  z = relu(y)   # Load y, compute, store z
  # 6 memory operations!

With compile:
  z = relu((a + b) * c)  # Fused into one kernel
  # 2 memory operations (load inputs, store output)

GPU memory is slow. Fewer trips = faster.
```

**Why It Matters:** torch.compile provides 2-3x speedup with one line of code. It's the "free lunch" of PyTorch optimization. Always use it for training.

---

• *115:00 - 125:00: Flash Attention*
    1. "flash attention is a kernel Fusion operation"
    2. "it never gets read or written to the hbm"
    3. *Technical analysis*: Replaces the standard attention implementation with Flash Attention. This algorithm fuses the attention computation steps and uses techniques like online softmax to avoid materializing the full `(T, T)` attention matrix in high-bandwidth memory (HBM), dramatically speeding up the operation.

### 💡 Intuitive Understanding

**Analogy:** Standard attention is like calculating all pairwise distances in a city, writing them in a giant book, then looking up the ones you need. Flash Attention calculates distances on-demand without the book—faster and uses less paper.

**Mental Model:** Flash Attention memory optimization:
```
Standard Attention:
  Q, K, V: [batch, heads, seq, head_dim]
  Attention matrix: [batch, heads, seq, seq]

  For seq=2048, batch=8, heads=12:
  Attention matrix = 8 × 12 × 2048 × 2048 = 400M floats = 1.6GB!

Flash Attention:
  Never materialize the full attention matrix
  Use "tiling" to compute in small chunks
  Keep intermediate results in fast SRAM (not slow HBM)

  Memory: O(seq) instead of O(seq²)
  Speed: 2-4x faster
```

**Why It Matters:** Flash Attention enables longer context windows and is always faster. It's not optional—it's table stakes for serious LLM training.

---

• *125:00 - 135:00: Numerical Efficiency: "Nice" Tensor Shapes*
    1. "you always want to use nice numbers"
    2. "we are seeing a roughly 4% Improvement"
    3. *Technical analysis*: Pads the vocabulary size from 50257 to 50304 (a multiple of 128). Many CUDA kernels are optimized for dimensions that are multiples of powers of two. Using "ugly" numbers can trigger inefficient boundary-case kernels, while padding leads to more efficient computation despite slightly more FLOPs.

### 💡 Intuitive Understanding

**Analogy:** CUDA kernels are like assembly lines optimized for batches of 128 items. If you send 127 items, the line runs half-empty. Padding to 128 (even with a dummy item) is faster overall.

**Mental Model:** Nice numbers:
```
Bad: 50257 (prime-ish, no factors of 2)
Good: 50304 = 50257 + 47 = 128 × 393

Why multiples of 128?
  - GPU tensor cores work in tiles (typically 16×16, 32×32)
  - Memory alignment (128-byte boundaries)
  - CUDA kernels have optimized fast paths for these sizes

Extra parameters: 47 × 768 = 36K
Speed gain: ~4%
Worth it!
```

**Why It Matters:** This is "free" performance. A tiny amount of padding yields measurable speedups. Always round up to nice numbers.

---

• *135:00 - 155:00: Adopting GPT-3 Hyperparameters*
    1. "we're going to go to gp3 paper to follow along"
    2. "we clip the global Norm of the gradient at 1.0"
    3. *Technical analysis*: Configures optimizer (AdamW betas, epsilon), gradient clipping, and a cosine learning rate schedule with warmup as per the GPT-3 paper. This establishes a robust, modern training configuration. Also implements weight decay only on 2D parameters (embeddings, linear weights).

### 💡 Intuitive Understanding

**Analogy:** GPT-3's hyperparameters are a "known-good recipe" tested at massive scale. Instead of experimenting, copy what's proven to work.

**Mental Model:** GPT-3 hyperparameters:
```
Optimizer: AdamW
  - betas: (0.9, 0.95)  # Momentum and second moment decay
  - eps: 1e-8           # Numerical stability
  - weight_decay: 0.1   # Regularization

Learning rate:
  - Warmup: 375M tokens linear ramp
  - Then: cosine decay to 0.1 × max_lr
  - Max LR: 6e-4 for 124M model

Gradient clipping:
  - clip_grad_norm = 1.0
  - Prevents exploding gradients

Weight decay:
  - Only on 2D tensors (embeddings, linear weights)
  - NOT on biases or LayerNorm
```

**Why It Matters:** These hyperparameters represent hundreds of thousands of dollars of experiments. Use them. Don't reinvent the wheel.

---

• *155:00 - 166:00: Gradient Accumulation*
    1. "we need to use what's called gradient accumulation"
    2. "we are missing the normalizer"
    3. *Technical analysis*: Implements gradient accumulation to simulate a large batch size (0.5M tokens) with limited GPU memory. The key nuance is scaling the loss by `1 / accumulation_steps` to maintain the correct gradient averaging semantics, as `loss.backward()` sums gradients.

### 💡 Intuitive Understanding

**Analogy:** If you can only carry 4 groceries at a time but need 16, you make 4 trips. Gradient accumulation makes multiple passes and combines the gradients before stepping.

**Mental Model:** Gradient accumulation:
```
Target batch size: 0.5M tokens
GPU memory: fits 16K tokens

accumulation_steps = 0.5M / 16K = 32

for step in range(accumulation_steps):
    loss = compute_loss(micro_batch) / accumulation_steps  # Scale!
    loss.backward()  # Gradients accumulate (sum)

optimizer.step()  # Step with accumulated gradients
optimizer.zero_grad()  # Reset for next macro-batch

Key: divide loss by accumulation_steps
  - backward() SUMS gradients
  - We want AVERAGE gradients
  - So pre-divide the loss
```

**Why It Matters:** Large batch sizes are often required for LLM training (stability, efficiency). Gradient accumulation lets you achieve them regardless of GPU memory.

---

• *166:00 - 185:00: Distributed Data Parallel (DDP) Training*
    1. "they are going to collaborate and optimize over tokens"
    2. "it will call all reduce and it basically does an average"
    3. *Technical analysis*: Scales training across 8 GPUs using DDP. Each GPU hosts a model replica and processes a unique shard of data. DDP averages gradients across all processes after backward passes. The implementation carefully handles data loader sharding, gradient synchronization only on the final accumulation step, and loss averaging across processes.

### 💡 Intuitive Understanding

**Analogy:** 8 workers reading different parts of a textbook, taking notes, then meeting to combine their notes into one improved summary. Each worker has a copy of the model and sees different data.

**Mental Model:** DDP mechanics:
```
8 GPUs, each with:
  - Complete copy of the model
  - 1/8 of the data

Each step:
  1. Each GPU computes loss on its data shard
  2. Each GPU computes gradients
  3. All-reduce: average gradients across all GPUs
  4. Each GPU applies same update → models stay in sync

Result: 8x throughput, same model behavior

Code:
  model = DDP(model)  # Wrap model
  # Everything else "just works"
```

**Why It Matters:** DDP is how you scale beyond one GPU. It's almost linear speedup. With 8 GPUs, you train ~8x faster.

---

• *185:00 - 200:00: Scaling to a Large Dataset (FineWeb-Edu)*
    1. "we want to upgrade to a more serious data set"
    2. "we pre-process and pre-tokenize all of the data"
    3. *Technical analysis*: Moves from TinyShakespeare to the FineWeb-Edu dataset (10B token sample). Implements a sharded data loader to stream from multiple files. Sets training steps based on token counts to match GPT-3's schedule (e.g., 375M token warmup).

### 💡 Intuitive Understanding

**Analogy:** Moving from practicing on a small pond to training in the ocean. TinyShakespeare is 1M tokens. FineWeb-Edu is 10B tokens. Real training needs real data.

**Mental Model:** Data scaling:
```
TinyShakespeare: 1M tokens, fits in memory
  - Good for debugging
  - Model memorizes it quickly

FineWeb-Edu: 10B tokens, 20GB on disk
  - High-quality educational text
  - Sharded into many files
  - Streaming data loader required

Training budget:
  - GPT-2: ~40B tokens
  - GPT-3: 300B tokens
  - Modern: 1-2T tokens

More data = better model (with proper scaling)
```

**Why It Matters:** Data is the bottleneck. FineWeb-Edu is curated for quality, not just quantity. This is why the reproduced GPT-2 can match the original with 10x fewer tokens.

---

• *200:00 - 215:00: Adding Validation and Sampling*
    1. "we want to evaluate on the validation split"
    2. "periodically we simply generate samples"
    3. *Technical analysis*: Adds periodic evaluation on a held-out validation set to monitor generalization. Integrates the earlier text generation code to sample from the model during training, providing a qualitative measure of progress.

### 💡 Intuitive Understanding

**Analogy:** Training loss is like your practice exam score. Validation loss is the real exam. Sampling is asking the model to write an essay to see if it actually makes sense.

**Mental Model:** Monitoring training:
```
Every N steps:
  1. Training loss: is the model fitting the data?
  2. Validation loss: is it generalizing or memorizing?
  3. Sample generation: does the output look reasonable?

If train loss ↓ but val loss ↑: overfitting
If train loss flat: learning rate too low or bug
If samples are gibberish: something wrong

All three signals together tell the full story.
```

**Why It Matters:** You can't just train blindly. Validation loss catches overfitting. Sampling catches subtle failure modes that loss doesn't show.

---

• *215:00 - 235:00: Quantitative Evaluation with HellaSwag*
    1. "HellaSwag is a smooth eval with early signal"
    2. "we just have to assign probabilities"
    3. *Technical analysis*: Implements HellaSwag evaluation, a common LLM benchmark. The model is evaluated by computing the average per-token loss for each possible continuation of a context and selecting the one with the lowest loss. This tests world knowledge and reasoning. The reported score for the original GPT-2 124M (29.5%) is set as the target to beat.

### 💡 Intuitive Understanding

**Analogy:** HellaSwag is a multiple-choice reading comprehension test. Given a scenario, which ending is most plausible? The model picks by asking: "Which continuation has the highest probability?"

**Mental Model:** HellaSwag evaluation:
```
Prompt: "A man is standing on a ladder. He"

Choices:
  A) "climbs higher to change the lightbulb"  ← Plausible
  B) "eats a sandwich while juggling"         ← Weird
  C) "transforms into a butterfly"            ← Absurd
  D) "reads a book on quantum physics"        ← Unlikely

Model scoring:
  - Compute loss for each choice given prompt
  - Pick choice with LOWEST loss (highest probability)
  - Compare to correct answer

GPT-2 124M target: 29.5% accuracy
(Random = 25%, so not much better than chance)
```

**Why It Matters:** HellaSwag tests common-sense reasoning and world knowledge. It's a standard benchmark that gives a single number to compare models. The goal is to match or beat 29.5%.

---

• *235:00 - 240:00: Training Results and Checkpointing*
    1. "we are surpassing the validation loss"
    2. "we basically surpassed the gpt2 124m model"
    3. *Technical analysis*: After training for ~10B tokens (1 epoch), the custom model surpasses the original GPT-2 124M's validation loss on FineWeb-Edu and matches its HellaSwag score, but with 10x fewer training tokens. Suggests gains may be due to higher-quality data and better optimization. Extending training to 40B tokens (4 epochs) nearly matches the larger GPT-3 124M's HellaSwag performance. Implements model checkpointing.

### 💡 Intuitive Understanding

**Analogy:** You've built a replica that's as good as the original, using a fraction of the fuel. Better data quality and modern techniques made up for the token count difference.

**Mental Model:** Training results:
```
Original GPT-2 124M:
  - Trained on WebText (40B tokens)
  - HellaSwag: 29.5%

Reproduced GPT-2:
  - Trained on FineWeb-Edu (10B tokens)
  - HellaSwag: ~29.5% (matches with 1/4 the tokens!)

Extended training (40B tokens):
  - HellaSwag: ~32% (approaches GPT-3 124M level)

Why better sample efficiency?
  - Higher quality data (curated educational text)
  - Modern optimization (AdamW, cosine schedule, etc.)
  - Flash Attention, BF16, etc.
```

**Why It Matters:** This validates the entire process. You built it, trained it, and matched the reference. Now you understand every component of LLM training.

---

*Conclusion*
• *Summary of key technical takeaways*: Successfully built GPT-2 from scratch, verified correctness via weight loading, and implemented a full training pipeline. Key performance optimizations included mixed precision (BF16), `torch.compile`, Flash Attention, and multi-GPU DDP training. Using modern hyperparameters (GPT-3) and a quality dataset (FineWeb-Edu) led to sample-efficient training, matching the original model's performance with significantly fewer tokens.
• *Practical applications*: The codebase serves as a complete, understandable template for pre-training medium-sized autoregressive language models. It demonstrates how to effectively utilize modern hardware (A100 GPUs) and software (PyTorch) stacks for LLM training.
• *Long-term recommendations*: Improve data loading with proper shuffling across epochs. Resolve incompatibility between `torch.compile` and evaluation/generation. Extend the evaluation suite with more benchmarks. Explore more aggressive learning rates and other hyperparameter tunings for faster convergence. The implementation is a foundation that can be scaled to larger models and datasets.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Weight Tying:** Explain weight tying and why it's beneficial. What's the memory savings for GPT-2?

2. **Precision Hierarchy:** Order these by precision and speed: FP32, BF16, FP16, TF32. When would you use each?

3. **Flash Attention:** Why is Flash Attention faster despite not reducing the number of operations? What bottleneck does it address?

4. **Nice Numbers:** Why does padding vocab from 50257 to 50304 improve speed? How do you find "nice" numbers?

5. **Gradient Accumulation:** If batch size is 32 and you want effective batch size 512, how many accumulation steps? Why divide loss by accumulation_steps?

6. **DDP Mechanics:** How does DDP keep models in sync? What happens during all-reduce?

### Coding Challenges

1. **Implement Weight Tying:**
   ```python
   class GPT(nn.Module):
       def __init__(self, ...):
           self.wte = nn.Embedding(vocab_size, n_embed)
           # How do you tie lm_head to wte?
   ```

2. **Add torch.compile:**
   Take any PyTorch model and add torch.compile. Measure speedup.

3. **Implement Gradient Accumulation:**
   ```python
   for step in range(steps):
       for micro_step in range(accumulation_steps):
           # Fill in the training loop
           pass
       optimizer.step()
       optimizer.zero_grad()
   ```

4. **Set Up DDP:**
   Take a single-GPU training script and convert it to multi-GPU DDP.

5. **Implement HellaSwag Evaluation:**
   ```python
   def evaluate_hellaswag(model, examples):
       """Return accuracy on HellaSwag benchmark."""
       pass
   ```

### Reflection

- The reproduced GPT-2 matches the original with 10x fewer tokens. Is this because of better data, better optimization, or both? Design an experiment to isolate the factors.

- torch.compile and Flash Attention are described as "free" performance. What are the downsides or limitations of each?

- The lecture trains to match GPT-2 124M. What would change to train GPT-2 1.5B? Make a list of what scales (architecture, compute, data) and what stays the same (techniques, hyperparameters).

- Model checkpointing is mentioned but not detailed. Research best practices for checkpoint management: how often, what to save, how to resume.
