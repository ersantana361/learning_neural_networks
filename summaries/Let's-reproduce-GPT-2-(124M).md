---
title: Let's reproduce GPT-2 (124M)
tags:
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

• *3:40 - 13:48: Loading and Inspecting the Target Model*
    1. "load the GPT to 124 M model as it was released"
    2. "these are just the parameters these are weights floats"
    3. *Technical analysis*: Demonstrates loading OpenAI's weights via Hugging Face to establish a ground-truth target. Analysis of positional embeddings shows they are learned, not fixed sinusoids, and can reveal training status (e.g., "undertrained").

• *13:48 - 28:01: Implementing the GPT-2 Architecture from Scratch*
    1. "we want to write this from scratch ourselves"
    2. "gpt2 is a decoder only Transformer"
    3. *Technical analysis*: Details architectural deviations from the original Transformer: removal of encoder, pre-normalization (layer norms before attention/MLP), and an added final layer norm. The implementation carefully names modules to align with Hugging Face's key names for easy weight loading.

• *28:01 - 31:00: Verifying the Custom Implementation*
    1. "we can load the weights and the biases into our module"
    2. "they look sensible and now we want to initialize from scratch"
    3. *Technical analysis*: Successfully ports pre-trained weights into the custom `GPT` class, validating the implementation's correctness. This creates a verified baseline from which to begin training from random initialization.

• *31:00 - 46:00: Setting Up Training Infrastructure*
    1. "to train the model we're going to need some data set"
    2. "we want to actually iterate these XY batches"
    3. *Technical analysis*: Implements a basic data loader for the TinyShakespeare dataset. Explains the crucial pattern of creating input/target tensors (`x` and `y`) by offsetting a 1D token sequence, then viewing it as a 2D `(batch, time)` tensor.

• *46:00 - 56:00: Initial Training Loop and Loss*
    1. "we want to calculate the loss"
    2. "we expect something around 10.82"
    3. *Technical analysis*: Adds loss calculation (cross-entropy) to the forward pass. Sanity-checks the initial loss against the theoretical value for a uniform output distribution over the vocabulary (~1/50257), confirming proper initialization.

• *56:00 - 66:00: Optimization and Hyperparameter Initialization*
    1. "we're using the AdamW optimizer"
    2. "we are overfitting a single batch"
    3. *Technical analysis*: Sets up a basic training loop with AdamW. Demonstrates successful overfitting on a single batch, a good sanity check. Introduces learning rate (3e-4) and the concept of checking loss convergence.

• *66:00 - 74:00: Correcting Implementation Details: Weight Tying and Initialization*
    1. "this is a common weight tying scheme"
    2. "scale the weight of residual layers by factor of one over sqrt(n)"
    3. *Technical analysis*: Correctly implements weight tying between the token embedding and the LM head, saving parameters and adding a useful inductive bias. Applies GPT-2's specific initialization scheme (std=0.02 for weights, zero for biases) and scales residual block outputs to control activation growth.

• *74:00 - 85:00: Hardware Utilization and Precision (TF32)*
    1. "deep learning can tolerate significantly lower precisions"
    2. "tf32 is a lower Precision format"
    3. *Technical analysis*: Motivates the use of lower precision to increase computational throughput and reduce memory bandwidth pressure. Enables TensorFloat-32 (TF32) in PyTorch, which offers ~8x potential FLOPs gain on Ampere GPUs by truncating mantissa bits inside tensor core operations.

• *85:00 - 100:00: Mixed Precision Training (BF16)*
    1. "we are going to drop down to B float 16"
    2. "our activations have been changed to BF16"
    3. *Technical analysis*: Implements BF16 mixed precision training via `torch.autocast`. BF16 preserves the dynamic range of FP32 (unlike FP16) by keeping the same exponent bits, simplifying training by often avoiding the need for gradient scaling. Parameters remain in FP32.

• *100:00 - 115:00: Compilation with `torch.compile`*
    1. "torch. compile is really quite incredible infrastructure"
    2. "this is about 2.3x Improvement"
    3. *Technical analysis*: Uses `torch.compile` to fuse operations and reduce Python overhead. The compiler analyzes the entire model graph, enabling optimizations like kernel fusion (e.g., combining elementwise ops in GELU), which minimizes expensive round-trips to GPU memory (HBM).

• *115:00 - 125:00: Flash Attention*
    1. "flash attention is a kernel Fusion operation"
    2. "it never gets read or written to the hbm"
    3. *Technical analysis*: Replaces the standard attention implementation with Flash Attention. This algorithm fuses the attention computation steps and uses techniques like online softmax to avoid materializing the full `(T, T)` attention matrix in high-bandwidth memory (HBM), dramatically speeding up the operation.

• *125:00 - 135:00: Numerical Efficiency: "Nice" Tensor Shapes*
    1. "you always want to use nice numbers"
    2. "we are seeing a roughly 4% Improvement"
    3. *Technical analysis*: Pads the vocabulary size from 50257 to 50304 (a multiple of 128). Many CUDA kernels are optimized for dimensions that are multiples of powers of two. Using "ugly" numbers can trigger inefficient boundary-case kernels, while padding leads to more efficient computation despite slightly more FLOPs.

• *135:00 - 155:00: Adopting GPT-3 Hyperparameters*
    1. "we're going to go to gp3 paper to follow along"
    2. "we clip the global Norm of the gradient at 1.0"
    3. *Technical analysis*: Configures optimizer (AdamW betas, epsilon), gradient clipping, and a cosine learning rate schedule with warmup as per the GPT-3 paper. This establishes a robust, modern training configuration. Also implements weight decay only on 2D parameters (embeddings, linear weights).

• *155:00 - 166:00: Gradient Accumulation*
    1. "we need to use what's called gradient accumulation"
    2. "we are missing the normalizer"
    3. *Technical analysis*: Implements gradient accumulation to simulate a large batch size (0.5M tokens) with limited GPU memory. The key nuance is scaling the loss by `1 / accumulation_steps` to maintain the correct gradient averaging semantics, as `loss.backward()` sums gradients.

• *166:00 - 185:00: Distributed Data Parallel (DDP) Training*
    1. "they are going to collaborate and optimize over tokens"
    2. "it will call all reduce and it basically does an average"
    3. *Technical analysis*: Scales training across 8 GPUs using DDP. Each GPU hosts a model replica and processes a unique shard of data. DDP averages gradients across all processes after backward passes. The implementation carefully handles data loader sharding, gradient synchronization only on the final accumulation step, and loss averaging across processes.

• *185:00 - 200:00: Scaling to a Large Dataset (FineWeb-Edu)*
    1. "we want to upgrade to a more serious data set"
    2. "we pre-process and pre-tokenize all of the data"
    3. *Technical analysis*: Moves from TinyShakespeare to the FineWeb-Edu dataset (10B token sample). Implements a sharded data loader to stream from multiple files. Sets training steps based on token counts to match GPT-3's schedule (e.g., 375M token warmup).

• *200:00 - 215:00: Adding Validation and Sampling*
    1. "we want to evaluate on the validation split"
    2. "periodically we simply generate samples"
    3. *Technical analysis*: Adds periodic evaluation on a held-out validation set to monitor generalization. Integrates the earlier text generation code to sample from the model during training, providing a qualitative measure of progress.

• *215:00 - 235:00: Quantitative Evaluation with HellaSwag*
    1. "HellaSwag is a smooth eval with early signal"
    2. "we just have to assign probabilities"
    3. *Technical analysis*: Implements HellaSwag evaluation, a common LLM benchmark. The model is evaluated by computing the average per-token loss for each possible continuation of a context and selecting the one with the lowest loss. This tests world knowledge and reasoning. The reported score for the original GPT-2 124M (29.5%) is set as the target to beat.

• *235:00 - 240:00: Training Results and Checkpointing*
    1. "we are surpassing the validation loss"
    2. "we basically surpassed the gpt2 124m model"
    3. *Technical analysis*: After training for ~10B tokens (1 epoch), the custom model surpasses the original GPT-2 124M's validation loss on FineWeb-Edu and matches its HellaSwag score, but with 10x fewer training tokens. Suggests gains may be due to higher-quality data and better optimization. Extending training to 40B tokens (4 epochs) nearly matches the larger GPT-3 124M's HellaSwag performance. Implements model checkpointing.

*Conclusion*
• *Summary of key technical takeaways*: Successfully built GPT-2 from scratch, verified correctness via weight loading, and implemented a full training pipeline. Key performance optimizations included mixed precision (BF16), `torch.compile`, Flash Attention, and multi-GPU DDP training. Using modern hyperparameters (GPT-3) and a quality dataset (FineWeb-Edu) led to sample-efficient training, matching the original model's performance with significantly fewer tokens.
• *Practical applications*: The codebase serves as a complete, understandable template for pre-training medium-sized autoregressive language models. It demonstrates how to effectively utilize modern hardware (A100 GPUs) and software (PyTorch) stacks for LLM training.
• *Long-term recommendations*: Improve data loading with proper shuffling across epochs. Resolve incompatibility between `torch.compile` and evaluation/generation. Extend the evaluation suite with more benchmarks. Explore more aggressive learning rates and other hyperparameter tunings for faster convergence. The implementation is a foundation that can be scaled to larger models and datasets.