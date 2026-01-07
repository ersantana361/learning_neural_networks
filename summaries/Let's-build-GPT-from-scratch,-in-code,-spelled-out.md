---
title: Let's build GPT: from scratch, in code, spelled out.
tags:
  - Transformer Architecture
  - Attention Is All You Need
  - Self-Attention Mechanism
  - Language Model Implementation
  - GPT
  - Character-Level Language Model
  - Autoregressive Model
  - Decoder-Only Transformer
  - Tokenization
  - Multi-Head Attention
  - Feed-Forward Network
  - Residual Connections
  - Layer Normalization
  - Training Loop
  - Cross-Entropy Loss
  - Tiny Shakespeare Dataset
  - Pre-training
  - Fine-tuning
  - ChatGPT
  - Neural Network from Scratch
---

*Introduction*
• *Key objectives of the video*: To explain the Transformer architecture from the "Attention Is All You Need" paper, and to build a GPT-like character-level language model from scratch, trained on the Tiny Shakespeare dataset.
• *Core themes and methodologies*: The video focuses on the mechanics of self-attention, the building blocks of the Transformer (attention heads, feed-forward networks, residual connections, layer normalization), and the process of training an autoregressive language model.
• *Target audience*: Individuals with proficiency in Python and a basic understanding of calculus and statistics, who are interested in the inner workings of models like ChatGPT.

*Detailed Analysis*

• **0:00 - 2:05: Introduction to Language Models and the Transformer**
    1. *"chat GPT is a probabilistic system and for any one prompt it can give us multiple answers"*
    2. *"it is what we call a language model... it models the sequence of words or characters or tokens"*
    • *Technical analysis and implications*: Introduces the core concept of language modeling as sequence completion. Highlights the probabilistic nature of modern LLMs, which allows for varied outputs from a single prompt. Sets the stage for the Transformer as the foundational architecture.

### 💡 Intuitive Understanding

**Analogy:** A language model is like an incredibly well-read autocomplete system. Given "To be or not to", it doesn't just know one answer—it understands that "be" is most likely, but "exist" or "continue" are also possible. It samples from this probability distribution, which is why it can give different answers each time.

**Mental Model:** The core equation of language modeling:
```
P(next_token | previous_tokens)

Given: "The cat sat on the"
Model outputs: {mat: 0.3, floor: 0.2, chair: 0.15, dog: 0.001, ...}

Sampling from this distribution gives varied but sensible completions.
```

**Why It Matters:** Understanding that LLMs are fundamentally about probability distributions over tokens demystifies their behavior. "Creativity" is sampling randomness; "knowledge" is high probability for correct tokens; "hallucination" is confident probabilities for wrong tokens.

---

• **2:05 - 4:10: Introducing the Transformer and Project Scope**
    1. *"GPT is... generatively pre-trained Transformer so Transformer is the neural net that actually does all the heavy lifting"*
    2. *"I'd like to focus on just to train a Transformer based language model and in our case it's going to be a character level language model"*
    • *Technical analysis and implications*: Distills the GPT acronym and anchors the tutorial to the 2017 Transformer paper. The decision to use a character-level model on a small dataset (Tiny Shakespeare) makes the task computationally tractable for educational purposes while preserving the core architectural principles.

### 💡 Intuitive Understanding

**Analogy:** Building a Transformer on Tiny Shakespeare is like learning to bake with a small home oven. You use the same techniques as a professional bakery (mixing, rising, baking), just at a smaller scale. The fundamentals transfer when you scale up.

**Mental Model:** The GPT family decoded:
```
G = Generative    → It generates text (next-token prediction)
P = Pre-trained   → Trained on massive data before any specific task
T = Transformer   → The architecture doing the computation

Character-level = 26 letters + punctuation ≈ 65 tokens
Subword-level (real GPT) = 50,000+ tokens (words/subwords)

Same architecture, different vocabulary size.
```

**Why It Matters:** Character-level modeling is simpler (no tokenizer complexity) but requires the model to learn spelling from scratch. It's the perfect teaching ground because you see every concept without the noise of subword tokenization.

---

• **4:10 - 10:30: Data Preparation and Tokenization**
    1. *"we are going to basically model how these characters follow each other"*
    2. *"when people say tokenize they mean convert the raw text as a string to some sequence of integers"*
    • *Technical analysis and implications*: Establishes the character-level modeling task. Compares simple character tokenization (small vocabulary, long sequences) to subword tokenizers like BPE (used by GPT), which have a larger vocabulary but shorter sequences. This trade-off is a key practical consideration in LLM design.

### 💡 Intuitive Understanding

**Analogy:** Tokenization is like choosing a unit of currency. Character-level is like counting in pennies—simple but lots of coins. Subword tokenization (BPE) is like using mixed denominations—more complex but fewer pieces to handle.

**Mental Model:** The tokenization trade-off:
```
Character-level:
  Text: "Hello"
  Tokens: [H, e, l, l, o] → 5 tokens
  Vocab size: ~65

Subword (BPE):
  Text: "Hello"
  Tokens: [Hello] → 1 token (if "Hello" is in vocab)
  Vocab size: ~50,000

More vocab → fewer tokens per text → easier for model to learn
But: more complex tokenizer, larger embedding table
```

**Why It Matters:** Real GPT models use BPE because it dramatically shortens sequences. "ChatGPT" might be 1-2 tokens, not 7 characters. This matters for efficiency: attention is O(n²) in sequence length.

---

• **10:30 - 20:08: Batching and Data Loading**
    1. *"we only work with chunks of the data set... sample random little chunks"*
    2. *"in a chunk of nine characters there's actually eight individual examples packed in there"*
    • *Technical analysis and implications*: Explains the efficient training method of using random context windows (blocks). Crucially, a single block contains multiple training examples (predicting the next token for every position in the context), which also trains the model to work with varying context lengths from 1 up to the block size.

### 💡 Intuitive Understanding

**Analogy:** Imagine learning to predict by sliding a window over a book. For the sentence "Hello_World", you practice: "H→e", "He→l", "Hel→l", "Hell→o", etc. Every window position is a training example. The model learns from short contexts (just "H") and long contexts (the full window).

**Mental Model:** Packing training examples:
```
Block: [A, B, C, D, E, F, G, H]  (block_size = 8)

Training examples extracted:
  Context: [A]           → Target: B
  Context: [A, B]        → Target: C
  Context: [A, B, C]     → Target: D
  ...
  Context: [A,B,C,D,E,F,G] → Target: H

8 positions = 8 training examples per block!
```

**Why It Matters:** This is why language model training is so efficient. We don't waste any computation—every token position provides a training signal. And the model automatically learns to handle all context lengths from 1 to block_size.

---

• **20:08 - 29:00: Implementing a Bigram (Baseline) Language Model**
    1. *"the simplest possible neural network... is the Bigram language model"*
    2. *"every single integer in our input is going to refer to this embedding table and it's going to pluck out a row"*
    • *Technical analysis and implications*: Starts with a trivial baseline model that predicts the next token based only on the identity of the current token, implemented via an embedding lookup. This establishes the training loop (forward pass, cross-entropy loss, backpropagation) and a generation function, providing a foundation to build upon.

### 💡 Intuitive Understanding

**Analogy:** A bigram model is like a very forgetful predictor. "What comes after 'e'?" It can answer, but ask "What comes after 'th'?" and it only remembers the 'h', forgetting the 't'. No context beyond the immediate predecessor.

**Mental Model:** Bigram as embedding lookup:
```
vocab_size = 65

Embedding table: [65 x 65] matrix
  Row i = logits for "what comes after token i"

Input: token 5
Output: Row 5 of embedding table = [logit_0, logit_1, ..., logit_64]
Softmax → probability distribution over next token
```

**Why It Matters:** This baseline is intentionally terrible at generation because it can't use context. It establishes what we're trying to beat. Every improvement from here (attention, layers) is about using more context.

---

• **29:00 - 38:02: Mathematical Foundation for Self-Attention**
    1. *"the trick is that we can be very very efficient about doing this using matrix multiplication"*
    2. *"you can think of these zeros as... an interaction strength or like an affinity"*
    • *Technical analysis and implications*: Introduces the core mathematical trick: using a batched matrix multiply with a lower-triangular mask to perform a weighted aggregation of past information. This is the efficient engine that will power self-attention, where the uniform weights will become data-dependent affinities.

### 💡 Intuitive Understanding

**Analogy:** Imagine you're averaging reviews. The naive way: loop through all past reviews and average. The matrix way: put all reviews in a matrix, multiply by a weight matrix, done. Same result, but GPUs love matrix operations—they can do them in parallel.

**Mental Model:** The masked averaging trick:
```
Tokens: [A, B, C, D]

We want: for each position, average all previous tokens
  Position 0: just A
  Position 1: avg(A, B)
  Position 2: avg(A, B, C)
  Position 3: avg(A, B, C, D)

Weight matrix (before normalization):
  [[1, 0, 0, 0],    # Position 0: only uses A
   [1, 1, 0, 0],    # Position 1: uses A, B
   [1, 1, 1, 0],    # Position 2: uses A, B, C
   [1, 1, 1, 1]]    # Position 3: uses all

output = weights @ tokens  # One matrix multiply = all averages!
```

**Why It Matters:** This is the computational foundation of attention. The lower-triangular mask enforces causality—position 3 can't see position 4 (the future). Self-attention just makes the weights data-dependent instead of uniform.

---

• **38:02 - 58:03: Implementing a Single Head of Self-Attention**
    1. *"every single token at each position will emit two vectors it will emit a query and it will emit a key"*
    2. *"attention is a communication mechanism... where you have a number of nodes in a directed graph"*
    • *Technical analysis and implications*: Explains the heart of the Transformer. Tokens create data-dependent **queries** (what they're looking for) and **keys** (what they contain). The dot product of queries and keys creates affinities (**attention scores**). After masking (for causality) and softmax, these scores weight the aggregation of **value** vectors (what the token will communicate). This allows tokens to selectively gather information from their past context.

### 💡 Intuitive Understanding

**Analogy:** Think of attention as a cocktail party. Each person (token) has:
- A **query**: "I'm looking for someone who knows about X"
- A **key**: "I'm the expert on Y"
- A **value**: "Here's what I have to share about Y"

When your query matches someone's key (high dot product), you pay attention to their value. It's information retrieval, not averaging.

**Mental Model:** The attention mechanism:
```
For each token:
  Q = query = "What am I looking for?" = x @ W_Q
  K = key   = "What do I contain?"     = x @ W_K
  V = value = "What do I want to say?" = x @ W_V

Attention scores: scores = Q @ K.T / sqrt(d_k)
  - Each query asks: "How much do I care about each key?"
  - Divide by sqrt(d_k) to keep variance stable

Masked softmax: weights = softmax(mask(scores))
  - Mask future positions (can't look ahead)
  - Softmax normalizes to probabilities

Output: output = weights @ V
  - Weighted sum of values based on attention
```

**Why It Matters:** This is the core innovation of the Transformer. Unlike RNNs (sequential) or CNNs (fixed local), attention is a flexible, learnable routing mechanism. The model decides which tokens are relevant, and this decision is learned from data.

---

• **58:03 - 70:57: Scaling Attention and Adding Network Depth**
    1. *"multi-head attention... is just applying multiple attentions in parallel and concatenating their results"*
    2. *"the self attention is the communication and then once they've gathered all the data now they need to think on that data individually and that's what feed forward is doing"*
    • *Technical analysis and implications*: Scales up the communication mechanism by using multiple attention heads in parallel (like group convolutions), allowing the model to develop different communication channels. Adds a feed-forward network per token after attention, introducing non-linearity and computation on the gathered information. This creates the classic Transformer block: communicate (attention), then compute (FFN).

### 💡 Intuitive Understanding

**Analogy:** Multi-head attention is like having multiple specialists at a meeting. One head might focus on "who is the subject?", another on "what is the verb?", another on "what was mentioned earlier?". Each head attends to different aspects, and their findings are combined.

**Mental Model:** The Transformer block structure:
```
Input: x (token embeddings)

# Step 1: Communication (what do I need from others?)
attended = MultiHeadAttention(x)
x = x + attended  # Residual connection

# Step 2: Computation (now think about what I gathered)
computed = FeedForward(x)  # Token-wise MLP
x = x + computed  # Residual connection

Output: x (enriched token representations)
```

The FFN is usually: Linear(d, 4d) → ReLU → Linear(4d, d)
The 4x expansion gives the model "thinking room."

**Why It Matters:** Communication (attention) and computation (FFN) are complementary. Attention gathers relevant information; FFN processes it. This separation is why Transformers can be parallelized—all tokens compute simultaneously, unlike RNNs.

---

• **70:57 - 85:00: Enabling Deep Networks: Residual Connections and LayerNorm**
    1. *"you have this gradient super highway that goes directly from the supervision all the way to the input unimpeded"*
    2. *"layer Norm... normalizes the rows... a per token transformation that just normalizes the features"*
    • *Technical analysis and implications*: Addresses optimization challenges in deep networks. **Residual connections** (adding a block's input to its output) create an unimpeded path for gradient flow. **Layer normalization** stabilizes activations by normalizing features within each token's vector. These are critical innovations that enable the training of very deep Transformer models.

### 💡 Intuitive Understanding

**Analogy:** Residual connections are like adding express lanes on a highway. Even if the regular lanes (transformations) are congested (vanishing gradients), the express lanes let gradients flow directly from the end to the beginning. No matter how deep the network, there's always a clear path.

**Mental Model:** Residual connections:
```
Without residual:  x → Layer → y
  Gradient: dy/dx = Layer_gradient (can vanish)

With residual:     x → Layer → y; output = x + y
  Gradient: d(x+y)/dx = 1 + Layer_gradient (always at least 1!)

The "1" is the gradient superhighway.
```

LayerNorm vs BatchNorm:
```
BatchNorm: normalize across batch (all examples, per feature)
LayerNorm: normalize across features (per example, all features)

LayerNorm is independent per token—no batch dependency.
Better for variable-length sequences and small batches.
```

**Why It Matters:** Without residual connections, training 6+ layers is extremely difficult. GPT-3 has 96 layers. The residual highway makes this possible. LayerNorm stabilizes each token's representation, preventing exploding/vanishing activations.

---

• **85:00 - 102:04: Scaling Up the Model and Final Results**
    1. *"we can try to scale this up... I made the batch size be much larger... block size to be 256... embedding Dimension is now 384"*
    2. *"we get a validation loss of 1.48 which is actually quite a bit of an improvement"*
    • *Technical analysis and implications*: Demonstrates the impact of scaling model hyperparameters (embedding dimension, number of layers/heads, context length). Adding dropout regularizes the larger model. The result is a significant drop in loss and the generation of text that convincingly mimics Shakespearean style, validating the implemented architecture.

### 💡 Intuitive Understanding

**Analogy:** Scaling is like upgrading from a bicycle to a car. Same transportation principles, but more power, more capacity, more speed. The small model proved the design works; the large model shows what it can really do.

**Mental Model:** Scaling dimensions:
```
Small model:
  n_embed = 32, n_head = 4, n_layer = 3, block_size = 8
  ~10K parameters

Scaled model:
  n_embed = 384, n_head = 6, n_layer = 6, block_size = 256
  ~10M parameters

1000x more parameters = dramatically better text generation
```

Dropout for regularization:
```
During training: randomly zero out X% of neurons
During inference: use all neurons (scaled appropriately)

Effect: prevents co-adaptation, improves generalization
```

**Why It Matters:** This is the core message of "Scaling Laws": bigger models, more data, more compute → better performance. The architecture scales gracefully. The same code runs a toy model or GPT-3 (with more resources).

---

• **102:04 - 116:18: Architecture Context and Connecting to ChatGPT**
    1. *"what we implemented here is a decoder only Transformer... it has this Auto regressive property"*
    2. *"to train chat GPT there are roughly two stages first is the pre-training stage and then the fine-tuning stage"*
    • *Technical analysis and implications*: Clarifies that the built model is a **decoder-only** Transformer (due to the causal mask), suitable for language modeling. Contrasts it with the original paper's **encoder-decoder** architecture used for sequence-to-sequence tasks like translation. Outlines the two-stage process to create a system like ChatGPT: 1) **Pre-training** a large decoder-only model on internet-scale text (as demonstrated), and 2) **Fine-tuning & Alignment** (supervised fine-tuning, reward modeling, RLHF) to make the model a helpful assistant.

### 💡 Intuitive Understanding

**Analogy:** Pre-training is like general education—learning to read, write, and understand the world from vast amounts of text. Fine-tuning is like job training—learning to be specifically helpful, follow instructions, and avoid harmful outputs.

**Mental Model:** The path from GPT to ChatGPT:
```
Stage 1: Pre-training
  Data: Massive internet text (books, web, code)
  Task: Predict next token
  Result: A "completion machine" that knows a lot but isn't helpful

Stage 2: Supervised Fine-Tuning (SFT)
  Data: Human-written examples of helpful responses
  Task: Learn to respond like a helpful assistant
  Result: A model that tries to be helpful

Stage 3: RLHF (Reinforcement Learning from Human Feedback)
  Data: Human rankings of model outputs
  Task: Optimize for human preferences
  Result: ChatGPT-style helpful, harmless, honest assistant
```

**Why It Matters:** The model we built is essentially "Stage 1." It can complete text but isn't an assistant. The magic of ChatGPT comes from careful alignment (Stages 2-3), which is a separate discipline on top of the Transformer architecture.

---

*Conclusion*
• *Summary of key technical takeaways*: The Transformer is built on **self-attention**, a data-dependent communication mechanism between tokens. Key components include **multi-head attention**, **feed-forward networks**, **residual connections**, and **layer normalization**. Autoregressive language modeling is trained by predicting the next token in sequences of fixed context windows.
• *Practical applications*: The principles shown allow for training character-level or token-level language models on any corpus. The code can be extended and scaled to create base models similar to GPT.
• *Long-term recommendations*: To build a ChatGPT-like system, one must first pre-train a large decoder-only Transformer (as shown) on a massive dataset. This must then be followed by extensive fine-tuning stages (supervised fine-tuning, reinforcement learning from human feedback) to align the model's outputs with desired conversational behavior.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Query-Key-Value:** In your own words, explain what Query, Key, and Value represent in self-attention. Why do we need three separate projections instead of one?

2. **Causal Masking:** Why do we use a lower-triangular mask in the attention weights? What would happen if we removed it?

3. **Multi-Head Purpose:** If a single attention head can learn any attention pattern, why do we use multiple heads?

4. **Residual Flow:** Draw the gradient flow through a Transformer block. Why does the residual connection make deep networks trainable?

5. **LayerNorm vs BatchNorm:** Why does the Transformer use LayerNorm instead of BatchNorm? What property of text/sequences makes LayerNorm more appropriate?

6. **Scaling Behavior:** If you double the embedding dimension, approximately how much do the parameters increase? What about doubling the number of layers?

### Coding Challenges

1. **Implement Single-Head Attention:**
   ```python
   def single_head_attention(x, W_Q, W_K, W_V, mask):
       """
       x: [B, T, C] - input embeddings
       W_Q, W_K, W_V: [C, head_size] - projection matrices
       mask: [T, T] - causal mask
       Returns: [B, T, head_size]
       """
       pass
   ```

2. **Build Multi-Head Attention:**
   ```python
   class MultiHeadAttention(nn.Module):
       def __init__(self, n_embed, n_head):
           # n_head heads, each with head_size = n_embed // n_head
           pass

       def forward(self, x):
           # Apply all heads in parallel, concatenate results
           pass
   ```

3. **Complete Transformer Block:**
   ```python
   class TransformerBlock(nn.Module):
       def __init__(self, n_embed, n_head):
           # Multi-head attention + FFN + residuals + LayerNorm
           pass

       def forward(self, x):
           # x = x + self.attn(self.ln1(x))
           # x = x + self.ffn(self.ln2(x))
           pass
   ```

4. **Train on Your Own Dataset:**
   - Replace Tiny Shakespeare with your own text corpus (lyrics, code, recipes)
   - Train a character-level model
   - Generate samples and evaluate quality

5. **Attention Visualization:**
   - For a trained model, extract the attention weights for a given input
   - Visualize which tokens attend to which other tokens
   - Interpret the learned attention patterns (do they make linguistic sense?)

### Reflection

- The Transformer paper's title is "Attention Is All You Need." Is this literally true? What else is needed besides attention (think: FFN, LayerNorm, positional encoding)?

- GPT-4 reportedly has ~1.7 trillion parameters. The model in this lecture has ~10 million. How does scaling by 170,000x change the capabilities? Is it purely "more of the same" or are there qualitative differences?

- The lecture implements a decoder-only Transformer. Research the encoder-decoder architecture from the original paper. When would you use encoder-decoder vs decoder-only?

- Self-attention has O(n²) complexity in sequence length. For a document with 100,000 tokens, how many attention computations would be needed? Research "efficient attention" methods (Sparse Attention, Linear Attention, FlashAttention) and how they address this.
