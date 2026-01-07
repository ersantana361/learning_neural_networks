---
title: Building makemore Part 2: MLP
tags:
  - Multi-Layer Perceptron (MLP)
  - Character-Level Language Model
  - Neural Language Model
  - Embedding Lookup
  - Distributed Representations
  - Generalization
  - Curse of Dimensionality
  - Bengio et al. 2003
  - PyTorch Implementation
  - Tensor Manipulation
  - Forward Pass
  - Backward Pass
  - Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Mini-Batching
  - Cross-Entropy Loss
  - Numerical Stability
  - Log-Sum-Exp Trick
  - Learning Rate Tuning
  - Learning Rate Decay
  - Hyperparameter Optimization
  - Train/Dev/Test Split
  - Overfitting
  - Model Evaluation
  - Model Scaling
  - Embedding Visualization
  - Autoregressive Sampling
---

## Introduction

**Key objectives of the video**: To implement a multi-layer perceptron (MLP) character-level language model, moving beyond a simple bigram model to capture longer context without the exponential blow-up of a count-based approach. The tutorial follows the methodology from the Bengio et al. (2003) neural language model paper.

**Core themes and methodologies**: Implementing an MLP with embedding lookup tables, hidden layers with non-linearities, and output logits. The process includes dataset construction, manual forward/backward pass implementation, optimization with gradient descent, hyperparameter tuning (like learning rate), and model evaluation using train/dev/test splits.

**Target audience**: Individuals with foundational knowledge of neural networks and Python/PyTorch, looking to understand the practical implementation and training dynamics of early neural language models.

---

## Detailed Analysis

### Segment: Problem with Bigram Models and Introduction to MLPs (0:00 - 1:47)

**Key verbatim excerpts**:
- "if we are to take more context... things quickly blow up"
- "the whole thing just kind of explodes and doesn't work very well"

**Technical analysis**: The lecturer explains the fundamental limitation of count-based N-gram models: the context window size is severely restricted due to exponential growth in the number of possible contexts (*27^k* for *k* characters). This "curse of dimensionality" makes statistical estimation infeasible and motivates the shift to parameterized neural models that can generalize.

#### 💡 Intuitive Understanding: The Curse of Dimensionality

**Analogy - The Phone Book Problem**: Imagine creating a phone book where each entry requires the full history of a person's life choices. For a 5-choice history, you need 5 entries. For a 10-choice history, you need 100,000 entries. For a 20-choice history, you need more entries than atoms in the universe. This is the curse of dimensionality.

**Mental Model**: For N-gram language models:

```
Bigram (1 char context):    27^1 = 27 parameters
Trigram (2 char context):   27^2 = 729 parameters
4-gram (3 char context):    27^3 = 19,683 parameters
5-gram (4 char context):    27^4 = 531,441 parameters
10-gram (9 char context):   27^9 = 7.6 trillion parameters!

Problem: Most entries will have zero counts (data sparsity)
```

**Why It Matters**: This exponential explosion is why explicit counting doesn't scale. Neural networks solve this by learning to *share* information between similar contexts through embeddings and shared weights, rather than memorizing each context separately.

---

### Segment: Overview of the Bengio et al. (2003) Model (1:48 - 4:12)

**Key verbatim excerpts**:
- "they propose to take every one of these words... and associate... a feature vector"
- "these points or vectors are going to basically move around in this space"

**Technical analysis**: The paper's core innovation is using *learned, distributed representations* (embeddings) for words/characters. Similar entities become proximate in this learned vector space, allowing the model to generalize to unseen context sequences by leveraging similarity, not just exact matches. This is a foundational idea for all modern neural NLP.

#### 💡 Intuitive Understanding: What Are Embeddings?

**Analogy - A Seating Chart**: Imagine arranging people at a party based on interests. Sports fans cluster together, readers cluster together, gamers cluster together. The *position* in the room encodes *meaning*. Similarly, embeddings arrange characters (or words) in a space where similar items are close.

**Mental Model**: Each character gets a coordinate in a learned space:

```
Before training (random):
  'a' → [0.2, -0.8]    (random position)
  'e' → [-0.5, 0.3]    (random position)
  'q' → [0.9, 0.1]     (random position)

After training (meaningful):
  'a' → [0.8, 0.9]     (near other vowels)
  'e' → [0.7, 0.85]    (near other vowels)
  'i' → [0.75, 0.88]   (near other vowels)
  'q' → [-0.9, -0.7]   (isolated, rare)

The network LEARNED this structure from the data!
```

**Why It Matters**: Embeddings are the foundation of modern NLP—from Word2Vec to GPT. They compress discrete symbols into continuous vectors where relationships can be computed. This single idea enables neural networks to handle language.

---

### Segment: Intuition for Generalization via Embeddings (4:18 - 5:42)

**Key verbatim excerpts**:
- "you can transfer knowledge through that embedding"
- "you can generalize to novel scenarios"

**Technical analysis**: A concrete example ("a dog was running in a" vs. "the dog was running in a") illustrates the power of embeddings. The model learns that "a" and "the" have similar embeddings, enabling knowledge transfer. This semantic generalization is the key advantage over rigid, count-based models.

#### 💡 Intuitive Understanding: Knowledge Transfer

**Analogy - Learning to Cook**: If you've learned to make tomato sauce, you can make other red sauces (marinara, arrabbiata) because they share techniques. You don't need to learn each from scratch. Embeddings work similarly—knowledge about one word transfers to similar words.

**Mental Model**: How generalization works:

```
Training data includes:
  "the cat sat on the mat"

The model learns:
  - "the" and "a" have similar embeddings (both are articles)
  - "cat" and "dog" have similar embeddings (both are animals)

At inference, given novel context:
  "a dog sat on the ___"

The model can predict "mat" even though it never saw this exact sequence!
Why? Because "a dog" activates similar neurons as "the cat"
```

**Why It Matters**: This ability to generalize is why neural language models work despite not seeing every possible sentence. They learn *patterns* that transfer to new situations, not just memorize training data.

---

### Segment: Neural Network Architecture Diagram (5:43 - 8:52)

**Key verbatim excerpts**:
- "lookup table C is a matrix that is 17,000 by say 30"
- "this layer has 17,000 neurons... this is the expensive layer"

**Technical analysis**: The architecture is detailed: an embedding layer (C), a hidden layer with tanh activation, and a large output softmax layer. The high cost of the final layer (due to large vocabulary size) is noted, foreshadowing a major challenge in scaling such models that later architectures (like word2vec, transformers) would address.

#### 💡 Intuitive Understanding: The MLP Architecture

**Analogy - A Translation Pipeline**: Think of the network as a pipeline:
1. **Lookup** (C): Convert character IDs to rich descriptions (embeddings)
2. **Combine & Process** (Hidden): Mix the descriptions and extract patterns
3. **Decide** (Output): Vote on what character comes next

**Mental Model**: The architecture step by step:

```
Input: [5, 13, 13]  (indices for "e", "m", "m")
         ↓
┌────────────────────────────────────────────┐
│  EMBEDDING LAYER (C)                        │
│  Shape: (27, embedding_dim)                 │
│  Each index selects a row from C            │
│  Output: (3, embedding_dim) tensor          │
└────────────────────────────────────────────┘
         ↓ flatten to (3 * embedding_dim,)
┌────────────────────────────────────────────┐
│  HIDDEN LAYER                               │
│  Linear: (3*emb_dim) → hidden_size          │
│  Activation: tanh (introduces non-linearity)│
│  Output: (hidden_size,)                     │
└────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────┐
│  OUTPUT LAYER                               │
│  Linear: hidden_size → 27                   │
│  Softmax: convert to probabilities          │
│  Output: probability for each character     │
└────────────────────────────────────────────┘
         ↓
Prediction: "a" (highest probability)
```

**Why It Matters**: This architecture—embedding → hidden layers → output—is the template for virtually all neural language models. Understanding it lets you understand GPT, BERT, and beyond.

---

### Segment: Building the Dataset (9:02 - 12:03)

**Key verbatim excerpts**:
- "block size... is basically the context length"
- "we're always padding with dots"

**Technical analysis**: The code creates (input, target) pairs using a sliding window over the text. A fixed `block_size` defines the context length. Padding with a special token (`.`) at the start handles contexts shorter than the block size, a common technique in sequence modeling.

#### 💡 Intuitive Understanding: Sliding Window Dataset

**Analogy - Reading with a Flashlight**: Imagine reading a book in the dark with a flashlight that only illuminates 3 words. You slide the flashlight along, and at each position, you predict the next word based on what you see. That's the sliding window approach.

**Mental Model**: Creating training examples from "emma":

```
Block size: 3 (look at 3 previous characters)

Word: ".emma." (with start/end tokens)

Position 0: Context [., ., .] → Target: e
Position 1: Context [., ., e] → Target: m
Position 2: Context [., e, m] → Target: m
Position 3: Context [e, m, m] → Target: a
Position 4: Context [m, m, a] → Target: .

5 training examples from one word!
```

**Why It Matters**: This windowing approach is universal. Transformers use the same idea with much larger context windows. Understanding how to create these (context, target) pairs is essential for training any sequence model.

---

### Segment: Implementing Embedding Lookup (12:21 - 17:03)

**Key verbatim excerpts**:
- "to embed simultaneously all of the integers in x we can simply do `c[x]`"
- "pytorch indexing is awesome"

**Technical analysis**: Demonstrates the equivalence between one-hot encoding followed by a linear layer and direct integer indexing into an embedding matrix (`C`). The latter is vastly more efficient. PyTorch's flexible tensor indexing (`c[x]`) allows batch processing, a crucial efficiency gain.

#### 💡 Intuitive Understanding: Why Integer Indexing = Embedding Lookup

**Analogy - Looking Up in a Dictionary**: Instead of encoding "apple" as [1,0,0,0,...] and multiplying by a matrix (slow), just look up row 0 directly (fast). Same result, much faster!

**Mental Model**: The mathematical equivalence:

```
One-hot approach (slow):
  x = 3
  one_hot = [0, 0, 0, 1, 0, ...]  # 27-dimensional
  embedding = one_hot @ C          # Matrix multiply
  # Result: row 3 of C

Direct indexing (fast):
  x = 3
  embedding = C[3]                 # Direct lookup
  # Result: row 3 of C (same answer!)

Why faster? No multiplication needed—just memory access!
```

**Why It Matters**: This optimization is why modern frameworks can handle huge embedding tables efficiently. When you use `nn.Embedding` in PyTorch, this is exactly what happens under the hood.

---

### Segment: Constructing the Hidden Layer and Tensor Manipulation (18:37 - 29:00)

**Key verbatim excerpts**:
- "how do we transform this 32 by 3 by 2 into a 32 by 6"
- "`view` is an extremely efficient operation"

**Technical analysis**: Compares methods to flatten context embeddings: explicit concatenation (`torch.cat`) vs. using `torch.view`. The `view` operation is preferred as it manipulates tensor metadata (shape, stride) without copying data, making it memory-efficient. This highlights the importance of understanding tensor memory layout for performance.

#### 💡 Intuitive Understanding: View vs. Reshape vs. Concatenate

**Analogy - Reorganizing a Bookshelf**: `view` is like relabeling shelf sections without moving books (instant). `cat` is like physically combining multiple shelves into one (takes time). Both achieve similar layouts, but `view` is virtually free.

**Mental Model**: The difference in operations:

```
Input shape: (batch=32, context=3, embed=2)
Desired: (batch=32, context*embed=6)

Method 1: torch.cat (slow)
  - Allocates new memory
  - Copies all data
  - O(n) time and space

Method 2: torch.view (fast)
  - Changes metadata only
  - No data copying
  - O(1) time, O(0) extra space
  - ONLY works if tensor is contiguous!

emb.view(32, 6)  # or emb.view(emb.shape[0], -1)
```

**Why It Matters**: In deep learning, tensor operations run millions of times. Using `view` instead of `cat` can make training significantly faster. Understanding memory layout is a key skill for efficient implementations.

---

### Segment: Forward Pass and Loss Calculation (29:41 - 33:38)

**Key verbatim excerpts**:
- "we want to index into the rows of `prob`... pluck out the probability assigned to the correct character"
- "the loss here is 17"

**Technical analysis**: Manually computes the forward pass to produce logits, applies softmax to get probabilities, and calculates negative log-likelihood loss. This step-by-step build is pedagogical, reinforcing how the network's output is interpreted as a probability distribution over the next character.

#### 💡 Intuitive Understanding: The Complete Forward Pass

**Analogy - A Prediction Machine**: The forward pass is like asking an expert for a prediction. They look at the evidence (embeddings), think about it (hidden layer), and give you probabilities for each outcome (output layer).

**Mental Model**: Tracing through one example:

```
Input: x = [5, 13, 13] (characters "e", "m", "m")
Target: y = 1 (character "a")

Step 1: Embedding lookup
  emb = C[x]  # Shape: (3, embed_dim)

Step 2: Flatten
  emb_flat = emb.view(3 * embed_dim)

Step 3: Hidden layer
  h = tanh(emb_flat @ W1 + b1)

Step 4: Output layer
  logits = h @ W2 + b2  # Shape: (27,)

Step 5: Softmax
  probs = softmax(logits)  # Shape: (27,), sums to 1

Step 6: Loss (for this example)
  loss = -log(probs[y])  # -log(prob of correct answer)
  loss = -log(probs[1])  # -log(prob of "a")
```

**Why It Matters**: Understanding the forward pass in detail lets you debug issues, modify architectures, and reason about what the network is computing. Every complex model is just more layers of this same pattern.

---

### Segment: Using `F.cross_entropy` and Numerical Stability (33:38 - 37:57)

**Key verbatim excerpts**:
- "many good reasons to prefer `F.cross_entropy` over rolling your own"
- "pytorch... internally calculates the maximum value... and subtracts it"

**Technical analysis**: Advocates for using the built-in `cross_entropy` function for efficiency (fused kernels), simpler gradients, and *numerical stability*. The explanation of the log-sum-exp trick (subtracting the max logit) is critical, showing how to avoid overflow when exponentiating large numbers, a vital practical consideration.

#### 💡 Intuitive Understanding: The Log-Sum-Exp Trick

**Analogy - Preventing Calculator Overflow**: If you try to compute e^1000 on a calculator, it shows "overflow." The log-sum-exp trick is like saying "everything relative to the biggest number" so values stay manageable.

**Mental Model**: Why naive softmax fails and the fix:

```
Naive softmax (DANGEROUS):
  logits = [1000, 1001, 1002]
  exp(logits) = [inf, inf, inf]  # OVERFLOW!
  Can't compute probabilities

Log-sum-exp trick (SAFE):
  max_logit = 1002
  shifted = logits - max_logit = [-2, -1, 0]
  exp(shifted) = [0.135, 0.368, 1.0]  # Safe values!
  probs = [0.09, 0.24, 0.67]  # Same answer, no overflow

Mathematical proof: softmax(x) = softmax(x - c) for any constant c
```

**Why It Matters**: This isn't just theoretical—real networks produce logits with large magnitudes. Using `F.cross_entropy` instead of manual softmax+NLL prevents silent numerical bugs that could ruin your training.

---

### Segment: Training Loop and Mini-Batching (37:59 - 44:31)

**Key verbatim excerpts**:
- "we're doing way too much work forwarding and backwarding 220,000 examples"
- "it's much better to have an approximate gradient and just make more steps"

**Technical analysis**: Implements gradient descent with manual zeroing of gradients (`p.grad = None`), backward pass, and parameter update. Introduces *mini-batching* by randomly sampling indices, drastically improving iteration speed. This demonstrates the standard stochastic gradient descent (SGD) training paradigm.

#### 💡 Intuitive Understanding: Why Mini-Batching Works

**Analogy - Surveying Voters**: To predict election results, you don't ask every voter (expensive!). You sample 1,000 random people and get a good estimate. Similarly, mini-batching samples data to estimate the gradient.

**Mental Model**: Full-batch vs mini-batch trade-offs:

```
Full-batch gradient descent:
  - Computes exact gradient on ALL data
  - Very slow per step
  - Stable but few updates
  - Updates per hour: ~10

Mini-batch SGD:
  - Computes noisy gradient on 32 samples
  - Very fast per step
  - Noisy but many updates
  - Updates per hour: ~10,000

The key insight: Many noisy steps beat few perfect steps!
The noise even helps escape local minima (regularization effect)
```

**Why It Matters**: Mini-batching is why deep learning is practical. Without it, training would be impossibly slow. The noise from random sampling actually helps generalization—a happy accident!

---

### Segment: Learning Rate Selection and Decay (44:45 - 52:00)

**Key verbatim excerpts**:
- "how do you determine this learning rate"
- "we are spaced exponentially in this interval"

**Technical analysis**: Introduces a practical method for learning rate search: training over a range of rates (e.g., *10^{-3}* to *10^{0}*) spaced exponentially and plotting loss. The "sweet spot" is where loss decreases steadily without exploding. Also introduces *learning rate decay* as a common technique to refine optimization in later stages.

#### 💡 Intuitive Understanding: The Learning Rate Sweet Spot

**Analogy - Walking Downhill Blindfolded**: The learning rate is your step size.
- Too small: You'll reach the bottom eventually, but it takes forever
- Too big: You overshoot, stumble, and might climb back up by accident
- Just right: Steady progress down the hill

**Mental Model**: The learning rate sweep:

```
Learning rate experiment:
  lr=0.0001: Loss decreases very slowly (too small)
  lr=0.001:  Loss decreases steadily ✓
  lr=0.01:   Loss decreases quickly ✓
  lr=0.1:    Loss decreases then plateaus ✓
  lr=1.0:    Loss explodes to infinity! (too big)

Sweet spot: somewhere in [0.001, 0.1]

Decay strategy:
  Start at lr=0.1 (fast progress)
  After 10k steps, reduce to lr=0.01 (fine-tuning)
  Final steps at lr=0.001 (polish)
```

**Why It Matters**: Learning rate is often the most important hyperparameter. A systematic search early in a project saves hours of wasted training with bad rates.

---

### Segment: Train/Dev/Test Splits and Overfitting (52:00 - 56:19)

**Key verbatim excerpts**:
- "the standard in the field is to split up your data set into three splits"
- "you are only allowed to test... very very few times"

**Technical analysis**: Emphasizes rigorous evaluation by splitting data into training (parameter optimization), development/validation (hyperparameter tuning), and test (final evaluation) sets. This framework is essential to diagnose *overfitting* (gap between train and dev loss) and ensure the model generalizes.

#### 💡 Intuitive Understanding: Why Three Splits?

**Analogy - Studying for an Exam**:
- **Training set** = Practice problems (you learn from these)
- **Dev set** = Practice exams (you check your progress, adjust study strategy)
- **Test set** = The real exam (only see it once, measures true knowledge)

If you keep peeking at the real exam to guide your studying, your "score" becomes meaningless!

**Mental Model**: Detecting overfitting:

```
Healthy training:
  Train loss: 2.3 → 2.0 → 1.8 → 1.6
  Dev loss:   2.5 → 2.2 → 2.0 → 1.9
  (Both decreasing, small gap = good generalization)

Overfitting:
  Train loss: 2.3 → 1.5 → 0.8 → 0.3
  Dev loss:   2.5 → 2.3 → 2.4 → 2.5
  (Train keeps dropping, dev plateaus = memorizing, not learning)

Fix: Regularization, more data, smaller model
```

**Why It Matters**: Overfitting is the enemy of useful models. The train/dev/test framework is the standard defense, used from academic papers to production ML systems.

---

### Segment: Model Scaling and Hyperparameter Experimentation (56:31 - 70:59)

**Key verbatim excerpts**:
- "increasing the size of the model should help the neural net"
- "the bottleneck... could be these embeddings that are two dimensional"

**Technical analysis**: Systematically explores increasing model capacity: more hidden neurons and larger embedding dimensions. The process is empirical—observe train/dev loss, identify potential bottlenecks (e.g., low-dimensional embeddings), adjust, and re-train. This mirrors the experimental workflow in deep learning research.

#### 💡 Intuitive Understanding: Finding the Bottleneck

**Analogy - A Factory Assembly Line**: If one station is slow, the whole factory slows down. In neural networks, the "slowest station" (bottleneck) limits performance. Maybe embeddings are too small to capture character relationships, or the hidden layer can't learn complex patterns.

**Mental Model**: Systematic scaling:

```
Experiment log:

Config 1: emb=2, hidden=100
  Train: 2.4  Dev: 2.5
  Analysis: Similar losses = underfitting (model too small)

Config 2: emb=10, hidden=100
  Train: 2.1  Dev: 2.3
  Analysis: Improved! Embeddings were bottleneck

Config 3: emb=10, hidden=300
  Train: 1.9  Dev: 2.2
  Analysis: Still improving, try more hidden

Config 4: emb=10, hidden=500
  Train: 1.7  Dev: 2.3
  Analysis: Train improved but dev got worse = overfitting!

Best config: emb=10, hidden=300
```

**Why It Matters**: This empirical tuning process is how real ML engineering works. There's no formula—you try things, measure, and iterate. Understanding this workflow is as important as understanding the math.

---

### Segment: Visualization of Learned Embeddings (71:00 - 73:18)

**Key verbatim excerpts**:
- "the vowels a e i o u are clustered up here"
- "q is kind of treated as an exception"

**Technical analysis**: Visualizing 2D character embeddings reveals the model has learned meaningful structure: vowels cluster together, while rare letters like 'q' are outliers. This provides intuitive, post-hoc interpretability for the learned representations, validating the embedding approach.

#### 💡 Intuitive Understanding: Reading Embedding Plots

**Analogy - A Map of Characters**: The embedding plot is like a map where the network has placed each character based on how it behaves in names. Characters that behave similarly end up near each other.

**Mental Model**: What the network learned:

```
2D Embedding visualization:

       │ (vowels cluster)
     a │  e
       │ i   o
       │     u
───────┼──────────────
       │
 q x z │   (rare letters isolated)
       │
       │    b d g p t k
       │    (consonants cluster)

The network discovered:
- Vowels appear in similar contexts → cluster together
- 'q' almost always followed by 'u' → unique position
- Consonants share patterns → form their own cluster
```

**Why It Matters**: Visualization validates that embeddings capture meaningful structure. This interpretability is valuable for debugging and understanding what your model has learned.

---

### Segment: Sampling from the Trained Model (73:24 - 74:55)

**Key verbatim excerpts**:
- "we're going to generate 20 samples"
- "the words here are much more word like or name like"

**Technical analysis**: Demonstrates the generative capability of the model. The sampling algorithm feeds the model's own predictions back as context in an autoregressive loop. The improved quality of samples (compared to the bigram model) is a tangible, qualitative measure of the model's success.

#### 💡 Intuitive Understanding: Autoregressive Generation

**Analogy - Continuing a Story**: Each word you write becomes context for the next word. The model does the same—it generates a character, adds it to the context, and uses that to generate the next character.

**Mental Model**: The generation loop:

```python
context = ['.', '.', '.']  # Start tokens

while True:
    # Get prediction for next character
    probs = model(context)

    # Sample from distribution
    next_char = sample(probs)

    if next_char == '.':
        break  # End token

    # Add to output and update context
    output += next_char
    context = context[1:] + [next_char]  # Slide window

# Example generation:
context: ['.', '.', '.'] → predict → sample 'j'
context: ['.', '.', 'j'] → predict → sample 'a'
context: ['.', 'j', 'a'] → predict → sample 'n'
context: ['j', 'a', 'n'] → predict → sample 'e'
context: ['a', 'n', 'e'] → predict → sample '.'
Output: "jane"
```

**Why It Matters**: This autoregressive sampling is the generation mechanism for all language models, including ChatGPT. The only difference is scale—GPT uses thousands of tokens of context and billions of parameters, but the loop is identical.

---

## Conclusion

**Summary of key technical takeaways**: Successfully implemented an MLP-based language model that overcomes the statistical limitations of N-grams by using learned embeddings for generalization. Key implementation skills included tensor manipulation (`view`, indexing), manual forward/backward passes, mini-batch SGD, learning rate tuning, and proper dataset splitting for evaluation.

**Practical applications**: The core architecture forms the historical basis for neural language modeling. The principles of embedding lookup, multi-layer transformations, and cross-entropy training are foundational to more advanced models (RNNs, Transformers). The tutorial workflow is directly applicable to training simple neural networks for other classification/regression tasks.

**Long-term recommendations**: To improve the model, experiment with hyperparameters (embedding size, hidden layer size, context length), implement more advanced optimizers (Adam), add regularization (dropout, weight decay), and explore the extensions suggested in the Bengio et al. paper. The next conceptual step is to move towards recurrent neural networks (RNNs) or transformers to handle arbitrarily long contexts more effectively.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Explain the trade-off**: Why can't we just use a very large context window with the counting-based approach? What specifically prevents us from building a 10-gram model with counting?

2. **Embedding intuition**: Two characters have very similar embeddings. What does this tell you about how they're used in names? Give a concrete example.

3. **Bottleneck diagnosis**: Your model has train loss of 1.5 and dev loss of 1.6 (very close). Is this underfitting, overfitting, or good fit? What would you try next to improve?

4. **Mini-batch trade-offs**: Why do we use batch size 32 instead of batch size 1 or batch size = all data? What are the pros and cons of each?

5. **Numerical stability**: Explain in simple terms why computing `softmax([1000, 1001, 1002])` directly fails, and how subtracting the max fixes it.

### Coding Challenges

1. **Implement the MLP**: Build the character-level MLP from scratch:
   - Create the embedding matrix C
   - Build the hidden layer with tanh
   - Build the output layer
   - Implement the training loop with mini-batching

2. **Learning rate finder**: Implement a learning rate sweep:
   - Train for 1000 steps at each of 10 learning rates from 10^-4 to 10^0
   - Plot loss vs. learning rate
   - Identify the optimal range

3. **Context length experiment**: Train models with context lengths 2, 3, 4, and 5. Compare:
   - Final dev loss
   - Sample quality (generate 20 names from each)
   - Training time

4. **Embedding visualization**: After training, plot the 2D embeddings of all characters. Color-code vowels vs. consonants. Do they cluster?

5. **Regularization**: Add L2 weight decay to your training loop. Experiment with different regularization strengths. How does it affect the train/dev loss gap?

### Reflection

- **Historical context**: This architecture is from 2003. Modern language models use similar ideas but with attention mechanisms. What limitation of the MLP approach might have motivated the development of attention?

- **Scaling laws**: As we increased model size, performance improved—up to a point. What do you think happens if we had 10x more training data? Would the optimal model size change?

- **Generalization mystery**: The model generates novel names it never saw. How is this possible when neural networks just learn to minimize loss on training examples?
