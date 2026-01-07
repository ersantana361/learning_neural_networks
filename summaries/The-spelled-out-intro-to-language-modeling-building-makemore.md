---
title: The spelled-out intro to language modeling: building makemore
tags:
  - character-level language model
  - bigram model
  - neural network
  - PyTorch
  - gradient descent
  - backpropagation
  - maximum likelihood estimation
  - negative log likelihood
  - tensor operations
  - one-hot encoding
  - softmax
  - logits
  - model sampling
  - broadcasting
  - regularization
---

## Introduction

**Key objectives of the video**: To build a character-level language model from scratch, starting with a simple bigram model. The video demonstrates both a statistical counting approach and a neural network approach using gradient-based optimization, establishing the foundation for more complex models like transformers.

**Core themes and methodologies**: Character-level language modeling, bigram statistics, maximum likelihood estimation, neural network implementation, PyTorch tensor operations, backpropagation, and gradient descent.

**Target audience**: Individuals with foundational programming and machine learning knowledge (familiarity with concepts like *micrograd* is assumed) seeking to understand the inner workings of language models and neural network training.

---

## Detailed Analysis

### Segment: Introduction to the Project (0:00 - 1:30)

**Key verbatim excerpts**:
- "make more as the name suggests makes more of things that you give it"
- "under the hood make more is a character level language model"

**Technical analysis**: The presenter introduces the goal: creating a model that generates new, name-like strings. The core methodology is established as character-level sequence prediction, framing the problem for the audience.

#### 💡 Intuitive Understanding: What is a Language Model?

**Analogy - Autocomplete on Steroids**: A language model is like your phone's autocomplete, but it can generate entire words, sentences, or documents. Given what came before, it predicts what comes next. The model assigns probabilities to all possible continuations.

**Mental Model**: Think of a language model as answering the question: "Given this context, what's likely to come next?"

```
Context: "The cat sat on the ___"
Language Model's Answer:
  - "mat" → 25% probability
  - "floor" → 20% probability
  - "couch" → 15% probability
  - ... (probabilities for all possible words)
```

**Why It Matters**: Language models are the foundation of modern AI assistants, translation systems, and text generators. Understanding them at the character level builds intuition for more complex word-level and subword-level models like GPT.

---

### Segment: Data Loading and Exploration (1:30 - 4:25)

**Key verbatim excerpts**:
- "we have to realize here is that every single word here like isabella is actually quite a few examples packed in to that single word"
- "in the beginning what i'd like to start with is i'd like to start with building a bi-gram language model"

**Technical analysis**: The transcript explains how a single word provides multiple training examples (character transitions). It introduces the bigram model as a simple starting point, which only considers the previous character to predict the next, establishing a baseline model.

#### 💡 Intuitive Understanding: Training Examples from Words

**Analogy - Learning Dance Steps**: Learning to dance isn't about memorizing entire routines—it's about learning step transitions. Similarly, learning language isn't about memorizing words; it's about learning character (or word) transitions.

**Mental Model**: A single word like "emma" becomes multiple training examples:

```
Word: "emma"
Training examples (with special tokens . for start/end):
  . → e  (start is followed by 'e')
  e → m  ('e' is followed by 'm')
  m → m  ('m' is followed by 'm')
  m → a  ('m' is followed by 'a')
  a → .  ('a' is followed by end)

Total: 5 training examples from one 4-letter word!
```

**Why It Matters**: This insight is crucial—datasets contain far more training signal than they might appear. A dataset of 32,000 names contains hundreds of thousands of character transitions to learn from.

---

### Segment: Building the Bigram Model - Counting Approach (4:25 - 9:50)

**Key verbatim excerpts**:
- "the simplest way in the bigram language models is to simply do it by counting"
- "we're going to store this information in a 2d array... the rows are going to be the first character... and the columns are going to be the second character"

**Technical analysis**: The first implementation method is detailed: constructing a count matrix `N` where `N[i, j]` is the frequency of character `j` following character `i`. This is a classic statistical language model. Special start (`.`) and end tokens are introduced to model word boundaries.

#### 💡 Intuitive Understanding: The Count Matrix

**Analogy - Frequency Table for Letters**: Imagine you're analyzing letter patterns in names. You create a giant table where each row is "what letter I just saw" and each column is "what letter came next." You go through all names, tallying each transition.

**Mental Model**: The count matrix is a 27×27 grid (26 letters + 1 special token):

```
           → a   b   c   d   e  ...  z   .
        ┌─────────────────────────────────
  . →   │  4   2   5   1  12  ...  0   0
  a →   │  0   3   1   2   1  ...  1   8
  b →   │  2   0   0   0   4  ...  0   0
  ...   │
  z →   │  1   0   0   0   2  ...  0   0

N[i,j] = "how many times did character j follow character i?"
```

**Why It Matters**: This simple counting approach is the foundation of statistical language models. It's fast, interpretable, and mathematically equivalent to a neural network bigram model—making it perfect for building intuition.

---

### Segment: Tensor Operations and Visualization (9:50 - 15:00)

**Key verbatim excerpts**:
- "tensors allow us to really manipulate all the individual entries and do it very efficiently"
- "we create a probability vector... we want to divide... to create a probability distribution"

**Technical analysis**: The count matrix is converted to a probability matrix `P` by normalizing each row. This segment emphasizes practical PyTorch skills: tensor creation, indexing, and the crucial operation of normalizing rows using `sum(dim=1, keepdim=True)`.

#### 💡 Intuitive Understanding: From Counts to Probabilities

**Analogy - Converting Votes to Percentages**: If 100 people voted on their favorite color, and 40 chose blue, 35 chose red, and 25 chose green, you convert counts to probabilities: 40% blue, 35% red, 25% green. This is exactly what we do for each row of our count matrix.

**Mental Model**: For each row (each context character), divide each count by the row sum:

```
Counts for what follows 'a':
  a→a: 20,  a→b: 50,  a→c: 10,  ...  (row sum = 400)

Probabilities:
  P(a|a) = 20/400 = 0.05
  P(b|a) = 50/400 = 0.125
  P(c|a) = 10/400 = 0.025
  ...

Now each row sums to 1.0 → valid probability distribution!
```

**Why It Matters**: This normalization step is the bridge between "how often did X happen" and "what's the probability X will happen." It's the same mathematical operation that appears in softmax.

---

### Segment: Sampling from the Bigram Model (15:00 - 30:00)

**Key verbatim excerpts**:
- "to sample from these distributions we're going to use torch.multinomial"
- "the reason these samples are so terrible is that bigram language model is actually look just like really terrible"

**Technical analysis**: The process of autoregressive sampling is implemented: start with the start token, repeatedly sample the next character from the probability distribution given the current character, and stop at the end token. The poor quality of the generated names highlights the limitation of the bigram model's limited context.

#### 💡 Intuitive Understanding: Autoregressive Sampling

**Analogy - Rolling Loaded Dice**: Imagine you have 27 dice (one for each starting character), and each die is "loaded" differently based on what typically follows that character. You start with the "start" die, roll it, note the result, pick up that result's die, roll again, and repeat until you roll "end."

**Mental Model**: The sampling loop is simple:

```python
current = START_TOKEN
output = ""
while True:
    probabilities = P[current]  # Get row for current character
    next_char = sample_from(probabilities)  # Roll the loaded die
    if next_char == END_TOKEN:
        break
    output += next_char
    current = next_char
```

**Why It Matters**: This autoregressive ("use your own previous outputs") sampling is how ALL language models generate text—from bigrams to GPT-4. The only difference is how sophisticated the probability prediction is.

---

### Segment: Efficiency and Broadcasting (30:00 - 44:00)

**Key verbatim excerpts**:
- "what i'd like to do is i'd like to actually prepare a matrix capital p that will just have the probabilities in it"
- "i encourage you to treat this with respect... you can very quickly run into bugs"

**Technical analysis**: The code is optimized by pre-computing the probability matrix `P`. This leads to a deep dive into PyTorch broadcasting rules. A critical bug is demonstrated where omitting `keepdim=True` during normalization silently normalizes columns instead of rows, emphasizing the importance of understanding tensor shapes.

#### 💡 Intuitive Understanding: Broadcasting Dangers

**Analogy - Miscommunication in Measurements**: Imagine telling someone to "divide the table by its totals." They could divide each row by its row total, or each column by its column total. Without specifying, you might get the wrong result—and no error message!

**Mental Model**: The `keepdim=True` trap:

```python
# Correct: keepdim=True preserves dimensions for broadcasting
counts = [[10, 20, 30], [5, 5, 10]]  # Shape: (2, 3)
row_sums = counts.sum(dim=1, keepdim=True)  # Shape: (2, 1)
# row_sums = [[60], [20]]
probs = counts / row_sums  # Broadcasting works correctly!
# probs = [[0.17, 0.33, 0.50], [0.25, 0.25, 0.50]]

# BUG: keepdim=False changes shape
row_sums_bad = counts.sum(dim=1)  # Shape: (2,) → becomes (1, 2) for broadcast
# row_sums_bad = [60, 20]
probs_bad = counts / row_sums_bad  # WRONG! Divides columns, not rows!
```

**Why It Matters**: Broadcasting bugs are silent and insidious—your code runs without errors but produces wrong results. This is one of the most common sources of bugs in deep learning code. Always verify tensor shapes!

---

### Segment: Model Evaluation and Loss Function (44:00 - 60:00)

**Key verbatim excerpts**:
- "our goal is to maximize likelihood which is the product of all the probabilities"
- "the negative log likelihood now is just negative of it and so the negative log likelihood is a very nice loss function"

**Technical analysis**: The model's quality is quantified. The likelihood of the dataset is introduced, then transformed into the log-likelihood for numerical stability, and finally into the negative log-likelihood (NLL) to create a standard loss function (where lower is better). The average NLL becomes the key training metric.

#### 💡 Intuitive Understanding: Why Negative Log Likelihood?

**Analogy - Golf Scoring**: In golf, lower scores are better. We want our loss function to work the same way. Likelihood is "higher is better," so we flip it. We also use log because multiplying thousands of tiny probabilities would underflow to zero.

**Mental Model**: The transformation chain:

```
Step 1: Likelihood = P(data) = P(char1) × P(char2) × ... × P(charN)
        Problem: Product of many small numbers → numerical underflow

Step 2: Log-Likelihood = log(P(data)) = log(P(char1)) + log(P(char2)) + ...
        Better: Sum of logs is numerically stable
        But: Higher is still better (want to maximize)

Step 3: Negative Log-Likelihood = -log(P(data))
        Perfect: Lower is better (standard loss function convention)
        Interpretation: "Surprise" or "bits needed to encode"
```

**Why It Matters**: NLL is THE standard loss function for language models. When you see "perplexity" or "cross-entropy loss," they're closely related to NLL. Understanding this metric is essential for evaluating and comparing language models.

---

### Segment: Neural Network Approach - Data Preparation (60:00 - 70:00)

**Key verbatim excerpts**:
- "we will end up in a very very similar position but the approach will look very different because i would like to cast the problem... into the neural network framework"
- "a common way of encoding integers is what's called one hot encoding"

**Technical analysis**: The paradigm shifts from explicit counting to parameterized neural network training. The training set is structured as input-output pairs (current character, next character). Inputs are converted into one-hot encoded vectors, which are suitable for linear layers in a neural network.

#### 💡 Intuitive Understanding: One-Hot Encoding

**Analogy - Answering "Which One?"**: Imagine a row of 27 light bulbs, one for each character. One-hot encoding means exactly one bulb is on, telling you which character you're representing.

**Mental Model**: Converting characters to vectors:

```
Character 'a' (index 0) → [1, 0, 0, 0, ..., 0]  (27 dimensions)
Character 'b' (index 1) → [0, 1, 0, 0, ..., 0]
Character 'c' (index 2) → [0, 0, 1, 0, ..., 0]
...
Character 'z' (index 25) → [0, 0, 0, ..., 1, 0]
```

**Why It Matters**: Neural networks need numerical inputs. One-hot encoding is the simplest way to represent categorical data (like characters) as numbers. It's also mathematically elegant—multiplying a one-hot vector by a weight matrix is equivalent to selecting a row from the matrix.

---

### Segment: Neural Network Architecture and Forward Pass (70:00 - 85:00)

**Key verbatim excerpts**:
- "our neural network is going to be a still a bigram character level language model so it receives a single character as an input"
- "we're going to interpret these to be the log counts... then these will be sort of the counts largest exponentiated"

**Technical analysis**: A single linear layer (27 inputs, 27 outputs) is defined. Its outputs are interpreted as *logits* (log counts). Applying the `softmax` function (exponentiate and normalize) converts these logits into a probability distribution over the next character, mirroring the count-and-normalize process.

#### 💡 Intuitive Understanding: Logits and Softmax

**Analogy - Raw Scores to Probabilities**: In a talent show, judges give raw scores (logits). To determine win probability, you exponentiate the scores (so higher is exponentially more likely) and normalize (so probabilities sum to 1). This is exactly softmax.

**Mental Model**: The softmax function step by step:

```
Logits (raw neural network output): [2.0, 1.0, 0.1]

Step 1: Exponentiate each value
  exp([2.0, 1.0, 0.1]) = [7.39, 2.72, 1.11]

Step 2: Normalize (divide by sum)
  sum = 7.39 + 2.72 + 1.11 = 11.22
  [7.39/11.22, 2.72/11.22, 1.11/11.22] = [0.66, 0.24, 0.10]

Result: Valid probability distribution that sums to 1.0!
```

**Why It Matters**: Softmax is everywhere in deep learning—classification, attention mechanisms, language models. Understanding it deeply (including numerical stability tricks) is essential.

---

### Segment: Loss Calculation and Gradient Setup (85:00 - 100:00)

**Key verbatim excerpts**:
- "the loss here is 3.7 something and you see that this loss... is exactly as we've obtained before but this is a vectorized form"
- "pytorch actually requires that we pass in requires grad is true"

**Technical analysis**: The NLL loss is calculated efficiently using vectorized indexing (`probs[torch.arange(5), ys]`). The computational graph is prepared for backpropagation by setting `requires_grad=True` on the weight tensor `W`.

#### 💡 Intuitive Understanding: Fancy Indexing

**Analogy - Looking Up Grades**: Imagine a gradebook where rows are students and columns are assignments. To get each student's grade on their assigned project, you don't look at every cell—you look up `gradebook[student_i, assignment_i]` for each student.

**Mental Model**: Selecting specific probabilities:

```python
# We have probabilities for each example (rows) and each possible next char (columns)
probs = [[0.1, 0.5, 0.2, ...],  # Example 0's distribution
         [0.3, 0.1, 0.4, ...],  # Example 1's distribution
         ...]

# We want the probability assigned to the ACTUAL next character
actual_next = [2, 0, 5, ...]  # What actually came next

# Fancy indexing extracts exactly what we need:
selected = probs[torch.arange(n), actual_next]
# selected = [probs[0,2], probs[1,0], probs[2,5], ...]
```

**Why It Matters**: This indexing pattern appears constantly in loss calculation. It efficiently extracts the model's predicted probability for the correct answer, which is exactly what we need for NLL loss.

---

### Segment: Backpropagation and Gradient Descent (100:00 - 110:00)

**Key verbatim excerpts**:
- "when you then calculate the loss we can call a dot backward on it and that backward then fills in the gradients"
- "we simply do w dot data plus equals... negative 0.1 times w dot grad"

**Technical analysis**: The training loop is implemented: forward pass, loss calculation, gradient zeroing (`w.grad = None`), backward pass (`loss.backward()`), and parameter update via gradient descent. This mirrors the training process from foundational frameworks like *micrograd*.

#### 💡 Intuitive Understanding: The Training Loop in Practice

**Analogy - Adjusting by Feedback**: Training is like adjusting your aim in darts. Throw (forward pass), see how far you missed (loss), figure out which way to adjust (backward pass), make the adjustment (update), and throw again.

**Mental Model**: The complete training loop:

```python
for step in range(1000):
    # Forward pass: predict
    logits = xenc @ W                    # Linear layer
    probs = softmax(logits)              # Convert to probabilities
    loss = -log(probs[correct]).mean()   # NLL loss

    # Backward pass: compute gradients
    W.grad = None                        # Zero gradients (CRITICAL!)
    loss.backward()                      # Fill in W.grad

    # Update: adjust parameters
    W.data -= 0.1 * W.grad               # Gradient descent step
```

**Why It Matters**: This exact loop structure—forward, loss, zero grad, backward, update—is used in virtually all neural network training. The details vary, but the structure is universal.

---

### Segment: Training, Equivalence, and Regularization (110:00 - 117:30)

**Key verbatim excerpts**:
- "we are achieving the roughly the same result but with gradient based optimization"
- "this w here is literally the same as this array here but w remember is the log counts"

**Technical analysis**: Training on the full dataset converges to a loss similar to the counting method, proving the neural network learns an equivalent probability matrix. The weight matrix `W` is shown to be the log of the count matrix. The concept of regularization (e.g., L2 weight decay) is introduced as the neural network analogue to count smoothing.

#### 💡 Intuitive Understanding: Why Are They Equivalent?

**Analogy - Two Paths Up the Mountain**: Counting and gradient descent are two different routes to the same summit. Counting gets there directly; gradient descent takes small steps guided by the slope. Both end up at the maximum likelihood solution.

**Mental Model**: The mathematical equivalence:

```
Counting approach:
  P[i,j] = N[i,j] / sum(N[i,:])
  (Divide counts by row sums)

Neural network (after training):
  W = log(N)                    # Weights become log-counts
  logits = one_hot @ W          # Select row = log-counts
  probs = softmax(logits)       # exp and normalize = P[i,j]

Same result, different path!
```

**Why It Matters**: This equivalence demonstrates that neural networks aren't magic—they're finding the same statistical patterns that explicit counting finds. But neural networks can scale to contexts where counting becomes impossible (too many combinations).

---

## Conclusion

**Summary of key technical takeaways**: A character-level bigram language model can be implemented via direct statistical counting or by training a single linear layer neural network with softmax and NLL loss. Both methods yield equivalent probability matrices. The neural network approach, while overkill for bigrams, provides a scalable framework for more complex models.

**Practical applications**: The established pipeline—data preparation, one-hot encoding, neural network forward pass, softmax, NLL loss calculation, backpropagation, and gradient descent—is the foundational training loop for all subsequent, more advanced language models.

**Long-term recommendations**: Master tensor operations and broadcasting rules to avoid subtle bugs. Understand the equivalence between statistical models and simple neural networks. The next step is to extend the context window beyond a single character and increase model complexity (e.g., MLPs, RNNs, Transformers) while keeping the core training framework intact.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Explain the trade-off**: Why is a bigram model fast to train but produces poor quality text? What would a trigram model gain and lose compared to a bigram?

2. **Probability check**: If your trained bigram model outputs P("z"|"q") = 0.8, what does this mean in plain English? Is this reasonable for English names?

3. **Loss interpretation**: Your bigram model achieves an average NLL of 2.5. Another model achieves 2.0. How much better is the second model in terms of likelihood? (Hint: think about what log means)

4. **Debugging scenario**: You train a neural network bigram model, but the loss stays at exactly `log(27) ≈ 3.3` and never decreases. What might be wrong?

5. **Connect to counting**: In the neural network approach, why do we initialize weights randomly instead of to zero? What would happen if all weights started at zero?

### Coding Challenges

1. **Build it yourself**: Implement the counting-based bigram model from scratch. Load a list of names, build the count matrix, convert to probabilities, and sample 10 new names.

   ```python
   # Starter structure:
   names = open('names.txt').read().splitlines()
   # Build count matrix N (27x27)
   # Convert to probability matrix P
   # Sample using torch.multinomial
   ```

2. **Evaluate your model**: Calculate the average negative log-likelihood of your bigram model on a held-out test set. Compare the loss on training data vs. test data—what do you observe?

3. **Trigram extension**: Extend your model to a trigram (use the previous 2 characters to predict the next). How does the count matrix size change? How does sample quality change?

4. **Smoothing experiment**: Add "smoothing" by adding a small constant (e.g., 1) to all counts before normalizing. How does this affect:
   - The loss on training data?
   - The loss on test data with rare character combinations?

5. **Visualize the model**: Create a heatmap visualization of your probability matrix P. What patterns do you see? Which characters are most likely to start a name? To end a name?

### Reflection

- **Scaling intuition**: The bigram model uses a 27×27 matrix. If you wanted to model 10 characters of context with 27 possible characters, how many entries would you need? Why is this approach called "the curse of dimensionality"?

- **Connection to GPT**: Modern language models like GPT predict the next *token* given *all previous tokens*. How is this similar to what we built? How is it different?

- **Sampling temperature**: What would happen if, instead of sampling from the probability distribution, you always picked the most likely next character? How would the generated names differ?
