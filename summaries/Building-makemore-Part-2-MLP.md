---
title: Building makemore Part 2: MLP
tags:
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

*Introduction*
• *Key objectives of the video*: To implement a multi-layer perceptron (MLP) character-level language model, moving beyond a simple bigram model to capture longer context without the exponential blow-up of a count-based approach. The tutorial follows the methodology from the Bengio et al. (2003) neural language model paper.
• *Core themes and methodologies*: Implementing an MLP with embedding lookup tables, hidden layers with non-linearities, and output logits. The process includes dataset construction, manual forward/backward pass implementation, optimization with gradient descent, hyperparameter tuning (like learning rate), and model evaluation using train/dev/test splits.
• *Target audience*: Individuals with foundational knowledge of neural networks and Python/PyTorch, looking to understand the practical implementation and training dynamics of early neural language models.

*Detailed Analysis*

• *0:00 - 1:47: Problem with Bigram Models and Introduction to MLPs*
    1. *Key excerpts*:
        • "if we are to take more context... things quickly blow up"
        • "the whole thing just kind of explodes and doesn't work very well"
    2. *Technical analysis and implications*: The lecturer explains the fundamental limitation of count-based N-gram models: the context window size is severely restricted due to exponential growth in the number of possible contexts (*27^k* for *k* characters). This "curse of dimensionality" makes statistical estimation infeasible and motivates the shift to parameterized neural models that can generalize.

• *1:48 - 4:12: Overview of the Bengio et al. (2003) Model*
    1. *Key excerpts*:
        • "they propose to take every one of these words... and associate... a feature vector"
        • "these points or vectors are going to basically move around in this space"
    2. *Technical analysis and implications*: The paper's core innovation is using *learned, distributed representations* (embeddings) for words/characters. Similar entities become proximate in this learned vector space, allowing the model to generalize to unseen context sequences by leveraging similarity, not just exact matches. This is a foundational idea for all modern neural NLP.

• *4:18 - 5:42: Intuition for Generalization via Embeddings*
    1. *Key excerpts*:
        • "you can transfer knowledge through that embedding"
        • "you can generalize to novel scenarios"
    2. *Technical analysis and implications*: A concrete example ("a dog was running in a" vs. "the dog was running in a") illustrates the power of embeddings. The model learns that "a" and "the" have similar embeddings, enabling knowledge transfer. This semantic generalization is the key advantage over rigid, count-based models.

• *5:43 - 8:52: Neural Network Architecture Diagram*
    1. *Key excerpts*:
        • "lookup table C is a matrix that is 17,000 by say 30"
        • "this layer has 17,000 neurons... this is the expensive layer"
    2. *Technical analysis and implications*: The architecture is detailed: an embedding layer (C), a hidden layer with tanh activation, and a large output softmax layer. The high cost of the final layer (due to large vocabulary size) is noted, foreshadowing a major challenge in scaling such models that later architectures (like word2vec, transformers) would address.

• *9:02 - 12:03: Building the Dataset*
    1. *Key excerpts*:
        • "block size... is basically the context length"
        • "we're always padding with dots"
    2. *Technical analysis and implications*: The code creates (input, target) pairs using a sliding window over the text. A fixed `block_size` defines the context length. Padding with a special token (`.`) at the start handles contexts shorter than the block size, a common technique in sequence modeling.

• *12:21 - 17:03: Implementing Embedding Lookup*
    1. *Key excerpts*:
        • "to embed simultaneously all of the integers in x we can simply do `c[x]`"
        • "pytorch indexing is awesome"
    2. *Technical analysis and implications*: Demonstrates the equivalence between one-hot encoding followed by a linear layer and direct integer indexing into an embedding matrix (`C`). The latter is vastly more efficient. PyTorch's flexible tensor indexing (`c[x]`) allows batch processing, a crucial efficiency gain.

• *18:37 - 29:00: Constructing the Hidden Layer and Tensor Manipulation*
    1. *Key excerpts*:
        • "how do we transform this 32 by 3 by 2 into a 32 by 6"
        • "`view` is an extremely efficient operation"
    2. *Technical analysis and implications*: Compares methods to flatten context embeddings: explicit concatenation (`torch.cat`) vs. using `torch.view`. The `view` operation is preferred as it manipulates tensor metadata (shape, stride) without copying data, making it memory-efficient. This highlights the importance of understanding tensor memory layout for performance.

• *29:41 - 33:38: Forward Pass and Loss Calculation*
    1. *Key excerpts*:
        • "we want to index into the rows of `prob`... pluck out the probability assigned to the correct character"
        • "the loss here is 17"
    2. *Technical analysis and implications*: Manually computes the forward pass to produce logits, applies softmax to get probabilities, and calculates negative log-likelihood loss. This step-by-step build is pedagogical, reinforcing how the network's output is interpreted as a probability distribution over the next character.

• *33:38 - 37:57: Using `F.cross_entropy` and Numerical Stability*
    1. *Key excerpts*:
        • "many good reasons to prefer `F.cross_entropy` over rolling your own"
        • "pytorch... internally calculates the maximum value... and subtracts it"
    2. *Technical analysis and implications*: Advocates for using the built-in `cross_entropy` function for efficiency (fused kernels), simpler gradients, and *numerical stability*. The explanation of the log-sum-exp trick (subtracting the max logit) is critical, showing how to avoid overflow when exponentiating large numbers, a vital practical consideration.

• *37:59 - 44:31: Training Loop and Mini-Batching*
    1. *Key excerpts*:
        • "we're doing way too much work forwarding and backwarding 220,000 examples"
        • "it's much better to have an approximate gradient and just make more steps"
    2. *Technical analysis and implications*: Implements gradient descent with manual zeroing of gradients (`p.grad = None`), backward pass, and parameter update. Introduces *mini-batching* by randomly sampling indices, drastically improving iteration speed. This demonstrates the standard stochastic gradient descent (SGD) training paradigm.

• *44:45 - 52:00: Learning Rate Selection and Decay*
    1. *Key excerpts*:
        • "how do you determine this learning rate"
        • "we are spaced exponentially in this interval"
    2. *Technical analysis and implications*: Introduces a practical method for learning rate search: training over a range of rates (e.g., *10^{-3}* to *10^{0}*) spaced exponentially and plotting loss. The "sweet spot" is where loss decreases steadily without exploding. Also introduces *learning rate decay* as a common technique to refine optimization in later stages.

• *52:00 - 56:19: Train/Dev/Test Splits and Overfitting*
    1. *Key excerpts*:
        • "the standard in the field is to split up your data set into three splits"
        • "you are only allowed to test... very very few times"
    2. *Technical analysis and implications*: Emphasizes rigorous evaluation by splitting data into training (parameter optimization), development/validation (hyperparameter tuning), and test (final evaluation) sets. This framework is essential to diagnose *overfitting* (gap between train and dev loss) and ensure the model generalizes.

• *56:31 - 70:59: Model Scaling and Hyperparameter Experimentation*
    1. *Key excerpts*:
        • "increasing the size of the model should help the neural net"
        • "the bottleneck... could be these embeddings that are two dimensional"
    2. *Technical analysis and implications*: Systematically explores increasing model capacity: more hidden neurons and larger embedding dimensions. The process is empirical—observe train/dev loss, identify potential bottlenecks (e.g., low-dimensional embeddings), adjust, and re-train. This mirrors the experimental workflow in deep learning research.

• *71:00 - 73:18: Visualization of Learned Embeddings*
    1. *Key excerpts*:
        • "the vowels a e i o u are clustered up here"
        • "q is kind of treated as an exception"
    2. *Technical analysis and implications*: Visualizing 2D character embeddings reveals the model has learned meaningful structure: vowels cluster together, while rare letters like 'q' are outliers. This provides intuitive, post-hoc interpretability for the learned representations, validating the embedding approach.

• *73:24 - 74:55: Sampling from the Trained Model*
    1. *Key excerpts*:
        • "we're going to generate 20 samples"
        • "the words here are much more word like or name like"
    2. *Technical analysis and implications*: Demonstrates the generative capability of the model. The sampling algorithm feeds the model's own predictions back as context in an autoregressive loop. The improved quality of samples (compared to the bigram model) is a tangible, qualitative measure of the model's success.

*Conclusion*
• *Summary of key technical takeaways*: Successfully implemented an MLP-based language model that overcomes the statistical limitations of N-grams by using learned embeddings for generalization. Key implementation skills included tensor manipulation (`view`, indexing), manual forward/backward passes, mini-batch SGD, learning rate tuning, and proper dataset splitting for evaluation.
• *Practical applications*: The core architecture forms the historical basis for neural language modeling. The principles of embedding lookup, multi-layer transformations, and cross-entropy training are foundational to more advanced models (RNNs, Transformers). The tutorial workflow is directly applicable to training simple neural networks for other classification/regression tasks.
• *Long-term recommendations*: To improve the model, experiment with hyperparameters (embedding size, hidden layer size, context length), implement more advanced optimizers (Adam), add regularization (dropout, weight decay), and explore the extensions suggested in the Bengio et al. paper. The next conceptual step is to move towards recurrent neural networks (RNNs) or transformers to handle arbitrarily long contexts more effectively.