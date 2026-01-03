---
title: The spelled-out intro to language modeling: building makemore
tags:
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

*Introduction*
• *Key objectives of the video*: To build a character-level language model from scratch, starting with a simple bigram model. The video demonstrates both a statistical counting approach and a neural network approach using gradient-based optimization, establishing the foundation for more complex models like transformers.
• *Core themes and methodologies*: Character-level language modeling, bigram statistics, maximum likelihood estimation, neural network implementation, PyTorch tensor operations, backpropagation, and gradient descent.
• *Target audience*: Individuals with foundational programming and machine learning knowledge (familiarity with concepts like *micrograd* is assumed) seeking to understand the inner workings of language models and neural network training.

*Detailed Analysis*

• *0:00 - 1:30: Introduction to the Project*
    1. *"make more as the name suggests makes more of things that you give it"*
    2. *"under the hood make more is a character level language model"*
    • The presenter introduces the goal: creating a model that generates new, name-like strings. The core methodology is established as character-level sequence prediction, framing the problem for the audience.

• *1:30 - 4:25: Data Loading and Exploration*
    1. *"we have to realize here is that every single word here like isabella is actually quite a few examples packed in to that single word"*
    2. *"in the beginning what i'd like to start with is i'd like to start with building a bi-gram language model"*
    • The transcript explains how a single word provides multiple training examples (character transitions). It introduces the bigram model as a simple starting point, which only considers the previous character to predict the next, establishing a baseline model.

• *4:25 - 9:50: Building the Bigram Model - Counting Approach*
    1. *"the simplest way in the bigram language models is to simply do it by counting"*
    2. *"we're going to store this information in a 2d array... the rows are going to be the first character... and the columns are going to be the second character"*
    • The first implementation method is detailed: constructing a count matrix `N` where `N[i, j]` is the frequency of character `j` following character `i`. This is a classic statistical language model. Special start (`.`) and end tokens are introduced to model word boundaries.

• *9:50 - 15:00: Tensor Operations and Visualization*
    1. *"tensors allow us to really manipulate all the individual entries and do it very efficiently"*
    2. *"we create a probability vector... we want to divide... to create a probability distribution"*
    • The count matrix is converted to a probability matrix `P` by normalizing each row. This segment emphasizes practical PyTorch skills: tensor creation, indexing, and the crucial operation of normalizing rows using `sum(dim=1, keepdim=True)`.

• *15:00 - 30:00: Sampling from the Bigram Model*
    1. *"to sample from these distributions we're going to use torch.multinomial"*
    2. *"the reason these samples are so terrible is that bigram language model is actually look just like really terrible"*
    • The process of autoregressive sampling is implemented: start with the start token, repeatedly sample the next character from the probability distribution given the current character, and stop at the end token. The poor quality of the generated names highlights the limitation of the bigram model's limited context.

• *30:00 - 44:00: Efficiency and Broadcasting*
    1. *"what i'd like to do is i'd like to actually prepare a matrix capital p that will just have the probabilities in it"*
    2. *"i encourage you to treat this with respect... you can very quickly run into bugs"*
    • The code is optimized by pre-computing the probability matrix `P`. This leads to a deep dive into PyTorch broadcasting rules. A critical bug is demonstrated where omitting `keepdim=True` during normalization silently normalizes columns instead of rows, emphasizing the importance of understanding tensor shapes.

• *44:00 - 60:00: Model Evaluation and Loss Function*
    1. *"our goal is to maximize likelihood which is the product of all the probabilities"*
    2. *"the negative log likelihood now is just negative of it and so the negative log likelihood is a very nice loss function"*
    • The model's quality is quantified. The likelihood of the dataset is introduced, then transformed into the log-likelihood for numerical stability, and finally into the negative log-likelihood (NLL) to create a standard loss function (where lower is better). The average NLL becomes the key training metric.

• *60:00 - 70:00: Neural Network Approach - Data Preparation*
    1. *"we will end up in a very very similar position but the approach will look very different because i would like to cast the problem... into the neural network framework"*
    2. *"a common way of encoding integers is what's called one hot encoding"*
    • The paradigm shifts from explicit counting to parameterized neural network training. The training set is structured as input-output pairs (current character, next character). Inputs are converted into one-hot encoded vectors, which are suitable for linear layers in a neural network.

• *70:00 - 85:00: Neural Network Architecture and Forward Pass*
    1. *"our neural network is going to be a still a bigram character level language model so it receives a single character as an input"*
    2. *"we're going to interpret these to be the log counts... then these will be sort of the counts largest exponentiated"*
    • A single linear layer (27 inputs, 27 outputs) is defined. Its outputs are interpreted as *logits* (log counts). Applying the `softmax` function (exponentiate and normalize) converts these logits into a probability distribution over the next character, mirroring the count-and-normalize process.

• *85:00 - 100:00: Loss Calculation and Gradient Setup*
    1. *"the loss here is 3.7 something and you see that this loss... is exactly as we've obtained before but this is a vectorized form"*
    2. *"pytorch actually requires that we pass in requires grad is true"*
    • The NLL loss is calculated efficiently using vectorized indexing (`probs[torch.arange(5), ys]`). The computational graph is prepared for backpropagation by setting `requires_grad=True` on the weight tensor `W`.

• *100:00 - 110:00: Backpropagation and Gradient Descent*
    1. *"when you then calculate the loss we can call a dot backward on it and that backward then fills in the gradients"*
    2. *"we simply do w dot data plus equals... negative 0.1 times w dot grad"*
    • The training loop is implemented: forward pass, loss calculation, gradient zeroing (`w.grad = None`), backward pass (`loss.backward()`), and parameter update via gradient descent. This mirrors the training process from foundational frameworks like *micrograd*.

• *110:00 - 117:30: Training, Equivalence, and Regularization*
    1. *"we are achieving the roughly the same result but with gradient based optimization"*
    2. *"this w here is literally the same as this array here but w remember is the log counts"*
    • Training on the full dataset converges to a loss similar to the counting method, proving the neural network learns an equivalent probability matrix. The weight matrix `W` is shown to be the log of the count matrix. The concept of regularization (e.g., L2 weight decay) is introduced as the neural network analogue to count smoothing.

*Conclusion*
• *Summary of key technical takeaways*: A character-level bigram language model can be implemented via direct statistical counting or by training a single linear layer neural network with softmax and NLL loss. Both methods yield equivalent probability matrices. The neural network approach, while overkill for bigrams, provides a scalable framework for more complex models.
• *Practical applications*: The established pipeline—data preparation, one-hot encoding, neural network forward pass, softmax, NLL loss calculation, backpropagation, and gradient descent—is the foundational training loop for all subsequent, more advanced language models.
• *Long-term recommendations*: Master tensor operations and broadcasting rules to avoid subtle bugs. Understand the equivalence between statistical models and simple neural networks. The next step is to extend the context window beyond a single character and increase model complexity (e.g., MLPs, RNNs, Transformers) while keeping the core training framework intact.