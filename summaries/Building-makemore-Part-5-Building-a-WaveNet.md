---
title: Building makemore Part 5: Building a WaveNet
tags:
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

• **1:41 - 6:53: Code Refactoring and PyTorch-ification**
    1. "we want to think of these modules as building blocks and like a Lego building block bricks"
    2. "let's create layers for these and then we can add those layers to just our list"
    *Technical analysis*: The code is refactored to increase modularity. Custom `Embedding` and `Flatten` layers are created, and a `Sequential` container is implemented to manage layers. This mirrors PyTorch's `torch.nn` API, simplifying the forward pass and improving code organization.

• **6:58 - 9:16: Visualization and Debugging**
    1. "we need to average up some of these values to get a more sort of representative value"
    2. "we see that we basically made a lot of progress and then here this is the learning rate decay"
    *Technical analysis*: Demonstrates practical debugging and visualization. A rolling average is applied to the noisy loss curve for better interpretation, revealing the impact of learning rate decay on optimization convergence.

• **9:19 - 17:02: Implementing Hierarchical Structure**
    1. "we don't want to Matrix multiply 80... immediately instead we want to group these"
    2. "we want this to be a 4 by 4 by 20 where basically every two consecutive characters are packed"
    *Technical analysis*: The core architectural shift. Instead of flattening all context characters at once, they are grouped (e.g., in pairs). A new `FlattenConsecutive` layer is created to reshape the tensor (e.g., `[B, T, C]` to `[B, T//n, C*n]`), allowing linear layers to fuse small groups of characters in parallel across a new "batch" dimension.

• **17:02 - 20:51: Baseline Performance and Bug Discovery**
    1. "simply scaling up the context length from 3 to 8 gives us a performance of 2.02"
    2. "the batch norm is not doing what we need what we wanted to do"
    *Technical analysis*: Increasing context improves performance, establishing a baseline. A critical bug is found: the custom `BatchNorm1d` layer was incorrectly handling 3D inputs by only normalizing over the first dimension, not treating the new grouping dimension as part of the batch. This led to independent statistics per position, reducing stability.

• **20:51 - 24:33: Fixing BatchNorm and Final Results**
    1. "the dimension we want to reduce over is either 0 or the Tuple zero and one depending on the dimensionality"
    2. "we went from 2.029 to 2.022... we're estimating them using 32 times 4 numbers"
    *Technical analysis*: The `BatchNorm1d` is fixed to normalize over all dimensions except the channel dimension (e.g., dims=(0,1) for 3D input). This correctly pools statistics across the batch and the intra-example groups, leading to more stable estimates and a slight performance improvement.

• **24:33 - 28:10: Scaling Up and Concluding Insights**
    1. "we are now getting validation performance of 1.993 so we've crossed over the 2.0 territory"
    2. "the use of convolutions is strictly for efficiency it doesn't actually change the model we've implemented"
    *Technical analysis*: Scaling the model (embedding size, hidden units) yields further gains. The lecturer explains that the implemented hierarchical structure is functionally equivalent to a dilated causal convolutional network (like WaveNet); convolutions are an implementation optimization that reuses computations across the sequence.

• **28:10 - End: Development Process and Future Directions**
    1. "there's a ton of trying to make the shapes work and there's a lot of gymnastics"
    2. "I very often prototype these layers and implementations in jupyter notebooks"
    *Technical analysis*: Provides meta-commentary on the deep learning development workflow: heavy reliance on documentation (despite its flaws), meticulous shape debugging, prototyping in notebooks, and transferring to code repositories. Outlines future topics: convolutional implementation, residual/skip connections, experimental harnesses, RNNs, and Transformers.

*Conclusion*
• *Summary of key technical takeaways*: Hierarchical, progressive fusion of context (via grouping and flattening) is a powerful architecture for sequence modeling. Careful tensor shape manipulation is fundamental. Batch normalization must correctly aggregate statistics over all relevant batch dimensions. Modular layer design greatly simplifies building and experimenting with complex networks.
• *Practical applications*: The principles are directly applicable to implementing and understanding modern autoregressive models (WaveNet, Transformers). The debugging process (shape checking, loss visualization, fixing layer states) is essential for real-world model development.
• *Long-term recommendations*: Build a robust experimental harness for hyperparameter tuning. Implement the discussed efficiency optimizations using causal convolutions. Explore advanced architectural components like gated activations and residual connections from the WaveNet paper. Transition to using `torch.nn` directly now that its internal workings are understood.