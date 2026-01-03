---
title: Building makemore Part 3: Activations & Gradients, BatchNorm
tags:
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

• **4:19 - Diagnosing Improper Initialization**
    • *Excerpts:*
        • "the network is very confidently wrong... record very high loss"
        • "we want the logits to be roughly zero... at initialization"
    • *Technical analysis and implications:* Shows that extreme, uncalibrated logits from the final layer lead to a high initial loss. The expected loss for a uniform distribution over 27 classes is ~3.29 (negative log of 1/27). A much higher initial loss indicates poorly scaled weights, forcing the early optimization to merely squash weights instead of learning useful features.

• **8:54 - Fixing Output Layer Initialization**
    • *Excerpts:*
        • "B2 is just... zero at initialization"
        • "scale down W2 by 0.1"
    • *Technical analysis and implications:* Zero-initializing the final bias and scaling the final weight matrix ensures the softmax input (logits) starts near zero. This yields the expected uniform prediction and eliminates the initial "hockey stick" loss curve, leading to more efficient training and slightly improved final loss.

• **13:00 - Analyzing Hidden Layer Saturation**
    • *Excerpts:*
        • "many of the elements are one or negative one"
        • "if the outputs of your tanh are very close to 1... you're going to get a zero... killing the gradient"
    • *Technical analysis and implications:* Pre-activations that are too large cause tanh neurons to saturate, residing in flat regions where the local gradient (1 - t²) approaches zero. This impedes gradient flow during backpropagation, slowing or preventing learning in affected neurons.

• **24:07 - Fixing Hidden Layer Initialization**
    • *Excerpts:*
        • "H preact is too far off from zero... we want this preactivation to be closer to zero"
        • "multiply everything by 0.1"
    • *Technical analysis and implications:* Scaling down the weights (`W1`) feeding into the hidden layer reduces the magnitude of pre-activations. This prevents saturation, ensures gradients can flow effectively, and improves the final validation loss.

• **28:01 - Introducing Kaiming Initialization**
    • *Excerpts:*
        • "how do we scale these W's to preserve... [a] distribution to remain Gaussian?"
        • "you are supposed to divide by the square root of the fan-in"
    • *Technical analysis and implications:* For a linear layer with Gaussian input (mean 0, std 1), multiplying by weights with std `1/sqrt(fan_in)` preserves the output's standard deviation at 1. This is the core principle of variance-preserving initialization.

• **32:34 - Kaiming Initialization for Nonlinearities**
    • *Excerpts:*
        • "for the case of ReLU... you have to compensate... with a gain"
        • "they find that... standard deviation is sqrt(2 / fan_in)"
    • *Technical analysis and implications:* Nonlinearities like ReLU (which zeros out half the distribution) and tanh (a contracting function) distort the variance. A *gain* factor (e.g., `sqrt(2)` for ReLU, `5/3` for tanh) must be applied to the weight initialization to compensate and maintain stable activations throughout the network.

• **40:48 - Introducing Batch Normalization**
    • *Excerpts:*
        • "why not take the hidden States and... just normalize them to be Gaussian?"
        • "it's a perfectly differentiable operation"
    • *Technical analysis and implications:* Batch Norm (BN) explicitly standardizes a layer's inputs to have zero mean and unit variance *per batch*. This forcefully creates the desired activation distribution, making the network less sensitive to initial weight scaling.

• **46:19 - Batch Normalization Scale and Shift**
    • *Excerpts:*
        • "we have to also introduce this additional component... scale and shift"
        • "BN gain will multiply... and the BN bias will offset"
    • *Technical analysis and implications:* The learnable parameters `gamma` (scale) and `beta` (shift) after normalization restore the network's expressive power. Without them, the normalized distribution would be fixed, limiting the model's capacity.

• **52:51 - Batch Norm's Side Effects and Inference**
    • *Excerpts:*
        • "the examples in the batch are coupled mathematically"
        • "at test time we are going to fix [mean and std] and use them during inference"
    • *Technical analysis and implications:* BN introduces dependency between examples in a batch, which acts as a regularizer. For inference, running estimates of mean and variance (`running_mean`, `running_var`) are used instead of batch statistics, enabling evaluation on single examples.

• **60:56 - Practical Details of Batch Norm**
    • *Excerpts:*
        • "Epsilon is... preventing a division by zero"
        • "biases now are actually useless because... whatever bias you add here is going to get subtracted"
    • *Technical analysis and implications:* A small `epsilon` stabilizes training. When using BN after a linear/conv layer, the bias term in that preceding layer becomes redundant and should be omitted, as its effect is canceled by the subsequent subtraction of the mean.

• **74:11 - Module-based Implementation & Diagnostics**
    • *Excerpts:*
        • "torchify our code... structure our code into these modules"
        • "looking at the activation statistics both in the forward pass and in the backward pass"
    • *Technical analysis and implications:* Refactoring into modular layers (Linear, BatchNorm1d, Tanh) mimics PyTorch's design, improving code clarity and reusability. Systematic plotting of activation/gradient histograms and the *update-to-data ratio* provides crucial diagnostics for training health.

• **99:56 - The Update-to-Data Ratio**
    • *Excerpts:*
        • "what matters... is the update to the data ratio"
        • "should be roughly 1e-3... a good rough heuristic"
    • *Technical analysis and implications:* The ratio `(learning_rate * gradient_std) / weight_std` measures the relative change applied to parameters per step. A log value near -3 (~0.001) indicates a well-calibrated learning rate. Values much higher suggest unstable updates; much lower suggest slow learning.

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