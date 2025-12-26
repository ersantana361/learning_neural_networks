# Neural Networks: Zero to Hero

A structured learning project based on **Andrej Karpathy's Neural Networks: Zero to Hero** course.

## Overview

This course teaches neural networks from scratch, in code. Starting with the basics of backpropagation, we build up to modern deep neural networks like GPT. The focus is on language models as an excellent entry point to deep learning—most concepts transfer directly to other areas like computer vision.

**Instructor**: Andrej Karpathy (Former Director of AI at Tesla, OpenAI founding member)
**Source**: [Course Website](https://karpathy.ai/zero-to-hero.html)
**Total Duration**: ~14 hours of video content
**Discord Community**: [Join the discussion](https://discord.gg/3zy8kqD9Cp)

## Learning Objectives

By completing this course, you will be able to:

- Implement backpropagation from scratch (micrograd)
- Build character-level language models (makemore)
- Train multilayer perceptrons with proper initialization
- Understand activation statistics and gradient flow
- Implement Batch Normalization and understand its purpose
- Manually backpropagate through complex networks
- Build convolutional architectures (WaveNet-style)
- Implement a GPT model from scratch (nanoGPT)
- Build a BPE tokenizer from scratch

## Prerequisites

- **Required**: Solid Python programming skills
- **Required**: Intro-level math (derivatives, gaussian distributions)
- **Helpful**: Basic familiarity with NumPy
- **Helpful**: High school calculus

## Course Structure

### Part 1: Foundations

| # | Lecture | Duration | Description |
|---|---------|----------|-------------|
| 1 | [Building micrograd](https://youtu.be/VMj-3S1tku0) | 2h 25m | Step-by-step backpropagation and neural network training from scratch. Build a tiny autograd engine. |
| 2 | [Building makemore](https://youtu.be/PaCmpygFfXo) | 1h 57m | Bigram character-level language model. Introduction to `torch.Tensor`, language modeling framework, and negative log likelihood loss. |

### Part 2: MLPs and Training Dynamics

| # | Lecture | Duration | Description |
|---|---------|----------|-------------|
| 3 | [MLP](https://youtu.be/TCH_1BHY58I) | 1h 15m | Multilayer perceptron language model. ML basics: learning rate, hyperparameters, train/dev/test splits, under/overfitting. |
| 4 | [Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc) | 1h 55m | Forward pass activations, backward pass gradients, diagnostic tools. Introduction to Batch Normalization. |
| 5 | [Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI) | 1h 55m | Manual backpropagation through entire network without autograd. Deep intuition for gradient flow at tensor level. |

### Part 3: Advanced Architectures

| # | Lecture | Duration | Description |
|---|---------|----------|-------------|
| 6 | [Building a WaveNet](https://youtu.be/t3YJ5hKiMQ0) | 56m | Tree-like hierarchical architecture, convolutional neural networks. Deep dive into `torch.nn` internals. |
| 7 | [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 1h 56m | Build GPT from scratch following "Attention is All You Need". Self-attention, transformer blocks, training loop. |
| 8 | [GPT Tokenizer](https://youtu.be/zduSFxRajkE) | 2h 13m | Build BPE tokenizer from scratch. Understanding tokenization's role in LLM behavior and limitations. |

## Progress Tracker

### Part 1: Foundations
- [ ] 1. Building micrograd (2h 25m)
  - [ ] Watch video
  - [ ] Implement micrograd from scratch
  - [ ] Complete exercises
- [ ] 2. Building makemore (1h 57m)
  - [ ] Watch video
  - [ ] Implement bigram model
  - [ ] Experiment with torch.Tensor

### Part 2: MLPs and Training Dynamics
- [ ] 3. MLP (1h 15m)
  - [ ] Watch video
  - [ ] Implement MLP language model
  - [ ] Experiment with hyperparameters
- [ ] 4. Activations & Gradients, BatchNorm (1h 55m)
  - [ ] Watch video
  - [ ] Visualize activation/gradient statistics
  - [ ] Implement BatchNorm
- [ ] 5. Becoming a Backprop Ninja (1h 55m)
  - [ ] Watch video
  - [ ] Manual backprop through full network
  - [ ] Complete all gradient derivations

### Part 3: Advanced Architectures
- [ ] 6. Building a WaveNet (56m)
  - [ ] Watch video
  - [ ] Implement hierarchical model
  - [ ] Understand torch.nn internals
- [ ] 7. Let's build GPT (1h 56m)
  - [ ] Watch video
  - [ ] Implement GPT from scratch
  - [ ] Train on Shakespeare dataset
- [ ] 8. GPT Tokenizer (2h 13m)
  - [ ] Watch video
  - [ ] Implement BPE tokenizer
  - [ ] Understand tokenization issues

## Key Concepts by Lecture

### Lecture 1: micrograd
- Computational graphs
- Forward pass / backward pass
- Chain rule and backpropagation
- Gradient descent optimization
- `Value` class with autograd

### Lecture 2: makemore (Bigram)
- Character-level language modeling
- `torch.Tensor` operations
- Negative log likelihood loss
- Model training loop
- Sampling from the model

### Lecture 3: MLP
- Embedding layers
- Hidden layers and activations
- Learning rate schedules
- Train/dev/test splits
- Overfitting vs underfitting

### Lecture 4: Activations & Gradients
- Activation statistics (dead neurons, saturation)
- Gradient statistics (vanishing/exploding)
- Kaiming/He initialization
- Batch Normalization
- Diagnostic visualizations

### Lecture 5: Backprop Ninja
- Manual gradient computation
- Cross-entropy loss gradient
- Linear layer gradients
- Tanh gradient
- BatchNorm gradients
- Embedding gradients

### Lecture 6: WaveNet
- Hierarchical/tree architectures
- Dilated convolutions (concept)
- `torch.nn.Module`
- `torch.nn.Sequential`
- Development workflow

### Lecture 7: GPT
- Self-attention mechanism
- Multi-head attention
- Transformer blocks
- Positional encodings
- Causal masking
- "Attention is All You Need" paper

### Lecture 8: Tokenizer
- Byte Pair Encoding (BPE)
- `encode()` and `decode()`
- Unicode and UTF-8
- Special tokens
- Tokenization artifacts in LLMs

## Companion Resources

### Official GitHub Repositories
| Repository | Description | Link |
|------------|-------------|------|
| micrograd | Tiny autograd engine (~100 lines) | [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd) |
| makemore | Character-level language model | [github.com/karpathy/makemore](https://github.com/karpathy/makemore) |
| nanoGPT | Minimal GPT implementation | [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) |
| minbpe | Minimal BPE tokenizer | [github.com/karpathy/minbpe](https://github.com/karpathy/minbpe) |

### Essential Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017) - The Transformer
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) - DeepMind (2016)
- [Batch Normalization](https://arxiv.org/abs/1502.03167) - Ioffe & Szegedy (2015)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - BPE for NLP

### Additional Learning
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Visual intuition
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - Jay Alammar
- [minGPT](https://github.com/karpathy/minGPT) - Karpathy's earlier minimal GPT

### Books
- *Deep Learning* - Goodfellow, Bengio, Courville (free online)
- *Neural Networks and Deep Learning* - Michael Nielsen (free online)

## Project Structure

```
neural_networks/
├── README.md                    # This file
├── notes/                       # Personal annotations per lecture
│   ├── 01_micrograd.md
│   ├── 02_makemore_bigram.md
│   └── ...
├── code/                        # Your implementations
│   ├── micrograd/               # Lecture 1
│   │   └── micrograd.py
│   ├── makemore/                # Lectures 2-6
│   │   ├── bigram.py
│   │   ├── mlp.py
│   │   └── wavenet.py
│   ├── gpt/                     # Lecture 7
│   │   └── gpt.py
│   └── tokenizer/               # Lecture 8
│       └── bpe.py
├── notebooks/                   # Jupyter notebooks for experimentation
└── data/                        # Training data (shakespeare, names.txt, etc.)
```

## Suggested Learning Path

1. **Week 1-2**: Lectures 1-2 (Foundations)
   - Implement micrograd completely from scratch
   - Build bigram model, understand torch basics

2. **Week 3-4**: Lectures 3-5 (MLPs & Training)
   - Build MLP, experiment with hyperparameters
   - Study activation/gradient dynamics
   - Master manual backpropagation

3. **Week 5**: Lecture 6 (WaveNet)
   - Understand hierarchical architectures
   - Get comfortable with torch.nn

4. **Week 6-7**: Lectures 7-8 (GPT & Tokenizer)
   - Build GPT from scratch
   - Implement BPE tokenizer
   - Train on real data

## Tips for Success

1. **Code along**: Don't just watch—pause and implement
2. **Break things**: Modify the code, see what breaks
3. **Visualize**: Plot losses, activations, gradients
4. **Read the papers**: They're referenced for a reason
5. **Join Discord**: Ask questions, share progress
6. **Repeat**: Watch lectures multiple times if needed

## License

Course content by Andrej Karpathy. Personal notes and implementations for educational purposes.

---

*Course Status: Ongoing - Karpathy may add more content*
