export const course = {
  title: "Neural Networks: Zero to Hero",
  instructor: "Andrej Karpathy",
  description: "Build neural networks from scratch, in code. From backpropagation to GPT.",
  website: "https://karpathy.ai/zero-to-hero.html",
  discord: "https://discord.gg/3zy8kqD9Cp",
  totalDuration: "~25 hours",

  parts: [
    {
      id: "part1",
      title: "Part 1: Foundations",
      lectures: [
        {
          id: 1,
          title: "Building micrograd",
          duration: "2h 25m",
          videoId: "VMj-3S1tku0",
          summaryFile: "The-spelled-out-intro-to-neural-networks-and-backpropagation-building-micrograd.md",
          description: "Step-by-step backpropagation and neural network training from scratch. Build a tiny autograd engine.",
          topics: [
            "Computational graphs",
            "Forward pass / backward pass",
            "Chain rule and backpropagation",
            "Gradient descent optimization",
            "Value class with autograd"
          ],
          tasks: [
            { id: "1-watch", label: "Watch video" },
            { id: "1-summary", label: "Read video summary" },
            { id: "1-implement", label: "Implement micrograd from scratch" },
            { id: "1-exercises", label: "Complete exercises" }
          ],
          codeFiles: ["code/micrograd/micrograd.py"],
          resources: [
            { label: "micrograd repo", url: "https://github.com/karpathy/micrograd" }
          ]
        },
        {
          id: 2,
          title: "Building makemore",
          duration: "1h 57m",
          videoId: "PaCmpygFfXo",
          summaryFile: "The-spelled-out-intro-to-language-modeling-building-makemore.md",
          description: "Bigram character-level language model. Introduction to torch.Tensor, language modeling framework, and negative log likelihood loss.",
          topics: [
            "Character-level language modeling",
            "torch.Tensor operations",
            "Negative log likelihood loss",
            "Model training loop",
            "Sampling from the model"
          ],
          tasks: [
            { id: "2-watch", label: "Watch video" },
            { id: "2-summary", label: "Read video summary" },
            { id: "2-implement", label: "Implement bigram model" },
            { id: "2-experiment", label: "Experiment with torch.Tensor" }
          ],
          codeFiles: ["code/makemore/bigram.py"],
          resources: [
            { label: "makemore repo", url: "https://github.com/karpathy/makemore" }
          ]
        }
      ]
    },
    {
      id: "part2",
      title: "Part 2: MLPs and Training Dynamics",
      lectures: [
        {
          id: 3,
          title: "MLP",
          duration: "1h 15m",
          videoId: "TCH_1BHY58I",
          summaryFile: "Building-makemore-Part-2-MLP.md",
          description: "Multilayer perceptron language model. ML basics: learning rate, hyperparameters, train/dev/test splits, under/overfitting.",
          topics: [
            "Embedding layers",
            "Hidden layers and activations",
            "Learning rate schedules",
            "Train/dev/test splits",
            "Overfitting vs underfitting"
          ],
          tasks: [
            { id: "3-watch", label: "Watch video" },
            { id: "3-summary", label: "Read video summary" },
            { id: "3-implement", label: "Implement MLP language model" },
            { id: "3-experiment", label: "Experiment with hyperparameters" }
          ],
          codeFiles: ["code/makemore/mlp.py"],
          resources: []
        },
        {
          id: 4,
          title: "Activations & Gradients, BatchNorm",
          duration: "1h 55m",
          videoId: "P6sfmUTpUmc",
          summaryFile: "Building-makemore-Part-3-Activations-&-Gradients,-BatchNorm.md",
          description: "Forward pass activations, backward pass gradients, diagnostic tools. Introduction to Batch Normalization.",
          topics: [
            "Activation statistics (dead neurons, saturation)",
            "Gradient statistics (vanishing/exploding)",
            "Kaiming/He initialization",
            "Batch Normalization",
            "Diagnostic visualizations"
          ],
          tasks: [
            { id: "4-watch", label: "Watch video" },
            { id: "4-summary", label: "Read video summary" },
            { id: "4-visualize", label: "Visualize activation/gradient statistics" },
            { id: "4-batchnorm", label: "Implement BatchNorm" }
          ],
          codeFiles: ["code/makemore/mlp.py"],
          resources: [
            { label: "Batch Normalization paper", url: "https://arxiv.org/abs/1502.03167" }
          ]
        },
        {
          id: 5,
          title: "Becoming a Backprop Ninja",
          duration: "1h 55m",
          videoId: "q8SA3rM6ckI",
          summaryFile: "Building-makemore-Part-4-Becoming-a-Backprop-Ninja.md",
          description: "Manual backpropagation through entire network without autograd. Deep intuition for gradient flow at tensor level.",
          topics: [
            "Manual gradient computation",
            "Cross-entropy loss gradient",
            "Linear layer gradients",
            "Tanh gradient",
            "BatchNorm gradients",
            "Embedding gradients"
          ],
          tasks: [
            { id: "5-watch", label: "Watch video" },
            { id: "5-summary", label: "Read video summary" },
            { id: "5-manual", label: "Manual backprop through full network" },
            { id: "5-derivations", label: "Complete all gradient derivations" }
          ],
          codeFiles: ["code/makemore/mlp.py"],
          resources: []
        }
      ]
    },
    {
      id: "part3",
      title: "Part 3: Advanced Architectures",
      lectures: [
        {
          id: 6,
          title: "Building a WaveNet",
          duration: "56m",
          videoId: "t3YJ5hKiMQ0",
          summaryFile: "Building-makemore-Part-5-Building-a-WaveNet.md",
          description: "Tree-like hierarchical architecture, convolutional neural networks. Deep dive into torch.nn internals.",
          topics: [
            "Hierarchical/tree architectures",
            "Dilated convolutions (concept)",
            "torch.nn.Module",
            "torch.nn.Sequential",
            "Development workflow"
          ],
          tasks: [
            { id: "6-watch", label: "Watch video" },
            { id: "6-summary", label: "Read video summary" },
            { id: "6-implement", label: "Implement hierarchical model" },
            { id: "6-torchnn", label: "Understand torch.nn internals" }
          ],
          codeFiles: ["code/makemore/wavenet.py"],
          resources: [
            { label: "WaveNet paper", url: "https://arxiv.org/abs/1609.03499" }
          ]
        },
        {
          id: 7,
          title: "Let's build GPT",
          duration: "1h 56m",
          videoId: "kCc8FmEb1nY",
          summaryFile: "Let's-build-GPT-from-scratch,-in-code,-spelled-out.md",
          description: "Build GPT from scratch following \"Attention is All You Need\". Self-attention, transformer blocks, training loop.",
          topics: [
            "Self-attention mechanism",
            "Multi-head attention",
            "Transformer blocks",
            "Positional encodings",
            "Causal masking",
            "\"Attention is All You Need\" paper"
          ],
          tasks: [
            { id: "7-watch", label: "Watch video" },
            { id: "7-summary", label: "Read video summary" },
            { id: "7-implement", label: "Implement GPT from scratch" },
            { id: "7-train", label: "Train on Shakespeare dataset" }
          ],
          codeFiles: ["code/gpt/gpt.py"],
          resources: [
            { label: "nanoGPT repo", url: "https://github.com/karpathy/nanoGPT" },
            { label: "Attention Is All You Need", url: "https://arxiv.org/abs/1706.03762" },
            { label: "The Illustrated Transformer", url: "https://jalammar.github.io/illustrated-transformer/" }
          ]
        },
        {
          id: 8,
          title: "State of GPT",
          duration: "42m",
          videoId: "bZQun8Y4L2A",
          summaryFile: "State-of-GPT-BRK216HFS.md",
          description: "Deep dive into how GPT assistants are trained: pretraining, supervised finetuning, RLHF, and practical prompting strategies.",
          topics: [
            "LLM training pipeline",
            "Pretraining at scale",
            "Supervised Fine-Tuning (SFT)",
            "Reinforcement Learning from Human Feedback (RLHF)",
            "Prompt engineering strategies",
            "Chain-of-thought reasoning"
          ],
          tasks: [
            { id: "8-watch", label: "Watch video" },
            { id: "8-summary", label: "Read video summary" },
            { id: "8-understand", label: "Understand training pipeline" }
          ],
          codeFiles: [],
          resources: [
            { label: "Microsoft Build 2023 Talk", url: "https://www.youtube.com/watch?v=bZQun8Y4L2A" }
          ]
        },
        {
          id: 9,
          title: "GPT Tokenizer",
          duration: "2h 13m",
          videoId: "zduSFxRajkE",
          summaryFile: "Let's-build-the-GPT-Tokenizer.md",
          description: "Build BPE tokenizer from scratch. Understanding tokenization's role in LLM behavior and limitations.",
          topics: [
            "Byte Pair Encoding (BPE)",
            "encode() and decode()",
            "Unicode and UTF-8",
            "Special tokens",
            "Tokenization artifacts in LLMs"
          ],
          tasks: [
            { id: "9-watch", label: "Watch video" },
            { id: "9-summary", label: "Read video summary" },
            { id: "9-implement", label: "Implement BPE tokenizer" },
            { id: "9-issues", label: "Understand tokenization issues" }
          ],
          codeFiles: ["code/tokenizer/bpe.py"],
          resources: [
            { label: "minbpe repo", url: "https://github.com/karpathy/minbpe" },
            { label: "BPE for NLP paper", url: "https://arxiv.org/abs/1508.07909" }
          ]
        },
        {
          id: 10,
          title: "Reproducing GPT-2",
          duration: "4h 1m",
          videoId: "l8pRSuU81PU",
          summaryFile: "Let's-reproduce-GPT-2-(124M).md",
          description: "Reproduce GPT-2 (124M) from scratch. Complete training pipeline with modern optimizations: mixed precision, Flash Attention, DDP.",
          topics: [
            "GPT-2 architecture implementation",
            "Mixed precision training (BF16)",
            "torch.compile optimization",
            "Flash Attention",
            "Distributed Data Parallel (DDP)",
            "Gradient accumulation",
            "HellaSwag evaluation"
          ],
          tasks: [
            { id: "10-watch", label: "Watch video" },
            { id: "10-summary", label: "Read video summary" },
            { id: "10-implement", label: "Implement GPT-2 training" },
            { id: "10-optimize", label: "Apply performance optimizations" }
          ],
          codeFiles: ["code/gpt/gpt.py"],
          resources: [
            { label: "build-nanogpt repo", url: "https://github.com/karpathy/build-nanogpt" },
            { label: "FineWeb-Edu dataset", url: "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu" }
          ]
        }
      ]
    }
  ]
};

// Helper to get all lectures flat
export const getAllLectures = () => {
  return course.parts.flatMap(part => part.lectures);
};

// Helper to get all task IDs
export const getAllTaskIds = () => {
  return getAllLectures().flatMap(lecture => lecture.tasks.map(t => t.id));
};

// Helper to get lecture by ID
export const getLectureById = (id) => {
  return getAllLectures().find(l => l.id === parseInt(id));
};

// Helper to get part containing a lecture
export const getPartForLecture = (lectureId) => {
  return course.parts.find(part =>
    part.lectures.some(l => l.id === parseInt(lectureId))
  );
};
