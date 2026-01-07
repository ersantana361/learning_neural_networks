---
title: State of GPT | BRK216HFS
tags:
  - LLM Training Pipeline
  - Pretraining
  - Supervised Fine-Tuning (SFT)
  - Reinforcement Learning from Human Feedback (RLHF)
  - Model Alignment
  - Tokenization
  - Base Model vs Assistant
  - Prompt Engineering
  - Chain-of-Thought
  - System 1 vs System 2 Simulation
  - Tool-Augmented LLMs
  - Retrieval-Augmented Generation (RAG)
  - Parameter-Efficient Fine-Tuning
  - LLM Limitations
  - AI Safety
  - Human-in-the-Loop
  - Copilot Pattern
---

*Introduction*
• *Key objectives of the video*: To explain the multi-stage training process for creating GPT-based AI assistants and to provide practical guidance on effectively applying these models.
• *Core themes and methodologies*: The technical "recipe" for training large language models (pretraining, supervised fine-tuning, reinforcement learning from human feedback) and the cognitive differences between humans and LLMs that inform advanced prompting strategies.
• *Target audience*: Developers, researchers, and technically-inclined practitioners interested in understanding the creation and application of state-of-the-art language models.

*Detailed Analysis*

• **0:00 - Introduction & Talk Structure**
    • *Excerpts*: "tell you about the state of GPT... the rapidly growing ecosystem" / "partition the talk into two parts... how we train... and how we can use"
    • *Analysis*: Establishes the dual focus of the talk: the technical pipeline for creating models and the practical methodology for deploying them. Sets an expectation for a comprehensive overview.

### 💡 Intuitive Understanding

**Analogy:** Think of this talk as a guide to both how a car is manufactured (training pipeline) and how to drive it effectively (prompt engineering). You can be a great driver without building cars, but understanding the manufacturing helps you understand the vehicle's capabilities and quirks.

**Mental Model:** The two-part structure:
```
Part 1: How GPT is Made
  - Pretraining → SFT → RLHF
  - Understanding what each stage does

Part 2: How to Use GPT Effectively
  - Prompting strategies
  - System 2 simulation
  - Tool use and RAG
  - When NOT to use LLMs
```

**Why It Matters:** Most LLM users only see Part 2, but understanding Part 1 explains *why* certain prompts work, why the model sometimes fails, and how to work around limitations.

---

• **0:46 - The Four-Stage Training Recipe**
    • *Excerpts*: "four major stages, pretraining, supervised finetuning, reward modeling, reinforcement learning" / "pretraining stage... is where all of the computational work basically happens"
    • *Analysis*: Introduces the core framework for modern LLM development. Highlights the massive computational asymmetry, where pretraining consumes ~99% of resources, framing the subsequent stages as more accessible "finetuning."

### 💡 Intuitive Understanding

**Analogy:** The four stages are like education:
1. **Pretraining** = Elementary through college (massive, general knowledge)
2. **SFT** = Professional training (specific skills)
3. **Reward Modeling** = Learning what the boss values
4. **RLHF** = On-the-job feedback loops

**Mental Model:** The resource asymmetry:
```
Stage           | Compute | Data        | Outcome
----------------|---------|-------------|----------------------
Pretraining     | 99%     | Trillions   | Knows language, facts
SFT             | <1%     | Thousands   | Follows instructions
Reward Modeling | <1%     | Thousands   | Knows human preferences
RLHF            | <1%     | Thousands   | Optimized for preferences
```

**Why It Matters:** You can fine-tune an assistant for ~$100-1000. Pretraining from scratch costs $10M-100M+. The stages are designed to leverage the expensive pretraining work efficiently.

---

• **1:27 - Pretraining: Data, Tokenization, and Scale**
    • *Excerpts*: "Internet scale datasets with thousands of GPUs... months of training" / "tokenization... a lossless translation... into sequences of integers"
    • *Analysis*: Demystifies the initial data pipeline. Emphasizes the scale (trillions of tokens, billions in cost) and the foundational step of converting text to a model-native numerical format, which is crucial for understanding model inputs.

### 💡 Intuitive Understanding

**Analogy:** Tokenization is like translating text into a secret code that the model can read. "Hello world" becomes [15496, 995]. The model never sees letters—it sees sequences of these integer codes.

**Mental Model:** Scale of pretraining:
```
GPT-3 Scale:
  - 175 billion parameters
  - ~300 billion tokens of training data
  - Trained on ~10,000 GPUs
  - Cost: ~$4.6 million (at 2020 prices)
  - Months of continuous computation

What it learns:
  - Grammar and syntax (implicitly)
  - Facts about the world (compressed into weights)
  - Reasoning patterns (by predicting logical sequences)
  - Multiple languages, code, math, etc.
```

**Why It Matters:** The pretraining data is the model's "life experience." Quality and diversity of this data determines what the model knows and how it thinks. Garbage in, garbage out.

---

• **2:40 - Pretraining: The Core Language Modeling Task**
    • *Excerpts*: "transformer is going to try to predict the next token" / "low loss means... higher probability to the next correct integer"
    • *Analysis*: Explains the self-supervised learning objective. The model learns a powerful, general representation by performing the seemingly simple task of next-token prediction across a vast corpus, which implicitly teaches it grammar, facts, and reasoning.

### 💡 Intuitive Understanding

**Analogy:** Imagine learning a language by only playing one game: finish this sentence. After playing billions of times, you'd learn grammar, vocabulary, world knowledge, and reasoning—just by trying to guess the next word correctly.

**Mental Model:** What next-token prediction teaches:
```
Training example: "The capital of France is ___"
To predict "Paris", the model must:
  - Understand the question format
  - Know what a capital is
  - Know that France is a country
  - Know that Paris is France's capital

No explicit labels! The answer IS the next token.
```

**Why It Matters:** This is why LLMs seem "intelligent." They've compressed patterns from trillions of tokens—including how to solve problems, write code, and reason—into their weights. It's all emergent from next-token prediction.

---

• **8:08 - From Base Models to Assistants**
    • *Excerpts*: "base models are not assistants. They don't want to make answers... they want to complete documents" / "supervised finetuning stage... collect small but high quality datasets"
    • *Analysis*: Draws a critical distinction. Base models are document completers; creating a helpful assistant requires an additional alignment stage (SFT) using curated prompt-response pairs to shape behavior.

### 💡 Intuitive Understanding

**Analogy:** A base model is like a parrot that's read the entire internet. Ask it a question, and it might answer—or it might continue your sentence as if you're writing an article. SFT is like training the parrot to actually respond to questions in a helpful way.

**Mental Model:** Base model vs. Assistant:
```
Base Model Input: "What is 2+2?"
Base Model Output: "
What is 2+3?
What is 2+4?
..."  (completing a document of similar questions)

After SFT Input: "What is 2+2?"
After SFT Output: "2+2 equals 4."  (answering the question)

SFT teaches the model that questions deserve answers.
```

**Why It Matters:** This explains why you can't just download a pretrained model and expect it to be an assistant. The assistant behavior is a learned layer on top of language modeling. It also explains why base models can be better for certain tasks (completion, creative generation).

---

• **13:02 - Reinforcement Learning from Human Feedback (RLHF)**
    • *Excerpts*: "reward modeling step... shift our data collection to be of the form of comparisons" / "reinforcement learning... weighing the language modeling objective by the rewards"
    • *Analysis*: Details the advanced alignment technique. RLHF uses human preferences to train a reward model, which then guides the model via reinforcement learning to produce outputs humans prefer, often leading to more helpful and harmless responses.

### 💡 Intuitive Understanding

**Analogy:** SFT is like giving someone a manual. RLHF is like giving them ongoing feedback: "That response was good. That one was bad." Over time, they learn what you actually want, not just what's in the manual.

**Mental Model:** The RLHF process:
```
Step 1: Reward Modeling
  - Show humans pairs of model outputs
  - Human picks which is better
  - Train a "reward model" to predict human preferences

Step 2: Reinforcement Learning
  - Generate responses
  - Score them with the reward model
  - Update the LLM to produce higher-scoring responses

Result: Model learns nuanced preferences like:
  - Be helpful but not sycophantic
  - Admit uncertainty instead of making things up
  - Avoid harmful content
```

**Why It Matters:** RLHF is what makes ChatGPT feel "aligned" with human values. It's easier to say "this response is better" (comparison) than to write the perfect response (demonstration). RLHF leverages this.

---

• **17:14 - Model Comparisons and Trade-offs**
    • *Excerpts*: "RLHF models... work better... but they lose some entropy" / "base model... good at... generate more things like it"
    • *Analysis*: Provides a nuanced comparison. RLHF improves alignment but can reduce output diversity. The choice between base, SFT, and RLHF models depends on the use case (e.g., creative generation vs. precise assistance).

### 💡 Intuitive Understanding

**Analogy:** Base models are like jazz improvisers—they might go anywhere. RLHF models are like session musicians—they reliably play what you want, but they might not surprise you as much.

**Mental Model:** When to use each model type:
```
Base Model:
  ✓ Creative writing (needs diversity)
  ✓ Completion of existing content
  ✓ Generating training data
  ✗ Following specific instructions

SFT Model:
  ✓ Following instructions
  ✓ Specific task formats
  ✗ Complex value judgments

RLHF Model (ChatGPT-like):
  ✓ General assistance
  ✓ Following nuanced preferences
  ✓ Being helpful/harmless
  ✗ Maximum creativity
```

**Why It Matters:** "Best model" depends on the task. For research into model behavior, base models are informative. For production assistants, RLHF is typically better. Choose based on your use case.

---

• **20:26 - The Cognitive Gap: Humans vs. LLMs**
    • *Excerpts*: "a ton happens under the hood in terms of your internal monologue" / "GPT... just goes chunk, chunk, chunk... same amount of compute on every one"
    • *Analysis*: A pivotal conceptual framework. Contrasts human System 2 thinking (slow, deliberate, with tool use and reflection) with the LLM's default System 1-like token simulation, explaining the need for sophisticated prompting.

### 💡 Intuitive Understanding

**Analogy:** Humans have "draft" and "edit" modes. We can pause, reconsider, backtrack. LLMs commit to each word immediately, like speaking without thinking—every word is final once said.

**Mental Model:** System 1 vs System 2:
```
Human (System 2):
  - Can pause and think
  - Uses scratch paper
  - Checks work
  - Uses tools (calculator, search)
  - Backs up when stuck

LLM (default behavior):
  - No pause, just token → token → token
  - Same compute per token (simple or hard)
  - Can't backtrack
  - No inherent tool access
  - Commits to first response
```

**Why It Matters:** This explains why LLMs make mistakes humans wouldn't. They "think" the same amount about every token. Prompt engineering is about giving LLMs the "tokens to think"—space to reason before committing to an answer.

---

• **24:41 - Prompt Engineering as Compensatory Strategy**
    • *Excerpts*: "These transformers need tokens to think" / "prompting is just making up for this cognitive difference"
    • *Analysis*: Positions prompt engineering as a method to *simulate* deeper reasoning. Techniques like chain-of-thought ("let's think step-by-step") spread computation over more tokens, mimicking a slower, more reliable reasoning process.

### 💡 Intuitive Understanding

**Analogy:** If LLMs need to "talk through" problems to solve them (like showing your work in math class), then prompting is about giving them space to write out their reasoning.

**Mental Model:** Why "let's think step by step" works:
```
Without chain-of-thought:
  Q: "What's 17 * 24?"
  A: "408" (just guessing from pattern matching)

With chain-of-thought:
  Q: "What's 17 * 24? Think step by step."
  A: "17 * 24 = 17 * 20 + 17 * 4 = 340 + 68 = 408"

The intermediate tokens give the model "compute time."
Each token can build on the previous one.
```

**Why It Matters:** Prompt engineering isn't about tricks—it's about compensating for the LLM's cognitive architecture. Understanding *why* techniques work helps you invent new ones for your specific problems.

---

• **27:31 - Advanced Techniques: System 2 Simulation & Tool Use**
    • *Excerpts*: "recreating our System 2" / "symbiosis of Python Glue code and individual prompts"
    • *Analysis*: Describes the frontier of LLM application. Methods like Tree of Thoughts or ReAct use programmatic loops to manage multiple LLM calls, enabling planning, self-evaluation, and tool use (calculators, search) to overcome native model limitations.

### 💡 Intuitive Understanding

**Analogy:** Instead of asking for the answer, you give the LLM a process: "Generate ideas → Evaluate them → Pick the best → Refine it." Each step is a separate LLM call, connected by code.

**Mental Model:** System 2 patterns:
```
Simple prompting: One call → One answer

Chain-of-Thought: One call with reasoning → Answer

ReAct (Reason + Act):
  Loop:
    - LLM: "I should search for X"
    - Code: [executes search]
    - LLM: "Based on results, I think..."
    - Repeat until done

Tree of Thoughts:
  - Generate multiple approaches
  - Evaluate each
  - Pursue the best
  - Backtrack if needed
```

**Why It Matters:** This is where LLM applications are heading. Single-turn prompting is limited. Multi-turn, tool-augmented, reflective systems can solve much harder problems.

---

• **32:54 - Retrieval Augmentation and Fine-tuning**
    • *Excerpts*: "retrieval-augmented models... works extremely well" / "fine tuning... is becoming a lot more accessible"
    • *Analysis*: Covers two key scaling techniques. Retrieval augments the model's fixed context with relevant external data, while parameter-efficient fine-tuning (e.g., LoRA) allows cost-effective customization of model weights for specific tasks.

### 💡 Intuitive Understanding

**Analogy:** RAG is like giving the model access to a library during an exam (open-book test). Fine-tuning is like sending the model back to school for specialized training.

**Mental Model:** When to use which:
```
Retrieval-Augmented Generation (RAG):
  - Your data changes frequently
  - You need citations/sources
  - You have lots of documents
  - Model needs up-to-date information

Fine-tuning:
  - You need a specific style/format
  - The task is unusual
  - You want behavior changes
  - Speed/cost critical (smaller fine-tuned model)

Often combine both!
```

**Why It Matters:** RAG is often the better first choice—it's cheaper, more transparent, and doesn't risk "forgetting" general knowledge. Fine-tune when RAG isn't enough.

---

• **37:11 - Practical Recommendations & Limitations**
    • *Excerpts*: "best performance will currently come from GPT-4" / "use LLMs in low-stakes applications... think co-pilots"
    • *Analysis*: Offers a clear, prioritized action plan: start with detailed prompting on the most capable model, then optimize. Crucially, advises caution due to known limitations (hallucinations, bias, security vulnerabilities), recommending human-in-the-loop, "copilot" patterns.

### 💡 Intuitive Understanding

**Analogy:** LLMs are like very capable interns. They can draft emails, summarize documents, and generate ideas. But you don't let interns sign contracts or make final decisions without review.

**Mental Model:** The LLM deployment ladder:
```
1. Start with the best model (GPT-4 level)
2. Write detailed prompts with examples
3. Use retrieval for your domain data
4. Add chain-of-thought for reasoning tasks
5. Fine-tune only if needed

Safety rules:
- Hallucinations: Always verify facts
- Bias: Review outputs for fairness
- Security: Never trust LLM-generated code blindly
- Stakes: Use copilot pattern for high-stakes decisions
```

**Why It Matters:** LLMs are tools, not oracles. The "copilot" pattern—LLM drafts, human reviews—captures the right relationship. It maximizes value while managing risk.

---

*Conclusion*
• *Summary of key technical takeaways*: Modern LLM assistants are created through a sequential pipeline of large-scale pretraining followed by alignment stages (SFT, RLHF). Their operation is fundamentally different from human cognition, acting as token simulators without inherent reasoning, planning, or self-awareness.
• *Practical applications*: Maximize performance by using detailed prompts, few-shot examples, retrieval-augmentation, and chaining multiple calls with code. Treat the model's psychology as a key variable in prompt design. Fine-tuning is viable but more complex.
• *Long-term recommendations*: Approach LLMs as powerful but imperfect tools. Prioritize prompt engineering and system design around state-of-the-art models like GPT-4 for high-value tasks. Maintain human oversight, use models as copilots in low-stakes scenarios, and stay abreast of rapidly evolving techniques for reasoning and tool integration.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Training Stages:** Describe the purpose of each stage (Pretraining, SFT, Reward Modeling, RLHF). What would happen if you skipped SFT and went straight from pretraining to RLHF?

2. **Base vs Assistant:** Give an example prompt where a base model would produce a fundamentally different output than an RLHF model. Why?

3. **System 1 vs 2:** Explain why LLMs are like "System 1" thinkers. What does this imply about the types of tasks they'll struggle with?

4. **Chain of Thought:** Why does "let's think step by step" improve reasoning accuracy? What's happening mechanically?

5. **RAG vs Fine-tuning:** You're building a customer support bot for a company with 10,000 product pages. Would you use RAG, fine-tuning, or both? Justify your choice.

6. **Copilot Pattern:** What is the "copilot" pattern and why is it recommended for high-stakes applications?

### Prompt Engineering Challenges

1. **Transform Base to Assistant:**
   Write a system prompt that could make a base model (document completer) behave more like an assistant. Test it and iterate.

2. **Chain-of-Thought Design:**
   Design a prompt that forces the model to reason through a multi-step math problem. Compare accuracy with and without chain-of-thought.

3. **Few-Shot Learning:**
   Create a 3-shot prompt for a classification task (e.g., sentiment analysis). Compare to zero-shot. How many examples are needed for good performance?

4. **ReAct Pattern:**
   Design a multi-turn prompt sequence where:
   - Step 1: LLM identifies what information is needed
   - Step 2: You provide that information (simulating a tool)
   - Step 3: LLM synthesizes a final answer

5. **Adversarial Testing:**
   Try to get the model to hallucinate by asking about obscure or nonexistent topics. How does the model's response differ between base and RLHF versions?

### Reflection

- The talk mentions that LLMs can have "jagged" capabilities—very good at some things, poor at others. Research examples of this (e.g., good at poetry, bad at counting). What does this imply about how LLMs represent knowledge?

- RLHF optimizes for human preferences, but human preferences can be biased or incorrect. What are the risks of this? Research "reward hacking" and its implications for AI alignment.

- The "copilot" pattern keeps humans in the loop. As models improve, how should the human-AI division of labor change? Is full autonomy a goal or a risk?

- The talk is from 2023. Research what has changed since then (new models, new techniques, new understanding). How has the "state of GPT" evolved?
