---
title: Let's build the GPT Tokenizer
tags:
tags:
  - Tokenization
  - Byte Pair Encoding (BPE)
  - Large Language Models (LLMs)
  - Subword Tokenization
  - Vocabulary
  - GPT-2
  - GPT-4
  - SentencePiece
  - UTF-8 Encoding
  - Preprocessing
  - Model Efficiency
  - Special Tokens
  - Regex Pre-tokenization
  - Encoding/Decoding
  - Tokenization Issues
  - Vocabulary Size
  - Embedding Table
  - tiktoken
---

*Introduction*
• *Key objectives of the video*: To explain the process, importance, and complexities of tokenization in large language models (LLMs), covering the Byte Pair Encoding (BPE) algorithm, its implementation, and practical implications.
• *Core themes and methodologies*: Tokenization as a fundamental pre-processing step; the BPE algorithm for building vocabularies; differences between character-level, subword, and byte-level tokenization; analysis of real-world tokenizers (GPT-2, GPT-4, SentencePiece); and common issues stemming from tokenization.
• *Target audience*: Developers, researchers, and enthusiasts working with or seeking to understand the inner workings of LLMs, particularly those interested in data preprocessing and model input pipelines.

*Detailed Analysis*

• **0:00 - Introduction to Tokenization**
    1. *"tokenization is my least favorite part... but unfortunately it is necessary to understand"*
    2. *"a lot of oddness with large language models typically traces back to tokenization"*
    *Technical analysis*: Establishes tokenization as a critical, albeit complex, foundational component of LLMs. It translates raw text into integer sequences the model can process, and many model quirks originate here.

• **0:26 - Naive Character-Level Tokenization**
    1. *"we created a vocabulary of 65 possible characters"*
    2. *"token 18 47 etc... later we saw that the way we plug these tokens... is by using an embedding table"*
    *Technical analysis*: Demonstrates a simple tokenizer mapping characters to integers. This approach results in long sequences and a small vocabulary, which is inefficient for modern LLMs due to limited context windows.

• **2:21 - Introduction to Byte Pair Encoding (BPE)**
    1. *"we're dealing on chunk level... using algorithms such as... the byte pair encoding algorithm"*
    2. *"tokens are this like fundamental unit... the atom of large language models"*
    *Technical analysis*: Introduces BPE as the standard subword tokenization method. It starts with a base vocabulary (e.g., 256 bytes) and iteratively merges the most frequent adjacent pairs, creating a compressed representation and a larger, optimized vocabulary.

• **4:20 - Motivational Examples of Tokenization Issues**
    1. *"llms can't... do spelling tasks very easily... usually due to tokenization"*
    2. *"non-english languages can work much worse... this is due to tokenization"*
    *Technical analysis*: Highlights practical problems caused by tokenization: spelling difficulties (long tokens obscure character-level details), language bias (non-English text often uses more tokens), and arithmetic challenges (numbers tokenize arbitrarily).

• **5:59 - Interactive Tokenizer Demonstration**
    1. *"the token space is is token 318... the token space at is token 379"*
    2. *"egg by itself... is two tokens but here as a space egg is suddenly a single token"*
    *Technical analysis*: Shows the non-intuitive nature of token chunks. The same string can tokenize differently based on context (leading space), and token boundaries often include spaces or punctuation, which the model must learn to interpret.

• **11:25 - Tokenization Efficiency Across Models**
    1. *"gpt2 tokenizer creates a token count of 300... GPT-4 tokenizer... drops to 185"*
    2. *"the handling of the whitespace for python has improved a lot"*
    *Technical analysis*: Compares tokenizer efficiency. GPT-4's larger vocabulary (~100k vs. ~50k) and improved merging rules (e.g., grouping Python indentation spaces) lead to denser, more context-efficient representations.

• **15:01 - Unicode and UTF-8 Encoding Foundation**
    1. *"strings are immutable sequences of Unicode code points"*
    2. *"utf8 takes every single code point and translates it to a byte stream... between one to four bytes"*
    *Technical analysis*: Establishes the text preprocessing foundation. LLMs don't use Unicode code points directly due to a large, unstable vocabulary. UTF-8 encoding provides a consistent byte-level representation, which BPE then compresses.

• **22:35 - BPE Algorithm Walkthrough**
    1. *"iteratively find the pair of tokens that occur the most frequently"*
    2. *"replace that pair with just a single new token that we append to our vocabulary"*
    *Technical analysis*: Explains the core BPE algorithm. It's a compression technique that reduces sequence length by replacing common byte pairs with new single tokens, iteratively building a merge table.

• **27:03 - Implementing BPE Training**
    1. *"we'd like to iterate over here and find the pair of bytes that occur most frequently"*
    2. *"we're going to iterate over this entire list and every time we see 101 comma 32 we're going to swap that out for 256"*
    *Technical analysis*: Provides a step-by-step code implementation for finding stats and performing merges. The process is separate from LLM training and uses its own dataset to build the vocabulary (merge rules).

• **35:55 - Encoding and Decoding Implementation**
    1. *"vocab is a mapping... from the token ID to the bytes object for that token"*
    2. *"we are going to be given a string and we want to encode it into tokens"*
    *Technical analysis*: Details the two-way translation. Decoding maps token IDs back to bytes (with UTF-8 error handling), while encoding applies the learned merge rules in order to compress a string into a token ID sequence.

• **57:39 - GPT-2 Tokenizer: Regex Pre-tokenization**
    1. *"they want to enforce that some types of characters should never be merged together"*
    2. *"using this regex pattern to chunk up the text is just one way of enforcing that some merges are not to happen"*
    *Technical analysis*: Reveals that production tokenizers like GPT-2's use regex rules to pre-split text (e.g., separating letters, numbers, punctuation) before applying BPE. This prevents undesirable merges across semantic boundaries but adds complexity.

• **71:40 - Special Tokens**
    1. *"special tokens are used to delimit different parts of the data"*
    2. *"the end of text token is used to delimit documents in the training set"*
    *Technical analysis*: Special tokens (e.g., `<|endoftext|>`, chat message delimiters) are added to the vocabulary outside the BPE process. They require model surgery (extending embedding and output layers) and are handled separately during encoding/decoding.

• **88:43 - SentencePiece Library**
    1. *"sentence piece... works directly on the level of the code points themselves"*
    2. *"it has a ton of options and configurations... quite a bit of accumulated historical baggage"*
    *Technical analysis*: Contrasts SentencePiece (used by LLaMA) with OpenAI's approach. It operates on Unicode code points first, falling back to bytes for rare characters. It includes many pre-processing/normalization options, which can be a source of complexity and potential misconfiguration.

• **103:29 - Vocabulary Size Considerations**
    1. *"as vocab size increases this embedding table... is going to also grow"*
    2. *"if you have a very large vocabulary size... every one of these tokens is going to come up more and more rarely"*
    *Technical analysis*: Discusses the trade-off in choosing vocabulary size. Larger vocabs yield shorter sequences (better context usage) but increase embedding/head parameters and risk undertraining rare tokens. Current models use ~50k-100k tokens.

• **109:54 - Revisiting Tokenization Issues**
    1. *"default style turns out to be a single individual token... the model should not be very good at tasks related to spelling"*
    2. *"solid gold Magikarp... is a Reddit user... this token never gets activated"*
    *Technical analysis*: Deepens the analysis of earlier issues. Long tokens hinder character-level tasks. The "solid gold Magikarp" phenomenon occurs when a token frequent in the tokenizer's training data is absent from the LLM's training data, leaving its embedding untrained and causing erratic model behavior.

*Conclusion*
• *Summary of key technical takeaways*: Tokenization via BPE is a separate, critical preprocessing stage that converts text into a compressed integer sequence. It involves trade-offs in vocabulary size and merging rules, which directly impact model efficiency, capabilities, and observed quirks. Real-world implementations add layers like regex pre-tokenization and special token handling.
• *Practical applications*: Use existing efficient tokenizers (like `tiktoken`) when possible. When designing prompts or structured data, consider token efficiency (e.g., YAML may be denser than JSON). Be aware of edge cases like trailing spaces or partial tokens that can lead to degraded performance.
• *Long-term recommendations*: The field would benefit from "tokenization-free" models that process raw bytes efficiently. Until then, for custom tokenizer training, carefully configure tools like SentencePiece or aim for libraries that combine BPE training efficiency with the clarity of byte-level encoding. Always ensure alignment between the tokenizer's training data and the LLM's training data to avoid "unallocated" token embeddings.