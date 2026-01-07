---
title: Let's build the GPT Tokenizer
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

### 💡 Intuitive Understanding

**Analogy:** Tokenization is like cutting a sentence into LEGO blocks. The model never sees the actual text—it only sees numbered blocks. How you cut determines what the model can "see" and how efficiently it uses its context window.

**Mental Model:** The tokenization pipeline:
```
"Hello world!" (raw text)
       ↓
[15496, 995, 0] (token IDs)
       ↓
Model processes these integers

The model has NO concept of letters.
It only knows ~50,000 token IDs.
```

**Why It Matters:** Many LLM "bugs" are actually tokenization artifacts. Why can't GPT count letters in a word? Because it doesn't see letters—it sees tokens. Understanding tokenization explains these limitations.

---

• **0:26 - Naive Character-Level Tokenization**
    1. *"we created a vocabulary of 65 possible characters"*
    2. *"token 18 47 etc... later we saw that the way we plug these tokens... is by using an embedding table"*
    *Technical analysis*: Demonstrates a simple tokenizer mapping characters to integers. This approach results in long sequences and a small vocabulary, which is inefficient for modern LLMs due to limited context windows.

### 💡 Intuitive Understanding

**Analogy:** Character-level tokenization is like spelling out every word letter by letter when talking. It works, but it's slow and uses up your "speaking time" (context window) quickly.

**Mental Model:** Character-level trade-offs:
```
Character-level:
  "Hello" → [H, e, l, l, o] → 5 tokens
  Vocab size: ~65 (26 letters + symbols)
  Simple, but very long sequences

Subword (BPE):
  "Hello" → [Hello] → 1 token (maybe)
  Vocab size: ~50,000
  Complex, but short sequences

Context window = 4096 tokens
Character: ~4096 characters of text
Subword: ~20,000+ characters of text
```

**Why It Matters:** Context window is precious. Character-level wastes it. BPE compresses text efficiently, letting the model "see" more content at once.

---

• **2:21 - Introduction to Byte Pair Encoding (BPE)**
    1. *"we're dealing on chunk level... using algorithms such as... the byte pair encoding algorithm"*
    2. *"tokens are this like fundamental unit... the atom of large language models"*
    *Technical analysis*: Introduces BPE as the standard subword tokenization method. It starts with a base vocabulary (e.g., 256 bytes) and iteratively merges the most frequent adjacent pairs, creating a compressed representation and a larger, optimized vocabulary.

### 💡 Intuitive Understanding

**Analogy:** BPE is like creating shorthand. If you frequently write "the", you might create a symbol for it. If "the " (with space) appears even more, make a symbol for that too. BPE does this automatically for the most common patterns.

**Mental Model:** BPE algorithm:
```
Start: vocabulary = 256 bytes (0-255)
       text = [104, 101, 108, 108, 111] (h-e-l-l-o)

Step 1: Find most frequent pair → (108, 108) = "ll"
Step 2: Assign new token → 256 = "ll"
Step 3: Replace in text → [104, 101, 256, 111]

Repeat until desired vocab size:
Step 4: Find next frequent pair → (104, 101) = "he"
Step 5: Assign new token → 257 = "he"
Step 6: Replace → [257, 256, 111]

Final: text compressed from 5 to 3 tokens
```

**Why It Matters:** BPE is the compression algorithm that makes LLMs efficient. It's a simple, elegant solution to the vocabulary problem: let the data tell you what chunks are important.

---

• **4:20 - Motivational Examples of Tokenization Issues**
    1. *"llms can't... do spelling tasks very easily... usually due to tokenization"*
    2. *"non-english languages can work much worse... this is due to tokenization"*
    *Technical analysis*: Highlights practical problems caused by tokenization: spelling difficulties (long tokens obscure character-level details), language bias (non-English text often uses more tokens), and arithmetic challenges (numbers tokenize arbitrarily).

### 💡 Intuitive Understanding

**Analogy:** Imagine reading a book where some words are glued together into single blocks. "Thequickbrownfox" as one block. You can understand it, but if asked "How many letters are in 'quick'?", you'd have to mentally unpack the block first.

**Mental Model:** Tokenization-induced limitations:
```
Spelling:
  "strawberry" → likely 1 token
  Model can't easily see: s-t-r-a-w-b-e-r-r-y
  Result: Struggles to count 'r's or reverse the word

Non-English:
  English: "Hello" → 1 token
  Japanese: "こんにちは" → 3+ tokens
  Same meaning, 3x the token cost → less context for non-English

Arithmetic:
  "127" might be 1 token
  "1,270" might be [1, ,, 2, 7, 0] = 5 tokens
  Model can't easily see the relationship
```

**Why It Matters:** These aren't model intelligence limitations—they're tokenization artifacts. Understanding this helps you design prompts that work around these issues (e.g., space out letters for spelling tasks).

---

• **5:59 - Interactive Tokenizer Demonstration**
    1. *"the token space is is token 318... the token space at is token 379"*
    2. *"egg by itself... is two tokens but here as a space egg is suddenly a single token"*
    *Technical analysis*: Shows the non-intuitive nature of token chunks. The same string can tokenize differently based on context (leading space), and token boundaries often include spaces or punctuation, which the model must learn to interpret.

### 💡 Intuitive Understanding

**Analogy:** Spaces are "sticky" in tokenization. " cat" (space-cat) and "cat" are different tokens. It's like the difference between a word at the start of a sentence vs. in the middle.

**Mental Model:** Context-dependent tokenization:
```
"egg"    → [68, 7149]  (2 tokens: "e" + "gg")
" egg"   → [19463]     (1 token: " egg")
"eggs"   → [68, 7149, 82]  (3 tokens)
" eggs"  → [19463, 82]     (2 tokens)

The leading space changes everything!

This is why prompts can be sensitive to formatting.
"Tell me about eggs" vs "Tell me about  eggs" (double space)
→ Different tokens → potentially different behavior
```

**Why It Matters:** This explains strange prompt engineering tricks. Adding or removing a space can change tokenization and thus model behavior. It's not magic—it's tokenization.

---

• **11:25 - Tokenization Efficiency Across Models**
    1. *"gpt2 tokenizer creates a token count of 300... GPT-4 tokenizer... drops to 185"*
    2. *"the handling of the whitespace for python has improved a lot"*
    *Technical analysis*: Compares tokenizer efficiency. GPT-4's larger vocabulary (~100k vs. ~50k) and improved merging rules (e.g., grouping Python indentation spaces) lead to denser, more context-efficient representations.

### 💡 Intuitive Understanding

**Analogy:** GPT-4's tokenizer is like learning common phrases in a language. GPT-2 might know "thank" and "you" separately. GPT-4 learned "thank you" as a unit. More vocabulary = fewer tokens for the same text.

**Mental Model:** Tokenizer efficiency comparison:
```
Same Python code:
  GPT-2: 300 tokens (50k vocab)
  GPT-4: 185 tokens (100k vocab)

Why?
  - Larger vocabulary = more common patterns
  - "    " (4 spaces) is 1 token in GPT-4, 4 in GPT-2
  - Common function names might be single tokens

Efficiency gain: 38% fewer tokens
  → 38% more code fits in context
  → Faster, cheaper inference
```

**Why It Matters:** Tokenizer efficiency directly impacts cost and capability. GPT-4 can "see" almost twice as much code in the same context window as GPT-2. This is a real engineering advantage.

---

• **15:01 - Unicode and UTF-8 Encoding Foundation**
    1. *"strings are immutable sequences of Unicode code points"*
    2. *"utf8 takes every single code point and translates it to a byte stream... between one to four bytes"*
    *Technical analysis*: Establishes the text preprocessing foundation. LLMs don't use Unicode code points directly due to a large, unstable vocabulary. UTF-8 encoding provides a consistent byte-level representation, which BPE then compresses.

### 💡 Intuitive Understanding

**Analogy:** Unicode is like a giant phonebook listing every character ever invented. UTF-8 is a clever way to write down any phonebook entry using just bytes (0-255). BPE then compresses these bytes into tokens.

**Mental Model:** The encoding pipeline:
```
Text: "Hello 世界"

Step 1: Unicode code points
  H = U+0048
  世 = U+4E16
  界 = U+754C

Step 2: UTF-8 bytes
  H = [72]           (1 byte)
  世 = [228, 184, 150]  (3 bytes)
  界 = [231, 149, 140]  (3 bytes)

Step 3: BPE on bytes
  [72, ...] → token IDs

Why UTF-8?
  - Base vocabulary is exactly 256 (one byte each)
  - Any text can be represented
  - Handles all languages
```

**Why It Matters:** This is why BPE works on "bytes" not characters. UTF-8 gives a stable 256-element base vocabulary. Unicode alone would have 150,000+ code points—unmanageable as a base vocabulary.

---

• **22:35 - BPE Algorithm Walkthrough**
    1. *"iteratively find the pair of tokens that occur the most frequently"*
    2. *"replace that pair with just a single new token that we append to our vocabulary"*
    *Technical analysis*: Explains the core BPE algorithm. It's a compression technique that reduces sequence length by replacing common byte pairs with new single tokens, iteratively building a merge table.

### 💡 Intuitive Understanding

**Analogy:** BPE is like repeatedly finding the most popular two-person team and giving them a team name. After enough rounds, you have a directory of teams (tokens) that represent common combinations.

**Mental Model:** BPE training step-by-step:
```
Training corpus: "aaabdaaabac"
Initial tokens: [a, a, a, b, d, a, a, a, b, a, c]

Round 1:
  Count pairs: (a,a)=4, (a,b)=2, (b,d)=1, (d,a)=1, (b,a)=1, (a,c)=1
  Most frequent: (a,a)
  Create token: Z = "aa"
  Result: [Z, a, b, d, Z, a, b, a, c]

Round 2:
  Count pairs: (Z,a)=2, (a,b)=2, (b,d)=1, (d,Z)=1, (b,a)=1, (a,c)=1
  Tie: pick (Z,a) → Y = "aaa"
  Result: [Y, b, d, Y, b, a, c]

Continue until vocab_size reached...
```

**Why It Matters:** The merge order is crucial and saved as a "merge table." This table is what defines the tokenizer. During encoding, you apply merges in the same order they were learned.

---

• **27:03 - Implementing BPE Training**
    1. *"we'd like to iterate over here and find the pair of bytes that occur most frequently"*
    2. *"we're going to iterate over this entire list and every time we see 101 comma 32 we're going to swap that out for 256"*
    *Technical analysis*: Provides a step-by-step code implementation for finding stats and performing merges. The process is separate from LLM training and uses its own dataset to build the vocabulary (merge rules).

### 💡 Intuitive Understanding

**Analogy:** Training a tokenizer is like creating a dictionary for a language. You analyze a corpus, find common patterns, and codify them. This dictionary (merge table) is then used forever—it doesn't change during LLM training.

**Mental Model:** BPE training code structure:
```python
def train_bpe(text, vocab_size):
    tokens = list(text.encode('utf-8'))  # Start with bytes
    merges = {}  # Will hold our learned merges

    for i in range(vocab_size - 256):  # 256 base bytes
        # Count all adjacent pairs
        pairs = get_pair_counts(tokens)

        # Find most frequent
        best_pair = max(pairs, key=pairs.get)

        # Create new token
        new_token = 256 + i
        merges[best_pair] = new_token

        # Replace all occurrences
        tokens = merge(tokens, best_pair, new_token)

    return merges  # This IS the tokenizer
```

**Why It Matters:** The tokenizer is trained separately from the LLM, often on different data. This separation is important—you train the tokenizer once, then use it to process all LLM training data.

---

• **35:55 - Encoding and Decoding Implementation**
    1. *"vocab is a mapping... from the token ID to the bytes object for that token"*
    2. *"we are going to be given a string and we want to encode it into tokens"*
    *Technical analysis*: Details the two-way translation. Decoding maps token IDs back to bytes (with UTF-8 error handling), while encoding applies the learned merge rules in order to compress a string into a token ID sequence.

### 💡 Intuitive Understanding

**Analogy:** Encoding is like zipping a file (compress text to tokens). Decoding is like unzipping (tokens back to text). The merge table is the compression dictionary.

**Mental Model:** Encode and decode:
```python
# DECODE: Token IDs → Text
def decode(token_ids):
    bytes_list = [vocab[id] for id in token_ids]
    return b''.join(bytes_list).decode('utf-8', errors='replace')

# Example:
decode([15496, 995]) → "Hello world"


# ENCODE: Text → Token IDs
def encode(text):
    tokens = list(text.encode('utf-8'))

    # Apply merges in training order
    for (pair, new_token) in merges.items():
        tokens = merge(tokens, pair, new_token)

    return tokens

# Example:
encode("Hello world") → [15496, 995]
```

**Why It Matters:** The order of merge application matters! Encoding must apply merges in the same order they were learned during training. This is why tokenizers save ordered merge lists, not just merge sets.

---

• **57:39 - GPT-2 Tokenizer: Regex Pre-tokenization**
    1. *"they want to enforce that some types of characters should never be merged together"*
    2. *"using this regex pattern to chunk up the text is just one way of enforcing that some merges are not to happen"*
    *Technical analysis*: Reveals that production tokenizers like GPT-2's use regex rules to pre-split text (e.g., separating letters, numbers, punctuation) before applying BPE. This prevents undesirable merges across semantic boundaries but adds complexity.

### 💡 Intuitive Understanding

**Analogy:** Pre-tokenization is like enforcing word boundaries before compression. You don't want "cat." to merge with "Dog" across the sentence boundary. Regex rules prevent "nonsense" merges.

**Mental Model:** GPT-2's regex pre-tokenization:
```python
# GPT-2's pattern (simplified)
pattern = r"""
    'contractions|  # 'll, 're, 'd, etc.
    \s+[a-zA-Z]+|   # space + letters (word)
    \s+\d+|         # space + numbers
    \s+|            # whitespace only
    [^\s]+          # anything else
"""

"Hello, world!" → ["Hello", ",", " world", "!"]

Then BPE runs on each chunk separately.
This prevents:
  - "." from merging with next word
  - Numbers from merging with letters
  - Cross-word merges
```

**Why It Matters:** Pre-tokenization explains many tokenization quirks. It's why "," is its own token, and why " cat" and "cat" behave differently. The regex defines the "walls" that BPE can't merge across.

---

• **71:40 - Special Tokens**
    1. *"special tokens are used to delimit different parts of the data"*
    2. *"the end of text token is used to delimit documents in the training set"*
    *Technical analysis*: Special tokens (e.g., `<|endoftext|>`, chat message delimiters) are added to the vocabulary outside the BPE process. They require model surgery (extending embedding and output layers) and are handled separately during encoding/decoding.

### 💡 Intuitive Understanding

**Analogy:** Special tokens are like chapter markers in a book. They're not part of the text content—they're metadata signals to the model. "End of chapter," "New speaker," "System instruction begins."

**Mental Model:** Special tokens:
```
Common special tokens:
  <|endoftext|>   - Document boundary
  <|im_start|>    - Chat message start
  <|im_end|>      - Chat message end
  <|user|>        - User message marker
  <|assistant|>   - Assistant message marker

How they're added:
  1. Reserve token IDs (e.g., 50256+)
  2. Extend embedding table with new rows
  3. Extend LM head with new output columns
  4. Train on data with these tokens

They're NOT learned from BPE—they're manually defined.
```

**Why It Matters:** Special tokens are how you structure prompts for chat models. They tell the model "this is a system message" vs "this is user input." Misusing them breaks the model's expectations.

---

• **88:43 - SentencePiece Library**
    1. *"sentence piece... works directly on the level of the code points themselves"*
    2. *"it has a ton of options and configurations... quite a bit of accumulated historical baggage"*
    *Technical analysis*: Contrasts SentencePiece (used by LLaMA) with OpenAI's approach. It operates on Unicode code points first, falling back to bytes for rare characters. It includes many pre-processing/normalization options, which can be a source of complexity and potential misconfiguration.

### 💡 Intuitive Understanding

**Analogy:** SentencePiece is like an older, Swiss-army-knife tool. It can do everything (BPE, unigram, normalization) but has many switches that can be set wrong. Tiktoken (OpenAI) is like a newer, purpose-built tool—fewer options, but cleaner.

**Mental Model:** SentencePiece vs tiktoken:
```
SentencePiece (LLaMA, etc.):
  - Operates on Unicode code points first
  - Falls back to bytes for rare chars
  - Many options: normalization, casing, etc.
  - Can accidentally change your text

tiktoken (OpenAI GPT):
  - Operates on UTF-8 bytes directly
  - No text normalization
  - Fewer footguns
  - What you encode is what you get
```

**Why It Matters:** If you're training your own models, choose your tokenizer carefully. SentencePiece is powerful but complex. For most use cases, a simpler byte-level BPE (like tiktoken's approach) is safer.

---

• **103:29 - Vocabulary Size Considerations**
    1. *"as vocab size increases this embedding table... is going to also grow"*
    2. *"if you have a very large vocabulary size... every one of these tokens is going to come up more and more rarely"*
    *Technical analysis*: Discusses the trade-off in choosing vocabulary size. Larger vocabs yield shorter sequences (better context usage) but increase embedding/head parameters and risk undertraining rare tokens. Current models use ~50k-100k tokens.

### 💡 Intuitive Understanding

**Analogy:** Vocabulary size is like choosing how many words to learn. Too few: you spell everything out. Too many: you have words you never use (and never really learned).

**Mental Model:** Vocabulary size trade-offs:
```
Small vocab (e.g., 8k):
  + Each token seen many times (well-trained)
  + Smaller embedding table
  - Long sequences
  - Context window wasted

Large vocab (e.g., 200k):
  + Short sequences
  + More context fits
  - Huge embedding table
  - Rare tokens poorly trained

Sweet spot: ~50k-100k
  GPT-2: 50,257 tokens
  GPT-4: ~100,000 tokens
  LLaMA: 32,000 tokens
```

**Why It Matters:** Vocabulary size is a key design decision. It affects model size, efficiency, and quality. Larger isn't always better—undertrained tokens cause problems.

---

• **109:54 - Revisiting Tokenization Issues**
    1. *"default style turns out to be a single individual token... the model should not be very good at tasks related to spelling"*
    2. *"solid gold Magikarp... is a Reddit user... this token never gets activated"*
    *Technical analysis*: Deepens the analysis of earlier issues. Long tokens hinder character-level tasks. The "solid gold Magikarp" phenomenon occurs when a token frequent in the tokenizer's training data is absent from the LLM's training data, leaving its embedding untrained and causing erratic model behavior.

### 💡 Intuitive Understanding

**Analogy:** "SolidGoldMagikarp" is like a word in your vocabulary that you've never actually read in context. You know it exists (it's in the tokenizer), but you've never seen it used (not in LLM training data). When someone uses it, you have no idea what to do.

**Mental Model:** The SolidGoldMagikarp bug:
```
Tokenizer training data: Reddit (includes username "SolidGoldMagikarp")
LLM training data: Filtered web (username removed)

Result:
  - Token 120412 = "SolidGoldMagikarp"
  - This token was never seen during LLM training
  - Embedding for 120412 is random/untrained
  - When prompted with this token → erratic behavior

Lesson: Tokenizer and LLM training data must align!
```

**Why It Matters:** This is a real bug that caused real model issues. It demonstrates why tokenization isn't just a preprocessing step—it's deeply intertwined with model behavior. Misaligned tokenizer/model training can cause subtle, bizarre failures.

---

*Conclusion*
• *Summary of key technical takeaways*: Tokenization via BPE is a separate, critical preprocessing stage that converts text into a compressed integer sequence. It involves trade-offs in vocabulary size and merging rules, which directly impact model efficiency, capabilities, and observed quirks. Real-world implementations add layers like regex pre-tokenization and special token handling.
• *Practical applications*: Use existing efficient tokenizers (like `tiktoken`) when possible. When designing prompts or structured data, consider token efficiency (e.g., YAML may be denser than JSON). Be aware of edge cases like trailing spaces or partial tokens that can lead to degraded performance.
• *Long-term recommendations*: The field would benefit from "tokenization-free" models that process raw bytes efficiently. Until then, for custom tokenizer training, carefully configure tools like SentencePiece or aim for libraries that combine BPE training efficiency with the clarity of byte-level encoding. Always ensure alignment between the tokenizer's training data and the LLM's training data to avoid "unallocated" token embeddings.

---

## 📝 Exercises & Practice

### Conceptual Questions

1. **Why BPE?** Explain why BPE is preferred over character-level tokenization for LLMs. What are the trade-offs?

2. **UTF-8 Foundation:** Why do modern tokenizers operate on UTF-8 bytes rather than Unicode code points?

3. **Space Sensitivity:** Why does " cat" (with space) tokenize differently from "cat"? What implications does this have for prompt engineering?

4. **Vocabulary Size:** What happens if vocabulary size is too small? Too large? What's the current sweet spot?

5. **Special Tokens:** Why are special tokens handled separately from BPE? Give examples of how they're used.

6. **SolidGoldMagikarp:** Explain the "SolidGoldMagikarp" bug. What causes it and how can it be prevented?

### Coding Challenges

1. **Implement Basic BPE Training:**
   ```python
   def train_bpe(text: str, num_merges: int) -> dict:
       """
       Returns a dictionary of merges: {(byte1, byte2): new_token}
       """
       pass
   ```

2. **Implement BPE Encoding:**
   ```python
   def encode(text: str, merges: dict) -> list[int]:
       """
       Given text and learned merges, return token IDs.
       """
       pass
   ```

3. **Implement BPE Decoding:**
   ```python
   def decode(token_ids: list[int], vocab: dict) -> str:
       """
       Given token IDs and vocab mapping, return text.
       Handle UTF-8 decoding errors gracefully.
       """
       pass
   ```

4. **Tokenizer Comparison:**
   - Use tiktoken to tokenize the same text with GPT-2 and GPT-4 tokenizers
   - Compare token counts
   - Visualize which parts of the text are tokenized differently

5. **Edge Case Analysis:**
   - Find text where the same characters produce different token counts due to spacing
   - Find examples where non-English text is inefficiently tokenized
   - Test how numbers are tokenized (1, 10, 100, 1000)

### Reflection

- The lecture suggests "tokenization-free" models (processing raw bytes) might be the future. Research current efforts in this direction (e.g., MegaByte, byte-level transformers). What are the challenges?

- Tokenization was designed for efficiency, but it creates artifacts that affect model behavior. Are there tasks where these artifacts are particularly harmful? Beneficial?

- The tiktoken library is open-source. Read its implementation. How does it handle edge cases? What can you learn from production-quality tokenizer code?

- If you were designing a tokenizer for a specific domain (medical, legal, code), how would you approach it? Would you train from scratch or adapt an existing tokenizer?
