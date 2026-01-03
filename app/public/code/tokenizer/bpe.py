"""
BPE Tokenizer: Byte Pair Encoding from scratch
Based on Lecture 8 of Andrej Karpathy's Neural Networks: Zero to Hero

Video: https://youtu.be/zduSFxRajkE

This module implements:
- Byte Pair Encoding (BPE) tokenizer
- encode() and decode() functions
- Unicode and UTF-8 handling
- Special tokens
- Understanding tokenization artifacts
"""


def get_stats(ids):
    """
    Count frequency of consecutive pairs in the list of ids.

    Args:
        ids: list of integers

    Returns:
        dict mapping pairs (tuple) to their counts
    """
    # TODO: Count all adjacent pairs
    pass


def merge(ids, pair, idx):
    """
    Replace all occurrences of pair with idx in ids.

    Args:
        ids: list of integers
        pair: tuple of two integers to replace
        idx: integer to replace pair with

    Returns:
        new list with pair replaced by idx
    """
    # TODO: Merge consecutive pairs into single token
    pass


class BasicTokenizer:
    """Simple BPE tokenizer."""

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {}   # int -> bytes

    def train(self, text, vocab_size):
        """
        Train the tokenizer on text.

        Args:
            text: training text (string)
            vocab_size: desired vocabulary size (>= 256)
        """
        # TODO:
        # 1. Start with bytes (0-255 are base vocabulary)
        # 2. Find most common pair
        # 3. Merge that pair, add to vocabulary
        # 4. Repeat until vocab_size reached
        pass

    def encode(self, text):
        """
        Encode text to list of token ids.

        Args:
            text: string to encode

        Returns:
            list of integers (token ids)
        """
        # TODO:
        # 1. Convert text to bytes
        # 2. Apply merges in order of training
        pass

    def decode(self, ids):
        """
        Decode list of token ids back to text.

        Args:
            ids: list of integers (token ids)

        Returns:
            decoded string
        """
        # TODO:
        # 1. Look up each id in vocabulary
        # 2. Concatenate bytes
        # 3. Decode to string
        pass


class RegexTokenizer:
    """
    BPE tokenizer with regex-based pre-tokenization.
    Similar to GPT-2/GPT-4 tokenizers.
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.pattern = None  # Regex pattern for pre-tokenization

    def train(self, text, vocab_size):
        """Train with regex-based splitting."""
        # TODO:
        # 1. Split text using regex pattern (like GPT-4)
        # 2. Train BPE on each piece separately
        pass

    def encode(self, text):
        """Encode with pre-tokenization."""
        # TODO: Split first, then encode pieces
        pass

    def decode(self, ids):
        """Decode tokens to string."""
        # TODO: Look up and concatenate
        pass


def visualize_tokenization(tokenizer, text):
    """Helper to visualize how text gets tokenized."""
    ids = tokenizer.encode(text)
    print(f"Text: {text!r}")
    print(f"Tokens: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)!r}")
    print(f"Token count: {len(ids)}")


# Tokenization artifacts to explore:
# 1. Numbers: "123" vs "1 2 3" vs " 123"
# 2. Whitespace sensitivity: "hello" vs " hello"
# 3. Case: "Hello" vs "hello" vs "HELLO"
# 4. Special characters: punctuation, emojis
# 5. Code: indentation, common patterns
# 6. Non-English text: various Unicode


if __name__ == "__main__":
    print("BPE Tokenizer Implementation")
    print("=" * 40)

    # Simple test
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump!
    """

    # TODO: Train a tokenizer
    # tokenizer = BasicTokenizer()
    # tokenizer.train(sample_text * 100, vocab_size=300)

    # TODO: Test encoding/decoding
    # test = "Hello, world!"
    # visualize_tokenization(tokenizer, test)

    # TODO: Explore tokenization artifacts

    print("\nRun the notebook for interactive exploration!")
