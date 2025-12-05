"""
Simple tokenizer for the LLM.
Converts text to token IDs and vice versa.
"""

class Tokenizer:
    def __init__(self, corpus):
        """Build vocabulary from corpus."""
        # Get all unique words
        words = set()
        for sentence in corpus:
            words.update(sentence.lower().split())
        
        # Create vocabulary with special tokens
        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + sorted(list(words))
        self.vocab_size = len(self.vocab)
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def encode(self, text):
        """Convert text to list of token IDs."""
        words = text.lower().split()
        return [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
    
    def decode(self, token_ids):
        """Convert list of token IDs to text."""
        words = [self.idx_to_word[idx] for idx in token_ids if idx != self.word_to_idx['<PAD>']]
        return ' '.join(words)
    
    def get_vocab_size(self):
        """Return vocabulary size."""
        return self.vocab_size

