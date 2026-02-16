"""
Pure Python implementation of Byte Pair Encoding (BPE) tokenizer.
Zero dependencies, educational focus.

BPE is the tokenization method used in GPT-2/3/4, making this a valuable
educational addition to understand modern language model tokenization.
"""

import json


class BPETokenizer:
    """Byte Pair Encoding tokenizer.
    
    BPE works by iteratively merging the most frequent pair of tokens in the corpus.
    Starting with a character-level vocabulary, it builds up common subwords and words
    through greedy merging based on frequency.
    
    Attributes:
        vocab: Dictionary mapping token strings to their IDs
        merges: List of (token1, token2) pairs representing the merge order
        bos_token: Beginning of sequence token string
        bos_id: ID of the BOS token
    """
    
    def __init__(self):
        self.vocab = {}  # token string -> id
        self.merges = []  # list of (str, str) merge pairs in order
        self.bos_token = '<BOS>'
        self.bos_id = 0
        
    def train(self, docs, vocab_size=256, verbose=False):
        """Train BPE tokenizer on a corpus.
        
        Args:
            docs: List of document strings to train on
            vocab_size: Target vocabulary size (must be >= unique chars + 1 for BOS)
            verbose: Print training progress
            
        The algorithm:
        1. Start with character-level vocabulary + BOS token
        2. Tokenize all documents as character sequences
        3. Iteratively merge the most frequent adjacent pair until vocab_size reached
        4. BOS tokens are never merged with other tokens (they're special delimiters)
        """
        # Build initial character vocabulary with BOS token
        chars = sorted(set(''.join(docs)))
        base_vocab = [self.bos_token] + chars
        
        if vocab_size < len(base_vocab):
            raise ValueError(f"vocab_size ({vocab_size}) must be >= {len(base_vocab)} "
                           f"(unique chars {len(chars)} + BOS token)")
        
        # Initialize vocab with base characters
        self.vocab = {token: idx for idx, token in enumerate(base_vocab)}
        self.bos_id = self.vocab[self.bos_token]
        self.merges = []
        
        # Tokenize all documents as lists of tokens (initially characters)
        tokenized_docs = []
        for doc in docs:
            # Each doc is [BOS, char1, char2, ..., charN, BOS]
            tokens = [self.bos_token] + list(doc) + [self.bos_token]
            tokenized_docs.append(tokens)
        
        # Iteratively merge most frequent pairs
        num_merges = vocab_size - len(base_vocab)
        for merge_idx in range(num_merges):
            # Count all adjacent pairs across all documents
            # Skip pairs involving BOS token (BOS should not be merged)
            pair_counts = {}
            for tokens in tokenized_docs:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    # Skip if either token in the pair is BOS
                    if pair[0] == self.bos_token or pair[1] == self.bos_token:
                        continue
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            if not pair_counts:
                if verbose:
                    print(f"No more pairs to merge at iteration {merge_idx}")
                break
                
            # Find most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])
            pair, count = best_pair
            
            if count < 2:  # Don't merge pairs that occur only once
                if verbose:
                    print(f"Stopping merge: most frequent pair occurs only {count} time(s)")
                break
            
            # Create new token by concatenating the pair
            new_token = pair[0] + pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.merges.append(pair)
            
            if verbose and merge_idx % 10 == 0:
                print(f"Merge {merge_idx}/{num_merges}: {pair} -> '{new_token}' "
                      f"(count={count}, vocab_size={len(self.vocab)})")
            
            # Apply merge to all documents
            for doc_idx in range(len(tokenized_docs)):
                tokens = tokenized_docs[doc_idx]
                i = 0
                new_tokens = []
                while i < len(tokens):
                    # Check if current and next token match the pair to merge
                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        new_tokens.append(new_token)
                        i += 2  # Skip both tokens in the pair
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokenized_docs[doc_idx] = new_tokens
        
        if verbose:
            print(f"Training complete: vocab_size={len(self.vocab)}, "
                  f"num_merges={len(self.merges)}")
    
    def encode(self, text):
        """Encode text to token IDs using trained BPE.
        
        Args:
            text: String to encode
            
        Returns:
            List of token IDs (integers)
            
        The algorithm:
        1. Start with character-level tokenization
        2. Apply all learned merges in order
        3. Convert tokens to IDs
        """
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Start with character-level tokens
        tokens = list(text)
        
        # Apply merges in the order they were learned
        for pair in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                # Check if current and next token match the merge pair
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    # Merge the pair
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert tokens to IDs (handle unknown tokens by using character fallback)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Fallback: encode unknown token character by character
                for char in token:
                    if char in self.vocab:
                        ids.append(self.vocab[char])
                    # If character not in vocab, skip it (shouldn't happen if trained properly)
        
        return ids
    
    def decode(self, ids):
        """Decode token IDs back to text.
        
        Args:
            ids: List of token IDs (integers)
            
        Returns:
            Decoded text string
        """
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Create reverse mapping
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Convert IDs to tokens
        tokens = []
        for idx in ids:
            if idx in id_to_token:
                token = id_to_token[idx]
                # Skip BOS token in output
                if token != self.bos_token:
                    tokens.append(token)
        
        return ''.join(tokens)
    
    def save(self, path):
        """Save trained tokenizer to JSON file.
        
        Args:
            path: File path to save to
        """
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'bos_token': self.bos_token,
            'bos_id': self.bos_id,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path):
        """Load trained tokenizer from JSON file.
        
        Args:
            path: File path to load from
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = [tuple(pair) for pair in data['merges']]  # Convert lists back to tuples
        self.bos_token = data['bos_token']
        self.bos_id = data['bos_id']
    
    @property
    def vocab_size(self):
        """Return the size of the vocabulary."""
        return len(self.vocab)
    
    def get_vocab(self):
        """Return the vocabulary dictionary."""
        return self.vocab.copy()


def build_char_tokenizer(docs):
    """Build simple character-level tokenizer (original method).
    
    Args:
        docs: List of document strings
        
    Returns:
        Tuple of (vocab_size, stoi, itos, BOS)
        - vocab_size: int, size of vocabulary
        - stoi: dict, character to ID mapping
        - itos: dict, ID to character mapping  
        - BOS: int, ID of BOS token
    """
    chars = ['<BOS>'] + sorted(set(''.join(docs)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    BOS = stoi['<BOS>']
    return vocab_size, stoi, itos, BOS
