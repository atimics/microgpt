#!/usr/bin/env python3
"""
Tests for BPE tokenizer implementation.
"""

import os
import json
import tempfile
from tokenizer import BPETokenizer, build_char_tokenizer


def test_char_tokenizer():
    """Test character-level tokenizer (baseline)."""
    docs = ['hello', 'world', 'hello world']
    vocab_size, stoi, itos, BOS = build_char_tokenizer(docs)
    
    # Check basic properties
    assert vocab_size > 0
    assert BOS == 0
    assert stoi['<BOS>'] == 0
    assert itos[0] == '<BOS>'
    
    # Check all characters are in vocab
    for doc in docs:
        for char in doc:
            assert char in stoi
    
    print("  PASS: Character tokenizer")


def test_bpe_tokenizer_basic():
    """Test basic BPE tokenizer functionality."""
    tokenizer = BPETokenizer()
    
    # Train on simple corpus
    docs = ['hello', 'hello', 'world', 'hello world']
    tokenizer.train(docs, vocab_size=30, verbose=False)
    
    # Check vocab size
    assert tokenizer.vocab_size >= len(set(''.join(docs))) + 1  # chars + BOS
    assert tokenizer.vocab_size <= 30
    
    # Check BOS token
    assert tokenizer.bos_token == '<BOS>'
    assert tokenizer.bos_id == 0
    
    # Test encode/decode
    text = 'hello'
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    
    print("  PASS: Basic BPE tokenizer")


def test_bpe_encode_decode():
    """Test BPE encode/decode correctness."""
    tokenizer = BPETokenizer()
    
    # Train on larger corpus
    docs = [
        'the quick brown fox',
        'the lazy dog',
        'quick brown',
        'the fox jumps',
    ]
    tokenizer.train(docs, vocab_size=50, verbose=False)
    
    # Test each document can be encoded and decoded
    for doc in docs:
        ids = tokenizer.encode(doc)
        decoded = tokenizer.decode(ids)
        assert decoded == doc, f"Encode/decode failed: '{doc}' -> '{decoded}'"
    
    print("  PASS: BPE encode/decode")


def test_bpe_merges():
    """Test that BPE actually creates merged tokens."""
    tokenizer = BPETokenizer()
    
    # Train with repeated patterns
    docs = ['aa'] * 10 + ['ab'] * 10  # 'aa' and 'ab' should be merged
    tokenizer.train(docs, vocab_size=20, verbose=False)
    
    # Check that merges happened
    assert len(tokenizer.merges) > 0, "No merges occurred"
    
    # Check that merged tokens are in vocab
    vocab = tokenizer.get_vocab()
    
    # 'aa' should be merged since it appears 10 times
    # We can't guarantee exact tokens, but vocab should be larger than base chars
    base_chars = set(''.join(docs))
    assert len(vocab) > len(base_chars) + 1, "Vocab didn't grow beyond base characters"
    
    print("  PASS: BPE merges")


def test_bpe_save_load():
    """Test saving and loading BPE tokenizer."""
    tokenizer1 = BPETokenizer()
    
    # Train
    docs = ['hello world', 'hello', 'world']
    tokenizer1.train(docs, vocab_size=25, verbose=False)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        tokenizer1.save(temp_path)
        
        # Load into new tokenizer
        tokenizer2 = BPETokenizer()
        tokenizer2.load(temp_path)
        
        # Check they're equivalent
        assert tokenizer1.vocab == tokenizer2.vocab
        assert tokenizer1.merges == tokenizer2.merges
        assert tokenizer1.bos_token == tokenizer2.bos_token
        assert tokenizer1.bos_id == tokenizer2.bos_id
        
        # Test encode/decode with loaded tokenizer
        text = 'hello'
        ids1 = tokenizer1.encode(text)
        ids2 = tokenizer2.encode(text)
        assert ids1 == ids2
        assert tokenizer2.decode(ids2) == text
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print("  PASS: BPE save/load")


def test_bpe_with_documents():
    """Test BPE with document-style data (similar to training data)."""
    tokenizer = BPETokenizer()
    
    # Simulate names dataset
    docs = [
        'emma',
        'olivia',
        'ava',
        'isabella',
        'sophia',
        'mia',
    ]
    
    tokenizer.train(docs, vocab_size=40, verbose=False)
    
    # Test encoding with BOS tokens (like in train.py)
    for doc in docs:
        # Encode the document
        ids = tokenizer.encode(doc)
        
        # Add BOS at start and end (like train.py does)
        ids_with_bos = [tokenizer.bos_id] + ids + [tokenizer.bos_id]
        
        # Decode (without BOS)
        decoded = tokenizer.decode(ids)
        assert decoded == doc
    
    print("  PASS: BPE with documents")


def test_bpe_vocab_size_limit():
    """Test that BPE respects vocab size limit."""
    tokenizer = BPETokenizer()
    
    docs = ['a'] * 100 + ['b'] * 100  # Lots of data
    target_vocab = 15
    tokenizer.train(docs, vocab_size=target_vocab, verbose=False)
    
    # Vocab should not exceed target
    assert tokenizer.vocab_size <= target_vocab
    
    print("  PASS: BPE vocab size limit")


def run_all_tests():
    """Run all BPE tokenizer tests."""
    print("Running BPE tokenizer tests...")
    
    tests = [
        test_char_tokenizer,
        test_bpe_tokenizer_basic,
        test_bpe_encode_decode,
        test_bpe_merges,
        test_bpe_save_load,
        test_bpe_with_documents,
        test_bpe_vocab_size_limit,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAIL: {test.__name__}")
            raise
    
    print("\nAll BPE tokenizer tests passed!")
    return True


if __name__ == '__main__':
    import sys
    try:
        run_all_tests()
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
