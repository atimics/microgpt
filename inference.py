"""
Standalone inference script for MicroGPT.
Supports temperature, top-k, and top-p (nucleus) sampling with interactive mode.
"""

import os
import sys
import json
import math
import random
import argparse
import array as _array

_DA = _array.array

# Optional C extension for accelerated ops
try:
    import fastops as _C
    HAS_C = True
except ImportError:
    _C = None
    HAS_C = False

# CLI arguments
parser = argparse.ArgumentParser(description='Generate text using a trained MicroGPT model')
parser.add_argument('--model', type=str, required=True, help='Path to model JSON file')
parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling (0, 1]. Lower = more conservative, higher = more creative')
parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling: only sample from top k tokens (0 = disabled)')
parser.add_argument('--top-p', type=float, default=0.0, help='Top-p (nucleus) sampling: only sample from tokens that sum to p probability (0 = disabled)')
parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to generate')
parser.add_argument('--max-length', type=int, default=None, help='Maximum length of generated samples (default: block_size from model)')
parser.add_argument('--interactive', action='store_true', help='Interactive mode: generate one sample at a time')
parser.add_argument('--stream', action='store_true', help='Stream output: print characters as they are generated')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

# Load model
if not os.path.exists(args.model):
    print(f"Error: Model file not found: {args.model}", file=sys.stderr)
    sys.exit(1)

print(f"Loading model from {args.model}...")
with open(args.model, 'r') as f:
    model_data = json.load(f)

# Extract hyperparameters
hyperparams = model_data['hyperparams']
n_embd = hyperparams['n_embd']
n_layer = hyperparams['n_layer']
n_head = hyperparams['n_head']
block_size = hyperparams['block_size']
vocab_size = model_data['vocab_size']
head_dim = n_embd // n_head

# Extract tokenizer
stoi = {k: int(v) for k, v in model_data['tokenizer']['stoi'].items()}
itos = {int(k): v for k, v in model_data['tokenizer']['itos'].items()}
BOS = stoi['<BOS>']

# Reconstruct state_dict from saved weights
state_dict = {}
weights = model_data['weights']
for name, weight_data in weights.items():
    # Convert nested lists back to array.array format
    if isinstance(weight_data[0], list):
        # It's a 2D matrix
        state_dict[name] = [_DA('d', row) for row in weight_data]
    else:
        # It's a 1D vector (shouldn't happen in our case, but handle it)
        state_dict[name] = _DA('d', weight_data)

print(f"Model loaded: {n_layer} layers, {n_embd} embedding dim, {vocab_size} vocab size")

# Set max_length
max_length = args.max_length if args.max_length is not None else block_size
if max_length > block_size:
    print(f"Warning: max_length ({max_length}) exceeds model's block_size ({block_size}), using block_size", file=sys.stderr)
    max_length = block_size

# Plain-list matrix operations for inference (from train.py)
def rmsnorm_infer(x):
    """RMS normalization for inference."""
    eps = 1e-5
    ss = sum(xi * xi for xi in x)
    rms = math.sqrt(ss / len(x) + eps)
    return [xi / rms for xi in x]

def matvec_infer(mat, vec):
    """Matrix-vector multiply for inference."""
    if HAS_C:
        n_out, n_in = len(mat), len(mat[0])
        out = _DA('d', bytes(n_out * 8))
        _C.matvec_flat(
            _DA('d', [x for row in mat for x in row]),
            _DA('d', vec),
            out,
            n_out,
            n_in
        )
        return list(out)
    else:
        return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def vecadd_infer(a, b):
    """Vector addition for inference."""
    if HAS_C:
        out = _DA('d', a)
        _C.vec_axpy(1.0, _DA('d', b), out)
        return list(out)
    else:
        return [a[i] + b[i] for i in range(len(a))]

def squared_relu_infer(x):
    """Squared ReLU activation for inference."""
    return [xi * xi if xi > 0 else 0 for xi in x]

def attention_infer(q, keys, values, n_head, head_dim):
    """Multi-head attention for inference."""
    n_embd = len(q)
    T = len(keys)
    
    # Reshape q, k, v for multi-head attention
    q_heads = [[q[h * head_dim + i] for i in range(head_dim)] for h in range(n_head)]
    k_heads = [[[keys[t][h * head_dim + i] for i in range(head_dim)] for t in range(T)] for h in range(n_head)]
    v_heads = [[[values[t][h * head_dim + i] for i in range(head_dim)] for t in range(T)] for h in range(n_head)]
    
    # Attention for each head
    head_outputs = []
    for h in range(n_head):
        # Compute attention scores
        scores = [sum(q_heads[h][i] * k_heads[h][t][i] for i in range(head_dim)) / math.sqrt(head_dim) 
                  for t in range(T)]
        # Softmax
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        att_weights = [e / sum_exp for e in exp_scores]
        # Weighted sum of values
        head_out = [sum(att_weights[t] * v_heads[h][t][i] for t in range(T)) for i in range(head_dim)]
        head_outputs.extend(head_out)
    
    return head_outputs

def gpt_inference(token_id, pos_id, keys, values):
    """GPT forward pass for inference using plain Python lists."""
    # Token + position embedding
    tok_emb = list(state_dict['wte'][token_id])
    pos_emb = list(state_dict['wpe'][pos_id])
    x = vecadd_infer(tok_emb, pos_emb)
    x = rmsnorm_infer(x)
    
    # Transformer layers
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm_infer(x)
        
        # Attention
        q = matvec_infer(state_dict[f'layer{li}.attn_wq'], x)
        k = matvec_infer(state_dict[f'layer{li}.attn_wk'], x)
        v = matvec_infer(state_dict[f'layer{li}.attn_wv'], x)
        keys[li].append(k)
        values[li].append(v)
        x_attn = attention_infer(q, keys[li], values[li], n_head, head_dim)
        x = vecadd_infer(matvec_infer(state_dict[f'layer{li}.attn_wo'], x_attn), x_residual)
        
        # MLP
        x_residual = x
        x = rmsnorm_infer(x)
        x = matvec_infer(state_dict[f'layer{li}.mlp_fc1'], x)
        x = squared_relu_infer(x)
        x = matvec_infer(state_dict[f'layer{li}.mlp_fc2'], x)
        x = vecadd_infer(x, x_residual)
    
    # Output projection
    return matvec_infer(state_dict['lm_head'], x)

def sample_token(logits, temperature, top_k, top_p):
    """Sample a token from logits using temperature, top-k, and top-p sampling."""
    # Apply temperature
    logits_t = [l / temperature for l in logits]
    
    # Convert to probabilities
    max_val = max(logits_t)
    exps = [math.exp(l - max_val) for l in logits_t]
    total = sum(exps)
    probs = [e / total for e in exps]
    
    # Create list of (index, prob) pairs
    indexed_probs = list(enumerate(probs))
    
    # Apply top-k filtering
    if top_k > 0:
        # Sort by probability and keep only top-k
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        indexed_probs = indexed_probs[:top_k]
        # Renormalize
        total_prob = sum(p for _, p in indexed_probs)
        indexed_probs = [(idx, p / total_prob) for idx, p in indexed_probs]
    
    # Apply top-p (nucleus) filtering
    if top_p > 0.0 and top_p < 1.0:
        # Sort by probability
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        # Keep tokens until cumulative probability exceeds top_p
        cumulative = 0.0
        cutoff = len(indexed_probs)
        for i, (_, p) in enumerate(indexed_probs):
            cumulative += p
            if cumulative >= top_p:
                cutoff = i + 1
                break
        indexed_probs = indexed_probs[:cutoff]
        # Renormalize
        total_prob = sum(p for _, p in indexed_probs)
        indexed_probs = [(idx, p / total_prob) for idx, p in indexed_probs]
    
    # Sample from the filtered distribution
    indices = [idx for idx, _ in indexed_probs]
    weights = [p for _, p in indexed_probs]
    return random.choices(indices, weights=weights)[0]

def generate_sample(stream=False):
    """Generate a single sample."""
    keys_cache = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]
    token_id = BOS
    sample_chars = []
    
    for pos_id in range(max_length):
        logits = gpt_inference(token_id, pos_id, keys_cache, values_cache)
        token_id = sample_token(logits, args.temperature, args.top_k, args.top_p)
        
        if token_id == BOS:
            break
        
        char = itos[token_id]
        sample_chars.append(char)
        
        if stream:
            print(char, end='', flush=True)
    
    if stream:
        print()  # Newline after streaming
    
    return ''.join(sample_chars)

# Main generation loop
print(f"\nGenerating with temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
print("=" * 60)

if args.interactive:
    # Interactive mode
    sample_num = 1
    while True:
        try:
            input(f"\nPress Enter to generate sample {sample_num} (Ctrl+C to quit)...")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        if args.stream:
            print(f"Sample {sample_num}: ", end='', flush=True)
            generate_sample(stream=True)
        else:
            sample = generate_sample(stream=False)
            print(f"Sample {sample_num}: {sample}")
        
        sample_num += 1
else:
    # Batch mode
    for i in range(args.num_samples):
        if args.stream:
            print(f"Sample {i+1}: ", end='', flush=True)
            generate_sample(stream=True)
        else:
            sample = generate_sample(stream=False)
            print(f"Sample {i+1}: {sample}")
