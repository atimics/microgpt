"""
Fast training path for microGPT. Uses contiguous flat memory + C extension for speed.
Same algorithm as train.py (the pure-Python reference); use the equivalence test to verify.
Requires the fastops C extension (python setup.py build_ext --inplace).
"""

import os       # for os.path.exists
import sys      # for sys.stderr, sys.exit
import time     # for time.perf_counter
import math     # for math.log, math.exp
import random   # for random.seed, random.choices
import json     # for json.dump
import argparse # for argparse.ArgumentParser
import array as _array  # for contiguous double arrays
_DA = _array.array      # shorthand for array.array constructor

import fastops as _C

# Model/Optimizer Constants (grouped for visibility and documentation)
# These are architectural choices and numerical stability parameters that are typically fixed.
# They can be modified here if needed for experimentation.
# Note: RMSNorm epsilon (1e-5) is hardcoded in the C extension (fastops.c)

WEIGHT_INIT_STD = 0.02        # Standard deviation for weight initialization (Gaussian)
MLP_HIDDEN_DIM_MULTIPLIER = 4 # MLP hidden dimension as a multiple of n_embd (standard is 4x)
ADAM_BETA1 = 0.9              # Adam optimizer momentum parameter (first moment)
ADAM_BETA2 = 0.95             # Adam optimizer momentum parameter (second moment)
ADAM_EPS = 1e-8               # Adam epsilon for numerical stability

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n-embd', type=int, default=16, help='Number of channels in the Transformer')
parser.add_argument('--n-layer', type=int, default=1, help='Number of layers in the Transformer')
parser.add_argument('--block-size', type=int, default=8, help='Maximum sequence length')
parser.add_argument('--num-steps', type=int, default=500, help='Number of training steps')
parser.add_argument('--n-head', type=int, default=4, help='Number of attention heads in the Transformer')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--grad-clip', type=float, default=0.0, help='Max gradient norm (0 = disabled)')
parser.add_argument('--lr-schedule', type=str, default='linear', choices=['linear', 'cosine'], help='Learning rate schedule')
parser.add_argument('--warmup-steps', type=int, default=0, help='Number of warmup steps')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--temperature', type=float, default=0.5, help='Sampling temperature for inference; typical range (0, 1], lower = less random')
parser.add_argument('--num-samples', type=int, default=20, help='Number of samples to generate during inference')
parser.add_argument('--val-split', type=float, default=0.0, help='Fraction of data for validation (0 = no validation)')
parser.add_argument('--val-every', type=int, default=50, help='Evaluate validation loss every N steps')
parser.add_argument('--early-stop-patience', type=int, default=0, help='Stop after N evaluations without improvement (0 = no early stopping)')
args = parser.parse_args()
n_embd, block_size, n_layer, n_head = args.n_embd, args.block_size, args.n_layer, args.n_head
if n_embd < n_head:
    parser.error(f"n_embd ({n_embd}) must be >= n_head ({n_head}) to avoid division by zero in attention")

# Validate hyperparameter configurations
if n_embd <= 0:
    raise ValueError("n_embd must be positive")
if n_layer <= 0:
    raise ValueError("n_layer must be positive")
if n_head <= 0:
    raise ValueError("n_head must be positive")
if block_size <= 0:
    raise ValueError("block_size must be positive")
if args.num_steps <= 0:
    raise ValueError("num_steps must be positive")
if args.learning_rate <= 0:
    raise ValueError("learning_rate must be positive")
if n_embd % n_head != 0:
    raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
head_dim = n_embd // n_head
random.seed(args.seed)

# Dataset example: the names dataset (one name per line). rest of the code just assumes docs: list[str]
if not os.path.exists('input.txt'):
    import urllib.request
    import tempfile
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    tmp_path = None
    try:
        # Download to temporary file first for atomic operation
        fd, tmp_path = tempfile.mkstemp(dir='.', prefix='.input_', suffix='.txt.tmp')
        os.close(fd)  # Close the file descriptor, we'll use the path with urlretrieve
        urllib.request.urlretrieve(url, tmp_path)
        # Atomic rename to final location
        os.replace(tmp_path, 'input.txt')
    except Exception as e:
        # Clean up temp file if it exists
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"Error downloading dataset from {url}: {e}", file=sys.stderr)
        print("Please manually download the dataset and save it as 'input.txt', or provide your own text file.", file=sys.stderr)
        sys.exit(1)
with open('input.txt') as f:
    docs = [l.strip() for l in f.read().strip().split('\n') if l.strip()] # list[str] of documents
if not docs:
    print("Error: input.txt contains no non-empty lines", file=sys.stderr)
    sys.exit(1)
random.shuffle(docs)
# Split into train and validation sets (deterministic with seed)
if args.val_split > 0.0:
    val_size = int(len(docs) * args.val_split)
    if val_size == 0:
        val_size = 1  # ensure at least 1 validation document
    train_docs = docs[val_size:]
    val_docs = docs[:val_size]
    print(f"split: {len(train_docs)} train docs, {len(val_docs)} val docs")
else:
    train_docs = docs
    val_docs = []

# Tokenizer: simple character-level tokenization with a BOS token delimiter
chars = ['<BOS>'] + sorted(set(''.join(docs)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) } # string to integer
itos = { i:ch for i, ch in enumerate(chars) } # integer to string
BOS = stoi['<BOS>']
train_docs_tokenized = [[BOS] + [stoi[ch] for ch in doc] + [BOS] for doc in train_docs]
val_docs_tokenized = [[BOS] + [stoi[ch] for ch in doc] + [BOS] for doc in val_docs] if val_docs else []
print(f"vocab size: {vocab_size}, num docs: {len(docs)}")

# Autograd engine: vector-level (each node is a 1D vector, not a scalar)
class Tensor:
    """ stores a 1D vector of floats and its gradient """
    __slots__ = ('data', 'grad', '_backward', '_prev')

    def __init__(self, data, _children=()):
        self.data = data if isinstance(data, _array.array) else _DA('d', data)
        self.grad = None  # lazily allocated on first backward
        self._backward = lambda: None
        self._prev = _children

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = _DA('d', bytes(len(self.data) * 8))  # zero-initialized contiguous doubles

    def backward(self):
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack[-1]
            if processed:
                stack.pop()
                topo.append(node)
                continue
            if id(node) in visited:
                stack.pop()
                continue
            visited.add(id(node))
            stack[-1] = (node, True)
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))
        for v in topo:
            v._ensure_grad()
        self.grad[0] = 1.0
        for v in reversed(topo):
            v._backward()

class FlatParam:
    """
    2D weight matrix stored as a single contiguous array.array('d', nout * nin).
    This eliminates per-row Python API calls in the C extension — one buffer access
    instead of nout for matvec, linear_backward, adam_update, and zero_grad.
    """
    __slots__ = ('data', 'grad', 'm', 'v', 'nout', 'nin')

    def __init__(self, nout, nin, std=WEIGHT_INIT_STD):
        self.nout, self.nin = nout, nin
        self.data = _DA('d', [random.gauss(0, std) for _ in range(nout * nin)])
        self.grad = _DA('d', bytes(nout * nin * 8))
        self.m = _DA('d', bytes(nout * nin * 8))  # Adam first moment
        self.v = _DA('d', bytes(nout * nin * 8))  # Adam second moment

    def zero_grad(self):
        _C.zero_grad_flat(self.grad)

# Model parameter initialization
state_dict = {'wte': FlatParam(vocab_size, n_embd), 'wpe': FlatParam(block_size, n_embd), 'lm_head': FlatParam(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = FlatParam(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = FlatParam(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = FlatParam(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = FlatParam(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = FlatParam(MLP_HIDDEN_DIM_MULTIPLIER * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = FlatParam(n_embd, MLP_HIDDEN_DIM_MULTIPLIER * n_embd, std=0)
params = list(state_dict.values())
num_params = sum(p.nout * p.nin for p in params)
print(f"num params: {num_params}")

# Fused operations using flat C functions

def embedding(param: FlatParam, idx: int) -> Tensor:
    out = Tensor(_C.embedding_flat(param.data, idx, param.nin))
    def _backward():
        # Accumulate gradient into the correct row of the flat grad buffer
        og = out.grad
        offset = idx * param.nin
        pg = param.grad
        for j in range(param.nin):
            pg[offset + j] += og[j]
    out._backward = _backward
    return out

def linear(x: Tensor, w: FlatParam) -> Tensor:
    n_out, n_in = w.nout, w.nin
    out_data = _C.matvec_flat(w.data, x.data, n_out, n_in)
    out = Tensor(out_data, (x,))
    def _backward():
        _C.linear_backward_flat(out.grad, w.data, w.grad, x.data, x.grad,
                                n_out, n_in)
    out._backward = _backward
    return out

def rmsnorm(x: Tensor) -> Tensor:
    xd = x.data
    out_data, scale = _C.rmsnorm_forward(xd)
    out = Tensor(out_data, (x,))
    def _backward():
        _C.rmsnorm_backward(out.grad, xd, scale, x.grad)
    out._backward = _backward
    return out

def tensor_add(a: Tensor, b: Tensor) -> Tensor:
    out_data = _C.tensor_add(a.data, b.data)
    out = Tensor(out_data, (a, b))
    def _backward():
        _C.tensor_add_backward(out.grad, a.grad, b.grad)
    out._backward = _backward
    return out

def squared_relu(x: Tensor) -> Tensor:
    xd = x.data
    out_data = _C.squared_relu_forward(xd)
    out = Tensor(out_data, (x,))
    def _backward():
        _C.squared_relu_backward(xd, out.grad, x.grad)
    out._backward = _backward
    return out

def attention(q: Tensor, keys: list[Tensor], values: list[Tensor],
              n_head: int, head_dim: int) -> Tensor:
    T = len(keys)
    children = (q,) + tuple(keys) + tuple(values)
    k_data = [k.data for k in keys]
    v_data = [v.data for v in values]
    out_data, all_attn_weights = _C.attention_forward(q.data, k_data, v_data, n_head, head_dim)
    out = Tensor(out_data, children)
    qd = q.data
    def _backward():
        qg = q.grad
        k_grad = [k.grad for k in keys]
        v_grad = [v.grad for v in values]
        _C.attention_backward(out.grad, qd, k_data, v_data,
                              all_attn_weights, qg, k_grad, v_grad,
                              n_head, head_dim)
    out._backward = _backward
    return out

def cross_entropy(logits: Tensor, target: int) -> Tensor:
    loss, probs = _C.cross_entropy_forward(logits.data, target)
    out = Tensor([loss], (logits,))
    def _backward():
        _C.cross_entropy_backward(out.grad[0], probs, target, logits.grad)
    out._backward = _backward
    return out

def mean_loss(losses: list[Tensor]) -> Tensor:
    n = len(losses)
    avg = sum(l.data[0] for l in losses) / n
    out = Tensor([avg], tuple(losses))
    def _backward():
        g = out.grad[0] / n
        for l in losses:
            l.grad[0] += g
    out._backward = _backward
    return out

# Model architecture (training: builds autograd graph)
def gpt(token_id: int, pos_id: int, keys: list[list[Tensor]], 
        values: list[list[Tensor]]) -> Tensor:
    tok_emb = embedding(state_dict['wte'], token_id)
    pos_emb = embedding(state_dict['wpe'], pos_id)
    x = tensor_add(tok_emb, pos_emb)
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = attention(q, keys[li], values[li], n_head, head_dim)
        x = tensor_add(linear(x_attn, state_dict[f'layer{li}.attn_wo']), x_residual)
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = squared_relu(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = tensor_add(x, x_residual)
    return linear(x, state_dict['lm_head'])

# Model architecture (inference: plain floats, no autograd overhead)
def gpt_inference(token_id: int, pos_id: int, keys: list[list], 
                  values: list[list]) -> _array.array:
    sd = state_dict
    x = _C.tensor_add(
        _C.embedding_flat(sd['wte'].data, token_id, n_embd),
        _C.embedding_flat(sd['wpe'].data, pos_id, n_embd),
    )
    def _rmsnorm(x):
        out, _ = _C.rmsnorm_forward(x)
        return out
    _linear = lambda x, w: _C.matvec_flat(w.data, x, w.nout, w.nin)
    _sqrelu = lambda x: _C.squared_relu_forward(x)
    x = _rmsnorm(x)
    for li in range(n_layer):
        xr = _DA('d', x)
        x = _rmsnorm(x)
        q = _linear(x, sd[f'layer{li}.attn_wq'])
        k = _linear(x, sd[f'layer{li}.attn_wk'])
        v = _linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn, _ = _C.attention_forward(q, keys[li], values[li], n_head, head_dim)
        x = _linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = _C.tensor_add(x, xr)
        xr = _DA('d', x)
        x = _rmsnorm(x)
        x = _linear(x, sd[f'layer{li}.mlp_fc1'])
        x = _sqrelu(x)
        x = _linear(x, sd[f'layer{li}.mlp_fc2'])
        x = _C.tensor_add(x, xr)
    return _linear(x, sd['lm_head'])

# Adam optimizer
learning_rate = args.learning_rate
beta1, beta2, eps_adam = ADAM_BETA1, ADAM_BETA2, ADAM_EPS

# Validation evaluation function
def evaluate_validation():
    """Evaluate loss on validation set without updating gradients."""
    if not val_docs_tokenized:
        return None
    val_losses = []
    for doc_tokens in val_docs_tokenized:
        n = min(block_size, len(doc_tokens) - 1)
        keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        doc_losses = []
        for pos_id in range(n):
            token_id, target_id = doc_tokens[pos_id], doc_tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys_cache, values_cache)
            doc_losses.append(cross_entropy(logits, target_id))
        val_losses.append(mean_loss(doc_losses).data[0])
    return sum(val_losses) / len(val_losses)

# Training loop
lossf_history = []
val_lossf_history = []
best_val_loss = float('inf')
patience_counter = 0
t_start = time.perf_counter()
for step in range(args.num_steps):

    # Take a single training document (pre-tokenized, surrounded with BOS on both sides)
    tokens = train_docs_tokenized[step % len(train_docs)]
    n = min(block_size, len(tokens) - 1)

    # Zero gradients on all parameters
    for p in params:
        p.zero_grad()

    # Forward/backward through the document over time dimension
    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        losses.append(cross_entropy(logits, target_id))
    loss = mean_loss(losses)
    loss.backward()

    # Gradient clipping
    if args.grad_clip > 0.0:
        total_norm = 0.0
        for p in params:
            for i in range(len(p.grad)):
                total_norm += p.grad[i] * p.grad[i]
        total_norm = total_norm ** 0.5
        if total_norm > args.grad_clip:
            scale = args.grad_clip / total_norm
            for p in params:
                for i in range(len(p.grad)):
                    p.grad[i] *= scale

    # Adam update (flat — one C call per parameter, not per row)
    # Compute learning rate with schedule and warmup
    if step < args.warmup_steps:
        # Linear warmup
        lr_t = learning_rate * (step + 1) / args.warmup_steps
    elif args.lr_schedule == 'cosine':
        # Cosine annealing after warmup
        progress = (step - args.warmup_steps) / (args.num_steps - args.warmup_steps)
        lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    else:  # linear
        # Linear decay after warmup
        progress = (step - args.warmup_steps) / (args.num_steps - args.warmup_steps)
        lr_t = learning_rate * (1 - progress)
    bc1 = 1.0 - beta1 ** (step + 1)
    bc2 = 1.0 - beta2 ** (step + 1)
    for p in params:
        _C.adam_update_flat(p.data, p.grad, p.m, p.v,
                            lr_t, beta1, beta2, bc1, bc2, eps_adam)

    lossf_history.append(loss.data[0])
    
    # Evaluate on validation set periodically
    val_loss_str = ""
    if val_docs_tokenized and (step + 1) % args.val_every == 0:
        val_loss = evaluate_validation()
        val_lossf_history.append(val_loss)
        val_loss_str = f" | val_loss {val_loss:.4f}"
        
        # Early stopping logic
        if args.early_stop_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"step {step+1:4d} / {args.num_steps:4d} | loss {loss.data[0]:.4f}{val_loss_str}")
                    print(f"Early stopping: validation loss did not improve for {args.early_stop_patience} evaluations")
                    # Note: lossf_history already has correct length (step+1), trim is defensive
                    # val_lossf_history is correct since it's only appended during evaluation
                    lossf_history = lossf_history[:step+1]
                    break
    
    print(f"step {step+1:4d} / {args.num_steps:4d} | loss {loss.data[0]:.4f}{val_loss_str}")
print(f"mean loss last 50 steps: {sum(lossf_history[-50:]) / len(lossf_history[-50:]):.4f}")
print(f"training time: {time.perf_counter() - t_start:.2f}s")

# Inference: generate samples (no autograd)
temperature = args.temperature
generated_samples = []
print("\n--- inference ---")
for sample_idx in range(args.num_samples):
    keys_cache, values_cache = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample_chars = []
    for pos_id in range(block_size):
        logits = gpt_inference(token_id, pos_id, keys_cache, values_cache)
        logits_t = [l / temperature for l in logits]
        max_val = max(logits_t)
        exps = [math.exp(l - max_val) for l in logits_t]
        total = sum(exps)
        probs = [e / total for e in exps]
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample_chars.append(itos[token_id])
    sample_str = ''.join(sample_chars)
    generated_samples.append(sample_str)
    print(f"sample {sample_idx+1}: {sample_str}")

# Save run metrics to JSON for the training harness
training_time = time.perf_counter() - t_start
mean_loss_last_50 = sum(lossf_history[-50:]) / len(lossf_history[-50:])
run_metrics = {
    'hyperparams': {
        'n_embd': n_embd, 'n_layer': n_layer, 'n_head': n_head,
        'block_size': block_size, 'num_steps': args.num_steps,
        'learning_rate': args.learning_rate,
        'temperature': args.temperature,
        'num_samples': args.num_samples,
        'val_split': args.val_split,
        'val_every': args.val_every,
        'early_stop_patience': args.early_stop_patience,
    },
    'num_params': num_params,
    'vocab_size': vocab_size,
    'num_docs': len(docs),
    'num_train_docs': len(train_docs),
    'num_val_docs': len(val_docs),
    'loss_history': lossf_history,
    'val_loss_history': val_lossf_history,
    'loss_final': lossf_history[-1],
    'loss_mean_last_50': mean_loss_last_50,
    'training_time_seconds': round(training_time, 4),
    'generated_samples': generated_samples,
}
with open('_last_run.json', 'w') as f:
    json.dump(run_metrics, f, indent=2)
