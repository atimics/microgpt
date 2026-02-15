"""
Fast training path for microGPT. Uses contiguous flat memory + C extension for speed.
Same algorithm as train.py (the pure-Python reference); use the equivalence test to verify.
Requires the fastops C extension (python setup.py build_ext --inplace).
"""

import os       # for os.path.exists
import time     # for time.perf_counter
import math     # for math.log, math.exp
import random   # for random.seed, random.choices
import json     # for json.dump
import argparse # for argparse.ArgumentParser
import array as _array  # for contiguous double arrays
_DA = _array.array      # shorthand for array.array constructor

import fastops as _C

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n-embd', type=int, default=16, help='Number of channels in the Transformer')
parser.add_argument('--n-layer', type=int, default=1, help='Number of layers in the Transformer')
parser.add_argument('--block-size', type=int, default=8, help='Maximum sequence length')
parser.add_argument('--num-steps', type=int, default=500, help='Number of training steps')
parser.add_argument('--n-head', type=int, default=4, help='Number of attention heads in the Transformer')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()
n_embd, block_size, n_layer, n_head = args.n_embd, args.block_size, args.n_layer, args.n_head
head_dim = n_embd // n_head
random.seed(args.seed)

# Dataset example: the names dataset (one name per line). rest of the code just assumes docs: list[str]
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)

# Tokenizer: simple character-level tokenization with a BOS token delimiter
chars = ['<BOS>'] + sorted(set(''.join(docs)))
vocab_size = len(chars)
stoi = { ch:i for i, ch in enumerate(chars) } # string to integer
itos = { i:ch for i, ch in enumerate(chars) } # integer to string
BOS = stoi['<BOS>']
docs_tokenized = [[BOS] + [stoi[ch] for ch in doc] + [BOS] for doc in docs]
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

    def __init__(self, nout, nin, std=0.02):
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
    state_dict[f'layer{i}.mlp_fc1'] = FlatParam(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = FlatParam(n_embd, 4 * n_embd, std=0)
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
beta1, beta2, eps_adam = 0.9, 0.95, 1e-8

# Training loop
lossf_history = []
t_start = time.perf_counter()
for step in range(args.num_steps):

    # Take a single training document (pre-tokenized, surrounded with BOS on both sides)
    tokens = docs_tokenized[step % len(docs)]
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

    # Adam update (flat — one C call per parameter, not per row)
    lr_t = learning_rate * (1 - step / args.num_steps)
    bc1 = 1.0 - beta1 ** (step + 1)
    bc2 = 1.0 - beta2 ** (step + 1)
    for p in params:
        _C.adam_update_flat(p.data, p.grad, p.m, p.v,
                            lr_t, beta1, beta2, bc1, bc2, eps_adam)

    lossf_history.append(loss.data[0])
    print(f"step {step+1:4d} / {args.num_steps:4d} | loss {loss.data[0]:.4f}")
print(f"mean loss last 50 steps: {sum(lossf_history[-50:]) / len(lossf_history[-50:]):.4f}")
print(f"training time: {time.perf_counter() - t_start:.2f}s")

# Inference: generate 20 samples (no autograd)
temperature = 0.5
generated_samples = []
print("\n--- inference ---")
for sample_idx in range(20):
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
    },
    'num_params': num_params,
    'vocab_size': vocab_size,
    'num_docs': len(docs),
    'loss_history': lossf_history,
    'loss_final': lossf_history[-1],
    'loss_mean_last_50': mean_loss_last_50,
    'training_time_seconds': round(training_time, 4),
    'generated_samples': generated_samples,
}
with open('_last_run.json', 'w') as f:
    json.dump(run_metrics, f, indent=2)
