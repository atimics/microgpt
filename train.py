"""
The most atomic way to train and inference a GPT LLM in pure, dependency-free Python.
Differences from GPT-2 are minor: layer norm -> rmsnorm, no biases, GeLU -> square ReLU, no weight tying.
The contents of this file is everything algorithmically needed to train a GPT. Everything else is just efficiency.
Art project by @karpathy.
"""

import os       # for os.path.exists
import time     # for time.perf_counter
import math     # for math.log, math.exp
import random   # for random.seed, random.choices
import json     # for json.dump
import argparse # for argparse.ArgumentParser

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n-embd', type=int, default=16, help='Number of channels in the Transformer')
parser.add_argument('--n-layer', type=int, default=1, help='Number of layers in the Transformer')
parser.add_argument('--block-size', type=int, default=8, help='Maximum sequence length')
parser.add_argument('--num-steps', type=int, default=500, help='Number of training steps')
parser.add_argument('--n-head', type=int, default=4, help='Number of attention heads in the Transformer')
parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
args = parser.parse_args()
n_embd, block_size, n_layer, n_head = args.n_embd, args.block_size, args.n_layer, args.n_head
head_dim = n_embd // n_head
random.seed(42)

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
print(f"vocab size: {vocab_size}, num docs: {len(docs)}")

# Autograd engine: vector-level (each node is a 1D vector, not a scalar)
class Tensor:
    """ stores a 1D vector of floats and its gradient """
    __slots__ = ('data', 'grad', '_backward', '_prev')

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = [0.0] * len(data)
        self._backward = lambda: None
        self._prev = set(_children)

    def backward(self):
        # iterative topological sort (avoids recursion stack overflow on deep graphs)
        topo = []
        visited = set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))
        self.grad[0] = 1.0
        for v in reversed(topo):
            v._backward()

class Param:
    """ a 2D weight matrix with gradient and Adam optimizer state """
    __slots__ = ('data', 'grad', 'm', 'v')

    def __init__(self, nout, nin, std=0.02):
        self.data = [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]
        self.grad = [[0.0] * nin for _ in range(nout)]
        self.m = [[0.0] * nin for _ in range(nout)] # Adam first moment
        self.v = [[0.0] * nin for _ in range(nout)] # Adam second moment

    def zero_grad(self):
        for row in self.grad:
            for j in range(len(row)):
                row[j] = 0.0

# Model parameter initialization
state_dict = {'wte': Param(vocab_size, n_embd), 'wpe': Param(block_size, n_embd), 'lm_head': Param(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = Param(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = Param(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = Param(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = Param(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = Param(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = Param(n_embd, 4 * n_embd, std=0)
params = list(state_dict.values())
num_params = sum(len(p.data) * len(p.data[0]) for p in params)
print(f"num params: {num_params}")

# Fused operations: each computes an entire vector operation as a single autograd node

def embedding(param, idx):
    out = Tensor(list(param.data[idx]))
    def _backward():
        og = out.grad
        pg = param.grad[idx]
        for j in range(len(og)):
            pg[j] += og[j]
    out._backward = _backward
    return out

def linear(x, w):
    n_out, n_in = len(w.data), len(w.data[0])
    xd = x.data
    out_data = [0.0] * n_out
    for i in range(n_out):
        s = 0.0
        wi = w.data[i]
        for j in range(n_in):
            s += wi[j] * xd[j]
        out_data[i] = s
    out = Tensor(out_data, (x,))
    def _backward():
        xd = x.data
        xg = x.grad
        for i in range(n_out):
            gi = out.grad[i]
            if gi == 0.0:
                continue
            wi = w.data[i]
            wgi = w.grad[i]
            for j in range(n_in):
                wgi[j] += gi * xd[j]
                xg[j] += gi * wi[j]
    out._backward = _backward
    return out

def rmsnorm(x):
    n = len(x.data)
    xd = x.data
    ms = sum(xi * xi for xi in xd) / n
    scale = (ms + 1e-5) ** -0.5
    out = Tensor([xi * scale for xi in xd], (x,))
    def _backward():
        dot = sum(out.grad[i] * xd[i] for i in range(n))
        s3n = scale * scale * scale / n
        for j in range(n):
            x.grad[j] += out.grad[j] * scale - s3n * xd[j] * dot
    out._backward = _backward
    return out

def tensor_add(a, b):
    out = Tensor([ai + bi for ai, bi in zip(a.data, b.data)], (a, b))
    def _backward():
        for i in range(len(a.data)):
            a.grad[i] += out.grad[i]
            b.grad[i] += out.grad[i]
    out._backward = _backward
    return out

def squared_relu(x):
    xd = x.data
    out = Tensor([max(0.0, xi) ** 2 for xi in xd], (x,))
    def _backward():
        for i in range(len(xd)):
            if xd[i] > 0.0:
                x.grad[i] += 2.0 * xd[i] * out.grad[i]
    out._backward = _backward
    return out

def attention(q, keys, values, n_head, head_dim):
    T = len(keys)
    scale = head_dim ** 0.5
    out_data = [0.0] * (n_head * head_dim)
    all_attn_weights = []
    for h in range(n_head):
        hs = h * head_dim
        qd = q.data
        attn_logits = [0.0] * T
        for t in range(T):
            s = 0.0
            kd = keys[t].data
            for j in range(head_dim):
                s += qd[hs + j] * kd[hs + j]
            attn_logits[t] = s / scale
        max_val = max(attn_logits) if T > 0 else 0.0
        exps = [math.exp(a - max_val) for a in attn_logits]
        total = sum(exps)
        attn_w = [e / total for e in exps]
        all_attn_weights.append(attn_w)
        for j in range(head_dim):
            s = 0.0
            idx = hs + j
            for t in range(T):
                s += attn_w[t] * values[t].data[idx]
            out_data[idx] = s
    children = (q,) + tuple(keys) + tuple(values)
    out = Tensor(out_data, children)
    def _backward():
        for h in range(n_head):
            hs = h * head_dim
            attn_w = all_attn_weights[h]
            d_attn = [0.0] * T
            for j in range(head_dim):
                g = out.grad[hs + j]
                if g == 0.0:
                    continue
                idx = hs + j
                for t in range(T):
                    values[t].grad[idx] += g * attn_w[t]
                    d_attn[t] += g * values[t].data[idx]
            dot = sum(attn_w[t] * d_attn[t] for t in range(T))
            for t in range(T):
                dl = attn_w[t] * (d_attn[t] - dot) / scale
                if dl == 0.0:
                    continue
                for j in range(head_dim):
                    idx = hs + j
                    q.grad[idx] += dl * keys[t].data[idx]
                    keys[t].grad[idx] += dl * q.data[idx]
    out._backward = _backward
    return out

def cross_entropy(logits, target):
    max_val = max(logits.data)
    exps = [math.exp(v - max_val) for v in logits.data]
    total = sum(exps)
    probs = [e / total for e in exps]
    loss = -math.log(probs[target])
    out = Tensor([loss], (logits,))
    def _backward():
        g = out.grad[0]
        for i in range(len(logits.data)):
            logits.grad[i] += g * (probs[i] - (1.0 if i == target else 0.0))
    out._backward = _backward
    return out

def mean_loss(losses):
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
def gpt(token_id, pos_id, keys, values):
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
def gpt_inference(token_id, pos_id, keys, values):
    sd = state_dict
    x = [t + p for t, p in zip(sd['wte'].data[token_id], sd['wpe'].data[pos_id])]
    def _rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        return [xi * (ms + 1e-5) ** -0.5 for xi in x]
    def _linear(x, w):
        return [sum(wi[j] * x[j] for j in range(len(x))) for wi in w.data]
    x = _rmsnorm(x)
    for li in range(n_layer):
        xr = list(x)
        x = _rmsnorm(x)
        q = _linear(x, sd[f'layer{li}.attn_wq'])
        k = _linear(x, sd[f'layer{li}.attn_wk'])
        v = _linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            attn_logits = [sum(q[hs+j] * keys[li][t][hs+j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(keys[li]))]
            mx = max(attn_logits)
            exps = [math.exp(a - mx) for a in attn_logits]
            total = sum(exps)
            aw = [e / total for e in exps]
            for j in range(head_dim):
                x_attn.append(sum(aw[t] * values[li][t][hs+j] for t in range(len(values[li]))))
        x = _linear(x_attn, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, xr)]
        xr = list(x)
        x = _rmsnorm(x)
        x = _linear(x, sd[f'layer{li}.mlp_fc1'])
        x = [max(0.0, xi)**2 for xi in x]
        x = _linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, xr)]
    return _linear(x, sd['lm_head'])

# Adam optimizer
learning_rate = args.learning_rate
beta1, beta2, eps_adam = 0.9, 0.95, 1e-8

# Training loop
lossf_history = []
t_start = time.perf_counter()
for step in range(args.num_steps):

    # Take a single training document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
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

    # Adam update (optimizer)
    lr_t = learning_rate * (1 - step / args.num_steps)
    for p in params:
        for i in range(len(p.data)):
            pd, pg, pm, pv = p.data[i], p.grad[i], p.m[i], p.v[i]
            for j in range(len(pd)):
                g = pg[j]
                pm[j] = beta1 * pm[j] + (1 - beta1) * g
                pv[j] = beta2 * pv[j] + (1 - beta2) * g * g
                m_hat = pm[j] / (1 - beta1 ** (step + 1))
                v_hat = pv[j] / (1 - beta2 ** (step + 1))
                pd[j] -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

    lossf_history.append(loss.data[0])
    print(f"step {step+1:4d} / {args.num_steps:4d} | loss {loss.data[0]:.4f}")
print(f"mean loss last 50 steps: {sum(lossf_history[-50:]) / len(lossf_history[-50:]):.4f}") # ~usable for basic kwarg tuning
print(f"training time: {time.perf_counter() - t_start:.2f}s") # ~usable for basic performance benchmarking

# Inference: generate 20 samples (no autograd)
temperature = 0.5 # number in (0, 1] that controls the "creativity" of generated text, low to high
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
