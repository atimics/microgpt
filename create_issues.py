#!/usr/bin/env python3
"""
Create GitHub issues for MicroGPT based on code review and roadmap analysis.

Usage:
  # Via GitHub Actions (automatic when pushed to create-issues branch)
  # Via manual workflow dispatch in GitHub UI
  # Locally with gh CLI installed:
  python create_issues.py

Requires: gh CLI authenticated, or GH_TOKEN environment variable.
"""

import subprocess
import json
import sys

REPO = "atimics/microgpt"

# ============================================================================
# Issue definitions: bugs, then features organized by roadmap phase
# ============================================================================

ISSUES = [
    # ========================================================================
    # BUGS
    # ========================================================================
    {
        "title": "Bug: No validation that n_embd is divisible by n_head",
        "labels": ["bug", "priority: high"],
        "body": """## Description

`head_dim = n_embd // n_head` (train.py:43, train_fast.py:29) performs integer division without validating that `n_embd` is evenly divisible by `n_head`. This silently drops dimensions.

## Steps to Reproduce

```bash
python train.py --n-embd 16 --n-head 5
```

`head_dim` becomes 3 (16 // 5), but `n_head * head_dim = 15 != 16`. The last embedding dimension is silently ignored, producing incorrect attention computation.

## Expected Behavior

The script should validate at startup that `n_embd % n_head == 0` and exit with a clear error message if not.

## Affected Files
- `train.py:43`
- `train_fast.py:29`

## Suggested Fix

```python
assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
```
""",
    },
    {
        "title": "Bug: No bounds checking on target index in C cross_entropy_forward",
        "labels": ["bug", "priority: high", "C extension"],
        "body": """## Description

In `fastops.c:267-288`, the `cross_entropy_forward` function uses `target` as an array index into `buf` without any bounds checking. If `target >= vocab_size` or `target < 0`, this results in undefined behavior (out-of-bounds memory access).

## Code Location

```c
// fastops.c:288
double loss = -log(buf[target]);  // no bounds check on target
```

## Impact

- Memory access violation / segfault
- Potential data corruption if the memory happens to be valid
- Security concern in educational code that students may adapt

## Suggested Fix

```c
if (target < 0 || target >= n) {
    free(buf);
    darr_done(&lg);
    PyErr_Format(PyExc_IndexError, "target index %zd out of range [0, %zd)", target, n);
    return NULL;
}
```
""",
    },
    {
        "title": "Bug: No bounds checking on idx in C embedding_flat",
        "labels": ["bug", "priority: high", "C extension"],
        "body": """## Description

In `fastops.c:481`, `embedding_flat` uses `idx * dim` as an offset into the data buffer without verifying it's within bounds.

```c
// fastops.c:481
PyObject *result = darr_new(d.ptr + idx * dim, dim);
```

If `idx` is negative or `idx * dim + dim > buffer_length`, this reads out-of-bounds memory.

## Impact

- Out-of-bounds read, potential segfault
- Could return garbage data silently

## Suggested Fix

```c
if (idx < 0 || (idx + 1) * dim > d.n) {
    darr_done(&d);
    PyErr_Format(PyExc_IndexError, "embedding index %zd out of range for buffer of %zd elements with dim %zd", idx, d.n, dim);
    return NULL;
}
```
""",
    },
    {
        "title": "Bug: File handle leak when reading input.txt",
        "labels": ["bug", "priority: medium"],
        "body": """## Description

Both `train.py:50` and `train_fast.py:36` open `input.txt` without using a context manager:

```python
docs = [l.strip() for l in open('input.txt').read().strip().split('\\n') if l.strip()]
```

This relies on garbage collection to close the file handle, which is not guaranteed to happen promptly (especially on non-CPython implementations like PyPy).

## Suggested Fix

```python
with open('input.txt') as f:
    docs = [l.strip() for l in f.read().strip().split('\\n') if l.strip()]
```

## Affected Files
- `train.py:50`
- `train_fast.py:36`
""",
    },
    {
        "title": "Bug: Dataset download has no error handling",
        "labels": ["bug", "priority: medium"],
        "body": """## Description

In `train.py:48-49` and `train_fast.py:33-35`, the dataset download via `urllib.request.urlretrieve()` has no error handling:

```python
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
```

If the network is unavailable, the URL is unreachable, or the download is interrupted, the script crashes with an unhelpful traceback. Worse, a partial download could leave a corrupt `input.txt` that persists across runs.

## Suggested Fix

- Wrap in try/except with a clear error message
- Download to a temp file first, then rename (atomic)
- Provide a fallback message directing users to supply their own `input.txt`
""",
    },
    {
        "title": "Bug: Division by zero possible if head_dim is 0 in attention",
        "labels": ["bug", "priority: medium", "C extension"],
        "body": """## Description

In `train.py:221` and `fastops.c:346`:

```python
scale = head_dim ** 0.5  # Python
```
```c
double scale = sqrt((double)head_dim);  // C
```

If `head_dim` is 0 (e.g., `n_embd < n_head`), the scale becomes 0, leading to division by zero on `s / scale` in the attention computation.

While current CLI defaults prevent this, there's no validation preventing a user from setting `--n-embd 2 --n-head 4` which yields `head_dim = 0`.

## Affected Files
- `train.py:221, 233`
- `train_fast.py` (via C extension)
- `fastops.c:346, 355`
""",
    },
    {
        "title": "Bug: Empty dataset causes ZeroDivisionError in training loop",
        "labels": ["bug", "priority: medium"],
        "body": """## Description

If `input.txt` contains only blank lines or whitespace, the `docs` list will be empty after filtering:

```python
docs = [l.strip() for l in open('input.txt').read().strip().split('\\n') if l.strip()]
```

This causes `docs_tokenized[step % len(docs)]` to raise `ZeroDivisionError` since `len(docs)` is 0.

Additionally, `chars = ['<BOS>'] + sorted(set(''.join(docs)))` would produce a vocabulary of just `['<BOS>']` with `vocab_size = 1`, which makes the model meaningless.

## Suggested Fix

Add a validation check after loading docs:
```python
if not docs:
    print("Error: input.txt contains no non-empty lines", file=sys.stderr)
    sys.exit(1)
```
""",
    },
    {
        "title": "Bug: Potential numerical issue with log(0) in C cross_entropy",
        "labels": ["bug", "priority: low", "C extension"],
        "body": """## Description

In `fastops.c:288`:

```c
double loss = -log(buf[target]);
```

While the softmax normalization should ensure `buf[target] > 0`, extreme floating-point underflow could produce `buf[target] = 0.0`, making `log(0)` return `-inf` or `nan`, which then corrupts all subsequent gradient computations silently.

Similarly in `fastops.c:282-283`, `exp()` could overflow for large logit differences despite the max-subtraction trick.

## Suggested Fix

Add a small epsilon or clamp:
```c
double loss = -log(fmax(buf[target], 1e-30));
```
""",
    },
    # ========================================================================
    # FEATURE: Phase 1 - Documentation & Accessibility
    # ========================================================================
    {
        "title": "Add docstrings to all major functions in train.py",
        "labels": ["enhancement", "documentation", "phase: 1"],
        "body": """## Description

As an educational project, `train.py` would benefit from docstrings on all major functions explaining:
- What the function computes
- Parameter descriptions
- Return values
- Algorithmic complexity

## Functions Needing Docstrings

- `embedding()` - Row extraction with gradient accumulation
- `linear()` - Matrix-vector multiply with hand-unrolled loops
- `rmsnorm()` - Root mean square normalization
- `tensor_add()` - Element-wise vector addition
- `squared_relu()` - Squared ReLU activation
- `attention()` - Multi-head self-attention with causal masking
- `cross_entropy()` - Softmax + negative log likelihood loss
- `mean_loss()` - Average of per-token losses
- `gpt()` - Training forward pass (builds autograd graph)
- `gpt_inference()` - Inference forward pass (plain floats)

## Acceptance Criteria

- Each function has a docstring explaining its mathematical operation
- No external dependencies added
- Educational tone maintained

## Roadmap Reference
Phase 1: Documentation & Accessibility
""",
    },
    {
        "title": "Add inline documentation to fastops.c",
        "labels": ["enhancement", "documentation", "phase: 1", "C extension"],
        "body": """## Description

The C extension (`fastops.c`, 626 lines) has minimal comments. For an educational project, each function should document:
- The operation being performed
- Memory layout assumptions
- Buffer protocol vs list fallback behavior
- Numerical stability considerations

## Key Areas Needing Documentation

1. **DArr abstraction** (lines 17-66): Explain the zero-copy vs malloc strategy
2. **Attention forward/backward** (lines 320-471): Most complex functions, need step-by-step comments
3. **Memory management patterns**: When `calloc` vs `malloc` is used and why
4. **Error handling convention**: The cascading `darr_done()` cleanup pattern

## Roadmap Reference
Phase 1: Documentation & Accessibility
""",
    },
    {
        "title": "Create examples directory with different dataset configurations",
        "labels": ["enhancement", "documentation", "phase: 1"],
        "body": """## Description

Add an `examples/` directory demonstrating:
1. Training on different text datasets (Shakespeare, code, etc.)
2. Different model size configurations with expected loss curves
3. How to prepare custom datasets
4. Performance tuning for different hardware

## Acceptance Criteria

- Zero additional dependencies
- Each example runnable as a standalone script or documented command
- Include expected output/loss ranges

## Roadmap Reference
Phase 1: Documentation & Accessibility - Examples directory
""",
    },
    # ========================================================================
    # FEATURE: Phase 2 - Core Functionality
    # ========================================================================
    {
        "title": "Add model serialization (save/load trained models)",
        "labels": ["enhancement", "phase: 2", "priority: high"],
        "body": """## Description

Currently there is no way to save a trained model and load it later for inference or continued training. The model parameters are lost when the script exits.

## Proposed Implementation

Save model as JSON (consistent with zero-dependency philosophy):
- All weight matrices (as nested lists or base64-encoded)
- Model hyperparameters (n_embd, n_layer, n_head, block_size)
- Vocabulary mapping (stoi/itos)
- Optimizer state (optional, for training resumption)
- Training metadata (step count, loss history)

## API Design

```python
# Save
python train.py --num-steps 1000 --save-model model.json

# Load for inference
python train.py --load-model model.json --inference-only

# Resume training
python train.py --load-model model.json --num-steps 500
```

## Acceptance Criteria

- JSON format, human-readable
- Round-trip save/load produces identical inference outputs
- Works with both train.py and train_fast.py
- No external dependencies

## Roadmap Reference
Phase 2: Core Functionality - Model Serialization
""",
    },
    {
        "title": "Add train/validation split with validation loss tracking",
        "labels": ["enhancement", "phase: 2", "priority: high"],
        "body": """## Description

Currently all documents are used for training with no validation split. There is no way to detect overfitting.

## Proposed Implementation

1. Split documents into train/validation sets (configurable ratio, default 90/10)
2. Periodically evaluate on validation set (e.g., every N steps)
3. Report both training and validation loss
4. Optionally implement early stopping when validation loss stops improving

## CLI Arguments

```
--val-split 0.1          # fraction of data for validation
--val-every 50           # evaluate validation loss every N steps
--early-stop-patience 5  # stop after N evaluations without improvement
```

## Acceptance Criteria

- Validation split is deterministic given the same seed
- Validation loss reported alongside training loss
- Optional early stopping
- No external dependencies

## Roadmap Reference
Phase 2: Core Functionality - Improved Dataset Handling / Training Improvements
""",
    },
    {
        "title": "Add gradient clipping to prevent training instability",
        "labels": ["enhancement", "phase: 2"],
        "body": """## Description

The training loop has no gradient clipping, which can cause training instability (exploding gradients), especially with larger models or higher learning rates.

## Proposed Implementation

Add global gradient norm clipping before the Adam update:

```python
# Compute global gradient norm
total_norm = 0.0
for p in params:
    for row in p.grad:
        total_norm += sum(g * g for g in row)
total_norm = total_norm ** 0.5

# Clip
if total_norm > max_norm:
    scale = max_norm / total_norm
    for p in params:
        for row in p.grad:
            for j in range(len(row)):
                row[j] *= scale
```

## CLI Argument

```
--grad-clip 1.0    # max gradient norm (0 = disabled)
```

## Roadmap Reference
Phase 2: Core Functionality - Training Improvements
""",
    },
    {
        "title": "Add cosine learning rate schedule option",
        "labels": ["enhancement", "phase: 2"],
        "body": """## Description

Currently only linear learning rate decay is supported:

```python
lr_t = learning_rate * (1 - step / args.num_steps)
```

Cosine annealing is a widely-used alternative that often produces better results and is standard in transformer training.

## Proposed Implementation

```python
if args.lr_schedule == 'cosine':
    lr_t = args.learning_rate * 0.5 * (1 + math.cos(math.pi * step / args.num_steps))
elif args.lr_schedule == 'linear':
    lr_t = args.learning_rate * (1 - step / args.num_steps)
```

## CLI Argument

```
--lr-schedule {linear,cosine}   # default: linear
--warmup-steps 0                # optional warmup period
```

## Roadmap Reference
Phase 2: Core Functionality - Training Improvements
""",
    },
    {
        "title": "Add standalone inference script with temperature and top-k sampling",
        "labels": ["enhancement", "phase: 2"],
        "body": """## Description

Inference is currently embedded at the end of `train.py` with hardcoded parameters (temperature=0.5, 20 samples, block_size limit). A standalone inference script would be more flexible.

## Proposed Features

- Load a saved model (depends on model serialization)
- Configurable temperature, top-k, top-p (nucleus) sampling
- Configurable number and max length of samples
- Interactive mode (generate one at a time)
- Streaming output (print characters as they're generated)

## CLI Design

```
python inference.py --model model.json --temperature 0.8 --top-k 10 --num-samples 5
python inference.py --model model.json --interactive
```

## Roadmap Reference
Phase 2: Core Functionality - Inference Mode
""",
    },
    {
        "title": "Add support for multiple text file datasets",
        "labels": ["enhancement", "phase: 2"],
        "body": """## Description

Currently only a single `input.txt` file is supported. Supporting multiple files and directories would make the project more practical.

## Proposed Implementation

```
--data input.txt                    # single file (current behavior)
--data data/*.txt                   # glob pattern
--data-dir ./datasets/              # directory of text files
```

Each file treated as a collection of documents (one per line, as currently).

## Acceptance Criteria

- Backward compatible (default behavior unchanged)
- Multiple files concatenated into single document list
- Support for glob patterns
- No external dependencies

## Roadmap Reference
Phase 2: Core Functionality - Improved Dataset Handling
""",
    },
    {
        "title": "Implement BPE tokenizer (pure Python)",
        "labels": ["enhancement", "phase: 2"],
        "body": """## Description

The current character-level tokenizer severely limits the model's ability to learn meaningful patterns from text. A BPE (Byte Pair Encoding) tokenizer would allow working with larger vocabularies and more realistic language modeling.

## Proposed Implementation

- Pure Python BPE implementation (zero dependencies)
- Train BPE vocabulary from data
- Configurable vocabulary size
- Save/load trained tokenizer
- Fallback to character-level if BPE not trained

## CLI Arguments

```
--tokenizer {char,bpe}         # default: char
--bpe-vocab-size 256           # for BPE training
--bpe-model tokenizer.json     # save/load BPE model
```

## Educational Value

BPE is the standard tokenization method used in GPT-2/3/4, making this a valuable educational addition.

## Roadmap Reference
Phase 2: Core Functionality - Enhanced Tokenization
""",
    },
    # ========================================================================
    # FEATURE: Phase 3 - Performance & Scalability
    # ========================================================================
    {
        "title": "Add SIMD vectorization (AVX2/AVX512) to C extension",
        "labels": ["enhancement", "performance", "phase: 3", "C extension"],
        "body": """## Description

The C extension (`fastops.c`) uses scalar operations for all computation. Adding SIMD intrinsics would provide significant speedup on modern CPUs.

## Key Opportunities

1. **matvec_flat** (line 499-505): Dot product is the hot path, ideal for SIMD
2. **adam_update_flat** (line 564-568): Element-wise operations perfect for vectorization
3. **attention_forward** (line 348-371): QK dot products and softmax
4. **rmsnorm_forward** (line 146): Sum of squares reduction

## Implementation Notes

- Use `#ifdef __AVX2__` / `#ifdef __AVX512F__` for compile-time detection
- Provide scalar fallback for portability
- The `roofline.py` already detects AVX2/AVX512 support
- Current build uses `-march=native`, so SIMD will be auto-enabled where available

## Expected Speedup

2-4x for AVX2, 4-8x for AVX512 on vectorizable operations.

## Roadmap Reference
Phase 3: Performance & Scalability - Advanced C Optimizations
""",
    },
    {
        "title": "Add multi-threading for matrix operations in C extension",
        "labels": ["enhancement", "performance", "phase: 3", "C extension"],
        "body": """## Description

All C extension operations are single-threaded. For larger model configurations, the matrix-vector multiplications in `matvec_flat` and `linear_backward_flat` could benefit from OpenMP or pthread parallelism.

## Proposed Implementation

- Use OpenMP `#pragma omp parallel for` for independent row computations in matvec
- Compile with `-fopenmp` flag (add to setup.py)
- Environment variable to control thread count: `OMP_NUM_THREADS`
- Minimum matrix size threshold to avoid parallelism overhead on small problems

## Key Functions to Parallelize

1. `matvec_flat`: Each output row is independent
2. `linear_backward_flat`: Weight gradient rows are independent
3. `adam_update_flat`: All elements are independent
4. `attention_forward`: Each head is independent

## Roadmap Reference
Phase 3: Performance & Scalability - Advanced C Optimizations
""",
    },
    {
        "title": "Add batch training support for larger models",
        "labels": ["enhancement", "performance", "phase: 3"],
        "body": """## Description

Currently, training processes one document per step. For larger models, batch training (processing multiple documents per step) improves gradient estimates and training efficiency.

## Proposed Implementation

```python
--batch-size 4    # number of documents per training step
```

Accumulate gradients across batch items before the optimizer step. This requires:
1. Processing multiple documents per step
2. Averaging gradients across the batch
3. C extension support for batched operations (optional optimization)

## Considerations

- Memory usage increases linearly with batch size
- Effective learning rate should account for batch size
- Maintain backward compatibility (default batch_size=1)

## Roadmap Reference
Phase 3: Performance & Scalability - Larger Model Support
""",
    },
    {
        "title": "Add memory usage estimation and profiling tools",
        "labels": ["enhancement", "performance", "phase: 3"],
        "body": """## Description

There is no way to estimate memory usage before starting training. For larger model configurations, users should be able to check if their system has enough memory.

## Proposed Features

1. **Memory estimator**: Given model config, estimate total memory usage
   ```
   python train.py --n-embd 128 --n-layer 4 --estimate-memory
   # Output: Estimated memory usage: 42.5 MB (params: 20.1 MB, grads: 20.1 MB, optimizer: 2.3 MB)
   ```

2. **Runtime memory tracking**: Optional peak memory reporting during training
   ```
   python train.py --track-memory
   # Output: step 100 | loss 2.31 | mem 45.2 MB (peak: 52.1 MB)
   ```

## Roadmap Reference
Phase 3: Performance & Scalability - Memory Efficiency
""",
    },
    # ========================================================================
    # FEATURE: Phase 4 - Advanced Features
    # ========================================================================
    {
        "title": "Add attention pattern visualization",
        "labels": ["enhancement", "phase: 4", "analysis"],
        "body": """## Description

The attention mechanism is a core concept in transformers. Visualizing attention patterns would add significant educational value.

## Proposed Features

1. **Attention weight extraction**: Save attention weights during inference
2. **Text-based visualization**: ASCII heatmap of attention patterns (zero-dependency)
3. **SVG visualization**: Leverage existing `report_charts.py` SVG generation
4. **Per-head analysis**: Show how different heads attend to different patterns

## Example Output

```
Token:  <BOS>  J  o  h  n
<BOS>   0.95  0.01  0.02  0.01  0.01
J       0.30  0.50  0.10  0.05  0.05
o       0.10  0.20  0.55  0.10  0.05
h       0.05  0.15  0.20  0.50  0.10
n       0.05  0.10  0.15  0.20  0.50
```

## Roadmap Reference
Phase 4: Advanced Features - Model Interpretability
""",
    },
    {
        "title": "Add experiment management and hyperparameter search",
        "labels": ["enhancement", "phase: 4"],
        "body": """## Description

The project has basic run archiving (`run_utils.py`, `harness.py`) but lacks structured experiment management and hyperparameter search capabilities.

## Proposed Features

1. **Grid search**: Test combinations of hyperparameters
   ```
   python experiments.py grid --n-embd 16,32,64 --n-layer 1,2,4 --learning-rate 0.01,0.001
   ```

2. **Experiment comparison**: Compare results across runs
   ```
   python experiments.py compare runs/*.json --sort-by loss_final
   ```

3. **Best config finder**: Identify optimal hyperparameters from archived runs
   ```
   python experiments.py best --metric loss_mean_last_50
   ```

## Existing Infrastructure

- `runs/` directory with JSON archives already exists
- `run_utils.py` handles archiving with git/machine metadata
- `report_charts.py` can generate SVG charts

## Roadmap Reference
Phase 4: Advanced Features - Experiment Management
""",
    },
    # ========================================================================
    # FEATURE: Phase 5 - Community & Ecosystem
    # ========================================================================
    {
        "title": "Add GitHub issue and PR templates",
        "labels": ["enhancement", "phase: 5", "community"],
        "body": """## Description

Add structured templates for GitHub issues and pull requests to maintain quality and consistency.

## Proposed Templates

### Issue Templates
1. **Bug Report**: Steps to reproduce, expected vs actual behavior, environment info
2. **Feature Request**: Description, motivation, proposed implementation
3. **Performance Issue**: Model config, timing data, roofline analysis output

### PR Template
- Summary of changes
- Motivation/linked issue
- Test plan
- Performance impact (if applicable)
- Checklist: tests pass, no new dependencies, docstrings added

## Files to Create

```
.github/ISSUE_TEMPLATE/bug_report.yml
.github/ISSUE_TEMPLATE/feature_request.yml
.github/ISSUE_TEMPLATE/performance_issue.yml
.github/PULL_REQUEST_TEMPLATE.md
```

## Roadmap Reference
Phase 5: Community & Ecosystem - Community Guidelines
""",
    },
    {
        "title": "Add cross-platform CI testing (macOS, Windows)",
        "labels": ["enhancement", "phase: 5", "CI/CD"],
        "body": """## Description

The CI pipeline currently only tests on Ubuntu. The roadmap targets cross-platform compatibility (Linux, macOS, Windows).

## Current State

- CI runs on `ubuntu-latest` only
- C extension uses Linux-specific flags (`-lmvec`)
- `roofline.py` uses `lscpu` which is Linux-only

## Proposed Changes

1. Add macOS runner to CI matrix
2. Add Windows runner with MSVC compilation
3. Make `setup.py` platform-aware for compiler flags
4. Make `roofline.py` CPU detection work on macOS/Windows
5. Test that all smoke tests pass on all platforms

## Roadmap Reference
Phase 5: Community & Ecosystem - Success Metrics (Cross-platform compatibility)
""",
    },
    {
        "title": "Add code of conduct and maintainer guidelines",
        "labels": ["enhancement", "phase: 5", "community"],
        "body": """## Description

As the project grows, formal community guidelines are needed.

## Documents to Create

1. **CODE_OF_CONDUCT.md**: Based on Contributor Covenant or similar
2. **MAINTAINER.md**: Guidelines for maintainers
   - Issue triage process
   - PR review standards
   - Release process
   - Performance regression handling

## Roadmap Reference
Phase 5: Community & Ecosystem - Community Guidelines
""",
    },
    # ========================================================================
    # IMPROVEMENT: Code Quality
    # ========================================================================
    {
        "title": "Make hardcoded constants configurable via CLI",
        "labels": ["enhancement", "code quality"],
        "body": """## Description

Several important constants are hardcoded without explanation or configurability:

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `1e-5` | train.py:184 | RMSNorm epsilon | Numerical stability |
| `0.02` | train.py:108 | Weight init std | Parameter initialization |
| `0.9, 0.95` | train.py:368 | Adam beta1, beta2 | Optimizer momentum |
| `1e-8` | train.py:368 | Adam epsilon | Optimizer numerical stability |
| `4 * n_embd` | train.py:127 | MLP hidden dim | Architecture choice |
| `0.5` | train.py:415 | Temperature | Inference sampling |
| `20` | train.py:418 | Num samples | Inference count |

## Proposed Changes

1. Add CLI arguments for commonly-tuned values (temperature, num_samples)
2. Add comments explaining the less configurable constants
3. Group constants at the top of the file for visibility
""",
    },
    {
        "title": "Improve error messages for invalid configurations",
        "labels": ["enhancement", "code quality"],
        "body": """## Description

Invalid hyperparameter configurations currently produce confusing errors deep in the training loop. Startup validation should catch common mistakes early.

## Validations to Add

```python
# At startup, after parsing args:
assert n_embd > 0, "n_embd must be positive"
assert n_layer > 0, "n_layer must be positive"
assert n_head > 0, "n_head must be positive"
assert block_size > 0, "block_size must be positive"
assert args.num_steps > 0, "num_steps must be positive"
assert args.learning_rate > 0, "learning_rate must be positive"
assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
```

## Affected Files
- `train.py`
- `train_fast.py`
""",
    },
    {
        "title": "Add type hints to train.py for educational clarity",
        "labels": ["enhancement", "code quality", "documentation"],
        "body": """## Description

Adding type hints to function signatures would improve code readability and serve as documentation for the expected data types, which is especially valuable in an educational codebase.

## Example

```python
def embedding(param: Param, idx: int) -> Tensor:
    ...

def linear(x: Tensor, w: Param) -> Tensor:
    ...

def attention(q: Tensor, keys: list[Tensor], values: list[Tensor],
              n_head: int, head_dim: int) -> Tensor:
    ...
```

## Scope

- Function signatures only (not internal variables)
- Both `train.py` and `train_fast.py`
- Type aliases for clarity where needed
""",
    },
]


def create_issue(issue: dict) -> bool:
    """Create a single GitHub issue using the gh CLI."""
    title = issue["title"]
    body = issue["body"]
    labels = issue.get("labels", [])

    # First ensure all labels exist
    for label in labels:
        subprocess.run(
            ["gh", "label", "create", label, "--repo", REPO, "--force"],
            capture_output=True,
            text=True,
        )

    cmd = [
        "gh", "issue", "create",
        "--repo", REPO,
        "--title", title,
        "--body", body,
    ]
    for label in labels:
        cmd.extend(["--label", label])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        url = result.stdout.strip()
        print(f"  Created: {url}")
        return True
    else:
        print(f"  FAILED: {result.stderr.strip()}")
        return False


def check_existing_issues() -> set:
    """Get titles of existing open issues to avoid duplicates."""
    result = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "all",
         "--limit", "200", "--json", "title"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return set()
    try:
        issues = json.loads(result.stdout)
        return {i["title"] for i in issues}
    except (json.JSONDecodeError, KeyError):
        return set()


def main():
    # Check gh is available
    result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: gh CLI not found. Install from https://cli.github.com/")
        sys.exit(1)

    print(f"Creating {len(ISSUES)} issues for {REPO}...\n")

    # Check for existing issues to avoid duplicates
    existing = check_existing_issues()
    if existing:
        print(f"Found {len(existing)} existing issues, will skip duplicates.\n")

    created = 0
    skipped = 0
    failed = 0

    for i, issue in enumerate(ISSUES, 1):
        title = issue["title"]
        labels_str = ", ".join(issue.get("labels", []))
        print(f"[{i}/{len(ISSUES)}] {title}")
        print(f"  Labels: {labels_str}")

        if title in existing:
            print("  SKIPPED: issue with this title already exists")
            skipped += 1
            continue

        if create_issue(issue):
            created += 1
        else:
            failed += 1

    print(f"\nDone! Created: {created}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
