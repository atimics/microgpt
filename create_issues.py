#!/usr/bin/env python3
"""
Create GitHub issues for MicroGPT based on code review and roadmap analysis.

Usage:
  # Via GitHub Actions (automatic when pushed to create-issues branch)
  # Via manual workflow dispatch in GitHub UI
  # Locally with gh CLI installed:
  python create_issues.py

Requires: gh CLI authenticated, or GH_TOKEN environment variable.

Previously resolved issues (for reference):
  - Bug: No validation that n_embd is divisible by n_head (fixed: assert in train.py/train_fast.py)
  - Bug: No bounds checking on target index in C cross_entropy_forward (fixed: bounds check + epsilon)
  - Bug: No bounds checking on idx in C embedding_flat (fixed: bounds check + IndexError)
  - Bug: File handle leak when reading input.txt (fixed: context manager)
  - Bug: Dataset download has no error handling (fixed: try/except + atomic temp file)
  - Bug: Division by zero if head_dim is 0 (fixed: n_embd >= n_head validation)
  - Bug: Potential numerical issue with log(0) in C cross_entropy (fixed: fmax clamping)
  - Feature: Add docstrings to all major functions in train.py (done)
  - Feature: Add inline documentation to fastops.c (done)
  - Feature: Add gradient clipping (done: --grad-clip flag)
  - Feature: Add cross-platform CI testing (done: Ubuntu, macOS, Windows matrix)
  - Feature: Add code of conduct (done: in CONTRIBUTING.md)
  - Feature: Add type hints to train.py (done: all function signatures)
"""

import subprocess
import json
import sys

REPO = "atimics/microgpt"

# ============================================================================
# Issue definitions: remaining bugs, then features organized by roadmap phase
# ============================================================================

ISSUES = [
    # ========================================================================
    # REMAINING BUGS
    # ========================================================================
    {
        "title": "Bug: Empty dataset causes ZeroDivisionError in training loop",
        "labels": ["bug", "priority: medium"],
        "body": """## Description

If `input.txt` contains only blank lines or whitespace, the `docs` list will be empty after filtering:

```python
with open('input.txt') as f:
    docs = [l.strip() for l in f.read().strip().split('\\n') if l.strip()]
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

## Affected Files
- `train.py`
- `train_fast.py`
""",
    },
    {
        "title": "Bug: Missing positive value validation for hyperparameters",
        "labels": ["bug", "priority: medium"],
        "body": """## Description

While `n_embd % n_head` and `n_embd >= n_head` validation was added, other hyperparameters
lack basic positive value checks. For example, `--n-layer 0` or `--block-size -1` will produce
confusing errors deep in the training loop rather than a clear startup error.

## Validations to Add

```python
# At startup, after parsing args:
assert n_embd > 0, "n_embd must be positive"
assert n_layer > 0, "n_layer must be positive"
assert n_head > 0, "n_head must be positive"
assert block_size > 0, "block_size must be positive"
assert args.num_steps > 0, "num_steps must be positive"
assert args.learning_rate > 0, "learning_rate must be positive"
```

## Affected Files
- `train.py`
- `train_fast.py`
""",
    },
    {
        "title": "Bug: train_fast.py missing run archiving support",
        "labels": ["bug", "priority: low"],
        "body": """## Description

`train.py` has optional run archiving via `run_utils.archive_run()` and a `--no-archive` flag,
but `train_fast.py` does not include this functionality. Training runs with the fast path
are not automatically archived to the `runs/` directory.

## Expected Behavior

Both `train.py` and `train_fast.py` should have identical run archiving behavior.

## Suggested Fix

Add the same archive_run import and call to `train_fast.py`:
```python
try:
    from run_utils import archive_run as _archive_run
except ImportError:
    _archive_run = None

# ... at end of file:
if _archive_run and not args.no_archive:
    dest = _archive_run(run_metrics, 'training_fast')
    print(f"run archived: {dest}")
```

## Affected Files
- `train_fast.py`
""",
    },
    # ========================================================================
    # FEATURE: Phase 1 - Documentation (remaining items)
    # ========================================================================
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
    # FEATURE: Phase 2 - Core Functionality (next priority)
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
    {
        "title": "Make hardcoded constants configurable via CLI",
        "labels": ["enhancement", "phase: 2", "code quality"],
        "body": """## Description

Several important constants are hardcoded without explanation or configurability:

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `1e-5` | train.py:270 | RMSNorm epsilon | Numerical stability |
| `0.02` | train.py:130 | Weight init std | Parameter initialization |
| `0.9, 0.95` | train.py:641 | Adam beta1, beta2 | Optimizer momentum |
| `1e-8` | train.py:641 | Adam epsilon | Optimizer numerical stability |
| `4 * n_embd` | train.py:149 | MLP hidden dim | Architecture choice |
| `0.5` | train.py:702 | Temperature | Inference sampling |
| `20` | train.py:705 | Num samples | Inference count |

## Proposed Changes

1. Add CLI arguments for commonly-tuned values (temperature, num_samples)
2. Add comments explaining the less configurable constants
3. Group constants at the top of the file for visibility
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

1. **matvec_flat**: Dot product is the hot path, ideal for SIMD
2. **adam_update_flat**: Element-wise operations perfect for vectorization
3. **attention_forward**: QK dot products and softmax
4. **rmsnorm_forward**: Sum of squares reduction

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
        "title": "Add maintainer guidelines document",
        "labels": ["enhancement", "phase: 5", "community"],
        "body": """## Description

As the project grows, formal maintainer guidelines are needed to ensure consistent
quality and process across contributions.

## Document to Create

**MAINTAINER.md**: Guidelines for maintainers
- Issue triage process and response time expectations
- PR review standards and checklist
- Release process and versioning policy
- Performance regression handling procedures
- Branch management and merge strategy

## Roadmap Reference
Phase 5: Community & Ecosystem - Community Guidelines
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
