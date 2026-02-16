# Contributing to MicroGPT

Thank you for your interest in contributing to MicroGPT! This guide will help you understand our development process and project philosophy.

## 🎯 Project Philosophy

Before contributing, please understand our core principles:

1. **Educational First**: Code clarity and teachability trump production optimizations
2. **Zero Core Dependencies**: Don't add external dependencies for core functionality
3. **Simplicity Over Features**: Every line of code has a maintenance cost
4. **Performance Conscious**: Changes should not regress performance
5. **Well Tested**: All code changes must include tests

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/atimics/microgpt.git
cd microgpt

# Build the C extension (optional but recommended)
python setup.py build_ext --inplace

# Run tests to verify setup
python test_smoke.py --quick
```

### Understanding the Codebase

Before making changes, we recommend:

1. **Read `train.py` completely** - It's the heart of the project (~460 lines)
2. **Run a training session** - See the system in action
3. **Study the tests** - `test_smoke.py` shows expected behavior
4. **Review existing PRs** - Understand our code review standards

## 📝 Types of Contributions

### 1. Documentation

**What we need:**
- Docstrings for functions lacking them
- Inline comments explaining complex algorithms
- Tutorial notebooks or guides
- Examples of different use cases
- Improved README sections

**Guidelines:**
- Use clear, simple language
- Explain *why*, not just *what*
- Include code examples
- Keep it concise

### 2. Bug Fixes

**Process:**
1. Open an issue describing the bug (if one doesn't exist)
2. Write a test that reproduces the bug
3. Fix the bug with minimal code changes
4. Ensure all tests pass
5. Submit PR with bug report reference

**Guidelines:**
- Minimal changes to fix the issue
- Include regression test
- Update documentation if behavior changes

### 3. Performance Improvements

**Process:**
1. Run benchmark baseline: `python benchmark.py update-baseline`
2. Make your changes
3. Run benchmarks: `python benchmark.py run`
4. Compare: `python benchmark.py compare baseline.json current.json`
5. Ensure no regressions (or justify trade-offs)

**Guidelines:**
- Document the performance improvement in PR
- Include roofline analysis if relevant
- Don't sacrifice readability for micro-optimizations
- C code changes need extra scrutiny

### 4. New Features

**Before starting:**
1. Open an issue to discuss the feature
2. Wait for maintainer approval
3. Check if it aligns with the roadmap

**Guidelines:**
- Keep features simple and focused
- Maintain zero-dependency principle for core features
- Add comprehensive tests
- Update documentation
- Consider educational value

## 🧪 Testing Requirements

### Required Tests

All PRs must:
- Pass existing smoke tests: `python test_smoke.py`
- Include new tests for new functionality
- Maintain or improve test coverage

### Test Types

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test component interactions
3. **Equivalence Tests**: Verify Python/C implementations match
4. **Performance Tests**: Ensure no regressions

### Running Tests

```bash
# Quick tests (recommended during development)
python test_smoke.py --quick

# Full test suite (required before submitting PR)
python test_smoke.py

# Additional test suites
python test_cross_entropy_edge_cases.py   # numerical stability tests
python test_download_error_handling.py    # download robustness tests
python test_gradient_clipping.py          # gradient clipping tests
```

## 📊 Performance Benchmarking

### Local Benchmarking

```bash
# Update baseline
python benchmark.py update-baseline

# Make your changes...

# Run new benchmark
python benchmark.py run --output current.json

# Compare
python benchmark.py compare baseline.json current.json
```

### CI Benchmarking

- Benchmarks run automatically on all PRs
- Results posted as PR comment
- Regressions > 5% are flagged
- Small variations are considered noise

### Roofline Analysis

For performance-critical changes:

```bash
# Analyze performance characteristics
python roofline.py --all-configs

# Compare Python vs C
python roofline.py --compare-c
```

## 🎨 Code Style

### Python Style

- Follow PEP 8 with some relaxations for readability
- Line length: 100-120 characters (not strict)
- Use descriptive variable names
- Prefer clarity over cleverness

**Good:**
```python
def attention(q, keys, values, n_head, head_dim):
    """
    Multi-head self-attention with scaled dot-product.
    
    Args:
        q: Query tensor (n_head * head_dim,)
        keys: List of key tensors, one per timestep
        values: List of value tensors, one per timestep
        n_head: Number of attention heads
        head_dim: Dimension of each head
    
    Returns:
        Attention output tensor (n_head * head_dim,)
    """
    # Implementation...
```

**Avoid:**
```python
def attn(q, k, v, nh, hd):  # cryptic names, no docstring
    # magic numbers and unclear logic
    return [x for x in [sum(a*b for a,b in zip(q[i:i+hd],k[i:i+hd])) 
            for i in range(0,len(q),hd)]]
```

### C Style

- Match existing style in `fastops.c`
- Use descriptive variable names
- Comment complex pointer arithmetic
- Include bounds checking
- Test thoroughly for memory safety

### Commit Messages

**Format:**
```
Short description (50 chars or less)

Detailed explanation of what changed and why.
Reference issue numbers if applicable.

- Bullet points for multiple changes
- Keep it concise but informative
```

**Good examples:**
- `Add RMSNorm backward pass optimization`
- `Fix memory leak in attention_backward`
- `Update roofline analysis to detect AVX512`

**Bad examples:**
- `fixes` (what did you fix?)
- `WIP` (don't commit work in progress to main)
- `asdf` (not descriptive)

## 🔄 Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows project style
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] No performance regressions (or justified)
- [ ] Commit messages are descriptive

### PR Template

When opening a PR, include:

```markdown
## Description
Brief description of changes

## Motivation
Why this change is needed

## Changes
- Bullet list of specific changes
- Include file names if helpful

## Testing
How you tested these changes

## Performance Impact
Benchmark results if relevant

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No performance regressions
```

### Review Process

1. **Automated checks**: CI must pass (tests + benchmarks)
2. **Code review**: Maintainer reviews for:
   - Code quality and clarity
   - Test coverage
   - Documentation completeness
   - Performance impact
   - Alignment with project goals
3. **Revisions**: Address review feedback
4. **Approval**: Maintainer approves and merges

### After Merge

- PR is merged to main branch
- CI runs final benchmarks
- Baseline updated automatically (if on main)
- Your contribution is appreciated! 🎉

## 🐛 Bug Reports

### Good Bug Reports Include

1. **Description**: What's wrong?
2. **Reproduction**: Minimal code to reproduce
3. **Expected**: What should happen?
4. **Actual**: What actually happens?
5. **Environment**: Python version, OS, CPU

### Template

```markdown
**Bug Description**
Clear description of the bug

**Reproduction**
```python
# Minimal code to reproduce
python train.py --num-steps 10
```

**Expected Behavior**
Should train without errors

**Actual Behavior**
Crashes with: [error message]

**Environment**
- Python: 3.11.5
- OS: Ubuntu 22.04
- CPU: Intel i7-12700K
```

## 💡 Feature Requests

### Before Requesting

1. Check existing issues and roadmap
2. Consider if it fits project philosophy
3. Think about zero-dependency constraint

### Good Feature Requests

```markdown
**Feature**: Add BPE tokenizer

**Motivation**: Character-level tokenization limits model quality

**Proposed Solution**: Pure Python BPE implementation

**Alternatives Considered**: 
- Using existing library (rejected: adds dependency)
- Word-level tokenizer (simpler but less flexible)

**Implementation Complexity**: Medium (200-300 lines)

**Educational Value**: High (teaches tokenization fundamentals)
```

## 🎓 Learning Path for Contributors

### Beginner-Friendly Tasks

Good first contributions:
- Adding docstrings to functions
- Improving error messages
- Adding examples to documentation
- Writing tutorials or guides
- Finding and reporting bugs

### Intermediate Tasks

Once familiar with codebase:
- Performance optimizations
- Adding configuration options
- Improving test coverage
- Refactoring for clarity

### Advanced Tasks

Deep understanding required:
- C extension modifications
- New autograd operations
- Architecture changes
- Roofline analysis improvements

## 📚 Resources

### Understanding Transformers
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Karpathy's [minGPT](https://github.com/karpathy/mingpt)

### Understanding Autograd
- Karpathy's [micrograd](https://github.com/karpathy/micrograd)
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

### Performance Optimization
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)

## ❓ Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open an Issue
- **Feature ideas**: Open an Issue for discussion first
- **Security issues**: Contact maintainers privately

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on what's best for the community
- Show empathy towards others
- Accept constructive criticism gracefully
- Prioritize education and learning

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 🙏 Recognition

All contributors will be recognized in:
- Git commit history
- GitHub contributors page
- Release notes (for significant contributions)

Thank you for helping make MicroGPT a better educational resource! 🚀
