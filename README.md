# MicroGPT

> The most atomic way to train and inference a GPT language model in pure, dependency-free Python.

**MicroGPT** is an educational implementation of a GPT-style transformer, designed to teach the fundamentals of modern language models with crystal-clear code and zero dependencies.

## ✨ Features

- 🎓 **Pure Python**: Core implementation requires no dependencies
- ⚡ **Optional C Extension**: 10-30x speedup with `fastops.c`
- 📊 **Built-in Performance Analysis**: Roofline analysis and FLOP counting
- 🔬 **Comprehensive Testing**: Automated equivalence testing and smoke tests
- 📈 **Performance Tracking**: Automated benchmarking with regression detection
- 🎨 **Educational Focus**: Clear, readable code over production optimizations

## 🚀 Quick Start

### Basic Training (Pure Python)

```bash
# Train a tiny GPT on the names dataset (downloads automatically)
python train.py --num-steps 500

# Custom configuration
python train.py --n-embd 32 --n-layer 2 --num-steps 1000
```

### Fast Training (with C extension)

```bash
# Build the C extension
python setup.py build_ext --inplace

# Train with C acceleration (10-30x faster)
python train_fast.py --num-steps 500
```

### Performance Analysis

```bash
# Roofline analysis (compute vs memory bound)
python roofline.py --all-configs

# Run benchmarks
python benchmark.py run

# Compare against baseline
python benchmark.py compare baseline.json current.json
```

## 📚 What's Inside

### Core Training Files
- **`train.py`**: Reference pure Python implementation (~460 lines)
  - Custom autograd engine (vector-level, not scalar)
  - GPT architecture with RMSNorm, square ReLU, no biases
  - Character-level tokenization
  - Adam optimizer
  
- **`train_fast.py`**: Optimized version with C extensions (~340 lines)
  - Flat memory layout for zero-copy operations
  - C-accelerated hot paths
  - Identical algorithm to `train.py` (verified by tests)

- **`fastops.c`**: C extension for performance (~630 lines)
  - Matrix operations (matvec, linear backward)
  - Fused operations (RMSNorm, attention, etc.)
  - AVX-friendly memory layout

### Analysis & Tooling
- **`roofline.py`**: Performance analysis (~770 lines)
  - Theoretical FLOP counting
  - Memory bandwidth measurement
  - Compute vs memory bound determination
  - CPU feature detection and peak estimation

- **`benchmark.py`**: Regression tracking (~475 lines)
  - Compare performance across commits
  - Automated CI integration
  - Markdown reports for PRs

- **`report_charts.py`**: Zero-dependency visualization (~800 lines)
  - Pure Python SVG chart generation
  - Performance trend graphs
  - GitHub-friendly output

- **`test_smoke.py`**: Comprehensive test suite (~250 lines)
  - C extension verification
  - Training convergence tests
  - Python/C equivalence testing
  - Roofline analysis validation

### Supporting Files
- **`harness.py`**: Training wrapper with run archiving
- **`run_utils.py`**: Shared utilities for experiment tracking
- **`setup.py`**: C extension build configuration

## 🎯 Architecture Highlights

### Model Design (GPT-2 variant)
- **Layer Normalization**: RMSNorm instead of LayerNorm
- **Activation**: Square ReLU instead of GeLU
- **Simplifications**: No biases, no weight tying
- **Attention**: Standard multi-head self-attention
- **Position Encoding**: Learned positional embeddings

### Autograd Engine
- **Vector-level**: Each node is a 1D vector (not scalar like micrograd)
- **Lazy gradients**: Allocated only during backward pass
- **Iterative topological sort**: Avoids recursion stack overflow
- **Fused operations**: Entire vector ops as single autograd nodes

### Performance Philosophy
- **Pure Python baseline**: Readable reference implementation
- **Surgical C optimization**: Only hot paths accelerated
- **Zero-copy where possible**: Buffer protocol for arrays
- **Cache-aware**: Flat memory layout for better locality

## 📊 Performance

Example timing on modern CPU (AMD/Intel with AVX2):

| Config | Params | Python | C Extension | Speedup |
|--------|--------|--------|-------------|---------|
| e16_L1_b8 | 1,360 | ~15 ms/step | ~1.2 ms/step | 12.5x |
| e64_L2_b16 | 37,120 | ~120 ms/step | ~8 ms/step | 15x |
| e128_L4_b32 | 300,544 | ~950 ms/step | ~65 ms/step | 14.6x |

Roofline efficiency: 70-85% of theoretical peak on compute-bound operations.

## 🧪 Testing

```bash
# Quick smoke tests (skip slow benchmarks)
python test_smoke.py --quick

# Full test suite including roofline measurement
python test_smoke.py

# Verify Python/C equivalence
python test_smoke.py  # includes equivalence test
```

## 🛠️ Development

### Build C Extension

```bash
python setup.py build_ext --inplace
```

### Run All Checks

```bash
# Tests
python test_smoke.py

# Benchmark
python benchmark.py run

# Roofline analysis
python roofline.py --all-configs
```

### CI/CD

The project includes comprehensive GitHub Actions CI:
- Automated testing on Python 3.11 and 3.12
- Performance benchmarking on every commit
- Regression detection with baseline comparison
- Automated baseline updates on main branch
- Performance reports on PRs with charts

## 📖 Learning Resources

### Understanding the Code
1. Start with `train.py` - read it linearly from top to bottom
2. Understand the autograd engine (`Tensor` class)
3. Study the fused operations (embedding, linear, attention, etc.)
4. Compare with `train_fast.py` to see optimization techniques
5. Examine `fastops.c` for C-level implementation

### Key Concepts Demonstrated
- **Transformer architecture**: Attention, feedforward, residual connections
- **Autograd mechanics**: Computational graph, backward pass, gradient accumulation
- **Performance optimization**: Memory layout, cache awareness, SIMD potential
- **Benchmarking methodology**: Roofline analysis, FLOP counting, bottleneck identification

## 🤝 Contributing

We welcome contributions that align with the project philosophy:

1. **Simplicity over features**: Code should be educational first
2. **Zero core dependencies**: Don't add dependencies for core functionality
3. **Performance-conscious**: Benchmark your changes
4. **Well-tested**: Include tests for new features
5. **Documented**: Add docstrings and comments

See [ROADMAP.md](ROADMAP.md) for planned features and priorities.

## 📜 License

This is an art project by [@karpathy](https://twitter.com/karpathy). Feel free to use, modify, and learn from this code.

## 🎓 Educational Philosophy

**MicroGPT** is inspired by the "micro" projects philosophy:
- **micrograd**: Scalar-level autograd engine
- **minGPT**: Minimal GPT in PyTorch
- **nanoGPT**: Production-quality minimal GPT
- **MicroGPT**: Pure Python GPT with zero dependencies

The goal is **maximum understanding with minimum code**. Every line serves a teaching purpose. Complexity is only introduced when the educational value outweighs the cognitive cost.

## 🔗 Related Projects

- [karpathy/micrograd](https://github.com/karpathy/micrograd) - Scalar autograd engine
- [karpathy/minGPT](https://github.com/karpathy/mingpt) - Minimal GPT in PyTorch
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) - Production GPT training
- [karpathy/makemore](https://github.com/karpathy/makemore) - Character-level models

## 🙏 Acknowledgments

Created as an art project by Andrej Karpathy. This implementation demonstrates that you can understand transformers deeply without complex frameworks - just Python, math, and careful thinking about compute.

---

**"The only way to learn deep learning is to implement it from scratch."** - Wisdom from the AI community

Star ⭐ this repo if you find it useful for learning!
