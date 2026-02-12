#!/usr/bin/env python3
"""
Roofline analysis for microGPT training.

Computes theoretical FLOP counts, measures actual throughput, and identifies
whether each configuration is compute-bound or memory-bound. Zero dependencies.

Usage:
    python roofline.py                     # default config (n_embd=16)
    python roofline.py --n-embd 64         # wider model
    python roofline.py --all-configs       # analyze all 6 standard configs
    python roofline.py --compare-c         # compare Python vs C extension
    python roofline.py --no-measure        # skip live measurement (analytical only)
"""

import subprocess
import time
import math
import json
import os
import sys
import argparse


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------

def get_cpu_info():
    """Detect CPU model, frequency, and estimate peak FLOPS + memory bandwidth."""
    info = {
        'model': 'Unknown',
        'cores': os.cpu_count() or 1,
        'freq_ghz': 2.0,
        'has_avx2': False,
        'has_avx512': False,
        'has_fma': False,
        'l1d_kb': 32,
        'l2_kb': 512,
        'l3_kb': 32768,
    }

    try:
        out = subprocess.check_output(['lscpu'], text=True, stderr=subprocess.DEVNULL)
        for line in out.split('\n'):
            if 'Model name' in line:
                info['model'] = line.split(':', 1)[1].strip()
            if 'CPU max MHz' in line:
                info['freq_ghz'] = float(line.split(':')[1].strip()) / 1000
            elif 'CPU MHz' in line and 'CPU max' not in line:
                mhz = line.split(':')[1].strip()
                try:
                    info['freq_ghz'] = float(mhz) / 1000
                except ValueError:
                    pass
            if 'Flags' in line or 'flags' in line:
                flags = line.split(':', 1)[1].lower()
                info['has_avx2'] = 'avx2' in flags
                info['has_avx512'] = 'avx512' in flags
                info['has_fma'] = 'fma' in flags
            if 'BogoMIPS' in line:
                try:
                    bogo = float(line.split(':')[1].strip())
                    if info['freq_ghz'] == 2.0:
                        info['freq_ghz'] = bogo / 2 / 1000
                except ValueError:
                    pass
    except Exception:
        pass

    # Peak GFLOPS (single-thread, double-precision):
    #   FMA: 2 ops/instruction (mul + add)
    #   AVX2: 256-bit = 4 doubles per vector
    #   Modern AMD/Intel: 2 FMA units
    #   = 2 units * 2 ops * 4 doubles = 16 FLOP/cycle
    if info['has_fma'] and info['has_avx2']:
        flops_per_cycle = 16
    elif info['has_avx2']:
        flops_per_cycle = 8
    else:
        flops_per_cycle = 4  # SSE2

    info['flops_per_cycle'] = flops_per_cycle
    info['peak_gflops'] = info['freq_ghz'] * flops_per_cycle

    return info


# ---------------------------------------------------------------------------
# Memory bandwidth measurement (cache hierarchy characterization)
# ---------------------------------------------------------------------------

def measure_memory_bandwidth():
    """
    Measure memory bandwidth at multiple working set sizes.

    Returns:
        bw_python: GB/s for Python list traversal (pointer-chasing)
        bw_raw: GB/s for sequential copy at DRAM level
        cache_bw: list of (label, size_bytes, gb_s) tuples per cache level
    """
    # Python list traversal — this is what training actually experiences
    n = 2_000_000
    data = [float(i) for i in range(n)]
    _ = sum(data)
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
        s = sum(data)
    elapsed = time.perf_counter() - t0
    # Each element: 8-byte pointer + ~16 bytes from PyFloat internals
    bytes_read = n * 16 * iters
    bw_python = bytes_read / elapsed / 1e9

    # Sequential copy at different working set sizes to characterize cache hierarchy
    cache_bw = []
    for label, size_bytes in [
        ('L1  (16 KB)',   16 * 1024),
        ('L2  (256 KB)', 256 * 1024),
        ('L3  (8 MB)',     8 * 1024 * 1024),
        ('DRAM (64 MB)',  64 * 1024 * 1024),
    ]:
        buf = bytearray(size_bytes)
        _ = bytes(buf)  # warm up
        target_bytes = 500_000_000  # aim for ~0.5s of work
        iters_c = max(3, target_bytes // max(size_bytes, 1))
        t0 = time.perf_counter()
        for _ in range(iters_c):
            _ = bytes(buf)
        elapsed_c = time.perf_counter() - t0
        bw = size_bytes * iters_c / elapsed_c / 1e9
        cache_bw.append((label, size_bytes, bw))

    bw_raw = cache_bw[-1][2]  # DRAM bandwidth for roofline calculations

    return bw_python, bw_raw, cache_bw


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

def count_flops(n_embd, n_layer, n_head, block_size, vocab_size, seq_len=None):
    """
    Count FLOPs for one complete training step (forward + backward + Adam).

    Conventions:
    - 1 multiply = 1 FLOP, 1 add = 1 FLOP (so a += b*c = 2 FLOPs)
    - Dot product of length n = 2n FLOPs
    - Matrix-vector (m x n) @ (n,) = 2mn FLOPs
    """
    head_dim = n_embd // n_head
    T = seq_len or block_size

    # === FORWARD PASS (per position, then summed) ===

    # Embedding: just copy, 0 FLOPs
    # tensor_add (tok + pos): n_embd adds
    # rmsnorm: n_embd (square) + n_embd (sum) + ~3 (mean, eps, rsqrt) + n_embd (scale) ~ 3*n_embd

    fwd_init_per_pos = n_embd + 3 * n_embd  # add + rmsnorm

    # Per-layer per-position (non-attention):
    #   rmsnorm(pre-attn):    3 * n_embd
    #   linear Q:             2 * n_embd^2
    #   linear K:             2 * n_embd^2
    #   linear V:             2 * n_embd^2
    #   linear Wo:            2 * n_embd^2
    #   tensor_add(residual): n_embd
    #   rmsnorm(pre-mlp):     3 * n_embd
    #   linear fc1:           2 * 4*n_embd * n_embd = 8 * n_embd^2
    #   squared_relu:         2 * 4*n_embd (compare + square)
    #   linear fc2:           2 * n_embd * 4*n_embd = 8 * n_embd^2
    #   tensor_add(residual): n_embd
    fwd_linear_per_layer = 24 * n_embd * n_embd
    fwd_other_per_layer = 3 * n_embd + n_embd + 3 * n_embd + 8 * n_embd + n_embd  # rmsnorms + adds + relu
    fwd_fixed_per_layer = fwd_linear_per_layer + fwd_other_per_layer

    # Attention (position-dependent — at position t, sees t+1 keys):
    #   Per head: dot(q, k_t) for each t -> (t+1) * 2*head_dim
    #   Softmax: (t+1) * 3 (exp, sum, div)
    #   Weighted sum of values: (t+1) * 2*head_dim
    #   Total per head: (t+1) * (4*head_dim + 3)
    #   All heads: (t+1) * (4*n_embd + 3*n_head)
    # Summed over t=0..T-1: (4*n_embd + 3*n_head) * T*(T+1)/2
    fwd_attn_total = n_layer * (4 * n_embd + 3 * n_head) * T * (T + 1) // 2

    # lm_head: 2 * vocab_size * n_embd
    # cross_entropy: ~3 * vocab_size (exp + sum + div)
    fwd_head_per_pos = 2 * vocab_size * n_embd + 3 * vocab_size

    fwd_per_pos = fwd_init_per_pos + n_layer * fwd_fixed_per_layer + fwd_head_per_pos
    fwd_total = T * fwd_per_pos + fwd_attn_total

    # Linear FLOPs (forward only, for breakdown)
    fwd_linear_total = T * (n_layer * fwd_linear_per_layer + 2 * vocab_size * n_embd)

    # === BACKWARD PASS ===
    # Each linear backward: weight_grad (nout*nin muls + adds) + input_grad (same) = 2x forward
    bwd_linear_total = 2 * fwd_linear_total
    bwd_attn_total = fwd_attn_total  # similar complexity
    bwd_other = T * (fwd_init_per_pos + n_layer * fwd_other_per_layer + 3 * vocab_size)
    bwd_total = bwd_linear_total + bwd_attn_total + bwd_other

    # === ADAM OPTIMIZER ===
    # Per parameter:
    #   m = beta1*m + (1-beta1)*g            -> 3 FLOPs (mul, mul, add)
    #   v = beta2*v + (1-beta2)*g*g           -> 4 FLOPs (mul, mul, mul, add)
    #   w -= lr * (m/bc1) / (sqrt(v/bc2)+eps) -> 7 FLOPs (div, div, sqrt, add, div, mul, sub)
    # Total: 14 FLOPs/param
    num_params = (vocab_size * n_embd +
                  block_size * n_embd +
                  n_layer * 12 * n_embd * n_embd +
                  vocab_size * n_embd)
    adam_total = num_params * 14

    total = fwd_total + bwd_total + adam_total

    return {
        'fwd_linear': fwd_linear_total,
        'fwd_attention': fwd_attn_total,
        'fwd_other': fwd_total - fwd_linear_total - fwd_attn_total,
        'bwd_linear': bwd_linear_total,
        'bwd_attention': bwd_attn_total,
        'bwd_other': bwd_other,
        'adam': adam_total,
        'forward': fwd_total,
        'backward': bwd_total,
        'total': total,
        'num_params': num_params,
    }


# ---------------------------------------------------------------------------
# Memory traffic estimation
# ---------------------------------------------------------------------------

def count_bytes(n_embd, n_layer, n_head, block_size, vocab_size, seq_len=None, mode='python'):
    """
    Estimate bytes moved per training step.

    Modes:
      'python'    — traversing Python list[float]; each number costs ~36 bytes
                    (8-byte pointer + 28-byte PyFloat, scattered on heap)
      'c_ext'     — fastops extracts to contiguous double* (8 bytes/element)
                    for inner loops, but still returns Python objects
      'float32'   — hypothetical contiguous float32 arrays (4 bytes)
    """
    T = seq_len or block_size
    bpf = {'python': 36, 'c_ext': 8, 'float32': 4}[mode]

    num_params = (vocab_size * n_embd +
                  block_size * n_embd +
                  n_layer * 12 * n_embd * n_embd +
                  vocab_size * n_embd)

    # Linear forward: read W (nout*nin) + read x (nin) + write out (nout)
    # Per layer: Q,K,V,O: 4 * (n_embd^2 + 2*n_embd)
    #            fc1: 4*n_embd*n_embd + 5*n_embd
    #            fc2: 4*n_embd*n_embd + 5*n_embd
    fwd_linear_bytes_per_layer = bpf * (
        4 * (n_embd * n_embd + 2 * n_embd) +
        (4 * n_embd * n_embd + 5 * n_embd) +
        (4 * n_embd * n_embd + 5 * n_embd)
    )
    fwd_head_bytes = bpf * (vocab_size * n_embd + n_embd + vocab_size)
    fwd_bytes_per_pos = n_layer * fwd_linear_bytes_per_layer + fwd_head_bytes

    # Backward: ~2x forward (read W, og, x; write wgrad, xgrad)
    bwd_bytes_per_pos = 2 * fwd_bytes_per_pos

    # Adam: read+write data, grad, m, v = 4 arrays * 2 directions
    adam_bytes = num_params * bpf * 8

    total = T * (fwd_bytes_per_pos + bwd_bytes_per_pos) + adam_bytes

    return {
        'forward': T * fwd_bytes_per_pos,
        'backward': T * bwd_bytes_per_pos,
        'adam': adam_bytes,
        'total': total,
        'bytes_per_float': bpf,
        'num_params': num_params,
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_step_time(n_embd, n_layer, n_head, block_size, num_steps=20,
                      pure_python=False):
    """Run train.py for a few steps and return time per step."""
    env = os.environ.copy()
    if pure_python:
        env['MICROGPT_PURE_PYTHON'] = '1'
    cmd = [
        sys.executable, 'train.py',
        '--n-embd', str(n_embd),
        '--n-layer', str(n_layer),
        '--n-head', str(n_head),
        '--block-size', str(block_size),
        '--num-steps', str(num_steps),
        '--learning-rate', '0.01',
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
            env=env,
        )
        for line in result.stdout.split('\n'):
            if 'training time:' in line:
                t = float(line.split(':')[1].strip().rstrip('s'))
                return t / num_steps
    except Exception as e:
        print(f"  [!] Measurement failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_flops(flops):
    if flops >= 1e9:  return f"{flops/1e9:.2f} GFLOP"
    if flops >= 1e6:  return f"{flops/1e6:.2f} MFLOP"
    if flops >= 1e3:  return f"{flops/1e3:.1f} KFLOP"
    return f"{flops:.0f} FLOP"


def fmt_bytes(b):
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.2f} MB"
    if b >= 1e3:  return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


# ---------------------------------------------------------------------------
# ASCII roofline chart (with collision handling and data table)
# ---------------------------------------------------------------------------

def draw_roofline(peak_gflops, mem_bw_gbs, points, width=72, height=20):
    """ASCII roofline diagram with collision-aware point placement."""
    ridge = peak_gflops / mem_bw_gbs

    all_oi  = [p['oi'] for p in points if p.get('oi', 0) > 0]
    all_gf  = [p['gflops'] for p in points if p.get('gflops', 0) > 0]
    if not all_oi or not all_gf:
        print("  (no measured data for roofline chart)")
        return

    # log-scale bounds — widen to separate clustered points
    x_lo = math.floor(math.log10(min(min(all_oi) * 0.3, 0.005)) * 2) / 2
    x_hi = math.ceil(math.log10(max(max(all_oi) * 3, ridge * 4)) * 2) / 2
    y_lo = math.floor(math.log10(min(all_gf) * 0.3) * 2) / 2
    y_hi = math.ceil(math.log10(peak_gflops * 2) * 2) / 2

    def to_col(oi):
        if oi <= 0: return -1
        return round((math.log10(oi) - x_lo) / (x_hi - x_lo) * (width - 1))

    def to_row(gf):
        if gf <= 0: return -1
        r = round((math.log10(gf) - y_lo) / (y_hi - y_lo) * (height - 1))
        return height - 1 - r  # flip so top = high

    grid = [[' '] * width for _ in range(height)]

    # Draw the roofline envelope: perf = min(peak, bw * oi)
    for c in range(width):
        oi = 10 ** (x_lo + c / (width - 1) * (x_hi - x_lo))
        gf = min(peak_gflops, mem_bw_gbs * oi)
        r = to_row(gf)
        if 0 <= r < height:
            grid[r][c] = '/' if oi < ridge else '-'

    # Plot operating points with collision handling
    markers = 'ABCDEFGHIJKLmnop'
    occupied = {}  # (row, col) -> True
    legend = []

    for i, p in enumerate(points):
        oi, gf = p.get('oi', 0), p.get('gflops', 0)
        if oi <= 0 or gf <= 0:
            continue
        c, r = to_col(oi), to_row(gf)
        mk = markers[i % len(markers)]

        # Try original position, then nearby cells to avoid overlap
        placed = False
        for dc, dr in [(0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0),
                        (0, -1), (0, 1), (1, -1), (-1, -1), (3, 0)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < width and 0 <= nr < height and (nr, nc) not in occupied:
                grid[nr][nc] = mk
                occupied[(nr, nc)] = True
                placed = True
                break
        if not placed and 0 <= c < width and 0 <= r < height:
            grid[r][c] = mk  # overwrite as last resort

        legend.append(f"  {mk} = {p['label']}")

    # Print with Y-axis labels
    for r in range(height):
        gf = 10 ** (y_hi - r / (height - 1) * (y_hi - y_lo))
        if r % 4 == 0:
            label = f"{gf:.3f}" if gf < 1 else f"{gf:.1f}"
            print(f"  {label:>8} |{''.join(grid[r])}")
        else:
            print(f"          |{''.join(grid[r])}")

    # X-axis
    print(f"          +{'-' * width}")
    lo_label = f"{10**x_lo:.3f}"
    hi_label = f"{10**x_hi:.1f}"
    mid = width - len(lo_label) - len(hi_label)
    print(f"           {lo_label}{' ' * max(mid, 1)}{hi_label}")
    print(f"           Operational Intensity (FLOP/byte, log scale)")
    print(f"           Y-axis: Achieved GFLOPS (log scale)")
    print()
    for l in legend:
        print(l)

    # Data table with exact values (always readable, even when chart points overlap)
    print()
    print(f"  {'Pt':<4} {'Label':<22} {'OI (FLOP/B)':>12} {'GFLOPS':>10} {'% Peak':>8}")
    print(f"  {'-'*58}")
    for i, p in enumerate(points):
        mk = markers[i % len(markers)]
        pct = p['gflops'] / peak_gflops * 100 if peak_gflops > 0 else 0
        print(f"  {mk:<4} {p['label']:<22} {p['oi']:>12.4f} {p['gflops']:>10.4f} {pct:>7.3f}%")


# ---------------------------------------------------------------------------
# Per-config analysis
# ---------------------------------------------------------------------------

def analyze_config(n_embd, n_layer, n_head, block_size, vocab_size,
                   cpu_info, bw_python, bw_raw, do_measure=True,
                   pure_python=False):
    """Full roofline analysis for one configuration."""
    T = block_size
    flops = count_flops(n_embd, n_layer, n_head, block_size, vocab_size, T)

    bytes_py  = count_bytes(n_embd, n_layer, n_head, block_size, vocab_size, T, 'python')
    bytes_c   = count_bytes(n_embd, n_layer, n_head, block_size, vocab_size, T, 'c_ext')
    bytes_f32 = count_bytes(n_embd, n_layer, n_head, block_size, vocab_size, T, 'float32')

    total_flops = flops['total']
    oi_py  = total_flops / bytes_py['total']
    oi_c   = total_flops / bytes_c['total']
    oi_f32 = total_flops / bytes_f32['total']

    peak = cpu_info['peak_gflops']
    ridge = peak / bw_raw if bw_raw > 0 else 2.0

    mode_label = " (pure Python)" if pure_python else ""
    print(f"\n{'='*72}")
    print(f"  n_embd={n_embd}  n_layer={n_layer}  n_head={n_head}  "
          f"block_size={block_size}  params={flops['num_params']:,}{mode_label}")
    print(f"{'='*72}")

    # FLOP breakdown
    print(f"\n  FLOP breakdown (per step, seq_len={T}):")
    print(f"  {'Operation':<25} {'FLOPs':>14} {'%':>6}")
    print(f"  {'-'*47}")
    for key, label in [
        ('fwd_linear',    'Forward: linear'),
        ('fwd_attention',  'Forward: attention'),
        ('fwd_other',     'Forward: other'),
        ('bwd_linear',    'Backward: linear'),
        ('bwd_attention', 'Backward: attention'),
        ('bwd_other',     'Backward: other'),
        ('adam',          'Adam optimizer'),
    ]:
        v = flops[key]
        print(f"  {label:<25} {fmt_flops(v):>14} {100*v/total_flops:>5.1f}%")
    print(f"  {'-'*47}")
    print(f"  {'TOTAL':<25} {fmt_flops(total_flops):>14}")

    # Memory traffic
    print(f"\n  Memory traffic (per step):")
    print(f"  {'':.<20} {'Python (36B)':>14} {'C ext (8B)':>14} {'float32 (4B)':>14}")
    for phase in ['forward', 'backward', 'adam']:
        print(f"  {phase.title():<20} "
              f"{fmt_bytes(bytes_py[phase]):>14} "
              f"{fmt_bytes(bytes_c[phase]):>14} "
              f"{fmt_bytes(bytes_f32[phase]):>14}")
    print(f"  {'TOTAL':<20} "
          f"{fmt_bytes(bytes_py['total']):>14} "
          f"{fmt_bytes(bytes_c['total']):>14} "
          f"{fmt_bytes(bytes_f32['total']):>14}")

    # Operational intensity
    print(f"\n  Operational intensity (FLOP / byte):")
    print(f"    Python lists:       {oi_py:.4f}")
    print(f"    C extension:        {oi_c:.4f}")
    print(f"    Contiguous float32: {oi_f32:.4f}")
    print(f"    CPU ridge point:    {ridge:.2f}  (peak_compute / mem_bandwidth)")
    print(f"    --> All modes are MEMORY-BOUND (OI << ridge)" if oi_f32 < ridge
          else f"    --> float32 mode approaches compute-bound territory")

    # Measurement
    step_time = None
    if do_measure:
        label = "pure Python" if pure_python else "with C ext"
        print(f"\n  Measuring ({label}, 20 steps)...", end='', flush=True)
        step_time = measure_step_time(n_embd, n_layer, n_head, block_size, 20,
                                      pure_python=pure_python)
        if step_time:
            print(f" {step_time*1000:.1f} ms/step")
        else:
            print(f" failed")

    result = {
        'config': {
            'n_embd': n_embd, 'n_layer': n_layer,
            'n_head': n_head, 'block_size': block_size,
        },
        'num_params': flops['num_params'],
        'total_flops': total_flops,
        'flop_breakdown': {k: flops[k] for k in flops if k != 'total'},
        'bytes_python': bytes_py['total'],
        'bytes_c': bytes_c['total'],
        'bytes_f32': bytes_f32['total'],
        'oi_python': oi_py,
        'oi_c': oi_c,
        'oi_f32': oi_f32,
        'ridge_point': ridge,
        'pure_python': pure_python,
    }

    if step_time:
        achieved = total_flops / step_time / 1e9
        eff = achieved / peak * 100
        bw_limited = oi_py * bw_python  # max GFLOPS if fully memory-bound

        print(f"\n  Performance:")
        print(f"    Time/step:          {step_time*1000:.1f} ms")
        print(f"    Achieved throughput: {achieved:.4f} GFLOPS")
        print(f"    CPU peak (1 thread):{peak:.1f} GFLOPS")
        print(f"    Compute efficiency: {eff:.3f}%")
        print(f"    BW-limited ceiling: {bw_limited:.4f} GFLOPS  "
              f"(OI={oi_py:.4f} * BW={bw_python:.1f} GB/s)")

        headroom = bw_limited / achieved if achieved > 0 else 0
        print(f"    Headroom to BW ceiling: {headroom:.1f}x")

        if oi_py < ridge:
            print(f"\n    VERDICT: MEMORY-BOUND")
            print(f"    Python's 36 bytes/float (vs 4 for float32) wastes {36/4:.0f}x bandwidth.")
            print(f"    Moving to contiguous arrays would reduce traffic by ~{bytes_py['total']/bytes_f32['total']:.0f}x,")
            print(f"    raising OI from {oi_py:.4f} to {oi_f32:.4f}.")
        else:
            print(f"\n    VERDICT: COMPUTE-BOUND (rare for this model size)")

        result['step_time_ms'] = step_time * 1000
        result['achieved_gflops'] = achieved
        result['efficiency_pct'] = eff
        result['bottleneck'] = 'memory' if oi_py < ridge else 'compute'

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='microGPT roofline analysis')
    parser.add_argument('--n-embd',     type=int, default=16)
    parser.add_argument('--n-layer',    type=int, default=1)
    parser.add_argument('--n-head',     type=int, default=4)
    parser.add_argument('--block-size', type=int, default=8)
    parser.add_argument('--vocab-size', type=int, default=27)
    parser.add_argument('--all-configs', action='store_true',
                        help='Analyze all 6 standard benchmark configs')
    parser.add_argument('--compare-c', action='store_true',
                        help='Compare pure Python vs C extension performance')
    parser.add_argument('--no-measure', action='store_true',
                        help='Analytical only (skip running train.py)')
    parser.add_argument('--json', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    print("=" * 72)
    print("  microGPT Roofline Analysis")
    print("=" * 72)

    # --- CPU info ---
    cpu = get_cpu_info()
    print(f"\n  CPU:       {cpu['model']}")
    print(f"  Cores:     {cpu['cores']}  (analysis is single-threaded)")
    print(f"  Freq:      {cpu['freq_ghz']:.2f} GHz")
    print(f"  ISA:       {'AVX2' if cpu['has_avx2'] else 'SSE2'}"
          f"{' + FMA' if cpu['has_fma'] else ''}"
          f"{' + AVX-512' if cpu['has_avx512'] else ''}")
    print(f"  Peak FLOPS:{cpu['peak_gflops']:.1f} GFLOPS/thread  "
          f"({cpu['flops_per_cycle']} FLOP/cycle * {cpu['freq_ghz']:.2f} GHz)")

    # --- Memory bandwidth (cache hierarchy) ---
    print(f"\n  Measuring memory bandwidth...", end='', flush=True)
    bw_python, bw_raw, cache_bw = measure_memory_bandwidth()
    print(f" done")

    print(f"\n  Cache Hierarchy Bandwidth:")
    print(f"  {'Level':<16} {'Working Set':>12} {'Bandwidth':>12}")
    print(f"  {'-'*42}")
    for label, size_bytes, bw in cache_bw:
        size_str = fmt_bytes(size_bytes)
        print(f"  {label:<16} {size_str:>12} {bw:>9.1f} GB/s")
    print(f"  {'Python lists':<16} {'scattered':>12} {bw_python:>9.1f} GB/s")

    print(f"\n  Ridge point:  {cpu['peak_gflops']/bw_raw:.2f} FLOP/byte  "
          f"(peak_compute / DRAM_bandwidth)")

    # --- C extension ---
    has_c = False
    try:
        import fastops
        has_c = True
        print(f"\n  C extension (fastops): available")
    except ImportError:
        print(f"\n  C extension (fastops): not found")

    # --- Configs ---
    if args.all_configs:
        configs = [
            (16, 1, 4, 8),
            (32, 1, 4, 8),
            (64, 1, 4, 8),
            (16, 1, 4, 16),
            (32, 2, 4, 8),
            (64, 2, 4, 16),
        ]
    else:
        configs = [(args.n_embd, args.n_layer, args.n_head, args.block_size)]

    all_results = []
    points = []

    for n_embd, n_layer, n_head, block_size in configs:
        r = analyze_config(
            n_embd, n_layer, n_head, block_size, args.vocab_size,
            cpu, bw_python, bw_raw, do_measure=not args.no_measure,
        )
        all_results.append(r)
        if 'achieved_gflops' in r:
            c = r['config']
            points.append({
                'label': f"e{c['n_embd']}_L{c['n_layer']}_b{c['block_size']}",
                'oi': r['oi_python'],
                'gflops': r['achieved_gflops'],
            })

    # --- Roofline chart ---
    if points:
        print(f"\n{'='*72}")
        print(f"  Roofline Chart (all configs)")
        print(f"{'='*72}\n")
        draw_roofline(cpu['peak_gflops'], bw_raw, points)

    # --- Summary table ---
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print(f"  Summary Table")
        print(f"{'='*72}")
        hdr = (f"  {'Config':<20} {'Params':>8} {'FLOP/step':>12} "
               f"{'OI':>8} {'ms/step':>8} {'GFLOPS':>8} {'Eff%':>7} {'Bound':>7}")
        print(hdr)
        print(f"  {'-'*(len(hdr)-2)}")
        for r in all_results:
            c = r['config']
            label = f"e{c['n_embd']}_L{c['n_layer']}_b{c['block_size']}"
            ms  = f"{r['step_time_ms']:.1f}" if 'step_time_ms' in r else '-'
            gf  = f"{r['achieved_gflops']:.4f}" if 'achieved_gflops' in r else '-'
            eff = f"{r['efficiency_pct']:.3f}" if 'efficiency_pct' in r else '-'
            bnd = r.get('bottleneck', '-')
            print(f"  {label:<20} {r['num_params']:>8,} {fmt_flops(r['total_flops']):>12} "
                  f"{r['oi_python']:>7.4f} {ms:>8} {gf:>8} {eff:>6}% {bnd:>7}")

    # --- Python vs C extension comparison ---
    if args.compare_c and not args.no_measure:
        print(f"\n{'='*72}")
        print(f"  Python vs C Extension Comparison")
        print(f"{'='*72}")
        if not has_c:
            print("\n  C extension (fastops) not available — nothing to compare.")
            print("  Build it with: python setup.py build_ext --inplace")
        else:
            compare_results = []
            for n_embd, n_layer, n_head, block_size in configs:
                c_label = f"e{n_embd}_L{n_layer}_b{block_size}"
                print(f"\n  Measuring {c_label} (pure Python)...", end='', flush=True)
                t_py = measure_step_time(n_embd, n_layer, n_head, block_size, 20,
                                         pure_python=True)
                if t_py:
                    print(f" {t_py*1000:.1f} ms/step")
                else:
                    print(f" failed")
                compare_results.append((c_label, t_py))

            print(f"\n  {'Config':<20} {'Pure Python':>14} {'With C ext':>14} {'Speedup':>10}")
            print(f"  {'-'*60}")
            for i, (label, t_py) in enumerate(compare_results):
                ms_c = all_results[i].get('step_time_ms')
                ms_py = t_py * 1000 if t_py else None
                ms_c_str  = f"{ms_c:.1f} ms" if ms_c else "-"
                ms_py_str = f"{ms_py:.1f} ms" if ms_py else "-"
                if ms_c and ms_py:
                    speedup = f"{ms_py / ms_c:.2f}x"
                else:
                    speedup = "-"
                print(f"  {label:<20} {ms_py_str:>14} {ms_c_str:>14} {speedup:>10}")

    # --- Key insights ---
    print(f"\n{'='*72}")
    print(f"  Key Insights")
    print(f"{'='*72}")
    ridge = cpu['peak_gflops'] / bw_raw
    print(f"""
  1. DATA REPRESENTATION IS THE BOTTLENECK
     Python list[float] = 36 bytes per number (28-byte PyFloat + 8-byte pointer).
     Contiguous float32 = 4 bytes per number. That's a 9x memory bloat.

  2. EVERYTHING IS MEMORY-BOUND
     Operational intensity with Python objects: ~0.05 FLOP/byte
     CPU ridge point:                          ~{ridge:.1f} FLOP/byte
     We're {ridge/0.055:.0f}x below the ridge — arithmetic is essentially free;
     we're paying for pointer-chasing through scattered heap objects.

  3. PATH TO COMPUTE-BOUND
     To reach the ridge point ({ridge:.1f} FLOP/byte), we need:
     a) Contiguous memory (numpy ndarray, ctypes buffer, or C arrays)
        -> Raises OI from ~0.05 to ~0.5 FLOP/byte (still memory-bound)
     b) Cache tiling for matmul (keep blocks in L1/L2)
        -> Can push effective OI above ridge for large enough matrices
     c) SIMD vectorization (compiler auto-vec or intrinsics)
        -> Approaches peak FLOPS within compute-bound regime

  4. PRACTICAL NEXT STEPS (ordered by impact)
     - Replace list[float] with flat ctypes double array: ~5-9x less traffic
     - Tile matrix operations for L1 cache (32 KB): reduce memory stalls
     - Fuse operations (e.g., rmsnorm+linear): reduce intermediate traffic
     - Use float32 instead of float64: 2x less traffic, 2x more SIMD lanes
""")

    # --- Save JSON ---
    if args.json:
        out = {
            'cpu': {k: v for k, v in cpu.items()},
            'bandwidth': {
                'python_gbs': bw_python,
                'raw_gbs': bw_raw,
                'cache_hierarchy': [
                    {'level': label, 'size_bytes': sz, 'bandwidth_gbs': bw}
                    for label, sz, bw in cache_bw
                ],
            },
            'results': all_results,
        }
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  Results saved to {args.json}")


if __name__ == '__main__':
    main()
