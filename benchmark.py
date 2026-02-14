#!/usr/bin/env python3
"""
Benchmark comparison tool for microgpt.

Runs roofline benchmarks and compares results against a stored baseline to
detect performance regressions and improvements across commits.

Usage:
    python benchmark.py run                                    # run benchmark
    python benchmark.py run --output current.json              # save to file
    python benchmark.py compare baseline.json current.json     # text diff
    python benchmark.py compare baseline.json current.json --md  # markdown diff
    python benchmark.py update-baseline                        # run + save as baseline
"""

import json
import sys
import subprocess
import os
import argparse
import socket
import datetime

# Changes below this threshold are reported as noise (CI runners vary)
NOISE_THRESHOLD_PCT = 5.0


def run_benchmark(output_path='current.json'):
    """Run roofline analysis on all standard configs and save results."""
    cmd = [
        sys.executable, 'roofline.py',
        '--all-configs', '--json', output_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
    )
    if result.returncode != 0:
        print(f"Benchmark failed (rc={result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)
    return output_path


def load_results(path):
    """Load benchmark results from JSON, keyed by config label."""
    with open(path) as f:
        data = json.load(f)
    keyed = {}
    for r in data.get('results', []):
        c = r['config']
        label = f"e{c['n_embd']}_L{c['n_layer']}_b{c['block_size']}"
        keyed[label] = r
    return data, keyed


def compare(baseline_path, current_path):
    """Compare two benchmark results. Returns rows and metadata."""
    base_data, base = load_results(baseline_path)
    curr_data, curr = load_results(current_path)

    all_configs = sorted(set(list(base.keys()) + list(curr.keys())))

    rows = []
    for config in all_configs:
        b = base.get(config, {})
        c = curr.get(config, {})

        b_ms = b.get('step_time_ms')
        c_ms = c.get('step_time_ms')

        if b_ms and c_ms:
            # Positive pct = faster (lower ms is better)
            pct = (b_ms - c_ms) / b_ms * 100
            rows.append({
                'config': config,
                'base_ms': b_ms,
                'curr_ms': c_ms,
                'base_gflops': b.get('achieved_gflops', 0),
                'curr_gflops': c.get('achieved_gflops', 0),
                'base_eff': b.get('efficiency_pct', 0),
                'curr_eff': c.get('efficiency_pct', 0),
                'pct_change': pct,
                'significant': abs(pct) > NOISE_THRESHOLD_PCT,
            })
        elif c_ms:
            rows.append({
                'config': config,
                'base_ms': None,
                'curr_ms': c_ms,
                'base_gflops': 0,
                'curr_gflops': c.get('achieved_gflops', 0),
                'base_eff': 0,
                'curr_eff': c.get('efficiency_pct', 0),
                'pct_change': 0,
                'significant': False,
                'new': True,
            })

    return rows, base_data, curr_data


def change_icon(pct, significant):
    """Return an icon for the change direction."""
    if not significant:
        return '~'  # within noise
    return '+' if pct > 0 else '-'


def change_emoji(pct, significant):
    """Return a markdown emoji for the change direction."""
    if not significant:
        return ':heavy_minus_sign:'  # within noise
    return ':white_check_mark:' if pct > 0 else ':warning:'


def format_text(rows, base_data, curr_data):
    """Format comparison as plain text."""
    lines = []
    lines.append("Performance Comparison")
    lines.append("=" * 70)

    # CPU info
    cpu = curr_data.get('cpu', {})
    lines.append(f"  CPU: {cpu.get('model', '?')}")
    lines.append(f"  Noise threshold: +/-{NOISE_THRESHOLD_PCT}%")
    lines.append("")

    hdr = (f"  {'Config':<20} {'Baseline':>10} {'Current':>10} "
           f"{'Change':>8} {'':>3} {'GFLOPS':>10}")
    lines.append(hdr)
    lines.append(f"  {'-' * (len(hdr) - 2)}")

    regressions = 0
    improvements = 0
    for r in rows:
        if r.get('new'):
            lines.append(f"  {r['config']:<20} {'(new)':>10} "
                         f"{r['curr_ms']:>8.1f}ms {'':>8} {'':>3} "
                         f"{r['curr_gflops']:>9.4f}")
            continue

        icon = change_icon(r['pct_change'], r['significant'])
        sign = '+' if r['pct_change'] > 0 else ''
        lines.append(
            f"  {r['config']:<20} {r['base_ms']:>8.1f}ms "
            f"{r['curr_ms']:>8.1f}ms "
            f"{sign}{r['pct_change']:>6.1f}%  {icon} "
            f"{r['curr_gflops']:>9.4f}"
        )
        if r['significant'] and r['pct_change'] > 0:
            improvements += 1
        elif r['significant'] and r['pct_change'] < 0:
            regressions += 1

    lines.append("")
    if regressions:
        lines.append(f"  *** {regressions} regression(s) detected! ***")
    if improvements:
        lines.append(f"  {improvements} improvement(s) detected.")
    if not regressions and not improvements:
        lines.append("  No significant changes (all within noise threshold).")

    return '\n'.join(lines)


def format_markdown(rows, base_data, curr_data):
    """Format comparison as GitHub-flavored markdown for PR comments."""
    lines = []
    lines.append("<!-- microgpt-benchmark -->")
    lines.append("## Performance Report")
    lines.append("")

    cpu = curr_data.get('cpu', {})
    lines.append(f"**CPU:** {cpu.get('model', '?')} | "
                 f"**Noise threshold:** +/-{NOISE_THRESHOLD_PCT}%")
    lines.append("")

    lines.append("| Config | Baseline | Current | Change | Status | GFLOPS |")
    lines.append("|--------|----------|---------|--------|--------|--------|")

    regressions = 0
    improvements = 0
    for r in rows:
        if r.get('new'):
            lines.append(
                f"| {r['config']} | _(new)_ | {r['curr_ms']:.1f}ms | - | :new: | "
                f"{r['curr_gflops']:.4f} |"
            )
            continue

        emoji = change_emoji(r['pct_change'], r['significant'])
        sign = '+' if r['pct_change'] > 0 else ''
        lines.append(
            f"| {r['config']} | {r['base_ms']:.1f}ms | {r['curr_ms']:.1f}ms | "
            f"{sign}{r['pct_change']:.1f}% | {emoji} | {r['curr_gflops']:.4f} |"
        )
        if r['significant'] and r['pct_change'] > 0:
            improvements += 1
        elif r['significant'] and r['pct_change'] < 0:
            regressions += 1

    lines.append("")

    if regressions:
        lines.append(f"> :warning: **{regressions} regression(s) detected** "
                     f"(>{NOISE_THRESHOLD_PCT}% slower)")
    if improvements:
        lines.append(f"> :white_check_mark: **{improvements} improvement(s)** "
                     f"(>{NOISE_THRESHOLD_PCT}% faster)")
    if not regressions and not improvements:
        lines.append(f"> :heavy_minus_sign: No significant changes "
                     f"(all within +/-{NOISE_THRESHOLD_PCT}%)")

    lines.append("")
    lines.append("_Generated by `benchmark.py` on CI_")

    return '\n'.join(lines)


def save_to_runs(result_path):
    """Copy benchmark result into the runs/ directory with a timestamp."""
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hostname = socket.gethostname()
    dest = os.path.join(runs_dir, f'{ts}_{hostname}_bench.json')
    with open(result_path) as f:
        data = json.load(f)
    data['_meta'] = {
        'timestamp': ts,
        'hostname': hostname,
        'type': 'benchmark',
    }
    with open(dest, 'w') as f:
        json.dump(data, f, indent=2)
    return dest


def cmd_run(args):
    """Run benchmark and save results."""
    run_benchmark(args.output)
    dest = save_to_runs(args.output)
    print(f"\nResults saved to {args.output}")
    print(f"Archived to {dest}")


def cmd_compare(args):
    """Compare two benchmark results."""
    if not os.path.exists(args.baseline):
        print(f"Baseline not found: {args.baseline}")
        print("Run `python benchmark.py update-baseline` first.")
        sys.exit(1)
    if not os.path.exists(args.current):
        print(f"Current results not found: {args.current}")
        sys.exit(1)

    rows, base_data, curr_data = compare(args.baseline, args.current)

    if args.md:
        output = format_markdown(rows, base_data, curr_data)
    else:
        output = format_text(rows, base_data, curr_data)

    print(output)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\nReport saved to {args.output}")

    # Exit with code 1 if regressions found and --fail-on-regression is set
    if args.fail_on_regression:
        regressions = sum(1 for r in rows
                          if r.get('significant') and r['pct_change'] < 0)
        if regressions:
            sys.exit(1)


def cmd_update_baseline(args):
    """Run benchmark and save as the new baseline."""
    output = args.path
    run_benchmark(output)
    print(f"\nBaseline updated: {output}")


def main():
    parser = argparse.ArgumentParser(description='microgpt benchmark tool')
    sub = parser.add_subparsers(dest='command')

    p_run = sub.add_parser('run', help='Run benchmark')
    p_run.add_argument('--output', '-o', default='current.json',
                       help='Output file (default: current.json)')

    p_cmp = sub.add_parser('compare', help='Compare two benchmark results')
    p_cmp.add_argument('baseline', help='Path to baseline JSON')
    p_cmp.add_argument('current', help='Path to current JSON')
    p_cmp.add_argument('--md', action='store_true', help='Output as markdown')
    p_cmp.add_argument('--output', '-o', default=None, help='Save report to file')
    p_cmp.add_argument('--fail-on-regression', action='store_true',
                       help='Exit 1 if regressions found')

    p_base = sub.add_parser('update-baseline', help='Run and save as baseline')
    p_base.add_argument('--path', default='baseline.json',
                        help='Baseline file (default: baseline.json)')

    args = parser.parse_args()
    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'update-baseline':
        cmd_update_baseline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
