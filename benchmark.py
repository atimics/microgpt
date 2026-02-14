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

try:
    from run_utils import archive_run as _archive_run, load_runs
except ImportError:
    _archive_run = None
    load_runs = None

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
    """Archive benchmark result into runs/ with git/machine metadata."""
    if not _archive_run:
        return None
    with open(result_path) as f:
        data = json.load(f)
    return _archive_run(data, 'benchmark')


def cmd_run(args):
    """Run benchmark and save results."""
    run_benchmark(args.output)
    dest = save_to_runs(args.output)
    print(f"\nResults saved to {args.output}")
    if dest:
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


def cmd_trend(args):
    """Show performance trend across all archived benchmark runs."""
    if not load_runs:
        print("run_utils not available. Cannot load runs.")
        sys.exit(1)

    bench_runs = load_runs(run_type='benchmark')
    if not bench_runs:
        print("No benchmark runs found in runs/.")
        sys.exit(1)

    # Collect per-config time series
    configs_seen = {}
    entries = []
    for fname, data in bench_runs:
        meta = data.get('_meta', {})
        git = meta.get('git', {})
        ts = meta.get('timestamp', fname[:15])
        commit = git.get('commit_short', '?')

        row = {'file': fname, 'ts': ts[:19], 'commit': commit, 'configs': {}}
        for r in data.get('results', []):
            c = r.get('config', {})
            label = f"e{c.get('n_embd','?')}_L{c.get('n_layer','?')}_b{c.get('block_size','?')}"
            ms = r.get('step_time_ms')
            gf = r.get('achieved_gflops', 0)
            if ms is not None:
                row['configs'][label] = {'ms': ms, 'gflops': gf}
                configs_seen[label] = True
        entries.append(row)

    all_configs = sorted(configs_seen.keys())

    # Limit to last N runs
    n = args.last if hasattr(args, 'last') else 10
    entries = entries[-n:]

    # Print trend table
    print(f"\nBenchmark Trend (last {len(entries)} runs)")
    print("=" * 70)

    # Header
    cfg_width = 10
    hdr = f"  {'Date':<12} {'Commit':<10}"
    for c in all_configs:
        hdr += f" {c:>{cfg_width}}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")

    for e in entries:
        date_str = e['ts'][:10]
        line = f"  {date_str:<12} {e['commit']:<10}"
        for c in all_configs:
            if c in e['configs']:
                ms = e['configs'][c]['ms']
                line += f" {ms:>{cfg_width}.1f}"
            else:
                line += f" {'-':>{cfg_width}}"
        print(line)

    # Show improvement from first to last
    if len(entries) >= 2:
        print(f"\n  Speedup (first → last):")
        first, last = entries[0], entries[-1]
        for c in all_configs:
            if c in first['configs'] and c in last['configs']:
                f_ms = first['configs'][c]['ms']
                l_ms = last['configs'][c]['ms']
                speedup = f_ms / l_ms if l_ms > 0 else 0
                print(f"    {c}: {f_ms:.1f}ms → {l_ms:.1f}ms ({speedup:.1f}x)")


def cmd_latest(args):
    """Compare the most recent benchmark run against baseline."""
    baseline = args.baseline
    if not os.path.exists(baseline):
        print(f"Baseline not found: {baseline}")
        sys.exit(1)

    # Find latest benchmark run
    if not load_runs:
        print("run_utils not available.")
        sys.exit(1)

    bench_runs = load_runs(run_type='benchmark')
    if not bench_runs:
        print("No benchmark runs in runs/.")
        sys.exit(1)

    latest_fname, latest_data = bench_runs[-1]
    latest_path = os.path.join('runs', latest_fname)

    meta = latest_data.get('_meta', {})
    git = meta.get('git', {})
    print(f"Latest run: {latest_fname}")
    print(f"  Commit: {git.get('commit_short', '?')} ({git.get('branch', '?')})")
    print(f"  Time:   {meta.get('timestamp', '?')[:19]}")
    print()

    rows, base_data, curr_data = compare(baseline, latest_path)
    print(format_text(rows, base_data, curr_data))


def cmd_runs(args):
    """List all archived runs with summary info."""
    if not load_runs:
        print("run_utils not available.")
        sys.exit(1)

    run_type = args.type if hasattr(args, 'type') and args.type else None
    runs = load_runs(run_type=run_type)
    if not runs:
        print("No runs found in runs/.")
        return

    print(f"\n{'Type':<12} {'Date':<12} {'Commit':<10} {'Machine':<14} {'Details'}")
    print("-" * 75)

    for fname, data in runs:
        meta = data.get('_meta', {})
        git = meta.get('git', {})
        machine = meta.get('machine', data.get('machine', {}))
        rtype = meta.get('type', '?')
        ts = meta.get('timestamp', fname[:15])[:19]
        commit = git.get('commit_short', '?')
        mid = machine.get('machine_id', '?')

        if rtype == 'training':
            hp = data.get('hyperparams', {})
            details = (f"e{hp.get('n_embd','?')}_L{hp.get('n_layer','?')} "
                       f"loss={data.get('loss_final', 0):.3f} "
                       f"t={data.get('training_time_seconds', 0):.1f}s")
        elif rtype == 'benchmark':
            n_configs = len(data.get('results', []))
            details = f"{n_configs} configs"
        else:
            details = fname

        print(f"  {rtype:<12} {ts[:10]:<12} {commit:<10} {mid:<14} {details}")

    print(f"\n{len(runs)} run(s) total.")


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

    p_trend = sub.add_parser('trend', help='Show performance trend over time')
    p_trend.add_argument('--last', type=int, default=10,
                         help='Number of recent runs to show (default: 10)')

    p_latest = sub.add_parser('latest', help='Compare latest run against baseline')
    p_latest.add_argument('--baseline', default='baseline.json',
                          help='Baseline file (default: baseline.json)')

    p_runs = sub.add_parser('runs', help='List all archived runs')
    p_runs.add_argument('--type', choices=['training', 'benchmark'],
                        help='Filter by run type')

    args = parser.parse_args()
    commands = {
        'run': cmd_run,
        'compare': cmd_compare,
        'update-baseline': cmd_update_baseline,
        'trend': cmd_trend,
        'latest': cmd_latest,
        'runs': cmd_runs,
    }
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
