"""
Training harness for microgpt. Runs train.py, captures metrics, archives results
with machine identity for cross-run and cross-machine comparison.

Usage:
    python harness.py                          # train with defaults (20 steps)
    python harness.py --tag baseline           # label the run
    python harness.py --n-embd 32 --n-layer 2  # forward args to train.py
    python harness.py --compare                # compare all archived runs
"""

import os
import sys
import json
import time
import platform
import subprocess
import argparse
from datetime import datetime, timezone

from run_utils import archive_run, load_runs, get_machine_id


def run_training(train_args):
    """Run train.py as a subprocess, streaming output live."""
    cmd = [sys.executable, 'train.py', '--no-archive'] + train_args
    print(f">>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
    if result.returncode != 0:
        print(f"\ntrain.py exited with code {result.returncode}")
        sys.exit(result.returncode)


def archive_harness_run(tag=None):
    """Read _last_run.json, enrich and archive via run_utils.

    If train.py already auto-archived (it writes _meta), we only
    re-archive when a tag is provided that train.py didn't have.
    """
    metrics_path = '_last_run.json'
    if not os.path.exists(metrics_path):
        print("ERROR: _last_run.json not found. Did train.py complete successfully?")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    # train.py already auto-archives; only re-archive if harness adds a tag
    already_archived = '_meta' in metrics
    if not already_archived or tag:
        filepath = archive_run(metrics, 'training', tag=tag)
        print(f"\nRun archived: {filepath}")
    else:
        filepath = None

    os.remove(metrics_path)
    return filepath, metrics


def print_comparison(runs_tuples):
    """Print a formatted comparison table of all runs."""
    if not runs_tuples:
        print("No runs found in runs/ directory.")
        return

    # header
    cols = [
        ('Run', 20),
        ('Machine', 14),
        ('Commit', 10),
        ('Tag', 12),
        ('Steps', 7),
        ('n_embd', 7),
        ('n_layer', 8),
        ('Loss(fin)', 10),
        ('Time(s)', 9),
    ]
    header = '  '.join(name.ljust(width) for name, width in cols)
    print(header)
    print('-' * len(header))

    for fname, run in runs_tuples:
        hp = run.get('hyperparams', {})
        meta = run.get('_meta', {})
        machine = meta.get('machine', run.get('machine', {}))
        git = meta.get('git', {})
        row = [
            fname[:20].replace('.json', ''),
            machine.get('machine_id', '?'),
            git.get('commit_short', '?'),
            meta.get('tag', run.get('tag', '-'))[:12] if meta.get('tag') or run.get('tag') else '-',
            str(hp.get('num_steps', '?')),
            str(hp.get('n_embd', '?')),
            str(hp.get('n_layer', '?')),
            f"{run.get('loss_final', 0):.4f}",
            f"{run.get('training_time_seconds', 0):.1f}",
        ]
        line = '  '.join(val.ljust(width) for val, (_, width) in zip(row, cols))
        print(line)

    print(f"\n{len(runs_tuples)} run(s) total.")


def print_summary(metrics):
    """Print a quick summary of the just-completed run."""
    hp = metrics['hyperparams']
    meta = metrics.get('_meta', {})
    machine = meta.get('machine', metrics.get('machine', {}))
    git = meta.get('git', {})
    print(f"\n{'='*50}")
    print(f"  Machine:    {machine.get('machine_id', '?')} ({machine.get('hostname', '?')})")
    print(f"  Commit:     {git.get('commit_short', '?')} ({git.get('branch', '?')})")
    print(f"  Tag:        {meta.get('tag', metrics.get('tag', '-'))}")
    print(f"  Config:     embd={hp['n_embd']} layers={hp['n_layer']} heads={hp['n_head']} blk={hp['block_size']}")
    print(f"  Steps:      {hp['num_steps']}")
    print(f"  Params:     {metrics['num_params']}")
    print(f"  Loss final: {metrics['loss_final']:.4f}")
    print(f"  Loss m50:   {metrics['loss_mean_last_50']:.4f}")
    print(f"  Time:       {metrics['training_time_seconds']:.1f}s")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description='microgpt training harness',
        epilog='All unrecognized arguments are forwarded to train.py',
    )
    parser.add_argument('--compare', action='store_true', help='Compare all archived runs')
    parser.add_argument('--tag', type=str, default=None, help='Label for this run')
    known, train_args = parser.parse_known_args()

    if known.compare:
        runs = load_runs(run_type='training')
        print_comparison(runs)
        return

    # default to 20 steps if --num-steps not explicitly passed
    if '--num-steps' not in train_args:
        train_args.extend(['--num-steps', '20'])

    run_training(train_args)
    filepath, metrics = archive_harness_run(tag=known.tag)
    print_summary(metrics)


if __name__ == '__main__':
    main()
