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
import hashlib
import platform
import socket
import subprocess
import argparse
from datetime import datetime, timezone


def get_machine_id():
    """Stable machine identifier: /etc/machine-id (Linux) with hostname fallback."""
    machine_id_path = '/etc/machine-id'
    if os.path.exists(machine_id_path):
        with open(machine_id_path) as f:
            raw = f.read().strip()
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    # fallback: hash hostname + platform
    raw = f"{socket.gethostname()}-{platform.machine()}-{platform.system()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def get_machine_meta():
    """Collect machine metadata for the run record."""
    return {
        'machine_id': get_machine_id(),
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'arch': platform.machine(),
    }


def run_training(train_args):
    """Run train.py as a subprocess, streaming output live."""
    cmd = [sys.executable, 'train.py'] + train_args
    print(f">>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) or '.')
    if result.returncode != 0:
        print(f"\ntrain.py exited with code {result.returncode}")
        sys.exit(result.returncode)


def archive_run(tag=None):
    """Read _last_run.json, enrich with machine metadata, archive to runs/."""
    metrics_path = '_last_run.json'
    if not os.path.exists(metrics_path):
        print("ERROR: _last_run.json not found. Did train.py complete successfully?")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    # enrich with machine and run metadata
    now = datetime.now(timezone.utc)
    metrics['machine'] = get_machine_meta()
    metrics['timestamp'] = now.isoformat()
    if tag:
        metrics['tag'] = tag

    # save to runs/
    runs_dir = 'runs'
    os.makedirs(runs_dir, exist_ok=True)
    machine_id = metrics['machine']['machine_id']
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{machine_id}.json"
    filepath = os.path.join(runs_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

    # clean up temp file
    os.remove(metrics_path)

    print(f"\nRun archived: {filepath}")
    return filepath, metrics


def load_all_runs():
    """Load all archived runs from runs/ directory."""
    runs_dir = 'runs'
    if not os.path.exists(runs_dir):
        return []
    runs = []
    for fname in sorted(os.listdir(runs_dir)):
        if fname.endswith('.json'):
            with open(os.path.join(runs_dir, fname)) as f:
                run = json.load(f)
            run['_filename'] = fname
            runs.append(run)
    return runs


def print_comparison(runs):
    """Print a formatted comparison table of all runs."""
    if not runs:
        print("No runs found in runs/ directory.")
        return

    # header
    cols = [
        ('Run', 20),
        ('Machine', 14),
        ('Tag', 12),
        ('Steps', 7),
        ('n_embd', 7),
        ('n_layer', 8),
        ('n_head', 7),
        ('LR', 10),
        ('Loss(fin)', 10),
        ('Loss(m50)', 10),
        ('Time(s)', 9),
    ]
    header = '  '.join(name.ljust(width) for name, width in cols)
    print(header)
    print('-' * len(header))

    for run in runs:
        hp = run.get('hyperparams', {})
        machine = run.get('machine', {})
        row = [
            run.get('_filename', '?')[:20].replace('.json', ''),
            machine.get('machine_id', '?'),
            run.get('tag', '-')[:12],
            str(hp.get('num_steps', '?')),
            str(hp.get('n_embd', '?')),
            str(hp.get('n_layer', '?')),
            str(hp.get('n_head', '?')),
            f"{hp.get('learning_rate', 0):.0e}",
            f"{run.get('loss_final', 0):.4f}",
            f"{run.get('loss_mean_last_50', 0):.4f}",
            f"{run.get('training_time_seconds', 0):.1f}",
        ]
        line = '  '.join(val.ljust(width) for val, (_, width) in zip(row, cols))
        print(line)

    print(f"\n{len(runs)} run(s) total.")


def print_summary(metrics):
    """Print a quick summary of the just-completed run."""
    hp = metrics['hyperparams']
    machine = metrics['machine']
    print(f"\n{'='*50}")
    print(f"  Machine:    {machine['machine_id']} ({machine['hostname']})")
    print(f"  Tag:        {metrics.get('tag', '-')}")
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
        runs = load_all_runs()
        print_comparison(runs)
        return

    # default to 20 steps if --num-steps not explicitly passed
    if '--num-steps' not in train_args:
        train_args.extend(['--num-steps', '20'])

    run_training(train_args)
    filepath, metrics = archive_run(tag=known.tag)
    print_summary(metrics)


if __name__ == '__main__':
    main()
