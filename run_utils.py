"""
Shared utilities for microgpt run archiving and reporting.
All run producers (train.py, benchmark.py, harness.py) use these
to ensure consistent schemas and auto-archiving to runs/.
"""

import os
import json
import socket
import platform
import hashlib
import subprocess
import datetime


def get_machine_id():
    """Stable machine identifier from /etc/machine-id or hostname fallback."""
    mid_path = '/etc/machine-id'
    if os.path.exists(mid_path):
        with open(mid_path) as f:
            raw = f.read().strip()
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    raw = f"{socket.gethostname()}-{platform.machine()}-{platform.system()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def get_git_info():
    """Get current git commit hash and branch, or None if not in a repo."""
    info = {}
    try:
        info['commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        info['commit_short'] = info['commit'][:8]
        info['branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        info['dirty'] = subprocess.call(
            ['git', 'diff', '--quiet'],
            stderr=subprocess.DEVNULL
        ) != 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return info if info else None


def make_run_meta(run_type, tag=None):
    """Create a standardized _meta block for any run.

    Args:
        run_type: 'training' or 'benchmark'
        tag: optional user label

    Returns:
        dict with type, timestamp, machine info, git info
    """
    ts = datetime.datetime.now(datetime.timezone.utc)
    meta = {
        'type': run_type,
        'timestamp': ts.isoformat(),
        'timestamp_local': ts.astimezone().strftime('%Y%m%d_%H%M%S'),
        'machine': {
            'machine_id': get_machine_id(),
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python': platform.python_version(),
        },
    }
    git = get_git_info()
    if git:
        meta['git'] = git
    if tag:
        meta['tag'] = tag
    return meta


def archive_run(data, run_type, tag=None):
    """Save a run dict to runs/ with consistent naming.

    Adds _meta block with type, timestamp, machine, and git info.
    Returns the path to the archived file.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(root, 'runs')
    os.makedirs(runs_dir, exist_ok=True)

    meta = make_run_meta(run_type, tag=tag)
    data['_meta'] = meta

    ts = meta['timestamp_local']
    machine_id = meta['machine']['machine_id']
    suffix = f'_{tag}' if tag else ''
    filename = f'{ts}_{machine_id}_{run_type}{suffix}.json'
    filepath = os.path.join(runs_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return filepath


def load_runs(run_type=None, runs_dir=None):
    """Load all archived runs, optionally filtered by type.

    Args:
        run_type: 'training', 'benchmark', or None for all
        runs_dir: override runs directory path

    Returns:
        list of (filename, data) tuples sorted by filename (timestamp order)
    """
    if runs_dir is None:
        runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    if not os.path.exists(runs_dir):
        return []

    runs = []
    for fname in sorted(os.listdir(runs_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(runs_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Determine run type from _meta or filename heuristic
        meta = data.get('_meta', {})
        rtype = meta.get('type')
        if rtype is None:
            # Legacy files: infer type from content
            if 'results' in data and 'cpu' in data:
                rtype = 'benchmark'
            elif 'loss_history' in data:
                rtype = 'training'
            else:
                rtype = 'unknown'

        if run_type and rtype != run_type:
            continue

        runs.append((fname, data))

    return runs
