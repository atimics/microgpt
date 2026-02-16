#!/usr/bin/env python3
"""
Test learning rate schedules (linear, cosine, warmup).
"""

import subprocess
import sys
import json
import math


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kwargs)
    return result.returncode, result.stdout, result.stderr


def test_linear_schedule():
    """Test linear learning rate schedule (default)."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '10',
        '--lr-schedule', 'linear',
        '--no-archive'
    ])
    assert rc == 0, f"train.py with linear schedule failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: linear schedule training")


def test_cosine_schedule():
    """Test cosine learning rate schedule."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '10',
        '--lr-schedule', 'cosine',
        '--no-archive'
    ])
    assert rc == 0, f"train.py with cosine schedule failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: cosine schedule training")


def test_warmup():
    """Test learning rate warmup."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '20',
        '--lr-schedule', 'cosine',
        '--warmup-steps', '5',
        '--no-archive'
    ])
    assert rc == 0, f"train.py with warmup failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: warmup with cosine schedule")


def test_linear_schedule_fast():
    """Test linear learning rate schedule with fast training."""
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '10',
        '--lr-schedule', 'linear',
    ])
    assert rc == 0, f"train_fast.py with linear schedule failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: linear schedule fast training")


def test_cosine_schedule_fast():
    """Test cosine learning rate schedule with fast training."""
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '10',
        '--lr-schedule', 'cosine',
    ])
    assert rc == 0, f"train_fast.py with cosine schedule failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: cosine schedule fast training")


def test_warmup_fast():
    """Test learning rate warmup with fast training."""
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '20',
        '--lr-schedule', 'cosine',
        '--warmup-steps', '5',
    ])
    assert rc == 0, f"train_fast.py with warmup failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: warmup with cosine schedule (fast)")


def test_equivalence_with_schedules():
    """Verify reference and fast path produce same results with cosine schedule."""
    # Run reference with cosine
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '15',
        '--lr-schedule', 'cosine',
        '--seed', '42',
        '--no-archive'
    ])
    assert rc == 0, f"train.py failed (rc={rc}):\n{err}"
    with open('_last_run.json') as f:
        ref_metrics = json.load(f)

    # Run fast path with cosine
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '15',
        '--lr-schedule', 'cosine',
        '--seed', '42'
    ])
    assert rc == 0, f"train_fast.py failed (rc={rc}):\n{err}"
    with open('_last_run.json') as f:
        fast_metrics = json.load(f)

    # Compare loss histories
    ref_loss = ref_metrics['loss_history']
    fast_loss = fast_metrics['loss_history']
    assert len(ref_loss) == len(fast_loss), \
        f"Length mismatch: ref={len(ref_loss)}, fast={len(fast_loss)}"
    
    # Check losses match closely (allowing for small numerical differences)
    max_diff = max(abs(r - f) for r, f in zip(ref_loss, fast_loss))
    assert max_diff < 1e-6, f"Loss histories differ by {max_diff:.2e}"
    print(f"  PASS: equivalence with cosine schedule (max diff: {max_diff:.2e})")


def test_validation_warmup_equals_num_steps():
    """Test that warmup_steps >= num_steps is rejected."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '10',
        '--warmup-steps', '10',
        '--no-archive'
    ])
    assert rc != 0, "train.py should fail when warmup_steps >= num_steps"
    assert 'warmup_steps' in err or 'warmup_steps' in out, \
        f"Expected error message about warmup_steps:\n{err}\n{out}"
    print("  PASS: validation rejects warmup_steps >= num_steps")


def test_validation_warmup_exceeds_num_steps():
    """Test that warmup_steps > num_steps is rejected."""
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '10',
        '--warmup-steps', '15',
    ])
    assert rc != 0, "train_fast.py should fail when warmup_steps > num_steps"
    assert 'warmup_steps' in err or 'warmup_steps' in out, \
        f"Expected error message about warmup_steps:\n{err}\n{out}"
    print("  PASS: validation rejects warmup_steps > num_steps")


def main():
    """Run all learning rate schedule tests."""
    tests = [
        test_linear_schedule,
        test_cosine_schedule,
        test_warmup,
        test_linear_schedule_fast,
        test_cosine_schedule_fast,
        test_warmup_fast,
        test_equivalence_with_schedules,
        test_validation_warmup_equals_num_steps,
        test_validation_warmup_exceeds_num_steps,
    ]
    
    print("Running learning rate schedule tests...")
    failures = []
    for test in tests:
        try:
            test()
        except Exception as e:
            failures.append((test.__name__, str(e)))
            print(f"  FAIL: {test.__name__}")
            print(f"    {e}")
    
    if failures:
        print(f"\n{len(failures)}/{len(tests)} tests FAILED")
        for name, msg in failures:
            print(f"  - {name}")
        return 1
    else:
        print(f"\nAll {len(tests)} tests PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
