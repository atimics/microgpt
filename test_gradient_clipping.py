#!/usr/bin/env python3
"""
Test gradient clipping functionality.
Validates that gradient clipping works correctly in both train.py and train_fast.py.
"""

import subprocess
import sys
import json
import os


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kwargs)
    return result.returncode, result.stdout, result.stderr


def test_train_with_gradient_clipping():
    """Test train.py with gradient clipping enabled."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '5',
        '--grad-clip', '1.0',
        '--seed', '42'
    ])
    assert rc == 0, f"train.py with gradient clipping failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    
    # Verify training completed
    with open('_last_run.json') as f:
        metrics = json.load(f)
    assert len(metrics['loss_history']) == 5, \
        f"Expected 5 loss entries, got {len(metrics['loss_history'])}"
    
    print("  PASS: train.py with gradient clipping (grad_clip=1.0)")


def test_train_fast_with_gradient_clipping():
    """Test train_fast.py with gradient clipping enabled."""
    try:
        import fastops
    except ImportError:
        print("  SKIP: fastops not available")
        return
    
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '5',
        '--grad-clip', '1.0',
        '--seed', '42'
    ])
    assert rc == 0, f"train_fast.py with gradient clipping failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    
    # Verify training completed
    with open('_last_run.json') as f:
        metrics = json.load(f)
    assert len(metrics['loss_history']) == 5, \
        f"Expected 5 loss entries, got {len(metrics['loss_history'])}"
    
    print("  PASS: train_fast.py with gradient clipping (grad_clip=1.0)")


def test_gradient_clipping_disabled():
    """Test that gradient clipping is disabled when grad_clip=0 (default)."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '3',
        '--seed', '42'
    ])
    assert rc == 0, f"train.py without gradient clipping failed (rc={rc}):\n{err}\n{out}"
    
    with open('_last_run.json') as f:
        metrics = json.load(f)
    assert len(metrics['loss_history']) == 3, \
        f"Expected 3 loss entries, got {len(metrics['loss_history'])}"
    
    print("  PASS: train.py with gradient clipping disabled (default)")


def test_gradient_clipping_different_values():
    """Test gradient clipping with different max_norm values."""
    test_values = [0.5, 2.0, 5.0]
    
    for clip_value in test_values:
        rc, out, err = run([
            sys.executable, 'train.py',
            '--num-steps', '3',
            '--grad-clip', str(clip_value),
            '--seed', '42'
        ])
        assert rc == 0, f"train.py with grad_clip={clip_value} failed (rc={rc}):\n{err}"
        
        with open('_last_run.json') as f:
            metrics = json.load(f)
        assert len(metrics['loss_history']) == 3, \
            f"Expected 3 loss entries with grad_clip={clip_value}"
    
    print(f"  PASS: gradient clipping with values {test_values}")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    tests = [
        ('Gradient clipping train.py', test_train_with_gradient_clipping),
        ('Gradient clipping train_fast.py', test_train_fast_with_gradient_clipping),
        ('Gradient clipping disabled', test_gradient_clipping_disabled),
        ('Gradient clipping different values', test_gradient_clipping_different_values),
    ]
    
    passed, failed = 0, 0
    failures = []
    
    print(f"\nRunning {len(tests)} gradient clipping tests...\n")
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failures.append(name)
            failed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {type(e).__name__}: {e}")
            failures.append(name)
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failures:
        print(f"  Failures: {', '.join(failures)}")
    print(f"{'='*50}")
    
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
