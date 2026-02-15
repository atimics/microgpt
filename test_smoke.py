#!/usr/bin/env python3
"""
Smoke tests for microgpt. Validates training, inference, C extension, and roofline analysis.
Exit code 0 = all passed, 1 = failures.

Usage:
    python test_smoke.py              # run all tests
    python test_smoke.py --quick      # skip slow tests (roofline measurement)
"""

import subprocess
import sys
import json
import os
import argparse


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kwargs)
    return result.returncode, result.stdout, result.stderr


def test_c_extension_builds():
    """Verify C extension compiles."""
    rc, out, err = run([sys.executable, 'setup.py', 'build_ext', '--inplace'])
    assert rc == 0, f"C extension build failed:\n{err}"
    print("  PASS: C extension builds")


def test_c_extension_ops():
    """Verify C extension operations produce correct results."""
    import array
    try:
        import fastops
    except ImportError:
        print("  SKIP: C extension not available")
        return

    # vec_dot
    result = fastops.vec_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert abs(result - 32.0) < 1e-10, f"vec_dot: expected 32.0, got {result}"

    # vec_axpy
    y = [1.0, 2.0, 3.0]
    fastops.vec_axpy(2.0, [1.0, 1.0, 1.0], y)
    assert abs(y[0] - 3.0) < 1e-10 and abs(y[1] - 4.0) < 1e-10 and abs(y[2] - 5.0) < 1e-10, \
        f"vec_axpy: expected [3,4,5], got {y}"

    # matvec_flat
    W_flat = array.array('d', [1.0, 0.0, 0.0, 1.0])  # 2x2 identity
    x = array.array('d', [3.0, 4.0])
    result = fastops.matvec_flat(W_flat, x, 2, 2)
    assert abs(result[0] - 3.0) < 1e-10 and abs(result[1] - 4.0) < 1e-10, \
        f"matvec_flat identity: expected [3,4], got {list(result)}"

    # matvec_flat (general)
    W_flat = array.array('d', [1.0, 2.0, 3.0, 4.0])  # [[1,2],[3,4]]
    x = array.array('d', [1.0, 1.0])
    result = fastops.matvec_flat(W_flat, x, 2, 2)
    assert abs(result[0] - 3.0) < 1e-10 and abs(result[1] - 7.0) < 1e-10, \
        f"matvec_flat general: expected [3,7], got {list(result)}"

    # embedding_flat
    data = array.array('d', [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # 3x2
    row1 = fastops.embedding_flat(data, 1, 2)
    assert abs(row1[0] - 30.0) < 1e-10 and abs(row1[1] - 40.0) < 1e-10, \
        f"embedding_flat: expected [30,40], got {list(row1)}"

    print("  PASS: C extension ops correct")


def test_training_basic():
    """Train for 5 steps with reference, verify loss is recorded."""
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '5'])
    assert rc == 0, f"train.py failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"

    metrics_path = '_last_run.json'
    assert os.path.exists(metrics_path), "Missing _last_run.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert len(metrics['loss_history']) == 5, \
        f"Expected 5 loss entries, got {len(metrics['loss_history'])}"
    assert metrics['training_time_seconds'] > 0, "Training time should be positive"
    print("  PASS: basic training (5 steps)")


def test_training_fast():
    """Train for 5 steps with fast path, verify loss is recorded."""
    rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '5'])
    assert rc == 0, f"train_fast.py failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"

    with open('_last_run.json') as f:
        metrics = json.load(f)
    assert len(metrics['loss_history']) == 5, \
        f"Expected 5 loss entries, got {len(metrics['loss_history'])}"
    print("  PASS: fast training (5 steps)")


def test_training_alternate_config():
    """Train with non-default hyperparameters."""
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '3', '--n-embd', '32', '--n-layer', '2',
    ])
    assert rc == 0, f"train_fast.py (n_embd=32, n_layer=2) failed (rc={rc}):\n{err}\n{out}"
    assert 'step    3' in out, f"Expected 3 steps in output:\n{out}"
    print("  PASS: alternate config training (n_embd=32, n_layer=2)")


def test_training_loss_decreases():
    """Train for 50 steps and verify loss trend is downward."""
    rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '50'])
    assert rc == 0, f"train_fast.py failed (rc={rc}):\n{err}"

    with open('_last_run.json') as f:
        metrics = json.load(f)
    history = metrics['loss_history']

    # Compare first 5 avg vs last 5 avg
    first_5 = sum(history[:5]) / 5
    last_5 = sum(history[-5:]) / 5
    assert last_5 < first_5, \
        f"Loss did not decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"
    print(f"  PASS: loss decreased ({first_5:.4f} -> {last_5:.4f} over 50 steps)")


def test_equivalence():
    """Verify reference and fast path produce identical results."""
    # Run reference
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '20', '--seed', '42'])
    assert rc == 0, f"train.py failed (rc={rc}):\n{err}"
    with open('_last_run.json') as f:
        ref_metrics = json.load(f)

    # Run fast path
    rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '20', '--seed', '42'])
    assert rc == 0, f"train_fast.py failed (rc={rc}):\n{err}"
    with open('_last_run.json') as f:
        fast_metrics = json.load(f)

    # Compare loss histories
    ref_loss = ref_metrics['loss_history']
    fast_loss = fast_metrics['loss_history']
    assert len(ref_loss) == len(fast_loss), \
        f"Length mismatch: ref={len(ref_loss)}, fast={len(fast_loss)}"

    max_diff = 0.0
    for i, (r, f) in enumerate(zip(ref_loss, fast_loss)):
        diff = abs(r - f)
        max_diff = max(max_diff, diff)
        assert diff < 1e-6, \
            f"Loss diverged at step {i+1}: ref={r:.10f}, fast={f:.10f}, diff={diff:.2e}"

    print(f"  PASS: equivalence verified (20 steps, max diff={max_diff:.2e})")


def test_roofline_analytical():
    """Run roofline analysis without measurement (pure analytical)."""
    rc, out, err = run([sys.executable, 'roofline.py', '--no-measure'])
    assert rc == 0, f"roofline.py (analytical) failed (rc={rc}):\n{err}\n{out}"
    assert 'FLOP breakdown' in out, "Missing FLOP breakdown section"
    assert 'Memory traffic' in out, "Missing Memory traffic section"
    assert 'Operational intensity' in out, "Missing OI section"
    assert 'Cache Hierarchy Bandwidth' in out, "Missing cache bandwidth section"
    print("  PASS: roofline analytical mode")


def test_roofline_measured():
    """Run roofline with live measurement (default config only)."""
    rc, out, err = run([sys.executable, 'roofline.py', '--json', '/tmp/roofline_test.json'])
    assert rc == 0, f"roofline.py (measured) failed (rc={rc}):\n{err}\n{out}"
    assert 'ms/step' in out, "Missing timing measurement"
    assert 'VERDICT:' in out, "Missing verdict"

    assert os.path.exists('/tmp/roofline_test.json'), "Missing JSON output"
    with open('/tmp/roofline_test.json') as f:
        data = json.load(f)
    assert 'cpu' in data, "Missing cpu info in JSON"
    assert 'bandwidth' in data, "Missing bandwidth info in JSON"
    assert 'cache_hierarchy' in data['bandwidth'], "Missing cache hierarchy in JSON"
    assert len(data['results']) > 0, "No results in JSON"
    print("  PASS: roofline measured mode + JSON output")


def test_roofline_all_configs():
    """Run roofline with all configs (analytical only for speed)."""
    rc, out, err = run([sys.executable, 'roofline.py', '--all-configs', '--no-measure'])
    assert rc == 0, f"roofline.py (all-configs) failed (rc={rc}):\n{err}\n{out}"
    # Should have 6 config headers
    count = out.count('n_embd=')
    assert count == 6, f"Expected 6 config analyses, found {count}"
    print("  PASS: roofline all-configs analytical")


def test_validation_n_embd_divisibility():
    """Verify n_embd divisibility validation rejects invalid configs."""
    # Test train.py with invalid config (16 % 5 != 0)
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '16', '--n-head', '5', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with n_embd=16, n_head=5"
    assert 'n_embd (16) must be divisible by n_head (5)' in err, \
        f"Expected validation error message in stderr, got:\n{err}"
    
    # Test train_fast.py with invalid config (16 % 5 != 0)
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '16', '--n-head', '5', '--num-steps', '1'])
    assert rc != 0, "train_fast.py should fail with n_embd=16, n_head=5"
    assert 'n_embd (16) must be divisible by n_head (5)' in err, \
        f"Expected validation error message in stderr, got:\n{err}"
    
    # Test train.py with valid config (16 % 4 == 0)
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '16', '--n-head', '4', '--num-steps', '1'])
    assert rc == 0, f"train.py should succeed with n_embd=16, n_head=4:\n{err}"
    
    # Test train_fast.py with valid config (32 % 8 == 0)
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '32', '--n-head', '8', '--num-steps', '1'])
    assert rc == 0, f"train_fast.py should succeed with n_embd=32, n_head=8:\n{err}"
    
    print("  PASS: n_embd divisibility validation")


def main():
    parser = argparse.ArgumentParser(description='microgpt smoke tests')
    parser.add_argument('--quick', action='store_true',
                        help='Skip slow tests (roofline measurement)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    tests = [
        ('C extension build',     test_c_extension_builds),
        ('C extension ops',       test_c_extension_ops),
        ('Parameter validation',  test_validation_n_embd_divisibility),
        ('Reference training',    test_training_basic),
        ('Fast training',         test_training_fast),
        ('Alternate config',      test_training_alternate_config),
        ('Loss decreases',        test_training_loss_decreases),
        ('Ref/fast equivalence',  test_equivalence),
        ('Roofline analytical',   test_roofline_analytical),
        ('Roofline all-configs',  test_roofline_all_configs),
    ]

    if not args.quick:
        tests.append(('Roofline measured', test_roofline_measured))

    passed, failed, skipped = 0, 0, 0
    failures = []

    print(f"\nRunning {len(tests)} smoke tests...\n")
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
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print(f"  Failures: {', '.join(failures)}")
    print(f"{'='*50}")

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
