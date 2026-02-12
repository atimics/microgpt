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
    try:
        import fastops
    except ImportError:
        print("  SKIP: C extension not available")
        return

    # vec_dot
    result = fastops.vec_dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert abs(result - 32.0) < 1e-10, f"vec_dot: expected 32.0, got {result}"

    # matvec (identity)
    W = [[1.0, 0.0], [0.0, 1.0]]
    x = [3.0, 4.0]
    result = fastops.matvec(W, x)
    assert abs(result[0] - 3.0) < 1e-10 and abs(result[1] - 4.0) < 1e-10, \
        f"matvec identity: expected [3,4], got {result}"

    # matvec (general)
    W = [[1.0, 2.0], [3.0, 4.0]]
    x = [1.0, 1.0]
    result = fastops.matvec(W, x)
    assert abs(result[0] - 3.0) < 1e-10 and abs(result[1] - 7.0) < 1e-10, \
        f"matvec general: expected [3,7], got {result}"

    # vec_axpy
    y = [1.0, 2.0, 3.0]
    fastops.vec_axpy(2.0, [1.0, 1.0, 1.0], y)
    assert abs(y[0] - 3.0) < 1e-10 and abs(y[1] - 4.0) < 1e-10 and abs(y[2] - 5.0) < 1e-10, \
        f"vec_axpy: expected [3,4,5], got {y}"

    print("  PASS: C extension ops correct")


def test_training_basic():
    """Train for 5 steps with C extension, verify loss is recorded."""
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


def test_training_pure_python():
    """Train without C extension to verify pure Python fallback."""
    env = os.environ.copy()
    env['MICROGPT_PURE_PYTHON'] = '1'
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '5'], env=env)
    assert rc == 0, f"train.py (pure Python) failed (rc={rc}):\n{err}\n{out}"
    assert 'training time:' in out, f"Missing 'training time:' in output:\n{out}"
    print("  PASS: pure Python training (5 steps)")


def test_training_alternate_config():
    """Train with non-default hyperparameters."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '3', '--n-embd', '32', '--n-layer', '2',
    ])
    assert rc == 0, f"train.py (n_embd=32, n_layer=2) failed (rc={rc}):\n{err}\n{out}"
    assert 'step    3' in out, f"Expected 3 steps in output:\n{out}"
    print("  PASS: alternate config training (n_embd=32, n_layer=2)")


def test_training_loss_decreases():
    """Train for 50 steps and verify loss trend is downward."""
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '50'])
    assert rc == 0, f"train.py failed (rc={rc}):\n{err}"

    with open('_last_run.json') as f:
        metrics = json.load(f)
    history = metrics['loss_history']

    # Compare first 5 avg vs last 5 avg
    first_5 = sum(history[:5]) / 5
    last_5 = sum(history[-5:]) / 5
    assert last_5 < first_5, \
        f"Loss did not decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"
    print(f"  PASS: loss decreased ({first_5:.4f} -> {last_5:.4f} over 50 steps)")


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


def main():
    parser = argparse.ArgumentParser(description='microgpt smoke tests')
    parser.add_argument('--quick', action='store_true',
                        help='Skip slow tests (roofline measurement)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    tests = [
        ('C extension build',     test_c_extension_builds),
        ('C extension ops',       test_c_extension_ops),
        ('Basic training',        test_training_basic),
        ('Pure Python training',  test_training_pure_python),
        ('Alternate config',      test_training_alternate_config),
        ('Loss decreases',        test_training_loss_decreases),
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
