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
import tempfile


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


def test_c_extension_bounds_checking():
    """Verify C extension bounds checking works correctly."""
    import array
    try:
        import fastops
    except ImportError:
        print("  SKIP: C extension not available")
        return

    # Test embedding_flat bounds checking
    data = array.array('d', [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # 3x2 (3 rows, dim=2)
    
    # Valid access: idx=0 should work
    try:
        row0 = fastops.embedding_flat(data, 0, 2)
        assert abs(row0[0] - 10.0) < 1e-10 and abs(row0[1] - 20.0) < 1e-10, \
            f"embedding_flat(0): expected [10,20], got {list(row0)}"
    except Exception as e:
        raise AssertionError(f"embedding_flat(0) should succeed but got: {e}")
    
    # Valid access: idx=2 (last row) should work
    try:
        row2 = fastops.embedding_flat(data, 2, 2)
        assert abs(row2[0] - 50.0) < 1e-10 and abs(row2[1] - 60.0) < 1e-10, \
            f"embedding_flat(2): expected [50,60], got {list(row2)}"
    except Exception as e:
        raise AssertionError(f"embedding_flat(2) should succeed but got: {e}")
    
    # Negative index should raise IndexError
    try:
        fastops.embedding_flat(data, -1, 2)
        raise AssertionError("embedding_flat(-1) should raise IndexError but succeeded")
    except IndexError as e:
        assert "out of range" in str(e), f"Expected 'out of range' in error message, got: {e}"
    
    # Out of bounds positive index should raise IndexError
    try:
        fastops.embedding_flat(data, 3, 2)  # idx=3 is beyond the 3 rows (0,1,2)
        raise AssertionError("embedding_flat(3) should raise IndexError but succeeded")
    except IndexError as e:
        assert "out of range" in str(e), f"Expected 'out of range' in error message, got: {e}"
    
    # Way out of bounds should raise IndexError
    try:
        fastops.embedding_flat(data, 100, 2)
        raise AssertionError("embedding_flat(100) should raise IndexError but succeeded")
    except IndexError as e:
        assert "out of range" in str(e), f"Expected 'out of range' in error message, got: {e}"

    print("  PASS: C extension bounds checking correct")


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
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '16', '--n-head', '5', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with n_embd=16, n_head=5"
    assert 'n_embd (16) must be divisible by n_head (5)' in err, \
        f"Expected validation error message in stderr, got:\n{err}"
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '16', '--n-head', '5', '--num-steps', '1'])
    assert rc != 0, "train_fast.py should fail with n_embd=16, n_head=5"
    assert 'n_embd (16) must be divisible by n_head (5)' in err, \
        f"Expected validation error message in stderr, got:\n{err}"
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '16', '--n-head', '4', '--num-steps', '1'])
    assert rc == 0, f"train.py should succeed with n_embd=16, n_head=4:\n{err}"
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '32', '--n-head', '8', '--num-steps', '1'])
    assert rc == 0, f"train_fast.py should succeed with n_embd=32, n_head=8:\n{err}"
    print("  PASS: n_embd divisibility validation")


def test_validation_train():
    """Test that train.py validates n_embd >= n_head."""
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '2', '--n-head', '4', '--num-steps', '1'])
    assert rc != 0, f"train.py should have failed with n_embd=2, n_head=4 (rc={rc})"
    assert "must be >= n_head" in err or "must be >= n_head" in out, \
        f"Expected validation error message, got:\nstdout: {out}\nstderr: {err}"
    print("  PASS: train.py validates n_embd >= n_head")


def test_validation_train_fast():
    """Test that train_fast.py validates n_embd >= n_head."""
    try:
        import fastops
    except ImportError:
        print("  SKIP: fastops not available")
        return
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '2', '--n-head', '4', '--num-steps', '1'])
    assert rc != 0, f"train_fast.py should have failed with n_embd=2, n_head=4 (rc={rc})"
    assert "must be >= n_head" in err or "must be >= n_head" in out, \
        f"Expected validation error message, got:\nstdout: {out}\nstderr: {err}"
    print("  PASS: train_fast.py validates n_embd >= n_head")


def test_validation_roofline():
    """Test that roofline.py handles invalid configs gracefully."""
    rc, out, err = run([sys.executable, 'roofline.py', '--n-embd', '2', '--n-head', '4', '--no-measure'])
    assert rc == 0 or "ERROR: Config" in out or "must be >= n_head" in out, \
        f"roofline.py should handle n_embd=2, n_head=4 gracefully (rc={rc}):\nstdout: {out}\nstderr: {err}"
    if rc == 0:
        assert "ERROR: Config" in out or "skipped" in out, \
            f"Expected validation error/skip message, got:\n{out}"
    print("  PASS: roofline.py handles invalid n_embd/n_head gracefully")


def test_empty_dataset_handling():
    """Verify proper error handling for empty dataset."""
    input_backup = None
    if os.path.exists('input.txt'):
        with open('input.txt', 'r') as f:
            input_backup = f.read()
    try:
        with open('input.txt', 'w') as f:
            f.write('   \n\n  \n')
        rc, out, err = run([sys.executable, 'train.py', '--num-steps', '1'])
        assert rc == 1, f"train.py should exit with code 1 for empty dataset, got {rc}"
        assert 'Error: input.txt contains no non-empty lines' in err, \
            f"Expected error message in stderr, got:\n{err}"
        rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '1'])
        assert rc == 1, f"train_fast.py should exit with code 1 for empty dataset, got {rc}"
        assert 'Error: input.txt contains no non-empty lines' in err, \
            f"Expected error message in stderr, got:\n{err}"
        print("  PASS: empty dataset handling")
    finally:
        if input_backup is not None:
            with open('input.txt', 'w') as f:
                f.write(input_backup)
        elif os.path.exists('input.txt'):
            os.remove('input.txt')


def test_validation_negative_n_embd():
    """Verify negative n_embd is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '-1', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with negative n_embd"
    assert "n_embd must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: negative n_embd rejected")


def test_validation_zero_n_embd():
    """Verify zero n_embd is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '0', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with zero n_embd"
    assert "n_embd must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: zero n_embd rejected")


def test_validation_negative_n_layer():
    """Verify negative n_layer is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-layer', '-1', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with negative n_layer"
    assert "n_layer must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: negative n_layer rejected")


def test_validation_zero_n_layer():
    """Verify zero n_layer is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-layer', '0', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with zero n_layer"
    assert "n_layer must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: zero n_layer rejected")


def test_validation_negative_n_head():
    """Verify negative n_head is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-head', '-1', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with negative n_head"
    assert "n_head must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: negative n_head rejected")


def test_validation_zero_n_head():
    """Verify zero n_head is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-head', '0', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with zero n_head"
    assert "n_head must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: zero n_head rejected")


def test_validation_negative_block_size():
    """Verify negative block_size is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--block-size', '-1', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with negative block_size"
    assert "block_size must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: negative block_size rejected")


def test_validation_zero_block_size():
    """Verify zero block_size is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--block-size', '0', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with zero block_size"
    assert "block_size must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: zero block_size rejected")


def test_validation_negative_num_steps():
    """Verify negative num_steps is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '-1'])
    assert rc != 0, "train.py should fail with negative num_steps"
    assert "num_steps must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: negative num_steps rejected")


def test_validation_zero_num_steps():
    """Verify zero num_steps is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '0'])
    assert rc != 0, "train.py should fail with zero num_steps"
    assert "num_steps must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: zero num_steps rejected")


def test_validation_negative_learning_rate():
    """Verify negative learning_rate is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--learning-rate', '-0.1', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with negative learning_rate"
    assert "learning_rate must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: negative learning_rate rejected")


def test_validation_zero_learning_rate():
    """Verify zero learning_rate is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--learning-rate', '0', '--num-steps', '1'])
    assert rc != 0, "train.py should fail with zero learning_rate"
    assert "learning_rate must be positive" in err, f"Expected validation error in:\n{err}"
    print("  PASS: zero learning_rate rejected")


def test_validation_n_embd_not_divisible():
    """Verify n_embd not divisible by n_head is rejected."""
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '17', '--n-head', '4', '--num-steps', '1'])
    assert rc != 0, "train.py should fail when n_embd not divisible by n_head"
    assert "n_embd" in err and "must be divisible by" in err and "n_head" in err, \
        f"Expected divisibility validation error in:\n{err}"
    print("  PASS: n_embd not divisible by n_head rejected")


def test_validation_fast_path():
    """Verify validation works in train_fast.py as well."""
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '17', '--n-head', '4', '--num-steps', '1'])
    assert rc != 0, "train_fast.py should fail when n_embd not divisible by n_head"
    assert "n_embd" in err and "must be divisible by" in err and "n_head" in err, \
        f"Expected divisibility validation error in:\n{err}"
    print("  PASS: validation works in train_fast.py")


def test_inference_cli_flags_reference():
    """Test inference CLI flags (--temperature, --num-samples) on reference path."""
    rc, out, err = run([
        sys.executable, 'train.py',
        '--num-steps', '3', '--temperature', '0.7', '--num-samples', '7'
    ])
    assert rc == 0, f"train.py with inference flags failed (rc={rc}):\n{err}\n{out}"
    with open('_last_run.json') as f:
        metrics = json.load(f)
    assert 'temperature' in metrics['hyperparams'], "temperature not in hyperparams"
    assert metrics['hyperparams']['temperature'] == 0.7
    assert metrics['hyperparams']['num_samples'] == 7
    assert len(metrics['generated_samples']) == 7
    print("  PASS: inference CLI flags (reference path)")


def test_inference_cli_flags_fast():
    """Test inference CLI flags (--temperature, --num-samples) on fast path."""
    rc, out, err = run([
        sys.executable, 'train_fast.py',
        '--num-steps', '3', '--temperature', '0.3', '--num-samples', '5'
    ])
    assert rc == 0, f"train_fast.py with inference flags failed (rc={rc}):\n{err}\n{out}"
    with open('_last_run.json') as f:
        metrics = json.load(f)
    assert 'temperature' in metrics['hyperparams'], "temperature not in hyperparams"
    assert metrics['hyperparams']['temperature'] == 0.3
    assert metrics['hyperparams']['num_samples'] == 5
    assert len(metrics['generated_samples']) == 5
    print("  PASS: inference CLI flags (fast path)")


def test_validation_train():
    """Test that train.py validates n_embd >= n_head."""
    # Should fail with n_embd < n_head
    rc, out, err = run([sys.executable, 'train.py', '--n-embd', '2', '--n-head', '4', '--num-steps', '1'])
    assert rc != 0, f"train.py should have failed with n_embd=2, n_head=4 (rc={rc})"
    assert "must be >= n_head" in err or "must be >= n_head" in out, \
        f"Expected validation error message, got:\nstdout: {out}\nstderr: {err}"
    print("  PASS: train.py validates n_embd >= n_head")


def test_validation_train_fast():
    """Test that train_fast.py validates n_embd >= n_head."""
    try:
        import fastops
    except ImportError:
        print("  SKIP: fastops not available")
        return
    
    # Should fail with n_embd < n_head
    rc, out, err = run([sys.executable, 'train_fast.py', '--n-embd', '2', '--n-head', '4', '--num-steps', '1'])
    assert rc != 0, f"train_fast.py should have failed with n_embd=2, n_head=4 (rc={rc})"
    assert "must be >= n_head" in err or "must be >= n_head" in out, \
        f"Expected validation error message, got:\nstdout: {out}\nstderr: {err}"
    print("  PASS: train_fast.py validates n_embd >= n_head")


def test_validation_roofline():
    """Test that roofline.py handles invalid configs gracefully."""
    # Should skip invalid config but not crash
    rc, out, err = run([sys.executable, 'roofline.py', '--n-embd', '2', '--n-head', '4', '--no-measure'])
    assert rc == 0 or "ERROR: Config" in out or "must be >= n_head" in out, \
        f"roofline.py should handle n_embd=2, n_head=4 gracefully (rc={rc}):\nstdout: {out}\nstderr: {err}"
    if rc == 0:
        assert "ERROR: Config" in out or "skipped" in out, \
            f"Expected validation error/skip message, got:\n{out}"
    print("  PASS: roofline.py handles invalid n_embd/n_head gracefully")


def test_inference_script():
    """Test standalone inference script with saved model."""
    # Train a small model and save it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        model_path = f.name

    try:
        rc, out, err = run([sys.executable, 'train.py', '--num-steps', '20', '--seed', '42',
                            '--save-model', model_path])
        assert rc == 0, f"train.py with --save-model failed (rc={rc}):\n{err}"
        assert os.path.exists(model_path), f"Model file not created at {model_path}"

        # Test basic inference
        rc, out, err = run([sys.executable, 'inference.py', '--model', model_path,
                            '--num-samples', '3', '--seed', '100'])
        assert rc == 0, f"inference.py basic test failed (rc={rc}):\n{err}"
        assert 'Sample 1:' in out, "Missing sample 1 in output"
        assert 'Sample 2:' in out, "Missing sample 2 in output"
        assert 'Sample 3:' in out, "Missing sample 3 in output"

        # Test temperature parameter
        rc, out, err = run([sys.executable, 'inference.py', '--model', model_path,
                            '--temperature', '0.5', '--num-samples', '2', '--seed', '100'])
        assert rc == 0, f"inference.py with temperature failed (rc={rc}):\n{err}"
        assert 'temperature=0.5' in out, "Temperature not shown in output"

        # Test top-k sampling
        rc, out, err = run([sys.executable, 'inference.py', '--model', model_path,
                            '--top-k', '5', '--num-samples', '2', '--seed', '100'])
        assert rc == 0, f"inference.py with top-k failed (rc={rc}):\n{err}"
        assert 'top_k=5' in out, "Top-k not shown in output"

        # Test top-p sampling
        rc, out, err = run([sys.executable, 'inference.py', '--model', model_path,
                            '--top-p', '0.9', '--num-samples', '2', '--seed', '100'])
        assert rc == 0, f"inference.py with top-p failed (rc={rc}):\n{err}"
        assert 'top_p=0.9' in out, "Top-p not shown in output"

        # Test max-length parameter
        rc, out, err = run([sys.executable, 'inference.py', '--model', model_path,
                            '--max-length', '3', '--num-samples', '2', '--seed', '100'])
        assert rc == 0, f"inference.py with max-length failed (rc={rc}):\n{err}"

        # Test streaming output
        rc, out, err = run([sys.executable, 'inference.py', '--model', model_path,
                            '--stream', '--num-samples', '2', '--seed', '100'])
        assert rc == 0, f"inference.py with streaming failed (rc={rc}):\n{err}"
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.unlink(model_path)

    print("  PASS: inference script (all modes)")


def test_bpe_tokenizer():
    """Test BPE tokenizer integration with train.py and train_fast.py."""
    # Clean up tokenizer.json if it exists
    if os.path.exists('tokenizer.json'):
        os.unlink('tokenizer.json')

    # Test train.py with BPE
    rc, out, err = run([sys.executable, 'train.py', '--tokenizer', 'bpe', '--bpe-vocab-size', '40',
                        '--num-steps', '3', '--no-archive'])
    assert rc == 0, f"train.py with BPE failed (rc={rc}):\n{err}\n{out}"
    assert 'tokenizer: BPE' in out, "Missing BPE tokenizer indicator in output"
    assert 'Training BPE tokenizer' in out or 'Loading BPE model' in out, "No BPE training/loading message"
    assert os.path.exists('tokenizer.json'), "BPE model file not created"

    # Test train_fast.py with BPE (should load existing model)
    rc, out, err = run([sys.executable, 'train_fast.py', '--tokenizer', 'bpe', '--bpe-vocab-size', '40',
                        '--num-steps', '3'])
    assert rc == 0, f"train_fast.py with BPE failed (rc={rc}):\n{err}\n{out}"
    assert 'tokenizer: BPE' in out, "Missing BPE tokenizer indicator in output"
    assert 'Loading BPE model' in out, "Should load existing BPE model"

    # Clean up
    if os.path.exists('tokenizer.json'):
        os.unlink('tokenizer.json')

    print("  PASS: BPE tokenizer integration")


def main():
    parser = argparse.ArgumentParser(description='microgpt smoke tests')
    parser.add_argument('--quick', action='store_true',
                        help='Skip slow tests (roofline measurement)')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

    tests = [
        ('C extension build',     test_c_extension_builds),
        ('C extension ops',       test_c_extension_ops),
        ('C bounds checking',     test_c_extension_bounds_checking),
        ('Parameter validation',  test_validation_n_embd_divisibility),
        ('Reference training',    test_training_basic),
        ('Fast training',         test_training_fast),
        ('Alternate config',      test_training_alternate_config),
        ('Loss decreases',        test_training_loss_decreases),
        ('Ref/fast equivalence',  test_equivalence),
        ('Empty dataset handling', test_empty_dataset_handling),
        ('Validation: negative n_embd', test_validation_negative_n_embd),
        ('Validation: zero n_embd', test_validation_zero_n_embd),
        ('Validation: negative n_layer', test_validation_negative_n_layer),
        ('Validation: zero n_layer', test_validation_zero_n_layer),
        ('Validation: negative n_head', test_validation_negative_n_head),
        ('Validation: zero n_head', test_validation_zero_n_head),
        ('Validation: negative block_size', test_validation_negative_block_size),
        ('Validation: zero block_size', test_validation_zero_block_size),
        ('Validation: negative num_steps', test_validation_negative_num_steps),
        ('Validation: zero num_steps', test_validation_zero_num_steps),
        ('Validation: negative learning_rate', test_validation_negative_learning_rate),
        ('Validation: zero learning_rate', test_validation_zero_learning_rate),
        ('Validation: n_embd divisibility', test_validation_n_embd_not_divisible),
        ('Validation: fast path', test_validation_fast_path),
        ('Inference flags (ref)', test_inference_cli_flags_reference),
        ('Inference flags (fast)', test_inference_cli_flags_fast),
        ('Inference script',      test_inference_script),
        ('Roofline analytical',   test_roofline_analytical),
        ('Roofline all-configs',  test_roofline_all_configs),
        ('Validation train.py',   test_validation_train),
        ('Validation train_fast.py', test_validation_train_fast),
        ('Validation roofline.py', test_validation_roofline),
        ('BPE tokenizer',         test_bpe_tokenizer),
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
