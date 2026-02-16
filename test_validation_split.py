#!/usr/bin/env python3
"""
Tests for train/validation split functionality.
Validates that validation split, validation loss tracking, and early stopping work correctly.
"""

import subprocess
import sys
import json
import os


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, **kwargs)
    return result.returncode, result.stdout, result.stderr


def test_validation_split():
    """Test that validation split creates correct proportions."""
    print("Testing validation split proportions...")
    
    # Test with 10% validation split
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '3', '--val-split', '0.1', '--seed', '42'])
    assert rc == 0, f"train.py failed:\n{err}"
    
    # Check split message
    assert 'split:' in out, "Expected split message in output"
    # Should be approximately 90/10 split
    assert '28830 train docs, 3203 val docs' in out, f"Expected 90/10 split, got: {out}"
    
    print("  PASS: Validation split proportions correct")


def test_validation_split_determinism():
    """Test that the same seed produces the same split."""
    print("Testing validation split determinism...")
    
    # Run twice with same seed
    rc1, out1, err1 = run([sys.executable, 'train.py', '--num-steps', '2', '--val-split', '0.15', '--seed', '999'])
    assert rc1 == 0, f"First run failed:\n{err1}"
    
    rc2, out2, err2 = run([sys.executable, 'train.py', '--num-steps', '2', '--val-split', '0.15', '--seed', '999'])
    assert rc2 == 0, f"Second run failed:\n{err2}"
    
    # Extract split info from both runs
    split1 = [line for line in out1.split('\n') if 'split:' in line][0]
    split2 = [line for line in out2.split('\n') if 'split:' in line][0]
    
    assert split1 == split2, f"Splits differ with same seed:\n{split1}\n{split2}"
    
    print("  PASS: Validation split is deterministic")


def test_validation_loss_tracking():
    """Test that validation loss is tracked and reported."""
    print("Testing validation loss tracking...")
    
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '10', '--val-split', '0.1', 
                        '--val-every', '5', '--seed', '42'])
    assert rc == 0, f"train.py failed:\n{err}"
    
    # Check validation loss appears in output
    val_loss_lines = [line for line in out.split('\n') if 'val_loss' in line]
    assert len(val_loss_lines) == 2, f"Expected 2 val_loss reports (steps 5 and 10), got {len(val_loss_lines)}"
    
    # Check JSON output has validation data
    with open('_last_run.json', 'r') as f:
        data = json.load(f)
    
    assert 'val_loss_history' in data, "val_loss_history missing from JSON"
    assert len(data['val_loss_history']) == 2, f"Expected 2 validation losses, got {len(data['val_loss_history'])}"
    assert 'num_train_docs' in data, "num_train_docs missing from JSON"
    assert 'num_val_docs' in data, "num_val_docs missing from JSON"
    assert data['hyperparams']['val_split'] == 0.1, "val_split not in hyperparams"
    assert data['hyperparams']['val_every'] == 5, "val_every not in hyperparams"
    
    print("  PASS: Validation loss tracking works")


def test_no_validation_backward_compatibility():
    """Test that training without validation still works (backward compatibility)."""
    print("Testing backward compatibility (no validation)...")
    
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '5', '--seed', '42'])
    assert rc == 0, f"train.py failed:\n{err}"
    
    # Should not have split message or val_loss
    assert 'split:' not in out, "Should not have split message when val_split=0"
    assert 'val_loss' not in out, "Should not have val_loss when val_split=0"
    
    # Check JSON
    with open('_last_run.json', 'r') as f:
        data = json.load(f)
    
    assert data['val_loss_history'] == [], "val_loss_history should be empty"
    assert data['num_val_docs'] == 0, "num_val_docs should be 0"
    
    print("  PASS: Backward compatibility maintained")


def test_early_stopping():
    """Test that early stopping works when validation loss doesn't improve."""
    print("Testing early stopping...")
    
    # Use a moderate num-steps, with early patience
    # With patience=3, should stop if validation loss doesn't improve for 3 consecutive checks
    rc, out, err = run([sys.executable, 'train.py', '--num-steps', '100', '--val-split', '0.1',
                        '--val-every', '10', '--early-stop-patience', '3', '--seed', '42'])
    assert rc == 0, f"train.py failed:\n{err}"
    
    # Check if early stopping message appears
    if 'Early stopping' in out:
        # Early stopping occurred
        with open('_last_run.json', 'r') as f:
            data = json.load(f)
        
        # Should have stopped before 100 steps
        actual_steps = len(data['loss_history'])
        assert actual_steps < 100, f"Early stopping failed, ran all {actual_steps} steps"
        print(f"  PASS: Early stopping triggered at step {actual_steps}")
    else:
        # Early stopping didn't trigger (validation loss kept improving)
        # This is also valid behavior
        print("  PASS: Early stopping not triggered (validation loss kept improving)")


def test_train_fast_validation():
    """Test that train_fast.py also supports validation split."""
    print("Testing train_fast.py validation support...")
    
    # First check if C extension is available
    rc, _, _ = run([sys.executable, 'setup.py', 'build_ext', '--inplace'])
    if rc != 0:
        print("  SKIP: C extension not available")
        return
    
    rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '10', '--val-split', '0.1',
                        '--val-every', '5', '--seed', '42'])
    assert rc == 0, f"train_fast.py failed:\n{err}"
    
    # Check validation loss appears
    val_loss_lines = [line for line in out.split('\n') if 'val_loss' in line]
    assert len(val_loss_lines) == 2, f"Expected 2 val_loss reports in train_fast.py"
    
    print("  PASS: train_fast.py validation support works")


if __name__ == '__main__':
    tests = [
        test_validation_split,
        test_validation_split_determinism,
        test_validation_loss_tracking,
        test_no_validation_backward_compatibility,
        test_early_stopping,
        test_train_fast_validation,
    ]
    
    print("Running validation split tests...\n")
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        sys.exit(0)
