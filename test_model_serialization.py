#!/usr/bin/env python3
"""
Tests for model serialization (save/load functionality).
Validates save/load round-trip, resumption, inference-only mode, and cross-compatibility.

Usage:
    python test_model_serialization.py
"""

import subprocess
import sys
import json
import os
import tempfile


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kwargs)
    return result.returncode, result.stdout, result.stderr


def extract_samples(output):
    """Extract generated samples from stdout."""
    samples = []
    for line in output.split('\n'):
        if line.startswith('sample '):
            # Extract the sample text after "sample N: "
            parts = line.split(': ', 1)
            if len(parts) == 2:
                samples.append(parts[1])
    return samples


def test_save_and_load_train():
    """Test save and load with train.py produces identical inference outputs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        model_path = tmp.name
    
    try:
        # Train and save
        rc, out, err = run([sys.executable, 'train.py', '--num-steps', '10', '--seed', '42',
                           '--save-model', model_path, '--no-archive'])
        assert rc == 0, f"Training failed (rc={rc}):\n{err}\n{out}"
        assert os.path.exists(model_path), f"Model file not created at {model_path}"
        
        # Load and run inference with fixed seed
        rc, out1, err = run([sys.executable, 'train.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999', '--no-archive'])
        assert rc == 0, f"First inference failed (rc={rc}):\n{err}\n{out1}"
        assert 'Skipping training (inference-only mode)' in out1, "Expected inference-only message"
        samples1 = extract_samples(out1)
        assert len(samples1) == 20, f"Expected 20 samples, got {len(samples1)}"
        
        # Load again with same seed - should produce identical results
        rc, out2, err = run([sys.executable, 'train.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999', '--no-archive'])
        assert rc == 0, f"Second inference failed (rc={rc}):\n{err}\n{out2}"
        samples2 = extract_samples(out2)
        assert len(samples2) == 20, f"Expected 20 samples, got {len(samples2)}"
        
        # Compare samples (they should be identical with same seed)
        assert samples1 == samples2, \
            f"Samples differ between identical runs:\nRun1: {samples1[:5]}\nRun2: {samples2[:5]}"
        
        # Load with different seed - should produce different results
        rc, out3, err = run([sys.executable, 'train.py', '--load-model', model_path,
                           '--inference-only', '--seed', '888', '--no-archive'])
        assert rc == 0, f"Third inference failed (rc={rc}):\n{err}\n{out3}"
        samples3 = extract_samples(out3)
        assert samples1 != samples3, \
            "Samples should differ with different seeds (model may not be working correctly)"
        
        print("  PASS: train.py save/load round-trip produces consistent outputs")
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_save_and_load_train_fast():
    """Test save and load with train_fast.py produces identical inference outputs."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        model_path = tmp.name
    
    try:
        # Train and save
        rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '10', '--seed', '42',
                           '--save-model', model_path])
        assert rc == 0, f"Training failed (rc={rc}):\n{err}\n{out}"
        assert os.path.exists(model_path), f"Model file not created at {model_path}"
        
        # Load and run inference with fixed seed
        rc, out1, err = run([sys.executable, 'train_fast.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999'])
        assert rc == 0, f"First inference failed (rc={rc}):\n{err}\n{out1}"
        assert 'Skipping training (inference-only mode)' in out1, "Expected inference-only message"
        samples1 = extract_samples(out1)
        assert len(samples1) == 20, f"Expected 20 samples, got {len(samples1)}"
        
        # Load again with same seed - should produce identical results
        rc, out2, err = run([sys.executable, 'train_fast.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999'])
        assert rc == 0, f"Second inference failed (rc={rc}):\n{err}\n{out2}"
        samples2 = extract_samples(out2)
        assert len(samples2) == 20, f"Expected 20 samples, got {len(samples2)}"
        
        # Compare samples (they should be identical with same seed)
        assert samples1 == samples2, \
            f"Samples differ between identical runs:\nRun1: {samples1[:5]}\nRun2: {samples2[:5]}"
        
        # Load with different seed - should produce different results
        rc, out3, err = run([sys.executable, 'train_fast.py', '--load-model', model_path,
                           '--inference-only', '--seed', '888'])
        assert rc == 0, f"Third inference failed (rc={rc}):\n{err}\n{out3}"
        samples3 = extract_samples(out3)
        assert samples1 != samples3, \
            "Samples should differ with different seeds (model may not be working correctly)"
        
        print("  PASS: train_fast.py save/load round-trip produces consistent outputs")
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_resume_training():
    """Test resuming training from a checkpoint."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        model_path = tmp.name
    
    try:
        # Train for 5 steps and save
        rc, out, err = run([sys.executable, 'train.py', '--num-steps', '5', '--seed', '42',
                           '--save-model', model_path, '--no-archive'])
        assert rc == 0, f"Initial training failed (rc={rc}):\n{err}\n{out}"
        
        # Load model data
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        assert model_data['optimizer_step'] == 5, \
            f"Expected optimizer_step=5, got {model_data['optimizer_step']}"
        assert len(model_data['loss_history']) == 5, \
            f"Expected 5 loss entries, got {len(model_data['loss_history'])}"
        
        # Resume training for 3 more steps
        rc, out, err = run([sys.executable, 'train.py', '--load-model', model_path,
                           '--num-steps', '3', '--save-model', model_path, '--no-archive'])
        assert rc == 0, f"Resume training failed (rc={rc}):\n{err}\n{out}"
        assert 'Resumed from step 5' in out, "Expected resumption message"
        
        # Check updated model
        with open(model_path, 'r') as f:
            resumed_data = json.load(f)
        
        assert resumed_data['optimizer_step'] == 8, \
            f"Expected optimizer_step=8, got {resumed_data['optimizer_step']}"
        assert len(resumed_data['loss_history']) == 8, \
            f"Expected 8 loss entries, got {len(resumed_data['loss_history'])}"
        
        print("  PASS: resume training from checkpoint")
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_cross_compatibility():
    """Test that train.py and train_fast.py can load each other's models."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        model_path = tmp.name
    
    try:
        # Train with train.py and save
        rc, out, err = run([sys.executable, 'train.py', '--num-steps', '10', '--seed', '42',
                           '--save-model', model_path, '--no-archive'])
        assert rc == 0, f"train.py failed (rc={rc}):\n{err}\n{out}"
        
        # Load with train_fast.py - run twice with same seed to verify consistency
        rc, out1, err = run([sys.executable, 'train_fast.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999'])
        assert rc == 0, f"train_fast.py load failed (rc={rc}):\n{err}\n{out1}"
        samples1 = extract_samples(out1)
        
        rc, out2, err = run([sys.executable, 'train_fast.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999'])
        assert rc == 0, f"train_fast.py second load failed (rc={rc}):\n{err}\n{out2}"
        samples2 = extract_samples(out2)
        
        # Samples from train_fast.py should be consistent
        assert samples1 == samples2, \
            f"train_fast.py inconsistent with same seed when loading train.py model"
        
        # Now reverse: train with train_fast.py
        rc, out, err = run([sys.executable, 'train_fast.py', '--num-steps', '10', '--seed', '42',
                           '--save-model', model_path])
        assert rc == 0, f"train_fast.py failed (rc={rc}):\n{err}\n{out}"
        
        # Load with train.py - run twice with same seed to verify consistency
        rc, out1, err = run([sys.executable, 'train.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999', '--no-archive'])
        assert rc == 0, f"train.py load failed (rc={rc}):\n{err}\n{out1}"
        samples3 = extract_samples(out1)
        
        rc, out2, err = run([sys.executable, 'train.py', '--load-model', model_path,
                           '--inference-only', '--seed', '999', '--no-archive'])
        assert rc == 0, f"train.py second load failed (rc={rc}):\n{err}\n{out2}"
        samples4 = extract_samples(out2)
        
        # Samples from train.py should be consistent
        assert samples3 == samples4, \
            f"train.py inconsistent with same seed when loading train_fast.py model"
        
        print("  PASS: cross-compatibility between train.py and train_fast.py")
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_model_json_format():
    """Test that saved model has correct JSON structure."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        model_path = tmp.name
    
    try:
        # Train and save
        rc, out, err = run([sys.executable, 'train.py', '--num-steps', '5',
                           '--save-model', model_path, '--no-archive'])
        assert rc == 0, f"Training failed (rc={rc}):\n{err}\n{out}"
        
        # Load and validate JSON structure
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Check top-level keys
        required_keys = {'hyperparams', 'vocab', 'optimizer_step', 'loss_history', 'weights'}
        assert set(model_data.keys()) == required_keys, \
            f"Missing keys in model JSON: {required_keys - set(model_data.keys())}"
        
        # Check hyperparams
        hyperparam_keys = {'n_embd', 'n_layer', 'n_head', 'block_size', 'learning_rate'}
        assert set(model_data['hyperparams'].keys()) == hyperparam_keys, \
            f"Missing hyperparams: {hyperparam_keys - set(model_data['hyperparams'].keys())}"
        
        # Check vocab
        assert 'stoi' in model_data['vocab'], "Missing stoi in vocab"
        assert 'itos' in model_data['vocab'], "Missing itos in vocab"
        
        # Check weights structure
        for name, weight in model_data['weights'].items():
            assert 'nout' in weight, f"Missing nout in {name}"
            assert 'nin' in weight, f"Missing nin in {name}"
            assert 'data' in weight, f"Missing data in {name}"
            assert 'm' in weight, f"Missing m (Adam first moment) in {name}"
            assert 'v' in weight, f"Missing v (Adam second moment) in {name}"
            
            # Check dimensions match
            assert len(weight['data']) == weight['nout'], \
                f"{name}: data rows ({len(weight['data'])}) != nout ({weight['nout']})"
            assert all(len(row) == weight['nin'] for row in weight['data']), \
                f"{name}: inconsistent row lengths in data"
        
        print("  PASS: model JSON format is correct")
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_inference_only_requires_load_model():
    """Test that --inference-only requires --load-model."""
    # Test train.py
    rc, out, err = run([sys.executable, 'train.py', '--inference-only'])
    assert rc != 0, "train.py should fail with --inference-only but no --load-model"
    assert '--inference-only requires --load-model' in err or '--inference-only requires --load-model' in out, \
        f"Expected validation error message, got:\nstdout: {out}\nstderr: {err}"
    
    # Test train_fast.py
    rc, out, err = run([sys.executable, 'train_fast.py', '--inference-only'])
    assert rc != 0, "train_fast.py should fail with --inference-only but no --load-model"
    assert '--inference-only requires --load-model' in err or '--inference-only requires --load-model' in out, \
        f"Expected validation error message, got:\nstdout: {out}\nstderr: {err}"
    
    print("  PASS: --inference-only validation works correctly")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    tests = [
        ('Save/load train.py', test_save_and_load_train),
        ('Save/load train_fast.py', test_save_and_load_train_fast),
        ('Resume training', test_resume_training),
        ('Cross-compatibility', test_cross_compatibility),
        ('Model JSON format', test_model_json_format),
        ('Inference-only validation', test_inference_only_requires_load_model),
    ]
    
    passed, failed = 0, 0
    failures = []
    
    print(f"\nRunning {len(tests)} model serialization tests...\n")
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
