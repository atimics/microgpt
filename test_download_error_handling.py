#!/usr/bin/env python3
"""
Test dataset download error handling in train.py and train_fast.py.
Validates that network errors are handled gracefully with proper cleanup.
"""

import subprocess
import sys
import os
import tempfile
import shutil


def test_train_with_existing_file():
    """Verify train.py works when input.txt already exists."""
    # Run in a temp directory with an existing input.txt
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy train.py to temp directory
        shutil.copy('train.py', tmpdir)
        
        input_path = os.path.join(tmpdir, 'input.txt')
        with open(input_path, 'w') as f:
            f.write('alice\nbob\ncharlie\n')
        
        result = subprocess.run(
            [sys.executable, 'train.py', '--num-steps', '1'],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed when input.txt exists
        assert result.returncode == 0, f"train.py failed with existing input.txt:\n{result.stderr}\n{result.stdout}"
        print("  PASS: train.py works with existing input.txt")


def test_train_fast_with_existing_file():
    """Verify train_fast.py works when input.txt already exists."""
    # Check if fastops is available
    try:
        import fastops
    except ImportError:
        print("  SKIP: fastops not available")
        return
    
    # Run in a temp directory with an existing input.txt
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy train_fast.py and fastops to temp directory
        shutil.copy('train_fast.py', tmpdir)
        
        # Copy fastops extension module (Python version-specific)
        import glob
        for pattern in ['fastops*.so', 'fastops*.pyd', 'fastops*.dll']:
            for fastops_file in glob.glob(pattern):
                shutil.copy(fastops_file, tmpdir)
        
        input_path = os.path.join(tmpdir, 'input.txt')
        with open(input_path, 'w') as f:
            f.write('alice\nbob\ncharlie\n')
        
        result = subprocess.run(
            [sys.executable, 'train_fast.py', '--num-steps', '1'],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed when input.txt exists
        assert result.returncode == 0, f"train_fast.py failed with existing input.txt:\n{result.stderr}\n{result.stdout}"
        print("  PASS: train_fast.py works with existing input.txt")


def test_download_with_bad_url():
    """Test that a bad URL is handled gracefully with clear error message."""
    # Create a temporary modified version of train.py with a bad URL
    with tempfile.TemporaryDirectory() as tmpdir:
        # Read original train.py
        with open('train.py', 'r') as f:
            content = f.read()
        
        # Replace URL with a bad one
        modified_content = content.replace(
            'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',
            'https://invalid.url.that.does.not.exist.example.com/nonexistent.txt'
        )
        
        # Write modified version
        modified_path = os.path.join(tmpdir, 'train_modified.py')
        with open(modified_path, 'w') as f:
            f.write(modified_content)
        
        # Run it (should fail gracefully)
        result = subprocess.run(
            [sys.executable, modified_path, '--num-steps', '1'],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should exit with error code 1
        assert result.returncode == 1, "Expected exit code 1 for bad URL"
        
        # Should have helpful error message
        stderr = result.stderr.lower()
        assert 'error downloading' in stderr or 'error' in stderr, \
            f"Expected error message in stderr, got:\n{result.stderr}"
        assert 'manually download' in stderr or 'input.txt' in stderr, \
            f"Expected fallback instructions in stderr, got:\n{result.stderr}"
        
        # Should NOT create a partial input.txt file
        assert not os.path.exists(os.path.join(tmpdir, 'input.txt')), \
            "Partial input.txt should not exist after failed download"
        
        # Should NOT leave temp files around
        tmp_files = [f for f in os.listdir(tmpdir) if f.startswith('.input_')]
        assert len(tmp_files) == 0, f"Temporary files not cleaned up: {tmp_files}"
        
        print("  PASS: Bad URL handled gracefully with cleanup")


def test_successful_download():
    """Test that download works correctly with the actual URL (integration test)."""
    # This test actually downloads from the real URL to ensure it works
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy train.py to temp directory
        shutil.copy('train.py', tmpdir)
        
        # Don't create input.txt, force download
        result = subprocess.run(
            [sys.executable, 'train.py', '--num-steps', '1'],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed
        assert result.returncode == 0, f"train.py failed during download:\n{result.stderr}\n{result.stdout}"
        
        # input.txt should exist
        input_path = os.path.join(tmpdir, 'input.txt')
        assert os.path.exists(input_path), "input.txt was not created"
        
        # Should contain actual data (names dataset)
        with open(input_path) as f:
            content = f.read()
        assert len(content) > 100, "Downloaded file seems too small"
        assert '\n' in content, "File should contain newlines"
        
        # Should NOT leave temp files around
        tmp_files = [f for f in os.listdir(tmpdir) if f.startswith('.input_')]
        assert len(tmp_files) == 0, f"Temporary files not cleaned up: {tmp_files}"
        
        print("  PASS: Successful download and cleanup")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    tests = [
        ('Train with existing file', test_train_with_existing_file),
        ('Train fast with existing file', test_train_fast_with_existing_file),
        ('Bad URL error handling', test_download_with_bad_url),
        ('Successful download', test_successful_download),
    ]
    
    passed = 0
    failed = 0
    failures = []
    
    print("\nRunning download error handling tests...\n")
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
