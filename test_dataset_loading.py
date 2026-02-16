#!/usr/bin/env python3
"""
Tests for multiple dataset file support (--data and --data-dir).
Tests backward compatibility, glob patterns, and directory loading.
"""

import subprocess
import sys
import os
import tempfile
import shutil


def run(cmd, **kwargs):
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kwargs)
    return result.returncode, result.stdout, result.stderr


def test_backward_compatibility():
    """Verify default behavior still works without new flags (uses input.txt)."""
    # Run in temp dir with input.txt
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple input.txt
        input_path = os.path.join(tmpdir, 'input.txt')
        with open(input_path, 'w') as f:
            f.write('alice\nbob\ncharlie\n')
        
        # Copy train.py to temp dir to isolate the test
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        # Run train.py with default behavior (should use input.txt)
        rc, out, err = run([sys.executable, train_script, '--num-steps', '3'], cwd=tmpdir)
        assert rc == 0, f"train.py failed:\n{err}\n{out}"
        assert 'vocab size:' in out, f"Expected vocab size in output:\n{out}"
        assert 'num docs: 3' in out, f"Expected 3 docs in output:\n{out}"
        print("  PASS: backward compatibility (input.txt)")


def test_single_data_file():
    """Test --data with a single file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a custom data file
        data_path = os.path.join(tmpdir, 'custom.txt')
        with open(data_path, 'w') as f:
            f.write('one\ntwo\nthree\n')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        rc, out, err = run([sys.executable, train_script, '--num-steps', '3', '--data', data_path], cwd=tmpdir)
        assert rc == 0, f"train.py --data failed:\n{err}\n{out}"
        assert 'num docs: 3' in out, f"Expected 3 docs in output:\n{out}"
        print("  PASS: single --data file")


def test_multiple_data_files():
    """Test --data with multiple files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple data files
        file1 = os.path.join(tmpdir, 'data1.txt')
        file2 = os.path.join(tmpdir, 'data2.txt')
        with open(file1, 'w') as f:
            f.write('alpha\nbeta\n')
        with open(file2, 'w') as f:
            f.write('gamma\ndelta\n')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        rc, out, err = run([
            sys.executable, train_script, '--num-steps', '3',
            '--data', file1, '--data', file2
        ], cwd=tmpdir)
        assert rc == 0, f"train.py --data (multiple) failed:\n{err}\n{out}"
        assert 'num docs: 4' in out, f"Expected 4 docs (2+2) in output:\n{out}"
        print("  PASS: multiple --data files")


def test_glob_pattern():
    """Test --data with glob pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple files matching a pattern
        for i in range(3):
            filepath = os.path.join(tmpdir, f'file{i}.txt')
            with open(filepath, 'w') as f:
                f.write(f'doc{i}\n')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        pattern = os.path.join(tmpdir, 'file*.txt')
        rc, out, err = run([sys.executable, train_script, '--num-steps', '3', '--data', pattern], cwd=tmpdir)
        assert rc == 0, f"train.py --data (glob) failed:\n{err}\n{out}"
        assert 'num docs: 3' in out, f"Expected 3 docs in output:\n{out}"
        print("  PASS: glob pattern")


def test_data_dir():
    """Test --data-dir with directory of text files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a data directory with text files
        data_dir = os.path.join(tmpdir, 'datasets')
        os.makedirs(data_dir)
        
        for i in range(2):
            filepath = os.path.join(data_dir, f'data{i}.txt')
            with open(filepath, 'w') as f:
                f.write(f'line{i}a\nline{i}b\n')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        rc, out, err = run([sys.executable, train_script, '--num-steps', '3', '--data-dir', data_dir], cwd=tmpdir)
        assert rc == 0, f"train.py --data-dir failed:\n{err}\n{out}"
        assert 'num docs: 4' in out, f"Expected 4 docs (2+2) in output:\n{out}"
        print("  PASS: --data-dir")


def test_combined_data_and_data_dir():
    """Test combining --data and --data-dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create individual file
        file1 = os.path.join(tmpdir, 'single.txt')
        with open(file1, 'w') as f:
            f.write('single1\nsingle2\n')
        
        # Create directory with files
        data_dir = os.path.join(tmpdir, 'dir')
        os.makedirs(data_dir)
        file2 = os.path.join(data_dir, 'dir1.txt')
        with open(file2, 'w') as f:
            f.write('dir1\ndir2\n')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        rc, out, err = run([
            sys.executable, train_script, '--num-steps', '3',
            '--data', file1, '--data-dir', data_dir
        ], cwd=tmpdir)
        assert rc == 0, f"train.py --data + --data-dir failed:\n{err}\n{out}"
        assert 'num docs: 4' in out, f"Expected 4 docs (2+2) in output:\n{out}"
        print("  PASS: combined --data and --data-dir")


def test_train_fast_compatibility():
    """Verify train_fast.py also supports the new flags."""
    # Skip if fastops is not available
    try:
        import fastops
    except ImportError:
        print("  SKIP: fastops not available")
        return
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, 'fast_data.txt')
        with open(data_path, 'w') as f:
            f.write('fast1\nfast2\nfast3\n')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train_fast.py')
        
        rc, out, err = run([sys.executable, train_script, '--num-steps', '3', '--data', data_path], cwd=tmpdir)
        assert rc == 0, f"train_fast.py --data failed:\n{err}\n{out}"
        assert 'num docs: 3' in out, f"Expected 3 docs in output:\n{out}"
        print("  PASS: train_fast.py --data")


def test_nonexistent_file_error():
    """Verify warning for glob patterns that match no files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        # When using a pattern that matches no files, should warn but continue with input.txt
        rc, out, err = run([
            sys.executable, train_script, '--num-steps', '3',
            '--data', '/nonexistent/*.txt'
        ], cwd=tmpdir)
        # Should either succeed with fallback to input.txt or fail
        # The warning should be present
        if rc == 0:
            # Succeeded with fallback
            assert 'Warning' in err or 'matched no files' in err, f"Expected warning:\n{err}"
        else:
            # Failed (no input.txt to fall back to)
            assert 'Error' in err or 'downloading' in err or 'No such file' in err, f"Expected error:\n{err}"
        print("  PASS: glob pattern with no matches handling")


def test_nonexistent_dir_error():
    """Verify error handling for nonexistent directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        train_script = os.path.join(script_dir, 'train.py')
        
        rc, out, err = run([
            sys.executable, train_script, '--num-steps', '3',
            '--data-dir', '/nonexistent/dir'
        ], cwd=tmpdir)
        assert rc != 0, "train.py should fail with nonexistent directory"
        assert 'not a directory' in err or 'Error' in err, f"Expected error message:\n{err}"
        print("  PASS: nonexistent directory error handling")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    tests = [
        ('Backward compatibility', test_backward_compatibility),
        ('Single --data file', test_single_data_file),
        ('Multiple --data files', test_multiple_data_files),
        ('Glob pattern', test_glob_pattern),
        ('--data-dir', test_data_dir),
        ('Combined --data and --data-dir', test_combined_data_and_data_dir),
        ('train_fast.py compatibility', test_train_fast_compatibility),
        ('Glob pattern with no matches', test_nonexistent_file_error),
        ('Nonexistent dir error', test_nonexistent_dir_error),
    ]
    
    passed, failed = 0, 0
    failures = []
    
    print(f"\nRunning {len(tests)} dataset loading tests...\n")
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
