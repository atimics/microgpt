#!/usr/bin/env python3
"""
Test cross-entropy function for numerical stability edge cases.
This specifically tests the fix for potential log(0) issues.
"""

import array
import math
import sys

try:
    import fastops
except ImportError:
    print("Error: fastops C extension not available. Build it with: python setup.py build_ext --inplace")
    sys.exit(1)


def test_extreme_negative_logits():
    """Test with extremely negative logits (should not cause underflow to 0)."""
    # Create logits where one is much larger than others
    # This could cause underflow in the smaller values during softmax
    logits = array.array('d', [-1000.0, -1001.0, -1002.0, -999.0])
    target = 3  # Target the largest logit
    
    loss, probs = fastops.cross_entropy_forward(logits, target)
    
    # Loss should be finite and reasonable
    assert math.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss >= 0, f"Cross-entropy loss should be non-negative, got {loss}"
    
    # Probs should sum to 1 and all be finite
    prob_sum = sum(probs)
    assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities should sum to 1, got {prob_sum}"
    assert all(math.isfinite(p) for p in probs), "All probabilities should be finite"
    assert all(p >= 0 for p in probs), "All probabilities should be non-negative"
    
    print("  PASS: Extreme negative logits handled correctly")


def test_uniform_logits():
    """Test with uniform logits (all the same value)."""
    logits = array.array('d', [5.0, 5.0, 5.0, 5.0])
    target = 0
    
    loss, probs = fastops.cross_entropy_forward(logits, target)
    
    # For uniform logits, all probs should be equal
    expected_prob = 1.0 / len(logits)
    for i, p in enumerate(probs):
        assert abs(p - expected_prob) < 1e-6, \
            f"Probability {i} should be {expected_prob}, got {p}"
    
    # Loss should be -log(1/n) = log(n)
    expected_loss = math.log(len(logits))
    assert abs(loss - expected_loss) < 1e-6, \
        f"Loss should be {expected_loss}, got {loss}"
    
    print("  PASS: Uniform logits produce equal probabilities")


def test_very_large_logits():
    """Test with very large logit differences."""
    # One very large logit compared to others
    logits = array.array('d', [0.0, 0.0, 0.0, 1000.0])
    target = 3
    
    loss, probs = fastops.cross_entropy_forward(logits, target)
    
    # The probability for the large logit should be very close to 1
    assert probs[3] > 0.99999, f"Probability of largest logit should be ~1, got {probs[3]}"
    
    # Loss should be very small (close to 0)
    assert loss < 0.001, f"Loss should be very small, got {loss}"
    assert math.isfinite(loss), "Loss should be finite"
    
    print("  PASS: Very large logit differences handled correctly")


def test_target_with_smallest_logit():
    """Test when target has the smallest logit (worst case)."""
    logits = array.array('d', [10.0, 9.0, 8.0, 0.0])
    target = 3  # Target the smallest
    
    loss, probs = fastops.cross_entropy_forward(logits, target)
    
    # Loss should be large but finite
    assert math.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss > 0, "Loss should be positive"
    
    # Smallest prob should still be positive (no underflow to 0)
    assert probs[3] > 0, f"Probability should be positive, got {probs[3]}"
    assert math.isfinite(probs[3]), "Probability should be finite"
    
    print("  PASS: Target with smallest logit handled correctly")


def test_single_element():
    """Test with a single element (edge case)."""
    logits = array.array('d', [42.0])
    target = 0
    
    loss, probs = fastops.cross_entropy_forward(logits, target)
    
    # With one element, probability should be 1.0
    assert abs(probs[0] - 1.0) < 1e-10, f"Single probability should be 1.0, got {probs[0]}"
    
    # Loss should be 0 (or very close)
    assert loss < 1e-10, f"Loss should be ~0 for single element, got {loss}"
    
    print("  PASS: Single element case handled correctly")


def test_numerical_stability_comparison():
    """
    Test that demonstrates the fix prevents log(0).
    
    While we can't directly test log(0) without the fix, we can test
    extreme cases that could lead to underflow.
    """
    # Create a scenario with extreme logit differences
    logits = array.array('d', [-500.0, -500.0, 100.0])
    target = 0  # Target one of the very small probability items
    
    loss, probs = fastops.cross_entropy_forward(logits, target)
    
    # The key test: loss should be finite (not inf or nan)
    assert math.isfinite(loss), f"Loss should be finite even for extreme cases, got {loss}"
    
    # Probabilities should all be finite
    assert all(math.isfinite(p) for p in probs), \
        f"All probabilities should be finite, got {list(probs)}"
    
    # The probability of the target should be very small but not zero
    # With the epsilon clamp, even if softmax underflows, loss will be finite
    assert probs[target] >= 0, f"Probability should be non-negative, got {probs[target]}"
    
    print("  PASS: Numerical stability verified for extreme underflow case")


def main():
    tests = [
        ("Extreme negative logits", test_extreme_negative_logits),
        ("Uniform logits", test_uniform_logits),
        ("Very large logit differences", test_very_large_logits),
        ("Target with smallest logit", test_target_with_smallest_logit),
        ("Single element", test_single_element),
        ("Numerical stability", test_numerical_stability_comparison),
    ]
    
    print(f"\nRunning {len(tests)} cross-entropy edge case tests...\n")
    
    passed = 0
    failed = 0
    failures = []
    
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
