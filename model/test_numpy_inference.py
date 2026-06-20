"""
Test NumPy inference accuracy vs ONNX model on MNIST test set.

Usage:
    python model/test_numpy_inference.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from inference_numpy import DigitCNNNumpy

# Try importing ONNX Runtime for comparison
try:
    from inference import DigitRecognizer as ONNXRecognizer
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def load_mnist_test():
    """Load MNIST test set from raw files."""
    import struct
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "MNIST", "raw")

    with open(os.path.join(data_dir, 't10k-images-idx3-ubyte'), 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'), 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def main():
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(MODEL_DIR, "digit_full_weights.npz")
    onnx_path = os.path.join(MODEL_DIR, "digit_full.onnx")

    print("=" * 60)
    print("NumPy CNN Inference — Accuracy Test")
    print("=" * 60)

    # Load NumPy model
    print(f"\nLoading NumPy model: {weights_path}")
    np_model = DigitCNNNumpy(weights_path)
    print("NumPy model loaded.")

    # Load ONNX model for comparison
    onnx_model = None
    if HAS_ONNX and os.path.exists(onnx_path):
        print(f"Loading ONNX model: {onnx_path}")
        onnx_model = ONNXRecognizer(onnx_path)
        print("ONNX model loaded.")
    else:
        print("ONNX model not available — skipping comparison.")

    # Load test data
    print("\nLoading MNIST test set...")
    images, labels = load_mnist_test()
    print(f"Loaded {len(images)} test images.")

    # Benchmark
    print("\n--- NumPy Inference Benchmark ---")
    np_correct = 0
    np_times = []
    onnx_correct = 0
    onnx_times = []
    mismatches = 0

    # Test on subset for speed (full 10K for accuracy check)
    n_test = min(10000, len(images))
    for i in range(n_test):
        # Preprocess: normalize to [0,1] and reshape
        img = images[i].astype(np.float32) / 255.0
        x = img.reshape(1, 1, 28, 28)
        label = int(labels[i])

        # NumPy inference
        t0 = time.perf_counter()
        np_pred, np_conf = np_model.predict(x)
        t1 = time.perf_counter()
        np_times.append(t1 - t0)
        if np_pred == label:
            np_correct += 1

        # ONNX inference (first 1000 only — slower)
        if onnx_model and i < 1000:
            t0 = time.perf_counter()
            onnx_pred, onnx_conf = onnx_model.predict(x)
            t1 = time.perf_counter()
            onnx_times.append(t1 - t0)
            if onnx_pred == label:
                onnx_correct += 1

            # Check for NumPy vs ONNX disagreement
            if np_pred != onnx_pred:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  MISMATCH #{mismatches}: image {i}, "
                          f"label={label}, numpy={np_pred}({np_conf:.3f}), "
                          f"onnx={onnx_pred}({onnx_conf:.3f})")

        # Progress
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{n_test} (acc so far: {np_correct/(i+1)*100:.2f}%)")

    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    np_acc = np_correct / n_test * 100
    print(f"NumPy accuracy: {np_correct}/{n_test} = {np_acc:.2f}%")
    print(f"NumPy avg time:  {np.mean(np_times)*1000:.2f} ms ({np.std(np_times)*1000:.2f} ms std)")

    if onnx_model:
        onnx_acc = onnx_correct / 1000 * 100
        print(f"ONNX accuracy:  {onnx_correct}/1000 = {onnx_acc:.2f}%")
        print(f"ONNX avg time:  {np.mean(onnx_times)*1000:.2f} ms")
        print(f"NumPy vs ONNX disagreements: {mismatches}/1000")

    # Verdict
    print()
    if np_acc >= 99.0:
        print("[PASS] NumPy accuracy >= 99% — meets production target!")
    elif np_acc >= 98.0:
        print("[WARN] NumPy accuracy slightly below 99% — check for numerical issues")
    else:
        print("[FAIL] NumPy accuracy below 98% — investigate implementation")

    # Test on a few synthetic samples from each digit class
    print("\n--- Per-digit Accuracy ---")
    per_digit = {d: {'total': 0, 'correct': 0} for d in range(10)}
    for i in range(n_test):
        img = images[i].astype(np.float32) / 255.0
        x = img.reshape(1, 1, 28, 28)
        label = int(labels[i])
        np_pred, _ = np_model.predict(x)
        per_digit[label]['total'] += 1
        if np_pred == label:
            per_digit[label]['correct'] += 1

    for d in range(10):
        total = per_digit[d]['total']
        correct = per_digit[d]['correct']
        acc = correct / total * 100 if total > 0 else 0
        bar = '#' * int(acc / 5)
        print(f"  Digit {d}: {correct:4d}/{total:4d} = {acc:5.2f}% {bar}")


if __name__ == "__main__":
    main()
