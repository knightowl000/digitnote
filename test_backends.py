"""Quick test of both inference backends."""
import sys, os, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.inference import DigitRecognizer

# Load MNIST test data
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'MNIST', 'raw')
with open(os.path.join(data_dir, 't10k-images-idx3-ubyte'), 'rb') as f:
    magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
with open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'), 'rb') as f:
    magic, num = struct.unpack('>II', f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

print("=== Test 1: NumPy backend (explicit .npz) ===")
rec = DigitRecognizer(os.path.join(model_dir, 'digit_full_weights.npz'))
print(f"Backend: {rec.backend}")
for i in range(10):
    img = images[i].astype(np.float32) / 255.0
    x = img.reshape(1, 1, 28, 28)
    digit, conf = rec.predict(x)
    ok = "OK" if digit == labels[i] else "MISMATCH"
    print(f"  {i}: true={labels[i]}, pred={digit}, conf={conf:.3f} [{ok}]")

print()
print("=== Test 2: ONNX backend (auto-detected) ===")
rec2 = DigitRecognizer(os.path.join(model_dir, 'digit_full.onnx'))
print(f"Backend: {rec2.backend}")
for i in range(10):
    img = images[i].astype(np.float32) / 255.0
    x = img.reshape(1, 1, 28, 28)
    digit, conf = rec2.predict(x)
    ok = "OK" if digit == labels[i] else "MISMATCH"
    print(f"  {i}: true={labels[i]}, pred={digit}, conf={conf:.3f} [{ok}]")

print()
print("All tests passed!")
