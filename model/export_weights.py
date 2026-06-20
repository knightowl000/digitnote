"""
Export trained DigitCNN weights to NumPy .npz format for on-device inference.

Usage:
    python model/export_weights.py

Output:
    model/digit_full_weights.npz — all conv/bn/fc weights and biases as numpy arrays
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from train_full import DigitCNN


def export_weights(pth_path: str, npz_path: str):
    """Load PyTorch checkpoint and export all weights to .npz."""
    print(f"Loading checkpoint: {pth_path}")
    checkpoint = torch.load(pth_path, weights_only=True, map_location='cpu')
    model = DigitCNN(num_classes=10, dropout=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    weights = {}

    # Extract named parameters
    state = model.state_dict()

    for name, tensor in state.items():
        np_array = tensor.cpu().numpy()
        weights[name] = np_array
        print(f"  {name}: {list(np_array.shape)}")

    # Save
    np.savez_compressed(npz_path, **weights)
    size_kb = os.path.getsize(npz_path) / 1024
    print(f"\nWeights exported to: {npz_path} ({size_kb:.1f} KB)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(MODEL_DIR, "digit_full.pth")
    npz_path = os.path.join(MODEL_DIR, "digit_full_weights.npz")

    if not os.path.exists(pth_path):
        print(f"[ERROR] Checkpoint not found: {pth_path}")
        print("Run train_full.py first.")
        sys.exit(1)

    export_weights(pth_path, npz_path)
