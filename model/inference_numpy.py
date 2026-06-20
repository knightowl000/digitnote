"""
Pure NumPy CNN inference for DigitNote.

Implements the full DigitCNN forward pass using only NumPy — no PyTorch,
no ONNX Runtime. This enables Android deployment via python-for-android
(which has a NumPy recipe but no onnxruntime recipe).

Architecture (DigitCNN):
    conv1: Conv2d(1,32,k=5,p=2) → BN → ReLU → MaxPool(2)  # 28→14
    conv2: Conv2d(32,64,k=5,p=2) → BN → ReLU → MaxPool(2)  # 14→7
    conv3: Conv2d(64,128,k=3,p=1) → BN → ReLU             # 7→7
    fc1: Linear(128*7*7, 256) → ReLU
    fc2: Linear(256, 10) → softmax

Accuracy: 99.71% on MNIST (same weights as ONNX model)
"""

import os
import numpy as np


class DigitCNNNumpy:
    """Pure NumPy implementation of DigitCNN for inference only."""

    def __init__(self, weights_path: str):
        """
        Load model weights from .npz file.

        Args:
            weights_path: Path to digit_full_weights.npz
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        data = np.load(weights_path)
        self.w = {}  # weights dict
        for key in data.files:
            self.w[key] = data[key]

        # Verify expected architecture
        assert self.w['conv1.weight'].shape == (32, 1, 5, 5)
        assert self.w['conv2.weight'].shape == (64, 32, 5, 5)
        assert self.w['conv3.weight'].shape == (128, 64, 3, 3)
        assert self.w['fc1.weight'].shape == (256, 6272)
        assert self.w['fc2.weight'].shape == (10, 256)

    # ================================================================
    # Layer implementations
    # ================================================================

    @staticmethod
    def _conv2d(x, weight, bias, stride=1, padding=0):
        """
        Conv2d via im2col + matrix multiply.

        Args:
            x:      (N, C_in, H, W)  float32
            weight: (C_out, C_in, kH, kW)  float32
            bias:   (C_out,)  float32
            stride: int
            padding: int

        Returns:
            out: (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = weight.shape

        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                       mode='constant', constant_values=0)

        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1

        # im2col via as_strided
        shape = (N, C_in, kH, kW, H_out, W_out)
        strides = (x.strides[0], x.strides[1],
                   x.strides[2], x.strides[3],
                   x.strides[2] * stride,
                   x.strides[3] * stride)
        cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides,
                                               writeable=False)
        # Reshape to (N * H_out * W_out, C_in * kH * kW)
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(-1, C_in * kH * kW)

        # Matrix multiply: (M, C_in*kH*kW) @ (C_in*kH*kW, C_out) → (M, C_out)
        w_flat = weight.reshape(C_out, -1)
        out = cols @ w_flat.T + bias
        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)

        return out.astype(np.float32)

    @staticmethod
    def _batch_norm(x, weight, bias, running_mean, running_var, eps=1e-5):
        """
        BatchNorm2d in inference mode.

        y = gamma * (x - mean) / sqrt(var + eps) + beta
        gamma = weight, beta = bias
        """
        gamma = weight.reshape(1, -1, 1, 1)
        beta = bias.reshape(1, -1, 1, 1)
        mean = running_mean.reshape(1, -1, 1, 1)
        var = running_var.reshape(1, -1, 1, 1)
        return (gamma * (x - mean) / np.sqrt(var + eps) + beta).astype(np.float32)

    @staticmethod
    def _max_pool2d(x, kernel_size=2, stride=2):
        """MaxPool2d via as_strided + max."""
        N, C, H, W = x.shape
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        shape = (N, C, kernel_size, kernel_size, H_out, W_out)
        strides = (x.strides[0], x.strides[1],
                   x.strides[2], x.strides[3],
                   x.strides[2] * stride,
                   x.strides[3] * stride)
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides,
                                                  writeable=False)
        return patches.max(axis=(2, 3)).astype(np.float32)

    @staticmethod
    def _linear(x, weight, bias):
        """Fully connected layer: y = x @ W.T + b."""
        return (x @ weight.T + bias).astype(np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0, dtype=np.float32)

    @staticmethod
    def _softmax(x):
        """Numerically stable softmax along last axis."""
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    # ================================================================
    # Forward pass
    # ================================================================

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run full forward pass.

        Args:
            x: Input image, shape (1, 1, 28, 28), float32, range [0, 1]

        Returns:
            probs: Class probabilities, shape (10,), float32
        """
        # Block 1
        x = self._conv2d(x, self.w['conv1.weight'], self.w['conv1.bias'],
                         stride=1, padding=2)
        x = self._batch_norm(x, self.w['bn1.weight'], self.w['bn1.bias'],
                             self.w['bn1.running_mean'], self.w['bn1.running_var'])
        x = self._relu(x)
        x = self._max_pool2d(x)  # 28→14

        # Block 2
        x = self._conv2d(x, self.w['conv2.weight'], self.w['conv2.bias'],
                         stride=1, padding=2)
        x = self._batch_norm(x, self.w['bn2.weight'], self.w['bn2.bias'],
                             self.w['bn2.running_mean'], self.w['bn2.running_var'])
        x = self._relu(x)
        x = self._max_pool2d(x)  # 14→7

        # Block 3
        x = self._conv2d(x, self.w['conv3.weight'], self.w['conv3.bias'],
                         stride=1, padding=1)
        x = self._batch_norm(x, self.w['bn3.weight'], self.w['bn3.bias'],
                             self.w['bn3.running_mean'], self.w['bn3.running_var'])
        x = self._relu(x)  # 7→7 (no pooling)

        # Classifier
        x = x.reshape(1, -1)  # flatten → (1, 128*7*7) = (1, 6272)
        x = self._linear(x, self.w['fc1.weight'], self.w['fc1.bias'])
        x = self._relu(x)
        x = self._linear(x, self.w['fc2.weight'], self.w['fc2.bias'])

        return self._softmax(x)[0]  # (10,)

    def predict(self, input_array: np.ndarray) -> tuple:
        """
        Predict digit and confidence from preprocessed input.

        Args:
            input_array: shape (1, 1, 28, 28), float32, range [0, 1]

        Returns:
            tuple: (predicted_digit: int, confidence: float)
        """
        if input_array.shape != (1, 1, 28, 28):
            raise ValueError(
                f"Expected shape (1,1,28,28), got {input_array.shape}"
            )

        probs = self.forward(input_array)
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        return predicted, confidence
