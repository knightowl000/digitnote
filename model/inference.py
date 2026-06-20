"""
Inference wrapper — ONNX Runtime with NumPy fallback.

Priority:
    1. ONNX Runtime (fast C++ backend) — desktop, GPU-capable platforms
    2. Pure NumPy (zero extra deps) — Android, platforms without onnxruntime

Both backends share the same DigitRecognizer API:
    recognizer = DigitRecognizer(model_path)   # .onnx or .npz
    digit, confidence = recognizer.predict(input_array)
"""

import os
import numpy as np

# Lazy imports for onnxruntime
_ort_available = False
try:
    import onnxruntime as ort
    _ort_available = True
except ImportError:
    pass


class DigitRecognizer:
    """
    Handwritten digit recognizer.

    Automatically selects the best available backend:
      - ONNX Runtime (preferred, requires onnxruntime package)
      - Pure NumPy (fallback, works everywhere, requires numpy + .npz weights)

    Usage:
        # Auto-detect backend
        rec = DigitRecognizer("model/digit_full.onnx")

        # Force NumPy backend
        rec = DigitRecognizer("model/digit_full_weights.npz")

        digit, conf = rec.predict(input_array)  # input: (1,1,28,28) float32
    """

    def __init__(self, model_path: str):
        """
        Initialize recognizer.

        Args:
            model_path: Path to .onnx model (ONNX backend) or .npz weights (NumPy backend).

        Raises:
            FileNotFoundError: Model file not found.
            ImportError: Neither onnxruntime nor the NumPy fallback weights are available.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backend = None
        self._ort_session = None
        self._np_model = None

        ext = os.path.splitext(model_path)[1].lower()

        if ext == '.npz':
            # Explicit NumPy backend
            self._init_numpy(model_path)

        elif ext == '.onnx':
            if _ort_available:
                self._init_onnx(model_path)
            else:
                # Try NumPy fallback by substituting .onnx → _weights.npz
                npz_path = model_path.replace('.onnx', '_weights.npz')
                if os.path.exists(npz_path):
                    print("[DigitRecognizer] ONNX Runtime not available, "
                          f"falling back to NumPy backend")
                    self._init_numpy(npz_path)
                else:
                    raise ImportError(
                        "onnxruntime is not installed, and no NumPy weights "
                        f"found at {npz_path}. Install onnxruntime or run "
                        "model/export_weights.py first."
                    )
        else:
            raise ValueError(f"Unknown model format: {ext} (expected .onnx or .npz)")

    def _init_onnx(self, model_path: str):
        """Initialize ONNX Runtime backend."""
        self._ort_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self._input_name = self._ort_session.get_inputs()[0].name
        self._output_name = self._ort_session.get_outputs()[0].name
        self._input_shape = self._ort_session.get_inputs()[0].shape
        self.backend = 'onnx'
        print(f"[DigitRecognizer] Backend: ONNX Runtime | Input: "
              f"{self._input_name} {self._input_shape}")

    def _init_numpy(self, weights_path: str):
        """Initialize pure NumPy backend."""
        # Import here — avoids loading the module unless needed
        from model.inference_numpy import DigitCNNNumpy
        self._np_model = DigitCNNNumpy(weights_path)
        self.backend = 'numpy'
        print("[DigitRecognizer] Backend: NumPy (pure Python, no C deps)")

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def predict(self, input_array: np.ndarray) -> tuple:
        """
        Predict digit and confidence.

        Args:
            input_array: Preprocessed image, shape (1, 1, 28, 28), dtype float32.

        Returns:
            tuple: (predicted_digit: int, confidence: float)
        """
        if input_array.shape != (1, 1, 28, 28):
            raise ValueError(
                f"Expected input shape (1, 1, 28, 28), got {input_array.shape}"
            )

        if self.backend == 'onnx':
            return self._predict_onnx(input_array)
        else:
            return self._predict_numpy(input_array)

    def predict_top_k(self, input_array: np.ndarray, k: int = 3) -> list:
        """
        Return top-K predictions.

        Args:
            input_array: shape (1, 1, 28, 28), float32.
            k: number of top results.

        Returns:
            list of (digit, confidence) tuples, descending by confidence.
        """
        if self.backend == 'onnx':
            return self._predict_top_k_onnx(input_array, k)
        else:
            return self._predict_top_k_numpy(input_array, k)

    # ----------------------------------------------------------------
    # Backend implementations
    # ----------------------------------------------------------------

    def _predict_onnx(self, input_array: np.ndarray) -> tuple:
        outputs = self._ort_session.run(
            [self._output_name],
            {self._input_name: input_array}
        )
        logits = outputs[0]
        probs = self._softmax(logits[0])
        digit = int(np.argmax(probs))
        conf = float(probs[digit])
        return digit, conf

    def _predict_top_k_onnx(self, input_array: np.ndarray, k: int) -> list:
        outputs = self._ort_session.run(
            [self._output_name],
            {self._input_name: input_array}
        )
        probs = self._softmax(outputs[0][0])
        top_idx = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top_idx]

    def _predict_numpy(self, input_array: np.ndarray) -> tuple:
        return self._np_model.predict(input_array)

    def _predict_top_k_numpy(self, input_array: np.ndarray, k: int) -> list:
        probs = self._np_model.forward(input_array)
        top_idx = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top_idx]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
