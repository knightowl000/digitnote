# DigitNote — Handwritten Digit Recognition

Cross-platform handwritten digit recognition app built with **Kivy** + **PyTorch** + **ONNX Runtime**. Draw a digit on the canvas (mouse or touch), and the app recognizes it in real time using a deep CNN trained on MNIST.

## Features

| Feature | Description |
|---------|-------------|
| ✍️ **Handwriting Canvas** | Draw digits with mouse or touch (stylus/finger). White background, smooth strokes. |
| 🔢 **Digit Recognition** | CNN model recognizes digits 0–9 with confidence score. Color-coded: green ≥80%, orange ≥50%, red <50%. |
| ⚡ **Auto Mode** | Toggle auto-recognition — recognizes your digit 0.8 seconds after you stop drawing. |
| 📜 **History** | SQLite-backed history with timestamps, confidence scores, and stored images. View, delete individual records, or clear all. |
| 📷 **Image Import** | Import digit photos from your device. Adaptive preprocessing handles varied lighting and backgrounds. |
| 🎓 **GPU Training** | Full training script with data augmentation, TensorBoard logging, Grad-CAM visualization. Achieves **99.71%** MNIST accuracy. |

## Tech Stack

```
UI Framework    —  Kivy 2.3.1 + KivyMD
Deep Learning   —  PyTorch (training) → ONNX (export) → ONNX Runtime (inference)
Image Processing—  OpenCV + Pillow + NumPy
Storage         —  SQLite (via Python built-in sqlite3)
Desktop Packing —  PyInstaller
Mobile Packing  —  Buildozer (Android)
Training Viz    —  TensorBoard + matplotlib (Grad-CAM)
Language        —  Python 3.10+
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Windows, use Python 3.11 or later. The project is tested with Python 3.11.6.

### 2. Train the model (or use pre-trained)

The repository includes pre-trained models (`digit_full.onnx` at 99.71% accuracy). To retrain:

```bash
# Quick MVP model (~98.9%, trains in ~1 min on CPU)
python model/train_mvp.py

# Full model with data augmentation (~99.7%, requires GPU)
python model/train_full.py
```

### 3. Launch the app

```bash
python main.py
```

### 4. Draw and recognize

1. Draw a digit (0–9) on the canvas with your mouse or touch
2. Click **Recognize** to see the prediction and confidence
3. Toggle **Auto: ON** for automatic recognition after you stop drawing
4. Use **Clear** to erase and try again

## Project Structure

```
digitnote/
├── main.py                    # App entry point
├── requirements.txt           # PyPI dependencies
├── digitnote.spec             # PyInstaller spec (desktop packaging)
├── buildozer.spec             # Buildozer spec (Android packaging)
├── README.md                  # This file
├── model/
│   ├── inference.py           # Unified inference wrapper (ONNX + NumPy fallback)
│   ├── inference_numpy.py     # Pure NumPy CNN forward pass (Android-ready)
│   ├── export_weights.py      # PyTorch → NumPy .npz weight exporter
│   ├── train_mvp.py           # MVP model training (Mini-LeNet, CPU)
│   ├── train_full.py          # Full model training (DigitCNN, GPU)
│   ├── digit_full.onnx        # Trained ONNX model (99.71%, 6.6 MB)
│   ├── digit_full_weights.npz # NumPy weights for Android (99.71%, 6.3 MB)
│   ├── digit_full.pth         # PyTorch checkpoint
│   ├── digit_mvp.onnx         # MVP ONNX model (fallback)
│   └── gradcam_grid.png       # Grad-CAM visualization
├── ui/
│   ├── canvas_widget.py       # Drawing canvas widget
│   ├── main_screen.py         # Main UI layout + controls
│   └── history_popup.py       # History viewer popup
└── utils/
    ├── database.py            # SQLite database manager
    └── preprocessing.py       # Image preprocessing pipelines
```

## Desktop Packaging (PyInstaller)

Build a standalone Windows `.exe`:

```bash
cd digitnote
pip install pyinstaller
pyinstaller digitnote.spec
```

Output: `dist/DigitNote/DigitNote.exe` — double-click to run, no Python installation needed.

> **macOS/Linux**: Edit `digitnote.spec` to adjust kivy_deps paths (Linux uses system SDL2, macOS uses `kivy_deps.sdl2` from pip). The spec file includes comments for these cases.

## Android Packaging (Buildozer)

> **Prerequisite:** Buildozer requires a **Linux** environment. Windows users must use **WSL** (Windows Subsystem for Linux).

### Inference: Pure NumPy CNN (No ONNX Runtime!)

DigitNote uses a **pure NumPy implementation** of the CNN forward pass on Android — no ONNX Runtime needed. This was built specifically because `onnxruntime` lacks a python-for-android recipe. The NumPy backend achieves **99.71% accuracy** (identical to the ONNX model) with ~0.8ms inference time.

| Backend | Accuracy | Speed | Dependencies | Android |
|---------|----------|-------|-------------|---------|
| ONNX Runtime | 99.71% | 0.15 ms | onnxruntime | ❌ No p4a recipe |
| **NumPy CNN** | **99.71%** | **0.83 ms** | **numpy only** | **✅ Works!** |

### Setup WSL (Windows)

```powershell
# In PowerShell (Admin), install WSL + Ubuntu if not already installed:
wsl --install -d Ubuntu
```

### Build the APK

```bash
# Enter WSL
wsl

# Install build dependencies
sudo apt update
sudo apt install -y python3-pip openjdk-17-jdk git autoconf libtool zip unzip
pip install buildozer

# Navigate to the project
cd /mnt/c/Users/user/Desktop/code/pytorch/digitnote

# Build debug APK
buildozer android debug
```

Output: `bin/digitnote-*-debug.apk`

### Requirements (all have standard p4a recipes)
```
python3, kivy==2.3.1, Pillow, numpy, opencv
```

All packages are compiled from source by python-for-android's NDK toolchain — no prebuilt wheels needed.

## Model Training Details

### Architectures

| Model | Architecture | Parameters | Accuracy | Training Time |
|-------|-------------|------------|----------|---------------|
| **MVP** | Mini-LeNet (2 conv layers) | 207K | 98.90% | ~1 min (CPU) |
| **Full** | DigitCNN (3 conv blocks) | 1.73M | 99.71% | ~15 min (GPU) |

### Full Model Features
- 3 convolutional blocks: 32→64→128 filters with BatchNorm
- Data augmentation: rotation ±12°, translation, scaling, shear
- CosineAnnealingLR scheduler (LR: 0.001 → 1e-5 over 30 epochs)
- Mixed precision training (AMP) on CUDA
- TensorBoard logging (`tensorboard --logdir runs/`)
- Grad-CAM heatmap visualization for model interpretability

### Run TensorBoard

```bash
tensorboard --logdir runs/
```

### Training Data

MNIST dataset is auto-downloaded by torchvision to `data/MNIST/` on first run (60K train + 10K test images, ~55 MB total).

## Image Preprocessing Pipeline

### Canvas drawings
1. Export canvas to PNG (white background, black strokes)
2. Read as grayscale → invert (match MNIST: white digit on black)
3. Resize to 28×28 with anti-aliasing
4. Normalize to [0, 1]
5. Reshape to (1, 1, 28, 28) for ONNX input

### Imported images (camera/gallery)
1. Read as grayscale → Gaussian blur
2. Adaptive threshold (handles varied lighting)
3. Find contours → extract largest digit region
4. Pad + resize to 20×20, center in 28×28 canvas
5. Normalize to [0, 1]

## License

This project is provided for educational and personal use.

## Acknowledgments

- [Kivy](https://kivy.org/) — cross-platform UI framework
- [PyTorch](https://pytorch.org/) — deep learning framework
- [ONNX Runtime](https://onnxruntime.ai/) — cross-platform inference engine
- [MNIST](http://yann.lecun.com/exdb/mnist/) — handwritten digit dataset
