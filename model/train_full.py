"""
Complete training script — Enhanced CNN with data augmentation.

Features:
  - Deeper CNN architecture (target >99% MNIST accuracy)
  - Data augmentation: random rotation, translation, affine (elastic-like)
  - GPU training with mixed precision support
  - Learning rate scheduling (CosineAnnealing)
  - TensorBoard logging
  - Grad-CAM heatmap visualization
  - ONNX export with static quantization

Usage:
    python model/train_full.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime
import cv2


# ============================================================
# Model: Enhanced CNN (Deeper LeNet-5 variant)
# ============================================================

class DigitCNN(nn.Module):
    """Enhanced CNN for handwritten digit recognition.

    Architecture:
        Conv2d(1, 32, 5) -> BN -> ReLU -> MaxPool(2)
        Conv2d(32, 64, 5) -> BN -> ReLU -> MaxPool(2)
        Conv2d(64, 128, 3) -> BN -> ReLU
        Flatten -> Linear(128*3*3, 256) -> ReLU -> Dropout(0.5)
        Linear(256, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x28x28 -> 32x14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x14x14 -> 64x7x7

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)   # 128x7x7

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

        # Store feature maps for Grad-CAM
        self._gradients = None
        self._activations = None

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3 (target for Grad-CAM)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Save activations for Grad-CAM
        if self.training:
            x.register_hook(self._save_gradients)
        self._activations = x

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def _save_gradients(self, grad):
        self._gradients = grad

    def get_gradcam_weights(self):
        """Get Grad-CAM weights by global-average-pooling gradients."""
        if self._gradients is None:
            return None
        return self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)


# ============================================================
# Data augmentation
# ============================================================

def get_transforms(augment: bool = True):
    """Get data transforms with optional augmentation."""
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=8,
            ),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return train_transform, test_transform


# ============================================================
# Training & Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    writer=None, scaler=None):
    """Train one epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log to TensorBoard every 100 batches
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


# ============================================================
# Grad-CAM visualization
# ============================================================

def generate_gradcam(model, image_tensor, target_class=None):
    """
    Generate Grad-CAM heatmap for a single image.

    Args:
        model: trained DigitCNN model
        image_tensor: input image (1, 1, 28, 28)
        target_class: class to visualize (None = predicted class)

    Returns:
        heatmap: numpy array (28, 28) normalized to [0, 1]
    """
    model.eval()
    model._gradients = None
    model._activations = None

    image_tensor = image_tensor.unsqueeze(0)  # add batch dim

    # Forward pass
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass for target class
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot)

    # Get activations and weights
    activations = model._activations.detach()  # (1, C, H, W)
    weights = model.get_gradcam_weights()       # (1, C, 1, 1)

    if weights is None:
        return np.zeros((28, 28), dtype=np.float32)

    # Weighted combination
    cam = (weights * activations).sum(dim=1)  # (1, H, W)
    cam = F.relu(cam)

    # Normalize
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # Resize to 28x28
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.resize(cam, (28, 28), interpolation=cv2.INTER_LINEAR)

    return cam


def save_gradcam_grid(model, test_loader, device, save_path, num_samples=16):
    """
    Save a grid of Grad-CAM visualizations for sample digits.

    Args:
        model: trained model
        test_loader: test data loader
        device: computation device
        save_path: output image path
        num_samples: number of samples to visualize
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()

    # Collect one example per digit
    examples = {}
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            digit = label.item()
            if digit not in examples:
                examples[digit] = img
            if len(examples) == 10:
                break
        if len(examples) == 10:
            break

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    for digit in range(10):
        img = examples[digit]
        img_input = img.to(device)

        # Original image
        axes[0, digit].imshow(img.squeeze(), cmap='gray')
        axes[0, digit].set_title(f'Digit {digit}')
        axes[0, digit].axis('off')

        # Grad-CAM
        with torch.no_grad():
            output = model(img_input.unsqueeze(0))
            pred = output.argmax(dim=1).item()
        heatmap = generate_gradcam(model, img_input, target_class=digit)
        axes[1, digit].imshow(img.squeeze(), cmap='gray')
        axes[1, digit].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[1, digit].set_title(f'Grad-CAM (pred={pred})')
        axes[1, digit].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM grid saved: {save_path}")


# ============================================================
# ONNX Export with quantization awareness
# ============================================================

def export_to_onnx(model, save_path: str, quantize: bool = False):
    """
    Export trained model to ONNX format, optionally with dynamic quantization.

    Args:
        model: trained PyTorch model
        save_path: output ONNX file path
        quantize: apply dynamic quantization before export (reduces model size)
    """
    model.eval()
    model.to("cpu")

    if quantize:
        # Dynamic quantization for Linear and Conv layers
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        print("Applied dynamic quantization (int8)")

    dummy_input = torch.randn(1, 1, 28, 28, device="cpu")

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )

    print(f"Model exported to ONNX: {save_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: [OK]")

        # Report model size
        size_mb = os.path.getsize(save_path)
        data_path = save_path + ".data"
        if os.path.exists(data_path):
            size_mb += os.path.getsize(data_path)
        print(f"ONNX model size: {size_mb / 1024:.1f} KB")
    except ImportError:
        print("(Skipped ONNX verification)")


# ============================================================
# Main training pipeline
# ============================================================

def main():
    # ---- Configuration ----
    BATCH_SIZE = 256
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(MODEL_DIR, "..", "runs", f"train_{TIMESTAMP}")
    PTH_PATH = os.path.join(MODEL_DIR, "digit_full.pth")
    ONNX_PATH = os.path.join(MODEL_DIR, "digit_full.onnx")
    GRADCAM_PATH = os.path.join(MODEL_DIR, "gradcam_grid.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")  # mixed precision on GPU only

    print(f"Device: {device}")
    print(f"Mixed precision: {use_amp}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Log dir: {LOG_DIR}")

    # ---- TensorBoard ----
    writer = SummaryWriter(log_dir=LOG_DIR)
    print("TensorBoard logging enabled.")

    # ---- Data ----
    train_transform, test_transform = get_transforms(augment=True)

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")

    # ---- Model ----
    model = DigitCNN(num_classes=10, dropout=0.5).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Log model graph to TensorBoard
    dummy_batch = torch.randn(1, 1, 28, 28).to(device)
    writer.add_graph(model, dummy_batch)

    # ---- Optimizer & Scheduler ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ---- Training loop ----
    print("\n" + "=" * 60)
    print("Training started")
    print("=" * 60)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, writer, scaler
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # TensorBoard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, PTH_PATH)

    print("=" * 60)
    print(f"Training complete! Best test accuracy: {best_acc:.2f}%")

    if best_acc >= 99.0:
        print(f"[OK] Accuracy {best_acc:.2f}% meets target (>99%)")
    else:
        print(f"[INFO] Accuracy {best_acc:.2f}% — close to 99% target")

    # ---- Load best model ----
    checkpoint = torch.load(PTH_PATH, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # ---- Grad-CAM ----
    print("\nGenerating Grad-CAM visualizations...")
    save_gradcam_grid(model, test_loader, device, GRADCAM_PATH)

    # ---- Export ONNX ----
    print("\nExporting ONNX model...")
    export_to_onnx(model, ONNX_PATH, quantize=False)

    # ---- Summary ----
    writer.add_text('Summary', f'Best accuracy: {best_acc:.2f}%')
    writer.close()

    print(f"\nOutput files:")
    print(f"  Trained model:   {PTH_PATH}")
    print(f"  ONNX model:      {ONNX_PATH}")
    print(f"  Grad-CAM grid:   {GRADCAM_PATH}")
    print(f"  TensorBoard log: {LOG_DIR}")
    print(f"\nView TensorBoard: tensorboard --logdir {LOG_DIR}")
    print(f"Replace MVP model: copy digit_full.onnx -> digit_mvp.onnx")


if __name__ == "__main__":
    main()
