"""
MVP 阶段模型训练脚本
快速训练一个简单 CNN 用于手写数字识别，并导出为 ONNX 格式。

模型架构（Mini-LeNet）：
    Conv2d(1, 16, 3) → ReLU → MaxPool(2)
    Conv2d(16, 32, 3) → ReLU → MaxPool(2)
    Flatten → Linear(32*5*5, 128) → ReLU → Dropout(0.5)
    Linear(128, 10) → Softmax

目标：MNIST 准确率 > 95%，CPU 训练时间 < 5 分钟。
"""

import os
import sys

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 模型定义
# ============================================================

class MiniLeNet(nn.Module):
    """简化版 LeNet-5，适合在 CPU 上快速训练。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1×28×28 → 16×12×12
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16×14×14 → 32×5×5
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 训练与评估
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """评估模型"""
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


def export_to_onnx(model, save_path: str, device: str = "cpu"):
    """
    将 PyTorch 模型导出为 ONNX 格式。

    Args:
        model: PyTorch 模型
        save_path: 导出的 ONNX 文件路径
        device: 导出设备
    """
    model.eval()
    model.to("cpu")

    # 创建虚拟输入
    dummy_input = torch.randn(1, 1, 28, 28, device="cpu")

    # 导出（使用较高 opset 版本以兼容新版 PyTorch）
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

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: [OK]")
    except ImportError:
        print("(Skipped ONNX verification: onnx package not installed)")


def main():
    """Main training pipeline"""
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 0.001
    MODEL_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    ONNX_PATH = os.path.join(MODEL_SAVE_DIR, "digit_mvp.onnx")
    PTH_PATH = os.path.join(MODEL_SAVE_DIR, "digit_mvp.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")

    # Data preparation — only ToTensor, NO Normalize
    # This ensures pixel range [0, 1] matches inference preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("\nDownloading / loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device.type == "cuda")
    )

    print(f"Train set: {len(train_dataset)} images, Test set: {len(test_dataset)} images")

    # Model, loss, optimizer
    model = MiniLeNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("\n" + "=" * 50)
    print("Training started")
    print("=" * 50)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), PTH_PATH)

    print("=" * 50)
    print(f"Training complete! Best test accuracy: {best_acc:.2f}%")

    # Load best model and export ONNX
    if best_acc >= 95.0:
        print(f"\n[OK] Accuracy {best_acc:.2f}% meets MVP target (>95%)")
    else:
        print(f"\n[WARN] Accuracy {best_acc:.2f}% below 95%, more epochs may help")

    model.load_state_dict(torch.load(PTH_PATH, weights_only=True))
    export_to_onnx(model, ONNX_PATH)

    print(f"\nOutput files:")
    print(f"  PyTorch model: {PTH_PATH}")
    print(f"  ONNX model:    {ONNX_PATH}")
    print(f"\nRun 'python main.py' to launch DigitNote.")


if __name__ == "__main__":
    main()
