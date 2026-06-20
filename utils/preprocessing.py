"""
图像预处理模块
负责将手写画布输出转换为 MNIST 格式（28×28 灰度图，白字黑底）
"""

import numpy as np
import cv2
from PIL import Image


def preprocess_canvas_image(image_path: str, invert: bool = True) -> np.ndarray:
    """
    将画布导出的图像预处理为 MNIST 推理输入格式。

    处理流程：
    1. 读取图像并转为灰度
    2. 反色（白字黑底 → 黑底白字，与 MNIST 一致）
    3. 缩放到 28×28
    4. 归一化到 [0, 1]
    5. 添加 batch 和 channel 维度 → (1, 1, 28, 28)

    Args:
        image_path: 画布导出图像的文件路径
        invert: 是否反色。画布通常是黑笔白底，MNIST 是白字黑底，需反色。

    Returns:
        np.ndarray, shape (1, 1, 28, 28), dtype float32, 值域 [0, 1]
    """
    # 1. 读取并转为灰度
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # 2. 反色：黑笔白底 → 白字黑底（MNIST 格式）
    if invert:
        img = 255 - img

    # 3. 缩放到 28×28（使用抗锯齿插值）
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 4. 归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0

    # 5. 添加 channel 和 batch 维度
    img = img.reshape(1, 1, 28, 28)

    return img


def preprocess_imported_image(image_path: str) -> np.ndarray:
    """
    处理用户导入的图片（相册/拍照），适配 MNIST 推理。

    处理流程：
    1. 读取并灰度化
    2. 自适应二值化
    3. 反色（确保数字为白色）
    4. 查找数字轮廓，裁剪到边界框
    5. 保持宽高比缩放到 20×20，居中放入 28×28 画布
    6. 归一化到 [0, 1]

    Args:
        image_path: 用户导入的图片路径

    Returns:
        np.ndarray, shape (1, 1, 28, 28), dtype float32
    """
    # 1. 读取并灰度化
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # 2. 高斯模糊 + 自适应二值化
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 3. 此时数字为白色(255)，背景为黑色(0) — 查找白色区域轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # 无轮廓时，回退到直接缩放
        resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        result = resized.astype(np.float32) / 255.0
        if np.mean(result) > 0.5:
            result = 1.0 - result  # 确保白字黑底
        return result.reshape(1, 1, 28, 28)

    # 4. 找到最大的轮廓（假设是数字）
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 扩展边界框（留一些边距）
    margin = max(w, h) // 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(binary.shape[1] - x, w + 2 * margin)
    h = min(binary.shape[0] - y, h + 2 * margin)

    digit_roi = binary[y:y + h, x:x + w]

    # 5. 保持宽高比缩放到 20×20，居中放入 28×28
    scale = min(20.0 / w, 20.0 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        digit_resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        digit_resized = digit_roi

    # 居中放入 28×28 黑色画布
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - digit_resized.shape[0]) // 2
    x_offset = (28 - digit_resized.shape[1]) // 2
    canvas[
        y_offset:y_offset + digit_resized.shape[0],
        x_offset:x_offset + digit_resized.shape[1]
    ] = digit_resized

    # 6. 归一化
    result = canvas.astype(np.float32) / 255.0
    return result.reshape(1, 1, 28, 28)


def preprocess_from_array(pixels: np.ndarray, invert: bool = True) -> np.ndarray:
    """
    从原始像素数组（如 Kivy 纹理数据）直接预处理。

    Args:
        pixels: 原始像素数组，shape (H, W) 或 (H, W, C)
        invert: 是否反色

    Returns:
        np.ndarray, shape (1, 1, 28, 28), dtype float32
    """
    # 转为灰度（如果是彩色）
    if len(pixels.shape) == 3 and pixels.shape[2] >= 3:
        # 使用加权灰度转换
        gray = (0.299 * pixels[:, :, 0] +
                0.587 * pixels[:, :, 1] +
                0.114 * pixels[:, :, 2])
    elif len(pixels.shape) == 3 and pixels.shape[2] == 1:
        gray = pixels[:, :, 0]
    else:
        gray = pixels

    gray = gray.astype(np.uint8)

    if invert:
        gray = 255 - gray

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, 1, 28, 28)
