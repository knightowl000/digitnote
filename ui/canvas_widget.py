"""
手写画布控件
基于 Kivy Widget，支持鼠标和触屏绘制，可导出为图像。
"""

import os
import tempfile

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line, Rectangle, Ellipse
from kivy.properties import ListProperty, NumericProperty
from kivy.core.window import Window


class HandwritingCanvas(Widget):
    """
    手写画布控件。

    特性：
    - 支持鼠标（按下拖动）和触屏（单指）绘制
    - 白色背景，黑色笔画
    - 笔画宽度可配置
    - 支持清空画布
    - 支持导出为 PNG 图像
    """

    # 笔画颜色 (R, G, B, A) — 默认黑色
    stroke_color = ListProperty([0, 0, 0, 1])

    # 笔画宽度
    stroke_width = NumericProperty(15)

    def __init__(self, stroke_end_callback=None, **kwargs):
        super().__init__(**kwargs)

        # 绑定尺寸变化以重绘背景
        self.bind(size=self._on_size)
        self.bind(pos=self._on_size)

        # 追踪当前笔画
        self._current_line = None
        self._current_ellipse = None
        self._has_strokes = False  # track if user has drawn anything

        # Auto-recognition: callback called when a stroke ends
        self._stroke_end_callback = stroke_end_callback

        # 初始化背景
        self._draw_background()

    def _draw_background(self):
        """绘制白色背景"""
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 1)  # 白色
            self._bg_rect = Rectangle(
                pos=self.pos,
                size=self.size
            )

    def _on_size(self, instance, value):
        """尺寸变化时更新背景大小"""
        if hasattr(self, '_bg_rect'):
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size

    def on_touch_down(self, touch):
        """触屏/鼠标按下 — 开始新笔画"""
        if self.collide_point(*touch.pos):
            self._has_strokes = True
            # 使用椭圆笔触使笔画更平滑（尤其在拐角处）
            with self.canvas:
                Color(*self.stroke_color)
                self._current_ellipse = Ellipse(
                    pos=(touch.x - self.stroke_width / 2,
                         touch.y - self.stroke_width / 2),
                    size=(self.stroke_width, self.stroke_width)
                )
                self._current_line = Line(
                    points=[touch.x, touch.y],
                    width=self.stroke_width,
                    joint='round',
                    cap='round'
                )
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """触屏/鼠标移动 — 继续绘制"""
        if self.collide_point(*touch.pos) and self._current_line:
            # 更新线条点
            self._current_line.points += [touch.x, touch.y]

            # 更新端点椭圆（让笔画末端圆润）
            if self._current_ellipse:
                self._current_ellipse.pos = (
                    touch.x - self.stroke_width / 2,
                    touch.y - self.stroke_width / 2
                )
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """触屏/鼠标抬起 — 结束当前笔画"""
        if self._current_line is not None:
            self._current_line = None
            # 移除最后一个椭圆
            if self._current_ellipse:
                self.canvas.remove(self._current_ellipse)
                self._current_ellipse = None
            # Notify that a stroke ended (for auto-recognition)
            if self._stroke_end_callback:
                self._stroke_end_callback()
        return super().on_touch_up(touch)

    def clear_canvas(self):
        """清空画布，仅保留背景"""
        # 清除所有绘制内容（保留 canvas.before 中的背景）
        self.canvas.clear()

        # 重新绑定背景（canvas.before 在 clear 后也被清除了）
        self._draw_background()

        # 重置状态
        self._current_line = None
        self._current_ellipse = None
        self._has_strokes = False

    def export_to_image(self, filepath: str = None) -> str:
        """
        将当前画布内容导出为 PNG 图像。

        Args:
            filepath: 可选的保存路径。若不指定，使用临时文件。

        Returns:
            str: 导出的 PNG 文件路径
        """
        if filepath is None:
            fd, filepath = tempfile.mkstemp(suffix='.png', prefix='digitnote_')
            os.close(fd)

        # Kivy Widget 的 export_to_png 方法
        self.export_to_png(filepath)
        return filepath

    def is_empty(self) -> bool:
        """检查画布是否为空（只有背景，无笔画）"""
        return not self._has_strokes
