import numpy as np
# from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


class TimelineRenderer:
    """Draws the colored label bar and tick marks."""

    def __init__(self, timeline_label: QLabel, tick_label: QLabel):
        self.timeline_label = timeline_label
        self.tick_label = tick_label

    def render(self, annotator, color_for_mode, format_frame_display, timeline_divisions: int):
        if not annotator.cap:
            self.timeline_label.clear()
            self.tick_label.clear()
            return

        width = self.timeline_label.width()
        height = self.timeline_label.height()
        if width <= 1 or height <= 1:
            return

        total_frames = annotator.frame_count
        if total_frames <= 0:
            return

        pixels_per_frame = width / total_frames

        def color_for_frame(idx: int) -> str:
            lab = annotator.annotations.get(idx)
            if lab:
                mode = lab.get("mode")
                if mode:
                    return color_for_mode(mode)
            return "#AAAAAA"

        segments: list[tuple[int, int, str]] = []
        current_start = 0
        current_color = color_for_frame(0)
        for i in range(1, total_frames):
            col = color_for_frame(i)
            if col != current_color:
                segments.append((current_start, i, current_color))
                current_start = i
                current_color = col
        segments.append((current_start, total_frames, current_color))

        bar = np.full((height, width, 3), 220, dtype=np.uint8)

        for start, end, color_hex in segments:
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)

            x1 = int(start * pixels_per_frame)
            x2 = max(x1 + 1, int(end * pixels_per_frame))
            bar[:, x1:x2] = (r, g, b)

        stride = bar.strides[0]

        qimg = QImage(bar.data, width, height, stride,
                      QImage.Format.Format_RGB888)
        self.timeline_label.setPixmap(QPixmap.fromImage(qimg))
        self._draw_ticks(annotator, format_frame_display, timeline_divisions)

    def _draw_ticks(self, annotator, format_frame_display, timeline_divisions: int):
        if not annotator.cap:
            self.tick_label.clear()
            return

        width = self.tick_label.width()
        height = self.tick_label.height()
        if width <= 10 or height <= 5:
            return

        img = QImage(width, height, QImage.Format.Format_RGB888)
        img.fill(QColor("#1b1e22"))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        tick_pen = QPen(QColor("#59606a"))
        tick_pen.setWidth(1)
        painter.setPen(tick_pen)

        text_pen = QPen(QColor("#dfe3e8"))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        total_frames = max(1, annotator.frame_count - 1)
        sections = max(1, int(timeline_divisions))
        for i in range(sections + 1):
            x = int(i * (width - 1) / sections)
            painter.drawLine(x, 0, x, height // 2)

            frame_at_tick = int(total_frames * (i / sections))
            label = format_frame_display(frame_at_tick)
            painter.setPen(text_pen)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label)
            text_h = metrics.height()
            tx = max(0, min(width - text_w, x - text_w // 2))
            ty = height - 4
            painter.drawText(tx, ty, label)
            painter.setPen(tick_pen)

        painter.end()
        self.tick_label.setPixmap(QPixmap.fromImage(img))
