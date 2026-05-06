from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


class TimelineRenderer:
    """Draws the colored label bar and tick marks."""

    def __init__(self, timeline_label: QLabel, tick_label: QLabel):
        self.timeline_label = timeline_label
        self.tick_label = tick_label

    def render(
        self,
        annotator,
        color_for_mode,
        format_frame_display,
        timeline_divisions: int,
        selected_section_id: int | None = None,
    ):
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

        img = QImage(width, height, QImage.Format.Format_ARGB32)
        img.fill(QColor("#15171b"))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        section_h = max(34, int(height * 0.58))
        annotation_y = section_h + 5
        annotation_h = max(12, height - annotation_y - 3)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1f2228"))
        painter.drawRoundedRect(0, 4, width - 1, section_h - 6, 6, 6)
        painter.setBrush(QColor("#24272d"))
        painter.drawRoundedRect(0, annotation_y, width - 1, annotation_h, 4, 4)

        painter.setPen(Qt.PenStyle.NoPen)
        for start, end, color_hex in segments:
            x1 = int(start * pixels_per_frame)
            x2 = max(x1 + 1, int(end * pixels_per_frame))
            painter.setBrush(QColor(color_hex))
            painter.drawRect(x1, annotation_y, x2 - x1, annotation_h)

        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        for section in annotator.sections:
            x1 = int(section.start_frame * pixels_per_frame)
            x2 = max(x1 + 4, int((section.end_frame + 1) * pixels_per_frame))
            rect = QRect(x1, 9, min(width - x1 - 1, x2 - x1), max(20, section_h - 14))
            selected = section.section_id == selected_section_id
            fill = QColor("#62c58f" if selected else "#386b55")
            fill.setAlpha(195 if selected else 145)
            painter.setBrush(fill)
            painter.setPen(QPen(QColor("#8ff0bc" if selected else "#5fb887"), 2 if selected else 1))
            painter.drawRoundedRect(rect, 6, 6)

            handle_pen = QPen(QColor("#d6ffe5" if selected else "#89c9a4"), 2)
            painter.setPen(handle_pen)
            painter.drawLine(rect.left() + 3, rect.top() + 4, rect.left() + 3, rect.bottom() - 4)
            painter.drawLine(rect.right() - 3, rect.top() + 4, rect.right() - 3, rect.bottom() - 4)

            label = section.label.strip() or f"Section {section.section_id}"
            text = f"{section.section_id}: {label}"
            painter.setPen(QPen(QColor("#f7fff9")))
            text_rect = rect.adjusted(10, 0, -8, 0)
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)

        if not annotator.sections:
            painter.setPen(QPen(QColor("#748091")))
            painter.drawText(
                QRect(0, 0, width, section_h),
                int(Qt.AlignmentFlag.AlignCenter),
                "Drag here to add a section",
            )

        if annotator.frame_count > 1:
            cursor_x = int(annotator.current_frame / (annotator.frame_count - 1) * (width - 1))
            painter.setPen(QPen(QColor("#64d692"), 2))
            painter.drawLine(cursor_x, 0, cursor_x, height)

        painter.end()
        self.timeline_label.setPixmap(QPixmap.fromImage(img))
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
