from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel


class SectionTimelineWidget(QLabel):
    sectionSelected = pyqtSignal(int)
    sectionPreviewChanged = pyqtSignal(int, int, int)
    sectionChanged = pyqtSignal(int, int, int)
    sectionCreated = pyqtSignal(int, int)
    sectionLabelEditRequested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.annotator = None
        self.selected_section_id: int | None = None
        self.default_length_frames = 300
        self.enable_real_time_section_update = True
        self._drag_mode: str | None = None
        self._drag_section_id: int | None = None
        self._press_frame = 0
        self._orig_start = 0
        self._orig_end = 0
        self._create_start = 0
        self._last_preview: tuple[int, int, int] | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_context(
        self,
        annotator,
        selected_section_id: int | None,
        default_length_frames: int,
        enable_real_time_section_update: bool = True,
    ):
        self.annotator = annotator
        self.selected_section_id = selected_section_id
        self.default_length_frames = max(1, int(default_length_frames))
        self.enable_real_time_section_update = bool(enable_real_time_section_update)

    def _frame_at_x(self, x: int) -> int:
        if not self.annotator or self.annotator.frame_count <= 1:
            return 0
        width = max(1, self.width() - 1)
        ratio = max(0.0, min(1.0, x / width))
        return int(round(ratio * (self.annotator.frame_count - 1)))

    def _x_for_frame(self, frame: int) -> int:
        if not self.annotator or self.annotator.frame_count <= 1:
            return 0
        return int(frame / (self.annotator.frame_count - 1) * max(1, self.width() - 1))

    def _hit_section(self, x: int):
        if not self.annotator:
            return None, None
        handle_px = 7
        for section in sorted(self.annotator.sections, key=lambda item: item.section_id, reverse=True):
            x1 = self._x_for_frame(section.start_frame)
            x2 = self._x_for_frame(section.end_frame)
            if abs(x - x1) <= handle_px:
                return section, "resize_left"
            if abs(x - x2) <= handle_px:
                return section, "resize_right"
            if x1 < x < x2:
                return section, "drag"
        return None, None

    def _section_bounds_for_frame(self, mode: str | None, frame: int):
        if not self.annotator or mode not in {"drag", "resize_left", "resize_right"}:
            return None

        if mode == "drag":
            delta = frame - self._press_frame
            start = self._orig_start + delta
            end = self._orig_end + delta
        elif mode == "resize_left":
            start = frame
            end = self._orig_end
        else:
            start = self._orig_start
            end = frame

        max_frame = max(0, self.annotator.frame_count - 1)
        length = max(1, end - start)
        if start < 0:
            end = min(max_frame, end - start)
            start = 0
        if end > max_frame:
            start = max(0, start - (end - max_frame))
            end = max_frame
        if end <= start:
            if mode == "resize_left":
                start = max(0, end - length)
            else:
                end = min(max_frame, start + length)
        return start, end

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseDoubleClickEvent(event)
        section, _mode = self._hit_section(int(event.position().x()))
        if section:
            self.sectionSelected.emit(section.section_id)
            self.sectionLabelEditRequested.emit(section.section_id)
            return
        start = self._frame_at_x(int(event.position().x()))
        end = start + self.default_length_frames - 1
        self.sectionCreated.emit(start, end)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        x = int(event.position().x())
        frame = self._frame_at_x(x)
        section, mode = self._hit_section(x)
        self._press_frame = frame
        if section:
            self.sectionSelected.emit(section.section_id)
            self._drag_mode = mode
            self._drag_section_id = section.section_id
            self._orig_start = section.start_frame
            self._orig_end = section.end_frame
            self._last_preview = None
            return
        self._drag_mode = "create"
        self._drag_section_id = None
        self._create_start = frame
        self._last_preview = None

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.annotator:
            return super().mouseReleaseEvent(event)
        frame = self._frame_at_x(int(event.position().x()))
        mode = self._drag_mode
        section_id = self._drag_section_id
        self._drag_mode = None
        self._drag_section_id = None

        if mode == "create":
            start = min(self._create_start, frame)
            end = max(self._create_start, frame)
            if abs(end - start) < 2:
                end = start + self.default_length_frames - 1
            self.sectionCreated.emit(start, end)
            return

        if section_id is None:
            return
        bounds = self._section_bounds_for_frame(mode, frame)
        if bounds is None:
            return
        start, end = bounds
        self._last_preview = None
        self.sectionChanged.emit(section_id, start, end)

    def mouseMoveEvent(self, event):
        if self._drag_mode:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            if (
                self.enable_real_time_section_update
                and self._drag_section_id is not None
                and self.annotator is not None
            ):
                frame = self._frame_at_x(int(event.position().x()))
                bounds = self._section_bounds_for_frame(self._drag_mode, frame)
                if bounds is not None:
                    start, end = bounds
                    preview = (self._drag_section_id, start, end)
                    if preview != self._last_preview:
                        self._last_preview = preview
                        self.sectionPreviewChanged.emit(self._drag_section_id, start, end)
            return
        section, mode = self._hit_section(int(event.position().x()))
        if section and mode in {"resize_left", "resize_right"}:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif section:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
