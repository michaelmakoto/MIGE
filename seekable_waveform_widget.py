from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel


class SeekableWaveformWidget(QLabel):
    frameRequested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.annotator = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_context(self, annotator):
        self.annotator = annotator

    def _frame_at_x(self, x: int) -> int:
        if not self.annotator or self.annotator.frame_count <= 1:
            return 0
        width = max(1, self.width() - 1)
        ratio = max(0.0, min(1.0, x / width))
        return int(round(ratio * (self.annotator.frame_count - 1)))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self.frameRequested.emit(self._frame_at_x(int(event.position().x())))

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frameRequested.emit(self._frame_at_x(int(event.position().x())))
            return
        self.setCursor(Qt.CursorShape.PointingHandCursor)
