import numpy as np
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel


class WaveformRenderer:
    def __init__(self, label: QLabel):
        self.label = label
        self.samples: np.ndarray | None = None
        self.sample_rate: int = 44100
        self.frame_count: int = 0
        self.fps: float = 30.0
        self._static_pixmap: QPixmap | None = None  # cached waveform without cursor

    def set_audio(self, samples: np.ndarray, sample_rate: int, frame_count: int, fps: float):
        self.samples = samples
        self.sample_rate = sample_rate
        self.frame_count = frame_count
        self.fps = fps
        self._static_pixmap = None  # invalidate cache

    def clear(self):
        self.samples = None
        self._static_pixmap = None
        self._paint(current_frame=0)

    def render(self, current_frame: int):
        self._paint(current_frame)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _build_static(self, w: int, h: int) -> QPixmap:
        """Paint waveform bars once; cursor is added separately each frame."""
        pix = QPixmap(w, h)
        pix.fill(QColor("#0f1114"))
        if self.samples is None or len(self.samples) == 0:
            return pix

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        mid = h // 2
        total = len(self.samples)
        painter.setPen(QPen(QColor("#3d7bc6"), 1))

        # Map each pixel directly to a sample range — no frame-count dependency.
        for px in range(w):
            s_start = int(px * total / w)
            s_end = int((px + 1) * total / w)
            s_end = max(s_start + 1, min(s_end, total))

            chunk = self.samples[s_start:s_end]
            amp = float(np.max(np.abs(chunk)))
            bar_h = max(1, int(amp * (mid - 1)))
            painter.drawLine(px, mid - bar_h, px, mid + bar_h)

        painter.end()
        return pix

    def _paint(self, current_frame: int):
        w = self.label.width()
        h = self.label.height()
        if w <= 1 or h <= 1:
            return

        # Rebuild static only when size changes (zoom / resize).
        if (self._static_pixmap is None
                or self._static_pixmap.width() != w
                or self._static_pixmap.height() != h):
            self._static_pixmap = self._build_static(w, h)

        # Copy static and draw cursor — fast path for every frame tick.
        pix = QPixmap(self._static_pixmap)

        if self.samples is not None and self.frame_count > 0 and len(self.samples) > 0:
            total = len(self.samples)
            # Convert frame → sample → pixel using the same mapping as the waveform.
            sample_pos = int(current_frame / max(1.0, self.fps) * self.sample_rate)
            sample_pos = max(0, min(sample_pos, total - 1))
            cx = int(sample_pos / total * (w - 1))

            painter = QPainter(pix)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawLine(cx, 0, cx, h)
            painter.end()

        self.label.setPixmap(pix)
