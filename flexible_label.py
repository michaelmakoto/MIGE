from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QLabel


class FlexibleLabel(QLabel):
    """QLabel that does not force a minimum size based on its pixmap."""

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(0, 0)

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(0, 0)
