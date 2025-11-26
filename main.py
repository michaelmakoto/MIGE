import sys
from PyQt6.QtWidgets import QApplication
from gaze_encoder_app import GazeEncoderApp


def main():
    app = QApplication(sys.argv)
    w = GazeEncoderApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
