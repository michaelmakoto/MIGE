import os

import cv2
import numpy as np

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from flexible_label import FlexibleLabel
from settings_loader import SettingsLoader
from timeline_renderer import TimelineRenderer
from video_annotator import VideoAnnotatorCore

DEFAULT_COLOR = "#AAAAAA"


class GazeEncoderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Encoder (PyQt6)")
        self.setMinimumSize(300, 200)
        self.resize(1600, 900)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1b1e22;
                color: #e5e5e5;
                font-size: 12pt;
            }
            QLabel#heading {
                font-size: 14pt;
                font-weight: 600;
                color: #f2f2f2;
            }
            QLabel#section {
                font-size: 11pt;
                font-weight: 600;
                color: #cfd2d6;
                margin-top: 8px;
            }
            QPushButton {
                background-color: #2a2e33;
                color: #f7f7f7;
                border: 1px solid #3a3f45;
                border-radius: 4px;
                padding: 8px 10px;
            }
            QPushButton:hover { background-color: #343941; }
            QPushButton:pressed { background-color: #1f2227; }
            QLineEdit {
                background-color: #111317;
                color: #e5e5e5;
                border: 1px solid #3a3f45;
                border-radius: 4px;
                padding: 6px 8px;
            }
            QListWidget {
                background-color: #111317;
                border: 1px solid #2c3036;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px 6px;
            }
            QListWidget::item:selected {
                background-color: #3b4854;
                color: #f9f9f9;
            }
            QScrollArea {
                border: none;
            }
            QSlider::groove:horizontal {
                background: #2f343a;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                border: none;
                width: 12px;
                height: 16px;
                margin: -5px 0;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal { background: #3d7bc6; }
            QSlider::add-page:horizontal { background: #2f343a; }
            """
        )

        self.settings = SettingsLoader()
        self.label_map = self._build_label_map()
        self.app_actions = self._normalize_app_keys(self.settings.app_keys)
        self.qt_to_token = self._build_qt_keymap()
        self.label_delay_ms = self.settings.timings.get("label_delay_ms", 80)
        self.long_press_ms = self.settings.timings.get(
            "long_press_threshold_ms", 1500)
        self.auto_label_interval_ms = self.settings.timings.get(
            "auto_label_interval_ms", 33)
        self.playback_interval_ms = self.settings.timings.get(
            "playback_interval_ms", 33)

        self.wheel_step = max(1, int(self.settings.mouse.get("wheel_step", 1)))
        self.wheel_fast_multiplier = max(
            1, int(self.settings.mouse.get("wheel_fast_multiplier", 1)))
        self.timeline_format = str(
            self.settings.timeline.get("format", "hh:mm:ss:ff"))
        try:
            self.timeline_divisions = max(
                1, int(self.settings.timeline.get("divisions", 10)))
        except Exception:
            self.timeline_divisions = 10

        self.annotator = VideoAnnotatorCore()
        self.timeline_renderer: TimelineRenderer | None = None
        self.last_frame_np: np.ndarray | None = None
        self.video_list: list[str] = []
        self.video_index: int = -1

        self.label_timer = QTimer()
        self.label_timer.timeout.connect(self.auto_label_step)

        self.label_delay_timer = QTimer()
        self.label_delay_timer.setSingleShot(True)
        self.label_delay_timer.timeout.connect(self.start_labeling_after_delay)

        self.active_label_char: str | None = None

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        self.play_speed = 1.0

        self._selected_icon = self._make_red_dot_icon()
        self._empty_icon = QIcon()
        self.display_mode = "frames"

        self.init_ui()

    # ==================================================
    # UI builders
    # ==================================================
    def init_ui(self):
        self.encoding_mode = "default"
        browser_panel = self._build_browser_panel()
        video_frame = self._build_video_display()
        inspector_scroll = self._build_inspector_panel()
        center_split = self._build_center_split(
            browser_panel, video_frame, inspector_scroll)
        bottom_layout = self._build_timeline_area()

        main_vertical = QVBoxLayout()
        main_vertical.setContentsMargins(8, 8, 8, 8)
        main_vertical.setSpacing(8)
        main_vertical.addWidget(center_split, 1)
        main_vertical.addLayout(bottom_layout)
        self.setLayout(main_vertical)

    def _build_browser_panel(self) -> QFrame:
        browser_title = QLabel("Video Browser")
        browser_title.setObjectName("heading")

        self.video_list_widget = QListWidget()
        self.video_list_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video_list_widget.itemClicked.connect(self._on_video_item_clicked)
        self.video_list_widget.currentRowChanged.connect(
            lambda _row: self._update_video_icons())
        self.video_list_widget.setStyleSheet(
            """
            QListWidget {
                background-color: #111317;
                border: 1px solid #2c3036;
                padding: 4px;
            }
            QListWidget::item {
                padding: 10px 8px;
                color: #e5e5e5;
            }
            QListWidget::item:selected {
                background: #111317;
                color: #e5e5e5;
            }
            """
        )

        self.add_video_button = QPushButton("+ Add Video")
        self.add_video_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_video_button.clicked.connect(self.select_video)

        browser_layout = QVBoxLayout()
        browser_layout.setContentsMargins(10, 10, 10, 10)
        browser_layout.setSpacing(10)
        browser_layout.addWidget(browser_title)
        browser_layout.addWidget(self.add_video_button)
        browser_layout.addWidget(self.video_list_widget)
        browser_layout.addStretch()

        browser_panel = QFrame()
        browser_panel.setLayout(browser_layout)
        browser_panel.setMinimumWidth(240)
        browser_panel.setMaximumWidth(360)
        browser_panel.setFrameShape(QFrame.Shape.NoFrame)
        return browser_panel

    def _build_video_display(self) -> QFrame:
        self.video_label = FlexibleLabel("Drag & drop a video to load")
        self.video_label.setMinimumSize(0, 0)
        self.video_label.setMinimumHeight(360)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(False)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #0f1114;")

        video_frame = QFrame()
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.video_label)
        video_frame.setLayout(video_layout)
        video_frame.setMinimumWidth(420)
        return video_frame

    def _build_inspector_panel(self) -> QScrollArea:
        self.full_path_label = QLabel("No video loaded")
        self.full_path_label.setWordWrap(True)
        self.full_path_label.setMinimumWidth(0)
        self.full_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.filename_label = QLabel("")
        self.filename_label.setWordWrap(True)
        self.filename_label.setMinimumWidth(0)
        self.filename_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.info_label = QLabel("Frame - | unlabeled")
        self.info_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_label = QLabel("Mode: default")
        self.mode_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.help_label = QLabel(self._build_help_text())
        self.help_label.setWordWrap(True)
        self.help_label.setMinimumWidth(320)
        self.help_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        inspector_title = QLabel("Inspector & Help")
        inspector_title.setObjectName("heading")
        # details_label = QLabel("-- Video Details --")
        # details_label.setObjectName("section")
        status_label = QLabel("-- Status --")
        status_label.setObjectName("section")
        shortcuts_label = QLabel("-- Data & Shortcuts --")
        shortcuts_label.setObjectName("section")

        nav_buttons = QHBoxLayout()
        nav_buttons.setSpacing(8)
        self.prev_button = QPushButton("◀ Prev")
        self.prev_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button = QPushButton("Next ▶")
        self.next_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_button.clicked.connect(self.next_frame)
        nav_buttons.addWidget(self.prev_button)
        nav_buttons.addWidget(self.next_button)
        nav_buttons.addStretch()

        inspector_layout = QVBoxLayout()
        inspector_layout.setContentsMargins(10, 10, 10, 10)
        inspector_layout.setSpacing(10)
        inspector_layout.addWidget(inspector_title)
        # inspector_layout.addWidget(details_label)
        inspector_layout.addWidget(QLabel("-- Directory:"))
        inspector_layout.addWidget(self.full_path_label)
        inspector_layout.addWidget(QLabel("-- File:"))
        inspector_layout.addWidget(self.filename_label)
        inspector_layout.addSpacing(6)
        inspector_layout.addWidget(QLabel("-- Quick Navigation:"))
        inspector_layout.addLayout(nav_buttons)
        inspector_layout.addWidget(status_label)
        inspector_layout.addWidget(self.info_label)
        inspector_layout.addWidget(self.mode_label)
        inspector_layout.addWidget(shortcuts_label)
        inspector_layout.addWidget(self.help_label)
        inspector_layout.addStretch()

        inspector_inner = QWidget()
        inspector_inner.setLayout(inspector_layout)

        inspector_scroll = QScrollArea()
        inspector_scroll.setWidget(inspector_inner)
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inspector_scroll.setMinimumWidth(320)
        return inspector_scroll

    def _build_timeline_area(self) -> QVBoxLayout:
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(0)
        self.seek_slider.setValue(0)
        self.seek_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.seek_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.seek_slider.valueChanged.connect(self.seek_changed)

        self.long_press_timer = QTimer()
        self.long_press_timer.setSingleShot(True)
        self.long_press_timer.timeout.connect(self.start_continuous_labeling)
        self.is_long_press = False

        self.timeline_label = QLabel()
        self.timeline_label.setMinimumHeight(26)
        self.timeline_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.timeline_label.setStyleSheet("background-color: #24272d;")

        self.tick_label = QLabel()
        self.tick_label.setMinimumHeight(28)
        self.tick_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tick_label.setStyleSheet("background-color: #1b1e22;")

        self.timeline_renderer = TimelineRenderer(
            self.timeline_label, self.tick_label)

        slider_and_timeline = QVBoxLayout()
        slider_and_timeline.setContentsMargins(0, 0, 0, 0)
        slider_and_timeline.setSpacing(4)
        slider_and_timeline.addWidget(self.tick_label)
        slider_and_timeline.addWidget(self.timeline_label)
        slider_and_timeline.addWidget(self.seek_slider)

        bottom_layout = QVBoxLayout()
        bottom_layout.addLayout(slider_and_timeline)
        return bottom_layout

    def _build_center_split(self, browser_panel: QFrame, video_frame: QFrame, inspector_scroll: QScrollArea) -> QSplitter:
        center_split = QSplitter(Qt.Orientation.Horizontal)
        center_split.addWidget(browser_panel)
        center_split.addWidget(video_frame)
        center_split.addWidget(inspector_scroll)
        center_split.setChildrenCollapsible(False)
        center_split.setStretchFactor(0, 1)
        center_split.setStretchFactor(1, 5)
        center_split.setStretchFactor(2, 2)
        center_split.setSizes([260, 1100, 360])
        return center_split

    # ==================================================
    # Build key/color map from settings
    # ==================================================
    def _build_label_map(self):
        mapping = {}
        for key_char, data in self.settings.labels.items():
            token = str(key_char).upper()
            mapping[token] = {
                "mode": data.get("name"),
                "group": data.get("group"),
                "color": data.get("color", DEFAULT_COLOR),
            }
        return mapping

    def _normalize_app_keys(self, app_keys: dict):
        normalized = {}
        for key_char, action in app_keys.items():
            token = str(key_char).upper()
            normalized[token] = action
        return normalized

    def _build_qt_keymap(self):
        qt_map = {}
        all_tokens = set(list(self.label_map.keys()) +
                         list(self.app_actions.keys()))
        for key_char in all_tokens:
            if len(key_char) == 1:
                qt_const = getattr(Qt.Key, f"Key_{key_char}", None)
                if qt_const is None and key_char.isdigit():
                    qt_const = getattr(Qt.Key, f"Key_{key_char}", None)
                if qt_const is not None:
                    qt_map[qt_const] = key_char

        special_lookup = {
            "left": Qt.Key.Key_Left,
            "right": Qt.Key.Key_Right,
            "up": Qt.Key.Key_Up,
            "down": Qt.Key.Key_Down,
            "space": Qt.Key.Key_Space,
        }
        for token in all_tokens:
            lower = token.lower()
            if lower in special_lookup:
                qt_map[special_lookup[lower]] = token
        return qt_map

    def _format_frame_display(self, frame_idx: int) -> str:
        if self.display_mode == "time" and self.annotator.fps:
            return self._format_timecode(frame_idx)
        return f"{frame_idx}"

    def _format_timecode(self, frame_idx: int) -> str:
        fps = self.annotator.fps if self.annotator.fps else 30.0
        frames_per_second = max(1, int(round(fps)))
        total_seconds = frame_idx / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        frames = int(frame_idx % frames_per_second)

        fmt = (self.timeline_format or "hh:mm:ss:ff").lower()
        token_map = {
            "hh": f"{hours:02}",
            "mm": f"{minutes:02}",
            "ss": f"{seconds:02}",
            "ff": f"{frames:02}",
        }
        result = fmt
        for token, val in token_map.items():
            result = result.replace(token, val)
        return result

    def _make_red_dot_icon(self) -> QIcon:
        size = 14
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor("#ff4d4f"))
        painter.setPen(QColor("#ff4d4f"))
        painter.drawEllipse(2, 2, size - 4, size - 4)
        painter.end()
        return QIcon(pix)

    def _update_video_icons(self):
        current = self.video_list_widget.currentRow()
        for i in range(self.video_list_widget.count()):
            item = self.video_list_widget.item(i)
            item.setIcon(self._selected_icon if i ==
                         current else self._empty_icon)

    def _toggle_display_format(self):
        self.display_mode = "time" if self.display_mode == "frames" else "frames"
        self.update_info_label()
        self.update_timeline()
        if self.last_frame_np is not None:
            self.show_frame(self.last_frame_np, store_last=False)

    def color_for_mode(self, mode: str):
        for data in self.label_map.values():
            if data["mode"] == mode:
                return data.get("color", DEFAULT_COLOR)
        return DEFAULT_COLOR

    def _label_stats_lines(self) -> str:
        fps = self.annotator.fps if self.annotator.fps else 30.0
        counts: dict[str, int] = {}
        for ann in self.annotator.annotations.values():
            mode = ann.get("mode")
            if not mode:
                continue
            counts[mode] = counts.get(mode, 0) + 1

        if not counts:
            return "    (Please load a video)"

        max_mode_len = max(len(mode) for mode in counts.keys())
        max_frames = max(counts.values())
        max_msec = int(round(max_frames * (1 / fps) * 1000))
        frames_width = len(str(max_frames))
        msec_width = len(str(max_msec))

        lines = []
        for mode, frames in sorted(counts.items()):
            msec = int(round(frames * (1 / fps) * 1000))
            lines.append(
                f"    {mode.ljust(max_mode_len, ' ')} : "
                f"{str(frames).rjust(frames_width)} -> "
                f"{str(msec).rjust(msec_width)} (msec)"
            )

        total_frames = sum(counts.values())
        total_msec = int(round(total_frames * (1 / fps) * 1000))
        pad = max(max_mode_len - 5, 0)
        lines.append(
            f"    Total{' '.ljust(pad)} : "
            f"{str(total_frames).rjust(frames_width)} -> "
            f"{str(total_msec).rjust(msec_width)} (msec)"
        )
        return "\n".join(lines)

    def _build_help_text(self) -> str:
        label_lines = "\n".join(
            [f"    {k}: {v['mode']} ({v.get('group', '')})" for k, v in
                self.label_map.items()]
        )
        app_lines = "\n".join(
            [f"    {k}: {v}" for k, v in self.app_actions.items()])
        return (
            "Label stats:\n"
            f"{self._label_stats_lines()}"
            "\n\n"
            "App keys:\n"
            f"{app_lines}\n"
            "\n"
            "Labeling keys:\n"
            f"{label_lines}\n"
            "\n"
            "Switch mode with default/scroll\n"
            "    default: Press key to label -> next frame / hold for continuous\n"
            "    scroll: Hold key while using wheel to fill\n"
            "\n"
        )

    def refresh_help_label(self):
        self.help_label.setText(self._build_help_text())

    def _token_from_event(self, event):
        token = self.qt_to_token.get(event.key())
        if token:
            return token
        text = event.text()
        if text:
            return text.upper()
        return None

    # ==================================================
    # Redraw timeline on resize
    # ==================================================
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_timeline()
        if self.last_frame_np is not None:
            self.show_frame(self.last_frame_np, store_last=False)

    # ==================================================
    # Video loading
    # ==================================================
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.load_video(path)

    def load_video(self, path):
        ok = self.annotator.load_video(path)
        if not ok:
            QMessageBox.warning(self, "Error", "Could not open video")
            return
        self.build_video_list(path)

        self.seek_slider.setMaximum(self.annotator.frame_count - 1)
        self.seek_slider.setValue(0)

        frame = self.annotator.get_frame(0)
        if frame is not None:
            self.show_frame(frame)
        self.update_info_label()

        fps_interval = int(
            1000 / self.annotator.fps) if self.annotator.fps else 33
        self.label_timer.setInterval(
            self.auto_label_interval_ms or fps_interval)
        playback_interval = self.playback_interval_ms or fps_interval
        self.play_timer.setInterval(int(playback_interval / self.play_speed))

        self.active_label_char = None
        self.label_timer.stop()
        self.label_delay_timer.stop()
        self.long_press_timer.stop()
        self.play_timer.stop()

        dir_path = os.path.dirname(path)
        file_name = os.path.basename(path)
        self.full_path_label.setText(dir_path)
        self.filename_label.setText(file_name)

        self.update_timeline()
        self.refresh_help_label()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        if self.last_frame_np is not None:
            self.show_frame(self.last_frame_np, store_last=False)

    def build_video_list(self, path: str):
        if not path:
            self.video_list = []
            self.video_index = -1
            self._refresh_video_list_widget()
            return
        directory = os.path.dirname(path)
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        files = []
        for name in os.listdir(directory):
            if name.startswith(".") or name.startswith("._"):
                continue
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(directory, name))
        files.sort()
        self.video_list = files
        try:
            self.video_index = self.video_list.index(path)
        except ValueError:
            self.video_index = -1
        self._refresh_video_list_widget()

    def load_adjacent_video(self, delta: int):
        if not self.video_list or self.video_index < 0:
            return
        new_idx = self.video_index + delta
        new_idx = max(0, min(len(self.video_list) - 1, new_idx))
        if new_idx == self.video_index:
            return
        self.video_index = new_idx
        self.load_video(self.video_list[self.video_index])

    def _refresh_video_list_widget(self):
        self.video_list_widget.clear()
        for path in self.video_list:
            item = QListWidgetItem(os.path.basename(path))
            item.setToolTip(path)
            self.video_list_widget.addItem(item)
        if 0 <= self.video_index < len(self.video_list):
            self.video_list_widget.setCurrentRow(self.video_index)
        self._update_video_icons()

    def _on_video_item_clicked(self, item: QListWidgetItem):
        row = self.video_list_widget.row(item)
        if 0 <= row < len(self.video_list):
            if row != self.video_index or not self.annotator.cap:
                self.video_index = row
                self.load_video(self.video_list[row])

    # ==================================================
    # Timeline rendering
    # ==================================================
    def update_timeline(self):
        if not self.timeline_renderer:
            return
        self.timeline_renderer.render(
            self.annotator,
            self.color_for_mode,
            self._format_frame_display,
            self.timeline_divisions,
        )

    # ==================================================
    # Frame rendering (with overlay)
    # ==================================================
    def show_frame(self, frame, store_last=True):
        h, w = frame.shape[:2]
        max_w = self.video_label.width()
        max_h = self.video_label.height()
        if max_w <= 1 or max_h <= 1:
            return
        scale = min(max_w / w, max_h / h)

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        lab = self.annotator.get_label(self.annotator.current_frame)
        if lab is not None:
            mode = lab.get("mode", "")
            label = lab.get("group", "")
            frame_txt = self._format_frame_display(
                self.annotator.current_frame)
            txt = f"{frame_txt} | {mode} | {label}"

            color_hex = self.color_for_mode(mode) if mode else DEFAULT_COLOR
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            color_rgb = (r, g, b)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale_text = 1.0
            thickness = 2
            (tw, th), _ = cv2.getTextSize(txt, font, scale_text, thickness)

            pad = 12
            x = int((new_w - tw) / 2)
            y = new_h - th - 20

            cv2.rectangle(resized, (x - pad, y - th - pad),
                          (x + tw + pad, y + pad), color_bgr, -1)
            text_color = (255, 255, 255) if sum(color_rgb) < 300 else (0, 0, 0)

            cv2.putText(resized, txt, (x, y), font, scale_text,
                        text_color, thickness, cv2.LINE_AA)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, new_w, new_h, 3 * new_w,
                      QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        if store_last:
            self.last_frame_np = frame

    # ==================================================
    # Info display
    # ==================================================
    def update_info_label(self):
        if not self.annotator.cap:
            self.info_label.setText("Frame - | unlabeled")
            return

        f = self.annotator.current_frame
        lab = self.annotator.get_label(f)
        label_text = "unlabeled" if lab is None else f"{lab['mode']} | {lab.get('group', '')}"
        frame_txt = self._format_frame_display(f)
        self.info_label.setText(f"Frame {frame_txt} | {label_text}")

    # ==================================================
    # CSV save
    # ==================================================
    def save_csv(self):
        if not self.annotator.cap:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        self.annotator.save_csv()
        QMessageBox.information(self, "Saved", "CSV saved")

    # ==================================================
    # Frame navigation + seek bar
    # ==================================================
    def goto_frame(self, idx, do_label=False):
        frame = self.annotator.get_frame(idx)
        if frame is None:
            return
        if do_label and self.active_label_char is not None:
            info = self.label_map.get(self.active_label_char)
            if info:
                self.annotator.set_label(
                    idx, info["mode"], info.get("group", ""))
                self.update_timeline()
                self.refresh_help_label()

        self.annotator.current_frame = idx
        self.show_frame(frame)
        self.seek_slider.setValue(idx)
        self.update_info_label()

    def prev_frame(self):
        idx = max(0, self.annotator.current_frame - 1)
        self.goto_frame(idx)

    def next_frame(self):
        idx = min(self.annotator.frame_count - 1,
                  self.annotator.current_frame + 1)
        self.goto_frame(idx)

    def seek_changed(self, value):
        if self.annotator.cap:
            self.goto_frame(value)

    # ==================================================
    # Playback
    # ==================================================
    def toggle_play(self):
        if not self.annotator.cap:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
        else:
            self.play_timer.start()

    def play_next_frame(self):
        frame = self.annotator.read_next_frame()
        if frame is None:
            self.toggle_play()
            return

        self.show_frame(frame)
        self.seek_slider.setValue(self.annotator.current_frame)
        self.update_info_label()

    # ==================================================
    # Labeling (key)
    # ==================================================
    def keyPressEvent(self, event):
        token = self._token_from_event(event)
        action = self.app_actions.get(token) if token else None
        if action == "toggle_mode":
            self.encoding_mode = "scroll" if self.encoding_mode == "default" else "default"
            self.mode_label.setText(f"Mode: {self.encoding_mode}")
            return
        if action == "toggle_play":
            self.toggle_play()
            return
        if action == "prev_frame":
            self.prev_frame()
            return
        if action == "next_frame":
            self.next_frame()
            return
        if action == "prev_video":
            self.load_adjacent_video(-1)
            return
        if action == "next_video":
            self.load_adjacent_video(+1)
            return
        if action == "fillin":
            self.fill_between_labels()
            return
        if action == "toggle_display_format":
            self._toggle_display_format()
            return

        if token in self.label_map:
            self.active_label_char = token
            self.label_delay_timer.start(self.label_delay_ms)
            self.long_press_timer.start(self.long_press_ms)
            self.is_long_press = False
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        token = self._token_from_event(event)
        if token and self.active_label_char == token:
            self.active_label_char = None
            self.label_timer.stop()
            self.label_delay_timer.stop()
            self.long_press_timer.stop()
            return
        super().keyReleaseEvent(event)

    def start_labeling_after_delay(self):
        if self.active_label_char and self.annotator.cap:
            if self.encoding_mode == "scroll":
                info = self.label_map.get(self.active_label_char)
                if info:
                    self.annotator.set_label(
                        self.annotator.current_frame, info["mode"], info.get("group", ""))
                    self.update_info_label()
                    self.update_timeline()
                    self.refresh_help_label()
                    return

            if not self.is_long_press:
                info = self.label_map.get(self.active_label_char)
                if info:
                    self.annotator.set_label(
                        self.annotator.current_frame, info["mode"], info.get("group", ""))
                    self.update_timeline()
                    self.refresh_help_label()
                    next_idx = min(self.annotator.frame_count - 1,
                                   self.annotator.current_frame + 1)
                    self.goto_frame(next_idx, do_label=False)

    def start_continuous_labeling(self):
        if self.encoding_mode == "scroll":
            return
        if self.active_label_char is None:
            return
        self.is_long_press = True
        if not self.label_timer.isActive():
            self.label_timer.start()

    def auto_label_step(self):
        if not self.annotator.cap or self.active_label_char is None:
            return
        info = self.label_map.get(self.active_label_char)
        if info:
            self.annotator.set_label(
                self.annotator.current_frame, info["mode"], info.get("group", ""))
            self.update_timeline()
            self.refresh_help_label()

        next_idx = self.annotator.current_frame + 1
        if next_idx >= self.annotator.frame_count:
            self.label_timer.stop()
            return
        self.goto_frame(next_idx, do_label=False)

    # ==================================================
    # Mouse wheel navigation
    # ==================================================
    def wheelEvent(self, event):
        if not self.annotator.cap:
            return

        delta_pt = event.angleDelta()
        delta = delta_pt.y() or delta_pt.x()
        base_step = self.wheel_step
        step = -base_step if delta > 0 else base_step
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step *= self.wheel_fast_multiplier

        idx = self.annotator.current_frame + step
        idx = max(0, min(self.annotator.frame_count - 1, idx))

        if self.encoding_mode == "scroll":
            if self.active_label_char is not None:
                self.goto_frame(idx, do_label=True)
            else:
                self.goto_frame(idx, do_label=False)
        else:
            do_label = self.active_label_char is not None
            self.goto_frame(idx, do_label)

    # ==================================================
    # Drag & Drop
    # ==================================================
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in {".mp4", ".avi", ".mov", ".mkv"}:
                self.load_video(path)

    # ==================================================
    # Save on exit
    # ==================================================
    def closeEvent(self, event):
        self.annotator.save_csv()
        super().closeEvent(event)

    # ==================================================
    # Fill-in feature
    # ==================================================
    def _find_neighbor_label(self, start_idx: int, step: int):
        idx = start_idx + step
        while 0 <= idx < self.annotator.frame_count:
            lab = self.annotator.annotations.get(idx)
            if lab is not None:
                return idx, lab
            idx += step
        return None, None

    def fill_between_labels(self):
        if not self.annotator.cap:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return

        cur = self.annotator.current_frame
        current_label = self.annotator.get_label(cur)
        if current_label is not None:
            QMessageBox.information(self, "Warning", "Run Fillin on an unlabeled frame")
            return

        prev_idx, prev_label = self._find_neighbor_label(cur, -1)
        next_idx, next_label = self._find_neighbor_label(cur, +1)
        if prev_label is None or next_label is None:
            QMessageBox.warning(
                self, "Warning", "When using Fillin, labels before and after the cursor must match")
            return

        same_mode = prev_label["mode"] == next_label["mode"]
        same_group = prev_label.get("group", "") == next_label.get("group", "")
        if not (same_mode and same_group):
            QMessageBox.warning(
                self, "Warning", "When using Fillin, labels before and after the cursor must match")
            return

        fill_mode = prev_label["mode"]
        fill_group = prev_label.get("group", "")

        filled = 0
        for idx in range(prev_idx + 1, next_idx):
            if self.annotator.annotations.get(idx) is None:
                self.annotator.annotations[idx] = {
                    "mode": fill_mode,
                    "group": fill_group,
                }
                filled += 1

        if filled == 0:
            QMessageBox.information(self, "Info", "No unlabeled frames between labels")
            return

        self.annotator.save_csv()
        self.update_timeline()
        self.refresh_help_label()
        self.goto_frame(cur, do_label=False)
        self.update_info_label()
