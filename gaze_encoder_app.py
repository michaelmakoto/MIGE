import os
import re
import shutil
import subprocess

import cv2
import numpy as np

from PyQt6.QtCore import QBuffer, QByteArray, QIODeviceBase, QTimer, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
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
from section_timeline_widget import SectionTimelineWidget
from settings_dialog import SettingsDialog
from settings_loader import SettingsLoader
from timeline_renderer import TimelineRenderer
from video_annotator import VideoAnnotatorCore, VideoSection
from waveform_renderer import WaveformRenderer

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
            QComboBox, QDoubleSpinBox {
                background-color: #111317;
                color: #e5e5e5;
                border: 1px solid #3a3f45;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QGroupBox {
                border: 1px solid #2c3036;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #dfe3e8;
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
        self.forward_num_frames = self.settings.timings.get("forward_num_frames", 1)
        self.backward_num_frames = self.settings.timings.get("backward_num_frames", 1)

        self.wheel_step = max(1, int(self.settings.mouse.get("wheel_step", 1)))
        self.wheel_fast_multiplier = max(
            1, int(self.settings.mouse.get("wheel_fast_multiplier", 1)))
        self.timeline_format = str(
            self.settings.timeline.get("format", "hh:mm:ss:ff"))
        self.default_section_seconds = self.settings.default_section_seconds()
        self.group_ids = list(self.settings.groups)
        try:
            self.timeline_divisions = max(
                1, int(self.settings.timeline.get("divisions", 10)))
        except Exception:
            self.timeline_divisions = 10

        self.annotator = VideoAnnotatorCore()
        self.timeline_renderer: TimelineRenderer | None = None
        self.waveform_renderer: WaveformRenderer | None = None
        self.last_frame_np: np.ndarray | None = None
        self.video_list: list[str] = []
        self.video_index: int = -1
        self.selected_section_id: int | None = None
        self._updating_section_fields = False
        self._has_shown_move_warning = False
        self.preview_brightness = 0
        self.preview_contrast = 100

        self.label_timer = QTimer()
        self.label_timer.timeout.connect(self.auto_label_step)

        self.label_delay_timer = QTimer()
        self.label_delay_timer.setSingleShot(True)
        self.label_delay_timer.timeout.connect(self.start_labeling_after_delay)

        self.active_label_char: str | None = None

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.play_next_frame)
        self.play_speed = 1.0

        self.autosave_timer = QTimer()
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self._flush_autosave)

        self._selected_icon = self._make_red_dot_icon()
        self._empty_icon = QIcon()
        self.display_mode = "frames"

        self._audio_samples: np.ndarray | None = None
        self._audio_sample_rate: int = 0
        self._audio_sink: QAudioSink | None = None
        self._audio_buffer: QBuffer | None = None
        self._audio_ba: QByteArray | None = None

        self._zoom_level: int = 1

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

        settings_box = QGroupBox("Settings")
        settings_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        settings_layout = QFormLayout()
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(8)

        self.section_group_combo = QComboBox()
        self.section_group_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._reload_group_combo()

        self.default_duration_spin = QDoubleSpinBox()
        self.default_duration_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.default_duration_spin.setRange(0.1, 3600.0)
        self.default_duration_spin.setDecimals(2)
        self.default_duration_spin.setSuffix(" sec")
        self.default_duration_spin.setValue(self.default_section_seconds)
        self.default_duration_spin.valueChanged.connect(self._on_default_duration_changed)

        self.settings_button = QPushButton("Edit Keys / Modes")
        self.settings_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.settings_button.clicked.connect(self.open_settings_dialog)

        settings_layout.addRow("Group ID", self.section_group_combo)
        settings_layout.addRow("New length", self.default_duration_spin)
        settings_layout.addRow(self.settings_button)
        settings_box.setLayout(settings_layout)

        browser_layout = QVBoxLayout()
        browser_layout.setContentsMargins(10, 10, 10, 10)
        browser_layout.setSpacing(10)
        browser_layout.addWidget(browser_title)
        browser_layout.addWidget(self.add_video_button)
        browser_layout.addWidget(self.video_list_widget)
        browser_layout.addWidget(settings_box)

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
        self.filename_label = QLabel("")
        self.filename_label.setWordWrap(True)
        self.filename_label.setMinimumWidth(0)
        self.filename_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.info_label = QLabel("Frame - | unlabeled")
        self.info_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mode_label = QLabel("Mode: default")
        self.mode_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.active_section_status_label = QLabel("Section: none")
        self.active_section_status_label.setWordWrap(True)
        self.help_label = QLabel(self._build_help_text())
        self.help_label.setWordWrap(True)
        self.help_label.setMinimumWidth(320)
        self.help_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        inspector_title = QLabel("Editor")
        inspector_title.setObjectName("heading")
        status_label = QLabel("-- Status --")
        status_label.setObjectName("section")

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

        self.section_list_widget = QListWidget()
        self.section_list_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.section_list_widget.currentRowChanged.connect(self._on_section_row_changed)

        self.add_section_button = QPushButton("+ Add Section")
        self.add_section_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_section_button.clicked.connect(self.add_section_at_cursor)

        self.delete_section_button = QPushButton("Delete Section")
        self.delete_section_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.delete_section_button.clicked.connect(self.delete_selected_section)

        section_buttons = QHBoxLayout()
        section_buttons.addWidget(self.add_section_button)
        section_buttons.addWidget(self.delete_section_button)

        self.section_id_value = QLabel("-")
        self.section_label_edit = QLineEdit()
        self.section_label_edit.setPlaceholderText("Section label")
        self.section_group_edit = QComboBox()
        self._reload_section_editor_group_combo()
        self.section_start_spin = QDoubleSpinBox()
        self.section_start_spin.setDecimals(3)
        self.section_start_spin.setSuffix(" sec")
        self.section_start_spin.setRange(0, 24 * 60 * 60)
        self.section_end_spin = QDoubleSpinBox()
        self.section_end_spin.setDecimals(3)
        self.section_end_spin.setSuffix(" sec")
        self.section_end_spin.setRange(0, 24 * 60 * 60)
        self.apply_section_button = QPushButton("Apply Section")
        self.apply_section_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.apply_section_button.clicked.connect(self.apply_section_editor)

        section_form = QFormLayout()
        section_form.setContentsMargins(0, 0, 0, 0)
        section_form.addRow("Section ID", self.section_id_value)
        section_form.addRow("Label", self.section_label_edit)
        section_form.addRow("Group ID", self.section_group_edit)
        section_form.addRow("Start", self.section_start_spin)
        section_form.addRow("End", self.section_end_spin)
        section_form.addRow(self.apply_section_button)

        section_box = QGroupBox("Sections")
        section_layout = QVBoxLayout()
        section_layout.addWidget(self.section_list_widget)
        section_layout.addLayout(section_buttons)
        section_layout.addLayout(section_form)
        section_box.setLayout(section_layout)

        self.brightness_value_label = QLabel("0")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self._on_preview_effect_changed)
        self.contrast_value_label = QLabel("100%")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._on_preview_effect_changed)

        effects_form = QFormLayout()
        effects_form.setContentsMargins(0, 0, 0, 0)
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self.brightness_slider)
        brightness_row.addWidget(self.brightness_value_label)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self.contrast_slider)
        contrast_row.addWidget(self.contrast_value_label)
        effects_form.addRow("Brightness", brightness_row)
        effects_form.addRow("Contrast", contrast_row)

        effects_box = QGroupBox("Video Effects")
        effects_box.setLayout(effects_form)

        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.save_csv)
        self.export_calc_button = QPushButton("Export Calculated CSV")
        self.export_calc_button.clicked.connect(self.export_calculated_csv)
        self.export_section_video_button = QPushButton("Export Selected Video")
        self.export_section_video_button.clicked.connect(self.export_selected_section_video)

        export_box = QGroupBox("Export")
        export_layout = QVBoxLayout()
        export_layout.addWidget(self.export_csv_button)
        export_layout.addWidget(self.export_calc_button)
        export_layout.addWidget(self.export_section_video_button)
        export_box.setLayout(export_layout)

        inspector_layout = QVBoxLayout()
        inspector_layout.setContentsMargins(10, 10, 10, 10)
        inspector_layout.setSpacing(10)
        inspector_layout.addWidget(inspector_title)
        inspector_layout.addWidget(QLabel("-- File:"))
        inspector_layout.addWidget(self.filename_label)
        inspector_layout.addSpacing(6)
        inspector_layout.addWidget(QLabel("-- Quick Navigation:"))
        inspector_layout.addLayout(nav_buttons)
        inspector_layout.addWidget(status_label)
        inspector_layout.addWidget(self.info_label)
        inspector_layout.addWidget(self.mode_label)
        inspector_layout.addWidget(self.active_section_status_label)
        inspector_layout.addWidget(section_box, 2)
        inspector_layout.addWidget(effects_box)
        inspector_layout.addWidget(export_box)
        shortcuts_label = QLabel("-- Keys --")
        shortcuts_label.setObjectName("section")
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

        self.timeline_label = SectionTimelineWidget()
        self.timeline_label.sectionSelected.connect(self.select_section)
        self.timeline_label.sectionChanged.connect(self.move_section_from_timeline)
        self.timeline_label.sectionCreated.connect(self.create_section_from_timeline)
        self.timeline_label.sectionLabelEditRequested.connect(self.edit_section_label_inline)
        self.timeline_label.setMinimumHeight(96)
        self.timeline_label.setMaximumHeight(96)
        self.timeline_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.timeline_label.setStyleSheet("background-color: #15171b;")

        self.tick_label = QLabel()
        self.tick_label.setMinimumHeight(28)
        self.tick_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tick_label.setStyleSheet("background-color: #1b1e22;")

        self.timeline_renderer = TimelineRenderer(
            self.timeline_label, self.tick_label)

        self.waveform_label = QLabel()
        self.waveform_label.setMinimumHeight(60)
        self.waveform_label.setMaximumHeight(60)
        self.waveform_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.waveform_label.setStyleSheet("background-color: #0f1114;")

        self.waveform_renderer = WaveformRenderer(self.waveform_label)

        self._zoom_slider = QSlider(Qt.Orientation.Vertical)
        self._zoom_slider.setMinimum(1)
        self._zoom_slider.setMaximum(16)
        self._zoom_slider.setValue(1)
        self._zoom_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._zoom_slider.setFixedWidth(18)
        self._zoom_slider.setStyleSheet(
            """
            QSlider::groove:vertical {
                background: #2f343a;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #58a6ff;
                border: none;
                width: 16px;
                height: 12px;
                margin: 0 -5px;
                border-radius: 2px;
            }
            QSlider::sub-page:vertical { background: #2f343a; }
            QSlider::add-page:vertical { background: #3d7bc6; }
            """
        )
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)

        self._zoom_level_label = QLabel("x1")
        self._zoom_level_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._zoom_level_label.setFixedWidth(28)
        self._zoom_level_label.setStyleSheet("color: #666; font-size: 8pt;")

        zoom_col = QVBoxLayout()
        zoom_col.setContentsMargins(0, 0, 4, 0)
        zoom_col.setSpacing(2)
        zoom_col.addWidget(self._zoom_slider, 1)
        zoom_col.addWidget(self._zoom_level_label)

        self._zoomable_content = QWidget()
        zoomable_layout = QVBoxLayout()
        zoomable_layout.setContentsMargins(0, 0, 0, 0)
        zoomable_layout.setSpacing(4)
        zoomable_layout.addWidget(self.waveform_label)
        zoomable_layout.addWidget(self.tick_label)
        zoomable_layout.addWidget(self.timeline_label)
        self._zoomable_content.setLayout(zoomable_layout)

        self._zoom_scroll = QScrollArea()
        self._zoom_scroll.setWidget(self._zoomable_content)
        self._zoom_scroll.setWidgetResizable(False)
        self._zoom_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._zoom_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._zoom_scroll.setFrameShape(QFrame.Shape.NoFrame)

        timeline_row = QHBoxLayout()
        timeline_row.setContentsMargins(0, 0, 0, 0)
        timeline_row.setSpacing(0)
        timeline_row.addLayout(zoom_col)
        timeline_row.addWidget(self._zoom_scroll, 1)

        seek_spacer = QWidget()
        seek_spacer.setFixedWidth(32)  # matches zoom_col width (28px label + 4px right margin)

        seek_row = QHBoxLayout()
        seek_row.setContentsMargins(0, 0, 0, 0)
        seek_row.setSpacing(0)
        seek_row.addWidget(seek_spacer)
        seek_row.addWidget(self.seek_slider, 1)

        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)
        bottom_layout.addLayout(timeline_row)
        bottom_layout.addLayout(seek_row)
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

    def _set_combo_items(self, combo: QComboBox, items: list[str], current: str = ""):
        combo.blockSignals(True)
        combo.clear()
        clean_items = [item for item in items if item]
        if not clean_items:
            clean_items = ["default"]
        combo.addItems(clean_items)
        if current and current in clean_items:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _reload_group_combo(self):
        self.group_ids = list(self.settings.groups)
        if hasattr(self, "section_group_combo"):
            current = self.section_group_combo.currentText()
            self._set_combo_items(self.section_group_combo, self.group_ids, current)

    def _reload_section_editor_group_combo(self):
        if hasattr(self, "section_group_edit"):
            current = self.section_group_edit.currentText()
            self._set_combo_items(self.section_group_edit, self.group_ids, current)
        else:
            self.section_group_edit = QComboBox()
            self._set_combo_items(self.section_group_edit, self.group_ids)

    def _on_default_duration_changed(self, value: float):
        self.default_section_seconds = float(value)
        self.settings.data.setdefault("sections", {})["default_duration_seconds"] = self.default_section_seconds
        self.settings.save()
        self._refresh_timeline_context()

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings.reload()
            self.label_map = self._build_label_map()
            self.app_actions = self._normalize_app_keys(self.settings.app_keys)
            self.qt_to_token = self._build_qt_keymap()
            self.default_section_seconds = self.settings.default_section_seconds()
            self.timeline_format = str(self.settings.timeline.get("format", "hh:mm:ss:ff"))
            self.timeline_divisions = max(1, int(self.settings.timeline.get("divisions", 10)))
            self._reload_group_combo()
            self._reload_section_editor_group_combo()
            self.default_duration_spin.blockSignals(True)
            self.default_duration_spin.setValue(self.default_section_seconds)
            self.default_duration_spin.blockSignals(False)
            self.update_timeline()
            self.refresh_help_label()

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
        self._apply_zoom()
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
        self._flush_autosave()
        if self.annotator.sections_dirty:
            self.annotator.save_sections()
        ok = self.annotator.load_video(path)
        if not ok:
            QMessageBox.warning(self, "Error", "Could not open video")
            return
        self.selected_section_id = None
        self._has_shown_move_warning = False
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
        self._audio_stop()
        self._zoom_slider.setValue(1)

        file_name = os.path.basename(path)
        self.filename_label.setText(file_name)

        self._refresh_sections_ui()
        self.update_timeline()
        self.refresh_help_label()
        self._load_waveform()
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
    # Zoom
    # ==================================================
    def _on_zoom_changed(self, value: int):
        self._zoom_level = value
        self._zoom_level_label.setText(f"x{value}")
        self._apply_zoom()

    def _apply_zoom(self):
        if not hasattr(self, "_zoom_scroll"):
            return
        vw = self._zoom_scroll.viewport().width()
        if vw <= 1:
            return
        content_w = vw * self._zoom_level
        for lbl in (self.waveform_label, self.tick_label, self.timeline_label):
            lbl.setFixedWidth(content_w)
        self._zoomable_content.setFixedWidth(content_w)
        self.update_waveform()
        self.update_timeline()
        self._scroll_to_frame(self.annotator.current_frame)

    def _scroll_to_frame(self, frame_idx: int):
        if not hasattr(self, "_zoom_scroll") or self._zoom_level <= 1:
            return
        if self.annotator.frame_count <= 1:
            return
        vw = self._zoom_scroll.viewport().width()
        content_w = vw * self._zoom_level
        cx = int(frame_idx / (self.annotator.frame_count - 1) * (content_w - 1))
        target = max(0, cx - vw // 2)
        self._zoom_scroll.horizontalScrollBar().setValue(target)

    # ==================================================
    # Waveform + Audio
    # ==================================================
    def _load_waveform(self):
        if not self.waveform_renderer:
            return
        samples, sample_rate = self.annotator.get_audio_samples()
        self._audio_samples = samples
        self._audio_sample_rate = sample_rate
        if samples is not None:
            self.waveform_renderer.set_audio(
                samples, sample_rate, self.annotator.frame_count, self.annotator.fps)
            self._setup_audio_sink()
        else:
            self.waveform_renderer.clear()
            self._audio_sink = None
            self._audio_ba = None
        self.waveform_renderer.render(self.annotator.current_frame)

    def _setup_audio_sink(self):
        if self._audio_samples is None or self._audio_sample_rate == 0:
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(self._audio_sample_rate)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        pcm = (self._audio_samples * 32767).astype(np.int16).tobytes()
        self._audio_ba = QByteArray(pcm)
        self._audio_sink = QAudioSink(fmt)

    def _audio_play_from_frame(self, frame_idx: int):
        if self._audio_sink is None or self._audio_ba is None:
            return
        self._audio_sink.stop()
        if self._audio_buffer:
            self._audio_buffer.close()
        sample_offset = int(frame_idx / max(1.0, self.annotator.fps) * self._audio_sample_rate)
        sample_offset = max(0, min(sample_offset, self._audio_ba.size() // 2 - 1))
        self._audio_buffer = QBuffer(self._audio_ba)
        self._audio_buffer.open(QIODeviceBase.OpenModeFlag.ReadOnly)
        self._audio_buffer.seek(sample_offset * 2)
        self._audio_sink.start(self._audio_buffer)

    def _audio_stop(self):
        if self._audio_sink:
            self._audio_sink.stop()
        if self._audio_buffer:
            self._audio_buffer.close()
            self._audio_buffer = None

    def update_waveform(self):
        if self.waveform_renderer:
            self.waveform_renderer.render(self.annotator.current_frame)

    # ==================================================
    # Timeline rendering
    # ==================================================
    def _default_section_frames(self) -> int:
        fps = self.annotator.fps if self.annotator.fps else 30.0
        return max(1, int(round(self.default_section_seconds * fps)))

    def _refresh_timeline_context(self):
        if isinstance(self.timeline_label, SectionTimelineWidget):
            self.timeline_label.set_context(
                self.annotator,
                self.selected_section_id,
                self._default_section_frames(),
            )

    def update_timeline(self):
        if not self.timeline_renderer:
            return
        self._refresh_timeline_context()
        self.timeline_renderer.render(
            self.annotator,
            self.color_for_mode,
            self._format_frame_display,
            self.timeline_divisions,
            self.selected_section_id,
        )

    def _frame_to_seconds(self, frame: int) -> float:
        return frame / max(self.annotator.fps, 1.0)

    def _seconds_to_frame(self, seconds: float) -> int:
        return int(round(seconds * max(self.annotator.fps, 1.0)))

    def _selected_section(self) -> VideoSection | None:
        if self.selected_section_id is None:
            return None
        return self.annotator.section_by_id(self.selected_section_id)

    def _section_display_name(self, section: VideoSection) -> str:
        label = section.label.strip() or f"Section {section.section_id}"
        start = self._format_timecode(section.start_frame)
        end = self._format_timecode(section.end_frame)
        return f"{section.section_id} | {label} | {section.group_id} | {start}-{end}"

    def _refresh_sections_ui(self):
        if not hasattr(self, "section_list_widget"):
            return
        self._updating_section_fields = True
        self.section_list_widget.clear()
        for section in self.annotator.sections:
            item = QListWidgetItem(self._section_display_name(section))
            item.setData(Qt.ItemDataRole.UserRole, section.section_id)
            self.section_list_widget.addItem(item)
            if section.section_id == self.selected_section_id:
                self.section_list_widget.setCurrentItem(item)
        self._updating_section_fields = False
        self._load_section_editor()

    def _load_section_editor(self):
        section = self._selected_section()
        self._updating_section_fields = True
        if section is None:
            self.section_id_value.setText("-")
            self.section_label_edit.setText("")
            self.section_start_spin.setValue(0)
            self.section_end_spin.setValue(0)
        else:
            self.section_id_value.setText(str(section.section_id))
            self.section_label_edit.setText(section.label)
            self.section_group_edit.setCurrentText(section.group_id)
            self.section_start_spin.setValue(self._frame_to_seconds(section.start_frame))
            self.section_end_spin.setValue(self._frame_to_seconds(section.end_frame))
        self._updating_section_fields = False

    def _on_section_row_changed(self, row: int):
        if self._updating_section_fields or row < 0:
            return
        item = self.section_list_widget.item(row)
        if item is None:
            return
        section_id = item.data(Qt.ItemDataRole.UserRole)
        self.select_section(section_id, jump=True)

    def select_section(self, section_id: int, jump: bool = False):
        section = self.annotator.section_by_id(section_id)
        if section is None:
            return
        self.selected_section_id = section.section_id
        self._refresh_sections_ui()
        self.update_timeline()
        if jump:
            self.goto_frame(section.start_frame)

    def add_section_at_cursor(self):
        if not self.annotator.cap:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        start = self.annotator.current_frame
        end = start + self._default_section_frames() - 1
        self.create_section_from_timeline(start, end)

    def create_section_from_timeline(self, start_frame: int, end_frame: int):
        if not self.annotator.cap:
            return
        group_id = self.section_group_combo.currentText() if hasattr(self, "section_group_combo") else ""
        next_id = self.annotator.next_section_id()
        section = self.annotator.add_section(
            start_frame,
            end_frame,
            group_id=group_id,
            label=f"Section {next_id}",
        )
        self.selected_section_id = section.section_id
        self._refresh_sections_ui()
        self.update_timeline()
        self.goto_frame(section.start_frame)

    def _warn_before_moving_section(self, section_id: int):
        if self._has_shown_move_warning:
            return
        if self.annotator.annotation_count_for_section_id(section_id) <= 0:
            return
        QMessageBox.information(
            self,
            "Section moved",
            "Existing encoded rows will stay on their original frames. Moving the section only changes the highlighted range for future encoding.",
        )
        self._has_shown_move_warning = True

    def move_section_from_timeline(self, section_id: int, start_frame: int, end_frame: int):
        self._warn_before_moving_section(section_id)
        section = self.annotator.update_section(
            section_id,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        if section:
            self.selected_section_id = section.section_id
        self._refresh_sections_ui()
        self.update_timeline()

    def apply_section_editor(self):
        if self._updating_section_fields:
            return
        section = self._selected_section()
        if section is None:
            QMessageBox.warning(self, "Warning", "Select a section first")
            return
        start_frame = self._seconds_to_frame(self.section_start_spin.value())
        end_frame = self._seconds_to_frame(self.section_end_spin.value())
        if start_frame != section.start_frame or end_frame != section.end_frame:
            self._warn_before_moving_section(section.section_id)
        self.annotator.update_section(
            section.section_id,
            label=self.section_label_edit.text().strip() or f"Section {section.section_id}",
            group_id=self.section_group_edit.currentText(),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self._refresh_sections_ui()
        self.update_timeline()
        self.update_info_label()
        if self.last_frame_np is not None:
            self.show_frame(self.last_frame_np, store_last=False)

    def delete_selected_section(self):
        section = self._selected_section()
        if section is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete section",
            f"Delete section {section.section_id}? Existing encoded rows will remain in the CSV.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.annotator.delete_section(section.section_id)
        self.selected_section_id = None
        self._refresh_sections_ui()
        self.update_timeline()
        self.update_info_label()

    def edit_section_label_inline(self, section_id: int):
        section = self.annotator.section_by_id(section_id)
        if section is None:
            return
        text, ok = QInputDialog.getText(
            self,
            "Section label",
            "Label",
            text=section.label,
        )
        if ok:
            self.annotator.update_section(
                section.section_id,
                label=text.strip() or f"Section {section.section_id}",
            )
            self.selected_section_id = section.section_id
            self._refresh_sections_ui()
            self.update_timeline()
            self.update_info_label()

    # ==================================================
    # Frame rendering (with overlay)
    # ==================================================
    def _apply_preview_effects(self, frame: np.ndarray) -> np.ndarray:
        alpha = max(0.1, self.preview_contrast / 100.0)
        beta = int(self.preview_brightness)
        if alpha == 1.0 and beta == 0:
            return frame
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def _draw_badge(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color_hex: str,
        align_center: bool = False,
    ):
        if not text:
            return
        color_hex = color_hex if re.match(r"^#[0-9a-fA-F]{6}$", color_hex) else DEFAULT_COLOR
        color_rgb = (
            int(color_hex[1:3], 16),
            int(color_hex[3:5], 16),
            int(color_hex[5:7], 16),
        )
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_text = 0.8
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale_text, thickness)
        max_text_w = max(40, image.shape[1] - 36)
        while tw > max_text_w and scale_text > 0.35:
            scale_text -= 0.05
            (tw, th), _ = cv2.getTextSize(text, font, scale_text, thickness)
        pad = 10
        if align_center:
            x = int((image.shape[1] - tw) / 2)
        x = max(pad, min(image.shape[1] - tw - pad, x))
        y = max(th + pad, min(image.shape[0] - pad, y))
        cv2.rectangle(
            image,
            (x - pad, y - th - pad),
            (x + tw + pad, y + pad),
            color_bgr,
            -1,
        )
        text_color = (255, 255, 255) if sum(color_rgb) < 300 else (0, 0, 0)
        cv2.putText(image, text, (x, y), font, scale_text, text_color, thickness, cv2.LINE_AA)

    def show_frame(self, frame, store_last=True):
        h, w = frame.shape[:2]
        max_w = self.video_label.width()
        max_h = self.video_label.height()
        if max_w <= 1 or max_h <= 1:
            return
        scale = min(max_w / w, max_h / h)

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        resized = self._apply_preview_effects(resized)

        section = self.annotator.section_at_frame(self.annotator.current_frame)
        if section is not None:
            section_label = section.label.strip() or f"Section {section.section_id}"
            self._draw_badge(
                resized,
                f"Section {section.section_id} | {section_label}",
                18,
                38,
                "#62c58f",
            )

        lab = self.annotator.get_label(self.annotator.current_frame)
        if lab is not None:
            mode = lab.get("mode", "")
            label = lab.get("group_ID") or lab.get("group", "")
            frame_txt = self._format_frame_display(
                self.annotator.current_frame)
            txt = f"{frame_txt} | {mode} | {label}"

            color_hex = self.color_for_mode(mode) if mode else DEFAULT_COLOR
            self._draw_badge(resized, txt, 18, new_h - 28, color_hex, align_center=True)

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
            self.active_section_status_label.setText("Section: none")
            return

        f = self.annotator.current_frame
        lab = self.annotator.get_label(f)
        label_text = "unlabeled" if lab is None else f"{lab['mode']} | {lab.get('group_ID') or lab.get('group', '')}"
        frame_txt = self._format_frame_display(f)
        self.info_label.setText(f"Frame {frame_txt} | {label_text}")
        section = self.annotator.section_at_frame(f)
        if section is None:
            self.active_section_status_label.setText("Section: none")
        else:
            label = section.label.strip() or f"Section {section.section_id}"
            self.active_section_status_label.setText(
                f"Section: {section.section_id} | {label} | {section.group_id}"
            )

    # ==================================================
    # CSV save
    # ==================================================
    def save_csv(self):
        if not self.annotator.cap:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        self._flush_autosave()
        self.annotator.save_csv()
        QMessageBox.information(self, "Saved", "CSV saved")

    def _schedule_autosave(self):
        self.autosave_timer.start(1500)

    def _flush_autosave(self):
        if self.autosave_timer.isActive():
            self.autosave_timer.stop()
        if self.annotator.dirty:
            self.annotator.save_csv()

    def _set_annotation(self, frame: int, mode: str, group: str):
        section = self.annotator.section_at_frame(frame)
        self.annotator.set_label(frame, mode, group, section, autosave=False)
        self._schedule_autosave()

    def export_calculated_csv(self):
        if not self.annotator.cap:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        self._flush_autosave()
        base, _ = os.path.splitext(self.annotator.path or "labels")
        default_path = base + "_calculated.csv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Calculated CSV",
            default_path,
            "CSV Files (*.csv)",
        )
        if not path:
            return
        self.annotator.export_calculated_csv(path)
        QMessageBox.information(self, "Saved", f"Calculated CSV exported:\n{path}")

    def _safe_filename(self, text: str) -> str:
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
        return text.strip("_") or "section"

    def export_selected_section_video(self):
        if not self.annotator.cap or not self.annotator.path:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        section = self._selected_section() or self.annotator.section_at_frame(self.annotator.current_frame)
        if section is None:
            QMessageBox.warning(self, "Warning", "Select a section first")
            return
        self.selected_section_id = section.section_id
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            QMessageBox.warning(
                self,
                "FFmpeg not found",
                "Install FFmpeg to export selected sections without re-encoding.",
            )
            return
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose Export Folder",
            os.path.dirname(self.annotator.path),
        )
        if not output_dir:
            return

        base_name = os.path.splitext(os.path.basename(self.annotator.path))[0]
        label = self._safe_filename(section.label or f"section_{section.section_id}")
        output_path = os.path.join(
            output_dir,
            f"{base_name}_section-{section.section_id:03d}_{label}.mp4",
        )
        start_seconds = section.start_frame / max(self.annotator.fps, 1.0)
        end_seconds = (section.end_frame + 1) / max(self.annotator.fps, 1.0)
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_seconds:.6f}",
            "-to",
            f"{end_seconds:.6f}",
            "-i",
            self.annotator.path,
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            output_path,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            QMessageBox.warning(
                self,
                "Export failed",
                exc.stderr[-1200:] if exc.stderr else "FFmpeg export failed.",
            )
            return
        QMessageBox.information(self, "Exported", f"Video exported:\n{output_path}")

    def _on_preview_effect_changed(self, _value=None):
        self.preview_brightness = self.brightness_slider.value()
        self.preview_contrast = self.contrast_slider.value()
        self.brightness_value_label.setText(str(self.preview_brightness))
        self.contrast_value_label.setText(f"{self.preview_contrast}%")
        if self.last_frame_np is not None:
            self.show_frame(self.last_frame_np, store_last=False)

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
                self._set_annotation(idx, info["mode"], info.get("group", ""))
                self.update_timeline()
                self.refresh_help_label()

        self.annotator.current_frame = idx
        self.show_frame(frame)
        self._set_seek_slider_value(idx)
        self.update_info_label()
        self.update_waveform()
        self._scroll_to_frame(idx)

    def _set_seek_slider_value(self, idx: int):
        previous = self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(idx)
        self.seek_slider.blockSignals(previous)

    def prev_frame(self):
        idx = max(0, self.annotator.current_frame - self.backward_num_frames)
        self.goto_frame(idx)

    def next_frame(self):
        idx = min(self.annotator.frame_count - 1,
                  self.annotator.current_frame + self.forward_num_frames)
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
            self._audio_stop()
        else:
            self._audio_play_from_frame(self.annotator.current_frame)
            self.play_timer.start()

    def play_next_frame(self):
        frame = self.annotator.read_next_frame()
        if frame is None:
            self.toggle_play()
            return

        self.show_frame(frame)
        self._set_seek_slider_value(self.annotator.current_frame)
        self.update_info_label()
        self.update_waveform()
        self._scroll_to_frame(self.annotator.current_frame)

    def _next_encoding_frame_after(self, frame_idx: int) -> int:
        section = self.annotator.section_at_frame(frame_idx)
        if section and frame_idx >= section.end_frame:
            next_section = self.annotator.next_section_after(section.end_frame)
            if next_section:
                return next_section.start_frame
        return min(self.annotator.frame_count - 1, frame_idx + 1)

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
                    self._set_annotation(
                        self.annotator.current_frame, info["mode"], info.get("group", ""))
                    self.update_info_label()
                    self.update_timeline()
                    self.refresh_help_label()
                    return

            if not self.is_long_press:
                info = self.label_map.get(self.active_label_char)
                if info:
                    self._set_annotation(
                        self.annotator.current_frame, info["mode"], info.get("group", ""))
                    self.update_timeline()
                    self.refresh_help_label()
                    next_idx = self._next_encoding_frame_after(self.annotator.current_frame)
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
            self._set_annotation(
                self.annotator.current_frame, info["mode"], info.get("group", ""))
            self.update_timeline()
            self.refresh_help_label()

        next_idx = self._next_encoding_frame_after(self.annotator.current_frame)
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
        self._flush_autosave()
        if self.annotator.sections_dirty:
            self.annotator.save_sections()
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
        same_group = (
            prev_label.get("group_ID") or prev_label.get("group", "")
        ) == (
            next_label.get("group_ID") or next_label.get("group", "")
        )
        if not (same_mode and same_group):
            QMessageBox.warning(
                self, "Warning", "When using Fillin, labels before and after the cursor must match")
            return

        fill_mode = prev_label["mode"]
        fill_group = prev_label.get("group_ID") or prev_label.get("group", "")

        filled = 0
        for idx in range(prev_idx + 1, next_idx):
            if self.annotator.annotations.get(idx) is None:
                self.annotator.annotations[idx] = {
                    "mode": fill_mode,
                    "group": fill_group,
                    "group_ID": fill_group,
                    "section_ID": prev_label.get("section_ID", ""),
                    "section_label": prev_label.get("section_label", ""),
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
