import csv
import os
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


EXPECTED_COLUMNS = ["phase", "event", "onset", "offset", "duration"]


@dataclass
class ExperimentCsvSection:
    label: str
    start_seconds: float
    end_seconds: float
    source_row: int


class ExperimentCsvDropLabel(QLabel):
    csvDropped = pyqtSignal(str)

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(42)
        self.setWordWrap(True)
        self.setStyleSheet(
            "border: 1px dashed #4a5360; border-radius: 4px; padding: 8px; color: #cfd2d6;"
        )

    def dragEnterEvent(self, event):
        if self._csv_path_from_event(event) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        path = self._csv_path_from_event(event)
        if path is None:
            event.ignore()
            return
        self.csvDropped.emit(path)
        event.acceptProposedAction()

    def _csv_path_from_event(self, event) -> str | None:
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() == ".csv":
                return path
        return None


class ExperimentCsvImportDialog(QDialog):
    def __init__(self, csv_path: str, timeline_zero_seconds: float, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.headers, self.rows = self._read_csv(csv_path)
        self.imported_sections: list[ExperimentCsvSection] = []

        self.setWindowTitle("Read Experiment CSV File")
        self.resize(1100, 650)

        self.phase_combo = self._build_column_combo(["phase"])
        self.section_combo = self._build_column_combo(["event", "section_ID", "section_label"])
        self.start_combo = self._build_column_combo(["onset", "start", "start_time"])
        self.end_combo = self._build_column_combo(["offset", "end", "end_time"])
        self.phase_value_edit = QLineEdit("stimulus")

        self.timeline_zero_spin = QDoubleSpinBox()
        self.timeline_zero_spin.setRange(0.0, 24 * 60 * 60.0)
        self.timeline_zero_spin.setDecimals(3)
        self.timeline_zero_spin.setSuffix(" sec")
        self.timeline_zero_spin.setValue(max(0.0, timeline_zero_seconds))
        self.timeline_zero_spin.valueChanged.connect(self._update_preview_columns)

        for combo in (
            self.phase_combo,
            self.section_combo,
            self.start_combo,
            self.end_combo,
        ):
            combo.currentTextChanged.connect(self._update_preview_columns)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(
            """
            QTableWidget {
                background-color: #23262E;
                alternate-background-color: #23262E;
                color: #D2CBD5;
                gridline-color: #3a3f45;
            }
            QTableWidget::item {
                background-color: #23262E;
            }
            QTableWidget::item:selected {
                background-color: #3b4854;
                color: #f2f2f2;
            }
            QHeaderView::section {
                background-color: #1b1e22;
                color: #D2CBD5;
                border: 1px solid #3a3f45;
                padding: 4px;
            }
            """
        )

        select_stimulus_button = QPushButton("Select Stimulus Rows")
        select_stimulus_button.clicked.connect(self._select_matching_rows)
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(lambda: self._set_all_rows_checked(True))
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self._set_all_rows_checked(False))

        selection_buttons = QHBoxLayout()
        selection_buttons.addWidget(select_stimulus_button)
        selection_buttons.addWidget(select_all_button)
        selection_buttons.addWidget(clear_button)
        selection_buttons.addStretch()

        form = QFormLayout()
        form.addRow("CSV file", QLabel(os.path.basename(csv_path)))
        form.addRow("Phase column", self.phase_combo)
        form.addRow("Auto-select phase", self.phase_value_edit)
        form.addRow("Section ID column", self.section_combo)
        form.addRow("Start column", self.start_combo)
        form.addRow("End column", self.end_combo)
        form.addRow("CSV time 0 in timeline", self.timeline_zero_spin)

        missing = self._missing_expected_columns()
        self.missing_label = QLabel("")
        self.missing_label.setWordWrap(True)
        if missing:
            self.missing_label.setText(
                "Missing expected columns: " + ", ".join(missing) + ". Choose matching columns manually."
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import Sections")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        if missing:
            layout.addWidget(self.missing_label)
        layout.addLayout(selection_buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._populate_table()
        self._select_matching_rows()
        self._update_preview_columns()

    def _read_csv(self, csv_path: str) -> tuple[list[str], list[dict[str, str]]]:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = [header for header in (reader.fieldnames or []) if header]
            rows = []
            for row in reader:
                rows.append({header: row.get(header, "") or "" for header in headers})
        return headers, rows

    def _build_column_combo(self, preferred_names: list[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(self.headers)
        target = self._find_column(preferred_names)
        if target:
            combo.setCurrentText(target)
        return combo

    def _find_column(self, preferred_names: list[str]) -> str:
        lower_to_header = {header.lower(): header for header in self.headers}
        for name in preferred_names:
            header = lower_to_header.get(name.lower())
            if header:
                return header
        return self.headers[0] if self.headers else ""

    def _missing_expected_columns(self) -> list[str]:
        present = {header.lower() for header in self.headers}
        return [name for name in EXPECTED_COLUMNS if name.lower() not in present]

    def _populate_table(self):
        columns = ["Use", "CSV Row", "Timeline Start", "Timeline End", "Section ID"]
        columns.extend(self.headers)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(self.rows))

        for row_index, row in enumerate(self.rows):
            use_item = self._table_item("", row_index)
            use_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            use_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_index, 0, use_item)
            self.table.setItem(row_index, 1, self._table_item(str(row_index + 2), row_index))
            self.table.setItem(row_index, 2, self._table_item("", row_index))
            self.table.setItem(row_index, 3, self._table_item("", row_index))
            self.table.setItem(row_index, 4, self._table_item("", row_index))
            for col_index, header in enumerate(self.headers, start=5):
                self.table.setItem(
                    row_index,
                    col_index,
                    self._table_item(row.get(header, ""), row_index),
                )

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for col in range(min(5, self.table.columnCount())):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.resizeColumnsToContents()

    def _table_item(self, text: str, row_index: int) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setBackground(QBrush(QColor("#23262E")))
        text_color = "#D2CBD5" if row_index % 2 == 0 else "#999999"
        item.setForeground(QBrush(QColor(text_color)))
        return item

    def _set_all_rows_checked(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 0)
            if item:
                item.setCheckState(state)

    def _select_matching_rows(self):
        phase_col = self.phase_combo.currentText()
        phase_value = self.phase_value_edit.text().strip().lower()
        for row_index, row in enumerate(self.rows):
            value = row.get(phase_col, "").strip().lower()
            checked = bool(phase_value) and value == phase_value
            item = self.table.item(row_index, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _update_preview_columns(self):
        timeline_zero = self.timeline_zero_spin.value()
        section_col = self.section_combo.currentText()
        start_col = self.start_combo.currentText()
        end_col = self.end_combo.currentText()

        for row_index, row in enumerate(self.rows):
            label = row.get(section_col, "")
            start_value = self._to_float(row.get(start_col, ""))
            end_value = self._to_float(row.get(end_col, ""))
            self.table.item(row_index, 4).setText(label)
            self.table.item(row_index, 2).setText(
                "" if start_value is None else f"{timeline_zero + start_value:.3f}"
            )
            self.table.item(row_index, 3).setText(
                "" if end_value is None else f"{timeline_zero + end_value:.3f}"
            )

    def _to_float(self, value: str) -> float | None:
        try:
            return float(str(value).strip())
        except Exception:
            return None

    def accept(self):
        sections, errors = self._collect_sections()
        if errors:
            preview = "\n".join(errors[:8])
            if len(errors) > 8:
                preview += f"\n...and {len(errors) - 8} more"
            QMessageBox.warning(self, "Import Sections", preview)
            return
        if not sections:
            QMessageBox.warning(self, "Import Sections", "Select at least one row to import.")
            return
        self.imported_sections = sections
        super().accept()

    def _collect_sections(self) -> tuple[list[ExperimentCsvSection], list[str]]:
        timeline_zero = self.timeline_zero_spin.value()
        section_col = self.section_combo.currentText()
        start_col = self.start_combo.currentText()
        end_col = self.end_combo.currentText()
        sections: list[ExperimentCsvSection] = []
        errors: list[str] = []

        for row_index, row in enumerate(self.rows):
            use_item = self.table.item(row_index, 0)
            if not use_item or use_item.checkState() != Qt.CheckState.Checked:
                continue

            csv_row = row_index + 2
            label = row.get(section_col, "").strip()
            start_value = self._to_float(row.get(start_col, ""))
            end_value = self._to_float(row.get(end_col, ""))
            if not label:
                errors.append(f"Row {csv_row}: section ID is empty.")
                continue
            if start_value is None or end_value is None:
                errors.append(f"Row {csv_row}: start or end time is not numeric.")
                continue
            start_seconds = timeline_zero + start_value
            end_seconds = timeline_zero + end_value
            if end_seconds <= start_seconds:
                errors.append(f"Row {csv_row}: end time must be after start time.")
                continue
            sections.append(
                ExperimentCsvSection(
                    label=label,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    source_row=csv_row,
                )
            )

        return sections, errors
