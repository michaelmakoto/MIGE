from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Encoding Settings")
        self.resize(760, 560)

        self.default_duration_input = QDoubleSpinBox()
        self.default_duration_input.setRange(0.1, 3600.0)
        self.default_duration_input.setDecimals(2)
        self.default_duration_input.setSuffix(" sec")
        self.default_duration_input.setValue(settings.default_section_seconds())

        self.groups_input = QLineEdit(", ".join(settings.groups))

        form = QFormLayout()
        form.addRow("Default section duration", self.default_duration_input)
        form.addRow("Group IDs", self.groups_input)

        self.label_table = QTableWidget(0, 4)
        self.label_table.setHorizontalHeaderLabels(["Key", "Mode", "Group ID", "Color"])
        self.label_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for key, data in settings.labels.items():
            self._add_label_row(key, data.get("name", ""), data.get("group", ""), data.get("color", "#AAAAAA"))

        self.app_table = QTableWidget(0, 2)
        self.app_table.setHorizontalHeaderLabels(["Key", "Action"])
        self.app_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for key, action in settings.app_keys.items():
            self._add_app_row(key, action)

        add_label_button = QPushButton("+ Label")
        add_label_button.clicked.connect(lambda: self._add_label_row("", "", "", "#AAAAAA"))
        add_app_button = QPushButton("+ App Key")
        add_app_button.clicked.connect(lambda: self._add_app_row("", ""))

        label_buttons = QHBoxLayout()
        label_buttons.addWidget(add_label_button)
        label_buttons.addStretch()

        app_buttons = QHBoxLayout()
        app_buttons.addWidget(add_app_button)
        app_buttons.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(QLabel("Label keys"))
        layout.addWidget(self.label_table, 1)
        layout.addLayout(label_buttons)
        layout.addWidget(QLabel("App keys"))
        layout.addWidget(self.app_table, 1)
        layout.addLayout(app_buttons)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _add_label_row(self, key: str, mode: str, group: str, color: str):
        row = self.label_table.rowCount()
        self.label_table.insertRow(row)
        for col, value in enumerate([key, mode, group, color]):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label_table.setItem(row, col, item)

    def _add_app_row(self, key: str, action: str):
        row = self.app_table.rowCount()
        self.app_table.insertRow(row)
        for col, value in enumerate([key, action]):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.app_table.setItem(row, col, item)

    def _table_text(self, table: QTableWidget, row: int, col: int) -> str:
        item = table.item(row, col)
        return item.text().strip() if item else ""

    def save_settings(self):
        labels = {}
        for row in range(self.label_table.rowCount()):
            key = self._table_text(self.label_table, row, 0).lower()
            mode = self._table_text(self.label_table, row, 1)
            group = self._table_text(self.label_table, row, 2)
            color = self._table_text(self.label_table, row, 3) or "#AAAAAA"
            if not key or not mode:
                continue
            labels[key] = {"name": mode, "group": group, "color": color}

        app_keys = {}
        for row in range(self.app_table.rowCount()):
            key = self._table_text(self.app_table, row, 0).lower()
            action = self._table_text(self.app_table, row, 1)
            if key and action:
                app_keys[key] = action

        groups = [
            group.strip()
            for group in self.groups_input.text().split(",")
            if group.strip()
        ]

        self.settings.data["labels"] = labels
        self.settings.data["app_keys"] = app_keys
        self.settings.data["groups"] = groups
        self.settings.data.setdefault("sections", {})[
            "default_duration_seconds"
        ] = self.default_duration_input.value()
        self.settings.save()
        self.accept()
