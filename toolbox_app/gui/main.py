from __future__ import annotations

import json
import typing
import types
from datetime import datetime
import shutil
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, get_args, get_origin

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pydantic import BaseModel

APP_STYLESHEET = """
QWidget {
    color: #1f2933;
}
QWidget#AppRoot {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f7f2e9, stop:1 #e1edf7);
}
QListWidget#ToolList {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(17, 24, 39, 0.12);
    border-radius: 14px;
    padding: 6px;
}
QListWidget#ToolList::item {
    padding: 8px 10px;
    margin: 2px 4px;
    border-radius: 8px;
}
QListWidget#ToolList::item:selected {
    background: #1f4b6e;
    color: #f8fafc;
}
QWidget#RightPanel {
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid rgba(17, 24, 39, 0.1);
    border-radius: 16px;
}
QLabel#TitleLabel {
    font-size: 18px;
    font-weight: 600;
    color: #0f172a;
}
QLabel#HomeTitle {
    font-size: 20px;
    font-weight: 600;
    color: #0f172a;
}
QLabel#HomeSubtitle,
QLabel#SubtitleLabel,
QLabel#DescriptionLabel {
    color: #475569;
}
QLabel#SectionLabel {
    font-weight: 600;
    color: #0f172a;
}
QLabel#StatusValue {
    color: #1f4b6e;
}
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QTextEdit {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(17, 24, 39, 0.15);
    border-radius: 8px;
    padding: 6px 8px;
}
QComboBox::drop-down {
    border: none;
}
QPushButton {
    background: #1f4b6e;
    color: #f8fafc;
    border: none;
    border-radius: 8px;
    padding: 6px 12px;
}
QPushButton:disabled {
    background: #a8b3c1;
    color: #f8fafc;
}
QPushButton#SecondaryButton {
    background: #e7edf3;
    color: #1f2937;
    border: 1px solid rgba(17, 24, 39, 0.12);
}
QProgressBar {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(17, 24, 39, 0.12);
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: #0f172a;
}
QProgressBar::chunk {
    background: #3b82f6;
    border-radius: 7px;
}
QWidget#OutputPanel {
    background: rgba(255, 255, 255, 0.6);
    border: 1px dashed rgba(17, 24, 39, 0.16);
    border-radius: 12px;
}
QSplitter::handle {
    background: rgba(17, 24, 39, 0.08);
}
"""

from toolbox_app.core.loader import discover_tools
from toolbox_app.core.logging import configure_logging
from toolbox_app.core.runner import ToolRunner
from toolbox_app.core.schema_utils import validate_inputs
from toolbox_app.core.paths import user_data_dir
from toolbox_app.core.settings import load_settings, save_settings


def _is_optional(t: Any) -> bool:
    origin = get_origin(t)
    if origin in (typing.Union, types.UnionType):
        return type(None) in get_args(t)
    return False


def _strip_optional(t: Any) -> Any:
    if _is_optional(t):
        args = [a for a in get_args(t) if a is not type(None)]
        return args[0] if len(args) == 1 else t
    return t


class TrimmedDoubleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:  # type: ignore[override]
        text = self.locale().toString(value, "f", self.decimals())
        dec = self.locale().decimalPoint()
        if dec in text:
            text = text.rstrip("0").rstrip(dec)
        return text


class FieldWidget:
    def __init__(self, widget, kind: str, field_name: str) -> None:
        self.widget = widget
        self.kind = kind
        self.field_name = field_name

    def get_value(self) -> Any:
        w = self.widget
        if self.kind == "int":
            return int(w.value())
        if self.kind == "float":
            return float(w.value())
        if self.kind == "bool":
            return bool(w.isChecked())
        if self.kind == "path":
            return w.text().strip()
        if self.kind == "enum":
            data = w.currentData()
            if data is None:
                text = w.currentText().strip()
                return None if text == "" else text
            return data
        return w.text().strip()

    def clear(self) -> None:
        w = self.widget
        if self.kind == "enum":
            if w.count() > 0:
                w.setCurrentIndex(0)
            return
        if self.kind in ("path", "str"):
            w.setText("")
            return
        if self.kind == "bool":
            w.setChecked(False)
            return

    def set_enabled(self, enabled: bool) -> None:
        self.widget.setEnabled(enabled)

    def set_value(self, value: Any) -> None:
        w = self.widget
        if self.kind == "int":
            w.setValue(int(value))
            return
        if self.kind == "float":
            w.setValue(float(value))
            return
        if self.kind == "bool":
            w.setChecked(bool(value))
            return
        if self.kind == "enum":
            if value is None or value == "":
                if w.count() > 0:
                    w.setCurrentIndex(0)
                return
            idx = w.findData(str(value))
            if idx >= 0:
                w.setCurrentIndex(idx)
            return
        if self.kind in ("path", "str"):
            w.setText("" if value is None else str(value))
            return


class HomeWidget(QWidget):
    tool_requested = Signal(str)  # tool_id

    def __init__(self, tools: List[Any]) -> None:
        super().__init__()
        self._tools = tools
        self._tool_by_id = {t.meta.id: t for t in tools}
        self._settings = load_settings()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        title = QLabel("Home")
        title.setObjectName("HomeTitle")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        title.setFont(f)

        subtitle = QLabel("Search or pick a tool. Recents are stored locally per user.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("HomeSubtitle")

        # Controls row
        controls = QWidget()
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search tools (name, category, id)...")
        self.search.textChanged.connect(self._refresh_lists)

        self.category = QComboBox()
        self._refresh_categories()
        self.category.currentIndexChanged.connect(self._refresh_lists)

        cl.addWidget(self.search, 1)
        cl.addWidget(self.category)

        # Recent list
        self.recent_label = QLabel("Recent tools")
        self.recent_label.setObjectName("SectionLabel")
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._recent_open)

        # All tools list
        self.all_label = QLabel("All tools")
        self.all_label.setObjectName("SectionLabel")
        self.all_list = QListWidget()
        self.all_list.itemDoubleClicked.connect(self._all_open)

        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addWidget(controls)
        lay.addWidget(self.recent_label)
        lay.addWidget(self.recent_list, 1)
        lay.addWidget(self.all_label)
        lay.addWidget(self.all_list, 3)

        self._refresh_lists()

    def set_tools(self, tools: List[Any]) -> None:
        self._tools = tools
        self._tool_by_id = {t.meta.id: t for t in tools}
        self._refresh_categories()
        self._refresh_lists()

    def _refresh_categories(self) -> None:
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("All categories", None)
        cats = sorted({t.meta.category for t in self._tools})
        for c in cats:
            self.category.addItem(c, c)
        self.category.blockSignals(False)

    def _matches_filters(self, tool: Any) -> bool:
        q = (self.search.text() or "").strip().lower()
        cat = self.category.currentData()

        if cat and tool.meta.category != cat:
            return False

        if not q:
            return True

        hay = f"{tool.meta.id} {tool.meta.name} {tool.meta.category}".lower()
        return q in hay

    def _recent_ids(self) -> List[str]:
        recents = self._settings.get("recent_tools", [])
        if isinstance(recents, list):
            return [str(x) for x in recents]
        return []

    def _refresh_lists(self) -> None:
        # Recent list
        self.recent_list.clear()
        recent_ids = self._recent_ids()

        shown_any_recent = False
        for tid in recent_ids:
            t = self._tool_by_id.get(tid)
            if not t:
                continue
            if not self._matches_filters(t):
                continue
            item = QListWidgetItem(f"{t.meta.category} • {t.meta.name}")
            item.setData(Qt.UserRole, t.meta.id)
            self.recent_list.addItem(item)
            shown_any_recent = True

        self.recent_label.setVisible(shown_any_recent)
        self.recent_list.setVisible(shown_any_recent)

        # All tools list
        self.all_list.clear()
        for t in sorted(self._tools, key=lambda x: (x.meta.category.lower(), x.meta.name.lower())):
            if not self._matches_filters(t):
                continue
            item = QListWidgetItem(f"{t.meta.category} • {t.meta.name}")
            item.setData(Qt.UserRole, t.meta.id)
            self.all_list.addItem(item)

    def _recent_open(self, item: QListWidgetItem) -> None:
        tid = item.data(Qt.UserRole)
        if tid:
            self.tool_requested.emit(str(tid))

    def _all_open(self, item: QListWidgetItem) -> None:
        tid = item.data(Qt.UserRole)
        if tid:
            self.tool_requested.emit(str(tid))

    def note_recent(self, tool_id: str) -> None:
        self._settings = load_settings()
        recents = self._settings.get("recent_tools", [])
        if not isinstance(recents, list):
            recents = []
        # Move to front, unique, cap length
        recents = [x for x in recents if str(x) != tool_id]
        recents.insert(0, tool_id)
        recents = recents[:10]
        self._settings["recent_tools"] = recents
        save_settings(self._settings)
        self._refresh_lists()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Engineering Toolbox")
        self.resize(1180, 740)

        self.tools: List[Any] = []
        self.tool_by_id: Dict[str, Any] = {}
        self.active_tool_id: Optional[str] = None

        self.runner = ToolRunner()
        self.field_widgets: Dict[str, FieldWidget] = {}
        self._shape_family_field: Optional[FieldWidget] = None
        self._section_fields: List[tuple[FieldWidget, str]] = []
        self._material_preset_bindings: List[dict[str, Any]] = []
        self._last_output: Any = None
        self._last_tool_id: Optional[str] = None
        self._choices_cache: Dict[str, List[str]] = {}
        self._aisc_labels_cache: Optional[Dict[str, List[str]]] = None
        self._aisc_hss_cache: Optional[tuple[List[str], List[str]]] = None

        splitter = QSplitter(Qt.Horizontal)

        # Left: tool list (always available)
        self.list = QListWidget()
        self.list.setMinimumWidth(330)
        self.list.setObjectName("ToolList")
        self.list.setFrameShape(QFrame.NoFrame)

        self._rebuild_tool_list(loading=True)
        self.list.setEnabled(False)

        self.list.currentItemChanged.connect(self.on_tool_changed)
        splitter.addWidget(self.list)

        # Right: panel
        right = QWidget()
        right.setObjectName("RightPanel")
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(16, 16, 16, 16)
        rlay.setSpacing(10)

        # Smart Home widget
        self.home = HomeWidget([])
        self.home.tool_requested.connect(self._select_tool_by_id)

        # Tool header
        self.title = QLabel("")
        self.title.setObjectName("TitleLabel")
        f = QFont()
        f.setPointSize(16)
        f.setBold(True)
        self.title.setFont(f)

        self.sub = QLabel("")
        self.sub.setObjectName("SubtitleLabel")
        self.sub.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setObjectName("DescriptionLabel")

        # Form
        self.form = QWidget()
        form_row = QHBoxLayout(self.form)
        form_row.setContentsMargins(0, 0, 0, 0)
        form_row.setSpacing(16)
        self.form_left = QWidget()
        self.form_right = QWidget()
        self.form_layout_left = QFormLayout(self.form_left)
        self.form_layout_right = QFormLayout(self.form_right)
        self.form_layout_left.setLabelAlignment(Qt.AlignRight)
        self.form_layout_right.setLabelAlignment(Qt.AlignRight)
        form_row.addWidget(self.form_left, 1)
        form_row.addWidget(self.form_right, 1)
        self._form_row_index = 0
        self._field_label_by_widget: Dict[QWidget, QLabel] = {}

        # Actions
        btn_row = QWidget()
        self.btn_row = btn_row
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(8)

        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.run_active_tool)
        self.run_btn.setEnabled(False)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("SecondaryButton")
        self.cancel_btn.clicked.connect(self.cancel_run)
        self.cancel_btn.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)

        btn_lay.addWidget(self.run_btn)
        btn_lay.addWidget(self.cancel_btn)
        btn_lay.addWidget(self.progress, 1)

        # Status + details
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("StatusValue")

        status_title = QLabel("Status")
        status_title.setObjectName("SectionLabel")

        self.output_toggle = QPushButton("Details")
        self.output_toggle.setCheckable(True)
        self.output_toggle.setChecked(False)
        self.output_toggle.setObjectName("SecondaryButton")
        self.output_toggle.toggled.connect(self._toggle_output)

        status_row = QWidget()
        status_lay = QHBoxLayout(status_row)
        status_lay.setContentsMargins(0, 0, 0, 0)
        status_lay.setSpacing(8)
        status_lay.addWidget(status_title)
        status_lay.addWidget(self.status_label, 1)
        status_lay.addWidget(self.output_toggle)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setVisible(False)
        self.output.setMaximumHeight(140)

        self.output_panel = QWidget()
        self.output_panel.setObjectName("OutputPanel")
        op_lay = QVBoxLayout(self.output_panel)
        op_lay.setContentsMargins(10, 8, 10, 8)
        op_lay.setSpacing(6)

        self.actions_row = QWidget()
        actions_lay = QHBoxLayout(self.actions_row)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        actions_lay.setSpacing(8)

        self.btn_capture = QPushButton("Capture results")
        self.btn_export_excel = QPushButton("Export to Excel")
        self.btn_export_mathcad = QPushButton("Mathcad handoff")
        self.btn_save_report = QPushButton("Save report HTML")
        for btn in (self.btn_capture, self.btn_export_excel, self.btn_export_mathcad, self.btn_save_report):
            btn.setObjectName("SecondaryButton")
            btn.setEnabled(False)

        self.btn_capture.clicked.connect(self._capture_results)
        self.btn_export_excel.clicked.connect(self._export_excel)
        self.btn_export_mathcad.clicked.connect(self._export_mathcad)
        self.btn_save_report.clicked.connect(self._save_report_html)

        actions_lay.addWidget(self.btn_capture)
        actions_lay.addWidget(self.btn_export_excel)
        actions_lay.addWidget(self.btn_export_mathcad)
        actions_lay.addWidget(self.btn_save_report)
        actions_lay.addStretch(1)

        op_lay.addWidget(self.actions_row)
        op_lay.addWidget(status_row)
        op_lay.addWidget(self.output)

        # Report panel (full-screen view)
        self.report_panel = QWidget()
        self.report_panel.setVisible(False)
        rp_lay = QVBoxLayout(self.report_panel)
        rp_lay.setContentsMargins(0, 0, 0, 0)
        rp_lay.setSpacing(8)

        report_actions = QWidget()
        ra_lay = QHBoxLayout(report_actions)
        ra_lay.setContentsMargins(0, 0, 0, 0)
        ra_lay.setSpacing(8)

        self.btn_report_back = QPushButton("Back to inputs")
        self.btn_report_back.setObjectName("SecondaryButton")
        self.btn_report_back.clicked.connect(self._show_inputs_view)
        ra_lay.addWidget(self.btn_report_back)
        ra_lay.addStretch(1)

        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)

        rp_lay.addWidget(report_actions)
        rp_lay.addWidget(self.report_view, 1)

        # Add widgets (home first; tool panel below)
        rlay.addWidget(self.home, 1)

        rlay.addWidget(self.title)
        rlay.addWidget(self.sub)
        rlay.addWidget(self.desc)
        rlay.addWidget(self.form)
        rlay.addWidget(self.report_panel, 1)
        rlay.addWidget(btn_row)
        rlay.addWidget(self.output_panel)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        container.setObjectName("AppRoot")
        lay = QHBoxLayout(container)
        lay.addWidget(splitter)
        self.setCentralWidget(container)

        # Start on Home
        self.list.setCurrentRow(0)  # Home
        self._show_home()
        QTimer.singleShot(0, self._load_tools)

    def _select_tool_by_id(self, tool_id: str) -> None:
        # Select the matching item in the left list
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.UserRole) == tool_id:
                self.list.setCurrentRow(i)
                return

    def _show_home(self) -> None:
        self.active_tool_id = None

        # Hide tool panel widgets by clearing them
        self.title.setText("")
        self.sub.setText("")
        self.desc.setText("")
        self.report_panel.setVisible(False)
        self._clear_form_layouts()
        self.field_widgets.clear()
        self._shape_family_field = None
        self._section_fields = []
        self._material_preset_bindings = []
        self._last_output = None
        self._last_tool_id = None
        self._set_action_buttons_enabled(False)
        self.output.clear()
        self.status_label.setText("Ready.")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)
        self.progress.setValue(0)
        self.output_panel.setVisible(False)
        self.btn_row.setVisible(False)

        # Make home visible
        self.home.setVisible(True)

    def _load_tools(self) -> None:
        self.status_label.setText("Loading tools...")
        try:
            tools = discover_tools()
        except Exception as exc:
            QMessageBox.critical(self, "Tool load failed", str(exc))
            tools = []
        self.tools = tools
        self.tool_by_id = {t.meta.id: t for t in self.tools}
        self.home.set_tools(self.tools)
        self._rebuild_tool_list(loading=False)
        self.list.setEnabled(True)
        self._show_home()

    def _rebuild_tool_list(self, loading: bool = False) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        home_item = QListWidgetItem("Home")
        home_item.setData(Qt.UserRole, "__HOME__")
        self.list.addItem(home_item)
        if loading:
            loading_item = QListWidgetItem("Loading tools...")
            loading_item.setFlags(loading_item.flags() & ~Qt.ItemIsEnabled)
            self.list.addItem(loading_item)
        else:
            for t in self.tools:
                item = QListWidgetItem(f"{t.meta.category} • {t.meta.name}")
                item.setData(Qt.UserRole, t.meta.id)
                self.list.addItem(item)
        if self.list.count() > 0:
            self.list.setCurrentRow(0)
        self.list.blockSignals(False)

    def _show_tool_panel(self) -> None:
        self.home.setVisible(False)
        self.report_panel.setVisible(False)
        self.output_panel.setVisible(True)
        self.btn_row.setVisible(True)

    def on_tool_changed(self, current: QListWidgetItem, _prev: QListWidgetItem) -> None:
        if current is None:
            return

        tool_id = current.data(Qt.UserRole)
        if tool_id == "__HOME__":
            self._show_home()
            return

        tool = self.tool_by_id.get(tool_id)
        if tool is None:
            return

        self._show_tool_panel()
        self.active_tool_id = str(tool_id)

        self.title.setText(tool.meta.name)
        self.sub.setText(
            f"ID: {tool.meta.id}     Version: {tool.meta.version}     Category: {tool.meta.category}"
        )
        self.desc.setText(tool.meta.description)
        self.sub.setVisible(True)
        self.desc.setVisible(True)
        self.form.setVisible(True)

        # Record recent immediately on selection
        self.home.note_recent(tool.meta.id)

        # Clear old form
        self._clear_form_layouts()
        self.field_widgets.clear()
        self._shape_family_field = None
        self._section_fields = []
        self._material_preset_bindings = []
        self._last_output = None
        self._last_tool_id = None
        self._set_action_buttons_enabled(False)

        # Build form from Pydantic schema if available
        model: Optional[Type[BaseModel]] = getattr(tool, "InputModel", None)

        if model is not None:
            defaults = model().model_dump()
            for fname, field in model.model_fields.items():  # type: ignore[attr-defined]
                label = field.title or fname
                if self.active_tool_id == "c49_overhang_bracket" and field.description:
                    label = field.description
                hint = field.description or ""
                choices = None
                choices_source = None
                ui_tab = None
                presets = None
                custom_value = None
                preset_targets = None
                override_field = None
                if isinstance(field.json_schema_extra, dict):
                    choices = field.json_schema_extra.get("choices")
                    choices_source = field.json_schema_extra.get("choices_source")
                    ui_tab = field.json_schema_extra.get("ui_tab")
                    presets = field.json_schema_extra.get("material_presets")
                    custom_value = field.json_schema_extra.get("custom_value")
                    preset_targets = field.json_schema_extra.get("preset_targets")
                    override_field = field.json_schema_extra.get("override_field")
                if choices is None and choices_source:
                    choices = self._load_dynamic_choices(choices_source)
                w = self._widget_for_field(fname, field.annotation, defaults.get(fname), hint, choices)
                self._add_form_row(label, w.widget)
                self.field_widgets[fname] = w
                if ui_tab:
                    self._section_fields.append((w, str(ui_tab)))
                if presets and preset_targets:
                    self._material_preset_bindings.append(
                        {
                            "source": fname,
                            "presets": presets,
                            "custom_value": custom_value or "Custom",
                            "targets": preset_targets,
                            "override_field": override_field,
                        }
                    )
                if fname == "shape_family":
                    self._shape_family_field = w
                    if isinstance(w.widget, QComboBox):
                        w.widget.currentIndexChanged.connect(self._apply_section_visibility)
        else:
            defaults = tool.default_inputs()
            for k, v in defaults.items():
                le = QLineEdit(str(v))
                le.setPlaceholderText(k)
                self._add_form_row(k, le)
                self.field_widgets[k] = FieldWidget(le, "str", k)

        self.output.clear()
        self.status_label.setText("Ready.")
        self.run_btn.setEnabled(True)
        self._setup_material_presets()
        self._setup_c49_girder_defaults()
        self._apply_section_visibility()

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self.btn_capture, self.btn_export_excel, self.btn_export_mathcad, self.btn_save_report):
            btn.setEnabled(enabled)

    def _show_report_view(self, html: str) -> None:
        self.report_view.setHtml(str(html))
        self.report_panel.setVisible(True)
        self.form.setVisible(False)
        self.btn_row.setVisible(False)
        self.output_panel.setVisible(False)
        self.sub.setVisible(False)
        self.desc.setVisible(False)

    def _show_inputs_view(self) -> None:
        if not self.active_tool_id:
            return
        self.report_panel.setVisible(False)
        self.form.setVisible(True)
        self.btn_row.setVisible(True)
        self.output_panel.setVisible(True)
        self.sub.setVisible(True)
        self.desc.setVisible(True)

    def _get_output_dir(self) -> Path:
        if isinstance(self._last_output, dict):
            run_dir = self._last_output.get("run_dir")
            if run_dir:
                p = Path(str(run_dir))
                p.mkdir(parents=True, exist_ok=True)
                return p
        tool_id = self._last_tool_id or "tool"
        p = user_data_dir() / "outputs" / tool_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _capture_results(self) -> None:
        if not isinstance(self._last_output, dict):
            QMessageBox.information(self, "No results", "Run a tool first.")
            return
        out_dir = self._get_output_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"capture_{stamp}.json"
        try:
            path.write_text(json.dumps(self._last_output, indent=2), encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Capture failed", str(e))
            return
        QMessageBox.information(self, "Captured", f"Saved:\n{path}")

    def _export_excel(self) -> None:
        if not isinstance(self._last_output, dict):
            QMessageBox.information(self, "No results", "Run a tool first.")
            return
        exports = self._last_output.get("exports", {})
        src = exports.get("excel_xlsx") if isinstance(exports, dict) else None
        if not src or not Path(str(src)).exists():
            QMessageBox.warning(self, "Export unavailable", "No Excel export was generated for this run.")
            return
        out_dir = self._get_output_dir()
        default_path = out_dir / Path(str(src)).name
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel export", str(default_path), "Excel Workbook (*.xlsx)")
        if not path:
            return
        try:
            shutil.copyfile(str(src), path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Excel export", f"Saved:\n{path}")

    def _export_mathcad(self) -> None:
        if not isinstance(self._last_output, dict):
            QMessageBox.information(self, "No results", "Run a tool first.")
            return
        exports = self._last_output.get("exports", {})
        if not isinstance(exports, dict):
            QMessageBox.warning(self, "Export unavailable", "No Mathcad export was generated for this run.")
            return
        srcs = [
            exports.get("mathcad_json"),
            exports.get("mathcad_csv"),
            exports.get("mathcad_assignments_txt"),
        ]
        srcs = [s for s in srcs if s and Path(str(s)).exists()]
        if not srcs:
            QMessageBox.warning(self, "Export unavailable", "No Mathcad export files were generated for this run.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select folder for Mathcad handoff", str(self._get_output_dir()))
        if not out_dir:
            return
        try:
            for src in srcs:
                shutil.copyfile(str(src), str(Path(out_dir) / Path(str(src)).name))
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Mathcad handoff", f"Wrote files to:\n{out_dir}")

    def _save_report_html(self) -> None:
        if not isinstance(self._last_output, dict):
            QMessageBox.information(self, "No results", "Run a tool first.")
            return
        html = self._last_output.get("report_html")
        exports = self._last_output.get("exports", {})
        src = exports.get("report_html") if isinstance(exports, dict) else None
        if not html and not src:
            QMessageBox.information(self, "Not available", "This tool did not provide an HTML report.")
            return
        out_dir = self._get_output_dir()
        default_path = out_dir / "report.html"
        path, _ = QFileDialog.getSaveFileName(self, "Save report HTML", str(default_path), "HTML (*.html)")
        if not path:
            return
        try:
            if html:
                Path(path).write_text(str(html), encoding="utf-8")
            elif src:
                shutil.copyfile(str(src), path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        QMessageBox.information(self, "Report saved", f"Saved:\n{path}")

    def _load_dynamic_choices(self, meta: Any) -> List[str]:
        try:
            key = json.dumps(meta, sort_keys=True)
        except Exception:
            key = str(meta)
        if key in self._choices_cache:
            return self._choices_cache[key]

        provider = meta.get("provider") if isinstance(meta, dict) else None
        choices: List[str] = []

        if provider == "aisc_shapes_v16":
            try:
                if self._aisc_labels_cache is None:
                    # Centralized AISC Shapes DB lives under toolbox_app.blocks.
                    # (Tool-local shapes_db.py was removed; keep UI dropdowns working.)
                    from toolbox_app.blocks.aisc_shapes_db import ShapeDatabase

                    db = ShapeDatabase()
                    self._aisc_labels_cache = db.list_labels_by_typecode()
                    self._aisc_hss_cache = db.partition_hss_labels(
                        self._aisc_labels_cache.get("HSS", [])
                    )
                labels = self._aisc_labels_cache or {}
                hss = meta.get("hss")
                if hss == "rect":
                    choices = list(self._aisc_hss_cache[0]) if self._aisc_hss_cache else []
                elif hss == "round":
                    choices = list(self._aisc_hss_cache[1]) if self._aisc_hss_cache else []
                else:
                    type_codes = meta.get("type_codes") or []
                    for code in type_codes:
                        choices.extend(labels.get(code, []))
            except Exception:
                choices = []

        # Dedupe while preserving order
        seen: set[str] = set()
        ordered: List[str] = []
        for item in choices:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)

        self._choices_cache[key] = ordered
        return ordered

    def _clear_form_layouts(self) -> None:
        for lay in (self.form_layout_left, self.form_layout_right):
            while lay.rowCount():
                lay.removeRow(0)
        self._form_row_index = 0
        self._field_label_by_widget.clear()

    def _add_form_row(self, label_text: str, widget: QWidget) -> None:
        label = QLabel(str(label_text))
        target = self.form_layout_left if self._form_row_index % 2 == 0 else self.form_layout_right
        target.addRow(label, widget)
        self._field_label_by_widget[widget] = label
        self._form_row_index += 1

    def _widget_for_field(
        self,
        name: str,
        annotation: Any,
        default: Any,
        hint: str,
        choices: Optional[List[str]] = None,
    ) -> FieldWidget:
        is_optional = _is_optional(annotation)
        t = _strip_optional(annotation)

        if choices is not None:
            cb = QComboBox()
            if hint:
                cb.setToolTip(hint)
            if is_optional:
                cb.addItem("", None)
            for choice in choices:
                cb.addItem(str(choice), str(choice))
            if default is not None:
                idx = cb.findData(str(default))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return FieldWidget(cb, "enum", name)

        # No list/table editors in this template
        if get_origin(t) is list:
            le = QLineEdit("" if default is None else json.dumps(default))
            if hint:
                le.setToolTip(hint)
            return FieldWidget(le, "str", name)

        if t is bool:
            cb = QCheckBox()
            cb.setChecked(bool(default))
            if hint:
                cb.setToolTip(hint)
            return FieldWidget(cb, "bool", name)

        if t is int:
            sb = QSpinBox()
            sb.setRange(-10**9, 10**9)
            sb.setValue(int(default) if default is not None else 0)
            sb.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            if hint:
                sb.setToolTip(hint)
            return FieldWidget(sb, "int", name)

        if t is float:
            dsb = TrimmedDoubleSpinBox()
            dsb.setDecimals(6)
            dsb.setRange(-1e18, 1e18)
            dsb.setValue(float(default) if default is not None else 0.0)
            dsb.setSingleStep(0.1)
            dsb.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
            dsb.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            if hint:
                dsb.setToolTip(hint)
            return FieldWidget(dsb, "float", name)

        if isinstance(t, type) and issubclass(t, Enum):
            cb = QComboBox()
            if hint:
                cb.setToolTip(hint)
            if is_optional:
                cb.addItem("", None)
            for member in t:
                label = str(getattr(member, "value", member))
                cb.addItem(label, label)
            default_value = getattr(default, "value", default)
            if default_value is not None:
                idx = cb.findData(str(default_value))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return FieldWidget(cb, "enum", name)

        if get_origin(t) is Literal:
            cb = QComboBox()
            if hint:
                cb.setToolTip(hint)
            if is_optional:
                cb.addItem("", None)
            for choice in get_args(t):
                label = str(choice)
                cb.addItem(label, label)
            if default is not None:
                idx = cb.findData(str(default))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return FieldWidget(cb, "enum", name)

        if t is Path or (t is str and any(k in name.lower() for k in ("path", "file", "template"))):
            le = QLineEdit("" if default is None else str(default))
            le.setPlaceholderText(hint or "Select a file…")
            btn = QPushButton("Browse…")
            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            lay.addWidget(le, 1)
            lay.addWidget(btn)

            def browse() -> None:
                p, _ = QFileDialog.getOpenFileName(self, "Select file")
                if p:
                    le.setText(p)

            btn.clicked.connect(browse)
            return FieldWidget(le, "path", name)

        le = QLineEdit("" if default is None else str(default))
        if hint:
            le.setToolTip(hint)
        return FieldWidget(le, "str", name)

    def _apply_section_visibility(self) -> None:
        if not self._shape_family_field or not self._section_fields:
            return
        family_value = self._shape_family_field.get_value()
        family_label = str(getattr(family_value, "value", family_value)) if family_value is not None else ""
        for fw, tag in self._section_fields:
            show = family_label == tag
            label = self._field_label_by_widget.get(fw.widget)
            if label is not None:
                label.setVisible(show)
            fw.widget.setVisible(show)
            if not show:
                fw.clear()

    def _setup_material_presets(self) -> None:
        if not self._material_preset_bindings:
            return
        for binding in self._material_preset_bindings:
            src = self.field_widgets.get(binding["source"])
            if src is None or not isinstance(src.widget, QComboBox):
                continue
            src.widget.currentIndexChanged.connect(
                lambda _idx, b=binding: self._apply_material_preset(b)
            )
            self._apply_material_preset(binding)

    def _apply_material_preset(self, binding: dict[str, Any]) -> None:
        src = self.field_widgets.get(binding["source"])
        if src is None:
            return
        grade = src.get_value()
        grade_label = str(grade) if grade is not None else ""
        presets = binding.get("presets", {})
        custom_value = binding.get("custom_value", "Custom")
        targets = binding.get("targets", [])
        override_field = binding.get("override_field")

        allow_edit = grade_label == custom_value
        preset = presets.get(grade_label, {})

        for target_name in targets:
            fw = self.field_widgets.get(target_name)
            if fw is None:
                continue
            fw.set_enabled(allow_edit)
            if not allow_edit:
                if target_name in preset:
                    fw.set_value(preset[target_name])

        if override_field:
            ofw = self.field_widgets.get(override_field)
            if ofw is not None and ofw.kind == "bool":
                ofw.set_enabled(allow_edit)
                ofw.set_value(allow_edit)

    def _setup_c49_girder_defaults(self) -> None:
        if self.active_tool_id != "c49_overhang_bracket":
            return
        src = self.field_widgets.get("girder_type")
        if src is None or not isinstance(src.widget, QComboBox):
            return

        def apply_defaults() -> None:
            from toolbox_app.tools.c49_overhang_bracket.db.txdot_girders import get_txdot_profile

            girder_type = src.get_value()
            if not girder_type:
                return
            profile = get_txdot_profile(str(girder_type))
            defaults = {
                "girder_depth_in": profile.depth_in,
                "girder_top_flange_width_in": profile.top_flange_width_in,
                "girder_bottom_flange_width_in": profile.bottom_flange_width_in,
                "girder_web_thickness_in": profile.web_thickness_in,
                "girder_top_flange_thickness_in": 3.5,
                "girder_bottom_flange_thickness_in": profile.F_in,
            }
            for key, value in defaults.items():
                fw = self.field_widgets.get(key)
                if fw is None:
                    continue
                fw.set_value(value)

        src.widget.currentIndexChanged.connect(lambda _idx: apply_defaults())
        apply_defaults()

    def cancel_run(self) -> None:
        self.runner.cancel()
        self.cancel_btn.setEnabled(False)
        self._update_status("Cancel requested.")

    def _collect_inputs(self) -> Dict[str, Any]:
        return {k: fw.get_value() for k, fw in self.field_widgets.items()}

    def _toggle_output(self, checked: bool) -> None:
        self.output.setVisible(checked)
        self.output_toggle.setText("Hide details" if checked else "Details")

    def _update_status(self, message: str) -> None:
        text = str(message)
        self.status_label.setText(text)
        self.output.append(text)

    def _set_running_ui(self) -> None:
        self.output.clear()
        self._update_status("Running...")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)

    def _finish_ui(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)

    def run_active_tool(self) -> None:
        if not self.active_tool_id:
            return
        tool = self.tool_by_id[self.active_tool_id]

        raw = self._collect_inputs()
        model: Optional[Type[BaseModel]] = getattr(tool, "InputModel", None)
        validated, err = validate_inputs(model, raw)
        if err:
            QMessageBox.warning(self, "Input validation error", err)
            return

        # UI-thread tools (HTML/WebEngine wrappers, future UI launchers)
        if getattr(tool, "RUNS_ON_UI_THREAD", False):
            try:
                self._set_running_ui()
                out = tool.run(validated)

                class _R:
                    ok = True
                    output = out
                    error = None

                self._on_finished(_R())
            except Exception as e:
                self._finish_ui()
                self.status_label.setText("Failed.")
                QMessageBox.critical(self, "Tool error", str(e))
            return

        # Background runner for compute / I/O tools
        self._set_running_ui()
        runnable = self.runner.start(tool, validated)
        runnable.signals.progress.connect(self.progress.setValue)
        runnable.signals.status.connect(self._update_status)
        runnable.signals.finished.connect(self._on_finished)

    @Slot(object)
    def _on_finished(self, result) -> None:
        self._finish_ui()
        if not result.ok:
            self.status_label.setText("Failed.")
            QMessageBox.critical(self, "Tool error", result.error or "Unknown error")
            return
        self.status_label.setText("Done.")
        output = result.output
        self._last_output = output
        self._last_tool_id = self.active_tool_id
        self._set_action_buttons_enabled(True)
        if isinstance(output, dict) and output.get("report_html"):
            self._show_report_view(str(output["report_html"]))
        else:
            self.output.append(json.dumps(output, indent=2))
            self.output.append("\nDone.")


def main() -> None:
    configure_logging()
    app = QApplication([])
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)
    app.setStyleSheet(APP_STYLESHEET)
    w = MainWindow()
    w.show()
    w.raise_()
    w.activateWindow()
    app.exec()


if __name__ == "__main__":
    main()
