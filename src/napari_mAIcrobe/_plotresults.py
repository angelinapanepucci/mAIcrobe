"""
Module responsible for the napari GUI used to load mAIcrobe CSV result files
and generate morphology plots.

The widget collects user input, but delegates experiment design, CSV loading,
and plotting to the backend results_analysis modules.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figurenapari 

from .results_analysis.group_conditions import ExperimentDesign
from .results_analysis.plotting import MorphologyPlotConfig, plot_from_config

if TYPE_CHECKING:
    import napari


MORPHOLOGY_COLUMNS = [
    "Area",
    "Perimeter",
    "Eccentricity",
    "Width",
    "Length",
]


class ResultsAnalysisWidget(QWidget):
    """
    Napari widget for loading grouped mAIcrobe CSV result files and plotting
    measurements by condition.
    """

    def __init__(
        self,
        Viewer: "napari.Viewer" | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.viewer = Viewer
        self.experiment = ExperimentDesign()

        self.selected_csv_paths: list[Path] = []
        self.combined_df: pd.DataFrame | None = None
        self.load_result = None

        self._build_widget()
        self._connect_events()
        self._set_plot_controls_enabled(False)

    def _build_widget(self) -> None:
        """
        Build the full widget layout.
        """

        self.setLayout(QVBoxLayout())

        self.layout().addWidget(self._make_group_box())
        self.layout().addWidget(self._make_current_groups_box())
        self.layout().addWidget(self._make_load_box())
        self.layout().addWidget(self._make_plot_box())
        self.layout().addWidget(self._make_status_box())

    def _make_group_box(self) -> QGroupBox:
        """
        Create controls for adding one condition group.
        """

        group_box = QGroupBox("Add CSV files to a condition")
        layout = QVBoxLayout()
        form = QFormLayout()

        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText("WT")

        self.replicate_edit = QLineEdit()
        self.replicate_edit.setPlaceholderText("1")

        self.group_name_edit = QLineEdit()
        self.group_name_edit.setPlaceholderText("Optional name, e.g. WT rep 1")

        form.addRow("Condition:", self.condition_edit)
        form.addRow("Replicate:", self.replicate_edit)
        form.addRow("Group name:", self.group_name_edit)

        self.select_files_button = QPushButton("Select CSV files")
        self.selected_files_label = QLabel("No CSV files selected.")
        self.selected_files_label.setWordWrap(True)

        self.add_group_button = QPushButton("Add condition group")

        layout.addLayout(form)
        layout.addWidget(self.select_files_button)
        layout.addWidget(self.selected_files_label)
        layout.addWidget(self.add_group_button)

        group_box.setLayout(layout)

        return group_box

    def _make_current_groups_box(self) -> QGroupBox:
        """
        Create the table showing groups already added by the user.
        """

        group_box = QGroupBox("Current experiment groups")
        layout = QVBoxLayout()

        self.groups_table = QTableWidget(0, 5)
        self.groups_table.setHorizontalHeaderLabels(
            [
                "Condition",
                "Replicate",
                "Group name",
                "n files",
                "Files",
            ]
        )
        self.groups_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.groups_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.groups_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        button_layout = QHBoxLayout()

        self.remove_group_button = QPushButton("Remove selected group")
        self.clear_groups_button = QPushButton("Clear groups")

        button_layout.addWidget(self.remove_group_button)
        button_layout.addWidget(self.clear_groups_button)

        layout.addWidget(self.groups_table)
        layout.addLayout(button_layout)

        group_box.setLayout(layout)

        return group_box

    def _make_load_box(self) -> QGroupBox:
        """
        Create controls for loading all added groups.
        """

        group_box = QGroupBox("Load grouped CSV data")
        layout = QVBoxLayout()

        self.keep_failed_check = QCheckBox("Keep failed shape-analysis rows")
        self.keep_failed_check.setChecked(False)

        self.load_button = QPushButton("Load all groups")

        layout.addWidget(self.keep_failed_check)
        layout.addWidget(self.load_button)

        group_box.setLayout(layout)

        return group_box

    def _make_plot_box(self) -> QGroupBox:
        """
        Create controls for graph generation.
        """

        group_box = QGroupBox("Generate graph")
        layout = QVBoxLayout()
        form = QFormLayout()

        self.measurement_combo = QComboBox()
        self.condition_col_combo = QComboBox()
        self.hue_col_combo = QComboBox()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItem("Violin + box + points", "violin_box")
        self.plot_type_combo.addItem("Box plot", "box")
        self.plot_type_combo.addItem("Violin plot", "violin")
        self.plot_type_combo.addItem("Strip plot", "strip")
        self.plot_type_combo.addItem("Swarm plot", "swarm")

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Optional graph title")

        self.y_label_edit = QLineEdit()
        self.y_label_edit.setPlaceholderText("Optional y-axis label")

        self.show_points_check = QCheckBox("Show individual cell points")
        self.show_points_check.setChecked(True)

        self.show_n_check = QCheckBox("Show n per condition")
        self.show_n_check.setChecked(True)

        form.addRow("Measurement:", self.measurement_combo)
        form.addRow("Condition column:", self.condition_col_combo)
        form.addRow("Hue/grouping column:", self.hue_col_combo)
        form.addRow("Plot type:", self.plot_type_combo)
        form.addRow("Title:", self.title_edit)
        form.addRow("Y-axis label:", self.y_label_edit)

        layout.addLayout(form)
        layout.addWidget(self.show_points_check)
        layout.addWidget(self.show_n_check)

        button_layout = QHBoxLayout()

        self.plot_button = QPushButton("Generate graph")
        self.save_button = QPushButton("Save graph")

        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.canvas)

        group_box.setLayout(layout)

        return group_box

    def _make_status_box(self) -> QGroupBox:
        """
        Create the status/report display.
        """

        group_box = QGroupBox("Status")
        layout = QVBoxLayout()

        self.status_box = QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(180)

        layout.addWidget(self.status_box)
        group_box.setLayout(layout)

        return group_box

    def _connect_events(self) -> None:
        """
        Connect buttons to callback methods.
        """

        self.select_files_button.clicked.connect(self._select_csv_files)
        self.add_group_button.clicked.connect(self._add_condition_group)

        self.remove_group_button.clicked.connect(self._remove_selected_group)
        self.clear_groups_button.clicked.connect(self._clear_groups)

        self.load_button.clicked.connect(self._load_all_groups)

        self.plot_button.clicked.connect(self._generate_plot)
        self.save_button.clicked.connect(self._save_graph)

    def _select_csv_files(self) -> None:
        """
        Open a file dialog for selecting one or more CSV result files.
        """

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select mAIcrobe CSV result files",
            "",
            "CSV files (*.csv);;All files (*)",
        )

        if not file_paths:
            return

        self.selected_csv_paths = [Path(path) for path in file_paths]

        file_names = [path.name for path in self.selected_csv_paths]
        self.selected_files_label.setText(", ".join(file_names))

    def _add_condition_group(self) -> None:
        """
        Add the currently selected CSV files as one condition group.
        """

        condition = self.condition_edit.text().strip()
        replicate = self.replicate_edit.text().strip() or None
        group_name = self.group_name_edit.text().strip() or None

        if not condition:
            self._set_status("Please enter a condition name before adding files.")
            return

        if not self.selected_csv_paths:
            self._set_status("Please select at least one CSV file.")
            return

        try:
            self.experiment.add_group(
                condition=condition,
                replicate=replicate,
                group_name=group_name,
                csv_paths=self.selected_csv_paths,
            )
        except Exception as exc:
            self._set_status(f"Could not add condition group:\n{exc}")
            return

        self._update_groups_table()

        self.selected_csv_paths = []
        self.selected_files_label.setText("No CSV files selected.")
        self.group_name_edit.clear()

        self._set_status(f"Added condition group: {condition}")

    def _remove_selected_group(self) -> None:
        """
        Remove the selected condition group from the experiment.
        """

        row = self.groups_table.currentRow()

        if row < 0:
            self._set_status("Please select a group to remove.")
            return

        try:
            removed = self.experiment.remove_group(row)
        except Exception as exc:
            self._set_status(f"Could not remove group:\n{exc}")
            return

        self._update_groups_table()
        self._set_status(f"Removed group: {removed.label}")

    def _clear_groups(self) -> None:
        """
        Clear all condition groups.
        """

        self.experiment.clear()
        self._update_groups_table()
        self._set_plot_controls_enabled(False)

        self.combined_df = None
        self.load_result = None

        self._set_status("Cleared all condition groups.")

    def _update_groups_table(self) -> None:
        """
        Refresh the experiment summary table.
        """

        self.groups_table.setRowCount(len(self.experiment.groups))

        for row, group in enumerate(self.experiment.groups):
            files = ", ".join(path.name for path in group.csv_paths)

            values = [
                group.condition,
                "" if group.replicate is None else str(group.replicate),
                "" if group.group_name is None else str(group.group_name),
                str(group.n_files),
                files,
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.groups_table.setItem(row, column, item)

    def _load_all_groups(self) -> None:
        """
        Load all added condition groups into one combined DataFrame.
        """

        if self.experiment.n_groups == 0:
            self._set_status("No condition groups have been added.")
            return

        try:
            result = self.experiment.load(
                keep_failed=self.keep_failed_check.isChecked()
            )
        except Exception as exc:
            self._set_status(f"Could not load CSV files:\n{exc}")
            return

        self.load_result = result
        self.combined_df = result.data

        self._populate_plot_controls()
        self._set_plot_controls_enabled(not self.combined_df.empty)
        self._show_load_report()

    def _populate_plot_controls(self) -> None:
        """
        Populate plot dropdown menus using columns from the loaded DataFrame.
        """

        if self.combined_df is None or self.combined_df.empty:
            return

        df = self.combined_df

        numeric_columns = [
            column for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]

        preferred_columns = [
            column for column in MORPHOLOGY_COLUMNS
            if column in numeric_columns
        ]

        remaining_columns = [
            column for column in numeric_columns
            if column not in preferred_columns
        ]

        measurement_columns = preferred_columns + remaining_columns

        self.measurement_combo.clear()

        for column in measurement_columns:
            self.measurement_combo.addItem(column, column)

        if "Width" in measurement_columns:
            index = self.measurement_combo.findData("Width")
            self.measurement_combo.setCurrentIndex(index)

        self.condition_col_combo.clear()

        for column in df.columns:
            self.condition_col_combo.addItem(column, column)

        if "condition" in df.columns:
            index = self.condition_col_combo.findData("condition")
            self.condition_col_combo.setCurrentIndex(index)

        self.hue_col_combo.clear()
        self.hue_col_combo.addItem("None", None)

        useful_hue_columns = [
            "replicate",
            "image",
            "condition_group",
            "source_file",
        ]

        for column in useful_hue_columns:
            if column in df.columns:
                self.hue_col_combo.addItem(column, column)

        for column in df.columns:
            if column not in useful_hue_columns and column != "condition":
                if df[column].nunique(dropna=True) <= 20:
                    self.hue_col_combo.addItem(column, column)

    def _generate_plot(self) -> None:
        """
        Generate the selected graph from the loaded DataFrame.
        """

        if self.combined_df is None or self.combined_df.empty:
            self._set_status("No data has been loaded yet.")
            return

        measurement_col = self.measurement_combo.currentData()
        condition_col = self.condition_col_combo.currentData()
        hue_col = self.hue_col_combo.currentData()
        plot_type = self.plot_type_combo.currentData()

        if measurement_col is None:
            self._set_status("Please select a measurement column.")
            return

        if condition_col is None:
            self._set_status("Please select a condition column.")
            return

        title = self.title_edit.text().strip() or None
        y_label = self.y_label_edit.text().strip() or None

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Build the plot configuration dataclass from the GUI values.
        ui_values = {
            "measurement_col": measurement_col,
            "condition_col": condition_col,
            "hue_col": hue_col,
            "plot_type": plot_type,
            "title": title,
            "y_label": y_label,
            "ylabel": y_label,
            "show_points": self.show_points_check.isChecked(),
            "show_n": self.show_n_check.isChecked(),
            "max_points_for_scatter": 5000,
            "sample_n": 5000,
            "random_state": 0,
        }

        mapped_values = self._map_ui__kwargs_to_plot_config(ui_values)
        config = self._make_plot_config(mapped_values)

        try:
            self._call_plot_from_config(
                df=self.combined_df,
                config=config,
                ax=ax,
            )
        except Exception as exc:
            self.figure.clear()
            self.canvas.draw_idle()
            self._set_status(f"Could not generate plot:\n{exc}")
            return

        self.figure.tight_layout()
        self.canvas.draw_idle()

        self._set_status("Generated graph.")

    def _make_plot_config(self, values: dict[str, Any]) -> MorphologyPlotConfig:
        """
        Build a MorphologyPlotConfig while ignoring GUI values that are not
        fields in the current config dataclass.

        This makes the widget easier to maintain while the plotting config is
        still changing during development.
        """

        if is_dataclass(MorphologyPlotConfig):
            valid_fields = {field.name for field in fields(MorphologyPlotConfig)}
            values = {
                key: value for key, value in values.items()
                if key in valid_fields
            }

        return MorphologyPlotConfig(**values)
    
    def _map_ui__kwargs_to_plot_config(self, ui_kwargs: dict[str, Any]) ->dict[str, Any]:
        """
        Map the UI kwargs to the plot config kwargs.

        This is a temporary function to help with the transition to the new
        plotting API. It can be removed once the plotting API is stable.
        """

        mapping = {
            "measurement_col": "measurement",
            "condition_col": "condition_col",
            "hue_col": "hue_col",
            "plot_type": "plot_type",
            "title": "title",
            "y_label": "ylabel",
            "show_points": "show_points",
            "show_n": "show_n",
            "max_points_for_scatter": "max_points_for_scatter",
            "sample_n": "sample_n",
            #add more as needed
        }

        mapped: dict[str, Any] = {}
        for k, v in ui_kwargs.items():
            if k in mapping:
                mapped[mapping[k]] = v
            else:
                mapped[k] = v

        return mapped

    def _call_plot_from_config(
        self,
        df: pd.DataFrame,
        config: MorphologyPlotConfig,
        ax,
    ):
        """
        Call the plotting backend.

        The preferred plotting.py API is:

            plot_from_config(df, config, ax=ax)

        The fallback is kept only to make development easier while the plotting
        backend is still being adjusted.
        """

        try:
            # Ensure we have an Axes object to pass to the preferred API.
            if ax is None:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=config.figsize)

            return plot_from_config(df, config, ax=ax)
        except TypeError:
            # Fallback to the older API that doesn't accept an ax argument.
            return plot_from_config(df, config)

    def _save_graph(self) -> None:
        """
        Save the current graph to disk.
        """

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save graph",
            "maicrobe_graph.png",
            "PNG image (*.png);;PDF file (*.pdf);;SVG file (*.svg);;All files (*)",
        )

        if not file_path:
            return

        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            self._set_status(f"Could not save graph:\n{exc}")
            return

        self._set_status(f"Saved graph to:\n{file_path}")

    def _show_load_report(self) -> None:
        """
        Show a readable report after loading data.
        """

        if self.load_result is None:
            return

        report = self.load_result.report
        df = self.load_result.data

        lines = [
            "Loaded mAIcrobe CSV results.",
            "",
            f"Number of files: {report.n_files}",
            f"Number of rows: {report.n_rows}",
            f"Number computed: {report.n_computed}",
            f"Number failed: {report.n_failed}",
            "",
            "Columns:",
            ", ".join(report.columns),
        ]

        if "condition" in df.columns:
            lines.extend(
                [
                    "",
                    "Condition counts:",
                    str(df["condition"].value_counts()),
                ]
            )

        if report.warnings:
            lines.extend(
                [
                    "",
                    "Warnings:",
                    *[f"- {warning}" for warning in report.warnings],
                ]
            )

        self._set_status("\n".join(lines))

    def _set_plot_controls_enabled(self, enabled: bool) -> None:
        """
        Enable or disable graph controls.
        """

        widgets = [
            self.measurement_combo,
            self.condition_col_combo,
            self.hue_col_combo,
            self.plot_type_combo,
            self.title_edit,
            self.y_label_edit,
            self.show_points_check,
            self.show_n_check,
            self.plot_button,
            self.save_button,
        ]

        for widget in widgets:
            widget.setEnabled(enabled)

    def _set_status(self, message: str) -> None:
        """
        Display a status message in the widget.
        """

        self.status_box.setPlainText(message)


def make_results_analysis_widget(
    Viewer: "napari.Viewer" | None = None,
) -> ResultsAnalysisWidget:
    """
    Create the mAIcrobe results-analysis widget for napari.
    """

    return ResultsAnalysisWidget(Viewer)