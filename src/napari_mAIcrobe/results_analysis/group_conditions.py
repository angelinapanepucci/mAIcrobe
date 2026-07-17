"""
Module responsible for storing experimental CSV groups before loading
mAIcrobe result tables.

This module does not read or clean CSV files directly. Instead, it stores
the user's experimental design, builds file-level metadata, and delegates
CSV loading to the existing mAIcrobe results loader.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .load_results import LoadResult, load_maicrobe_csvs


Metadata = dict[str, Any]


@dataclass
class ConditionGroup:
    """
    A group of CSV files belonging to one experimental condition.

    A ConditionGroup represents one user-defined batch, for example:
    - condition = "WT"
    - replicate = 1
    - csv_paths = five mAIcrobe result CSV files

    The napari widget should create one ConditionGroup every time the user
    clicks an "Add condition group" button.
    """

    condition: str
    csv_paths: list[Path]
    replicate: int | str | None = None
    group_name: str | None = None
    extra_metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        self.condition = str(self.condition).strip()
        self.csv_paths = [Path(path).expanduser() for path in self.csv_paths]

        if self.replicate is not None:
            self.replicate = str(self.replicate).strip()

        if self.group_name is not None:
            self.group_name = str(self.group_name).strip()

        if self.extra_metadata is None:
            self.extra_metadata = {}

        if not self.condition:
            raise ValueError("Condition name cannot be empty.")

        if not self.csv_paths:
            raise ValueError(
                f"No CSV files were provided for condition {self.condition!r}."
            )

    @property
    def n_files(self) -> int:
        """
        Return the number of CSV files in this condition group.
        """

        return len(self.csv_paths)

    @property
    def label(self) -> str:

        if self.group_name:
            return self.group_name

        if self.replicate is not None:
            return f"{self.condition}, replicate {self.replicate}"

        return self.condition

    def metadata_for_file(self, path: str | Path) -> Metadata:
        """
        Build metadata for one CSV file in this condition group.

        Parameters
        ----------
        path:
            Path to one CSV file belonging to this group.

        Returns
        -------
        metadata:
            Dictionary that can be passed to load_maicrobe_csvs through
            metadata_by_file.
        """

        path = Path(path)

        metadata: Metadata = {
            "condition": self.condition,
            "image": path.stem,
            "condition_group": self.label,
        }

        if self.replicate is not None:
            metadata["replicate"] = self.replicate

        metadata.update(dict(self.extra_metadata))

        return metadata

    def missing_files(self) -> list[Path]:
        """
        Return files in this group that do not currently exist.
        """

        return [path for path in self.csv_paths if not path.exists()]

    def non_csv_files(self) -> list[Path]:
        """
        Return files in this group that do not have a .csv extension.
        """

        return [
            path for path in self.csv_paths
            if path.suffix.lower() != ".csv"
        ]


@dataclass
class ExperimentDesign:
    """
    Container for all condition groups in one mAIcrobe graphing experiment.

    The napari GUI should keep one ExperimentDesign object alive while the
    user adds condition groups. When the user clicks "Load all", the widget
    should call experiment.load().
    """

    groups: list[ConditionGroup] = field(default_factory=list)

    def add_group(
        self,
        condition: str,
        csv_paths: Iterable[str | Path],
        replicate: int | str | None = None,
        group_name: str | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> ConditionGroup:
        """
        Add one condition group to the experiment.

        Parameters
        ----------
        condition:
            Experimental condition name, for example "WT" or "condition_1".

        csv_paths:
            CSV files belonging to this condition group.

        replicate:
            Optional biological replicate label.

        group_name:
            Optional user-facing name for this group.

        extra_metadata:
            Optional additional metadata to attach to every row from this
            group. For example: strain, treatment, timepoint, or microscope.

        Returns
        -------
        group:
            The ConditionGroup that was added.
        """

        group = ConditionGroup(
            condition=condition,
            replicate=replicate,
            csv_paths=list(csv_paths),
            group_name=group_name,
            extra_metadata=extra_metadata,
        )

        self.groups.append(group)

        return group

    def remove_group(self, index: int) -> ConditionGroup:
        """
        Remove one condition group by index.

        This is useful for a GUI "Remove selected group" button.
        """

        return self.groups.pop(index)

    def clear(self) -> None:
        """
        Remove all condition groups from the experiment.
        """

        self.groups.clear()

    @property
    def n_groups(self) -> int:
        """
        Return the number of condition groups.
        """

        return len(self.groups)

    @property
    def n_files(self) -> int:
        """
        Return the total number of CSV files across all groups.
        """

        return sum(group.n_files for group in self.groups)

    def all_csv_paths(self) -> list[Path]:
        """
        Return all CSV paths across all condition groups.
        """

        csv_paths: list[Path] = []

        for group in self.groups:
            csv_paths.extend(group.csv_paths)

        return csv_paths

    def build_metadata_by_file(self) -> dict[str, Metadata]:
        """
        Build the metadata dictionary expected by load_maicrobe_csvs.

        The full path string is used as the safest key. Filename and stem keys
        are also added only when they are unique, so the dictionary remains
        compatible with loaders that match metadata by path.name or path.stem.
        """

        metadata_by_file: dict[str, Metadata] = {}
        paths = self.all_csv_paths()

        name_counts = Counter(path.name for path in paths)
        stem_counts = Counter(path.stem for path in paths)

        for group in self.groups:
            for path in group.csv_paths:
                metadata = group.metadata_for_file(path)

                # Safest key. This avoids collisions when two folders contain
                # files with the same name.
                metadata_by_file[str(path)] = metadata

                # Compatibility keys for the current loader implementation.
                # These are only safe when unique.
                if name_counts[path.name] == 1:
                    metadata_by_file[path.name] = metadata

                if stem_counts[path.stem] == 1:
                    metadata_by_file[path.stem] = metadata

        return metadata_by_file

    def validate(self, check_files: bool = True) -> list[str]:
        """
        Validate the current experiment design.

        Parameters
        ----------
        check_files:
            Whether to check that files exist on disk.

        Returns
        -------
        warnings:
            List of warnings that can be shown in the napari GUI.
        """

        warnings: list[str] = []

        if not self.groups:
            warnings.append("No condition groups have been added.")
            return warnings

        paths = self.all_csv_paths()

        duplicate_paths = [
            path for path, count in Counter(paths).items()
            if count > 1
        ]

        if duplicate_paths:
            warnings.append(
                "Some CSV files were added more than once: "
                + ", ".join(str(path) for path in duplicate_paths)
            )

        duplicate_names = [
            name for name, count in Counter(path.name for path in paths).items()
            if count > 1
        ]

        if duplicate_names:
            warnings.append(
                "Some CSV files have the same filename. This is okay only if "
                "the loader checks metadata_by_file using str(path) before "
                "path.name or path.stem. Duplicate filenames: "
                + ", ".join(duplicate_names)
            )

        for group in self.groups:
            if group.replicate is None:
                warnings.append(
                    f"Group {group.label!r} has no replicate value. "
                    "This is okay for simple condition plots, but replicate-level "
                    "statistics will require a replicate column."
                )

            non_csv_files = group.non_csv_files()
            if non_csv_files:
                warnings.append(
                    f"Group {group.label!r} contains non-CSV files: "
                    + ", ".join(path.name for path in non_csv_files)
                )

            if check_files:
                missing_files = group.missing_files()
                if missing_files:
                    warnings.append(
                        f"Group {group.label!r} contains missing files: "
                        + ", ".join(str(path) for path in missing_files)
                    )

        return warnings

    def summary(self) -> pd.DataFrame:
        """
        Return a summary table of the current experiment design.

        This is useful for displaying the currently added groups in a napari
        widget before the user clicks "Load all".
        """

        rows: list[dict[str, Any]] = []

        for index, group in enumerate(self.groups):
            rows.append(
                {
                    "index": index,
                    "condition": group.condition,
                    "replicate": group.replicate,
                    "group_name": group.group_name,
                    "n_files": group.n_files,
                    "files": ", ".join(path.name for path in group.csv_paths),
                }
            )

        return pd.DataFrame(rows)

    def load(
        self,
        *,
        keep_failed: bool = False,
        check_files: bool = True,
    ) -> LoadResult:
        """
        Load all condition groups into one combined mAIcrobe result table.

        This method delegates CSV reading, cleaning, validation, and
        concatenation to load_maicrobe_csvs.

        Parameters
        ----------
        keep_failed:
            Whether to keep rows where Shape Analysis Status is not computed.

        check_files:
            Whether to check files before calling the loader.

        Returns
        -------
        result:
            LoadResult containing the combined DataFrame and load report.
        """

        warnings = self.validate(check_files=check_files)

        if not self.groups:
            raise ValueError("No condition groups have been added.")

        csv_paths = self.all_csv_paths()
        metadata_by_file = self.build_metadata_by_file()

        result = load_maicrobe_csvs(
            csv_paths=csv_paths,
            metadata_by_file=metadata_by_file,
            keep_failed=keep_failed,
        )

        result.report.warnings = warnings + result.report.warnings

        return result
