"""
This module provides functions to load and process results from the mAIcrobe pipeline. It will combine multiple CSV files into a single DataFrame, 
and ensure the data is clean and well-structured for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import warnings

import pandas as pd

Metadata = Mapping[str, Any]
FilenameParser = Callable[[Path | str], Metadata]

mAIcrobe_numeric_columns = [
    "frame",
    "label",
    "Area",
    "Perimeter",
    "Eccentricity",
    "Width",
    "Length",
    "Baseline",
    "Cell Median",
    "Membrane Median",
    "Septum Median",
    "Cytoplasm Median",
    "Fluor Ratio",
    "Fluor Ratio 75%",
    "Fluor Ratio 25%",
    "Fluor Ratio 10%",
    "Cell Cycle Phase",
    "DNA Ratio",
]

mAIcrobe_recommended_columns = [
    "label",
    "Area",
    "Perimeter",
    "Eccentricity",
    "Width",
    "Length",
    "Shape Analysis Status",
]

@dataclass
class CSVLoadReport:
    """
    A dataclass to hold the report of loading CSV files.
    """
    n_files: int
    n_rows: int
    n_computed: int
    n_failed: int
    columns: list[str]
    numeric_columns: list[str]
    warnings: list[str] = field(default_factory=list)

@dataclass
class LoadResult:
    """
    Combined mAIcrobe reults table and loading report.
    """
    data: pd.DataFrame
    report: CSVLoadReport

def clean_maicrobe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the mAIcrobe DataFrame by ensuring numeric columns are of the correct type and handling missing values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame.
    """
    df = df.copy()
    # Ensure numeric columns are of the correct type

    df.columns = [str(col).strip() for col in df.columns]  # Strip whitespace from column names

    columns_to_drop = []
    for col in df.columns:
       lowered = col.lower()

       if col == "":
           columns_to_drop.append(col)

       elif lowered.startswith("unnamed:"):
           columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    return df

def coerce_maicrobe_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Convert known mAIcrobe measurement columns to numeric values.
    If numeric_columns is None, prefer the module default; if that's not
    available, infer numeric-like columns from the dataframe.
    """
    df = df.copy()
    warnings: list[str] = []
    converted_columns: list[str] = []

    # Decide which columns to convert: explicit -> module default -> inference
    if numeric_columns is None:
        try:
            # prefer the explicit module-level default list
            if mAIcrobe_numeric_columns:
                numeric_columns = mAIcrobe_numeric_columns
            else:
                # force fallback to inference if list empty
                raise NameError("module default empty")
        except NameError:
            # Infer numeric-like columns:
            inferred: list[str] = []
            for col in df.columns:
                # fast path: pandas numeric dtype
                if pd.api.types.is_numeric_dtype(df[col]):
                    inferred.append(col)
                    continue
                # otherwise, check whether any values look like numbers
                if df[col].dtype == object:
                    s = df[col].astype(str)
                    if s.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False).any():
                        inferred.append(col)
            numeric_columns = inferred
            warnings.append(
                "No explicit numeric column list found; inferred numeric-like columns: "
                + f"{numeric_columns}"
            )

    # Convert listed columns to numeric, warn on newly coerced-to-NaN values
    for col in numeric_columns:
        if col not in df.columns:
            continue

        before_missing = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_missing = df[col].isna().sum()

        converted_columns.append(col)

        newly_missing = after_missing - before_missing
        if newly_missing > 0:
            warnings.append(
                f"Column {col!r} had {newly_missing} values that could not be converted to numbers."
            )

    return df, converted_columns, warnings

def validate_maicrobe_table(df: pd.DataFrame, file_name: str) -> list[str]:
    """
    Check whether a mAIcrobe CSV has the expected basic structure.
    """

    warnings: list[str] = []

    missing_columns = [
        col for col in mAIcrobe_recommended_columns if col not in df.columns
    ]

    if missing_columns:
        warnings.append(
            f"{file_name} is missing recommended mAIcrobe columns: "
            f"{missing_columns}"
        )

    if "Shape Analysis Status" in df.columns:
        # normalize values before comparison
        statuses = df["Shape Analysis Status"].astype(str).str.strip().str.lower()
        n_failed = (statuses != "computed").sum()

        if n_failed > 0:
            warnings.append(
                f"{file_name} contains {n_failed} rows where "
                "Shape Analysis Status is not 'computed'."
            )

    if "Width" in df.columns:
        n_zero_width = (df["Width"] == 0).sum()

        if n_zero_width > 0:
            warnings.append(
                f"{file_name} contains {n_zero_width} rows with Width = 0."
            )

    if "Length" in df.columns:
        n_zero_length = (df["Length"] == 0).sum()

        if n_zero_length > 0:
            warnings.append(
                f"{file_name} contains {n_zero_length} rows with Length = 0."
            )

    if "DNA Ratio" in df.columns and df["DNA Ratio"].isna().all():
        warnings.append(
            f"{file_name} has a DNA Ratio column, but it is completely empty."
        )

    return warnings

def read_single_maicrobe_csv(
    path: str | Path,
    *,
    metadata: Metadata | None = None,
    numeric_columns: list[str] | None = None,   
    filename_parser: FilenameParser | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Read one mAIcrobe CSV file and add provenance metadata.

    Parameters
    ----------
    path:
        Path to one mAIcrobe CSV file.

    metadata:
        Optional metadata to add to every row.

        Example
        -------
        {
            "condition": "WT",
            "replicate": 1,
            "image": "img01"
        }

    filename_parser:
        Optional function that extracts metadata from the file name.

    Returns
    -------
    df:
        Cleaned result table.

    numeric_columns:
        Numeric columns found in this file.

    warnings:
        Warnings generated while reading this file.
    """

    path = Path(path)
    warnings: list[str] = []

    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    df = pd.read_csv(path)
    df = clean_maicrobe_columns(df)

    # Normalize status text so later filtering/counts are consistent
    if "Shape Analysis Status" in df.columns:
       df["Shape Analysis Status"] = (
            df["Shape Analysis Status"].astype(str).str.strip().str.lower()
       )

    # Convert measurement columns before validation (use caller-provided numeric_columns).
    df, numeric_columns, numeric_warnings = coerce_maicrobe_numeric_columns(df, numeric_columns=numeric_columns)
    warnings.extend(numeric_warnings)

    # Add source/provenance information.
    df["source_file"] = path.name
    df["source_stem"] = path.stem
    df["source_path"] = str(path)

    # Add metadata parsed from file name.
    if filename_parser is not None:
        parsed_metadata = filename_parser(path)
        for key, value in parsed_metadata.items():
            df[key] = value

    # Add manually supplied metadata.
    # This happens after filename parsing so manual metadata can override it.
    if metadata is not None:
        for key, value in metadata.items():
            df[key] = value

    # If image was not provided, use the file stem as the image ID.
    if "image" not in df.columns:
        df["image"] = path.stem

    # Make an object ID that is unique across multiple CSV files.
    if "frame" in df.columns and "label" in df.columns:
        df["global_object_id"] = (
            df["source_stem"].astype(str)
            + "_frame"
            + df["frame"].astype(str)
            + "_label"
            + df["label"].astype(str)
        )
    elif "label" in df.columns:
        df["global_object_id"] = (
            df["source_stem"].astype(str)
            + "_label"
            + df["label"].astype(str)
        )
    else:
        df["global_object_id"] = (
            df["source_stem"].astype(str)
            + "_row"
            + df.index.astype(str)
        )

    warnings.extend(validate_maicrobe_table(df, path.name))

    return df, numeric_columns, warnings

def load_maicrobe_csvs(
    csv_paths: Iterable[str | Path],
    *,
    metadata_by_file: Mapping[str, Metadata] | None = None,
    filename_parser: FilenameParser | None = None,
    numeric_columns: list[str] | None = None,
    keep_failed: bool = True,
) -> LoadResult:
    """
    Load multiple mAIcrobe result CSV files into one combined table.

    Parameters
    ----------
    csv_paths:
        List of mAIcrobe CSV result files.

    metadata_by_file:
        Optional dictionary mapping file name or file stem to metadata.

        Example
        -------
        {
            "WT_rep1_img01_results": {
                "condition": "WT",
                "replicate": 1,
                "image": "img01"
            }
        }

    filename_parser:
        Optional function that extracts condition, replicate, or image metadata
        from file names.

    keep_failed:
        Whether to keep rows where Shape Analysis Status is not "computed".

        For plotting raw output, True is useful.
        For morphology statistics, False may be safer.

    Returns
    -------
    LoadedResults
        Combined DataFrame and loading report.
    """

    paths = [Path(p) for p in csv_paths]

    if not paths:
        raise ValueError("No CSV files were provided.")

    combined_tables: list[pd.DataFrame] = []
    all_warnings: list[str] = []
    all_numeric_columns: set[str] = set()

    for path in paths:
        metadata = None
        if metadata_by_file is not None:
            metadata = (
                metadata_by_file.get(str(path))
                or metadata_by_file.get(path.name)
                or metadata_by_file.get(path.stem)
            )

        try:
            df, numeric_columns, warnings = read_single_maicrobe_csv(
                path,
                metadata=metadata,
                numeric_columns=numeric_columns,
                filename_parser=filename_parser,
            )
        except Exception as exc:
            all_warnings.append(f"Failed to read {path}: {exc}")
            # Optionally continue to next file instead of aborting
            continue

        all_numeric_columns.update(numeric_columns)
        all_warnings.extend(warnings)
        combined_tables.append(df)

    # If no files were successfully read, avoid pd.concat on an empty list.
    if not combined_tables:
        all_warnings.append("No CSV files were successfully read.")
        combined = pd.DataFrame()
    else:
        combined = pd.concat(combined_tables, ignore_index=True)
        if not keep_failed and "Shape Analysis Status" in combined.columns:
            combined = combined[combined["Shape Analysis Status"] == "computed"].copy()

    # Warn if key experimental metadata is missing.
    for col in ["condition", "replicate", "image"]:
        if col not in combined.columns:
            all_warnings.append(
                f"Metadata column {col!r} is missing. "
                "This is important for grouped plots and replicate-level statistics."
            )

    if "Shape Analysis Status" in combined.columns:
        n_computed = int((combined["Shape Analysis Status"] == "computed").sum())
        n_failed = int((combined["Shape Analysis Status"] != "computed").sum())
    else:
        n_computed = 0
        n_failed = 0

    report = CSVLoadReport(
        n_files=len(paths),
        n_rows=len(combined),
        n_computed=n_computed,
        n_failed=n_failed,
        columns=list(combined.columns),
        numeric_columns=sorted(all_numeric_columns),
        warnings=all_warnings,
    )

    return LoadResult(data=combined, report=report)