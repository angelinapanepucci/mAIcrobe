print("test.py started")
from pathlib import Path
import sys
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[2]))  # Add src/napari_mAIcrobe to sys.path

from .load_results import LoadResult, load_maicrobe_csvs
from .group_conditions import ExperimentDesign
from .plotting import (
    get_morphology_columns,
    plot_measurement_by_condition,
    save_figure,
)


def make_fake_maicrobe_csv(path: Path, width_shift: float = 0.0):
    """
    Create a tiny fake mAIcrobe-style CSV for testing.
    """

    df = pd.DataFrame(
        {
            "frame": [0, 0, 0, 0, 0],
            "label": [1, 2, 3, 4, 5],
            "Area": [1.1, 1.3, 1.2, 1.4, 1.5],
            "Perimeter": [4.2, 4.5, 4.3, 4.6, 4.8],
            "Eccentricity": [0.8, 0.85, 0.82, 0.88, 0.9],
            "Width": [
                0.75 + width_shift,
                0.80 + width_shift,
                0.78 + width_shift,
                0.82 + width_shift,
                0.79 + width_shift,
            ],
            "Length": [2.1, 2.3, 2.2, 2.5, 2.4],
            "Shape Analysis Status": [
                "computed",
                "computed",
                "computed",
                "computed",
                "computed",
            ],
        }
    )

    df.to_csv(path, index=False)


def main():
    test_dir = Path("test_graphing_data")
    test_dir.mkdir(exist_ok=True)

    wt_files = []
    condition1_files = []

    # Make fake WT CSVs
    for i in range(1, 4):
        path = test_dir / f"WT_img{i}.csv"
        make_fake_maicrobe_csv(path, width_shift=0.0)
        wt_files.append(path)

    # Make fake condition_1 CSVs with slightly larger width
    for i in range(1, 4):
        path = test_dir / f"condition1_img{i}.csv"
        make_fake_maicrobe_csv(path, width_shift=0.15)
        condition1_files.append(path)

    experiment = ExperimentDesign()

    experiment.add_group(
        condition="WT",
        replicate=1,
        csv_paths=wt_files,
    )

    experiment.add_group(
        condition="condition_1",
        replicate=1,
        csv_paths=condition1_files,
    )

    print("\nExperiment summary:")
    print(experiment.summary())

    print("\nExperiment warnings:")
    for warning in experiment.validate():
        print("-", warning)

    result = experiment.load(keep_failed=False)

    combined_df = result.data
    report = result.report

    print("\nLoad report:")
    print("Number of files:", report.n_files)
    print("Number of rows:", report.n_rows)
    print("Number computed:", report.n_computed)
    print("Number failed:", report.n_failed)

    print("\nCombined DataFrame columns:")
    print(combined_df.columns.tolist())

    print("\nCondition counts:")
    print(combined_df["condition"].value_counts())

    print("\nFirst few rows:")
    print(
        combined_df[
            [
                "source_file",
                "condition",
                "replicate",
                "image",
                "label",
                "Width",
                "Length",
            ]
        ].head()
    )

    morphology_columns = get_morphology_columns(combined_df)

    print("\nAvailable morphology columns:")
    print(morphology_columns)

    plot_result = plot_measurement_by_condition(
        combined_df,
        measurement="Width",
        condition_col="condition",
        plot_type="violin_box",
        title="Test plot: cell width by condition",
        xlabel="Condition",
        ylabel="Width (µm)",
        show_n=True,
        keep_failed=False,
    )

    print("\nPlotted cell counts:")
    print(plot_result.counts)

    output_path = save_figure(
        plot_result.fig,
        test_dir / "test_width_violin_box.png",
    )

    print(f"\nSaved test plot to: {output_path}")


if __name__ == "__main__":
    main()