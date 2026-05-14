#!/usr/bin/env python3
"""Print a LaTeX table summarizing the sampled OpenML datasets.

The sampled files in data/openml/sampled are whitespace-separated with the
class label in the first column and features in the remaining columns.
This script reports, for each dataset:

- |D|: number of examples
- |F|: number of features
- sum_f |W^f|: total number of candidate split values across all features

The output matches the two-column table layout used in the thesis notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table summarizing the datasets in data/openml/sampled."
    )
    parser.add_argument(
        "--sampled-dir",
        type=Path,
        default=Path("data/openml/sampled"),
        help="Directory containing whitespace-separated sampled datasets.",
    )
    return parser.parse_args()


def list_sampled_datasets(sampled_dir: Path) -> List[Path]:
    if not sampled_dir.exists():
        raise FileNotFoundError(f"Sampled dataset directory not found: {sampled_dir}")

    datasets = sorted(path for path in sampled_dir.iterdir() if path.is_file())
    if not datasets:
        raise ValueError(f"No sampled datasets found in {sampled_dir}")
    return datasets


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None)


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for raw, replacement in replacements.items():
        escaped = escaped.replace(raw, replacement)
    return escaped


def count_candidate_splits(frame: pd.DataFrame) -> int:
    total = 0
    for column in frame.columns[1:]:
        unique_values = frame[column].dropna().nunique()
        total += max(int(unique_values) - 1, 0)
    return total


def summarize_dataset(path: Path) -> Tuple[str, int, int, int]:
    frame = load_dataset(path)
    dataset_name = escape_latex(path.stem)
    n_examples = int(frame.shape[0])
    n_features = int(frame.shape[1] - 1)
    n_splits = count_candidate_splits(frame)
    return dataset_name, n_examples, n_features, n_splits


def split_pairs(items: Sequence[Tuple[str, int, int, int]]) -> Iterable[Tuple[Tuple[str, int, int, int], Tuple[str, int, int, int] | None]]:
    midpoint = (len(items) + 1) // 2
    left_column = items[:midpoint]
    right_column = items[midpoint:]

    for index, left in enumerate(left_column):
        right = right_column[index] if index < len(right_column) else None
        yield left, right


def format_row(left: Tuple[str, int, int, int], right: Tuple[str, int, int, int] | None) -> str:
    left_name, left_d, left_f, left_w = left
    if right is None:
        return f"{left_name} & {left_d} & {left_f} & {left_w} &  &  &  &  \\\\"

    right_name, right_d, right_f, right_w = right
    return f"{left_name} & {left_d} & {left_f} & {left_w} & {right_name} & {right_d} & {right_f} & {right_w} \\\\"


def main() -> None:
    args = parse_args()
    datasets = list_sampled_datasets(args.sampled_dir)
    summary = [summarize_dataset(path) for path in datasets]

    header = (
        '        Dataset & $|D|$ & $|F|$ & $\\sum_f |W^f|$ & '
        'Dataset & $|D|$ & $|F|$ & $\\sum_f |W^f|$ '
        + r'\\'
    )

    print("\\begin{table}[!htbp]")
    print("    \\centering")
    print("    \\begin{tabular}{lrrr@{\\hspace{1.5em}}lrrr}")
    print("        \\hline")
    print(header)
    print("        \\hline")

    for left, right in split_pairs(summary):
        print(f"        {format_row(left, right)}")

    print("        \\hline")
    print("    \\end{tabular}")
    print(
        "    \\caption{Datasets used in the experiments with the number of examples $\\D$, "
        "number of features $F$, and number of feature tests $\\sum_f |W^f|$.}"
    )
    print("    \\label{tab:datasets}")
    print("\\end{table}")


if __name__ == "__main__":
    main()