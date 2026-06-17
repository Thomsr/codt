#!/usr/bin/env python3
"""Print a LaTeX table comparing Witty and CODT on sampled datasets.

The output follows the two-column table layout used by
scripts/summarize_sampled_datasets.py. For each dataset, this script reports
the obtained branch-node count and runtime for Witty and CODT's
and-or-dfs-prio run. The runtime of the solver that found a tree faster is
printed in bold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Result = Dict[str, Any]
SummaryRow = Tuple[
    str,
    Optional[int],
    Optional[float],
    bool,
    bool,
    Optional[int],
    bool,
    Optional[float],
    bool,
    bool,
]


DISPLAY_NAMES = {
    "arcene_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_true": r"arcene\_seed\_0",
    "arcene_seed_3_nrows_2000_nclasses_10_ncols_100_stratify_true": r"arcene\_seed\_3",
    "climate-model-simulation-crashes": r"CMSC\footnote{climate-model-simulation-crashes}",
    "credit_approval_classification": r"credit\_approval",
    "german-credit-data-creditability-2": r"GCDC-2\footnote{german-credit-data-creditability}",
    "german-credit-data-creditability": r"GCDC\footnote{german-credit-data-creditability}",
    "visualizing_environmental": "environmental",
    "visualizing_hamster": "hamster",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table comparing Witty and CODT and-or-dfs-prio results."
    )
    parser.add_argument(
        "--sampled-dir",
        type=Path,
        default=None,
        help="Optional sampled dataset directory used for row order and filtering.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/codt-witty-sampled"),
        help="Directory containing Witty and CODT cache subdirectories.",
    )
    parser.add_argument(
        "--witty-cache",
        type=Path,
        default=None,
        help="Directory containing Witty JSON cache files.",
    )
    parser.add_argument(
        "--codt-cache",
        type=Path,
        default=None,
        help="Directory containing CODT and-or-dfs-prio JSON cache files.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=1800.0,
        help="Runtime threshold used to identify runs that reached the timeout.",
    )
    return parser.parse_args()


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


def format_dataset_name(dataset_name: str) -> str:
    if dataset_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[dataset_name]
    return escape_latex(dataset_name)


def list_dataset_names(
    sampled_dir: Optional[Path], witty_cache_dir: Path, codt_cache_dir: Path
) -> List[str]:
    if sampled_dir is not None and sampled_dir.exists():
        names = sorted(path.stem for path in sampled_dir.iterdir() if path.is_file())
        if names:
            return names

    names = {path.stem for path in witty_cache_dir.glob("*.json")}
    names.update(path.stem for path in codt_cache_dir.glob("*.json"))
    if not names:
        raise ValueError(
            f"No result JSON files found in {witty_cache_dir} or {codt_cache_dir}"
        )
    return sorted(names)


def read_result(cache_dir: Path, dataset_name: str) -> Optional[Result]:
    path = cache_dir / f"{dataset_name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def found_tree(result: Optional[Result]) -> bool:
    if result is None:
        return False
    if result.get("timed_out") is True or result.get("error") == "timeout":
        return False
    return (
        result.get("optimal") is True
        or result.get("solved") is True
        or positive_int(result.get("tree_size")) is not None
    )


def tree_size(result: Optional[Result]) -> Optional[int]:
    if not found_tree(result):
        return None
    branch_count = positive_int(result.get("branch_count"))
    if branch_count is not None:
        return branch_count
    return positive_int(result.get("tree_size"))


def runtime_seconds(result: Optional[Result]) -> Optional[float]:
    if result is None:
        return None
    return numeric(result.get("runtime_seconds"))


def timeout_threshold(result: Optional[Result], timeout_seconds: float) -> float:
    if result is None:
        return timeout_seconds
    configured_timeout = numeric(result.get("timeout_seconds_arg"))
    return configured_timeout if configured_timeout is not None else timeout_seconds


def reached_timeout(result: Optional[Result], timeout_seconds: float) -> bool:
    if result is None:
        return False
    if result.get("timed_out") is True or result.get("error") == "timeout":
        return True

    threshold = timeout_threshold(result, timeout_seconds)
    runtime = runtime_seconds(result)
    return runtime is not None and runtime >= threshold


def summarize_dataset(
    dataset_name: str,
    witty_cache_dir: Path,
    codt_cache_dir: Path,
    timeout_seconds: float,
) -> SummaryRow:
    witty = read_result(witty_cache_dir, dataset_name)
    codt = read_result(codt_cache_dir, dataset_name)

    witty_runtime = runtime_seconds(witty)
    codt_runtime = runtime_seconds(codt)
    witty_found = found_tree(witty)
    codt_found = found_tree(codt)
    witty_timeout = reached_timeout(witty, timeout_seconds)
    codt_timeout = reached_timeout(codt, timeout_seconds)

    bold_witty = False
    bold_codt = False
    if witty_found and codt_found and witty_runtime is not None and codt_runtime is not None:
        if witty_runtime < codt_runtime:
            bold_witty = True
        elif codt_runtime < witty_runtime:
            bold_codt = True
    elif witty_found and witty_runtime is not None:
        bold_witty = True
    elif codt_found and codt_runtime is not None:
        bold_codt = True

    codt_size = tree_size(codt)
    bold_codt_size = (
        codt_size is not None
        and witty_timeout
        and codt_timeout
    )
    if bold_codt_size:
        bold_codt = False
    if witty_timeout:
        witty_runtime = timeout_threshold(witty, timeout_seconds)
    if codt_timeout:
        codt_runtime = timeout_threshold(codt, timeout_seconds)

    return (
        format_dataset_name(dataset_name),
        tree_size(witty),
        witty_runtime,
        witty_timeout,
        bold_witty,
        codt_size,
        bold_codt_size,
        codt_runtime,
        codt_timeout,
        bold_codt,
    )


def split_pairs(items: Sequence[SummaryRow]) -> Iterable[Tuple[SummaryRow, SummaryRow | None]]:
    midpoint = (len(items) + 1) // 2
    left_column = items[:midpoint]
    right_column = items[midpoint:]

    for index, left in enumerate(left_column):
        right = right_column[index] if index < len(right_column) else None
        yield left, right


def format_tree_size(value: Optional[int], bold: bool = False) -> str:
    if value is None:
        return "--"
    formatted = str(value)
    if bold:
        return rf"\textbf{{{formatted}}}"
    return formatted


def format_runtime(value: Optional[float], reached_timeout: bool, bold: bool) -> str:
    if value is None:
        return "--"
    formatted = str(int(value)) if reached_timeout else f"{value:.2f}"
    if bold:
        return rf"\textbf{{{formatted}}}"
    return formatted


def format_half(row: SummaryRow | None) -> str:
    if row is None:
        return " &  &  &  & "

    (
        dataset_name,
        witty_size,
        witty_runtime,
        witty_timeout,
        bold_witty,
        codt_size,
        bold_codt_size,
        codt_runtime,
        codt_timeout,
        bold_codt,
    ) = row
    return (
        f"{dataset_name} & "
        f"{format_tree_size(witty_size)} & {format_runtime(witty_runtime, witty_timeout, bold_witty)} & "
        f"{format_tree_size(codt_size, bold_codt_size)} & {format_runtime(codt_runtime, codt_timeout, bold_codt)}"
    )


def format_row(left: SummaryRow, right: SummaryRow | None) -> str:
    return f"{format_half(left)} & {format_half(right)} \\\\"


def main() -> None:
    args = parse_args()
    witty_cache_dir = args.witty_cache or args.results_dir / "witty-cache"
    codt_cache_dir = args.codt_cache or args.results_dir / "codt-and-or-dfs-prio-cache"

    dataset_names = list_dataset_names(args.sampled_dir, witty_cache_dir, codt_cache_dir)
    summary = [
        summarize_dataset(dataset_name, witty_cache_dir, codt_cache_dir, args.timeout_seconds)
        for dataset_name in dataset_names
    ]

    header = (
        r"        Dataset & \multicolumn{2}{c}{Witty} & \multicolumn{2}{c}{CODT} & "
        r"Dataset & \multicolumn{2}{c}{Witty} & \multicolumn{2}{c}{CODT} \\"
    )
    subheader = (
        r"         & $|T|$ & Time (s) & $|T|$ & Time (s) & "
        r" & $|T|$ & Time (s) & $|T|$ & Time (s) \\"
    )

    print("\\begin{table}[!htbp]")
    print("    \\centering")
    print("    \\begin{tabular}{lrrrr@{\\hspace{1.5em}}lrrrr}")
    print("        \\hline")
    print(header)
    print(subheader)
    print("        \\hline")

    for left, right in split_pairs(summary):
        print(f"        {format_row(left, right)}")

    print("        \\hline")
    print("    \\end{tabular}")
    print(
        "    \\caption{Comparison of Witty and CODT with the and-or-dfs-prio strategy. "
        "Tree size is reported as the number of branch nodes. Runtime is in seconds; "
        "bold marks the solver that found a tree faster.}"
    )
    print("    \\label{tab:witty-codt-sampled}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
