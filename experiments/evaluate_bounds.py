#!/usr/bin/env python3
"""Evaluate CODT lower/upper bound configurations by expanded search nodes.

The experiment isolates two kinds of pruning choices:

* lower-bound configurations, with the default upper bound and CART upper bound
* CART upper-bound enabled/disabled, with all lower bounds enabled
* all bounds enabled/disabled as an end-to-end baseline

Each run is cached as JSON under the output directory. The aggregate CSVs and
figures are regenerated from the cache at the end, so the script is resumable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import tab10_colors


DEFAULT_DATASETS = [
    "analcatdata_bankruptcy",
    "analcatdata_election2000",
    "appendicitis_test_edsa",
    "breast-w",
    "chscase_adopt",
    "divorce_prediction",
    "kc1-top5",
    "wine",
]

DEFAULT_UPPERBOUND = "for-remaining-interval"
DEFAULT_CART_UPPERBOUND = "enabled"
DEFAULT_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
BOUND_EXPERIMENT_STRATEGY = "and-or-dfs-prio"
LOWERBOUNDS = ["class-count", "one-off", "pair", "improvement"]
CART_UPPERBOUNDS = ["disabled", "enabled"]
LEGEND_LABELS = {
    "bounds": {
        "none": "No bounds",
        "all": "All bounds",
    },
    "lowerbound": {
        "none": "No lower bound",
        "class-count": "Class count",
        "pair": "Pair",
        "improvement": "Pair-size",
        "one-off": "One-off",
        "all": "All lower bounds",
    },
    "upperbound": {
        "disabled": "CART disabled",
        "enabled": "CART enabled",
    },
}
@dataclass(frozen=True)
class BoundConfig:
    group: str
    name: str
    lowerbound: str
    upperbound: str
    cart_upperbound: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CODT bound configurations using expanded search nodes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/openml/sampled"),
        help="Directory containing sampled whitespace-separated OpenML datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/bounds"),
        help="Directory for cached runs, CSV summaries, and figures.",
    )
    parser.add_argument(
        "--codt-cli-binary",
        type=Path,
        default=Path("target/release/codt-cli"),
        help="Path to the release-built CODT CLI binary.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset basenames to run, without .txt. Defaults to a small mixed panel.",
    )
    parser.add_argument(
        "--strategy",
        default=BOUND_EXPERIMENT_STRATEGY,
        choices=[BOUND_EXPERIMENT_STRATEGY],
        help="CODT search strategy to use for every bound comparison.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-run timeout in seconds.",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=DEFAULT_MEMORY_LIMIT_BYTES,
        help="Per-run memory limit in bytes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached per-run JSON files and rerun everything.",
    )
    parser.add_argument(
        "--skip-pair",
        action="store_true",
        help="Skip configs using the pair lower bound, useful when Gurobi is unavailable.",
    )
    return parser.parse_args()


def load_dataset_metadata(path: Path) -> Dict[str, int]:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected labels plus at least one feature in {path}")

    return {
        "n_examples": int(frame.shape[0]),
        "n_features": int(frame.shape[1] - 1),
        "n_classes": int(frame.iloc[:, 0].nunique()),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_codt_cli_binary(binary_path: Path) -> Path:
    resolved = binary_path if binary_path.is_absolute() else repo_root() / binary_path
    if not resolved.exists():
        raise FileNotFoundError(
            f"CODT release binary not found at {resolved}. "
            "Build it first with `cargo build --release -p codt-cli`."
        )
    return resolved


def canonical_name(value: str) -> str:
    return (
        value.replace(",", "+")
        .replace("/", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )


def cache_path(output_dir: Path, dataset_name: str, config: BoundConfig) -> Path:
    file_name = f"{dataset_name}__{config.group}__{canonical_name(config.name)}.json"
    return output_dir / "runs" / file_name


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def build_configs(skip_pair: bool) -> List[BoundConfig]:
    lowerbounds = list(LOWERBOUNDS)

    if skip_pair:
        lowerbounds = [lb for lb in lowerbounds if lb != "pair"]

    all_lb = ",".join(lowerbounds)
    configs: List[BoundConfig] = []

    configs.append(
        BoundConfig(
            group="lowerbound",
            name="none",
            lowerbound="none",
            upperbound=DEFAULT_UPPERBOUND,
            cart_upperbound=DEFAULT_CART_UPPERBOUND,
        )
    )

    for lowerbound in lowerbounds:
        configs.append(
            BoundConfig(
                group="lowerbound",
                name=lowerbound,
                lowerbound=lowerbound,
                upperbound=DEFAULT_UPPERBOUND,
                cart_upperbound=DEFAULT_CART_UPPERBOUND,
            )
        )

    configs.append(
        BoundConfig(
            group="lowerbound",
            name="all",
            lowerbound=all_lb,
            upperbound=DEFAULT_UPPERBOUND,
            cart_upperbound=DEFAULT_CART_UPPERBOUND,
        )
    )

    for cart_upperbound in CART_UPPERBOUNDS:
        configs.append(
            BoundConfig(
                group="upperbound",
                name=cart_upperbound,
                lowerbound=all_lb,
                upperbound=DEFAULT_UPPERBOUND,
                cart_upperbound=cart_upperbound,
            )
        )

    configs.extend(
        [
            BoundConfig(
                group="bounds",
                name="none",
                lowerbound="none",
                upperbound=DEFAULT_UPPERBOUND,
                cart_upperbound="disabled",
            ),
            BoundConfig(
                group="bounds",
                name="all",
                lowerbound=all_lb,
                upperbound=DEFAULT_UPPERBOUND,
                cart_upperbound=DEFAULT_CART_UPPERBOUND,
            ),
        ]
    )

    seen = set()
    unique_configs = []
    for config in configs:
        key = (config.group, config.name)
        if key not in seen:
            unique_configs.append(config)
            seen.add(key)
    return unique_configs


def classify_status(
    solver_status: str,
    runtime_seconds: float,
    timeout: int,
    error: Optional[str],
) -> str:
    if error is not None:
        message = error.lower()
        if "timeout" in message:
            return "timeout"
        if "memory" in message:
            return "memory-limit"
        return "error"
    if runtime_seconds >= timeout and solver_status != "perfect-tree-found":
        return "timeout"
    return solver_status


def run_one(
    dataset_name: str,
    dataset_path: Path,
    metadata: Dict[str, int],
    config: BoundConfig,
    strategy: str,
    timeout: int,
    memory_limit: int,
    codt_cli_binary: Path,
) -> Dict[str, Any]:
    command = [
        str(codt_cli_binary),
        "--file",
        str(dataset_path),
        "--strategy",
        strategy,
        "--lowerbound",
    ]
    if config.lowerbound != "none":
        command.append(config.lowerbound)
    command.extend(
        [
            "--upperbound",
            config.upperbound,
            "--cart-upperbound",
            str(config.cart_upperbound == "enabled").lower(),
            "--cart-upper-bound-patience",
            "5",
            "--cache",
            "true",
            "--cache-max-branch-budget",
            "3",
            "--data-reduction",
            "true",
            "--memory-limit",
            str(memory_limit),
            "--timeout",
            str(timeout),
        ]
    )

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        output = "\n".join(part for part in (stdout, stderr) if part)
        is_perfect = "No perfect tree exists for the given data and constraints." not in output
        solver_status = "perfect-tree-found" if is_perfect else "no-perfect-tree"
        error = None

        branch_match = re.search(r"Branch nodes:\s*(\d+)", output)
        branch_count = int(branch_match.group(1)) if branch_match else None
        tree_size = 2 * branch_count + 1 if branch_count is not None else None

        expansion_match = re.search(r"Graph expansions:\s*(\d+)", output)
        expansions = int(expansion_match.group(1)) if expansion_match else None

        memory_match = re.search(r"Max memory usage \(MB\):\s*([0-9.]+)", output)
        memory_usage_bytes = (
            int(float(memory_match.group(1)) * 1024 * 1024)
            if memory_match
            else None
        )

        if completed.returncode != 0:
            error = output.strip() or f"codt-cli exited with status {completed.returncode}"
            solver_status = "error"
            is_perfect = False
            tree_size = None
            branch_count = None
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - records hung runs.
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        error = f"codt-cli exceeded its {timeout + 30}s process timeout"
        solver_status = "error"
        is_perfect = False
        tree_size = None
        branch_count = None
        expansions = None
        memory_usage_bytes = None

    runtime_seconds = time.perf_counter() - started
    status = classify_status(solver_status, runtime_seconds, timeout, error)

    return {
        "dataset": dataset_name,
        **metadata,
        "group": config.group,
        "config": config.name,
        "strategy": strategy,
        "lowerbound": config.lowerbound,
        "upperbound": config.upperbound,
        "cart_upperbound": config.cart_upperbound,
        "status": status,
        "solver_status": solver_status,
        "error": error,
        "is_perfect": is_perfect,
        "tree_size": tree_size,
        "branch_count": branch_count,
        "expansions": expansions,
        "runtime_seconds": runtime_seconds,
        "memory_usage_bytes": memory_usage_bytes,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
    }


def run_experiment(
    args: argparse.Namespace,
    configs: List[BoundConfig],
    codt_cli_binary: Path,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for dataset_name in args.datasets:
        dataset_path = args.data_dir / f"{dataset_name}.txt"
        if not dataset_path.exists():
            print(f"Skipping {dataset_name}: missing {dataset_path}")
            continue

        metadata = load_dataset_metadata(dataset_path)
        print(
            f"\nDataset {dataset_name}: "
            f"{metadata['n_examples']} rows, {metadata['n_features']} features"
        )

        for config in configs:
            path = cache_path(args.output_dir, dataset_name, config)
            cached = None if args.force else read_json(path)
            if cached is not None and cached.get("strategy") != args.strategy:
                cached = None
            if cached is not None:
                results.append(cached)
                print(f"  cached {config.group}/{config.name}")
                continue

            print(f"  running {config.group}/{config.name}...", end=" ", flush=True)
            result = run_one(
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                metadata=metadata,
                config=config,
                strategy=args.strategy,
                timeout=args.timeout,
                memory_limit=args.memory_limit,
                codt_cli_binary=codt_cli_binary,
            )
            write_json(path, result)
            results.append(result)

            expansions = result["expansions"]
            expansion_text = "n/a" if expansions is None else f"{expansions:,}"
            print(f"status={result['status']}, expansions={expansion_text}")

    return results


def expansion_ratio_to_reference(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    references = {
        "bounds": "all",
        "lowerbound": "all",
        "upperbound": DEFAULT_CART_UPPERBOUND,
    }

    for group, reference_config in references.items():
        group_frame = frame[frame["group"] == group].copy()
        if group_frame.empty:
            continue

        ref = group_frame[group_frame["config"] == reference_config][
            ["dataset", "expansions"]
        ].rename(columns={"expansions": "reference_expansions"})
        merged = group_frame.merge(ref, on="dataset", how="left")
        # Use expansions + 1 for ratios so datasets solved at the root still
        # have a finite, interpretable comparison.
        merged["expansion_ratio_to_reference"] = (
            (merged["expansions"] + 1) / (merged["reference_expansions"] + 1)
        )
        rows.append(merged)

    if not rows:
        return frame.assign(expansion_ratio_to_reference=np.nan)
    return pd.concat(rows, ignore_index=True)


def save_tables(results: List[Dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame = expansion_ratio_to_reference(frame)
    frame.to_csv(output_dir / "raw_results.csv", index=False)

    solved = frame[frame["expansions"].notna()].copy()
    summary = (
        solved.groupby(["group", "config"], dropna=False)
        .agg(
            runs=("dataset", "count"),
            solved_runs=("status", lambda values: int((values != "error").sum())),
            perfect_runs=("is_perfect", "sum"),
            median_expansions=("expansions", "median"),
            mean_expansions=("expansions", "mean"),
            median_runtime_seconds=("runtime_seconds", "median"),
            median_expansion_ratio_to_reference=(
                "expansion_ratio_to_reference",
                "median",
            ),
        )
        .reset_index()
        .sort_values(["group", "median_expansion_ratio_to_reference", "median_expansions"])
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return frame


def plot_grouped_expansions(frame: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for group, group_frame in frame.groupby("group"):
        plot_frame = group_frame[group_frame["expansions"].notna()].copy()
        if plot_frame.empty:
            continue

        datasets = list(dict.fromkeys(plot_frame["dataset"].tolist()))
        configs = list(dict.fromkeys(plot_frame["config"].tolist()))
        x = np.arange(len(datasets))
        width = min(0.8 / max(len(configs), 1), 0.22)
        colors = tab10_colors(max(3, len(configs)))

        fig, ax = plt.subplots(figsize=(10, 4))

        for idx, config in enumerate(configs):
            subset = plot_frame[plot_frame["config"] == config].set_index("dataset")
            values = [
                subset.loc[dataset, "expansions"] if dataset in subset.index else np.nan
                for dataset in datasets
            ]
            offset = (idx - (len(configs) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[idx % len(colors)],
                label=LEGEND_LABELS[group].get(config, config),
            )

        ax.set_ylabel("Expanded search nodes")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(configs),
            frameon=False,
            columnspacing=1.4,
            handlelength=1.8,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / f"{group}_expansions.png", dpi=200)
        plt.close(fig)


def print_summary(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No results to summarize.")
        return

    display_columns = [
        "group",
        "config",
        "runs",
        "median_expansions",
        "median_expansion_ratio_to_reference",
        "median_runtime_seconds",
    ]
    summary = (
        frame[frame["expansions"].notna()]
        .groupby(["group", "config"], dropna=False)
        .agg(
            runs=("dataset", "count"),
            median_expansions=("expansions", "median"),
            median_expansion_ratio_to_reference=(
                "expansion_ratio_to_reference",
                "median",
            ),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .reset_index()
        .sort_values(["group", "median_expansion_ratio_to_reference", "median_expansions"])
    )
    if summary.empty:
        print("No successful runs to summarize.")
        return

    print("\nExpansion summary:")
    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "median_expansions": lambda value: f"{value:,.0f}",
                "median_expansion_ratio_to_reference": lambda value: (
                    "n/a" if math.isnan(value) else f"{value:.3g}x"
                ),
                "median_runtime_seconds": lambda value: f"{value:.3f}s",
            },
        )
    )


def main() -> None:
    args = parse_args()
    codt_cli_binary = resolve_codt_cli_binary(args.codt_cli_binary)
    configs = build_configs(skip_pair=args.skip_pair)

    print(f"Using {len(configs)} bound configurations:")
    for config in configs:
        print(
            f"  {config.group}/{config.name}: "
            f"lower={config.lowerbound}, upper={config.upperbound}, "
            f"cart={config.cart_upperbound}"
        )

    results = run_experiment(args, configs, codt_cli_binary)
    frame = save_tables(results, args.output_dir)
    plot_grouped_expansions(frame, args.output_dir)
    print_summary(frame)
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
