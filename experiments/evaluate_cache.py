#!/usr/bin/env python3
"""Evaluate CODT solution-cache effectiveness.

This experiment runs matched CODT configurations with the internal solution
cache enabled and disabled. It records runtime, expanded search nodes, memory
usage, and cache hit statistics from the CLI logs.

Each run is saved as JSON under the output directory. Aggregate CSVs and
figures are regenerated from the JSON cache, so interrupted runs can be resumed.
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
from typing import Any, Dict, Iterable, List, Optional

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
DEFAULT_LOWERBOUNDS = ["class-count", "improvement", "pair", "one-off"]
DEFAULT_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
METRICS = [
    ("runtime_seconds", "Runtime (s)", "cache_effect_runtime.png", False),
    ("expansions", "Expanded search nodes", "cache_effect_expansions.png", True),
    ("memory_usage_bytes", "Memory usage (MiB)", "cache_effect_memory.png", False),
]


@dataclass(frozen=True)
class CacheConfig:
    name: str
    cache_enabled: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CODT runs with the internal solution cache enabled and disabled."
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
        default=Path("experiments/results/cache"),
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
        "--all-datasets",
        action="store_true",
        help="Run every .txt dataset in --data-dir instead of the default panel.",
    )
    parser.add_argument(
        "--strategy",
        default="and-or-dfs-prio",
        help="CODT search strategy to use for every run.",
    )
    parser.add_argument(
        "--lowerbound",
        nargs="+",
        default=DEFAULT_LOWERBOUNDS,
        help="Lower bounds to pass to CODT. Use 'none' to disable lower bounds.",
    )
    parser.add_argument(
        "--upperbound",
        default="for-remaining-interval",
        help="Upper-bound strategy to pass to CODT.",
    )
    parser.add_argument(
        "--cart-upperbound",
        choices=["true", "false"],
        default="true",
        help="Whether CODT should use the CART upper bound.",
    )
    parser.add_argument(
        "--cart-upper-bound-patience",
        type=int,
        default=5,
        help="Patience passed to the CART upper-bound routine.",
    )
    parser.add_argument(
        "--cache-max-branch-budget",
        type=int,
        default=3,
        help="Cache subproblems whose bound gap allows at most this many more branch nodes.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-run solver timeout in seconds.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of matched cache-enabled/cache-disabled repetitions per dataset.",
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
        help="Ignore saved per-run JSON files and rerun everything.",
    )
    return parser.parse_args()


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


def list_datasets(data_dir: Path, requested: Iterable[str], all_datasets: bool) -> List[str]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if all_datasets:
        names = sorted(path.stem for path in data_dir.glob("*.txt"))
        if not names:
            raise ValueError(f"No .txt datasets found in {data_dir}")
        return names
    return list(requested)


def load_dataset_metadata(path: Path) -> Dict[str, int]:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected labels plus at least one feature in {path}")
    return {
        "n_examples": int(frame.shape[0]),
        "n_features": int(frame.shape[1] - 1),
        "n_classes": int(frame.iloc[:, 0].nunique()),
    }


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


def cache_path(output_dir: Path, dataset_name: str, config: CacheConfig, repetition: int) -> Path:
    return output_dir / "runs" / f"{dataset_name}__{config.name}__r{repetition:02d}.json"


def build_command(
    args: argparse.Namespace,
    dataset_path: Path,
    config: CacheConfig,
    codt_cli_binary: Path,
) -> List[str]:
    command = [
        str(codt_cli_binary),
        "--file",
        str(dataset_path),
        "--strategy",
        args.strategy,
        "--lowerbound",
    ]

    lowerbounds = [value for value in args.lowerbound if value != "none"]
    command.extend(lowerbounds)
    command.extend(
        [
            "--upperbound",
            args.upperbound,
            "--cart-upperbound",
            args.cart_upperbound,
            "--cart-upper-bound-patience",
            str(args.cart_upper_bound_patience),
            "--cache",
            str(config.cache_enabled).lower(),
            "--cache-max-branch-budget",
            str(args.cache_max_branch_budget),
            "--data-reduction",
            "true",
            "--memory-limit",
            str(args.memory_limit),
            "--timeout",
            str(args.timeout),
        ]
    )
    return command


def parse_codt_output(output: str) -> Dict[str, Any]:
    is_perfect = "No perfect tree exists for the given data and constraints." not in output
    solver_status = "perfect-tree-found" if is_perfect else "no-perfect-tree"

    branch_match = re.search(r"Branch nodes:\s*(\d+)", output)
    branch_count = int(branch_match.group(1)) if branch_match else None
    tree_size = 2 * branch_count + 1 if branch_count is not None else None

    expansion_match = re.search(r"Graph expansions:\s*(\d+)", output)
    expansions = int(expansion_match.group(1)) if expansion_match else None

    memory_match = re.search(r"Max memory usage \(MB\):\s*([0-9.]+)", output)
    memory_usage_bytes = (
        int(float(memory_match.group(1)) * 1024 * 1024) if memory_match else None
    )

    cache_match = re.search(
        r"Solution cache:\s*(\d+) useful hits / (\d+) lookups \(([0-9.]+)%\), (\d+) entries",
        output,
    )
    cache_useful_hits = int(cache_match.group(1)) if cache_match else None
    cache_lookups = int(cache_match.group(2)) if cache_match else None
    cache_hit_rate = float(cache_match.group(3)) / 100.0 if cache_match else None
    cache_entries = int(cache_match.group(4)) if cache_match else None

    return {
        "solver_status": solver_status,
        "is_perfect": is_perfect,
        "tree_size": tree_size,
        "branch_count": branch_count,
        "expansions": expansions,
        "memory_usage_bytes": memory_usage_bytes,
        "cache_useful_hits": cache_useful_hits,
        "cache_lookups": cache_lookups,
        "cache_hit_rate": cache_hit_rate,
        "cache_entries": cache_entries,
    }


def classify_status(
    solver_status: str,
    runtime_seconds: float,
    timeout: int,
    memory_usage_bytes: Optional[int],
    memory_limit: int,
    error: Optional[str],
) -> str:
    if error is not None:
        message = error.lower()
        if "timeout" in message:
            return "timeout"
        if "memory" in message:
            return "memory-limit"
        return "error"
    if memory_usage_bytes is not None and memory_usage_bytes >= memory_limit:
        return "memory-limit"
    if runtime_seconds >= timeout and solver_status != "perfect-tree-found":
        return "timeout"
    return solver_status


def run_one(
    args: argparse.Namespace,
    dataset_name: str,
    dataset_path: Path,
    metadata: Dict[str, int],
    config: CacheConfig,
    repetition: int,
    codt_cli_binary: Path,
) -> Dict[str, Any]:
    command = build_command(args, dataset_path, config, codt_cli_binary)
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout + 30,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        output = "\n".join(part for part in (stdout, stderr) if part)
        parsed = parse_codt_output(output)
        error = None
        if completed.returncode != 0:
            error = output.strip() or f"codt-cli exited with status {completed.returncode}"
            parsed["solver_status"] = "error"
            parsed["is_perfect"] = False
            parsed["tree_size"] = None
            parsed["branch_count"] = None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        parsed = {
            "solver_status": "error",
            "is_perfect": False,
            "tree_size": None,
            "branch_count": None,
            "expansions": None,
            "memory_usage_bytes": None,
            "cache_useful_hits": None,
            "cache_lookups": None,
            "cache_hit_rate": None,
            "cache_entries": None,
        }
        error = f"codt-cli exceeded its {args.timeout + 30}s process timeout"

    runtime_seconds = time.perf_counter() - started
    status = classify_status(
        parsed["solver_status"],
        runtime_seconds,
        args.timeout,
        parsed["memory_usage_bytes"],
        args.memory_limit,
        error,
    )

    return {
        "dataset": dataset_name,
        **metadata,
        "config": config.name,
        "cache_enabled": config.cache_enabled,
        "repetition": repetition,
        "strategy": args.strategy,
        "lowerbound": ",".join(args.lowerbound),
        "upperbound": args.upperbound,
        "cart_upperbound": args.cart_upperbound,
        "cache_max_branch_budget": args.cache_max_branch_budget,
        "status": status,
        "error": error,
        "runtime_seconds": runtime_seconds,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        **parsed,
    }


def run_experiment(args: argparse.Namespace, codt_cli_binary: Path) -> List[Dict[str, Any]]:
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")

    configs = [
        CacheConfig(name="cache-disabled", cache_enabled=False),
        CacheConfig(name="cache-enabled", cache_enabled=True),
    ]
    dataset_names = list_datasets(args.data_dir, args.datasets, args.all_datasets)
    results: List[Dict[str, Any]] = []

    for dataset_index, dataset_name in enumerate(dataset_names):
        dataset_path = args.data_dir / f"{dataset_name}.txt"
        if not dataset_path.exists():
            print(f"Skipping {dataset_name}: missing {dataset_path}")
            continue

        metadata = load_dataset_metadata(dataset_path)
        print(
            f"\nDataset {dataset_name}: "
            f"{metadata['n_examples']} rows, {metadata['n_features']} features"
        )

        for repetition in range(args.repetitions):
            run_configs = configs if (dataset_index + repetition) % 2 == 0 else list(reversed(configs))
            for config in run_configs:
                path = cache_path(args.output_dir, dataset_name, config, repetition)
                cached = None if args.force else read_json(path)
                if cached is not None and cached.get("strategy") != args.strategy:
                    cached = None
                if cached is not None:
                    results.append(cached)
                    print(f"  cached {config.name} r{repetition}")
                    continue

                print(f"  running {config.name} r{repetition}...", end=" ", flush=True)
                result = run_one(
                    args=args,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    metadata=metadata,
                    config=config,
                    repetition=repetition,
                    codt_cli_binary=codt_cli_binary,
                )
                write_json(path, result)
                results.append(result)

                expansions = result["expansions"]
                memory_mib = (
                    None
                    if result["memory_usage_bytes"] is None
                    else result["memory_usage_bytes"] / 1024 / 1024
                )
                expansion_text = "n/a" if expansions is None else f"{expansions:,}"
                memory_text = "n/a" if memory_mib is None else f"{memory_mib:.1f} MiB"
                print(
                    f"status={result['status']}, runtime={result['runtime_seconds']:.3f}s, "
                    f"expansions={expansion_text}, memory={memory_text}"
                )

    return results


def add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    baseline = frame[frame["config"] == "cache-disabled"][
        ["dataset", "repetition", "runtime_seconds", "expansions", "memory_usage_bytes"]
    ].rename(
        columns={
            "runtime_seconds": "baseline_runtime_seconds",
            "expansions": "baseline_expansions",
            "memory_usage_bytes": "baseline_memory_usage_bytes",
        }
    )
    merged = frame.merge(baseline, on=["dataset", "repetition"], how="left")
    merged["runtime_ratio_to_no_cache"] = (
        merged["runtime_seconds"] / merged["baseline_runtime_seconds"]
    )
    merged["expansion_ratio_to_no_cache"] = (
        (merged["expansions"] + 1) / (merged["baseline_expansions"] + 1)
    )
    merged["memory_ratio_to_no_cache"] = (
        merged["memory_usage_bytes"] / merged["baseline_memory_usage_bytes"]
    )
    return merged


def save_tables(results: List[Dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_ratios(pd.DataFrame(results))
    frame.to_csv(output_dir / "raw_results.csv", index=False)

    if frame.empty:
        pd.DataFrame().to_csv(output_dir / "summary.csv", index=False)
        return frame

    summary = (
        frame.groupby("config", dropna=False)
        .agg(
            runs=("dataset", "count"),
            perfect_runs=("is_perfect", "sum"),
            median_runtime_seconds=("runtime_seconds", "median"),
            median_expansions=("expansions", "median"),
            median_memory_mib=(
                "memory_usage_bytes",
                lambda values: values.median() / 1024 / 1024,
            ),
            median_runtime_ratio_to_no_cache=("runtime_ratio_to_no_cache", "median"),
            median_expansion_ratio_to_no_cache=("expansion_ratio_to_no_cache", "median"),
            median_memory_ratio_to_no_cache=("memory_ratio_to_no_cache", "median"),
            median_cache_hit_rate=("cache_hit_rate", "median"),
            median_cache_entries=("cache_entries", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    return frame


def plot_metric(frame: pd.DataFrame, output_dir: Path, metric: str, ylabel: str, filename: str, log_scale: bool) -> None:
    plot_frame = frame[frame[metric].notna()].copy()
    if plot_frame.empty:
        return
    if metric == "memory_usage_bytes":
        plot_frame[metric] = plot_frame[metric] / 1024 / 1024
    plot_frame = (
        plot_frame.groupby(["dataset", "config"], as_index=False)[metric]
        .median()
    )

    datasets = list(dict.fromkeys(plot_frame["dataset"].tolist()))
    configs = ["cache-disabled", "cache-enabled"]
    labels = {"cache-disabled": "Cache disabled", "cache-enabled": "Cache enabled"}
    colors = tab10_colors(2)
    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 4))
    for index, config in enumerate(configs):
        subset = plot_frame[plot_frame["config"] == config].set_index("dataset")
        values = [
            subset.loc[dataset, metric] if dataset in subset.index else np.nan
            for dataset in datasets
        ]
        offset = (index - 0.5) * width
        ax.bar(x + offset, values, width=width, color=colors[index], label=labels[config])

    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, dpi=200)
    plt.close(fig)


def plot_ratios(frame: pd.DataFrame, output_dir: Path) -> None:
    enabled = frame[frame["config"] == "cache-enabled"].copy()
    ratio_columns = [
        ("runtime_ratio_to_no_cache", "Runtime"),
        ("expansion_ratio_to_no_cache", "Nodes"),
        ("memory_ratio_to_no_cache", "Memory"),
    ]
    enabled = enabled.dropna(subset=[column for column, _ in ratio_columns], how="all")
    if enabled.empty:
        return
    enabled = (
        enabled.groupby("dataset", as_index=False)[
            [column for column, _ in ratio_columns]
        ]
        .median()
    )

    datasets = enabled["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.24
    colors = tab10_colors(len(ratio_columns))

    fig, ax = plt.subplots(figsize=(10, 4))
    for index, (column, label) in enumerate(ratio_columns):
        offset = (index - (len(ratio_columns) - 1) / 2) * width
        ax.bar(x + offset, enabled[column], width=width, color=colors[index], label=label)

    ax.axhline(1.0, color="black", linewidth=1, alpha=0.7)
    ax.set_ylabel("Cache enabled / cache disabled")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "cache_effect_ratios.png", dpi=200)
    plt.close(fig)


def plot_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    for metric, ylabel, filename, log_scale in METRICS:
        plot_metric(frame, output_dir, metric, ylabel, filename, log_scale)
    plot_ratios(frame, output_dir)


def print_summary(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No results to summarize.")
        return

    enabled = frame[frame["config"] == "cache-enabled"]
    if enabled.empty:
        print("No cache-enabled runs to summarize.")
        return
    enabled = (
        enabled.groupby("dataset", as_index=False)
        .agg(
            runtime_ratio_to_no_cache=("runtime_ratio_to_no_cache", "median"),
            expansion_ratio_to_no_cache=("expansion_ratio_to_no_cache", "median"),
            memory_ratio_to_no_cache=("memory_ratio_to_no_cache", "median"),
            cache_useful_hits=("cache_useful_hits", "median"),
            cache_lookups=("cache_lookups", "median"),
            cache_entries=("cache_entries", "median"),
        )
    )

    print("\nCache-enabled ratios vs cache-disabled:")
    columns = [
        "dataset",
        "runtime_ratio_to_no_cache",
        "expansion_ratio_to_no_cache",
        "memory_ratio_to_no_cache",
        "cache_useful_hits",
        "cache_lookups",
        "cache_entries",
    ]
    print(
        enabled[columns].to_string(
            index=False,
            formatters={
                "runtime_ratio_to_no_cache": lambda value: "n/a" if math.isnan(value) else f"{value:.3g}x",
                "expansion_ratio_to_no_cache": lambda value: "n/a" if math.isnan(value) else f"{value:.3g}x",
                "memory_ratio_to_no_cache": lambda value: "n/a" if math.isnan(value) else f"{value:.3g}x",
            },
        )
    )


def main() -> None:
    args = parse_args()
    codt_cli_binary = resolve_codt_cli_binary(args.codt_cli_binary)

    results = run_experiment(args, codt_cli_binary)
    frame = save_tables(results, args.output_dir)
    plot_figures(frame, args.output_dir)
    print_summary(frame)
    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
