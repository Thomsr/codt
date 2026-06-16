#!/usr/bin/env python3
"""Run CODT and Witty on sampled datasets with resumable per-dataset caching.

The sampled datasets in `data/openml/sampled` are whitespace-separated files with
the class label in the first column and features in the remaining columns.
Witty expects CSV input with a header row, the features first, and the class
label last, so this script converts each sampled file into a cached Witty-ready
CSV before launching the JAR.

Each solver result is cached separately on disk. If the script is interrupted,
already-finished datasets will be skipped on the next run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    sampled_dir: Path
    output_dir: Path
    witty_jar: Path
    codt_cli_binary: Path
    codt_timeout_seconds: Optional[int]
    witty_timeout_seconds: int
    witty_algorithm_id: int
    witty_max_tree_size: Optional[int]
    witty_upper_bound_time_ms: int
    force: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CODT and Witty on sampled datasets with resumable caching."
    )
    parser.add_argument(
        "--sampled-dir",
        type=Path,
        default=Path("data/openml/sampled"),
        help="Directory containing the sampled whitespace-separated datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/codt-witty-sampled"),
        help="Directory for cached inputs, cached results, and the summary CSV.",
    )
    parser.add_argument(
        "--codt-cli-binary",
        type=Path,
        default=Path("target/release/codt-cli"),
        help="Path to the release-built CODT CLI binary.",
    )
    parser.add_argument(
        "--witty-jar",
        type=Path,
        required=True,
        help="Path to the Witty JAR file.",
    )
    parser.add_argument(
        "--codt-timeout-seconds",
        type=int,
        default=1800,
        help="Timeout for CODT fits in seconds. Defaults to 30 minutes.",
    )
    parser.add_argument(
        "--witty-timeout-seconds",
        type=int,
        default=1800,
        help="Timeout passed to Witty in seconds. Defaults to 30 minutes.",
    )
    parser.add_argument(
        "--witty-algorithm-id",
        type=int,
        default=5,
        help="Witty algorithm id to run. The default is the fully enabled Witty solver.",
    )
    parser.add_argument(
        "--witty-max-tree-size",
        type=int,
        default=None,
        help="Maximum tree size passed to Witty. Defaults to the number of examples in the dataset.",
    )
    parser.add_argument(
        "--witty-upper-bound-time-ms",
        type=int,
        default=30000,
        help="Time in milliseconds spent calculating the upper bound, written to Witty's output arguments.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore caches and rerun every dataset.",
    )
    return parser.parse_args()


def load_sampled_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_codt_cli_binary(binary_path: Path) -> Path:
    resolved = binary_path if binary_path.is_absolute() else repo_root() / binary_path
    if not resolved.exists():
        raise FileNotFoundError(
            f"CODT release binary not found at {resolved}. Build it first with `cargo build --release -p codt-cli`."
        )
    return resolved


def list_sampled_datasets(sampled_dir: Path) -> List[Path]:
    if not sampled_dir.exists():
        raise FileNotFoundError(f"Sampled dataset directory not found: {sampled_dir}")

    datasets = sorted(path for path in sampled_dir.iterdir() if path.is_file())
    if not datasets:
        raise ValueError(f"No sampled datasets found in {sampled_dir}")
    return datasets


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def convert_for_witty(source_path: Path, destination_path: Path) -> Dict[str, Any]:
    frame = load_sampled_dataset(source_path)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least one feature and one label column in {source_path}")

    classes = frame.iloc[:, 0].to_numpy()
    features = frame.iloc[:, 1:].to_numpy()

    unique_classes = np.unique(classes)
    if unique_classes.size != 2:
        raise ValueError(
            f"Witty expects binary labels, but {source_path.name} has {unique_classes.size} classes: "
            f"{unique_classes.tolist()}"
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    header = [f"feature_{index}" for index in range(features.shape[1])] + ["class"]
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_features, row_class in zip(features, classes):
            writer.writerow([*row_features.tolist(), row_class])
    tmp_path.replace(destination_path)

    return {
        "source": str(source_path),
        "destination": str(destination_path),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "binary_classes": unique_classes.tolist(),
    }


def parse_codt_cli_output(output: str) -> Dict[str, Any]:
    status = "no-perfect-tree"
    solved = False
    accuracy: Optional[float] = None
    tree_size: Optional[int] = None
    branch_count: Optional[int] = None
    expansions: Optional[int] = None
    memory_usage_bytes: Optional[int] = None
    tree_repr: Optional[str] = None
    error: Optional[str] = None

    if "No perfect tree exists for the given data and constraints." not in output:
        status = "perfect-tree-found"
        solved = True

    accuracy_match = re.search(r"Accuracy:\s*([0-9.]+)%", output)
    if accuracy_match:
        accuracy = float(accuracy_match.group(1)) / 100.0

    branch_match = re.search(r"Branch nodes:\s*(\d+)", output)
    if branch_match:
        branch_count = int(branch_match.group(1))
        tree_size = 2 * branch_count + 1

    expansion_match = re.search(r"Graph expansions:\s*(\d+)", output)
    if expansion_match:
        expansions = int(expansion_match.group(1))

    memory_match = re.search(r"Max memory usage \(MB\):\s*([0-9.]+)", output)
    if memory_match:
        memory_usage_bytes = int(float(memory_match.group(1)) * 1024 * 1024)

    tree_match = re.search(r"Tree:\s*(.+)", output)
    if tree_match:
        tree_repr = tree_match.group(1).strip()

    if "timeout" in output.lower() and status == "no-perfect-tree":
        error = "timeout"

    return {
        "status": status,
        "solved": solved,
        "accuracy": accuracy,
        "tree_size": tree_size,
        "branch_count": branch_count,
        "expansions": expansions,
        "memory_usage_bytes": memory_usage_bytes,
        "tree": tree_repr,
        "error": error,
    }


def run_codt(dataset_path: Path, timeout_seconds: Optional[int], codt_cli_binary: Path) -> Dict[str, Any]:
    frame = load_sampled_dataset(dataset_path)

    command = [
        str(codt_cli_binary),
        "--file",
        str(dataset_path),
        "--strategy",
        "and-or-dfs-prio",
        "--lowerbound",
        "pair",
        "--lowerbound",
        "improvement",
        "--lowerbound",
        "class-count",
        "--lowerbound",
        "one-off",
        "--upperbound",
        "for-remaining-interval",
        "--cart-upperbound",
        "true",
        "--cart-upper-bound-patience",
        "5",
        "--cache-max-branch-budget",
        "3",
        "--data-reduction",
        "true",
        "--memory-limit",
        str(4 * 1024 * 1024 * 1024),
    ]
    if timeout_seconds is not None:
        command.extend(["--timeout", str(timeout_seconds)])

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=None if timeout_seconds is None else timeout_seconds + 30,
    )
    runtime_seconds = time.perf_counter() - started

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    parsed = parse_codt_cli_output(output)

    if completed.returncode != 0 and parsed["error"] is None:
        parsed["status"] = "error"
        parsed["solved"] = False
        parsed["error"] = output.strip() or f"codt-cli exited with status {completed.returncode}"

    frame_shape = frame.shape

    return {
        "solver": "codt",
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path),
        "n_examples": int(frame_shape[0]),
        "n_features": int(frame_shape[1] - 1),
        "runtime_seconds": runtime_seconds,
        **parsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def run_witty(
    jar_path: Path,
    witty_input_dir: Path,
    dataset_name: str,
    output_path: Path,
    problem_id: int,
    max_tree_size: int,
    timeout_seconds: int,
    algorithm_id: int,
    upper_bound: int,
    upper_bound_time_ms: int,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    command = [
        "java",
        "-jar",
        str(jar_path),
        str(witty_input_dir),
        dataset_name,
        str(tmp_output),
        "1.0",
        "0",
        str(max_tree_size),
        str(timeout_seconds),
        str(problem_id),
        str(algorithm_id),
        str(upper_bound),
        str(upper_bound_time_ms),
    ]

    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds + 30)
    runtime_seconds = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            "Witty failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )

    if not tmp_output.exists():
        raise FileNotFoundError(f"Witty did not create its output file: {tmp_output}")

    lines = [line.strip() for line in tmp_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Witty output file is empty: {tmp_output}")

    result_line = lines[-1]
    fields = result_line.split(";")
    if len(fields) < 36:
        raise ValueError(f"Unexpected Witty output format ({len(fields)} fields): {result_line}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output.replace(output_path)

    return {
        "solver": "witty",
        "dataset": dataset_name,
        "problem_id": int(fields[0]),
        "algorithm_id": int(fields[1]),
        "dataset_name": fields[2],
        "n_examples": int(fields[3]),
        "subset_ratio": float(fields[4]),
        "subset_seed": int(fields[5]),
        "n_features": int(fields[6]),
        "max_tree_size_arg": int(fields[7]),
        "timeout_seconds_arg": int(fields[8]),
        "runtime_milliseconds": int(fields[9]),
        "max_memory_mib": int(fields[10]),
        "timed_out": fields[11] == "true",
        "optimal": fields[12] == "true",
        "tree_size": int(fields[13]),
        "accuracy": float(fields[14]),
        "tree": fields[35],
        "runtime_seconds": runtime_seconds,
        "output_file": str(output_path),
        "raw_line": result_line,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def save_summary_csv(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    frame = pd.DataFrame(list(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    config = RunConfig(
        sampled_dir=args.sampled_dir,
        output_dir=args.output_dir,
        witty_jar=args.witty_jar,
        codt_cli_binary=args.codt_cli_binary,
        codt_timeout_seconds=args.codt_timeout_seconds,
        witty_timeout_seconds=args.witty_timeout_seconds,
        witty_algorithm_id=args.witty_algorithm_id,
        witty_max_tree_size=args.witty_max_tree_size,
        witty_upper_bound_time_ms=args.witty_upper_bound_time_ms,
        force=args.force,
    )

    codt_cli_binary = resolve_codt_cli_binary(config.codt_cli_binary)

    datasets = list_sampled_datasets(config.sampled_dir)
    witty_input_dir = config.output_dir / "witty-inputs"
    codt_cache_dir = config.output_dir / "codt-cache"
    witty_cache_dir = config.output_dir / "witty-cache"
    summary_path = config.output_dir / "summary.csv"

    rows: List[Dict[str, Any]] = []
    total_runs = len(datasets) * 2
    run_index = 0

    for problem_id, dataset_path in enumerate(datasets):
        dataset_name = dataset_path.name
        frame = load_sampled_dataset(dataset_path)
        max_tree_size = config.witty_max_tree_size or int(frame.shape[0])

        witty_input_path = witty_input_dir / f"{dataset_path.stem}.csv"
        witty_cache_path = witty_cache_dir / f"{dataset_path.stem}.json"
        codt_cache_path = codt_cache_dir / f"{dataset_path.stem}.json"

        witty_from_cache = False
        if config.force or not witty_cache_path.exists():
            convert_for_witty(dataset_path, witty_input_path)
            run_index += 1
            print(
                f"[{run_index}/{total_runs}] dataset={dataset_name} solver=witty running...",
                flush=True,
            )
            witty_output_path = witty_cache_dir / f"{dataset_path.stem}.txt"
            witty_result = run_witty(
                jar_path=config.witty_jar,
                witty_input_dir=witty_input_dir,
                dataset_name=witty_input_path.name,
                output_path=witty_output_path,
                problem_id=problem_id,
                max_tree_size=max_tree_size,
                timeout_seconds=config.witty_timeout_seconds,
                algorithm_id=config.witty_algorithm_id,
                upper_bound=max_tree_size,
                upper_bound_time_ms=config.witty_upper_bound_time_ms,
            )
            write_json(witty_cache_path, witty_result)
        else:
            witty_result = read_json(witty_cache_path)
            witty_from_cache = True

        if witty_result is None:
            raise RuntimeError(f"Failed to load cached Witty result for {dataset_name}")
        witty_result = dict(witty_result)
        witty_result["from_cache"] = witty_from_cache
        witty_result["source_dataset"] = dataset_name
        rows.append(witty_result)

        codt_from_cache = False
        if config.force or not codt_cache_path.exists():
            run_index += 1
            print(
                f"[{run_index}/{total_runs}] dataset={dataset_name} solver=codt running...",
                flush=True,
            )
            codt_result = run_codt(dataset_path, config.codt_timeout_seconds, codt_cli_binary)
            write_json(codt_cache_path, codt_result)
        else:
            codt_result = read_json(codt_cache_path)
            codt_from_cache = True

        if codt_result is None:
            raise RuntimeError(f"Failed to load cached CODT result for {dataset_name}")
        codt_result = dict(codt_result)
        codt_result["from_cache"] = codt_from_cache
        codt_result["source_dataset"] = dataset_name
        rows.append(codt_result)

        save_summary_csv(rows, summary_path)

    save_summary_csv(rows, summary_path)
    print(f"Wrote summary to {summary_path}")
    print(f"Cached Witty inputs in {witty_input_dir}")
    print(f"Cached CODT results in {codt_cache_dir}")
    print(f"Cached Witty results in {witty_cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
