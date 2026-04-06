#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codt_py import OptimalDecisionTreeClassifier


@dataclass(frozen=True)
class SolverConfig:
    name: str
    strategy: str
    lowerbound: str
    upperbound: str
    timeout: Optional[int]
    memory_limit: Optional[int]
    intermediates: bool

    def to_model_kwargs(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "lowerbound": self.lowerbound,
            "upperbound": self.upperbound,
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "intermediates": self.intermediates,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two CODT solver configurations on sampled datasets ordered by difficulty."
        )
    )

    parser.add_argument(
        "--difficulty-file",
        type=Path,
        default=Path("experiments/sampled_difficulty_order.txt"),
        help="Text file with one sampled dataset filename per line, from easiest to hardest.",
    )
    parser.add_argument(
        "--sampled-dir",
        type=Path,
        default=Path("data/sampled"),
        help="Directory containing sampled datasets.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=50,
        help="How many instances to run from easiest to hardest.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("experiments/cache/compare_two_solvers_cache.json"),
        help="JSON cache file for per-instance solver runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for CSV output and plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached entries and rerun everything.",
    )

    add_solver_args(parser, "solver-a", "candidate_a")
    add_solver_args(parser, "solver-b", "candidate_b")

    return parser.parse_args()


def add_solver_args(parser: argparse.ArgumentParser, prefix: str, default_name: str) -> None:
    parser.add_argument(
        f"--{prefix}-name",
        type=str,
        default=default_name,
        help=f"Display name for {prefix} in plots.",
    )
    parser.add_argument(
        f"--{prefix}-strategy",
        type=str,
        default="dfs-prio",
        help=f"Search strategy for {prefix}.",
    )
    parser.add_argument(
        f"--{prefix}-lowerbound",
        type=str,
        default="class-count",
        help=f"Lower bound strategy for {prefix}.",
    )
    parser.add_argument(
        f"--{prefix}-upperbound",
        type=str,
        default="for-remaining-interval",
        help=f"Upper bound strategy for {prefix}.",
    )
    parser.add_argument(
        f"--{prefix}-timeout",
        type=int,
        default=None,
        help=f"Timeout in seconds for {prefix}.",
    )
    parser.add_argument(
        f"--{prefix}-memory-limit",
        type=int,
        default=None,
        help=f"Memory limit in MiB for {prefix}.",
    )
    parser.add_argument(
        f"--{prefix}-intermediates",
        action="store_true",
        help=f"Collect intermediate bounds for {prefix}.",
    )


def build_solver_config(args: argparse.Namespace, prefix: str) -> SolverConfig:
    key = prefix.replace("-", "_")
    return SolverConfig(
        name=getattr(args, f"{key}_name"),
        strategy=getattr(args, f"{key}_strategy"),
        lowerbound=getattr(args, f"{key}_lowerbound"),
        upperbound=getattr(args, f"{key}_upperbound"),
        timeout=getattr(args, f"{key}_timeout"),
        memory_limit=getattr(args, f"{key}_memory_limit"),
        intermediates=getattr(args, f"{key}_intermediates"),
    )


def read_difficulty_order(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Difficulty file not found: {path}")

    filenames: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            filenames.append(line)
    if not filenames:
        raise ValueError(f"Difficulty file is empty: {path}")
    return filenames


def select_instances(ordered_files: Sequence[str], sampled_dir: Path, count: int) -> List[Tuple[int, Path]]:
    if count <= 0:
        raise ValueError("--num-instances must be greater than 0")

    selected: List[Tuple[int, Path]] = []
    missing: List[str] = []

    for rank, filename in enumerate(ordered_files, start=1):
        candidate = sampled_dir / filename
        if candidate.exists():
            selected.append((rank, candidate))
        else:
            missing.append(filename)
        if len(selected) >= count:
            break

    if len(selected) < count:
        raise ValueError(
            f"Requested {count} instances, but only found {len(selected)} in {sampled_dir}. "
            f"Missing {len(missing)} files."
        )

    return selected


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    x = frame.iloc[:, 1:].to_numpy()
    y = frame.iloc[:, 0].to_numpy()
    return x, y


def hash_run_key(instance_name: str, solver_config: SolverConfig) -> str:
    payload = {
        "instance": instance_name,
        "solver_config": asdict(solver_config),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("version") != 1:
        return {}
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    return entries


def save_cache(path: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {"version": 1, "entries": entries}
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def run_solver(instance_path: Path, rank: int, solver: SolverConfig) -> Dict[str, Any]:
    x, y = load_dataset(instance_path)

    model = OptimalDecisionTreeClassifier(**solver.to_model_kwargs())

    started = time.perf_counter()
    status: str
    solved: bool
    error: Optional[str] = None
    tree_size: Optional[int] = None
    branch_count: Optional[int] = None
    expansions: Optional[int] = None

    try:
        model.fit(x, y)
        status = model.status()
        solved = bool(model.is_perfect())
        tree_size = int(model.tree_size())
        branch_count = int(model.branch_count())
        expansions = int(model.expansions())
    except Exception as exc:  # pragma: no cover
        status = "error"
        solved = False
        error = str(exc)

    runtime_seconds = time.perf_counter() - started

    return {
        "instance": instance_path.name,
        "instance_path": str(instance_path),
        "difficulty_rank": rank,
        "solver_name": solver.name,
        "solver_strategy": solver.strategy,
        "solver_lowerbound": solver.lowerbound,
        "solver_upperbound": solver.upperbound,
        "solver_timeout": solver.timeout,
        "solver_memory_limit": solver.memory_limit,
        "solver_intermediates": solver.intermediates,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "solved": solved,
        "error": error,
        "tree_size": tree_size,
        "branch_count": branch_count,
        "expansions": expansions,
    }


def build_long_results(
    instances: Sequence[Tuple[int, Path]],
    solvers: Sequence[SolverConfig],
    cache_entries: Dict[str, Dict[str, Any]],
    force: bool,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total_runs = len(instances) * len(solvers)
    run_index = 0

    for rank, instance_path in instances:
        for solver in solvers:
            run_index += 1
            key = hash_run_key(instance_path.name, solver)
            if not force and key in cache_entries:
                row = dict(cache_entries[key])
                row["from_cache"] = True
                source = "cache"
            else:
                row = run_solver(instance_path, rank, solver)
                cache_entries[key] = row
                row["from_cache"] = False
                source = "fresh"

            print(
                f"[{run_index}/{total_runs}] "
                f"{instance_path.name} | solver={solver.name} | {source} | "
                f"status={row['status']} | runtime={row['runtime_seconds']:.3f}s",
                flush=True,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_paired_results(df: pd.DataFrame, solver_a: SolverConfig, solver_b: SolverConfig) -> pd.DataFrame:
    first = df[df["solver_name"] == solver_a.name].set_index("instance")
    second = df[df["solver_name"] == solver_b.name].set_index("instance")

    paired = first.join(second, lsuffix="_a", rsuffix="_b", how="inner")
    paired = paired.reset_index().rename(columns={"instance": "instance"})
    paired["hardness_norm"] = (paired["difficulty_rank_a"] - 1) / max(len(paired) - 1, 1)
    return paired


def save_results(long_df: pd.DataFrame, paired_df: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    long_path = output_dir / f"comparison_long_{timestamp}.csv"
    paired_path = output_dir / f"comparison_paired_{timestamp}.csv"
    latest_long = output_dir / "comparison_long_latest.csv"
    latest_paired = output_dir / "comparison_paired_latest.csv"

    long_df.to_csv(long_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    long_df.to_csv(latest_long, index=False)
    paired_df.to_csv(latest_paired, index=False)

    return long_path, paired_path


def make_cactus_plot(long_df: pd.DataFrame, solver_a: SolverConfig, solver_b: SolverConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))

    for solver in (solver_a, solver_b):
        subset = long_df[(long_df["solver_name"] == solver.name) & (long_df["solved"])].copy()
        runtimes = np.sort(subset["runtime_seconds"].to_numpy())
        x = np.arange(1, len(runtimes) + 1)
        plt.step(x, runtimes, where="post", label=solver.name)

    plt.xlabel("Instances solved")
    plt.ylabel("Runtime (seconds)")
    plt.title("Cactus Plot: Runtime vs. Instances Solved")
    plt.grid(alpha=0.3)
    plt.legend()

    out_path = output_dir / "cactus_runtime_vs_instances_solved.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def make_scatter_plot(paired_df: pd.DataFrame, solver_a: SolverConfig, solver_b: SolverConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    x = paired_df["runtime_seconds_b"].to_numpy()
    y = paired_df["runtime_seconds_a"].to_numpy()
    hardness = paired_df["hardness_norm"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        x,
        y,
        c=hardness,
        cmap="RdYlGn_r",
        s=55,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
    )

    x_lim = float(np.max(x)) if len(x) else 1.0
    y_lim = float(np.max(y)) if len(y) else 1.0
    parity_lim = min(x_lim, y_lim)
    ax.plot([0, parity_lim], [0, parity_lim], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(0, x_lim * 1.02)
    ax.set_ylim(0, y_lim * 1.02)
    ax.set_xlabel(f"{solver_b.name} runtime (seconds)")
    ax.set_ylabel(f"{solver_a.name} runtime (seconds)")
    ax.set_title("Runtime Scatter: Solver Comparison by Instance")
    ax.grid(alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Difficulty (green = easy, red = hard)")

    out_path = output_dir / "scatter_runtime_solver_a_vs_b.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    solver_a = build_solver_config(args, "solver-a")
    solver_b = build_solver_config(args, "solver-b")

    ordered_files = read_difficulty_order(args.difficulty_file)
    instances = select_instances(ordered_files, args.sampled_dir, args.num_instances)

    cache_entries = load_cache(args.cache_file)
    long_df = build_long_results(instances, [solver_a, solver_b], cache_entries, args.force)
    save_cache(args.cache_file, cache_entries)

    paired_df = build_paired_results(long_df, solver_a, solver_b)
    long_csv, paired_csv = save_results(long_df, paired_df, args.output_dir)

    cactus_path = make_cactus_plot(long_df, solver_a, solver_b, args.output_dir)
    scatter_path = make_scatter_plot(paired_df, solver_a, solver_b, args.output_dir)

    cache_hits = int(long_df["from_cache"].sum())
    total_runs = int(len(long_df))

    print(f"Selected instances: {len(instances)}")
    print(f"Evaluations (2 solvers x instances): {total_runs}")
    print(f"Loaded from cache: {cache_hits}")
    print(f"Computed fresh: {total_runs - cache_hits}")
    print(f"Long-form results: {long_csv}")
    print(f"Paired results: {paired_csv}")
    print(f"Cactus plot: {cactus_path}")
    print(f"Scatter plot: {scatter_path}")


if __name__ == "__main__":
    main()
