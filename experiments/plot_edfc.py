from typing import Optional, Dict
#!/usr/bin/env python3
"""
Plot EDFC (Empirical Distribution Function of Computation) comparing CODT and Witty.
Shows runtime vs instances solved, with special handling for timeouts and memory errors.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_solver_results(cache_dir: Path, solver_name: str) -> Dict[str, Dict]:
    """Load all JSON result files from a solver's cache directory."""
    results = {}
    for json_file in sorted(cache_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                # Extract dataset name from filename
                dataset_name = json_file.stem
                results[dataset_name] = data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return results


def extract_runtime_and_status(results: Dict[str, Dict], solver_name: str = "unknown", 
                                memory_limit_gb: float = 4.0) -> Tuple[List[float], List[str], List[bool], List[bool]]:
    """
    Extract runtime, status, solved flag, and memory error flag from results.
    Handles both CODT (uses 'solved') and Witty (uses 'optimal') result formats.
    
    Detects memory errors in CODT even if a tree was found (solved=true but memory exceeded).
    The default memory cutoff matches the runner's 4 GiB limit.
    
    Returns:
        - runtimes: list of runtime in seconds
        - statuses: list of status strings (timeout, memory_error, solved, etc)
        - solved: list of boolean indicating if solved
        - has_memory_error: list of boolean indicating if memory error occurred
    """
    runtimes = []
    statuses = []
    solved_flags = []
    memory_error_flags = []
    memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
    timeout_threshold = 1800
    
    for dataset_name, result in sorted(results.items()):
        runtime = result.get("runtime_seconds", 0)
        error = result.get("error")
        
        # Check for memory error (can occur even if a tree was found)
        has_memory_error = False
        if error:
            if "memory" in str(error).lower() or "out of memory" in str(error).lower():
                has_memory_error = True
        
        # For CODT, also check if memory usage exceeded limit
        if not has_memory_error and "memory_usage_bytes" in result:
            mem_used = result.get("memory_usage_bytes", 0)
            if mem_used > memory_limit_bytes:
                has_memory_error = True
        
        # Determine if solved (handle both CODT and Witty formats)
        if "optimal" in result:  # Witty format
            solved = result.get("optimal", False)
            timed_out = result.get("timed_out", False)
        else:  # CODT format
            solved = result.get("solved", False)
            timed_out = False
        
        # Determine status
        if has_memory_error:
            status = "memory_error"
        elif error:
            if "memory" in str(error).lower() or "out of memory" in str(error).lower():
                status = "memory_error"
            elif "timeout" in str(error).lower():
                status = "timeout"
            else:
                status = "error"
        elif timed_out or not solved or runtime > timeout_threshold:
            # Check if it's a timeout case.
            if timed_out or runtime > timeout_threshold:
                status = "timeout"
            else:
                status = "unsolved"
        else:
            status = "solved"
        
        runtimes.append(runtime)
        statuses.append(status)
        solved_flags.append(status == "solved")
        memory_error_flags.append(has_memory_error)
    
    return runtimes, statuses, solved_flags, memory_error_flags


def load_all_results(results_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Load all solver caches in a results directory.

    Expects subdirectories named like `<solver>-cache`. Returns a mapping
    `{solver_name: {dataset: result_dict}}`.
    """
    solvers: Dict[str, Dict[str, Dict]] = {}
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return solvers

    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.endswith("-cache"):
            solver_name = sub.name[: -len("-cache")]
            print(f"Loading results for solver: {solver_name} from {sub}")
            solvers[solver_name] = load_solver_results(sub, solver_name)

    return solvers


def plot_edfc(all_results: Dict[str, Dict[str, Dict]],
              output_path: Path = None, timeout_seconds: float = 1800):
    """Plot ECDFs for all solvers in `all_results` on a single plot."""
    if not all_results:
        print("No solver results provided to plot.")
        return

    # Prepare colors
    solver_names = list(all_results.keys())
    colors = sns.color_palette("tab10", n_colors=max(3, len(solver_names)))

    # Figure
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))

    summary_stats = {}

    for i, solver in enumerate(solver_names):
        results = all_results[solver]
        runtimes, statuses, solved_flags, mem_errors = extract_runtime_and_status(results, solver)

        total = len(solved_flags)
        solved_runtimes = [rt for rt, s in zip(runtimes, solved_flags) if s]
        # Split unsolved into timeouts, memory errors, and others so we can mark them
        timeout_runtimes = [
            timeout_seconds
            for rt, status, s in zip(runtimes, statuses, solved_flags)
            if (not s) and status == 'timeout'
        ]
        memerr_runtimes = [
            (timeout_seconds if status == 'timeout' else rt)  # keep timeout boundary if timed out
            for rt, status, s in zip(runtimes, statuses, solved_flags)
            if (not s) and status == 'memory_error'
        ]
        other_unsolved = [
            (timeout_seconds if status == 'timeout' else rt)
            for rt, status, s in zip(runtimes, statuses, solved_flags)
            if (not s) and status not in ('timeout', 'memory_error')
        ]

        if solved_runtimes:
            sorted_r = np.sort(np.asarray(solved_runtimes))
            fraction = np.arange(1, len(sorted_r) + 1) / total
            ax1.step(
                sorted_r,
                fraction,
                where='post',
                linewidth=3,
                color=colors[i % len(colors)],
                label=solver,
            )

        # Plot unsolved cases: timeouts and memory errors as crosses, others as 'x'
        if other_unsolved:
            ax1.scatter(
                other_unsolved,
                np.zeros(len(other_unsolved)),
                marker='x',
                s=35,
                color=colors[i % len(colors)],
                alpha=0.7,
            )
        if timeout_runtimes:
            ax1.scatter(
                timeout_runtimes,
                np.zeros(len(timeout_runtimes)),
                marker='x',
                s=35,
                color=colors[i % len(colors)],
                alpha=0.7,
            )
        if memerr_runtimes:
            ax1.scatter(
                memerr_runtimes,
                np.zeros(len(memerr_runtimes)),
                marker='x',
                s=35,
                color=colors[i % len(colors)],
                alpha=0.7,
            )

        # collect stats
        summary_stats[solver] = {
            'total': total,
            'solved': sum(solved_flags),
            'timeouts': sum(1 for s in statuses if s == 'timeout'),
            'memory_errors': sum(1 for s in statuses if s == 'memory_error'),
            'avg_solved_runtime': float(np.mean([r for r, s in zip(runtimes, solved_flags) if s])) if any(solved_flags) else None,
            'median_solved_runtime': float(np.median([r for r, s in zip(runtimes, solved_flags) if s])) if any(solved_flags) else None,
        }

    ax1.set_xscale('log')
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel('Runtime (seconds, log scale)')
    ax1.set_ylabel('Fraction of instances solved')
    ax1.grid(True, which='major', alpha=0.4)
    ax1.grid(True, which='minor', alpha=0.4, linestyle=':')
    ax1.legend()
    ax1.tick_params(axis='y')
    ax1.tick_params(axis='x')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    # Plot search nodes for all solvers that have node info
    plot_search_nodes(all_results, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SOLVER COMPARISON SUMMARY")
    print("=" * 60)
    for solver, stats in summary_stats.items():
        print(f"\n{solver}:")
        print(f"  Solved:          {stats['solved']}/{stats['total']}")
        print(f"  Memory errors:   {stats['memory_errors']}")
        print(f"  Timeouts:        {stats['timeouts']}")
        if stats['avg_solved_runtime'] is not None:
            print(f"  Avg runtime (solved): {stats['avg_solved_runtime']:.2f}s")
            print(f"  Median runtime (solved): {stats['median_solved_runtime']:.2f}s")
    print("=" * 60 + "\n")


def plot_search_nodes(all_results: Dict[str, Dict[str, Dict]], output_path: Path = None):
    """Plot search effort (expansions / nodes checked) for all solvers."""
    solver_names = list(all_results.keys())
    colors = sns.color_palette("tab10", n_colors=max(3, len(solver_names)))

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    any_plotted = False
    for i, solver in enumerate(solver_names):
        items = sorted(all_results[solver].items())
        solved_nodes: List[int] = []
        unsolved_nodes: List[int] = []

        for _, result in items:
            # Prefer CODT 'expansions', fallback to Witty search tree nodes
            expansions = result.get("expansions")
            nodes_checked = None
            if expansions is None:
                nodes_checked = _get_witty_search_tree_nodes_checked(result)
            else:
                nodes_checked = expansions

            if nodes_checked is None:
                continue

            is_solved = False
            if 'solved' in result:
                is_solved = bool(result.get('solved', False)) and not _is_memory_error_result(result) and not _is_timeout_result(result)
            elif 'optimal' in result:
                is_solved = bool(result.get('optimal', False)) and not result.get('timed_out', False)

            if is_solved:
                solved_nodes.append(int(nodes_checked))
            else:
                unsolved_nodes.append(int(nodes_checked))

        if solved_nodes:
            sorted_r = np.sort(np.asarray(solved_nodes))
            fraction = np.arange(1, len(sorted_r) + 1) / max(1, len(items))
            ax.step(
                sorted_r,
                fraction,
                where='post',
                linewidth=3,
                color=colors[i % len(colors)],
                label=f"{solver}",
            )
            any_plotted = True

        if unsolved_nodes:
            ax.scatter(
                unsolved_nodes,
                np.zeros(len(unsolved_nodes)),
                marker='x',
                s=35,
                color=colors[i % len(colors)],
                alpha=0.7,
            )

    if not any_plotted:
        print("No node/expansion info found for any solver; skipping node plot.")
        return

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Search effort (log scale)')
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.4, linestyle=':')
    ax.legend()
    ax.tick_params(axis='y')
    ax.tick_params(axis='x')

    plt.tight_layout()

    if output_path:
        nodes_path = output_path.with_name(f"{output_path.stem}_nodes{output_path.suffix}")
        plt.savefig(nodes_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {nodes_path}")

    plt.show()


def _is_memory_error_result(result: Dict[str, Dict]) -> bool:
    error = result.get("error")
    if error and ("memory" in str(error).lower() or "out of memory" in str(error).lower()):
        return True
    memory_usage_bytes = result.get("memory_usage_bytes")
    return memory_usage_bytes is not None and int(memory_usage_bytes) > 4 * 1024 * 1024 * 1024


def _is_timeout_result(result: Dict[str, Dict]) -> bool:
    error = result.get("error")
    if error and "timeout" in str(error).lower():
        return True

    if result.get("timed_out", False):
        return True

    status = result.get("status")
    if status == "timeout":
        return True

    runtime_seconds = result.get("runtime_seconds", 0)
    return runtime_seconds > 1800


def _get_witty_search_tree_nodes_checked(result: Dict[str, Dict]) -> Optional[int]:
    nodes_checked = result.get("search_tree_nodes_checked")
    if nodes_checked is not None:
        return int(nodes_checked)

    raw_line = result.get("raw_line")
    if not raw_line:
        return None

    fields = raw_line.split(";")
    if len(fields) <= 15:
        return None

    try:
        return int(fields[15])
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot EDFC comparing multiple solver results (detects *-cache dirs)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results/codt-witty-sampled"),
        help="Directory containing codt-cache and witty-cache subdirectories"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/edfc_comparison.png"),
        help="Output file for the plot"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Timeout value in seconds (used for unsolved instances)"
    )
    
    args = parser.parse_args()
    
    # Load all solver caches found in results dir
    print("Loading solver caches from results directory...")
    all_results = load_all_results(args.results_dir)
    for s, r in all_results.items():
        print(f"  Found {len(r)} results for solver: {s}")

    # Create plot
    print(f"Creating EDFC plot...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_edfc(all_results, args.output, args.timeout)


if __name__ == "__main__":
    main()
