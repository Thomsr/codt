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


def plot_edfc(codt_results: Dict[str, Dict], witty_results: Dict[str, Dict], 
              output_path: Path = None, timeout_seconds: float = 1800):
    """
    Plot ECDF comparing two solvers.
    Only plots solved instances, normalizing by total instances so y-axis reaches actual solve fraction.
    """
    # Extract data
    codt_runtimes, codt_statuses, codt_solved, codt_mem_errors = extract_runtime_and_status(codt_results, "codt")
    witty_runtimes, witty_statuses, witty_solved, witty_mem_errors = extract_runtime_and_status(witty_results, "witty")
    
    codt_items = len(codt_solved)
    witty_items = len(witty_solved)
    
    # Extract only solved runtimes.
    # Timeouts are shown at the timeout boundary so they do not look like short runs.
    codt_solved_runtimes = [rt for rt, solved in zip(codt_runtimes, codt_solved) if solved]
    codt_unsolved_runtimes = [
        timeout_seconds if status == "timeout" else rt
        for rt, status, solved in zip(codt_runtimes, codt_statuses, codt_solved)
        if not solved
    ]

    witty_solved_runtimes = [rt for rt, solved in zip(witty_runtimes, witty_solved) if solved]
    witty_unsolved_runtimes = [
        timeout_seconds if status == "timeout" else rt
        for rt, status, solved in zip(witty_runtimes, witty_statuses, witty_solved)
        if not solved
    ]
    
    # Count metrics
    codt_timeouts = sum(1 for s in codt_statuses if s == "timeout")
    codt_mem_errors = sum(1 for s in codt_statuses if s == "memory_error")
    codt_solved_count = sum(codt_solved)
    
    witty_timeouts = sum(1 for s in witty_statuses if s == "timeout")
    witty_mem_errors = sum(1 for s in witty_statuses if s == "memory_error")
    witty_solved_count = sum(witty_solved)

    # Create figure for runtime comparison
    fig, ax1 = plt.subplots(1, 1, figsize=(8.5, 5.5))
    
    # Plot CODT solved instances
    if codt_solved_runtimes:
        codt_sorted = np.sort(np.asarray(codt_solved_runtimes))
        codt_fraction = np.arange(1, len(codt_sorted) + 1) / codt_items
        ax1.step(
            codt_sorted,
            codt_fraction,
            where='post',
            linewidth=4,
            color="#00A6D6",
            label="CodTree",
        )
    
    # Plot Witty solved instances
    if witty_solved_runtimes:
        witty_sorted = np.sort(np.asarray(witty_solved_runtimes))
        witty_fraction = np.arange(1, len(witty_sorted) + 1) / witty_items
        ax1.step(
            witty_sorted,
            witty_fraction,
            where='post',
            linewidth=4,
            color="#BABABA",
            label="Witty",
        )
    
    # Plot unsolved instances as scatter points at y=0
    if codt_unsolved_runtimes:
        ax1.scatter(
            codt_unsolved_runtimes,
            np.zeros(len(codt_unsolved_runtimes)),
            marker='x',
            s=35,
            color="#00A6D6",
            alpha=0.7,
        )
    
    if witty_unsolved_runtimes:
        ax1.scatter(
            witty_unsolved_runtimes,
            np.zeros(len(witty_unsolved_runtimes)),
            marker='x',
            s=35,
            color="#BABABA",
            alpha=0.7,
        )
    
    ax1.set_xscale('log')
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel('Runtime (seconds, log scale)', fontsize=18)
    ax1.set_ylabel('Fraction of instances solved', fontsize=18)
    # ax1.set_title('Runtime vs Instances Solved', fontsize=20)
    ax1.grid(True, which='major', alpha=0.4)
    ax1.grid(True, which='minor', alpha=0.4, linestyle=':')
    ax1.legend(fontsize=15)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

    plot_search_nodes(codt_results, witty_results, output_path)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SOLVER COMPARISON SUMMARY")
    print("="*60)
    print(f"\nCODT:")
    print(f"  Solved:          {codt_solved_count}/{len(codt_solved)}")
    print(f"  Memory errors:   {codt_mem_errors} (returned tree but exceeded memory limit)")
    print(f"  Timeouts:        {codt_timeouts}")
    if sum(codt_solved) > 0:
        print(f"  Avg runtime (solved): {np.mean([r for r, s in zip(codt_runtimes, codt_solved) if s]):.2f}s")
        print(f"  Median runtime (solved): {np.median([r for r, s in zip(codt_runtimes, codt_solved) if s]):.2f}s")
    
    print(f"\nWitty:")
    print(f"  Solved:          {witty_solved_count}/{len(witty_solved)}")
    print(f"  Memory errors:   {witty_mem_errors}")
    print(f"  Timeouts:        {witty_timeouts}")
    if sum(witty_solved) > 0:
        print(f"  Avg runtime (solved): {np.mean([r for r, s in zip(witty_runtimes, witty_solved) if s]):.2f}s")
        print(f"  Median runtime (solved): {np.median([r for r, s in zip(witty_runtimes, witty_solved) if s]):.2f}s")
    
    # Head-to-head comparison
    print(f"\nHead-to-head:")
    codt_faster = sum(1 for cr, wr, cs, ws in zip(codt_runtimes, witty_runtimes, codt_solved, witty_solved)
                      if cs and ws and cr < wr)
    witty_faster = sum(1 for cr, wr, cs, ws in zip(codt_runtimes, witty_runtimes, codt_solved, witty_solved)
                       if cs and ws and wr < cr)
    print(f"  CODT faster: {codt_faster}")
    print(f"  Witty faster: {witty_faster}")
    print("="*60 + "\n")


def plot_search_nodes(codt_results: Dict[str, Dict], witty_results: Dict[str, Dict], output_path: Path = None):
    """Plot CODT expansions against Witty search-tree nodes checked."""
    codt_items = sorted(codt_results.items())
    witty_items = sorted(witty_results.items())

    codt_solved_nodes: List[int] = []
    codt_unsolved_nodes: List[int] = []
    for _, result in codt_items:
        expansions = result.get("expansions")
        if expansions is None:
            continue
        if result.get("solved", False) and not _is_memory_error_result(result) and not _is_timeout_result(result):
            codt_solved_nodes.append(int(expansions))
        else:
            codt_unsolved_nodes.append(int(expansions))

    witty_solved_nodes: List[int] = []
    witty_unsolved_nodes: List[int] = []
    for _, result in witty_items:
        nodes_checked = _get_witty_search_tree_nodes_checked(result)
        if nodes_checked is None:
            continue
        if result.get("optimal", False) and not result.get("timed_out", False):
            witty_solved_nodes.append(int(nodes_checked))
        else:
            witty_unsolved_nodes.append(int(nodes_checked))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))

    if codt_solved_nodes:
        codt_sorted = np.sort(np.asarray(codt_solved_nodes))
        codt_fraction = np.arange(1, len(codt_sorted) + 1) / len(codt_items)
        ax.step(
            codt_sorted,
            codt_fraction,
            where='post',
            linewidth=4,
            color="#00A6D6",
            label="CodTree expansions",
        )

    if witty_solved_nodes:
        witty_sorted = np.sort(np.asarray(witty_solved_nodes))
        witty_fraction = np.arange(1, len(witty_sorted) + 1) / len(witty_items)
        ax.step(
            witty_sorted,
            witty_fraction,
            where='post',
            linewidth=4,
            color="#BABABA",
            label="Witty nodes checked",
        )

    if codt_unsolved_nodes:
        ax.scatter(
            codt_unsolved_nodes,
            np.zeros(len(codt_unsolved_nodes)),
            marker='x',
            s=35,
            color="#00A6D6",
            alpha=0.7,
        )

    if witty_unsolved_nodes:
        ax.scatter(
            witty_unsolved_nodes,
            np.zeros(len(witty_unsolved_nodes)),
            marker='x',
            s=35,
            color="#BABABA",
            alpha=0.7,
        )

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Search effort (log scale)', fontsize=18)
    # ax.set_title('CODT Expansions vs Witty Nodes Checked', fontsize=20)
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.4, linestyle=':')
    ax.legend(fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

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
        description="Plot EDFC comparing CODT and Witty solver results"
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
    
    # Load results
    print("Loading CODT results...")
    codt_results = load_solver_results(args.results_dir / "codt-cache", "codt")
    print(f"  Found {len(codt_results)} CODT results")
    
    print("Loading Witty results...")
    witty_results = load_solver_results(args.results_dir / "witty-cache", "witty")
    print(f"  Found {len(witty_results)} Witty results")
    
    # Create plot
    print(f"Creating EDFC plot...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_edfc(codt_results, witty_results, args.output, args.timeout)


if __name__ == "__main__":
    main()
