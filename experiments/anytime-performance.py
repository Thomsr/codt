# anytime-performance.py
"""
Run codt with different search strategies and generate anytime performance plots.

This script:
1. Loads datasets from data/openml/sampled/
2. Runs OptimalDecisionTreeClassifier with different strategies
3. Extracts intermediate bounds
4. Generates anytime performance plots
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from collections import defaultdict
from codt_py import OptimalDecisionTreeClassifier, all_search_strategies


def set_style():
    sns.set_context('paper')
    plt.rc('font', size=10, family='sans-serif')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    plt.rc('xtick', labelsize='small')
    plt.rc('ytick', labelsize='small')
    plt.rc('axes', labelsize='medium', grid=True)
    plt.rc('legend', fontsize='small')
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('text', usetex=False)
    sns.set_palette("colorblind")


def find_data_dir():
    """Find the data/openml/sampled directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'openml', 'sampled'))


def load_dataset(path):
    """Load a dataset from a whitespace-delimited file."""
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    return df


def extract_cost_components(cost):
    """Return (misclassification_score, branch_count) from a CODT cost value."""
    if hasattr(cost, "primary") and hasattr(cost, "secondary"):
        return int(cost.primary), int(cost.secondary)
    if isinstance(cost, (tuple, list)) and len(cost) >= 2:
        return int(cost[0]), int(cost[1])
    return int(cost), 0


def serialize_cost(cost):
    """Convert a cost value into a JSON-friendly representation."""
    primary, secondary = extract_cost_components(cost)
    return [primary, secondary]


def serialize_intermediates(intermediates):
    """Convert intermediate bounds into a JSON-friendly representation."""
    return [
        {
            "cost": serialize_cost(cost),
            "expansions": int(expansions),
            "time": float(elapsed),
        }
        for cost, expansions, elapsed in intermediates
    ]


def format_expansion_tick(value, _):
    """Format graph-expansion ticks using k/M suffixes."""
    abs_value = abs(value)

    if abs_value >= 1_000_000:
        scaled = value / 1_000_000
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"

    if abs_value >= 1_000:
        scaled = value / 1_000
        text = f"{scaled:.0f}" if scaled.is_integer() else f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"

    return f"{int(value)}"


def deserialize_intermediates(intermediates):
    """Restore intermediate bounds from the JSON cache."""
    return [
        ((entry["cost"][0], entry["cost"][1]), entry["expansions"], entry["time"])
        for entry in intermediates
    ]


def classify_solver_status(status, runtime_seconds, timeout_seconds, memory_usage_bytes, memory_limit_bytes, error=None):
    """Normalize solver outcomes into a small set of saved statuses."""
    message = "" if error is None else str(error).lower()

    if error is not None:
        if "timeout" in message:
            return "timeout"
        if "memory" in message or "out of memory" in message:
            return "memory-limit"
        return "error"

    if memory_limit_bytes is not None and memory_usage_bytes is not None and memory_usage_bytes >= memory_limit_bytes:
        return "memory-limit"

    if timeout_seconds is not None and runtime_seconds is not None and runtime_seconds >= timeout_seconds and status != "perfect-tree-found":
        return "timeout"

    return status


def save_results(results_list, results_file):
    """Write all experiment results to disk."""
    serializable = []
    for result in results_list:
        result_copy = dict(result)
        result_copy["cost"] = serialize_cost(result_copy["cost"]) if result_copy["cost"] is not None else None
        result_copy["intermediate_ubs"] = serialize_intermediates(result_copy["intermediate_ubs"])
        result_copy["intermediate_lbs"] = serialize_intermediates(result_copy["intermediate_lbs"])
        serializable.append(result_copy)

    with results_file.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def save_result(result, results_dir):
    """Write one experiment result to its own JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{result['dataset']}__{result['strategy']}.json"
    result_file = results_dir / file_name
    payload = dict(result)
    payload["cost"] = serialize_cost(payload["cost"]) if payload["cost"] is not None else None
    payload["intermediate_ubs"] = serialize_intermediates(payload["intermediate_ubs"])
    payload["intermediate_lbs"] = serialize_intermediates(payload["intermediate_lbs"])
    with result_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_results(results_file):
    """Load experiment results from disk."""
    with results_file.open("r", encoding="utf-8") as handle:
        raw_results = json.load(handle)

    results = []
    for result in raw_results:
        result_copy = dict(result)
        if result_copy.get("cost") is not None:
            result_copy["cost"] = tuple(result_copy["cost"])
        result_copy["intermediate_ubs"] = deserialize_intermediates(result_copy["intermediate_ubs"])
        result_copy["intermediate_lbs"] = deserialize_intermediates(result_copy["intermediate_lbs"])
        results.append(result_copy)

    return results


def result_key(dataset_name, strategy):
    """Unique key for a dataset/strategy run."""
    return (dataset_name, strategy)


def print_intermediate_bounds(result):
    """Print intermediate upper and lower bounds for one solver run."""
    print("    Intermediate upper bounds:")
    for score, expansions, elapsed in result["intermediate_ubs"]:
        misclassifications, branch_count = extract_cost_components(score)
        print(
            f"      ub: misclassifications={misclassifications}, branch_count={branch_count}, "
            f"expansions={expansions}, time={elapsed:.6f}s"
        )

    print("    Intermediate lower bounds:")
    for score, expansions, elapsed in result["intermediate_lbs"]:
        misclassifications, branch_count = extract_cost_components(score)
        print(
            f"      lb: misclassifications={misclassifications}, branch_count={branch_count}, "
            f"expansions={expansions}, time={elapsed:.6f}s"
        )


def get_available_datasets(data_dir):
    """Get list of available datasets."""
    datasets = []
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.txt'):
                datasets.append(f[:-4])  # Remove .txt extension
    return datasets


def run_experiment(dataset_name, X, y, strategy, timeout=600):
    """
    Run OptimalDecisionTreeClassifier with given parameters.
    
    Returns a dict with results including intermediate bounds.
    """
    started = pd.Timestamp.utcnow()
    memory_limit_bytes = 4 * 1024 * 1024 * 1024  # 4 GB

    clf = OptimalDecisionTreeClassifier(
        strategy=strategy,
        timeout=timeout,
        lowerbound="improvement",
        intermediates=True,
        memory_limit=memory_limit_bytes,
    )

    error = None
    try:
        clf.fit(X, y)
        solver_status = clf.status()
        is_perfect = clf.is_perfect()
        tree_size = clf.tree_size()
        branch_count = clf.branch_count()
        expansions = clf.expansions()
        memory_usage_bytes = clf.memory_usage_bytes()
        ubs = clf.intermediate_ubs()
        lbs = clf.intermediate_lbs()
    except Exception as exc:
        error = str(exc)
        solver_status = "error"
        is_perfect = False
        tree_size = None
        branch_count = None
        expansions = None
        memory_usage_bytes = None
        ubs = []
        lbs = []

    runtime_seconds = (pd.Timestamp.utcnow() - started).total_seconds()
    status = classify_solver_status(
        solver_status,
        runtime_seconds,
        timeout,
        memory_usage_bytes,
        memory_limit_bytes,
        error=error,
    )

    final_cost = None
    if ubs:
        final_cost = extract_cost_components(ubs[-1][0])

    results = {
        "dataset": dataset_name,
        "strategy": strategy,
        "status": status,
        "solver_status": solver_status,
        "error": error,
        "timed_out": status == "timeout",
        "memory_limit_exceeded": status == "memory-limit",
        "is_perfect": is_perfect,
        "tree_size": tree_size,
        "branch_count": branch_count,
        "expansions": expansions,
        "memory_usage_bytes": memory_usage_bytes,
        "runtime_seconds": runtime_seconds,
        "cost": final_cost,
        "intermediate_ubs": ubs,
        "intermediate_lbs": lbs,
    }

    return results


def plot_anytime_performance(results_list, output_dir, x_key="time"):
    """
    Plot anytime performance for different strategies.
    
    Args:
        results_list: List of result dicts from run_experiment()
        output_dir: Directory to save plots
        x_key: "time" or "expansions" for x-axis
    """
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    x_values = []
    y_values = []
    for result in results_list:
        for ub in result.get("intermediate_ubs", []):
            x_values.append(ub[2] if x_key == "time" else ub[1])
            y_values.append(extract_cost_components(ub[0])[1])
        for lb in result.get("intermediate_lbs", []):
            x_values.append(lb[2] if x_key == "time" else lb[1])
            y_values.append(extract_cost_components(lb[0])[1])

    if x_values:
        x_min = min(x_values)
        x_max = max(x_values)
        x_span = x_max - x_min
        x_pad = 0.05 * x_span if x_span > 0 else 1.0
        x_limits = (x_min - x_pad, x_max + x_pad)
    else:
        x_limits = None

    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        y_span = y_max - y_min
        y_pad = 0.05 * y_span if y_span > 0 else 1.0
        y_limits = (y_min - y_pad, y_max + y_pad)
    else:
        y_limits = None

    line_color = "#00A6D6"
    
    # Group results by strategy so each strategy gets its own figure.
    grouped = defaultdict(list)
    for result in results_list:
        key = result["strategy"]
        grouped[key].append(result)
    
    for strategy, group_results in grouped.items():
        if not group_results:
            continue
        
        fig, ax = plt.subplots(figsize=(3, 2.51))
        
        for result in group_results:
            ubs = result["intermediate_ubs"]
            lbs = result["intermediate_lbs"]
            
            if not ubs or not lbs:
                print(f"  Skipping {result['dataset']} for {strategy} (no intermediates)")
                continue
            
            # Extract branch counts and x-values
            ub_scores = [extract_cost_components(ub[0])[1] for ub in ubs]
            ub_xs = [ub[2] if x_key == "time" else ub[1] for ub in ubs]
            
            lb_scores = [extract_cost_components(lb[0])[1] for lb in lbs]
            lb_xs = [lb[2] if x_key == "time" else lb[1] for lb in lbs]
            
            # Plot upper and lower bounds using the same blue; lower bound dashed.
            ax.plot(ub_xs, ub_scores, marker='o', drawstyle="steps-post", linewidth=2.0, color=line_color, markersize=5)
            ax.plot(lb_xs, lb_scores, marker='s', drawstyle="steps-post", linewidth=2.0, linestyle='--', color=line_color, markersize=5)
        
        x_label = "Time (s)" if x_key == "time" else "Graph expansions"
        ax.set_xlabel(x_label, fontsize=8)
        if x_key == "expansions":
            ax.set_ylabel("Branch count", fontsize=8)
        else:
            ax.set_ylabel("")
        ax.set_title(f"{strategy}")
        ax.axhline(10, color="grey", linestyle="--", linewidth=1.5, alpha=0.8)
        if x_key == "expansions":
            ax.xaxis.set_major_formatter(FuncFormatter(format_expansion_tick))
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.3)
        
        filename = f"fig-anytime-{x_key}-{strategy}.png"
        plt.tight_layout()
        plt.savefig(output_dir / filename, bbox_inches="tight", pad_inches=0.03)
        plt.close()
        
        print(f"  Saved: {filename}")


def main():
    output_dir = Path("anytime_results")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "results.json"
    per_run_dir = output_dir / "runs"
    
    data_dir = find_data_dir()
    print(f"Data directory: {data_dir}")
    
    # Get available datasets
    datasets_available = get_available_datasets(data_dir)
    print(f"Found {len(datasets_available)} datasets: {datasets_available[:5]}...")
    
    # Select a subset for testing
    datasets_to_use = ["diggle_table_a1"]
    
    # Get available strategies from the library
    available_strategies = all_search_strategies()
    print(f"Available strategies: {available_strategies}")
    
    # # Select strategies to compare
    # strategies_to_use = [
    #     "dfs-prio",
    #     # "bfs-balance-small-lb",
    #     "and-or",
    # ]
    # strategies_to_use = [s for s in strategies_to_use if s in available_strategies]
    strategies_to_use = available_strategies  # Use all available strategies
    print(f"Using strategies: {strategies_to_use}")
    
    all_results = []
    results_by_key = {}

    if results_file.exists():
        print(f"\nLoading cached results from {results_file}...")
        all_results = load_results(results_file)
        results_by_key = {
            result_key(result["dataset"], result["strategy"]): result
            for result in all_results
        }

    # Run any missing experiments.
    print("\nRunning experiments...")
    for dataset_name in datasets_to_use:
        dataset_path = os.path.join(data_dir, f"{dataset_name}.txt")

        if not os.path.exists(dataset_path):
            print(f"  Skipping {dataset_name} (not found)")
            continue

        print(f"\nLoading {dataset_name}...")
        try:
            df = load_dataset(dataset_path)
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
            print(f"  Shape: {X.shape}")
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue

        for strategy in strategies_to_use:
            key = result_key(dataset_name, strategy)
            if key in results_by_key:
                print(f"  Skipping {dataset_name} strategy={strategy} (cached)")
                continue

            try:
                print(f"  Running {dataset_name} strategy={strategy}...", end=" ")
                result = run_experiment(dataset_name, X, y, strategy, timeout=1200)
                all_results.append(result)
                results_by_key[key] = result
                print(f"status={result['status']}")
                print_intermediate_bounds(result)
                save_result(result, per_run_dir)
                save_results(all_results, results_file)
                print(f"    Saved run to {per_run_dir / (dataset_name + '__' + strategy + '.json')}")
                print(f"    Updated aggregate cache at {results_file}")
            except Exception as e:
                print(f"Error: {e}")

    if all_results and not results_file.exists():
        save_results(all_results, results_file)
        print(f"\nSaved results to {results_file}")
    
    if not all_results:
        print("No results generated!")
        return
    
    # Generate plots
    print(f"\nGenerated {len(all_results)} results")
    print("\nGenerating anytime plots...")
    plot_anytime_performance(all_results, output_dir, x_key="time")
    print(f"Time-based plots saved to {output_dir}/")
    
    plot_anytime_performance(all_results, output_dir, x_key="expansions")
    print(f"Expansion-based plots saved to {output_dir}/")
    
    print("\nDone!")


if __name__ == "__main__":
    main()