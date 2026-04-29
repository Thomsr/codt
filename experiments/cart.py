from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def sklearn_branch_count(dataset_path: Path, random_state: int) -> int:
    data = np.loadtxt(dataset_path)
    y = data[:, 0]
    X = data[:, 1:]

    clf = DecisionTreeClassifier(
        criterion="gini",
        min_samples_leaf=1,
        min_samples_split=2,
        max_depth=None,
        random_state=random_state,
    )
    clf.fit(X, y)

    return int((clf.tree_.node_count - 1) // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/tmp/cart_subset_experiment.csv")
    parser.add_argument("--data-dir", default="data/sampled")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--plot-type",
        choices=["line", "patience"],
        default="line",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=1,
        help="Start iteration for the line plot.",
    )
    parser.add_argument(
        "--output",
        default="experiments/figures/avg_gap_vs_iteration.png",
    )
    args = parser.parse_args()

    if not Path(args.file).exists():
        raise FileNotFoundError(
            f"Input CSV not found: {args.file}\n"
            "Run the Rust test first:\n"
            "  cargo test -p codt --test cart cart_subset_experiment -- --nocapture"
        )

    df = pd.read_csv(args.file)
    df = df.sort_values(["dataset", "iteration"]).reset_index(drop=True)

    if args.plot_type == "patience":
        wait_lengths: list[int] = []
        for _, group in df.groupby("dataset", sort=False):
            group = group.sort_values("iteration")

            improvement_iters = []
            prev_best = None
            for _, row in group.iterrows():
                current_best = row["best_so_far"]
                if prev_best is None:
                    prev_best = current_best
                    continue
                if current_best < prev_best:
                    improvement_iters.append(int(row["iteration"]))
                prev_best = current_best

            anchors = [0] + improvement_iters
            for i in range(len(anchors) - 1):
                wait_lengths.append(anchors[i + 1] - anchors[i])

        if not wait_lengths:
            raise ValueError("No improvements found in the data; patience histogram is empty.")

        wait_series = pd.Series(wait_lengths, name="wait")
        counts = wait_series.value_counts().sort_index()

        plt.figure(figsize=(10, 6))
        plt.bar(counts.index.astype(int), counts.values, width=0.9, color="tab:blue")
        plt.xlabel("Iterations Until Next Improvement")
        plt.ylabel("Count Across Datasets")
        plt.title("Patience Distribution of CART Improvements")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to {output_path}")
        print(f"Total improvement intervals counted: {len(wait_lengths)}")
        print(f"Distinct patience values: {counts.index.min()}..{counts.index.max()}")
        return

    df["cart_gap"] = ((df["cart_size"] - df["optimal_size"]) / df["optimal_size"]) * 100
    df["best_gap"] = ((df["best_so_far"] - df["optimal_size"]) / df["optimal_size"]) * 100

    full_df = df[df["iteration"] == 0][["dataset", "cart_gap"]].rename(
        columns={"cart_gap": "gap"}
    )
    full_df["solver"] = "CODT full CART"

    data_dir = Path(args.data_dir)
    dataset_rows = (
        df.groupby("dataset", as_index=False)["optimal_size"].first().sort_values("dataset")
    )
    sklearn_rows = []
    for _, row in dataset_rows.iterrows():
        dataset_name = row["dataset"]
        optimal_size = row["optimal_size"]
        txt_name = str(dataset_name).removesuffix(".csv") + ".txt"
        dataset_path = data_dir / txt_name
        sklearn_size = sklearn_branch_count(dataset_path, args.random_state)
        sklearn_gap = ((sklearn_size - optimal_size) / optimal_size) * 100
        sklearn_rows.append({"dataset": dataset_name, "gap": sklearn_gap})

    sklearn_df = pd.DataFrame(sklearn_rows)
    sklearn_df["solver"] = f"sklearn CART (seed={args.random_state})"

    full_avg_gap = float(np.mean(full_df["gap"]))
    sklearn_avg_gap = float(np.mean(sklearn_df["gap"]))

    plt.figure(figsize=(10, 6))
    subset_best_curve = (
        df[df["iteration"] >= args.start_iteration]
        .groupby("iteration")["best_gap"]
        .mean()
        .reset_index()
        .sort_values("iteration")
    )
    if subset_best_curve.empty:
        raise ValueError(
            f"No data points with iteration >= {args.start_iteration}. "
            "Choose a smaller --start-iteration."
        )

    plt.plot(
        subset_best_curve["iteration"],
        subset_best_curve["best_gap"],
        linestyle="-",
        linewidth=2.5,
        label="CODT subset CART (best_so_far)",
    )
    plt.axhline(
        y=full_avg_gap,
        color="tab:green",
        linestyle="-",
        linewidth=2,
        label="CODT full CART (avg, iter=0)",
    )
    plt.axhline(
        y=sklearn_avg_gap,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=f"sklearn CART (avg, seed={args.random_state})",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Average % above optimal")
    plt.title("Average Distance from Optimal vs Iteration")
    plt.grid(alpha=0.3)
    plt.xlim(left=args.start_iteration)

    plt.legend()
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")
    print(f"CODT full CART average % above optimal: {full_avg_gap:.2f}")
    print(f"sklearn average % above optimal: {sklearn_avg_gap:.2f}")


if __name__ == "__main__":
    main()
