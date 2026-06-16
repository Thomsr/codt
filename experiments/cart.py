from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import TAB10_COLORS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="/tmp/cart_subset_experiment.csv")
    parser.add_argument(
        "--output",
        default="experiments/figures/cart_patience.png",
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

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        counts.index.astype(int),
        counts.values,
        width=0.85,
        color=TAB10_COLORS[0],
        edgecolor="none",
        alpha=0.9,
    )
    
    ax.set_xlabel("Iterations Until Next Improvement", fontsize=13, fontweight="semibold")
    ax.set_ylabel("Frequency", fontsize=13, fontweight="semibold")
    ax.set_title("Patience Distribution of CART Random Subset Improvements", fontsize=14, fontweight="bold", pad=16)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Style improvements
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")
    print(f"Total improvement intervals counted: {len(wait_lengths)}")
    print(f"Distinct patience values: {counts.index.min()}..{counts.index.max()}")


if __name__ == "__main__":
    main()
