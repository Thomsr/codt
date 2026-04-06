# Compare Two CODT Solver Configurations

This experiment compares two solver candidates on sampled datasets ordered by difficulty.

## Inputs

- Difficulty order: `experiments/sampled_difficulty_order.txt`
- Sampled datasets: `data/sampled/*.txt`

## What it does

- Creates two independent `OptimalDecisionTreeClassifier` instances.
- Lets you set strategy/lower bound/upper bound/timeout/memory for each one.
- Runs the first `N` instances from easiest to hardest.
- Caches per-instance results in JSON so reruns are fast.
- Produces:
  - Cactus plot: runtime vs instances solved.
  - Scatter plot: solver A runtime (y) vs solver B runtime (x), with point color from green (easy) to red (hard).

## Run

```bash
uv run experiments/compare_two_solvers.py \
  --num-instances 100 \
  --solver-a-name cart_cc \
  --solver-a-strategy dfs-prio \
  --solver-a-lowerbound class-count \
  --solver-a-upperbound cart \
  --solver-b-name fri_pc \
  --solver-b-strategy dfs-prio \
  --solver-b-lowerbound pair-count \
  --solver-b-upperbound for-remaining-interval
```

## Important flags

- `--force`: ignore cache and recompute all selected runs.
- `--cache-file`: choose custom cache location.
- `--output-dir`: choose where CSV files and plots are written.

## Output files

In `experiments/results` by default:

- `comparison_long_latest.csv`
- `comparison_paired_latest.csv`
- `cactus_runtime_vs_instances_solved.png`
- `scatter_runtime_solver_a_vs_b.png`
