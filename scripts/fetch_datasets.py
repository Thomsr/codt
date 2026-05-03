#!/usr/bin/env python3
"""
Fetch OpenML datasets with numeric features and binary targets.
Write full datasets to data/openml and 100-row chunks to data/openml/sampled.
"""

import json
from pathlib import Path

import openml
import pandas as pd

# Configure OpenML cache
openml.config.cache_directory = "/tmp/openml_cache"

BASE_DIR = Path(__file__).parent.parent
OPENML_DIR = BASE_DIR / "data" / "openml"
SAMPLED_DIR = OPENML_DIR / "sampled"
OPENML_DIR.mkdir(parents=True, exist_ok=True)
SAMPLED_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DATASET_COUNT = 300
SAMPLED_ROWS = 100


def _to_int(value):
    """Convert OpenML dataframe values to int, handling NaN/None."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _sanitize_filename(name):
    """Build a safe ASCII filename from an OpenML dataset name."""
    cleaned = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch.lower())
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    return out or "dataset"


def _output_stem(name, occurrence=0):
    """Build a clean filename stem, adding a letter suffix only when needed."""
    stem = _sanitize_filename(name)
    if occurrence == 0:
        return stem

    suffix = chr(ord("a") + occurrence)
    return f"{stem}_{suffix}"


def _format_row(label, feature_row):
    """Format one row as: label feature1 feature2 ..."""
    values = [str(int(label))]
    values.extend(f"{float(v):.12g}" for v in feature_row)
    return " ".join(values)


def _write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def _pick_evenly_spaced_rows(group, quota):
    """Pick evenly spaced rows from a dataframe group."""
    if quota <= 0:
        return group.iloc[0:0]
    if quota >= len(group):
        return group

    if quota == 1:
        return group.iloc[[len(group) // 2]]

    positions = []
    last = len(group) - 1
    for i in range(quota):
        pos = round(i * last / (quota - 1))
        positions.append(pos)

    positions = list(dict.fromkeys(positions))
    if len(positions) < quota:
        for pos in range(len(group)):
            if pos not in positions:
                positions.append(pos)
            if len(positions) == quota:
                break

    return group.iloc[sorted(positions[:quota])]


def _select_diverse_datasets(datasets_df, target_count=TARGET_DATASET_COUNT, bins=10):
    """Select a varied subset of datasets across instance-size buckets."""
    if len(datasets_df) <= target_count:
        return datasets_df.copy()

    df = datasets_df.copy()
    df["NumberOfInstances"] = df["NumberOfInstances"].fillna(0)
    df["NumberOfFeatures"] = df["NumberOfFeatures"].fillna(0)
    df = df.sort_values(["NumberOfInstances", "NumberOfFeatures", "name", "did"]).reset_index(drop=False)

    bucket_count = min(bins, len(df))
    df["_bucket"] = pd.qcut(df["NumberOfInstances"], q=bucket_count, duplicates="drop")
    grouped = [group.sort_values(["NumberOfInstances", "NumberOfFeatures", "name", "did"]) for _, group in df.groupby("_bucket", sort=True)]

    quotas = [target_count // len(grouped)] * len(grouped)
    for idx in range(target_count % len(grouped)):
        quotas[idx] += 1

    selected = []
    seen = set()
    for group, quota in zip(grouped, quotas):
        picked = _pick_evenly_spaced_rows(group, quota)
        for row_idx in picked["index"].tolist():
            if row_idx in seen:
                continue
            seen.add(row_idx)
            selected.append(row_idx)

    if len(selected) < target_count:
        for row_idx in df["index"].tolist():
            if row_idx in seen:
                continue
            selected.append(row_idx)
            seen.add(row_idx)
            if len(selected) == target_count:
                break

    return datasets_df.loc[selected].copy()


def _build_binary_labels(y):
    """Map target labels to {0, 1}; return None when not binary."""
    y_series = pd.Series(y)
    unique_vals = [v for v in y_series.dropna().unique()]
    if len(unique_vals) != 2:
        return None

    # Stable deterministic mapping.
    unique_vals = sorted(unique_vals, key=lambda v: str(v))
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    return y_series.map(mapping)


def fetch_datasets(limit=None, max_instances=10000, target_count=TARGET_DATASET_COUNT):
    """Fetch, save, and split suitable OpenML datasets."""
    print("Searching OpenML for binary classification datasets with numeric predictors...")
    print(f"Max dataset size: {max_instances} rows")

    datasets_df = openml.datasets.list_datasets(
        output_format="dataframe",
        status="active",
        number_classes=2,
        size=limit,
    )
    print(f"Total binary datasets from listing: {len(datasets_df)}")

    # Keep only datasets with at most one symbolic column (the class column).
    filtered_df = datasets_df[datasets_df["NumberOfSymbolicFeatures"].fillna(0) <= 1].copy()
    print(f"After symbolic-feature filter: {len(filtered_df)}")

    filtered_df = filtered_df[filtered_df["NumberOfInstances"].fillna(0) <= max_instances].copy()
    print(f"After size filter: {len(filtered_df)}")

    filtered_df = filtered_df[filtered_df["NumberOfFeatures"].fillna(0) <= 100].copy()
    print(f"After feature-count filter: {len(filtered_df)}")

    filtered_df = filtered_df[~filtered_df["name"].fillna("").astype(str).str.startswith("fri_c")].copy()
    print(f"After fri_c filter: {len(filtered_df)}")

    selected_df = _select_diverse_datasets(filtered_df, target_count=target_count)
    print(f"Selected diverse subset: {len(selected_df)} datasets")

    # Keep only one dataset per cleaned name.
    selected_df = selected_df.drop_duplicates(subset=["name"], keep="first").copy()
    print(f"After duplicate-name filter: {len(selected_df)} datasets")

    name_counts = {}
    stems = []
    for _, row in selected_df.iterrows():
        name = str(row.get("name", "unknown"))
        occurrence = name_counts.get(name, 0)
        name_counts[name] = occurrence + 1
        stems.append(_output_stem(name, occurrence))

    saved = []
    skipped = []

    for idx, ((_, row), stem) in enumerate(zip(selected_df.iterrows(), stems), start=1):
        did = _to_int(row.get("did"))
        name = str(row.get("name", "unknown"))
        listed_instances = _to_int(row.get("NumberOfInstances"))
        dataset_file = OPENML_DIR / f"{stem}.txt"

        if idx % 25 == 0:
            print(f"Processing {idx}/{len(filtered_df)}: {name} ({did})", flush=True)

        if listed_instances > max_instances:
            skipped.append(
                {
                    "id": did,
                    "name": name,
                    "reason": f"too large ({listed_instances} rows > {max_instances})",
                }
            )
            continue

        # Resume behavior: if already written, keep existing file.
        if dataset_file.exists():
            with open(dataset_file, "r", encoding="utf-8") as existing_file:
                row_count = sum(1 for _ in existing_file)
            sample_file = SAMPLED_DIR / f"{stem}.txt"
            sample_count = 1 if sample_file.exists() else 0
            saved.append(
                {
                    "id": did,
                    "name": name,
                    "rows": row_count,
                    "features": _to_int(row.get("NumberOfFeatures")) - 1,
                    "file": dataset_file.name,
                    "sampled_parts": sample_count,
                }
            )
            continue

        try:
            dataset = openml.datasets.get_dataset(did, download_data=True)
            target_attr = dataset.default_target_attribute
            if not target_attr:
                skipped.append({"id": did, "name": name, "reason": "missing target"})
                continue

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_attr)

            # Ensure features are numeric.
            X = X.apply(pd.to_numeric, errors="coerce")
            y_bin = _build_binary_labels(y)
            if y_bin is None:
                skipped.append({"id": did, "name": name, "reason": "target is not binary"})
                continue

            valid_mask = X.notna().all(axis=1) & y_bin.notna()
            X = X[valid_mask]
            y_bin = y_bin[valid_mask]

            if len(X) == 0:
                skipped.append({"id": did, "name": name, "reason": "no valid numeric rows"})
                continue

            # Write full dataset.
            rows = [_format_row(label, feats) for label, feats in zip(y_bin.to_numpy(), X.to_numpy())]
            _write_lines(dataset_file, rows)

            # Keep only one sampled file per dataset to bound the sample set.
            sample_file = SAMPLED_DIR / f"{stem}.txt"
            sample_rows = rows[:SAMPLED_ROWS]
            _write_lines(sample_file, sample_rows)
            sample_count = 1

            info = {
                "id": did,
                "name": name,
                "rows": len(rows),
                "features": int(X.shape[1]),
                "file": dataset_file.name,
                "sampled_parts": sample_count,
            }
            saved.append(info)
            print(
                f"Saved {name} ({did}): rows={info['rows']} features={info['features']} sampled={'yes' if sample_count else 'no'}",
                flush=True,
            )

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as exc:
            skipped.append({"id": did, "name": name, "reason": str(exc)})
            print(f"Skipped {name} ({did}): {exc}", flush=True)

    metadata = {
        "saved_count": len(saved),
        "skipped_count": len(skipped),
        "saved": saved,
        "skipped": skipped,
    }

    metadata_file = OPENML_DIR / "openml_binary_numerical_datasets.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved datasets: {len(saved)}")
    print(f"Skipped datasets: {len(skipped)}")
    print(f"Metadata: {metadata_file}")
    print(f"OpenML data directory: {OPENML_DIR}")
    print(f"Sampled directory: {SAMPLED_DIR}")


if __name__ == "__main__":
    import sys

    limit = None
    max_instances = 10000
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Applying OpenML listing size limit: {limit}")
        except ValueError:
            print("Usage: python3 scripts/fetch_datasets.py [limit] [max_instances]")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            max_instances = int(sys.argv[2])
            print(f"Applying max instances filter: {max_instances}")
        except ValueError:
            print("Usage: python3 scripts/fetch_datasets.py [limit] [max_instances]")
            sys.exit(1)

    fetch_datasets(limit=limit, max_instances=max_instances)
