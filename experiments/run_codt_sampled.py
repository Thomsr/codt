#!/usr/bin/env python3
"""
Load a dataset from data/openml/sampled and run the `codt` Python classifier.

Usage:
  SAMPLED_DATASET=iris.txt python experiments/run_codt_sampled.py
"""
import os
import sys
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report

from codt_py import OptimalDecisionTreeClassifier


def find_data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'openml', 'sampled'))


def load_dataset(path):
    # Try common whitespace-delimited format used in sampled files
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        return df
    except Exception as e:
        raise


def main(dataset_name=None):
    data_dir = find_data_dir()
    if dataset_name is None:
        dataset_name = os.environ.get('SAMPLED_DATASET', 'analcatdata_bankruptcy.txt')

    path = os.path.join(data_dir, dataset_name)
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        sys.exit(2)

    print(f"Loading dataset: {path} {dataset_name}")
    df = load_dataset(path)
    print("Loaded shape:", df.shape)

    if df.shape[1] < 2:
        print("Dataset has fewer than 2 columns — cannot form X and y")
        sys.exit(3)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    try:
        y = y.astype(int)
    except Exception:
        pass

    clf = OptimalDecisionTreeClassifier()
    print("Training OptimalDecisionTreeClassifier on full dataset (no split)...")
    clf.fit(X, y)

    # Evaluate on the same training data since user requested all data is training data
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Training accuracy: {acc:.4f}")
    try:
        print("Classification report:\n", classification_report(y, y_pred))
    except Exception:
        pass

if __name__ == '__main__':
    ds = None
    if len(sys.argv) > 1:
        ds = sys.argv[1]
    main(ds)
