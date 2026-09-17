"""
cross_validation.py

PURPOSE
-------
Robustness check requested BEFORE Step 9: run 5-fold Stratified
Cross-Validation on Iris for both models, using the EXACT SAME
algorithm and hyperparameters already used in the single train/test
split experiment (experiment.py / Step 8). No hyperparameter is
tuned separately for either model, and no algorithmic change is made
here relative to baseline_tree.py or bivariate_tree.py.

WHY THIS EXPERIMENT
--------------------
The Step 8 result was based on a SINGLE train/test split (seed=42),
so it could be a lucky or unlucky split. Cross-validation gives a
mean +/- standard deviation over 5 different train/test partitions,
which is a more robust (though still small-scale) estimate of how
each model performs on Iris.

WHAT WE REPORT PER MODEL
--------------------------
- accuracy:        mean +/- std  (over the 5 test folds)
- precision_macro:  mean +/- std
- recall_macro:     mean +/- std
- f1_macro:         mean +/- std
- avg training time
- avg tree depth
- avg number of nodes
- avg number of leaves
- avg number of bivariate nodes    (bivariate tree only; 0 for baseline)
- avg number of univariate fallback nodes (bivariate tree only; equals
  total decision node count for baseline, since ALL baseline splits
  are univariate by construction)

OUT OF SCOPE / NOT CHANGED
----------------------------
- No hyperparameter search or tuning is performed for either model.
- The bivariate tree still uses Gini-based greedy search with the
  fixed orientation set, min_impurity_decrease stopping rule, and
  local bivariate-vs-univariate fallback comparison described in
  bivariate_tree.py -- nothing about the algorithm changes here.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from src.baseline_tree import (
    train_baseline_tree,
    get_tree_depth,
    get_num_nodes,
    get_num_leaves,
)
from src.bivariate_tree import BivariateDecisionTree
from src.metrics import compute_classification_metrics

# -----------------------------------------------------------------------
# EXACT SAME configuration as Step 8 (experiment.py) -- not re-tuned here.
# -----------------------------------------------------------------------
RANDOM_SEED = 42
N_SPLITS = 5

BIVARIATE_MAX_DEPTH = 4
BIVARIATE_MIN_SAMPLES_SPLIT = 5
BIVARIATE_N_ORIENTATIONS = 18
BIVARIATE_MIN_IMPURITY_DECREASE = 1e-7

RESULTS_DIR = "results"


def load_full_iris():
    iris = load_iris()
    return iris.data, iris.target, list(iris.feature_names)


def run_cross_validation():
    X, y, feature_names = load_full_iris()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    baseline_fold_results = []
    bivariate_fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------------- Baseline (same call as Step 8) ----------------
        baseline_model, baseline_time = train_baseline_tree(
            X_train, y_train, max_depth=None, random_seed=RANDOM_SEED
        )
        y_pred_baseline = baseline_model.predict(X_test)
        baseline_cls_metrics = compute_classification_metrics(y_test, y_pred_baseline)

        baseline_fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred_baseline),
            "precision_macro": baseline_cls_metrics["precision"],
            "recall_macro": baseline_cls_metrics["recall"],
            "f1_macro": baseline_cls_metrics["f1"],
            "training_time_seconds": baseline_time,
            "depth": get_tree_depth(baseline_model),
            "num_nodes": get_num_nodes(baseline_model),
            "num_leaves": get_num_leaves(baseline_model),
            # All baseline splits are univariate by construction.
            "num_bivariate_nodes": 0,
            "num_univariate_nodes": get_num_nodes(baseline_model) - get_num_leaves(baseline_model),
        })

        # ---------------- Bivariate (same settings as Step 8) ----------------
        bivariate_model = BivariateDecisionTree(
            max_depth=BIVARIATE_MAX_DEPTH,
            min_samples_split=BIVARIATE_MIN_SAMPLES_SPLIT,
            min_impurity_decrease=BIVARIATE_MIN_IMPURITY_DECREASE,
            n_orientations=BIVARIATE_N_ORIENTATIONS,
            random_seed=RANDOM_SEED,
        )
        start = time.time()
        bivariate_model.fit(X_train, y_train)
        bivariate_time = time.time() - start

        y_pred_bivariate = bivariate_model.predict(X_test)
        bivariate_cls_metrics = compute_classification_metrics(y_test, y_pred_bivariate)

        decision_nodes = bivariate_model.get_decision_nodes()
        num_biv = sum(1 for n in decision_nodes if n["node_type"] == "bivariate")
        num_uni = sum(1 for n in decision_nodes if n["node_type"] == "univariate")

        bivariate_fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred_bivariate),
            "precision_macro": bivariate_cls_metrics["precision"],
            "recall_macro": bivariate_cls_metrics["recall"],
            "f1_macro": bivariate_cls_metrics["f1"],
            "training_time_seconds": bivariate_time,
            "depth": bivariate_model.get_depth(),
            "num_nodes": bivariate_model.get_num_nodes(),
            "num_leaves": bivariate_model.get_num_leaves(),
            "num_bivariate_nodes": num_biv,
            "num_univariate_nodes": num_uni,
        })

    return baseline_fold_results, bivariate_fold_results


def summarize_folds(fold_results, model_name):
    """
    Aggregate per-fold dictionaries into mean +/- std (for accuracy,
    precision, recall, f1) and plain averages (for the structural /
    timing statistics, where a std is less meaningful for a small,
    discrete count like node number, but we still report it for
    completeness).
    """
    df = pd.DataFrame(fold_results)

    summary = {"model_name": model_name}

    # Metrics reported as mean +/- std, as requested.
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        summary[f"{metric}_mean"] = df[metric].mean()
        summary[f"{metric}_std"] = df[metric].std(ddof=1)  # sample std over 5 folds

    # Averages requested (mean across folds).
    for stat in ["training_time_seconds", "depth", "num_nodes", "num_leaves",
                 "num_bivariate_nodes", "num_univariate_nodes"]:
        summary[f"{stat}_mean"] = df[stat].mean()
        summary[f"{stat}_std"] = df[stat].std(ddof=1)

    return summary, df


def save_cv_results(baseline_summary, bivariate_summary,
                     baseline_folds_df, bivariate_folds_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- CSV: one row per model, summary statistics only ---
    summary_rows = [baseline_summary, bivariate_summary]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "cross_validation_results.csv"), index=False)

    # --- JSON: summary + full per-fold breakdown for transparency ---
    full_json = {
        "config": {
            "random_seed": RANDOM_SEED,
            "n_splits": N_SPLITS,
            "bivariate_max_depth": BIVARIATE_MAX_DEPTH,
            "bivariate_min_samples_split": BIVARIATE_MIN_SAMPLES_SPLIT,
            "bivariate_n_orientations": BIVARIATE_N_ORIENTATIONS,
            "bivariate_min_impurity_decrease": BIVARIATE_MIN_IMPURITY_DECREASE,
            "note": "Same algorithm and hyperparameters as Step 8; no per-model tuning.",
        },
        "baseline_summary": baseline_summary,
        "bivariate_summary": bivariate_summary,
        "baseline_per_fold": baseline_folds_df.to_dict(orient="records"),
        "bivariate_per_fold": bivariate_folds_df.to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS_DIR, "cross_validation_results.json"), "w") as f:
        json.dump(full_json, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else o)


def print_cv_summary(baseline_summary, bivariate_summary):
    print("=" * 78)
    print("ROBUSTNESS CHECK -- 5-Fold Stratified Cross-Validation (Iris)")
    print("Same algorithm & hyperparameters as Step 8. No per-model tuning.")
    print("=" * 78)

    def fmt(summary):
        lines = []
        lines.append(f"  Accuracy:         {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        lines.append(f"  Precision (macro): {summary['precision_macro_mean']:.4f} ± {summary['precision_macro_std']:.4f}")
        lines.append(f"  Recall (macro):    {summary['recall_macro_mean']:.4f} ± {summary['recall_macro_std']:.4f}")
        lines.append(f"  F1 (macro):        {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
        lines.append(f"  Avg training time: {summary['training_time_seconds_mean']:.4f}s")
        lines.append(f"  Avg depth:         {summary['depth_mean']:.2f}")
        lines.append(f"  Avg num_nodes:     {summary['num_nodes_mean']:.2f}")
        lines.append(f"  Avg num_leaves:    {summary['num_leaves_mean']:.2f}")
        lines.append(f"  Avg bivariate nodes:  {summary['num_bivariate_nodes_mean']:.2f}")
        lines.append(f"  Avg univariate nodes: {summary['num_univariate_nodes_mean']:.2f}")
        return "\n".join(lines)

    print("\nBaseline Univariate Decision Tree (sklearn):")
    print(fmt(baseline_summary))

    print("\nSimplified Bivariate Decision Tree (custom):")
    print(fmt(bivariate_summary))

    print("\n" + "-" * 78)
    print("Results saved to:")
    print(f"  {RESULTS_DIR}/cross_validation_results.csv")
    print(f"  {RESULTS_DIR}/cross_validation_results.json")
    print("=" * 78)


if __name__ == "__main__":
    baseline_folds, bivariate_folds = run_cross_validation()
    baseline_summary, baseline_df = summarize_folds(baseline_folds, "Baseline Univariate Decision Tree (sklearn)")
    bivariate_summary, bivariate_df = summarize_folds(bivariate_folds, "Simplified Bivariate Decision Tree (custom)")

    save_cv_results(baseline_summary, bivariate_summary, baseline_df, bivariate_df)
    print_cv_summary(baseline_summary, bivariate_summary)
