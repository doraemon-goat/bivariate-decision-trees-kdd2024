"""
experiment.py

PURPOSE
-------
Orchestrates STEP 8 of the project: run a fair, side-by-side
experimental comparison between

    A. Standard (univariate) Decision Tree  -- src/baseline_tree.py
    B. Our Simplified Bivariate Decision Tree -- src/bivariate_tree.py

on the Iris dataset, using the SAME train/test split, SAME random
seed, and SAME evaluation code (src/metrics.py) for both models.

This file contains NO modeling logic itself -- it only calls into
baseline_tree.py / bivariate_tree.py / metrics.py, and produces:
    - a printed summary
    - results/*.csv, results/*.json  (numeric results)
    - plots/*.png                    (visualizations)

IMPORTANT SCOPE NOTE
---------------------
This experiment evaluates OUR simplified student implementation of
the bivariate-tree idea (see bivariate_tree.py docstring for exactly
what is reproduced from the paper and what is simplified/out of
scope). It does NOT reproduce the KDD 2024 paper's experiments,
datasets, or the full bivariate TAO algorithm, and the numbers here
should not be compared against the paper's own reported results.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.baseline_tree import (
    train_baseline_tree,
    get_accuracy,
    get_tree_depth,
    get_num_nodes,
    get_num_leaves,
)
from src.bivariate_tree import BivariateDecisionTree
from src.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    build_comparison_table,
)

# -----------------------------------------------------------------------
# Fixed configuration (shared by both models, as required)
# -----------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.3

# Bivariate tree settings -- explicitly stated per the task requirements.
BIVARIATE_MAX_DEPTH = 4
BIVARIATE_MIN_SAMPLES_SPLIT = 5
BIVARIATE_N_ORIENTATIONS = 18
BIVARIATE_MIN_IMPURITY_DECREASE = 1e-7

RESULTS_DIR = "results"
PLOTS_DIR = "plots"


# =========================================================================
# 1. Data loading (identical for both models)
# =========================================================================
def load_shared_data():
    """
    Load Iris and create ONE train/test split, used identically for
    both models, with a fixed random seed for reproducibility.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_names, class_names


# =========================================================================
# 2. Train + evaluate each model into a common summary format
# =========================================================================
def run_baseline(X_train, y_train, X_test, y_test):
    model, training_time = train_baseline_tree(
        X_train, y_train, max_depth=None, random_seed=RANDOM_SEED
    )

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    cls_metrics = compute_classification_metrics(y_test, y_pred_test)
    cm = compute_confusion_matrix(y_test, y_pred_test)

    summary = {
        "model_name": "Baseline Univariate Decision Tree (sklearn)",
        "train_accuracy": get_accuracy(model, X_train, y_train),
        "test_accuracy": get_accuracy(model, X_test, y_test),
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "training_time_seconds": training_time,
        "depth": get_tree_depth(model),
        "num_nodes": get_num_nodes(model),
        "num_leaves": get_num_leaves(model),
    }
    return model, summary, cm


def run_bivariate(X_train, y_train, X_test, y_test):
    import time

    model = BivariateDecisionTree(
        max_depth=BIVARIATE_MAX_DEPTH,
        min_samples_split=BIVARIATE_MIN_SAMPLES_SPLIT,
        min_impurity_decrease=BIVARIATE_MIN_IMPURITY_DECREASE,
        n_orientations=BIVARIATE_N_ORIENTATIONS,
        random_seed=RANDOM_SEED,
    )

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    cls_metrics = compute_classification_metrics(y_test, y_pred_test)
    cm = compute_confusion_matrix(y_test, y_pred_test)

    summary = {
        "model_name": "Simplified Bivariate Decision Tree (custom)",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "training_time_seconds": training_time,
        "depth": model.get_depth(),
        "num_nodes": model.get_num_nodes(),
        "num_leaves": model.get_num_leaves(),
    }
    return model, summary, cm


# =========================================================================
# 3. Bivariate-tree-specific reporting (node type breakdown)
# =========================================================================
def summarize_bivariate_node_types(model, feature_names):
    """
    Report how many decision nodes ended up bivariate vs univariate,
    and list the bivariate nodes' feature pairs and coefficients.
    This is specific to the bivariate tree (the baseline has no such
    notion, since ALL its splits are univariate by construction).
    """
    decision_nodes = model.get_decision_nodes()
    bivariate_nodes = [n for n in decision_nodes if n["node_type"] == "bivariate"]
    univariate_nodes = [n for n in decision_nodes if n["node_type"] == "univariate"]

    bivariate_details = []
    for n in bivariate_nodes:
        bivariate_details.append({
            "feature_pair": (feature_names[n["feat_j"]], feature_names[n["feat_k"]]),
            "w1": n["w1"],
            "w2": n["w2"],
            "b": n["b"],
            "impurity_gain": n["impurity_gain"],
            "n_samples": n["n_samples"],
        })

    return {
        "num_bivariate_nodes": len(bivariate_nodes),
        "num_univariate_nodes": len(univariate_nodes),
        "bivariate_node_details": bivariate_details,
    }


# =========================================================================
# 4. Plotting
# =========================================================================
def plot_accuracy_comparison(baseline_summary, bivariate_summary, save_path):
    labels = ["Train Accuracy", "Test Accuracy"]
    baseline_vals = [baseline_summary["train_accuracy"], baseline_summary["test_accuracy"]]
    bivariate_vals = [bivariate_summary["train_accuracy"], bivariate_summary["test_accuracy"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline (univariate)")
    ax.bar(x + width / 2, bivariate_vals, width, label="Bivariate (ours)")

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy Comparison: Baseline vs. Bivariate Tree")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, v in enumerate(baseline_vals):
        ax.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(bivariate_vals):
        ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tree_size_comparison(baseline_summary, bivariate_summary, save_path):
    labels = ["Depth", "Num. Nodes", "Num. Leaves"]
    baseline_vals = [baseline_summary["depth"], baseline_summary["num_nodes"], baseline_summary["num_leaves"]]
    bivariate_vals = [bivariate_summary["depth"], bivariate_summary["num_nodes"], bivariate_summary["num_leaves"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline (univariate)")
    ax.bar(x + width / 2, bivariate_vals, width, label="Bivariate (ours)")

    ax.set_ylabel("Count")
    ax.set_title("Tree Size Comparison: Baseline vs. Bivariate Tree")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, v in enumerate(baseline_vals):
        ax.text(i - width / 2, v + 0.1, str(v), ha="center", fontsize=9)
    for i, v in enumerate(bivariate_vals):
        ax.text(i + width / 2, v + 0.1, str(v), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cm_baseline, cm_bivariate, class_names, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, cm, title in zip(
        axes,
        [cm_baseline, cm_bivariate],
        ["Baseline (univariate)", "Bivariate (ours)"],
    ):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
    fig.suptitle("Confusion Matrices (Test Set)")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_bivariate_decision_boundary(model, X_train, y_train, feature_names,
                                      class_names, feat_j, feat_k, save_path):
    """
    Visualize the training data in the 2D space of (feat_j, feat_k)
    -- the feature pair used by one of the tree's actual bivariate
    nodes -- and overlay the linear boundary that node learned:

        w1*x[feat_j] + w2*x[feat_k] + b = 0

    NOTE / LIMITATION: this plot shows the boundary of ONE node in
    isolation, projected onto its two features, using ALL training
    points (not just the subset that reaches that node in the tree).
    This is a simplification for visualization purposes -- the
    node's true decision only applies to the subset of samples that
    reach it after earlier splits, but plotting the full dataset
    gives a clearer, more interpretable 2D picture of what that
    node's rule is doing, consistent with the paper's own idea of
    visualizing bivariate splits as 2D scatterplots.
    """
    decision_nodes = model.get_decision_nodes()
    target_node = None
    for n in decision_nodes:
        if n["node_type"] == "bivariate" and n["feat_j"] == feat_j and n["feat_k"] == feat_k:
            target_node = n
            break
    if target_node is None:
        raise ValueError("No bivariate node found for the given feature pair.")

    w1, w2, b = target_node["w1"], target_node["w2"], target_node["b"]

    x_vals = X_train[:, feat_j]
    y_vals = X_train[:, feat_k]

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_vals, y_vals, c=y_train, cmap="viridis", edgecolor="k", alpha=0.8)

    # Decision line: w1*x + w2*y + b = 0  =>  y = -(w1*x + b) / w2  (if w2 != 0)
    x_min, x_max = x_vals.min() - 0.5, x_vals.max() + 0.5
    if abs(w2) > 1e-9:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(w1 * x_line + b) / w2
        ax.plot(x_line, y_line, "r--", linewidth=2, label="Bivariate split boundary")
    else:
        # vertical line case (w2 == 0)
        x_line_val = -b / w1
        ax.axvline(x_line_val, color="r", linestyle="--", linewidth=2, label="Bivariate split boundary")

    ax.set_xlabel(feature_names[feat_j])
    ax.set_ylabel(feature_names[feat_k])
    ax.set_title(f"Bivariate Split Boundary\n({feature_names[feat_j]} vs. {feature_names[feat_k]})")
    ax.legend(loc="best")

    handles, _ = scatter.legend_elements()
    legend2 = ax.legend(handles, class_names, title="Class", loc="lower right")
    ax.add_artist(legend2)
    ax.legend(loc="upper left")  # re-add boundary legend since add_artist overwrote it

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# 5. Saving numeric results
# =========================================================================
def _to_jsonable(obj):
    """
    Recursively convert numpy scalar/array types into plain Python
    types so the results dictionary can be serialized with json.dump.
    This is a pure engineering utility, unrelated to the algorithm.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(comparison_table, baseline_summary, bivariate_summary,
                  bivariate_node_report, cm_baseline, cm_bivariate):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Comparison table as CSV
    df = pd.DataFrame(comparison_table)
    df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)

    # Full raw summaries + node report as JSON (for completeness)
    full_results = {
        "baseline_summary": baseline_summary,
        "bivariate_summary": bivariate_summary,
        "bivariate_node_report": bivariate_node_report,
        "confusion_matrix_baseline": cm_baseline.tolist(),
        "confusion_matrix_bivariate": cm_bivariate.tolist(),
        "config": {
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "bivariate_max_depth": BIVARIATE_MAX_DEPTH,
            "bivariate_min_samples_split": BIVARIATE_MIN_SAMPLES_SPLIT,
            "bivariate_n_orientations": BIVARIATE_N_ORIENTATIONS,
            "bivariate_min_impurity_decrease": BIVARIATE_MIN_IMPURITY_DECREASE,
        },
    }
    with open(os.path.join(RESULTS_DIR, "full_results.json"), "w") as f:
        json.dump(_to_jsonable(full_results), f, indent=2)


# =========================================================================
# 6. Main experiment runner
# =========================================================================
def run_experiment():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, class_names = load_shared_data()

    baseline_model, baseline_summary, cm_baseline = run_baseline(
        X_train, y_train, X_test, y_test
    )
    bivariate_model, bivariate_summary, cm_bivariate = run_bivariate(
        X_train, y_train, X_test, y_test
    )

    bivariate_node_report = summarize_bivariate_node_types(bivariate_model, feature_names)

    comparison_table = build_comparison_table(baseline_summary, bivariate_summary)

    # --- Plots ---
    plot_accuracy_comparison(
        baseline_summary, bivariate_summary,
        os.path.join(PLOTS_DIR, "accuracy_comparison.png"),
    )
    plot_tree_size_comparison(
        baseline_summary, bivariate_summary,
        os.path.join(PLOTS_DIR, "tree_size_comparison.png"),
    )
    plot_confusion_matrices(
        cm_baseline, cm_bivariate, class_names,
        os.path.join(PLOTS_DIR, "confusion_matrices.png"),
    )

    # Decision boundary plot: use the FIRST bivariate node found (if any)
    boundary_plot_path = None
    if bivariate_node_report["num_bivariate_nodes"] > 0:
        example = bivariate_node_report["bivariate_node_details"][0]
        feat_j_name, feat_k_name = example["feature_pair"]
        feat_j = feature_names.index(feat_j_name)
        feat_k = feature_names.index(feat_k_name)
        boundary_plot_path = os.path.join(
            PLOTS_DIR, f"decision_boundary_{feat_j_name.replace(' ', '_')}_{feat_k_name.replace(' ', '_')}.png"
        )
        plot_bivariate_decision_boundary(
            bivariate_model, X_train, y_train, feature_names, class_names,
            feat_j, feat_k, boundary_plot_path,
        )

    # --- Save numeric results ---
    save_results(
        comparison_table, baseline_summary, bivariate_summary,
        bivariate_node_report, cm_baseline, cm_bivariate,
    )

    return {
        "baseline_summary": baseline_summary,
        "bivariate_summary": bivariate_summary,
        "comparison_table": comparison_table,
        "bivariate_node_report": bivariate_node_report,
        "cm_baseline": cm_baseline,
        "cm_bivariate": cm_bivariate,
        "boundary_plot_path": boundary_plot_path,
    }


def print_summary(results):
    print("=" * 78)
    print("EXPERIMENTAL EVALUATION -- Baseline vs. Simplified Bivariate Tree")
    print("(This evaluates OUR simplified implementation, NOT the KDD 2024")
    print(" paper's own algorithm/results. See limitations below.)")
    print("=" * 78)
    print(f"Dataset: Iris | test_size={TEST_SIZE} | random_seed={RANDOM_SEED}")
    print(f"Bivariate tree settings: max_depth={BIVARIATE_MAX_DEPTH}, "
          f"min_samples_split={BIVARIATE_MIN_SAMPLES_SPLIT}, "
          f"n_orientations={BIVARIATE_N_ORIENTATIONS}, "
          f"min_impurity_decrease={BIVARIATE_MIN_IMPURITY_DECREASE}")
    print("-" * 78)

    table = results["comparison_table"]
    print(f"{'Metric':<24}{'Baseline (univariate)':<24}{'Bivariate (ours)':<24}")
    for row in table:
        b_val = row["baseline_univariate_tree"]
        v_val = row["bivariate_tree"]
        if isinstance(b_val, float):
            b_val = f"{b_val:.4f}"
        if isinstance(v_val, float):
            v_val = f"{v_val:.4f}"
        print(f"{row['metric']:<24}{str(b_val):<24}{str(v_val):<24}")

    print("-" * 78)
    node_report = results["bivariate_node_report"]
    print(f"Bivariate tree node breakdown: "
          f"{node_report['num_bivariate_nodes']} bivariate node(s), "
          f"{node_report['num_univariate_nodes']} univariate fallback node(s)")
    for i, n in enumerate(node_report["bivariate_node_details"]):
        print(f"  Bivariate node {i+1}: features={n['feature_pair']}, "
              f"w1={n['w1']:.4f}, w2={n['w2']:.4f}, b={n['b']:.4f}, "
              f"gain={n['impurity_gain']:.4f}, n_samples={n['n_samples']}")

    print("-" * 78)
    print(f"Plots saved to:   {PLOTS_DIR}/")
    print(f"Results saved to: {RESULTS_DIR}/")
    if results["boundary_plot_path"]:
        print(f"Decision boundary plot: {results['boundary_plot_path']}")
    else:
        print("NOTE: No bivariate node was found in this run, so no decision "
              "boundary plot could be generated for a bivariate split.")

    print("=" * 78)
    print("LIMITATIONS OF THIS EXPERIMENT (see also bivariate_tree.py docstring):")
    print("- Small dataset (Iris, 150 samples, 4 features): results here are")
    print("  illustrative, not a claim of general superiority over CART.")
    print("- Our bivariate tree uses Gini-based greedy search with a FIXED,")
    print("  small orientation set, not the paper's exact/TAO-optimized split.")
    print("- No lambda/C regularization or global tree optimization (TAO) was")
    print("  implemented; node type (bivariate vs univariate) is chosen by a")
    print("  simple local Gini-gain comparison, not the paper's formal rule.")
    print("- These results do NOT reproduce the KDD 2024 paper's own reported")
    print("  numbers, which used different datasets, the full TAO algorithm,")
    print("  and exact/degenerate-case handling not implemented here.")
    print("=" * 78)


if __name__ == "__main__":
    results = run_experiment()
    print_summary(results)
