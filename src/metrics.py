"""
metrics.py

PURPOSE
-------
Shared evaluation utilities used IDENTICALLY for both the baseline
(univariate) tree and our custom BivariateDecisionTree, so that the
comparison in experiment.py is fair: both models are scored with
exactly the same functions.

This file contains no modeling logic of its own -- only measurement
and reporting helpers.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(y_true, y_pred, average="macro"):
    """
    Compute precision, recall and F1-score.

    We use macro-averaging (average="macro") because Iris has 3
    classes and macro-averaging treats all 3 classes equally,
    regardless of class size (Iris classes are balanced anyway, but
    macro-averaging is still the standard, defensible default for a
    small multiclass problem like this).

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_confusion_matrix(y_true, y_pred, n_classes=3):
    """
    Compute the confusion matrix as a numpy array of shape
    (n_classes, n_classes), rows = true class, columns = predicted
    class.
    """
    labels = list(range(n_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def build_comparison_table(baseline_summary, bivariate_summary):
    """
    Combine the two models' summary dictionaries into one comparison
    table (list of rows), so it can be printed and/or saved as a CSV.

    Both `baseline_summary` and `bivariate_summary` are expected to
    share the same set of metric keys (train_accuracy, test_accuracy,
    precision, recall, f1, training_time_seconds, depth, num_nodes,
    num_leaves), which experiment.py is responsible for guaranteeing.
    """
    metric_keys = [
        "train_accuracy",
        "test_accuracy",
        "precision",
        "recall",
        "f1",
        "training_time_seconds",
        "depth",
        "num_nodes",
        "num_leaves",
    ]

    rows = []
    for key in metric_keys:
        rows.append({
            "metric": key,
            "baseline_univariate_tree": baseline_summary.get(key),
            "bivariate_tree": bivariate_summary.get(key),
        })
    return rows
