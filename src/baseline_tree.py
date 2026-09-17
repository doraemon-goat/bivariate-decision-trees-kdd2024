"""
baseline_tree.py

PURPOSE
-------
This file implements the BASELINE model for our project: a standard,
off-the-shelf UNIVARIATE decision tree (each split uses exactly one
feature), using scikit-learn's DecisionTreeClassifier.

This baseline exists so we can later compare it against our custom
BivariateDecisionTree (implemented separately in bivariate_tree.py)
on the same dataset and the same train/test split.

IMPORTANT: This file contains NO bivariate-tree logic. It only wraps
scikit-learn's tree and extracts simple statistics from it (accuracy,
depth, number of nodes, number of leaves). This keeps the "baseline"
completely separate from the custom algorithm we are studying.
"""

import time

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------
def load_iris_data(test_size=0.3, random_seed=42):
    """
    Load the Iris dataset and split it into train/test sets.

    We use all 4 original Iris features and all 3 classes, as required
    by the project brief. A fixed random_seed is used everywhere in
    this project so results are reproducible.

    Parameters
    ----------
    test_size : float
        Fraction of the data to hold out for testing.
    random_seed : int
        Seed used for the train/test split, so the SAME split can be
        reused later for the bivariate tree (fair comparison).

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names : list of str
    class_names : list of str
    """
    iris = load_iris()
    X = iris.data          # shape (150, 4): sepal length/width, petal length/width
    y = iris.target        # shape (150,): class labels 0, 1, 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,  # keep class proportions balanced in train/test
    )

    return X_train, X_test, y_train, y_test, list(iris.feature_names), list(iris.target_names)


# ---------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------
def train_baseline_tree(X_train, y_train, max_depth=None, random_seed=42):
    """
    Train a standard scikit-learn univariate decision tree.

    Parameters
    ----------
    X_train, y_train : training data
    max_depth : int or None
        Maximum depth allowed for the tree. None means nodes are
        expanded until all leaves are pure or contain too few samples
        (scikit-learn's default stopping behaviour).
    random_seed : int
        Random seed for reproducibility (sklearn's tree building has
        some randomness when choosing among equally good splits).

    Returns
    -------
    model : trained sklearn.tree.DecisionTreeClassifier
    training_time_seconds : float
    """
    model = DecisionTreeClassifier(
        criterion="gini",       # standard CART impurity measure
        max_depth=max_depth,
        random_state=random_seed,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time_seconds = time.time() - start_time

    return model, training_time_seconds


# ---------------------------------------------------------------------
# 3. Metric extraction helpers
# ---------------------------------------------------------------------
def get_accuracy(model, X, y):
    """
    Compute classification accuracy of the trained model on data (X, y).
    """
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)


def get_tree_depth(model):
    """
    Return the depth of the fitted tree.

    scikit-learn exposes this directly via model.get_depth().
    Depth = number of edges on the longest root-to-leaf path.
    """
    return model.get_depth()


def get_num_nodes(model):
    """
    Return the TOTAL number of nodes in the fitted tree
    (decision nodes + leaves combined).

    scikit-learn stores the whole tree structure in model.tree_,
    and model.tree_.node_count gives the total node count directly.
    """
    return model.tree_.node_count


def get_num_leaves(model):
    """
    Return the number of LEAF nodes in the fitted tree.

    scikit-learn exposes this via model.get_n_leaves().
    """
    return model.get_n_leaves()


def summarize_baseline_model(model, X_train, y_train, X_test, y_test):
    """
    Convenience function that collects all the statistics required
    by the project brief for the baseline model into one dictionary:
        - training accuracy
        - test accuracy
        - tree depth
        - number of nodes
        - number of leaves

    This dictionary format is designed to match the summary produced
    for the bivariate tree later, so the two models can be compared
    side by side in experiment.py.
    """
    summary = {
        "model_name": "Baseline Univariate Decision Tree (sklearn)",
        "train_accuracy": get_accuracy(model, X_train, y_train),
        "test_accuracy": get_accuracy(model, X_test, y_test),
        "depth": get_tree_depth(model),
        "num_nodes": get_num_nodes(model),
        "num_leaves": get_num_leaves(model),
    }
    return summary


# ---------------------------------------------------------------------
# 4. Simple manual check when running this file directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # This block lets us sanity-check baseline_tree.py in isolation,
    # without needing experiment.py or main.py yet.

    X_train, X_test, y_train, y_test, feature_names, class_names = load_iris_data()

    model, training_time = train_baseline_tree(X_train, y_train)

    summary = summarize_baseline_model(model, X_train, y_train, X_test, y_test)
    summary["training_time_seconds"] = training_time

    print("Baseline Decision Tree — Summary")
    print("---------------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")
