"""
bivariate_tree.py

PURPOSE
-------
This file implements our custom BivariateDecisionTree classifier,
following the simplified algorithm agreed in Step 4 of the project
plan, which is itself a simplified version of the bivariate-tree idea
from:

    Kairgeldin & Carreira-Perpinan, "Bivariate Decision Trees:
    Smaller, Interpretable, More Accurate", KDD 2024.

CORE IDEA (from the paper)
---------------------------
A univariate tree node asks a question about ONE feature. A bivariate
tree node instead asks a question about a LINEAR COMBINATION of TWO
features:

    score = w1 * x[feat_j] + w2 * x[feat_k] + b
    if score < 0: go LEFT
    else:         go RIGHT

This is exactly the routing rule defined in the paper (Section 4).
No decision node in this implementation is ever allowed to use more
than two features.

WHAT WE REPRODUCE FROM THE PAPER (see Step 2 of the plan)
-----------------------------------------------------------
- The bivariate routing rule itself (w1*x1 + w2*x2 + b < 0).
- The "at most 2 features per node" constraint.
- The paper's APPROXIMATE split-search idea: instead of solving the
  hard optimal-linear-split problem exactly, we pick a small, FIXED
  set of line orientations (angles between 0 and 180 degrees),
  project the two candidate features onto each orientation, and
  search for the best threshold along that 1-D projection. This
  mirrors Fig. 2 / Fig. 3 (pseudocode) of the paper.
- Growing the tree via greedy recursive partitioning (i.e. the
  general spirit of what the paper calls "bivariate CART"), rather
  than the slower, globally-optimized "bivariate TAO" algorithm.

WHAT WE SIMPLIFY / LEAVE OUT (see Step 3 of the plan) -- OUT OF SCOPE
-----------------------------------------------------------------------
- We do NOT implement bivariate TAO (alternating optimization over a
  fixed tree structure, reduced-set bookkeeping, reverse-BFS updates).
- We do NOT implement the paper's lambda/C regularization objective
  (Equation 1-2 in the paper) that lets a node automatically choose
  between zero/one/two features based on a global penalty. Instead,
  we use a simple, explicit, LOCAL fallback rule (see
  `_choose_best_split`) to decide bivariate vs univariate per node.
- We use GINI IMPURITY REDUCTION to score candidate splits, not the
  paper's exact 0/1-loss objective (Eq. 3). Gini is the standard
  CART-family criterion and is much cheaper to compute at every
  candidate threshold.
- We do not implement a regularization path over lambda, nor
  cost-complexity pruning. We rely on simple stopping conditions
  instead (max depth, min samples, purity, minimum impurity decrease).
- We search a small, explicit set of feature pairs (all pairs, since
  Iris only has 4 features -> 6 pairs total) rather than a general
  large-D scheme.

These choices are simplifications/engineering decisions for a
one-week student project, not claims about what the original paper
does at full scale.
"""

import itertools
import time

import numpy as np


# =======================================================================
# 1. Node representations
# =======================================================================
class DecisionNode:
    """
    An INTERNAL node of the tree. Represents ONE routing rule.

    A decision node is either:
      - "bivariate": uses TWO features via
            score = w1*x[feat_j] + w2*x[feat_k] + b
        Here feat_j and feat_k are both real feature indices, and
        both w1 and w2 are non-zero.

      - "univariate": uses ONE feature only (fallback case, see
        `_choose_best_split`). To keep the SAME routing formula for
        every decision node (bivariate or univariate), we represent a
        univariate split as a "degenerate" bivariate split where the
        second feature's weight is exactly 0:
            score = w1*x[feat_j] + 0*x[feat_k] + b
        feat_k is arbitrary in this case (we still record a real
        feature index for bookkeeping, but its weight is 0, so it has
        NO effect on the decision). This still respects the paper's
        "at most 2 features per node" constraint, since it is
        equivalent to using only 1 feature.

    In both cases, the node NEVER uses more than 2 features, and the
    exact same score/threshold rule (score < 0 -> left) is used for
    routing at prediction time.
    """

    def __init__(self, feat_j, feat_k, w1, w2, b, node_type, impurity_gain, n_samples):
        self.feat_j = feat_j
        self.feat_k = feat_k
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.node_type = node_type          # "bivariate" or "univariate"
        self.impurity_gain = impurity_gain  # Gini impurity reduction achieved by this split
        self.n_samples = n_samples          # how many training samples reached this node

        self.left = None   # child DecisionNode or LeafNode
        self.right = None  # child DecisionNode or LeafNode

    def route(self, x):
        """
        Decide whether sample x goes LEFT or RIGHT, using the paper's
        bivariate routing rule:
            score = w1*x[feat_j] + w2*x[feat_k] + b
            left if score < 0, else right
        """
        score = self.w1 * x[self.feat_j] + self.w2 * x[self.feat_k] + self.b
        return "left" if score < 0 else "right"

    def is_leaf(self):
        return False


class LeafNode:
    """
    A LEAF node of the tree. Stores a single constant class prediction,
    equal to the majority class of the training samples that reached
    this leaf (same idea as a standard decision tree leaf, and matches
    the paper's leaf rule: "majority class of the samples in the
    reduced set").
    """

    def __init__(self, predicted_class, n_samples, class_counts):
        self.predicted_class = predicted_class
        self.n_samples = n_samples
        self.class_counts = class_counts  # dict: class_label -> count, for inspection

    def is_leaf(self):
        return True


# =======================================================================
# 2. Impurity helper (Gini index)
# =======================================================================
def gini_impurity(y):
    """
    Compute the Gini impurity of a set of labels y.

    Gini impurity = 1 - sum_over_classes( p_c^2 )
    where p_c is the fraction of samples belonging to class c.

    Impurity is 0 when all samples belong to a single class (pure),
    and increases as classes become more mixed. This is the SAME
    impurity measure used internally by standard CART trees
    (including scikit-learn's DecisionTreeClassifier), so it is a
    fair, well-established choice for scoring candidate splits.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)


def weighted_child_impurity(y_left, y_right):
    """
    Compute the impurity of a split as the SAMPLE-SIZE-WEIGHTED
    average of the two children's Gini impurities. This is the
    standard way CART evaluates a candidate split: a split that
    produces two large, pure children is better than one that
    produces a tiny pure child and a huge mixed child.
    """
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    weighted = (n_left / n_total) * gini_impurity(y_left) + \
               (n_right / n_total) * gini_impurity(y_right)
    return weighted


def impurity_reduction(y_parent, y_left, y_right):
    """
    Impurity reduction ("information gain" in the Gini sense) achieved
    by splitting y_parent into y_left and y_right:

        gain = impurity(parent) - weighted_impurity(children)

    Higher gain = better split. This is the score we use to compare
    ALL candidate splits (bivariate AND univariate) at a node.
    """
    return gini_impurity(y_parent) - weighted_child_impurity(y_left, y_right)


# =======================================================================
# 3. Split search: BIVARIATE (paper-inspired, simplified)
# =======================================================================
def _best_threshold_for_projection(proj, y):
    """
    Given a 1-D array `proj` (the result of projecting samples onto
    some direction) and their labels y, find the threshold `b` that
    maximizes the Gini impurity reduction of the split:

        left  if (proj + b) < 0   <=>   proj < -b
        right otherwise

    This is equivalent to: sort the projected values, and try
    thresholds at the midpoints between consecutive DISTINCT sorted
    values (the standard CART approach to finding a 1-D threshold).

    Returns
    -------
    best_b : float or None (None if no valid split found, e.g. all
             projected values are identical)
    best_gain : float (impurity reduction of the best split found;
                -inf if no valid split found)
    """
    order = np.argsort(proj)
    proj_sorted = proj[order]
    y_sorted = y[order]

    best_b = None
    best_gain = -np.inf

    n = len(proj_sorted)
    for i in range(1, n):
        # Only consider a threshold between two DIFFERENT projected
        # values -- placing a threshold between two identical values
        # would not actually separate them.
        if proj_sorted[i] == proj_sorted[i - 1]:
            continue

        midpoint = (proj_sorted[i] + proj_sorted[i - 1]) / 2.0
        # We define: left if (proj + b) < 0  =>  b = -midpoint
        candidate_b = -midpoint

        y_left = y_sorted[:i]
        y_right = y_sorted[i:]

        gain = impurity_reduction(y_sorted, y_left, y_right)
        if gain > best_gain:
            best_gain = gain
            best_b = candidate_b

    return best_b, best_gain


def search_bivariate_split(X, y, feature_pairs, n_orientations):
    """
    THIS FUNCTION IS THE HEART OF THE BIVARIATE ALGORITHM.

    For every candidate pair of features (feat_j, feat_k), and for a
    small, FIXED set of `n_orientations` line orientations spanning
    0 to 180 degrees, we:

      1. Project the two chosen features onto that orientation:
             proj = w1*x[feat_j] + w2*x[feat_k]
         where (w1, w2) = (cos(angle), sin(angle)).
      2. Search for the best threshold b along that 1-D projection
         (via `_best_threshold_for_projection`).
      3. Score the resulting split using Gini impurity reduction.

    We keep track of the single best (feat_j, feat_k, w1, w2, b)
    combination found across ALL pairs and ALL orientations.

    This directly mirrors the paper's approximate bivariate split
    procedure (Fig. 2 / Fig. 3 pseudocode): a small fixed set of
    orientations is used instead of solving the (expensive, exact)
    optimal linear separator problem.

    SIMPLIFICATION: the paper uses the exact 0/1 loss as the
    objective (Eq. 3); we use Gini impurity reduction, which is the
    standard CART-family alternative and is documented as a
    simplification in Step 3 of the project plan.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    feature_pairs : list of (j, k) tuples
        Candidate feature-index pairs to try.
    n_orientations : int
        Number of angles H sampled uniformly in [0, 180) degrees.

    Returns
    -------
    best_split : dict or None
        Dictionary with keys: feat_j, feat_k, w1, w2, b, gain.
        None if no valid split was found at all (e.g. too few samples).
    """
    angles = np.linspace(0, np.pi, n_orientations, endpoint=False)

    best_split = None
    best_gain = -np.inf

    for (feat_j, feat_k) in feature_pairs:
        x_j = X[:, feat_j]
        x_k = X[:, feat_k]

        for angle in angles:
            w1 = np.cos(angle)
            w2 = np.sin(angle)

            proj = w1 * x_j + w2 * x_k

            b, gain = _best_threshold_for_projection(proj, y)
            if b is None:
                continue

            if gain > best_gain:
                best_gain = gain
                best_split = {
                    "feat_j": feat_j,
                    "feat_k": feat_k,
                    "w1": w1,
                    "w2": w2,
                    "b": b,
                    "gain": gain,
                }

    return best_split


# =======================================================================
# 4. Split search: UNIVARIATE FALLBACK
# =======================================================================
def search_univariate_split(X, y, n_features):
    """
    Standard, single-feature CART-style split search: for each
    feature individually, find the best threshold. This is used ONLY
    as an explicit FALLBACK when no bivariate split improves on it
    (see `_choose_best_split`).

    We represent a univariate split using the SAME (feat_j, feat_k,
    w1, w2, b) format as a bivariate split, but with w2 forced to 0.0,
    so that:
        score = w1*x[feat_j] + 0*x[feat_k] + b
    depends on feat_j only. This lets both bivariate and univariate
    nodes share exactly one routing implementation (`DecisionNode.route`).

    Returns
    -------
    best_split : dict or None (same keys as search_bivariate_split)
    """
    best_split = None
    best_gain = -np.inf

    for feat_j in range(n_features):
        x_j = X[:, feat_j]

        # Projecting onto a single feature is the same as an
        # orientation of angle 0 degrees: w1=1, w2=0.
        b, gain = _best_threshold_for_projection(x_j, y)
        if b is None:
            continue

        if gain > best_gain:
            best_gain = gain
            best_split = {
                "feat_j": feat_j,
                "feat_k": (feat_j + 1) % n_features,  # placeholder index, weight is 0
                "w1": 1.0,
                "w2": 0.0,
                "b": b,
                "gain": gain,
            }

    return best_split


# =======================================================================
# 5. Choosing between bivariate and univariate at a node
# =======================================================================
def _choose_best_split(X, y, feature_pairs, n_orientations, n_features):
    """
    ENGINEERING-LEVEL DECISION RULE (explicitly NOT from the paper's
    TAO node-type selection formula, which uses lambda/C
    regularization -- see Step 3, simplification #2).

    We compute BOTH the best bivariate split and the best univariate
    split at this node, and simply pick whichever has the HIGHER
    Gini impurity reduction. Ties are broken in favor of the simpler
    (univariate) split, to prefer interpretability when there is no
    real accuracy benefit to using two features.

    This is a transparent, easy-to-explain substitute for the paper's
    formal zero/uni/bivariate selection rule, and every choice made
    here is tagged with node_type = "bivariate" or "univariate" so it
    can be audited afterward.

    Returns
    -------
    chosen_split : dict (with an added "node_type" key), or None if
                   neither search found a usable split.
    """
    biv_split = search_bivariate_split(X, y, feature_pairs, n_orientations)
    uni_split = search_univariate_split(X, y, n_features)

    biv_gain = biv_split["gain"] if biv_split is not None else -np.inf
    uni_gain = uni_split["gain"] if uni_split is not None else -np.inf

    if biv_split is None and uni_split is None:
        return None

    if biv_gain > uni_gain:
        chosen = dict(biv_split)
        chosen["node_type"] = "bivariate"
    else:
        chosen = dict(uni_split)
        chosen["node_type"] = "univariate"

    return chosen


# =======================================================================
# 6. The BivariateDecisionTree classifier
# =======================================================================
class BivariateDecisionTree:
    """
    Custom decision tree classifier where each internal node makes a
    decision using AT MOST TWO features, via the linear routing rule:

        score = w1*x[feat_j] + w2*x[feat_k] + b
        left if score < 0, else right

    This is a SIMPLIFIED, student-project implementation of the
    bivariate tree idea from Kairgeldin & Carreira-Perpinan (KDD 2024).
    See the module docstring at the top of this file for exactly what
    is reproduced from the paper and what is simplified.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree (stopping condition).
    min_samples_split : int
        A node will not be split further if it has fewer than this
        many samples (stopping condition).
    min_impurity_decrease : float
        A split must achieve at least this much Gini impurity
        reduction to be accepted; otherwise the node becomes a leaf
        (stopping condition -- avoids near-useless splits).
    n_orientations : int
        Number of candidate line orientations (H in the paper) used
        by the bivariate split search, sampled uniformly in [0, 180)
        degrees. Kept small deliberately for speed (see Step 3).
    random_seed : int
        Stored for reproducibility / consistency with the rest of the
        project, even though this implementation is deterministic
        given the same data (no randomness is used internally).
    """

    def __init__(self, max_depth=4, min_samples_split=5,
                 min_impurity_decrease=1e-7, n_orientations=18,
                 random_seed=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_orientations = n_orientations
        self.random_seed = random_seed

        self.root = None
        self.n_features_ = None

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------
    def fit(self, X, y):
        """
        Grow the tree recursively, starting from the root, using
        greedy recursive partitioning (this is the general strategy
        the paper calls "bivariate CART" -- see Step 2).

        At EVERY node we:
          1. Check stopping conditions (max depth / too few samples /
             pure node). If any apply -> make a leaf.
          2. Otherwise, search for the best split (bivariate first,
             univariate as fallback -- see `_choose_best_split`).
          3. If the best split found does not improve impurity enough
             -> make a leaf anyway (min_impurity_decrease condition).
          4. Otherwise, split the data and recurse on each child.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.n_features_ = X.shape[1]

        # Candidate feature pairs: ALL pairs, since with a small
        # number of features (e.g. Iris has 4) this is cheap and
        # complete. For larger D one would need to subsample pairs
        # (explicitly noted as a limitation in Step 3).
        feature_pairs = list(itertools.combinations(range(self.n_features_), 2))
        if len(feature_pairs) == 0:
            raise ValueError("Need at least 2 features to build a bivariate tree.")
        self.feature_pairs_ = feature_pairs

        self.root = self._build_node(X, y, depth=0)
        return self

    def _build_node(self, X, y, depth):
        """
        Recursively build one node of the tree (and, if it is a
        decision node, its two children).
        """
        n_samples = len(y)

        # --- Stopping condition 1: max depth reached ---
        if depth >= self.max_depth:
            return self._make_leaf(y)

        # --- Stopping condition 2: too few samples to split ---
        if n_samples < self.min_samples_split:
            return self._make_leaf(y)

        # --- Stopping condition 3: node is already pure ---
        if gini_impurity(y) == 0.0:
            return self._make_leaf(y)

        # --- Search for the best split at this node ---
        split = _choose_best_split(
            X, y, self.feature_pairs_, self.n_orientations, self.n_features_
        )

        # --- Stopping condition 4: no usable split found, or the
        # best split found does not improve impurity enough ---
        if split is None or split["gain"] < self.min_impurity_decrease:
            return self._make_leaf(y)

        # --- Build this decision node ---
        node = DecisionNode(
            feat_j=split["feat_j"],
            feat_k=split["feat_k"],
            w1=split["w1"],
            w2=split["w2"],
            b=split["b"],
            node_type=split["node_type"],
            impurity_gain=split["gain"],
            n_samples=n_samples,
        )

        # --- Partition the data using this node's routing rule ---
        left_mask = np.array([node.route(x) == "left" for x in X])
        right_mask = ~left_mask

        # Safety check: a valid split must send at least 1 sample to
        # each side (this should already be guaranteed by the search,
        # but we check defensively).
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._make_leaf(y)

        # --- Recurse on children ---
        node.left = self._build_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_node(X[right_mask], y[right_mask], depth + 1)

        return node

    def _make_leaf(self, y):
        """
        Build a leaf node predicting the majority class among the
        samples that reached it (same rule as the paper's leaf
        solution: majority class of the reduced set).
        """
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        class_counts = dict(zip(classes.tolist(), counts.tolist()))
        return LeafNode(
            predicted_class=majority_class,
            n_samples=len(y),
            class_counts=class_counts,
        )

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict(self, X):
        """
        Predict class labels for each sample in X, by walking each
        sample from the root to a leaf using each decision node's
        `route()` method (the w1*x1+w2*x2+b<0 rule).
        """
        if self.root is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=float)
        predictions = np.array([self._predict_one(x, self.root) for x in X])
        return predictions

    def _predict_one(self, x, node):
        """
        Recursively route a single sample x from `node` down to a
        leaf, and return that leaf's predicted class.
        """
        if node.is_leaf():
            return node.predicted_class

        direction = node.route(x)
        child = node.left if direction == "left" else node.right
        return self._predict_one(x, child)

    # -------------------------------------------------------------
    # Structural helper methods (for evaluation + viva inspection)
    # -------------------------------------------------------------
    def get_depth(self):
        """Return the depth of the tree (0 if root is a single leaf)."""
        return self._depth_of(self.root)

    def _depth_of(self, node):
        if node.is_leaf():
            return 0
        return 1 + max(self._depth_of(node.left), self._depth_of(node.right))

    def get_num_nodes(self):
        """Return the TOTAL number of nodes (decision nodes + leaves)."""
        return self._count_nodes(self.root)

    def _count_nodes(self, node):
        if node.is_leaf():
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_num_leaves(self):
        """Return the number of LEAF nodes in the tree."""
        return self._count_leaves(self.root)

    def _count_leaves(self, node):
        if node.is_leaf():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def get_decision_nodes(self):
        """
        Return a flat list of dictionaries, one per DECISION node in
        the tree (leaves are excluded), each describing:
            feat_j, feat_k, w1, w2, b, node_type, impurity_gain, n_samples

        This is the key inspection method for the viva: it makes it
        explicit and checkable that the tree contains genuine
        two-feature ("bivariate") routing rules, not just single-
        feature splits relabeled as bivariate.
        """
        nodes_info = []
        self._collect_decision_nodes(self.root, nodes_info)
        return nodes_info

    def _collect_decision_nodes(self, node, nodes_info):
        if node.is_leaf():
            return
        nodes_info.append({
            "node_type": node.node_type,
            "feat_j": node.feat_j,
            "feat_k": node.feat_k if node.node_type == "bivariate" else None,
            "w1": node.w1,
            "w2": node.w2 if node.node_type == "bivariate" else None,
            "b": node.b,
            "impurity_gain": node.impurity_gain,
            "n_samples": node.n_samples,
        })
        self._collect_decision_nodes(node.left, nodes_info)
        self._collect_decision_nodes(node.right, nodes_info)

    def print_tree_structure(self, feature_names=None):
        """
        Human-readable printout of every decision node's routing rule,
        for viva inspection. If feature_names is given, feature
        indices are replaced with readable names.
        """
        def name(idx):
            if feature_names is not None and idx is not None:
                return feature_names[idx]
            return f"x{idx}"

        for info in self.get_decision_nodes():
            if info["node_type"] == "bivariate":
                rule = (f"{info['w1']:.3f}*{name(info['feat_j'])} + "
                        f"{info['w2']:.3f}*{name(info['feat_k'])} + "
                        f"{info['b']:.3f} < 0")
            else:
                rule = (f"{info['w1']:.3f}*{name(info['feat_j'])} + "
                        f"{info['b']:.3f} < 0   (univariate)")

            print(f"[{info['node_type'].upper():10s}] n_samples={info['n_samples']:3d} "
                  f"gain={info['impurity_gain']:.4f}  rule: {rule}")


# =======================================================================
# 7. Simple manual sanity check when running this file directly
# =======================================================================
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # --- Load Iris data (same style/seed as baseline_tree.py) ---
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- Train the bivariate tree ---
    model = BivariateDecisionTree(
        max_depth=4,
        min_samples_split=5,
        min_impurity_decrease=1e-7,
        n_orientations=18,
        random_seed=42,
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # --- Verify prediction works ---
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("=" * 70)
    print("BivariateDecisionTree — Sanity Check on Iris")
    print("=" * 70)
    print(f"Training time (s): {train_time:.4f}")
    print(f"Train accuracy:    {train_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
    print(f"Tree depth:        {model.get_depth()}")
    print(f"Number of nodes:   {model.get_num_nodes()}")
    print(f"Number of leaves:  {model.get_num_leaves()}")
    print()
    print("Decision node routing rules (proof of bivariate splits):")
    print("-" * 70)
    model.print_tree_structure(feature_names=feature_names)

    # --- Explicitly print at least one bivariate node's raw params ---
    print()
    print("Explicit raw parameters of decision nodes:")
    print("-" * 70)
    decision_nodes = model.get_decision_nodes()
    biv_nodes = [n for n in decision_nodes if n["node_type"] == "bivariate"]
    if biv_nodes:
        example = biv_nodes[0]
        print("Example BIVARIATE node found:")
        print(f"  feat_j = {example['feat_j']} ({feature_names[example['feat_j']]})")
        print(f"  feat_k = {example['feat_k']} ({feature_names[example['feat_k']]})")
        print(f"  w1 = {example['w1']:.4f}")
        print(f"  w2 = {example['w2']:.4f}")
        print(f"  b  = {example['b']:.4f}")
    else:
        print("No bivariate node was selected by the tree "
              "(univariate fallback won at every node for this run).")
