# Implementation Details

This document explains the actual code in `src/bivariate_tree.py` (and its supporting modules). Every component described here exists in the repository exactly as documented — nothing is aspirational.

---

## 1. Overview

The core deliverable is the `BivariateDecisionTree` class in `src/bivariate_tree.py`, written from scratch using only NumPy. **scikit-learn is never used to implement the bivariate algorithm** — it appears only for the baseline model, data loading, and standard metrics.

The module is organized in seven sections:

| Section | Contents |
|---|---|
| 1 | Node representations — `DecisionNode`, `LeafNode` |
| 2 | Impurity helpers — `gini_impurity`, `weighted_child_impurity`, `impurity_reduction` |
| 3 | Bivariate split search — `_best_threshold_for_projection`, `search_bivariate_split` |
| 4 | Univariate fallback — `search_univariate_split` |
| 5 | Node-type selection — `_choose_best_split` |
| 6 | The classifier — `BivariateDecisionTree` |
| 7 | Standalone sanity check (`__main__` block) |

---

## 2. `DecisionNode`

An internal node representing **one routing rule**. It stores:

```python
feat_j, feat_k, w1, w2, b, node_type, impurity_gain, n_samples
```

Its `route(x)` method implements the paper's rule directly:

```python
score = self.w1 * x[self.feat_j] + self.w2 * x[self.feat_k] + self.b
return "left" if score < 0 else "right"
```

**Design decision — one rule form for both node types.** A univariate node is stored as a *degenerate* bivariate node with `w2 = 0.0`:

```
score = w1·x[feat_j] + 0·x[feat_k] + b
```

Since the second weight is exactly zero, `feat_k` has no effect on the decision — the node genuinely uses only one feature. This lets both node types share a single routing implementation, and it still respects the "at most two features per node" constraint. The `node_type` attribute (`"bivariate"` or `"univariate"`) records which case applies, so the tree's composition can always be audited.

`is_leaf()` returns `False`.

---

## 3. `LeafNode`

A terminal node storing a constant prediction:

```python
predicted_class, n_samples, class_counts
```

The prediction is the **majority class** among the training samples that reached the leaf — the same rule used by standard decision trees and by the paper's leaf solution. `class_counts` is retained purely for inspection. `is_leaf()` returns `True`.

---

## 4. Gini Impurity

Three functions in Section 2 implement split scoring.

**`gini_impurity(y)`** — measures how mixed a set of labels is:

```
Gini(y) = 1 − Σ_c p_c²
```

where `p_c` is the fraction of samples of class `c`. It equals 0 for a pure node and increases with mixing. Returns `0.0` for an empty array.

**`weighted_child_impurity(y_left, y_right)`** — the sample-size-weighted average of the two children's impurities:

```
(n_left/n)·Gini(y_left) + (n_right/n)·Gini(y_right)
```

Weighting matters: a split producing two large, pure children is better than one producing a tiny pure child and a large mixed one.

**`impurity_reduction(y_parent, y_left, y_right)`** — the score used to compare *all* candidate splits:

```
gain = Gini(parent) − weighted_impurity(children)
```

Higher is better. This is the standard CART criterion, and it is a **documented simplification** of the paper's exact 0/1-loss objective.

---

## 5. Threshold Search — `_best_threshold_for_projection(proj, y)`

Given a 1-D array of projected values and their labels, this finds the threshold `b` maximizing Gini gain for the split

```
left if (proj + b) < 0   ⟺   proj < −b
```

The procedure:

1. Sort samples by their projected value.
2. Scan adjacent pairs; **skip any pair with identical values**, since a threshold between two equal values separates nothing.
3. For each valid boundary, take the midpoint and set `candidate_b = −midpoint`.
4. Score the resulting split with `impurity_reduction` and keep the best.

Returns `(best_b, best_gain)`, or `(None, −inf)` when no valid split exists — e.g. when all projected values are identical.

---

## 6. Bivariate Split Search — `search_bivariate_split(X, y, feature_pairs, n_orientations)`

**This is the heart of the algorithm.** Its structure is a triple loop:

```
for each feature pair (j, k):
    for each orientation θ in linspace(0, π, n_orientations, endpoint=False):
        w1, w2 = cos(θ), sin(θ)
        proj   = w1·X[:, j] + w2·X[:, k]        # projection
        b, gain = _best_threshold_for_projection(proj, y)   # threshold search
        keep (j, k, w1, w2, b) if gain is the best so far
```

**Orientation search.** Angles are sampled uniformly over `[0, π)` — i.e. 0° to 180°, excluding the endpoint. Angles beyond 180° are redundant, since they describe the same line with left and right swapped. Using the parameterization `(w1, w2) = (cos θ, sin θ)` guarantees a unit weight vector, so the geometry is fully described by the angle.

**Projection.** Computing `w1·x_j + w2·x_k` maps each 2-D point onto a 1-D line at angle θ. This is the key trick: it reduces the hard problem of finding an optimal 2-D linear separator to the easy, well-understood problem of finding a 1-D threshold.

**Result.** A dictionary with `feat_j, feat_k, w1, w2, b, gain`, or `None` if no valid split was found anywhere.

This mirrors the paper's approximate split procedure. Two documented simplifications apply: the objective is Gini rather than exact 0/1 loss, and `n_orientations` is smaller (18) than values typically used in the paper.

---

## 7. Univariate Fallback — `search_univariate_split(X, y, n_features)`

A standard single-feature CART search: for each feature in turn, call `_best_threshold_for_projection` on that raw column (equivalent to an orientation of 0°, i.e. `w1=1, w2=0`) and keep the best.

The result is returned in the **same dictionary format** as a bivariate split, with `w1 = 1.0`, `w2 = 0.0`, and `feat_k` set to a placeholder index whose weight is zero. This uniformity is what lets `DecisionNode.route` handle both node types with one formula.

---

## 8. Node-Type Selection — `_choose_best_split(...)`

At every node, **both** searches run. The function then picks whichever split has the higher Gini gain, with **ties broken in favor of the univariate split** (preferring simplicity when two features buy nothing). The chosen dictionary gains a `node_type` key of `"bivariate"` or `"univariate"`.

**This is an engineering-level decision rule, explicitly not from the paper.** The paper selects node type through a globally optimized λ/C-regularized objective. Our rule is purely *local* — it looks only at Gini gain at the current node — and is a transparent, easy-to-explain substitute chosen to keep the project tractable. It carries no optimality guarantee for the tree as a whole.

---

## 9. Recursive Tree Construction

**`fit(X, y)`** validates inputs (2-D `X`, matching lengths, at least two features), records `n_features_`, enumerates **all feature pairs** via `itertools.combinations` (6 pairs for Iris's 4 features), then calls `_build_node(X, y, depth=0)`.

**`_build_node(X, y, depth)`** applies four stopping criteria in order, returning a leaf if any fires:

1. `depth >= max_depth`
2. `n_samples < min_samples_split`
3. the node is already pure (`gini_impurity(y) == 0`)
4. no usable split exists, or the best gain is below `min_impurity_decrease`

Otherwise it constructs a `DecisionNode`, partitions the samples by calling `node.route(x)` on each one — so the *same* rule used at prediction time defines the training partition — and recurses on both sides with `depth + 1`. A defensive check converts the node to a leaf if either side would be empty.

**`_make_leaf(y)`** builds a `LeafNode` with the majority class and per-class counts.

---

## 10. Prediction

**`predict(X)`** raises `RuntimeError` if the model is unfitted, then routes each sample independently via `_predict_one`.

**`_predict_one(x, node)`** recurses: if the node is a leaf, return its `predicted_class`; otherwise call `node.route(x)` and descend into the corresponding child. Prediction uses exactly the same routing formula as training — there is no separate inference path.

---

## 11. Inspection Helpers

| Method | Purpose |
|---|---|
| `get_depth()` | Longest root-to-leaf path (0 if the root is a leaf) |
| `get_num_nodes()` | Total nodes — decision nodes plus leaves |
| `get_num_leaves()` | Leaf count |
| `get_decision_nodes()` | Flat list of dicts describing every decision node |
| `print_tree_structure(feature_names)` | Human-readable printout of each routing rule |

`get_decision_nodes()` is the key auditing method: it exposes each node's `node_type`, feature indices, weights, bias, gain, and sample count. For univariate nodes it reports `feat_k` and `w2` as `None`, making the distinction unambiguous. This is what allows the claim "the tree genuinely uses two-feature rules" to be *verified* rather than merely asserted — and it is what `experiment.py` and `cross_validation.py` use to count bivariate versus univariate nodes.

---

## 12. Supporting Modules

- **`src/baseline_tree.py`** — wraps `sklearn.tree.DecisionTreeClassifier` (Gini, `random_state=42`), loads/splits Iris, and extracts depth, node count and leaf count via `get_depth()`, `tree_.node_count` and `get_n_leaves()`. Contains no bivariate logic.
- **`src/metrics.py`** — shared macro-averaged precision/recall/F1, confusion matrices, and the comparison-table builder. Both models are scored by these *same* functions, which is what makes the comparison fair.
- **`src/experiment.py`** — orchestrates the single 70/30 experiment, generates all four plots, and writes `results/`. Contains no modeling logic.
- **`src/cross_validation.py`** — runs 5-fold `StratifiedKFold` with identical hyperparameters and no per-model tuning.

---

## 13. Explicitly Out of Scope

The following exist in the paper but are **not implemented** here:

- Bivariate TAO (alternating optimization over a fixed tree structure)
- The λ/C regularization objective and its regularization path
- Exact 0/1-loss node solutions and the zero-feature node case
- Cost-complexity pruning
- Vectorized/GPU projection
- Feature-pair subsampling for high-dimensional data

See the README's comparison table and `docs/paper-summary.md` for the full picture.
