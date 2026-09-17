# Experiments

This document records the experiments actually performed in this project and the exact numerical results obtained. **No results here are invented or estimated** — all figures come from the stored output files in `results/`.

---

## 1. Dataset

- **Iris**, loaded via `sklearn.datasets.load_iris` (a reference copy is included at `data/iris.csv`)
- **150 samples**, **4 features** (sepal length, sepal width, petal length, petal width), **3 classes** (setosa, versicolor, virginica)
- No feature scaling or other preprocessing was applied — the four original features were used directly

## 2. Models Compared

| | Model | Source |
|---|---|---|
| **A** | Standard univariate decision tree | `sklearn.tree.DecisionTreeClassifier`, Gini criterion, `max_depth=None`, `random_state=42` |
| **B** | Simplified bivariate decision tree | Custom `BivariateDecisionTree` in `src/bivariate_tree.py` |

Both models were evaluated using the **same data splits** and the **same scoring functions** (`src/metrics.py`), which is what makes the comparison fair.

## 3. Hyperparameters

The bivariate tree used these settings in **both** experiments. They were fixed in advance and **not tuned**, and no separate tuning was performed for either model:

```
max_depth             = 4
min_samples_split     = 5
n_orientations        = 18
min_impurity_decrease = 1e-7
random_state          = 42
```

Precision, recall and F1 are **macro-averaged** throughout, treating all three classes equally.

---

## 4. Experiment 1 — Single 70/30 Train/Test Split

**Protocol:** stratified split, `test_size=0.3`, `random_state=42` → 105 training samples, 45 test samples.
**Run with:** `python main.py`

### Results

| Metric | Baseline (univariate) | Bivariate (ours) |
|---|---|---|
| Train accuracy | 1.0000 | 0.9905 |
| Test accuracy | 0.9333 | 0.9556 |
| Precision (macro) | 0.9444 | 0.9556 |
| Recall (macro) | 0.9333 | 0.9556 |
| F1 (macro) | 0.9327 | 0.9556 |
| Training time (s) | 0.0014 | 0.8427 |
| Depth | 5 | 3 |
| Number of nodes | 15 | 7 |
| Number of leaves | 8 | 4 |

### Node composition of the bivariate tree

- **1 bivariate node**, **2 univariate fallback nodes**
- The bivariate node used the feature pair **(sepal length, petal width)**:

```
w1 = 0.1736
w2 = 0.9848
b  = -2.8116
Gini gain    = 0.4459
samples here = 70
```

i.e. the learned routing rule was
`0.1736 · sepal_length + 0.9848 · petal_width − 2.8116 < 0 → left`.

This is direct evidence that the tree performs a genuine two-feature linear decision, not a relabeled univariate split.

### Confusion matrices (test set)

**Baseline:** all 15 setosa correct; 12 of 15 versicolor correct with 3 misclassified as virginica; all 15 virginica correct.

**Bivariate:** all 15 setosa correct; 14 of 15 versicolor correct with 1 misclassified as virginica; 14 of 15 virginica correct with 1 misclassified as versicolor.

### Output files

- `results/comparison_table.csv`
- `results/full_results.json`

---

## 5. Experiment 2 — 5-Fold Stratified Cross-Validation

**Protocol:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the full 150-sample dataset. Identical hyperparameters to Experiment 1; no re-tuning.
**Run with:** `python -m src.cross_validation`

**Motivation:** Experiment 1 relies on a single 45-sample test set, which can be unrepresentative by chance. Cross-validation averages over five disjoint test partitions and reports variability, giving a more robust estimate.

### Results

| Metric | Baseline (univariate) | Bivariate (ours) |
|---|---|---|
| Accuracy (mean ± std) | 0.9533 ± 0.0380 | 0.9400 ± 0.0548 |
| Precision macro (mean ± std) | 0.9572 ± 0.0365 | 0.9434 ± 0.0523 |
| Recall macro (mean ± std) | 0.9533 ± 0.0380 | 0.9400 ± 0.0548 |
| F1 macro (mean ± std) | 0.9531 ± 0.0382 | 0.9396 ± 0.0552 |
| Avg. training time (s) | 0.0011 | 0.8749 |
| Avg. depth | 4.60 | 3.40 |
| Avg. number of nodes | 14.20 | 7.80 |
| Avg. number of leaves | 7.60 | 4.40 |
| Avg. bivariate nodes | 0.00 | 2.40 |
| Avg. univariate fallback nodes | 6.60 | 1.00 |

Standard deviations are sample standard deviations (`ddof=1`) across the five folds. The baseline's bivariate-node count is 0.00 by construction, since all its splits are univariate.

### Output files

- `results/cross_validation_results.csv` — one summary row per model
- `results/cross_validation_results.json` — summary plus the full per-fold breakdown

---

## 6. Plots Generated

All plots are produced by `python main.py` and stored in `plots/`:

| File | Contents |
|---|---|
| `accuracy_comparison.png` | Grouped bar chart of train and test accuracy for both models |
| `tree_size_comparison.png` | Grouped bar chart of depth, node count and leaf count |
| `confusion_matrices.png` | Side-by-side test-set confusion matrices |
| `decision_boundary_sepal_length_(cm)_petal_width_(cm).png` | Training data in the (sepal length, petal width) plane with the learned bivariate boundary overlaid as a dashed line |

**Note on the decision-boundary plot.** It shows one node's boundary in isolation, projected onto its two features, plotted against **all** training points rather than only the subset that actually reaches that node after earlier splits. This is a deliberate visualization simplification, documented in the source, chosen because it gives a clearer 2-D picture of what the rule does.

---

## 7. Key Observations

1. **Tree size:** the bivariate tree was consistently smaller in both experiments — 7 vs. 15 nodes and depth 3 vs. 5 on the single split, and 7.80 vs. 14.20 nodes and depth 3.40 vs. 4.60 on average across folds.
2. **Genuine bivariate usage:** across folds the tree averaged 2.40 bivariate nodes against 1.00 univariate fallback node, so the majority of its decisions used real two-feature rules.
3. **The two experiments disagree on accuracy.** The single split favored the bivariate tree (0.9556 vs. 0.9333); cross-validation favored the baseline (0.9533 vs. 0.9400 mean accuracy). Cross-validation is treated as the more reliable estimate.
4. **Variability:** the bivariate tree showed larger standard deviations across folds on every accuracy-type metric.
5. **Training cost:** the bivariate tree was roughly 800× slower to train, an expected consequence of its brute-force search over 6 feature pairs × 18 orientations × all candidate thresholds at every node.

---

## 8. Limitations of These Experiments

- **Single small dataset.** Iris has 150 samples, 4 features, and is close to linearly separable using single features, leaving little room for a two-feature rule to demonstrate an advantage.
- **Only 5 folds, one seed.** No repeated cross-validation and no multiple random seeds were run.
- **No statistical significance testing.** The accuracy differences observed are small relative to the fold-to-fold standard deviations, and no paired test was performed. **No claim of significance is made.**
- **No hyperparameter tuning.** Fixed hyperparameters were used deliberately for fairness, but neither model was optimized, so neither result represents best achievable performance.
- **Timing comparison is not like-for-like.** The baseline is a compiled, highly optimized library implementation; ours is pure Python/NumPy. The ~800× gap reflects implementation maturity as much as algorithmic cost.
- **Algorithmic simplifications.** Gini impurity instead of exact 0/1 loss, coarse 18-orientation sampling, no TAO, no λ/C regularization, and a local heuristic univariate fallback — each documented in `docs/implementation.md`.

**These results characterize our simplified implementation on one small dataset. They do not reproduce the KDD 2024 paper's experiments and should not be compared against its reported numbers.**

---

## 9. Reproducibility

Both experiments use `random_state=42` throughout and are reproducible from a clean environment via `pip install -r requirements.txt`.

Re-running `python main.py` regenerates `results/comparison_table.csv`, `results/full_results.json` and all plots. Every metric reproduces exactly; only the live-measured `training_time_seconds` field varies between runs, since it depends on machine speed and load.
