# Implementation and Experimental Evaluation of Bivariate Decision Trees on a Classification Dataset

> ⚠️ **This is a simplified student implementation, not an exact reproduction of the complete KDD 2024 algorithm.**
> It implements the *conceptual core* of the bivariate decision tree idea (a two-feature linear routing rule with an approximate orientation-based split search) at a one-week academic project scale. The paper's full bivariate TAO algorithm, its λ/C regularization, and its large-scale experiments are **not** implemented. See [Difference Between the Original Paper and Our Implementation](#19-difference-between-the-original-paper-and-our-implementation).

---

## 1. Project Title

**Implementation and Experimental Evaluation of Bivariate Decision Trees on a Classification Dataset**

## 2. Research Paper

- **Title:** *Bivariate Decision Trees: Smaller, Interpretable, More Accurate*
- **Authors:** Rasul Kairgeldin, Miguel Á. Carreira-Perpiñán

## 3. Venue

**KDD 2024** — 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

## 4. Project Objective

Implement a simplified, from-scratch version of the bivariate decision tree idea — in which each decision node routes samples using a learned linear rule over **at most two features** — and compare it experimentally against a conventional univariate decision tree on a classification dataset, using both a single train/test split and 5-fold stratified cross-validation.

## 5. Problem Statement

A standard ("univariate") decision tree makes each decision using a single feature at a time. When the true decision boundary depends on more than one feature simultaneously, the tree must approximate it with many zig-zagging axis-aligned splits, producing deeper and larger trees that are harder to interpret and may generalize worse.

This project investigates whether allowing a small, controlled increase in per-node complexity — at most two features per node, combined via a learned linear rule — can produce **smaller and/or more accurate** trees than a conventional univariate decision tree.

## 6. Bivariate Decision-Tree Concept

The original paper proposes a middle ground between two extremes:

| Model type | Features per node | Interpretability | Expressiveness |
|---|---|---|---|
| Univariate (CART) | 1 | Very high | Limited — axis-aligned only |
| **Bivariate (this paper)** | **≤ 2** | **High** | **Can express oblique boundaries** |
| Oblique / multivariate | All | Low | Very high |

By allowing two features per node, a bivariate tree can express a diagonal (oblique) boundary in a single split where a univariate tree would need many, producing a smaller tree without sacrificing much interpretability.

## 7. Mathematical Routing Rule

Each internal node stores `(feat_j, feat_k, w1, w2, b)` and routes a sample `x` as:

```
score = w1 · x[feat_j] + w2 · x[feat_k] + b

if score < 0  →  go to LEFT child
else          →  go to RIGHT child
```

Subject to the constraint that **no node uses more than two features**.

Univariate fallback nodes are represented in the same form with `w2 = 0`, so a single routing implementation handles both node types.

## 8. Simplified Implementation Methodology

At each node, the algorithm:

1. **Enumerates candidate feature pairs** — all pairs (with Iris's 4 features this is all 6 pairs).
2. **Samples a small fixed set of line orientations** — `n_orientations` angles spaced uniformly over 0°–180°, giving `(w1, w2) = (cos θ, sin θ)`.
3. **Projects** the two chosen features onto each orientation: `proj = w1·x_j + w2·x_k`.
4. **Searches for the best threshold** `b` along that 1-D projection, testing midpoints between sorted distinct values.
5. **Scores each candidate split** by **Gini impurity reduction**, keeping the best (pair, orientation, threshold) combination.
6. **Computes a univariate fallback split** (standard single-feature CART search) and uses whichever split has the higher Gini gain; ties favor the simpler univariate split.
7. **Recurses** on the left/right partitions until a stopping criterion is met.

**Stopping criteria:** maximum depth reached, fewer samples than `min_samples_split`, node already pure, or no split achieving at least `min_impurity_decrease`.

Every node is explicitly tagged `"bivariate"` or `"univariate"`, so the tree's composition is fully auditable.

## 9. Dataset

- **Dataset:** Iris (loaded via `sklearn.datasets.load_iris`; a reference copy is included at `data/iris.csv`)
- **Samples:** 150
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** 3 (setosa, versicolor, virginica)

## 10. Experimental Setup

**Bivariate tree hyperparameters** (identical in both experiments; no per-model tuning was performed):

```
max_depth             = 4
min_samples_split     = 5
n_orientations        = 18
min_impurity_decrease = 1e-7
random_state          = 42
```

**Baseline:** `sklearn.tree.DecisionTreeClassifier` with Gini criterion, default stopping (`max_depth=None`), `random_state=42`.

- **Single split:** stratified, `test_size=0.3`, `random_state=42`
- **Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the full dataset

## 11. Single Train/Test Results

| Metric | Baseline (univariate) | Bivariate (ours) |
|---|---|---|
| Train accuracy | 1.0000 | 0.9905 |
| Test accuracy | 0.9333 | **0.9556** |
| Precision (macro) | 0.9444 | 0.9556 |
| Recall (macro) | 0.9333 | 0.9556 |
| F1 (macro) | 0.9327 | 0.9556 |
| Training time (s) | **0.0014** | 0.8427 |
| Depth | 5 | **3** |
| Number of nodes | 15 | **7** |
| Number of leaves | 8 | **4** |

**Node breakdown:** 1 bivariate node, 2 univariate fallback nodes.
The bivariate node used **(sepal length, petal width)** with `w1 = 0.1736`, `w2 = 0.9848`, `b = -2.8116` (Gini gain 0.4459, 70 samples reaching the node).

## 12. 5-Fold Cross-Validation Results

| Metric | Baseline (univariate) | Bivariate (ours) |
|---|---|---|
| Accuracy (mean ± std) | **0.9533 ± 0.0380** | 0.9400 ± 0.0548 |
| Precision macro | **0.9572 ± 0.0365** | 0.9434 ± 0.0523 |
| Recall macro | **0.9533 ± 0.0380** | 0.9400 ± 0.0548 |
| F1 macro | **0.9531 ± 0.0382** | 0.9396 ± 0.0552 |
| Avg. training time (s) | **0.0011** | 0.8749 |
| Avg. depth | 4.60 | **3.40** |
| Avg. number of nodes | 14.20 | **7.80** |
| Avg. number of leaves | 7.60 | **4.40** |
| Avg. bivariate nodes | 0.00 | 2.40 |
| Avg. univariate fallback nodes | 6.60 | 1.00 |

## 13. Discussion

Across both evaluations, the bivariate tree consistently produced **smaller trees** than the baseline — fewer nodes, fewer leaves, lower depth — and this reduction is genuine: on average, more of its decision nodes used true two-feature rules (2.40) than univariate fallbacks (1.00).

However, the two evaluations **disagree on accuracy**. The single split suggested the bivariate tree was slightly *more* accurate (0.9556 vs. 0.9333 test accuracy). The 5-fold cross-validation shows the **opposite**: the baseline achieved marginally *higher* mean accuracy and F1 (0.9533 vs. 0.9400, 0.9531 vs. 0.9396), with the bivariate tree also showing greater variability across folds.

This disagreement is expected on a small dataset like Iris, where a single 45-sample test set can be unrepresentative by chance. **Cross-validation is treated here as the more reliable estimate**, and it does not show a clear accuracy advantage for our simplified implementation. The clearest and most consistent benefit observed is **model compactness, not accuracy**.

The bivariate tree was also substantially slower to train (~800× on average), which is expected given its brute-force per-node search over feature pairs and orientations versus scikit-learn's optimized single-feature search.

These findings reflect the behavior of *our specific simplified implementation* on one small, relatively easy dataset — they are not a judgment on the bivariate-tree idea as presented in the original paper.

## 14. Limitations

- **Iris is small and easy** (150 samples, 4 features, near-linearly separable). The paper evaluates on much larger, more varied datasets, so these results cannot be generalized.
- **Simplified split-quality measure:** Gini impurity reduction is used instead of the paper's exact 0/1 loss objective.
- **Coarse fixed orientation sampling:** only 18 orientations per feature pair; better splits falling between sampled angles can be missed.
- **No bivariate TAO:** the paper's slower, globally-optimized alternating-optimization algorithm is not implemented.
- **No λ/C regularization:** the paper's formal mechanism for choosing between zero-, one-, and two-feature nodes is not implemented.
- **Local, heuristic univariate fallback:** the bivariate-vs-univariate choice at each node is a simple local Gini-gain comparison, not a globally optimized decision.
- **Not a reproduction:** the numbers here should not be compared against, or used to validate/invalidate, the KDD 2024 paper's reported results.

## 15. Future Work

- Implement the full bivariate TAO algorithm (alternating optimization over all tree nodes).
- Implement the paper's λ/C regularization and regularization path.
- Evaluate on larger, higher-dimensional datasets used in the original paper.
- Compare more extensively against conventional and oblique/multivariate trees.
- Add statistical significance testing (paired tests across folds, repeated CV with multiple seeds).

## 16. How to Install

```bash
git clone https://github.com/<your-username>/bivariate-decision-trees-kdd2024.git
cd bivariate-decision-trees-kdd2024

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## 17. How to Run

Run the single train/test experiment (trains both models, saves results and plots, prints a summary):

```bash
python main.py
```

Run the 5-fold stratified cross-validation robustness check:

```bash
python -m src.cross_validation
```

All experiments use `random_state=42` and are reproducible from a clean environment.

> **Note:** `main.py` regenerates `results/comparison_table.csv`, `results/full_results.json`, and the plot files. All metrics reproduce exactly; only the live-measured `training_time_seconds` field varies between runs, since it depends on machine speed.

## 18. Project Structure

```
bivariate-decision-trees-kdd2024/
│
├── data/
│   └── iris.csv                    # Reference copy of the dataset
│
├── src/
│   ├── __init__.py
│   ├── baseline_tree.py            # Standard univariate tree (sklearn wrapper)
│   ├── bivariate_tree.py           # Custom BivariateDecisionTree (core deliverable)
│   ├── metrics.py                  # Shared evaluation utilities
│   ├── experiment.py               # Single train/test experiment
│   └── cross_validation.py         # 5-fold stratified CV robustness check
│
├── results/
│   ├── comparison_table.csv
│   ├── full_results.json
│   ├── cross_validation_results.csv
│   └── cross_validation_results.json
│
├── plots/
│   ├── accuracy_comparison.png
│   ├── tree_size_comparison.png
│   ├── confusion_matrices.png
│   └── decision_boundary_sepal_length_(cm)_petal_width_(cm).png
│
├── docs/
│   ├── paper-summary.md            # Summary of the original KDD 2024 paper
│   ├── implementation.md           # Detailed walkthrough of our code
│   ├── experiments.md              # Experimental protocol and results
│   └── viva-notes.md               # Concise revision notes for the viva
│
├── README.md
├── PROJECT_REPORT.md               # Concise submission-ready report
├── requirements.txt
├── .gitignore
└── main.py
```

## 19. Difference Between the Original Paper and Our Implementation

| Aspect | Original Paper (KDD 2024) | Our Student Implementation |
|---|---|---|
| Learning algorithms | Fast "bivariate CART" **and** slower, globally-optimized "bivariate TAO" | Only a simplified version of the fast CART-style idea; **TAO not implemented** |
| Split-quality objective | Exact 0/1 loss | Gini impurity reduction |
| Orientation search | Fixed set of H orientations (typically H = 30–90) | Fixed set, H = 18 (smaller, for speed) |
| Node-type selection | Formal, globally optimized rule using λ/C regularization | Simple local comparison of Gini gain at each node |
| Regularization / pruning | λ (tree size) and C (uni- vs. bivariate cost), with a full regularization path | Not implemented; only standard stopping conditions |
| Datasets | Multiple UCI benchmarks of varying size and dimensionality | Only Iris (150 samples, 4 features, 3 classes) |
| Evaluation protocol | Repeated splits with cross-validated hyperparameters | Single 70/30 split + 5-fold stratified CV, no tuning |
| Implementation scale | C++/parallel, GPU-vectorizable projection | Pure Python/NumPy, single-threaded |
| Feature-pair search | All D² pairs | All pairs (feasible with 4 features); would need subsampling for larger D |

**This project reproduces the conceptual core of the bivariate decision tree idea. It does not reproduce, and should not be interpreted as reproducing, the KDD 2024 paper's full algorithms, datasets, or experimental results.**

## 20. Citation

If you refer to the original work, please cite:

```bibtex
@inproceedings{kairgeldin2024bivariate,
  title     = {Bivariate Decision Trees: Smaller, Interpretable, More Accurate},
  author    = {Kairgeldin, Rasul and Carreira-Perpi{\~n}{\'a}n, Miguel {\'A}.},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2024}
}
```

Kairgeldin, R., & Carreira-Perpiñán, M. Á. (2024). *Bivariate Decision Trees: Smaller, Interpretable, More Accurate.* In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2024).

---

**Academic integrity note:** This repository contains an independent, simplified student reimplementation created for coursework. It is not affiliated with or endorsed by the original authors, and its results do not represent the performance of the methods described in the original paper.
