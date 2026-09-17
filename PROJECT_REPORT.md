# Project Report

## Implementation and Experimental Evaluation of Bivariate Decision Trees on a Classification Dataset

**Based on:** Kairgeldin, R. & Carreira-Perpiñán, M.Á. (2024). *Bivariate Decision Trees: Smaller, Interpretable, More Accurate.* KDD 2024.

---

### 1. Objective

Implement a simplified version of the bivariate decision tree idea — where each decision node routes samples using a linear rule over **at most two features** — and compare it experimentally against a standard univariate decision tree on the Iris dataset.

### 2. Core Idea Implemented

Each bivariate decision node stores `(feat_j, feat_k, w1, w2, b)` and routes a sample `x` using:

```
score = w1 * x[feat_j] + w2 * x[feat_k] + b
left if score < 0, otherwise right
```

The tree is grown greedily (recursive partitioning). At each node, the algorithm searches all feature pairs and a small fixed set of line orientations (0°–180°), projects the data onto each orientation, and finds the best threshold via Gini impurity reduction. A univariate (single-feature) split is also computed at each node as a fallback; whichever of the two achieves higher Gini gain is used, and every node is explicitly tagged `"bivariate"` or `"univariate"`.

### 3. What Was Reproduced vs. Simplified

| Reproduced from the paper | Simplified / Out of scope |
|---|---|
| Two-feature linear routing rule | No bivariate TAO (alternating global optimization) |
| ≤2 features per node constraint | No λ/C regularization or regularization path |
| Fixed-orientation projection + threshold search | Gini impurity used instead of exact 0/1 loss |
| Greedy recursive partitioning ("bivariate CART" spirit) | Local, heuristic bivariate-vs-univariate fallback rule |
| — | Evaluated only on Iris (paper uses much larger datasets) |

### 4. Experimental Setup

- **Dataset:** Iris — 150 samples, 4 features, 3 classes.
- **Single split:** `test_size=0.3`, `random_state=42`, stratified.
- **Cross-validation:** 5-fold Stratified K-Fold, `shuffle=True`, `random_state=42`.
- **Bivariate tree hyperparameters (fixed, not tuned per model):**
  `max_depth=4`, `min_samples_split=5`, `n_orientations=18`, `min_impurity_decrease=1e-7`.

### 5. Results

**Single train/test split:**

| Metric | Baseline | Bivariate |
|---|---|---|
| Test accuracy | 0.9333 | 0.9556 |
| F1 (macro) | 0.9327 | 0.9556 |
| Depth / Nodes / Leaves | 5 / 15 / 8 | 3 / 7 / 4 |
| Training time (s) | 0.0014 | 0.8427 |

**5-fold stratified cross-validation (mean ± std):**

| Metric | Baseline | Bivariate |
|---|---|---|
| Accuracy | 0.9533 ± 0.0380 | 0.9400 ± 0.0548 |
| F1 (macro) | 0.9531 ± 0.0382 | 0.9396 ± 0.0552 |
| Avg. depth / nodes / leaves | 4.60 / 14.20 / 7.60 | 3.40 / 7.80 / 4.40 |
| Avg. bivariate / univariate nodes | 0.00 / 6.60 | 2.40 / 1.00 |
| Avg. training time (s) | 0.0011 | 0.8749 |

### 6. Discussion (Summary)

The bivariate tree was **consistently smaller** than the baseline (fewer nodes, leaves, and lower depth) in both evaluations, and genuinely used two-feature rules in the majority of its decision nodes. The single split suggested the bivariate tree was also more accurate, but the more reliable 5-fold cross-validation showed the **opposite** — the baseline had marginally higher mean accuracy and F1, and the bivariate tree showed more variability across folds. This discrepancy is expected on a small dataset like Iris, where a single split can be unrepresentative; cross-validation is treated here as the more trustworthy estimate. The bivariate tree was also substantially slower to train due to its brute-force per-node search.

### 7. Limitations

Results are limited to a small, easy dataset (Iris); the split-search uses Gini impurity and a coarse fixed orientation set rather than the paper's exact objective; no TAO optimization or λ/C regularization was implemented; and the bivariate-vs-univariate choice at each node uses a simple local heuristic rather than a globally optimized rule. **These results do not reproduce, and should not be compared against, the KDD 2024 paper's own experiments or reported numbers.**

### 8. Conclusion

The implementation demonstrates the core mechanics of a bivariate decision tree — genuine two-feature linear routing, smaller resulting trees — but does not demonstrate a clear accuracy advantage over a standard decision tree on this small dataset once evaluated more robustly via cross-validation. This is consistent with the fact that only a simplified, greedy, unregularized version of the paper's fast algorithm was implemented, without its more powerful TAO-based optimization.

### 9. Future Work

- Implement bivariate TAO and λ/C regularization.
- Evaluate on larger, higher-dimensional datasets used in the original paper.
- Add statistical significance testing across folds.

---

*Full implementation details, complete discussion, and the paper-vs-implementation comparison table are provided in `README.md`. Numeric results are saved in `results/`, and plots in `plots/`.*
