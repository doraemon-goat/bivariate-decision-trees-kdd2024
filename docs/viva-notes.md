# Viva Notes

Concise revision notes. Everything here is drawn from the existing implementation and results — no new claims.

---

## 1. Research Problem

Standard univariate decision trees split on **one feature at a time**, so each cut is axis-aligned. When the true class boundary is diagonal — which happens whenever features are correlated — the tree must approximate it with a *staircase* of many small cuts. The result is a large, deep tree that is harder to interpret and may generalize worse.

## 2. Motivation

The classical fix is the **oblique tree**, where each node uses all D features. That solves the geometry but destroys interpretability — nobody can read a 50-weight rule.

The paper's insight: this is a false dichotomy. Allow **exactly up to two** features per node. Two features are enough to draw a diagonal line in a plane (the most common failure case of univariate trees), yet a two-variable rule can still be read as a sentence *and plotted on a 2-D scatterplot*. Fewer nodes offset the slightly higher per-node complexity.

## 3. The Mathematical Equation

```
score = w₁ · x_j + w₂ · x_k + b

score < 0  →  go LEFT
otherwise  →  go RIGHT
```

Constraint: **at most two non-zero weights per node.**

In our code this lives in `DecisionNode.route(x)`. A univariate node is stored in the same form with `w2 = 0`, so one formula handles both node types.

## 4. Bivariate vs. Univariate

| | Univariate | Bivariate | Oblique |
|---|---|---|---|
| Features/node | 1 | ≤ 2 | all D |
| Boundary shape | Axis-aligned only | Any line in a 2-D plane | Any hyperplane |
| Interpretable? | Very | Yes — can be drawn | No |
| Tree size | Large | Smaller | Small |

## 5. Gini Impurity

```
Gini(y) = 1 − Σ_c p_c²
```

0 = pure node; higher = more mixed. We score a split by **impurity reduction**:

```
gain = Gini(parent) − [ (n_L/n)·Gini(left) + (n_R/n)·Gini(right) ]
```

Children are weighted by size so a split yielding two large pure children beats one yielding a tiny pure child plus a large mixed one. Higher gain = better split. This is the standard CART criterion.

## 6. Orientation Search

The exact optimal 2-D linear split is very expensive to find (on the order of O(N³D²)). The paper's approximation: fix a **small set of H line orientations**, sampled uniformly by rotating a direction vector from **0° to 180°**.

- Why stop at 180°? Angles beyond it give the *same line* with left and right swapped — redundant.
- Parameterization: `(w₁, w₂) = (cos θ, sin θ)`, which automatically gives a unit weight vector.
- **Our H = 18.** The paper typically uses 30–90; ours is smaller for speed — a documented simplification.

## 7. Projection

For each orientation, compute

```
proj = w₁·x_j + w₂·x_k
```

This maps each 2-D point onto a 1-D line at angle θ. **This is the key trick**: it converts the hard 2-D "find the best separating line" problem into the easy, well-understood 1-D "find the best threshold" problem.

## 8. Threshold Search

On the projected values (`_best_threshold_for_projection`):

1. Sort samples by projected value.
2. Scan adjacent pairs, **skipping identical values** (a threshold between two equal values separates nothing).
3. Take the midpoint; set `b = −midpoint`.
4. Score with Gini gain; keep the best.

Returns `None` when no valid split exists (e.g. all values identical).

## 9. CART

Greedy recursive partitioning: at each node pick the locally best split by Gini, then recurse. It optimizes **no global objective** — each decision is made once and never revisited. Our tree-growing strategy is CART-style, corresponding in spirit to the paper's **fast bivariate CART** variant.

## 10. TAO — *Not implemented*

**Tree Alternating Optimization.** Fixes the tree *structure* and optimizes a **global objective** (0/1 loss + regularization) by alternating over nodes.

- Exploits **separability**: with the rest of the tree fixed, a node's parameters depend only on its **reduced set** (the instances reaching it), so each node can be optimized independently.
- Sweeps nodes in **reverse breadth-first order** (deepest → root); gives **monotonic descent** and finite-iteration convergence.
- At each node it compares **zero-feature / one-feature / two-feature** solutions and picks the lowest regularized loss — this is what lets the tree prune itself automatically.

TAO gives the paper's **best** results. We did not implement it. ⚠️ *Expect a question here.*

## 11. Regularization — *Not implemented*

Two hyperparameters in the paper: **λ** controls overall tree size; **C** sets the relative cost of using two features instead of one. Together they let each node decide how much complexity is justified, and drive a **regularization path** over λ.

We implement neither. Instead we use plain stopping criteria plus a **local** Gini-gain comparison to choose node type.

## 12. Why Our Implementation Is Simplified

| Aspect | Paper | Ours |
|---|---|---|
| Algorithms | Bivariate CART **and** TAO | CART-style only |
| Objective | Exact 0/1 loss | Gini impurity reduction |
| Orientations | H ≈ 30–90 | H = 18 |
| Node-type choice | Global λ/C rule | Local Gini comparison |
| Regularization | λ, C + full path | None |
| Datasets | Many UCI benchmarks | Iris only |
| Code | C++/parallel, GPU-vectorizable | Pure Python/NumPy |

**One-week project scope.** Everything above is documented in the code and README — nothing is hidden.

## 13. Results

**Single 70/30 split:**

| | Baseline | Bivariate |
|---|---|---|
| Test accuracy | 0.9333 | **0.9556** |
| Depth / nodes / leaves | 5 / 15 / 8 | **3 / 7 / 4** |
| Training time (s) | **0.0014** | 0.8427 |

Node breakdown: 1 bivariate, 2 univariate. The bivariate node: **(sepal length, petal width)**, `w1=0.1736`, `w2=0.9848`, `b=−2.8116`, gain 0.4459, 70 samples.

**5-fold stratified CV (mean ± std):**

| | Baseline | Bivariate |
|---|---|---|
| Accuracy | **0.9533 ± 0.0380** | 0.9400 ± 0.0548 |
| F1 macro | **0.9531 ± 0.0382** | 0.9396 ± 0.0552 |
| Avg. depth / nodes / leaves | 4.60 / 14.20 / 7.60 | **3.40 / 7.80 / 4.40** |
| Avg. bivariate / univariate nodes | 0.00 / 6.60 | 2.40 / 1.00 |

**The headline:** consistently **smaller trees with genuine bivariate nodes**, but **no accuracy advantage** once evaluated by cross-validation.

## 14. Limitations

Small easy dataset · Gini instead of 0/1 loss · coarse 18-orientation sampling · no TAO · no λ/C regularization · local heuristic fallback · single seed, 5 folds, no significance test · timing not like-for-like (compiled library vs. pure Python). **Not a reproduction of the paper.**

---

## 15. Likely Professor Questions

**Q1. How do you know this isn't just a decision tree with relabeled splits?**
`get_decision_nodes()` exposes every node's `node_type`, feature indices, weights and bias. The single-split tree contains a node with `w1=0.1736` **and** `w2=0.9848` — both non-zero, over two distinct features. Across CV folds it averaged 2.40 bivariate nodes vs. 1.00 univariate. It's verifiable, not asserted.

**Q2. Your CV accuracy is *worse* than the baseline. Doesn't that refute the paper?**
No. It says something about *our simplified implementation on Iris*, not about the paper's method. We omitted TAO and λ/C regularization — precisely the components producing the paper's best results — used a coarser orientation grid, and tested on one small, near-linearly-separable dataset. The honest conclusion: we reproduced the **"smaller"** claim, not the **"more accurate"** one.

**Q3. Why did the single split and CV disagree?**
The single split has a 45-sample test set; a handful of borderline samples swings accuracy by several points. CV averages five disjoint test partitions and reports variability, so it's the more reliable estimate — which is exactly why we ran it.

**Q4. Why Gini instead of the paper's 0/1 loss?**
Gini is the standard CART criterion, cheap to evaluate at every candidate threshold, and differentiates between splits of equal error rate but different class purity. Exact 0/1-loss optimization belongs to the TAO formulation we didn't implement. It's a documented simplification.

**Q5. Why 180° and not 360°?**
Angles past 180° produce the same line with the sides swapped — mathematically redundant. Sampling `[0, π)` covers every distinct orientation.

**Q6. Why is your tree ~800× slower?**
Per node we evaluate 6 feature pairs × 18 orientations × all candidate thresholds, each requiring a sort and a scan — versus scikit-learn's optimized single-feature search in compiled code. Part algorithmic, part implementation maturity. The paper addresses this with vectorized/GPU projection, which we didn't implement.

**Q7. What is TAO and why didn't you implement it?**
[See §10.] Not implemented because it requires fixed-structure alternating optimization, reduced-set bookkeeping, reverse-BFS node sweeps, and a λ regularization path — beyond a one-week scope. It's the top item in future work.

**Q8. What does the univariate fallback do, and is it in the paper?**
At every node we compute both the best bivariate and best univariate split and take whichever has higher Gini gain (ties → univariate, preferring simplicity). **It is not from the paper** — the paper uses a globally optimized λ/C rule. Ours is a transparent local substitute with no global optimality guarantee.

**Q9. Why does a univariate node still store two feature indices?**
So one routing formula serves both node types. A univariate node sets `w2 = 0`, making `feat_k` genuinely irrelevant to the decision. `get_decision_nodes()` reports `feat_k` and `w2` as `None` for univariate nodes, so the distinction stays unambiguous.

**Q10. Are your accuracy differences statistically significant?**
We make **no significance claim**. The gaps are small relative to fold-to-fold standard deviations (±0.038 to ±0.055) and we ran only 5 folds with a single seed and no paired test. Proper testing is listed under future work.

**Q11. Why Iris, given its known limitations?**
One-week scope: small, well understood, and easy to visualize in 2-D — which matters for showing the bivariate boundary. Its weakness is exactly what we report: it's nearly separable on single features, leaving little room for a two-feature rule to help. Larger benchmarks are future work.

**Q12. What's the single most important thing you learned?**
That evaluation methodology can reverse a conclusion. The single split said our model was more accurate; cross-validation said the opposite. Had we stopped at the first experiment, we'd have reported a result we couldn't support.
