# Paper Summary

**Paper:** *Bivariate Decision Trees: Smaller, Interpretable, More Accurate*
**Authors:** Rasul Kairgeldin, Miguel Á. Carreira-Perpiñán
**Venue:** KDD 2024 (30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining)

---

## 1. The Problem the Paper Addresses

Decision trees are popular largely because they are **interpretable**: you can read a path from the root to a leaf as a short chain of human-understandable questions. The most common variety is the **univariate** (or axis-aligned) tree, produced by algorithms like CART, where every internal node asks a question about exactly one feature:

> "Is petal length greater than 2.5?"

The limitation is geometric. A univariate node can only cut the input space with a hyperplane perpendicular to one axis. When the real class boundary is diagonal — which happens whenever features are correlated, i.e. most of the time in practice — the tree must approximate that diagonal with a *staircase* of many small axis-aligned cuts.

The consequences are:

- The tree becomes **large and deep**, requiring many nodes to express one simple boundary.
- A large tree is **harder to interpret**, undermining the main reason for using a tree at all.
- The staircase approximation can also **generalize poorly**, since it overfits the particular shape of the training data near the boundary.

## 2. The Existing Alternative and Why It Isn't Satisfactory

The classical fix is the **oblique** (multivariate) tree, where each node uses a linear combination of *all* input features:

```
w₁x₁ + w₂x₂ + ... + w_D x_D + b < 0
```

This solves the geometry problem — a single node can now express any hyperplane — but creates a new one. A node with 50 non-zero weights is no longer something a human can read at a glance. You trade interpretability for expressiveness, which defeats the original purpose.

## 3. The Paper's Proposal: Bivariate Trees

The paper's key insight is that this is a false dichotomy. Instead of choosing between "one feature" and "all features," it proposes a **middle ground**: allow each node to use **at most two features**.

The routing rule becomes:

```
score = w₁ · x_j + w₂ · x_k + b

if score < 0  →  go left
else          →  go right
```

subject to the constraint that no more than two of the weights are non-zero.

Why two is a good choice:

- **Two features can express a diagonal line in a 2-D plane.** This is enough to capture the most common and most costly failure case of univariate trees — a boundary between a pair of correlated features — so a single bivariate node can replace a long staircase of univariate nodes.
- **Two features remain interpretable.** A rule involving two variables can be read as a sentence, and crucially, it can be *drawn*: you can plot the two features on a 2-D scatterplot and see the line. This visual verifiability is lost entirely with oblique trees.
- **Smaller trees offset the added node complexity.** Each node is slightly more complex, but the tree has far fewer nodes, so total model complexity often goes *down*.

The paper's title captures the claim: bivariate trees are typically **smaller**, remain **interpretable**, and are **more accurate** than univariate trees.

## 4. The Core Technical Difficulty

Finding the *optimal* two-feature linear split at a node is computationally hard. Exactly enumerating all distinct linear separations of N points in 2-D, across all feature pairs, is very costly (the paper notes a cost on the order of O(N³D²)), which is impractical for real datasets.

The paper's solution is an **approximation based on orientation sampling**:

1. Fix a small set of **H line orientations**, sampled uniformly by rotating a direction vector from 0° to 180°. (Angles beyond 180° are redundant, since they produce the same line with the sides swapped.)
2. For a given feature pair, **project** every sample onto each orientation — i.e. compute `w₁x_j + w₂x_k` where `(w₁, w₂) = (cos θ, sin θ)`.
3. This reduces the hard 2-D problem to a familiar **1-D thresholding problem**, which can be solved efficiently by sorting the projected values and scanning candidate thresholds.
4. Repeat across all orientations and all feature pairs, and keep the best combination.

The resulting weight vector is **sparse** — only two entries are non-zero — which is exactly the bivariate constraint. The paper reports using H values typically between 30 and 90 in its experiments, and notes that the projection step can be vectorized (and GPU-accelerated) by computing all projections as a matrix product.

## 5. The Two Learning Algorithms

The paper presents two ways to build such trees, trading speed against quality.

### 5.1 Bivariate TAO (the high-quality algorithm)

TAO (Tree Alternating Optimization) treats tree learning as the optimization of a **well-defined global objective function** over a tree of *fixed structure*, rather than as greedy top-down growth. The objective combines:

- a **misclassification loss** (0/1 loss) over the training set, and
- a **regularization term** that penalizes node complexity differentially, controlled by hyperparameters **λ** (which controls overall tree size) and **C** (which sets the relative cost of using two features versus one).

The algorithm exploits a *separability* property: given the rest of the tree fixed, the parameters of a single node can be optimized independently, considering only the **reduced set** — the training instances that actually reach that node. TAO then alternates over the nodes (in reverse breadth-first order, from the deepest nodes up to the root), re-optimizing each in turn. This produces **monotonic descent** of the objective and converges in a finite number of iterations.

At each node, the reduced problem is solved by comparing three candidate solutions — a **zero-feature** solution (effectively removing the node), a **one-feature** (univariate) solution, and a **two-feature** (bivariate) solution — and selecting whichever minimizes the loss plus its regularization penalty, with ties broken in favor of fewer parameters. This is what allows the tree to *automatically* prune itself and decide, per node, how much complexity is justified.

TAO produces the paper's best results, but it is slower and requires sweeping λ over a regularization path.

### 5.2 Bivariate CART (the fast algorithm)

The same orientation-sampling split search can also be plugged into ordinary greedy recursive partitioning — i.e. CART. Two variants are described: modifying CART's Gini-based split step to use the partial enumeration over bivariate splits, or pre-constructing an augmented feature set containing the projected features and then running standard univariate CART on it.

This approach **does not optimize any global objective**, so it produces worse trees than bivariate TAO, but it is **much faster**.

## 6. Relevance to This Project

Our student implementation reproduces the *conceptual core* described in Section 3 and the *approximate split search* described in Section 4 — the two-feature routing rule, orientation sampling, projection, and 1-D threshold search — combined with greedy recursive partitioning, which corresponds in spirit to the fast **bivariate CART** approach of Section 5.2.

We do **not** implement bivariate TAO (Section 5.1), the λ/C regularization objective, or the exact 0/1-loss node solution. Those are documented as out of scope in [`implementation.md`](implementation.md) and in the project README.
