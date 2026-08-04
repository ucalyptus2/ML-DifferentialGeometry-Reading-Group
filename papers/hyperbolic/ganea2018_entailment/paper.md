# Hyperbolic Entailment Cones for Learning Hierarchical Embeddings

- **Authors:** Octavian-Eugen Ganea, Gary Bécigneul, Thomas Hofmann
- **Venue/Year:** ICML 2018
- **arXiv:** https://arxiv.org/abs/1804.01882
- **Category:** hyperbolic

## One-paragraph TL;DR
Instead of encoding entailment purely through hyperbolic distance (as Poincaré embeddings do), this paper assigns each embedded point a **geodesic cone**—a region of the Poincaré ball emanating from the origin with a point-dependent half-aperture—and declares $u \succeq v$ iff $v$ falls inside $u$'s cone. The aperture shrinks as a point moves deeper (toward the boundary), mirroring the exponential shrinkage of subtrees in hyperbolic space. This gives a *partial order* structure on embeddings that is more robust than pure distance for learning hierarchical taxonomies, with state-of-the-art results on WordNet noun and verb reconstruction.

## The problem
Poincaré embeddings encode hierarchy through *distance*: ancestors are closer to the origin, descendants near the boundary, and parent–child distance encodes depth. But distance alone does not give a **partial order**: it is symmetric and does not directly answer "does $u$ entail $v$?" Furthermore, in a tree, a node's subtree occupies an *angular sector* at the origin, and the angular width of this sector shrinks with depth (exponentially in hyperbolic space). A pure distance-based embedding ignores this angular/cone structure. The paper asks: can we explicitly model the entailment relation as geometric *containment* rather than as a distance threshold?

## Key idea(s)
- **Entailment cone.** Each point $u$ in the Poincaré ball is assigned a cone $C(u)$ with apex at the origin, axis along the direction of $u$, and half-aperture $\alpha(u)$. A point $v$ is a descendant of $u$ ($u \succeq v$) iff $v \in C(u)$.
- **Depth-dependent aperture.** The half-aperture $\alpha(u)$ is a decreasing function of $\|u\|$ (the Euclidean norm, which proxies depth). Deeper points have narrower cones, matching the exponential shrinkage of subtree angular width in hyperbolic space.
- **Cone-aware loss.** The training objective pushes true descendants inside the ancestor's cone and pushes non-descendants outside, using a smooth (sigmoid or hinge) surrogate for the containment indicator.

## The mathematics
In the Poincaré ball $\mathbb{B}_c^n$ of curvature $-c$, the Riemannian metric is conformal: $g_x = \lambda_x^2 g_E$ with $\lambda_x = \frac{2}{1 - c\|x\|^2}$. Because the metric is conformal, **angles at the origin are preserved** between the Euclidean and hyperbolic metrics ($\lambda_0 = 2$ is a constant scalar).

**Cone membership.** The entailment cone $C(u)$ is the set of points $v$ whose direction from the origin lies within angle $\alpha(u)$ of $u$'s direction:
$$v \in C(u) \iff \theta(u, v) \le \alpha(u), \qquad \theta(u, v) = \arccos\!\left(\frac{\langle u, v\rangle}{\|u\|\,\|v\|}\right)$$

**Half-aperture.** The aperture is determined by the requirement that, in hyperbolic space, a cone at depth $d(0, u)$ with apex at the origin should contain the angular sector of a balanced binary subtree rooted at $u$. Using the hyperbolic law of cosines and the relationship between Euclidean norm and hyperbolic distance, the half-aperture is:
$$\alpha(u) = \arcsin\!\left(\sin(\alpha_0)\,\frac{1 - c\|u\|^2}{2\sqrt{c}\,\|u\|}\right)$$

where $\alpha_0$ is the aperture of the root (a learnable parameter). Key properties:
- At the origin ($\|u\| = 0$): the formula diverges, so $C(\text{root})$ is the entire ball (the root entails everything).
- Near the boundary ($\|u\| \to 1/\sqrt{c}$): $\alpha(u) \to 0$, so leaves have degenerate (point-like) cones.
- The function is monotonically decreasing in $\|u\|$, matching the exponential shrinkage of subtree angular width.

**Entailment loss.** For positive pairs $(u, v)$ with $u \succeq v$ and negative pairs $(u, v')$ with $u \not\succeq v'$:
$$\mathcal{L} = \sum_{(u,v) \in \mathcal{P}} \max\!\big(0,\;\theta(u,v) - \alpha(u) + m\big) + \sum_{(u,v') \in \mathcal{N}} \max\!\big(0,\;\alpha(u) - \theta(u,v') + m\big)$$
where $m > 0$ is a margin. The first term pulls descendants inside the cone; the second pushes non-descendants outside.

## Method / architecture
- **Embeddings:** One point per node in $\mathbb{B}_c^n$, initialized near the origin (root) and spread outward.
- **Aperture parameter:** $\alpha_0$ (or equivalently a learnable scalar) shared across all nodes, learned jointly with embeddings.
- **Training:** Minimize the entailment loss with Riemannian SGD/Adam; project gradients to tangent spaces and retract via $\exp$ (or clip into the ball).
- **Inference:** $u \succeq v$ iff $v \in C(u)$, i.e., iff the angle $\theta(u,v) \le \alpha(u)$. This gives a *transitive* partial order (approximately, due to the aperture structure).
- Evaluated on WordNet noun/verb hypernymy reconstruction, outperforming Poincaré embeddings.

## Code
See `code/entailment_cones.py` — implements the cone half-aperture $\alpha(u)$, the angle-at-origin $\theta(u,v)$, cone membership $v \in C(u)$, and a hinge entailment loss trained on a tiny 5-node tree. The demo verifies that after training, true descendants fall inside their ancestors' cones while non-descendants fall outside. Official implementation: https://github.com/dalab/hyperbolic_cones

## Why it matters
Entailment cones were the first to make the entailment relation a *geometric containment* test rather than a distance comparison. This is conceptually cleaner: partial order $\supset$ distance. The depth-dependent aperture is a beautiful application of hyperbolic geometry—specifically, the fact that hyperbolic cone volumes shrink exponentially with depth, which is *exactly* what is needed for tree embeddings. The idea of attaching local geometric objects (cones, balls, half-spaces) to embedding points has influenced subsequent work on box embeddings and order embeddings.

## Reading questions / discussion
1. The cone has its apex at the **origin**, not at $u$. What would change if the apex were at $u$ itself, and why is the origin-apex version more natural for a rooted tree?
2. The aperture formula $\alpha(u) = \arcsin(\sin(\alpha_0)(1 - c\|u\|^2)/(2\sqrt{c}\|u\|))$ is derived for a *balanced binary tree*. How would you modify it for trees with non-constant branching factor?
3. Is the cone containment relation $v \in C(u) \Rightarrow u \succeq v$ actually *transitive*? Under what conditions on the apertures does transitivity hold exactly?
4. Compare entailment cones to Poincaré embeddings: both use the Poincaré ball, but one uses distance and the other uses angular containment. Can you combine both signals in a single loss?