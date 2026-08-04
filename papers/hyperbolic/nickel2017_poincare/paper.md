# Poincaré Embeddings for Learning Hierarchical Representations

- **Authors:** Maximilian Nickel, Douwe Kiela
- **Venue/Year:** NeurIPS 2017
- **arXiv:** https://arxiv.org/abs/1705.08039
- **Category:** hyperbolic

## One-paragraph TL;DR
The authors embed the nodes of a *hierarchy* (e.g. WordNet's noun hypernymy graph) as points in
the **Poincaré ball** — a model of hyperbolic space — instead of flat ℝⁿ. Because hyperbolic space
grows exponentially (like a tree), distances in the ball align with *ancestor–descendant* depth, so
the embedding respects hierarchy with far fewer dimensions than Euclidean embedding, and yields
state-of-the-art reconstruction + link-prediction on WordNet.

## The problem
Discrete hierarchical data (taxonomies, ontologies, DAGs/trees) is poorly modelled by Euclidean
embeddings: the number of leaves at depth $d$ grows exponentially, so a faithful Euclidean embedding
needs exponentially many dimensions. Hyperbolic space has *constant negative curvature*, whose
volume grows exponentially with radius, so it is the "natural" ambient space for trees.

## Key idea(s)
- Use the **Poincaré ball** model $\mathbb{B}^n = \{x:\|x\|<1\}$ with the Riemannian metric
  $g_x = \lambda_x^2 I$, $\lambda_x = \frac{2}{1-\|x\|^2}$.
- Distance
  $$d(x,y) = \operatorname{arcosh}\!\Big(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\Big)$$
  grows as points approach the boundary — a proxy for "deeper in the tree".
- Optimize with Riemannian SGD: project the Euclidean gradient onto the tangent space, retract via
  the **exponential map**, and *project back* onto the ball (renormalize if it escapes).

## The mathematics
The gradient of $d(x,y)$ w.r.t. $x$ in the Poincaré ball, projected to the tangent space at $x$, is
$$\nabla_x^{R} d(x,y) = \frac{g_x}{\|\nabla_x d\|^2_{g_x}}\,\nabla_x d \;=\; \lambda_x^2\,\nabla_x d$$
(up to the conformal factor and a normalization for numerical stability). Each step retracts via
$x \leftarrow \exp_x(-\eta\,\nabla_x^R d)$, or a cheaper surrogate: clip into the ball.

## Method / architecture
- Negative-sampling loss over the hierarchy: positive pairs $(u,v)$ with $u \succeq v$ pushed together
  in Poincaré distance; random negatives pushed apart.
- The "distance" is made *asymmetric* (small weight for the norm of $u$) to encode that the root sits
  near the centre and leaves near the boundary.

## Code
See `code/poincare_embeddings.py` — implements `poincare_dist`, the Riemannian-gradient step, and a
tiny hierarchy reconstruction demo.

## Why it matters
This paper sparked the whole line of *hyperbolic representation learning*: it showed that a single
geometric prior (negative curvature) can encode hierarchy with dramatically fewer parameters. It is
the direct ancestor of the hyperbolic neural nets, GCNs, and entailment cones that follow.

## Reading questions / discussion
1. Why does the *distance* (not the embedding) encode the hierarchy? What happens to a parent and a
   deep descendant as both move toward the boundary?
2. The Poincaré metric is conformal: only a *scalar* $\lambda_x$ differs from the Euclidean inner
   product. Where exactly does the exponential-growth volume come from?
3. Compare to embedding in flat space with a learned "depth" feature — what does the hyperbolic
   model get for free?
