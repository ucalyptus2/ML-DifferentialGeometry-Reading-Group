# Hyperbolic Graph Convolutional Neural Networks

- **Authors:** Ines Chami, Albert Ying, Christopher Ré, Jure Leskovec
- **Venue/Year:** NeurIPS 2019
- **arXiv:** https://arxiv.org/abs/1910.12933
- **Category:** hyperbolic

## One-paragraph TL;DR
HGCN generalizes the Euclidean graph convolutional network (GCN) to the Poincaré ball by performing the neighborhood aggregation step in the **tangent space at the origin** (where it reduces to a standard Euclidean GCN) and then mapping the result back to the ball via the exponential map. The paper introduces a trainable curvature parameter and a feature-to-ball initialization via $\exp_0$, and shows that hyperbolic GCNs outperform Euclidean GCNs on graph datasets with hierarchical structure (disease, airport, Ring-of-Cliques) while remaining competitive on non-hierarchical ones.

## The problem
Graph convolutional networks (Kipf & Welling 2017) aggregate neighbor features via $H^{(\ell+1)} = \sigma(\hat{A} H^{(\ell)} W)$, where $\hat{A}$ is the normalized adjacency. This operates entirely in flat $\mathbb{R}^n$, which is suboptimal when the graph has latent hierarchical or tree-like structure. Prior hyperbolic embedding methods (Poincaré embeddings, hyperbolic neural networks) did not handle *graph-structured* data with shared neighborhoods. The challenge is to combine the message-passing paradigm of GCNs with the non-Euclidean geometry of the Poincaré ball.

## Key idea(s)
- **Aggregate in tangent space.** The Poincaré ball is not closed under addition or scalar multiplication, so one cannot directly average features. Instead, map all node features to the tangent space at the origin via $\log_0$, perform the standard Euclidean GCN aggregation $\hat{A}(\cdot) W$, and map back via $\exp_0$.
- **Trainable curvature.** The sectional curvature $-c$ is a learnable parameter, allowing the model to interpolate between Euclidean ($c \to 0$) and maximally hyperbolic geometry.
- **Feature-to-ball initialization.** Real-valued input features are mapped into the ball via $\exp_0$ at the start, so the first layer receives ball-valued inputs.
- **Fermi-Dirac decoder.** Link prediction uses a probabilistic decoder based on hyperbolic distance rather than Euclidean inner product.

## The mathematics
Let $\mathbb{B}_c^n$ be the Poincaré ball with curvature $-c$, and let $\exp_0$, $\log_0$ denote the exponential and logarithmic maps at the origin $o$:

$$\exp_0(v) = \tanh(\sqrt{c}\,\|v\|)\,\frac{v}{\sqrt{c}\,\|v\|}, \qquad \log_0(y) = \operatorname{artanh}(\sqrt{c}\,\|y\|)\,\frac{y}{\sqrt{c}\,\|y\|}$$

**One HGCN layer.** Given ball-valued node features $H^{(\ell)} \in \mathbb{B}_c^{n \times d}$:

1. **Tangent projection:** $H_{\text{tan}}^{(\ell)} = \log_0(H^{(\ell)}) \in T_o\mathbb{B}_c^n \cong \mathbb{R}^d$
2. **Euclidean GCN aggregation:** $H_{\text{tan}}^{(\ell+1)} = \sigma\!\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}\, H_{\text{tan}}^{(\ell)}\, W^{(\ell)}\right)$
3. **Ball projection:** $H^{(\ell+1)} = \exp_0\!\left(H_{\text{tan}}^{(\ell+1)}\right)$

where $\hat{A} = A + I_n$ is the adjacency with self-loops and $\hat{D}$ is the degree matrix. The aggregation in step 2 is identical to a standard GCN because $\log_0$ identifies the tangent space at the origin with Euclidean space. The nonlinearity $\sigma$ is applied in the tangent space (before the $\exp_0$ map), which is equivalent to the hyperbolic activation $\sigma_\otimes(x) = \exp_0(\sigma(\log_0(x)))$ applied after the aggregation.

**Fermi-Dirac decoder** for link prediction:
$$p(e_{ij} = 1) = \left[e^{-(d(x_i, x_j) - r)^2 / \tau} + 1\right]^{-1}$$
where $d(\cdot,\cdot)$ is the hyperbolic distance and $r, \tau$ are trainable parameters.

## Method / architecture
- **Input:** Graph adjacency $A$ + node features $X \in \mathbb{R}^{n \times d}$. Map $X$ into the ball: $H^{(0)} = \exp_0(X)$.
- **Stack $L$ HGCN layers** as above, with trainable weight matrices $W^{(\ell)}$ and shared curvature $c$.
- **Output:** Ball-valued embeddings $H^{(L)}$; decode edges with the Fermi-Dirac decoder for link prediction, or map to tangent space for node classification.
- **Training:** Riemannian Adam with exponential-map retraction and ball projection.
- The paper also introduces a **hyperbolic feature attention** that scales the conformal factor per layer.

## Code
See `code/hgcn_layer.py` — implements batch $\log_0$/$\exp_0$, the normalized adjacency $\hat{A}$, one `HGNNLayer` (tangent-space GCN aggregation + ball projection), and a 2-layer forward/backward demo on a toy graph verifying outputs stay in the ball and gradients flow. Official implementation: https://github.com/HazyResearch/hgcn

## Why it matters
HGCN was the first graph neural network to operate natively in hyperbolic space, demonstrating that the tangent-space aggregation trick makes hyperbolic GNNs both practical and competitive. The key insight—"aggregate in the tangent space, map back to the manifold"—became the standard pattern for all subsequent Riemannian GNNs, including those on the SPD manifold and product manifolds. The trainable curvature idea is also widely adopted: it lets the model *discover* the right geometry rather than committing a priori.

## Reading questions / discussion
1. The aggregation happens in $T_o\mathbb{B}_c^n$, the tangent space at the *origin*. What geometric information is lost by not aggregating in each node's own tangent space (which would require parallel transport)?
2. The curvature $c$ is trainable. What happens to the Poincaré ball model as $c \to 0$? Does HGCN reduce exactly to a Euclidean GCN in that limit?
3. The Fermi-Dirac decoder depends on hyperbolic *distance* rather than inner product. Why is distance a better decoder for hierarchical graphs, and what failure mode of the inner-product decoder does it fix?
4. For a graph that is a single long chain (a path), would you expect HGCN to outperform GCN? Why or why not?
