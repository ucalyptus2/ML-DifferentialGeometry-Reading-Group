# Gauge Equivariant Convolutional Networks and the Icosahedral CNN

- **Authors:** Taco S. Cohen, Maurice Weiler, Berkay Kicanaoglu, Max Welling
- **Venue/Year:** ICML 2019
- **arXiv:** https://arxiv.org/abs/1902.04615
- **Category:** gauge

## One-paragraph TL;DR
Generalizes group-equivariant CNNs to **manifolds without a global symmetry** (like the sphere
under an icosahedral discretization) by working with a **$G$-structure** and **gauge**-equivariant
convolutions: features are sections of a vector bundle associated to a principal $G$-bundle, and
convolution kernels are *gauge-equivariant* — they transform correctly when the local frame
("gauge") at each point is changed. Applied to the icosahedron (a regular discretization of $S^2$),
this yields the **Icosahedral CNN**.

## The problem
Cohen & Welling's earlier group-CNNs need a *global* group action. But on a general manifold
(a curved surface, a mesh) there is no global group — only *local* frames that are related by
**gauge transformations**. A naive convolution that adds features at neighbouring points is not
well-defined: vectors at different tangent spaces live in different frames.

## Key idea(s)
- The base space $X$ carries a **$G$-structure**: a reduction of the frame bundle to a subgroup
  $G \subset GL(n)$ (e.g. $SO(2)$ for oriented surfaces). A **gauge** is a choice of local frame.
- Features are **sections of an associated bundle** $E = P \times_\rho V$ for a representation
  $\rho$ of $G$ on $V$ (e.g. scalar fields: $\rho$ trivial; vector fields: $\rho$ the defining rep).
- A kernel $K$ between neighbours is **gauge-equivariant** if under a gauge change $g \in G$ it
  transforms as $K \mapsto \rho(g)\,K\,\rho(g)^{-1}$; equivalently $K$ is an **intertwiner**.

## The mathematics
A gauge-equivariant convolution at $x$ reads
$$ (f * K)(x) = \sum_{y \sim x} \rho\!\big(\tau_{x \leftarrow y}\big)\, f(y), $$
where $\tau_{x \leftarrow y} \in G$ is the **parallel transport** of the local frame from $y$ to
$x$ along the edge. Gauge-equivariance means: if we flip the local frame at $x$ by $g_x$ and at $y$
by $g_y$, the transported feature transforms as $\rho(g_x)\,\rho(\tau)\,\rho(g_y)^{-1} f(y)$, and the
*kernel* is chosen so the whole sum is frame-independent. For $G = SO(2)$, kernels decompose into
angular-frequency modes (like spherical harmonics, but local).

## Method / architecture
On the icosahedron, each face has 6 neighbours and a consistent tangent frame; the gauge group is
$SO(2)$ (rotations of the local frame). The authors build **gauge-equivariant convolution layers**
with kernels that are intertwiners of $SO(2)$ (i.e. band-limited angular filters), pooling on the
icosahedral hierarchy, and a final $G$-invariant pooling.

## Code
`code/gauge_conv.py` — a toy gauge-equivariant layer on a small graph with a **$\mathbb{Z}_2$ gauge**
(sign flips of the local frame): parallel transport along an edge multiplies by $\pm 1$, and the
kernel is the $\mathbb{Z}_2$-intertwiner. Verified to be **gauge-equivariant** under local frame
flips.

## Why it matters
This is the paper that moved equivariance from *global groups* to *local gauges/connections*,
introducing the principal-bundle / parallel-transport language into deep learning and laying the
groundwork for gauge-equivariant mesh CNNs.

## Reading questions / discussion
1. What is a **gauge** in this paper, and why does a "gauge transformation" *not* change the
   physical feature — only its coordinate description?
2. Why is ordinary graph convolution **not** gauge-equivariant for vector features, even though it
   is permutation-equivariant?
3. The kernel must be an **intertwiner** of $G$. For $G = SO(2)$, what does that say about the
   kernel's angular-frequency structure?
