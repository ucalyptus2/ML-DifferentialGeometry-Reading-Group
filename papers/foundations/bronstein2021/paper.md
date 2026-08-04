# Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges

- **Authors:** Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković
- **Venue/Year:** arXiv, 2021 (subsequently a Nature MI paper, 2021)
- **arXiv:** https://arxiv.org/abs/2104.13478
- **Category:** foundations

## One-paragraph TL;DR
This is the "Blueprints for Geometric Deep Learning" position paper that makes precise the informal $\psi \circ P$ template of Bronstein et al. (2017). It formulates geometric DL as learning functions on domains that are *sets of points equipped with a symmetry group action*, and derives, from first principles, that every class of architecture is characterized by two categorical choices: **(1) the type of feature field** (a scalar field / a group representation) and **(2) the type of equivariance** the model is required to satisfy. Stacking these choices yields a 2-category whose objects are the familiar domains — grids, groups, graphs, geodesics (manifolds), gauges (fields). The paper's central message is that "the choice of architecture is a choice of geometry": once you fix a *group* and a *feature type*, the architecture is essentially forced.

## The problem
Deep learning architectures look bewilderingly diverse, but the paper observes that all of them process data living on "irregular domains" where the natural symmetries differ. Why does a CNN work on images but a GNN on graphs and an equivariant net on molecules — and how are these related? The difficulty is that the notion of *what should be equivariant* is subtle:
- On a **grid**, the symmetry is the translation group; features are indexed by pixels which form a group itself.
- On a **graph / manifold**, there is no global symmetry — only *local* gauge-like symmetries (re-labelling of neighbours, choice of local coordinates).
- On a **field**, the group acts *in the space of features* (gauge symmetry), not on positions.

To unify them one must separate the *feature space* from the *position space*, classify the possible equivariances of each, and build architectures from invariant/equivariant building blocks.

## Key idea(s)
- **The two axes**: every geometric DL architecture is determined by (1) the **feature field type** — a *positive* (scalar, $\mathbb{R}$-valued) field vs an *equivariant* (group- or vector-valued) field — and (2) the **aggregation type** — *neighbourhood aggregate* (sum/mean/max over neighbours) vs *gauge-dependent* message passing (needs parallel transport).
- **The 1-categorical Blueprint** ("kinematics"): for kinematic tasks (where features only need to be *carried* along the domain) it suffices to work with **positive scalar fields** plus a **neighbour aggregate**, giving architectures like CNNs, GNNs, and the 2017 geometric CNN's.
- **The 2-categorical Blueprint** ("dynamics"): for tasks where information must be *transported* between different points (non-kinematic), features become **equivariant fields** and messages must be compared via **parallel transport**, leading to gauge-equivariant kernels (Cohen et al. 2019; De Haan et al. 2020).
- **"The choice of architecture is a choice of geometry"**: fix the group $G$ and the feature representation $\rho$; the most general equivariant layer satisfying the blueprint is essentially unique up to choices of kernel/aggregator.

## The mathematics

Let data live on a **domain** $\Omega = (S, \mathcal G)$: a set of points $S$ together with a group $\mathcal G$ acting on it (positions). A **feature space** is a set $\mathcal F$ on which a related group acts (features). A feature field assigns a feature to each point,
$$f: S \to \mathcal F,$$
and the architecture must be **equivariant**: for a symmetry $g \in \mathcal G$ acting as $s \mapsto g \cdot s$ on points and $\rho(g)$ on features,
$$(\Phi f)(g\cdot s) = \rho(g)\,( \Phi f)(s).$$
When $\rho = 1$ (trivial representation) we recover invariant scalar fields.

**1-categorical (kinematic) blueprint.** Features are **positive scalar fields** on $S$, and the layer is
$$(\Phi f)(i) = \phi\Big(f(i),\ \bigoplus_{j \in \mathcal N(i)} \rho_\text{agg} \, \psi(f(i), f(j))\biggr)$$
where $\mathcal N(i)$ is a neighbour set, $\oplus$ a permutation-invariant aggregator (sum/mean/max), and $\psi,\phi$ are learned maps. For an image, $\mathcal N(i)$ is the $k\times k$ pixel patch and equivariance under the translation group restores standard convolution.

**2-categorical (dynamic) blueprint.** When the task requires transporting data between points $i,j$, scalar aggregation loses information because the *relative direction* matters. One must use **vectors/equivariant features** and compare them via **parallel transport** along a path from $j$ to $i$:
$$\mathrm{transp}_{\gamma}(\cdot): T_j M \to T_i M$$
with
$$(\Phi f)(i) = \phi\Big(f(i),\ \bigoplus_{j \in \mathcal N(i)} K\big(g_{ji}^{-1}\big)\, \mathrm{transp}_{j\to i}\big(\psi(f(i),f(j))\big)\Big),$$
where $g_{ji} \in G$ is the *gauge element* (holonomy) relating the frames at $j$ and $i$, and $K$ is a gauge-equivariant kernel. The transition map $g_{ji}$ is analogous to a connection-form / Christoffel symbol: it depends on the chosen local frames and must transform consistently under a change of gauge.

**Categories.** Composing these choices gives a *2-category*: objects are domains; 1-morphisms are equivariant maps (blueprints); 2-morphisms are gauge transformations.

## Method / architecture
The paper does not train any model; its "method" is the categorical derivation. It shows how each well-known architecture appears as a specific instance of the blueprints:

- **Grids (CNNs)**: $\Omega$ = $\mathbb{Z}^n$ with the translation group $\mathcal G = \mathbb{Z}^n$; positive feature fields; neighbour aggregate over pixel patches → *translation-equivariant convolution*.
- **Groups (group-equivariant CNNs / steerable nets)**: $\Omega$ = a group $G$ (e.g. $E(2)$ or $SO(3)$); equivariant feature fields; convolution is group convolution.
- **Graphs (GNNs)**: $\Omega$ = a graph, $\mathcal G$ = trivial (no global symmetry), only permutation of nodes; scalar message passing is the 1-categorical blueprint.
- **Geodesics (manifolds / GCNN)**: $\Omega$ = a Riemannian manifold; local geodesic polar patches; positive features + neighbour aggregate (Masci et al. 2015).
- **Gauges (gauge CNNs)**: $\Omega$ = a manifold with a $G$-structure; *equivariant vector fields* transported with parallel transport; this is the 2-categorical blueprint and the most expressive.

The key conceptual move is the **feature/anchor distinction**: the *anchor space* (where features live) need not equal the *index space* (positions). Classifying equivariance on each produces a precise recipe: pick a group, pick a representation, pick an aggregation; the result is a principled architecture with no free choice about whether it is equivariant.

`code/` implements the two blueprints directly: a 1-categorical scalar-message-passing layer and a 2-categorical gauge-equivariant layer with parallel transport on a small manifold patch.

## Code
See `code/` in this directory: `blueprints.py` contains (i) `ScalarMessagePassing` — the 1-categorical blueprint (positive scalar fields + neighbour aggregate, invariant under relabelling), and (ii) `GaugeEquivariantLayer` — the 2-categorical blueprint that transports equivariant features between neighbours with a transition map $g_{ji}$ and applies a kernel that is equivariant under the structure group $G=SO(2)$ (rotation). The `__main__` demo (a) checks scalar-message-passing invariance under relabelling, and (b) checks that the gauge layer is *equivariant*: rotating all local frames (a gauge transformation) transforms the output by the same rotation, up to floating point error. Original implementation, not the authors' code.

## Why it matters
This is the definitive framework paper of the subfield. It turns "design a neural net for data X" into "choose a geometry": the group, the feature field type, and the aggregation are the only design decisions, and they force the architecture. It also clearly isolates *when* parallel transport (the deepest diff-geometric notion) becomes necessary — exactly for non-kinematic (dynamic) tasks — which motivates the gauge/transport papers later in this reading group, and it introduces the 2-categorical perspective that unifies graphs, manifolds, and gauge theories under one calculus.

## Reading questions / discussion
1. Give an example of a task that *requires* the 2-categorical blueprint (parallel transport) and one for which the 1-categorical blueprint suffices. What property of the task distinguishes them?
2. In the gauge layer, why does the transition map $g_{ji}$ depend on the choice of local frames, and how does the kernel $K$ compensate so the whole layer is equivariant?
3. The paper claims "the choice of architecture is a choice of geometry." Where does this reduction come from (what is fixed once you fix $G$ and $\rho$)?
4. How does the 1-categorical blueprint specialize to an ordinary 2D CNN, and where does the *kernel sharing* of the CNN come from in this language?

