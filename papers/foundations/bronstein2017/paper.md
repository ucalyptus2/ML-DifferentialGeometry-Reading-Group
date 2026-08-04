# Geometric deep learning: going beyond Euclidean data

- **Authors:** Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, Pierre Vandergheynst
- **Venue/Year:** IEEE Signal Processing Magazine, 2017
- **arXiv:** https://arxiv.org/abs/1611.08097
- **Category:** foundations

## One-paragraph TL;DR
This position paper asks *why* deep learning works in Euclidean space and *what* structure makes convolution possible, then generalizes those ingredients to arbitrary geometric domains (grids, groups, graphs, manifolds). It argues that the two reusable building blocks of any convolutional architecture are a *local aggregator* $P$ (how a node pools information from its neighbours) and a *symmetric function* $\psi$ (how that pooled information is recombined) — both borrowed from the language of harmonic analysis on domains. The paper recasts CNNs, graph convnets, and geodesic convnets as instances of one unified template, making it the intellectual seed of the later "blueprint" of Bronstein et al. (2021).

## The problem
Classical deep learning operates on Euclidean grids: images, audio, voxels. The success of convolution there relies on three hidden assumptions — *translation equivariance* (shifting the input shifts the features), *local connectivity* (a kernel of fixed receptive field), and *parameter sharing*. These assumptions fail on non-Euclidean data: social networks, 3D surfaces, molecules, brain connectomes have no canonical global coordinate system, no translation group, and irregular neighbourhoods of varying size/density. Naively flattening such data destroys its intrinsic structure and either throws away geometry or forces wasteful gridification. The core difficulty is that **there is no intrinsic coordinate system** on a manifold or graph that plays the role of $\mathbb{R}^n$ for images.

## Key idea(s)
- **A common mathematical speech for all domains**: grids, groups, graphs, and manifolds are all instances of *domains* equipped with a local neighbourhood structure, so a single template covers them.
- **Convolution as $\psi \circ P$**: any convolutional layer decomposes into a *local aggregator* $P$ (permutation-invariant pooling over neighbours) followed by a *symmetric function* $\psi$ (a learnable combination of the pooled statistics). This is the paper's famous diagram.
- **Two families of architectures**: *spatial* methods (aggregate in the domain: spectral/heat-kernel or template-based patches like geodesic polar patches) and *spectral* methods (filter in the frequency domain of a Laplacian defined on the domain).

## The mathematics

The paper frames data as functions on a *domain* $\Omega$ with a measure. On a differentiable manifold $M$ the natural objects are:

- the **tangent space** $T_x M$ at $x \in M$;
- a **Riemannian metric** $g_x: T_x M \times T_x M \to \mathbb{R}$, giving local inner products and hence a distance
  $$d(x,x') = \inf_{\gamma}\int \sqrt{g(\dot\gamma,\dot\gamma)}\,dt$$
  over curves $\gamma$ joining $x,x'$ (geodesic distance);
- the **Laplace–Beltrami operator** $\Delta_M = -\operatorname{div}_g \nabla_g$, the intrinsic generalization of the Euclidean Laplacian, with spectral decomposition
  $$\Delta_M \varphi_k = \lambda_k \varphi_k, \qquad \lambda_0 \le \lambda_1 \le \cdots.$$

For a **group** $G$ acting on the domain, a key notion is **equivariance**: a map $f$ between feature spaces satisfies
$$f(g \cdot x) = \rho(g) \cdot f(x),$$
where $\rho$ is the group representation; when $\rho = \mathrm{Id}$ we get invariance $f(g\cdot x)=f(x)$. Convolution on a group uses the group (Haar) measure and translation by group elements:
$$(f \ast h)(g) = \int_G f(g')\, h(g^{-1}g')\,dg'.$$

The **aggregator** $P$ over a neighbourhood $\mathcal N(x)$ is any permutation-invariant statistic of features $f(y)$ for $y \in \mathcal N(x)$, e.g.
$$P_{f}(\mathcal N(x)) = \Big\{\max_{y \in \mathcal N(x)} f(y),\ \text{mean}_{y \in \mathcal N(x)} f(y),\ \text{sum}_{y \in \mathcal N(x)} f(y),\ \text{LSTM}\circ\{f(y)\}_{y \in \mathcal N(x)}\Big\}.$$
Permutation invariance encodes *symmetry under re-labelling of neighbours*, which is the correct inductive bias once global translation symmetry is gone. A generic convolution layer is then
$$h(x) = \psi\Big(P_{f}(\mathcal N(x))\Big),$$
with learnable (possibly multi-layer MLP) $\psi$.



## Method / architecture
The paper does not propose one network but a *taxonomy* cataloguing how existing convnets instantiate $\psi \circ P$:

1. **Euclidean CNNs**: $P$ = pooling over the $k\times k$ pixel patch, $\psi$ = the learned kernel summed over the patch (translation equivariant).
2. **Spectral / graph CNNs (Bruna et al. 2013; Defferrard et al. 2016; Kipf & Welling 2017)**: define a graph Laplacian $L = D - A$ (or normalized version) and filter in its eigenbasis, $x \mapsto \phi(L)x$ where $\phi$ is a spectral mask; $P$ = sum over neighbours weighted by $A$, $\psi$ = shared weight matrix. Localizing via Chebyshev polynomials recovers spatial neighbourhoods.
3. **Geodesic convnets (Masci et al. 2015)**: build local patches in geodesic polar coordinates, then $P$ = pool over patch bins (histogram), $\psi$ = kernel.
4. **Bruna et al. (2014) manifold scattering / hierarchical**: cascade of $\psi \circ P$ blocks with wavelets.

`code/` implements the generic $\psi \circ P$ aggregator template directly.

## Code
See `code/` in this directory: a self-contained PyTorch implementation of the two building blocks — a permutation-invariant aggregator `P` (mean/max/sum over neighbourhoods) and a symmetric function `\psi` — composed into a general `PsiComposeP` layer. A `__main__` demo builds a random graph, feeds node features, and checks (i) permutation equivariance/invariance of the pooled output and (ii) learning on a small graph signal. No official repository is needed; this is an original minimal implementation of the paper's central template.

## Why it matters
This is the "unifying moment" paper for ML × differential geometry: it shows that convolutions are not an accident of pixels but the answer to "what is translation equivariance on my domain?", and it supplies the exact two ingredients ($\psi, P$) that the 2021 follow-up elevates into a formal Blueprint with gauge theory. It also clearly separates the **choice of geometry** (how neighbourhoods/aggregation are defined) from the **learning machinery**, so every later result in this reading group — Laplacians, geodesic patches, parallel transport, gauge equivariance — plugs into this same socket.

## Reading questions / discussion
1. Where exactly does the "symmetric function" $\psi$ buy us equivariance, and what would break if aggregation were not permutation-invariant?
2. Spectral vs spatial methods: which better respects local geometry, and what are the costs (eigen-decomposition cost, non-Euclidean interpolation)?
3. How does the $\psi\circ P$ template generalize to the 1-categorical Blueprint (positive vs equivariant features) in Bronstein et al. 2021?
4. Translation equivariance is a symmetry of the *domain*; what symmetry of the *feature space* does each building block (mean vs max vs sum) presuppose?
