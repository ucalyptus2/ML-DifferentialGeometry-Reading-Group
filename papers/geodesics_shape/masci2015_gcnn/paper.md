# Geodesic Convolutional Neural Networks on Riemannian Manifolds

- **Authors:** Jonathan Masci, Davide Boscaini, Michael M. Bronstein, Pierre Vandergheynst
- **Venue/Year:** IEEE ICCV Workshops, 2015
- **arXiv:** https://arxiv.org/abs/1501.06297
- **Category:** geodesics_shape

## One-paragraph TL;DR
This paper builds convolutional networks directly on a Riemannian manifold (a triangulated surface) by replacing the flat, regularly-sampled pixel patch of a CNN with a **geodesic polar patch**: around each center point, the surface is charted in *geodesic polar coordinates* $(r, \theta)$ where $r$ is geodesic distance and $\theta$ is the geodesic direction relative to a reference point, transported locally. A careful feature (a histogram over $(r,\theta)$ bins) makes patches *intrinsic* — invariant to how the surface is rigidly placed or how it bends isometrically — so a shared convolutional kernel works at every point regardless of parametrization. This is one of the first "GCNN"s and a direct ancestor of the gauge-equivariant methods in this reading group.

## The problem
A standard 2D CNN relies on images being regular grids: every pixel patch has the same shape, size, and an unambiguous up/down/left/right. Surfaces (faces, lungs, ears, geometric objects) have **no such global coordinates**; they are given as arbitrary triangle meshes. Two obstacles arise:
1. **No canonical patch shape** — the local neighbourhood of a vertex has variable size and connectivity, so a fixed template cannot be overlaid directly.
2. **No canonical direction** ("up") — even if we chart a patch, we must choose a local reference direction, and the network must either be equivariant to that choice or consistent so that weights remain shareable.

If these are ignored, the learned filters become dependent on the arbitrary mesh parametrization and do not *transfer* between different triangulations of the same surface — a fatal flaw for shape analysis. The paper's answer is to describe each patch in coordinates that are intrinsic to the *geometry* (geodesic distance and geodesic direction), which survive re-meshing and isometric deformation.

## Key idea(s)
- **Geodesic polar coordinates (patches)**: a local chart around a center $c$ given by $(r, \theta)$ where $r = d_M(c, p)$ is the geodesic distance on the manifold $M$ and $\theta$ is the angle the geodesic from $c$ to $p$ makes with a fixed reference direction — computed via parallel transport / local frames.
- **Anisotropic local kernels with a common pattern**: the continuous version of a convolutional kernel becomes a function $\Xi(\Theta)$ of the geodesic polar patch, integrated over the patch, so that *the same kernel pattern* is re-parameterized at each point.
- **Patch as a vectorized histogram**: discretize $(r,\theta)\in[0,R]\times[0,2\pi)$ into bins and form a histogram per point; the network reads the histogram. Doing so with *weighted* geodesic kernels gives GCNNs (Boscaini et al. 2015).
- **Patch deformation difference**: features can be built on the histogram of a *single* surface (shape context of the patch) or of the *difference* between two aligned surfaces (for matching/detection).

## The mathematics

Let $M$ be a connected compact Riemannian manifold ("shape") with metric $g$, and let $x: M \to \mathbb{R}^3$ be its embedding. The **geodesic distance** is
$$d_M(c, p) = \inf_{\gamma} \int_0^1 \sqrt{\langle \dot\gamma(t), \dot\gamma(t)\rangle_{g}}\,dt, \qquad \gamma(0)=c,\ \gamma(1)=p,$$
the infimum over curves in $M$; on a triangulated mesh it is computed by Dijkstra on the weighted graph whose edge weights are the Euclidean lengths of the triangles' edges.

Fix a center $c$ and a **reference direction** $v \in T_c M$ (e.g. the tangent direction toward a reference point $p^*$). In geodesic polar coordinates a point $p$ in a small normal neighbourhood is described by
$$(r, \theta) = \Big(d_M(c,p),\ \angle\big(\exp_c^{-1}(p),\ v\big)\Big),$$
where $\exp_c: T_c M \to M$ is the Riemannian exponential and $\exp_c^{-1}(p)$ is the tangent vector (log map) of the shortest geodesic from $c$ to $p$. On a mesh the log map is approximated by projecting $p-c$ onto the tangent plane $T_c M$; the direction is then
$$\theta = \operatorname{atan2}\big(\langle t, u_2\rangle,\ \langle t, u_1\rangle\big), \qquad t = \Pi_{T_c M}(p - c)$$
for an orthonormal tangent basis $(u_1,u_2)$.

The **GCNN layer** integrates an anisotropic kernel over the geodesic polar patch:
$$f(c) = \int_{-\pi}^{\pi}\int_0^{R} \Xi(\Theta)\,\omega(r,\theta)\, d\theta\, dr,$$
where $\omega$ accounts for the surface measure in polar coordinates and $\Theta$ is the patch (collection of geodesic polar coordinates plus a feature value). Discretized over the polar histogram this becomes a **spatial convolution over the $(r,\theta)$ bins**:
$$f_i = \sum_{j=1}^{N} w_j\, h_{i,j},$$
with a single shared weight vector $w$ and $h_{i,j}$ the $j$-th bin value of point $i$'s patch histogram. Because $(r,\theta)$ are intrinsic, $w$ is *shared across all points* and *invariant to rigid and isometric transformations*.

## Method / architecture
1. **Input mesh** with per-vertex features (e.g. HKS, or point coordinates).
2. **Geodesic polar patch extraction**: for each center, compute $r$ (Dijkstra) and $\theta$ (tangent projection + reference direction) for vertices within radius $R$.
3. **Patch vectorization**: accumulate the per-bin statistic into a histogram vector $h_i \in \mathbb{R}^{N_r \cdot N_\theta}$.
4. **Convolution**: apply the shared kernel $w$ over bins (optionally two scales / anisotropic $\Xi$).
5. Stack several such layers + nonlinearity + spatial pooling to build a shallow GCNN for per-point tasks (classification, correspondence, segmentation).

`code/` implements the geodesic polar patch extraction and the $(r,\theta)$ histogram convolution on a synthetic sphere mesh, and trains it on a directional shape signal.

## Code
See `code/` in this directory: `geodesic_conv.py` builds a UV-sphere Riemannian mesh (vertices + faces), computes the **geodesic distance matrix** with Dijkstra on the mesh graph, constructs **geodesic polar patches** $(r,\theta)$ at every vertex using tangent-plane projection and a reference direction, forms **polar histograms**, and runs a tiny **polar-convolution** layer (PyTorch) trained to detect the *direction* of a feature blob — demonstrating that the anisotropic $(\theta)$ component carries signal. `__main__` also sanity-checks computed geodesic distances against the analytic great-circle distance. Original implementation; the authors' GCNN code is not public at a canonical URL.

## Why it matters
GCNN is the bridge between mesh signal processing and deep learning on curved shapes. Its key contributions — intrinsic patches, geodesic polar coordinates, shared anisotropic kernels — directly anticipate:
- the *geometry/feature* split formalized by the Blueprint paper (Bronstein et al. 2021);
- gauge-equivariant nets (Cohen et al. 2019), which replace the hand-picked reference direction with an explicit connection/parallel transport;
- surface networks / diffusion-based shape analysis (Kostrikov et al. 2018) in this folder.
Reading it grounds why "patches must be intrinsic" is the real lesson, not the specific mesh tricks.

## Reading questions / discussion
1. Where in the patch construction does the arbitrary *choice of reference direction* enter, and what happens if two nearby points choose inconsistent directions?
2. Why does the histogram-based kernel transfer across re-meshing and isometric bending, while a raw-coordinate kernel would not?
3. How would you replace the heuristic reference direction with a parallel-transport/connection so the network becomes properly gauge-equivariant (the 2-categorical blueprint)?
4. `exp_c^{-1}` is used here only through a tangent-plane projection; what breaks for large patches on high-curvature regions, and how do multi-scale / windowed geodesic kernels help?

