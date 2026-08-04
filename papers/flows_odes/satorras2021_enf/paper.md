# E(n) Equivariant Normalizing Flows

- **Authors:** Victor Garcia Satorras, Emiel Hoogeboom, Fabian Fuchs, Ingmar Posner, Max Welling
- **Venue/Year:** NeurIPS 2021
- **arXiv:** https://arxiv.org/abs/2105.09016
- **Category:** flows_odes

## One-paragraph TL;DR
E(n)-GNFs build a continuous normalizing flow (à la FFJORD) whose vector field is parameterized by
an **E(n)-equivariant graph neural network** (EGNN) rather than an arbitrary MLP. Because every
building block of the flow — the vector field, the integration scheme, and the base density — is
equivariant/invariant under rotation, translation, and permutation, the *entire generative model*
of a point cloud (e.g. 3D molecule coordinates) inherits those symmetries exactly, not
approximately, so the model never has to "learn" that a rotated molecule is the same molecule.

## The problem
Generative models of point clouds/molecules must assign the *same* likelihood to any rotation,
translation, or atom-relabelling of a configuration — the physics doesn't care about your
coordinate frame. A CNF with a generic (non-equivariant) vector field can approximate this
symmetry from data, but never exactly; every rotation of the input can get a slightly different
likelihood, and the model needs vastly more data to learn what could instead be built in.

## Key idea(s)
- Use an **EGNN layer** (Satorras et al. 2021, `papers/group_equivariance/satorras2021_egnn`) as
  the CNF's vector field $f_\theta(x,h,t)$: node features $h$ stay invariant, coordinate updates
  $x$ are computed only from **relative, invariant** quantities (pairwise distances), so the whole
  layer commutes with any $E(n)$ transform and any permutation of nodes.
- Stack the *same equivariant layer* across the ODE's continuous "time" (depth) dimension — since
  composing equivariant maps yields an equivariant map, the whole flow (many Euler/RK steps of the
  EGNN vector field) is equivariant, not just one layer.
- Use a rotation/translation/permutation-**invariant** base density (e.g. an isotropic Gaussian on
  the zero-center-of-mass subspace) so the full generative density $p(x) = p_0(f^{-1}(x))\cdot
  |\det J|^{-1}$ is invariant end-to-end (the Jacobian log-det term is itself invariant since it
  only depends on invariant quantities computed by the EGNN).

## The mathematics
A map $\Phi:X\to X$ is **$E(n)$-equivariant** if $\Phi(gx) = g\Phi(x)$ for every
$g=(R,t)\in E(n)$ acting as $gx = Rx+t$ (rotation + translation), and **permutation-equivariant**
if $\Phi(P x) = P\Phi(x)$ for any permutation matrix $P$ applied to the node ordering. Composing
equivariant maps stays equivariant: if $\Phi_1,\Phi_2$ are both $G$-equivariant, then
$\Phi_2\circ\Phi_1(gx) = \Phi_2(g\Phi_1(x)) = g\,\Phi_2(\Phi_1(x))$ — so a flow built entirely from
equivariant Euler/RK steps of an equivariant vector field is equivariant at *every* intermediate
time $t$, not merely at $t=0$ and $t=1$. This is the same "blueprint" as lesson 06 of this
tutorial: aggregate with an equivariant $\mathcal{P}$, and the whole architecture inherits the
symmetry group's action.

## Method / architecture
`dx/dt = EGNN(x,h,t)`, integrated with any ODE solver (RK4, adaptive), with the FFJORD trace trick
for the (already rotation/translation-invariant, since it depends only on invariant pairwise
features) log-density term. Node features $h$ are typically atom types (already permutation-indexed
consistently with $x$); center-of-mass is projected out so the base density is well-defined on the
translation-invariant subspace.

## Code
See `code/enf_flow_equivariance.py` — extends
`papers/group_equivariance/satorras2021_egnn/code/egnn_layer.py`'s single-layer check to a
**multi-step Euler-integrated flow** of an EGNN vector field, and verifies the *whole trajectory*
(not just one layer) commutes with rotation + translation + permutation.

## Why it matters
This is the direct synthesis point of the reading group: it combines Neural-ODE/CNF machinery
(Chen et al., FFJORD) with $E(n)$-equivariant architectures (Satorras et al. EGNN) to get a
generative model whose symmetry guarantees are exact by construction — the geometric deep learning
blueprint (lesson 06) applied to generative modeling.

## Reading questions / discussion
1. Why does composing equivariant layers preserve equivariance, but composing an equivariant layer
   with a *non*-equivariant one (e.g. a plain MLP mixing raw coordinates) generally destroy it?
2. The base density must be invariant under $E(n)$ and permutations. Why is an isotropic Gaussian
   on the *zero-center-of-mass* subspace a natural choice, and what breaks if you used a generic
   (non-isotropic, non-zero-mean) Gaussian instead?
3. FFJORD's Hutchinson trace estimator samples a random $\epsilon$. Does randomizing $\epsilon$
   risk breaking the flow's equivariance (since $\epsilon$ itself isn't tied to the data's
   symmetry)? How would you reason about this?
