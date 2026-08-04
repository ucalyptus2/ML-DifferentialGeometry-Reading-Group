# 03 — Groups & equivariance

## Groups and group actions

A **group** $(G, \cdot)$ is a set with an associative operation, an identity $e$, and inverses.
It **acts** on a set/space $X$ via
$$\rho: G \to \operatorname{Aut}(X),\qquad (g,h)\mapsto \rho(g)\circ\rho(h)=\rho(gh).$$

Real examples:
- the **translation group** $\mathbb{R}^2$ and **rotation group** $SO(2)$ acting on the plane,
- the **p4** group (rotations by $90^\circ$ + translations) acting on $\mathbb{Z}^2$ grids,
- the **E(n) group** (rotations/reflections $O(n)$, translations, permutations) acting on point clouds,
- $SO(3)$ acting on the sphere $S^2$ and on 3D objects.

A smooth group is a **Lie group**; its tangent space at the identity is the **Lie algebra**
$\mathfrak{g}$ (e.g. skew-symmetric matrices for $SO(n)$). The exponential map
$\exp:\mathfrak{g}\to G$ bridges them — precisely what the tutorial code uses for the rotation group.

## Equivariance

A function $f: X \to Y$ is **$G$-equivariant** if, for a group action $\rho_X$ on $X$ and $\rho_Y$ on $Y$,
$$f(\rho_X(g)\, x) = \rho_Y(g)\, f(x).$$
It is **$G$-invariant** when $\rho_Y$ is trivial, i.e. $f(\rho_X(g)x)=f(x)$.

### Example: a rotation-equivariant 90° filter
If we rotate the input image by $g$ and the network's first layer by $g$, and outputs match after
$g$, the layer is equivariant. Group-equivariant CNNs **lift** the input to the group
$\mathbb{R}^2 \rtimes G$ and convolve there.

## Why this matters for ML

- **Data efficiency & generalization**: an equivariant net never has to *learn* a symmetry; it gets
  it for free, so it sees far fewer effective examples.
- **Correct inductive bias**: for 3D/data on spheres or point clouds, invariance under $E(n)$,
  $SO(3)$, or $O(3)$ is physically correct.

## The geometric-deep-learning "blueprint" in one line

Every geometric architecture is built from two operations on feature fields over a space $X$:

> 1. **Local aggregation** (neighbourhood pooling $\mathcal{P}$)
> 2. **Nonlinearity / equivariant map** $\psi$ (a symmetric function)

This is exactly the formulation in the **bronstein2017** and **bronstein2021** papers.

## Code reference

See `papers/group_equivariance/cohen2016_group/code/` (implemented by the reading-group build) for
a working `p4` group convolution.

---
**Next:** [04 — Hyperbolic geometry (the Poincaré ball)](04_hyperbolic_geometry.md)
