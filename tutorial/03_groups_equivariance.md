# 03 — Groups & equivariance

## Groups and group actions

A **group** $(G, \cdot)$ is a set with an associative operation, an identity $e$, and inverses.
It **acts** on a set/space $X$ via a **group homomorphism**
$$\rho: G \to \operatorname{Aut}(X)$$
into the automorphisms (invertible structure-preserving self-maps) of $X$. Being a homomorphism
means it respects composition — for all $g,h \in G$:
$$\rho(g)\circ\rho(h)=\rho(gh), \qquad \rho(e) = \operatorname{id}_X, \qquad \rho(g)^{-1}=\rho(g^{-1}).$$
(Read $\rho(g)x$ as "$g$ acting on $x$"; the middle identity is *why* composing two actions in
sequence is the same as acting by the product $gh$ once — the property equivariance is built on.)

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

## Check yourself

1. Is "add 5 to every pixel value" a group action of $\mathbb{R}$ on an image? Is it equivariant
   for a CNN? *(Yes it's a valid action (intensity-shift group); a linear-then-ReLU CNN layer is
   generally NOT equivariant to it, since ReLU's threshold at 0 breaks with a shift — a concrete
   reason architectures must be *designed* for the symmetry they target, not just any group.)*
2. Why must $\rho(e)=\operatorname{id}_X$ follow from the homomorphism property alone (not be
   assumed separately)? *(Set $g=h=e$: $\rho(e)\circ\rho(e)=\rho(e)$, so $\rho(e)$ is idempotent
   and invertible in $\operatorname{Aut}(X)$ — the only idempotent invertible map is the
   identity.)*
3. $SO(2)$ acting on $\mathbb{R}^2$ by rotation is **invariant** for $f(x)=\|x\|$ but
   **equivariant** (not invariant) for $f(x)=x$ itself. What's the general pattern? *(Invariance
   is equivariance where the action on the output space $Y$ is trivial — $\|x\|$ is a distance,
   which rotation doesn't change, so $\rho_Y = \mathrm{id}$.)*

---
**Next:** [04 — Hyperbolic geometry (the Poincaré ball)](04_hyperbolic_geometry.md)
