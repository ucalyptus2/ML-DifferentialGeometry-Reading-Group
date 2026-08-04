# 06 — The geometric deep learning blueprint

## One blueprint to rule them all

The two **geometric deep learning** survey papers (Bronstein et al. 2017, 2021) make a powerful
point: **almost every deep network is a special case of one blueprint**. A layer on a space $X$
acts on *feature fields* over $X$ and is built from just two operations:

> **Local & geometric aggregation** $\ \mathcal{P}\ $ — combine features from a neighbourhood,
>
> **Equivariant nonlinearity** $\ \psi\ $ — a pointwise map that respects the symmetry.

For grids: $\mathcal{P}$ = convolution on $\mathbb{Z}^2$. For groups: convolution on $G$. For
graphs: message passing on edges. For geodesic/manifold data: aggregation along geodesic patches.
For gauge-equivariant models: aggregation with *parallel transport* to align local frames.

## The 1-categorical vs 2-categorical distinction (Bronstein et al. 2021)

This is the *single most important conceptual idea in the whole reading list*:

- **1-categorical** tasks: the feature is a **scalar field** (or a group representation) and you can
  aggregate neighbours *without* comparing local frames. Translation-equivariant CNNs, GCNs,
  hyperbolic models — these only need the neighbouring relation $\to$ and a symmetric aggregator.
- **2-categorical** tasks: features are **vectors/tensors** whose orientation matters (e.g.
  surface normals on a sphere). To add two vectors at different points you must first **transport**
  one to the other along a path. This requires a **connection** and **parallel transport** — the
  *gauge-equivariant* setting.

That is precisely why **gauge-equivariant mesh CNNs** (de Haan et al.) and **Gauge CNNs**
(Cohen et al.) talk about principal $G$-bundles, connections, and parallel transport: they are the
architectures that handle *non-kinematic* (vector/tensor) fields correctly.

## Connections & parallel transport, made concrete

A **connection** is the rule that tells you how to move a tangent vector from $T_yM$ to $T_xM$
along a path from $y$ to $x$ without "differentiating it against nothing" — plain component-wise
subtraction is meaningless since $T_xM$ and $T_yM$ are *different* vector spaces embedded
differently in ambient space. **Parallel transport** $\tau_{x\leftarrow y}: T_yM \to T_xM$ is the
connection's answer: transport a vector along a curve so it changes "as little as possible" (zero
covariant derivative along the curve). Two properties every connection has:

- it's a **linear isometry** $T_yM \to T_xM$ (preserves the metric $g$: $g_x(\tau v,\tau w) =
  g_y(v,w)$ — transporting can rotate a vector but never stretches it),
- transporting a vector around a **closed loop** need not return it to itself — the mismatch is
  exactly the **curvature** $K$ of lesson 02 (Gauss–Bonnet: holonomy angle $=\int K\,dA$).

On the sphere this has a clean closed form. If $u$ is a unit tangent at $x$ and $\gamma(t) =
\cos(t)x+\sin(t)u$ is the unit-speed geodesic through $x$ in direction $u$, then transporting
$w\in T_xM$ to $\gamma(t)$ is
$$\tau_{\gamma(t)\leftarrow x}(w) = w - \langle u,w\rangle(1-\cos t)\,u - \langle u,w\rangle \sin t\;x.$$
`tutorial/code/riemannian_pytorch.py` (`sphere_parallel_transport`, section 7) implements exactly
this, and chains three such transports around a spherical triangle to reproduce the 90°
holonomy — the same demo cited in lesson 02, now tied to *why* gauge-equivariant layers need
$\tau_{x\leftarrow y}$ at all: without it, "$f(y)$ for $y$ near $x$" lives in the wrong vector
space to be combined with $f(x)$.

## The blueprint in equations

For a $G$-structure with structure group $G$ (gauge):

```
feature fields  f : X -> V        (V = a rep. of G)
neighbour aggregate  P:  (f(x), {f(y)}_{y~x}) -> sum over transported neighbours
   parallel transport  tau_{x<-y} : T_y X -> T_x X   (the "connection")
gauge-equivariant conv:
   (P*f)(x) = sum_{y~x}  rho( tau_{x<-y} ) f(y)      (rho = the G-representation)
```

## How each folder instantiates the blueprint

| Folder | Space $X$ | Aggregator $\mathcal{P}$ | Equivariance |
|---|---|---|---|
| `group_equivariance` | $\mathbb{Z}^2$, $S^2$, point clouds | $G$-conv / spherical-harmonic conv / distance message passing | $G$, $SO(3)$, $E(n)$ |
| `gauge` | surfaces & meshes | gauge conv + parallel transport | gauge group $G$ |
| `geodesics_shape` | Riemannian manifolds/meshes | geodesic polar patches / surface diffusion | isometries |
| `hyperbolic` | Poincaré ball | Möbius/gyrovector ops, tangent aggregation | Möbius isometries |
| `spd_riemannian`, `flows_odes`, `diffusion_manifolds` | curved optim/manifolds | retractions, exp·log, Riemannian BM | manifold isometries |

## TL;DR

> **Learn to spot the two operations (aggregate + symmetric nonlinearity) in every paper.**
> Then figure out which *space* and which *symmetry* the authors used — the whole geometric-deep-
> learning literature becomes a catalogue of that blueprint on different geometries.

## Check yourself

1. Why is a translation-equivariant CNN "1-categorical" but a gauge-equivariant mesh CNN
   "2-categorical"? *(A CNN's features are scalars/channels at each pixel — no orientation to
   transport, just neighbours to sum. A mesh CNN's features are vectors in each face's *own*
   tangent plane, which differ face to face, so combining them needs $\tau_{x\leftarrow y}$
   first.)*
2. In the blueprint equation $(P{*}f)(x) = \sum_{y\sim x} \rho(\tau_{x\leftarrow y})f(y)$, what
   would break if you dropped the $\rho(\tau_{x\leftarrow y})$ term and just summed $f(y)$
   directly? *(You'd be adding vectors from different tangent spaces as if they were the same
   space — not merely inaccurate, but not well-defined, since $T_yM \neq T_xM$ have no shared
   coordinates without a connection to relate them.)*
3. Pick any paper in `papers/`. Which folder's row of the blueprint table does it belong to, and
   what plays the role of $\mathcal{P}$ and $\psi$ in its architecture?

---
**Start the papers:** browse [`papers/`](https://github.com/ucalyptus2/ML-DifferentialGeometry-Reading-Group/tree/master/papers) — foundation papers first.
