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

---
**Start the papers:** browse [`papers/`](../papers) — foundation papers first.
