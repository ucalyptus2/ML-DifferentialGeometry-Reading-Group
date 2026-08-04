# Tutorial — Machine Learning × Differential Geometry

This folder teaches the **differential-geometric ideas** behind every paper in this reading group,
with runnable code. It's intended as the on-ramp before diving into `papers/`.

## Lessons

| # | Lesson | Key ideas |
|---|---|---|
| 00 | [Why geometry matters](00_why_geometry.md) | manifolds, equivariance, information geometry |
| 01 | [Manifolds & tangent spaces](01_manifolds_tangent_spaces.md) | $M$, $T_xM$, charts |
| 02 | [Riemannian metrics & geodesics](02_riemannian_metric_geodesics.md) | $g_x$, $\exp/\log$, geodesic distance |
| 03 | [Groups & equivariance](03_groups_equivariance.md) | group action, $f(gx)=gf(x)$, Lie algebra |
| 04 | [Hyperbolic geometry](04_hyperbolic_geometry.md) | Poincaré ball, negative curvature |
| 05 | [Matrix manifolds: SPD](05_matrix_manifolds_spd.md) | affine-invariant metric, retractions |
| 06 | [Geometric DL blueprint](06_geometric_deep_learning_blueprint.md) | 1- vs 2-categorical, gauge |

## Runnable code (`code/`)

| File | Framework | Covers |
|---|---|---|
| `riemannian_pytorch.py` | PyTorch | sphere + Poincaré + SPD + SO(3) exp/log; Riemannian GD; SPD retraction check; sphere holonomy/curvature demo |
| `riemannian_jax.py` | JAX | same core in JAX (`pip install jax` — not exercised in this repo's CI/dev environment, kept for readers who have JAX installed) |

Run them:

```bash
cd code
python riemannian_pytorch.py
# or
python riemannian_jax.py
```

## Notation glossary

The same symbols recur across all 7 lessons — reconciled here once instead of per lesson:

| Symbol | Meaning | First defined |
|---|---|---|
| $M$, $x \in M$ | a manifold, a point on it | 01 |
| $T_xM$ | tangent space at $x$ (linear approx. of $M$ at $x$) | 01 |
| $g_x(\cdot,\cdot)$ | Riemannian metric: inner product on $T_xM$ | 02 |
| $\exp_x(v)$, $\log_x(y)$ | move along a geodesic by $v$; its inverse | 02 |
| $K$ | sectional curvature ($+$: sphere-like, $0$: flat, $-$: hyperbolic-like) | 02 |
| $\tau_{x\leftarrow y}$ | parallel transport, $T_yM \to T_xM$ (a connection) | 06 |
| $\kappa$ | the Poincaré ball's curvature magnitude, $K=-\kappa$ | 01, 02, 04 |
| $G$, $\rho$ | a (Lie) group, and its action $\rho:G\to\operatorname{Aut}(X)$ | 03 |
| $\mathfrak{g}$ | the Lie algebra of $G$ (tangent space at the identity) | 03 |
| $\operatorname{Retr}_x$ | a retraction: any first-order approximation of $\exp_x$ | 05 |

## How to build on this

After the tutorial, pick the paper that interests you in `../papers/`, read its `paper.md`, then run
its `code/` and try the *reading questions* at the bottom.
