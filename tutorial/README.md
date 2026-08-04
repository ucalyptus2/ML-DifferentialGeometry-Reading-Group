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
| `riemannian_pytorch.py` | PyTorch | sphere + Poincaré + SPD + SO(3) exp/log; Riemannian GD on the sphere |
| `riemannian_jax.py` | JAX | same core in JAX (`pip install jax`) |

Run them:

```bash
cd code
python riemannian_pytorch.py
# or
python riemannian_jax.py
```

## How to build on this

After the tutorial, pick the paper that interests you in `../papers/`, read its `paper.md`, then run
its `code/` and try the *reading questions* at the bottom.
