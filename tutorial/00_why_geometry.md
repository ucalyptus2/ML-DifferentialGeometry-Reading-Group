# 00 — Why geometry matters for machine learning

**Underlies:** `papers/foundations/` (the Bronstein et al. surveys formalize everything below).

A surprising amount of modern ML is really **geometry**. Three recurring themes tie every paper in
this reading group together:

1. **The data lives on a manifold, not in flat space.**
   Rotation matrices, symmetric positive-definite (SPD) covariance matrices, human poses, 3D
   shapes, and discrete probability distributions all form **curved** spaces with non-Euclidean
   structure. Interpolating or averaging them "naively" in a flat vector space gives wrong answers
   (e.g. linearly blending two rotation matrices is not a rotation).

2. **The model should respect the symmetries of the problem.**
   A neural net that classifies objects in a scene should give the same answer if everything is
   rotated, translated, or re-flipped. This is **equivariance** — making the architecture invariant
   to a **group** of transformations. Differential geometry gives the language (Lie groups, group
   actions, gauge structures) to build such architectures systematically.

3. **Information itself has a geometric structure.**
   The Fisher information metric endows spaces of distributions with a Riemannian geometry
   (**information geometry**); moving a distribution optimally is **optimal transport** with a
   metric (Wasserstein). Optimization on any curved space is **Riemannian optimization**.

### Three examples in one sentence each

| Problem | Curved geometry at play | Papers |
|---|---|---|
| Embed a *hierarchy* (animals→mammals→dogs) | **hyperbolic** space fits trees better than flat space | Nickel & Kiela, Ganea et al., Chami et al. |
| Classify anything on a sphere / under rotations | **SO(3)** group action + **gauge** on the sphere | Cohen et al., de Haan et al. |
| Learn a continuous generative model | **geodesics & flows** on the data manifold | Chen et al. (ODE), De Bortoli et al. (Riemannian diffusion) |

### The conceptual ladder

```
flat ℝⁿ  →  manifolds M  →  Riemannian metrics  →  geodesics/exp·log
        →  groups & equivariance  →  gauge structures  →  curvature (hyperbolic/spherical)
```

Every item on that ladder shows up in `papers/`. The `tutorial` walks up it with runnable code.

### Try it

```bash
cd tutorial/code
python riemannian_pytorch.py   # or: python riemannian_jax.py
```
Watch the *Riemannian gradient descent* demo: even a naive optimizer that "retracts" back onto the
sphere after every step stays on the manifold — the core trick of Riemannian optimization.

## Check yourself

1. Why is "linearly blending two rotation matrices" (e.g. $\tfrac12(R_1+R_2)$) usually not a
   rotation at all? *(Averaging isn't closed on $SO(3)$ — the constraint $R^\top R=I$ is
   quadratic, not linear/convex, so a convex combination of two solutions generally isn't a
   solution. Lessons 01–02 give the fix: average in the tangent space via $\log$, then map back
   with $\exp$.)*
2. Pick one row of the "three examples" table. Which of the three recurring themes (curved data /
   symmetry / information geometry) does it fall under, and why?

---
**Next:** [01 — Manifolds & tangent spaces](01_manifolds_tangent_spaces.md)
