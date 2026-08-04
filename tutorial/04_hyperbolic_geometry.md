# 04 — Hyperbolic geometry (the Poincaré ball)

## The idea

Hyperbolic space is a complete, simply-connected Riemannian manifold of **constant negative
curvature** $-\kappa$. Its volume grows *exponentially* with radius — exactly like a tree grows:
a binary tree of depth $d$ has $2^d$ nodes but "fits" with $O(\log)$ distances in hyperbolic space.
This is why **hierarchical / tree-structured data** can be embedded with tiny distortion, whereas
flat (Euclidean) space needs exponentially many dimensions.

## The Poincaré ball model

The Poincaré ball is the disk
$$\mathbb{B}^n_\kappa = \{ x \in \mathbb{R}^n : \kappa\|x\|^2 < 1 \}$$
with the Riemannian metric
$$g_x = \lambda^2_x\, I,\qquad \lambda_x = \frac{2}{1-\kappa\|x\|^2}.$$
(The conformal factor $\lambda_x$ grows toward the boundary — distances stretch there.)

**Geodesic distance**
$$d_\kappa(x,y) = \frac{1}{\sqrt\kappa}\operatorname{arcosh}\!\Big(1 + \frac{2\kappa\|x-y\|^2}{(1-\kappa\|x\|^2)(1-\kappa\|y\|^2)}\Big).$$

**Möbius addition** (the group operation making the ball a gyrogroup):
$$x \oplus_\kappa y = \frac{(1+2\kappa\langle x,y\rangle+\kappa\|y\|^2)\,x + (1-\kappa\|x\|^2)\,y}{1+2\kappa\langle x,y\rangle+\kappa^2\|x\|^2\|y\|^2}.$$

## What the papers do with it

| Paper | Idea |
|---|---|
| **Poincaré Embeddings** (Nickel & Kiela 2017) | Learn embeddings of a hierarchy by minimizing Poincaré distance between related nodes; loss pushes unrelated nodes apart |
| **Hyperbolic Neural Networks** (Ganea et al. 2018) | Do matrix-vector operations **inside** the ball using Möbius gyrovector spaces (analog of neurons) |
| **Hyperbolic GCN** (Chami et al. 2019) | Aggregate in tangent space (Euclidean), then map back with the exponential map + project to the ball |
| **Hyperbolic Entailment Cones** (Ganea et al. 2018) | Model a partial order: encode $u \preceq v$ as "cone of $u$ inside cone of $v$" |

## The two "extremes" worth internalizing

```
flat space:   circle grows ~ O(r)     -> balanced trees are squeezed
hyperbolic:   circle grows ~ O(e^r)   -> trees "fit naturally"
```

## Code

`tutorial/code/riemannian_pytorch.py` implements `poincare_dist` and `poincare_exp`; verify that
the distance grows dramatically as points approach the boundary:
```
d(a, -a)  = 0.564   # both near origin
d(far,-far)= 6.579  # both near boundary  -> hyperbolic stretching
```
Full hyperbolic implementations are in `papers/hyperbolic/*/code/`.

## Check yourself

1. Lesson 02 showed $K=-\kappa$ geodesics diverge like $\cosh(\sqrt\kappa\, t)$. Sketch why that
   forces $|\mathbb{B}^n_\kappa|$'s volume within radius $r$ to grow like $e^{\sqrt\kappa\,r}$
   rather than polynomially, as in flat space. *(Volume accumulates over a "sphere of radius
   $r$" whose surface area itself grows like $\sinh^{n-1}(\sqrt\kappa\,r)\sim e^{(n-1)\sqrt\kappa
   r}$ — exponential, not $r^{n-1}$ as in $\mathbb{R}^n$.)*
2. As $\kappa \to 0$, what does the Poincaré distance formula reduce to? *(Ordinary Euclidean
   distance $\|x-y\|$ — hyperbolic space "flattens out" in the small-curvature limit, consistent
   with $K=-\kappa\to 0$.)*
3. Why is Möbius addition $\oplus_\kappa$ **not** commutative in general (unlike ordinary vector
   addition), and why does that matter for building "hyperbolic neurons"? *(The ball has no
   linear structure; $\oplus_\kappa$ is a *gyrogroup* operation, so operation order matters —
   hyperbolic-NN layers (Ganea et al.) must fix a convention for which side the bias/weight acts
   on, unlike a Euclidean $Wx+b$.)*

---
**Next:** [05 — Matrix manifolds: SPD & retractions](05_matrix_manifolds_spd.md)
