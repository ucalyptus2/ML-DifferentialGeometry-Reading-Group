# 01 — Manifolds & tangent spaces

## What is a manifold?

A **smooth manifold** $M$ is a space that *locally* looks like Euclidean space $\mathbb{R}^n$, but
which globally can be curved. Around every point $x \in M$ there is a **chart** — a smooth bijection
$$\varphi_x : U \to \mathbb{R}^n$$ from a neighbourhood $U \subset M$ to an open set of $\mathbb{R}^n$.

Examples:
- the unit sphere $S^2 = \{x \in \mathbb{R}^3 : \|x\|=1\}$ — locally like $\mathbb{R}^2$,
- the **SPD cone** $\mathcal{S}^n_{++} = \{A \in \mathbb{R}^{n\times n} : A = A^\top \succ 0\}$,
- the **rotation group** $SO(3) = \{R : R^\top R = I,\ \det R = 1\}$,
- the **Poincaré ball** $B^n_c = \{x : \|x\| < 1/\sqrt{c}\}$ (hyperbolic space).

## Tangent space

At each point $x$ there is a **tangent space** $T_x M$ — the vector space of all "velocities" of
curves passing through $x$. It is the best *linear* approximation of $M$ at $x$ and has dimension
$n = \dim M$.

- On a sphere: $T_x S^2 = \{v : v \cdot x = 0\}$ (a 2-D plane tangent to the sphere).
- On any manifold, gradients of a function live in $T_x M$ — this is why *Euclidean* gradient
  descent is wrong on a manifold: the usual gradient points out of the tangent space.

## Why this matters for ML

Every modern "geometric" optimizer and architecture chooses a **tangent-space representation**:

1. compute an update vector $v \in T_x M$,
2. move to a new point via the **exponential map** $\exp_x(v)$ (lesson 02).

For a manifold embedded in a vector space (like the sphere), the "project-onto-tangent" step
appears everywhere — including the *E(n)-equivariant* and *gauge* networks later in the reading list.

## Code

```python
# sphere tangent space: v is tangent iff v . x == 0
v = torch.tensor([0.0, 1.0, 0.0])
x = torch.tensor([1.0, 0.0, 0.0])
print(torch.dot(x, v))   # 0.0  -> tangent to the sphere at x
```

Numeric demo of exp/log round-trips is in `tutorial/code/riemannian_pytorch.py` (section 1).

---
**Next:** [02 — Riemannian metrics & geodesics](02_riemannian_metric_geodesics.md)
