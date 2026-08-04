# 01 — Manifolds & tangent spaces

**Underlies:** every paper in `papers/` — this is the common substrate. Start here if you're new.

## What is a manifold?

A **smooth manifold** $M$ is a space that *locally* looks like Euclidean space $\mathbb{R}^n$, but
which globally can be curved. Around every point $x \in M$ there is a **chart** — a smooth bijection
$$\varphi_x : U \to \mathbb{R}^n$$ from a neighbourhood $U \subset M$ to an open set of $\mathbb{R}^n$.

Examples:
- the unit sphere $S^2 = \{x \in \mathbb{R}^3 : \|x\|=1\}$ — locally like $\mathbb{R}^2$,
- the **SPD cone** $\mathcal{S}^n_{++} = \{A \in \mathbb{R}^{n\times n} : A = A^\top \succ 0\}$,
- the **rotation group** $SO(3) = \{R : R^\top R = I,\ \det R = 1\}$,
- the **Poincaré ball** $\mathbb{B}^n_\kappa = \{x : \|x\| < 1/\sqrt{\kappa}\}$ (hyperbolic space,
  curvature $-\kappa$; same $\kappa$ notation used in lessons 02 and 04).

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

## Check yourself

1. Is $v=(1,1,0)$ tangent to $S^2$ at $x=(1,0,0)$? *(No: $v\cdot x = 1 \neq 0$. Only its
   component orthogonal to $x$, i.e. $(0,1,0)$, is tangent.)*
2. Why is $T_xM$ a **vector space** even when $M$ itself is not (e.g. the sphere isn't closed
   under addition)? *(It's the linearization of $M$ at a single point via the chart
   $\varphi_x$ — charts are what make $T_xM \cong \mathbb{R}^n$ with ordinary vector-space
   structure, independent of $M$'s global curvature.)*
3. $SO(3)$ has dimension 3, but the matrices are $3\times 3$ (9 numbers). Where do the other 6
   degrees of freedom go? *(They're pinned down by the constraints $R^\top R=I,\ \det R=1$ —
   6 independent scalar equations, leaving $9-6=3$ free parameters, matching $\dim SO(3)=3$.)*

---
**Next:** [02 — Riemannian metrics & geodesics](02_riemannian_metric_geodesics.md)
