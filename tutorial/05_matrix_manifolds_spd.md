# 05 — Matrix manifolds: SPD & retractions

## The SPD manifold

Covariance matrices, distance matrices, precisions, and many deep feature "Gram" matrices are
**symmetric positive-definite (SPD)**: $A = A^\top$ and $x^\top A x > 0$ for all $x \neq 0$. The set
$$\mathcal{S}^n_{++} = \{ A \in \mathbb{R}^{n\times n} : A = A^\top \succ 0 \}$$
is an *open convex cone*, but treating it as flat $\mathbb{R}^{n(n+1)/2}$ is wrong — e.g. the
arithmetic mean of two SPD matrices is SPD but loses the geometry, and "distances" become meaningless.

Equip it with the **affine-invariant metric**
$$g_A(X,Y) = \operatorname{tr}\big(A^{-1} X\, A^{-1} Y\big),\qquad X,Y \in T_A\mathcal{S}^n_{++}.$$
Under this metric:
$$\operatorname{Exp}_A(V) = A^{\tfrac12} e^{\,A^{-\tfrac12} V A^{-\tfrac12}} A^{\tfrac12},$$
$$\operatorname{Log}_A(B) = A^{\tfrac12} \log\!\big(A^{-\tfrac12} B A^{-\tfrac12}\big) A^{\tfrac12},$$
$$d(A,B) = \big\|\log(A^{-\tfrac12} B A^{-\tfrac12})\big\|_F.$$

## Retractions

Computing the true exponential map can be costly. A **retraction** is any map
$\operatorname{Retr}_x : T_xM \to M$ that is a first-order approximation of $\exp_x$:
$$\operatorname{Retr}_x(0)=x,\qquad \frac{d}{dt}\operatorname{Retr}_x(tv)\Big|_{t=0}=v.$$
Examples used in practice:
- SPD: $\operatorname{Retr}_A(V) = \exp(A^{-1}V)\,A$ (or Cholesky factorizations),
- Sphere: $\operatorname{Retr}_x(v) = \frac{x+v}{\|x+v\|}$,
- Grassmann: QR-decomposition based.

Riemannian optimizers (the **Riemannian Adam** paper) combine a retraction with a Riemannian
adaptive step to stay on the manifold.

## Why this matters for ML

- **SPD nets** (Huang & Van Gool) build deep networks whose *features* are SPD matrices, using
  bi-mapping, |·|-eigenvalue nonlinearities, and log-mapped pooling.
- **Riemannian optimization** (Bećigneul & Ganea) generalizes Adam/SGD to *any* Riemannian
  manifold, used for hyperbolic embeddings, SPD learning, and optimization on $SO(3)$.

## Code

```bash
cd tutorial/code && python riemannian_pytorch.py   # SPD exp/log round-trip (section 3)
```
Full Riemannian-SGD / Adam implementations are in `papers/spd_riemannian/*/code/`.

---
**Next:** [06 — The geometric deep learning blueprint](06_geometric_deep_learning_blueprint.md)
