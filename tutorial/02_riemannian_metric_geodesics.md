# 02 — Riemannian metrics & geodesics (exp / log maps)

## Riemannian metric

A **Riemannian metric** is a smoothly-varying inner product
$$g_x : T_x M \times T_x M \to \mathbb{R}$$
on the tangent space at each point. It turns $M$ into a *metric* (distance) space.

The metric completely determines:
- the **length** of a curve $\gamma:[0,1]\to M$: $L(\gamma)=\int_0^1 \sqrt{g_{\gamma(t)}(\dot\gamma,\dot\gamma)}\,dt$,
- the **geodesics** — locally shortest curves (generalizations of straight lines),
- the **exponential/log maps**.

## Geodesics, $\exp$ and $\log$

Given a point $x$ and a tangent vector $v \in T_xM$, let $\gamma_v(t)$ be the geodesic with
$\gamma_v(0)=x,\ \dot\gamma_v(0)=v$. Define

- **Exponential map**: $\ \exp_x(v) = \gamma_v(1),$  (move along the geodesic by $v$)
- **Logarithmic map**: $\ \log_x(y)$ is the unique $v\in T_xM$ with $\exp_x(v)=y$ (inverse),
- **Geodesic distance**: $\ d(x,y) = \|\log_x(y)\|_x = \sqrt{g_x(\log_x y,\ \log_x y)}.$

Concrete formulas used all over the reading list:

**Sphere $S^n$** (radius 1)
$$\exp_x(v) = \cos\|v\|\, x + \sin\|v\|\,\frac{v}{\|v\|}, \qquad
  d(x,y) = \arccos(x\cdot y).$$

**Poincaré ball $\mathbb{B}^n$** (hyperbolic, curvature $-c$)
$$d(x,y) = \frac{1}{\sqrt c}\,\operatorname{arcosh}\!\Big(1 + \frac{2c\,\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\Big).$$

**SPD affine-invariant** manifold
$$\operatorname{Exp}_A(V) = A^{\tfrac12} e^{\,A^{-\tfrac12}V A^{-\tfrac12}} A^{\tfrac12}.$$

**Rotation group $SO(3)$**
$$\exp([\omega]_\times) = I + \sin\theta\,[\hat\omega]_\times + (1-\cos\theta)\,[\hat\omega]_\times^2.$$

## Why this matters for ML

- **Riemannian gradient descent** (used in *spd_riemannian/*, *flows_odes/*):
  final step $x_{new} = \exp_x(-\eta\,\mathrm{grad}\,f)$ always stays on the manifold.
  Mathematically: $\ \mathrm{grad}\,f$ is the *metric transpose* of the Euclidean gradient,
  $\mathcal{G}^{-1}\nabla f$, and the exp-map keeps the iterate valid (e.g. SPD or rotations).
- **Log maps give features**: short geodesic distances / directions are the local coordinates used
  by geodesic CNNs and manifold networks.

## Code

```bash
cd tutorial/code && python riemannian_pytorch.py
```
Sections 1–4 verify the exp/log round-trips for the sphere, Poincaré ball, SPD, and SO(3).

---
**Next:** [03 — Groups & equivariance](03_groups_equivariance.md)
