# 02 — Riemannian metrics & geodesics (exp / log maps)

**Underlies:** `spd_riemannian/`, `hyperbolic/`, `flows_odes/`, `diffusion_manifolds/` — anything
that takes a gradient step or measures distance on a curved space.

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

**Poincaré ball $\mathbb{B}^n$** (hyperbolic, curvature $-\kappa$)
$$d(x,y) = \frac{1}{\sqrt \kappa}\,\operatorname{arcosh}\!\Big(1 + \frac{2\kappa\,\|x-y\|^2}{(1-\kappa\|x\|^2)(1-\kappa\|y\|^2)}\Big).$$

**SPD affine-invariant** manifold
$$\operatorname{Exp}_A(V) = A^{\tfrac12} e^{\,A^{-\tfrac12}V A^{-\tfrac12}} A^{\tfrac12}.$$

**Rotation group $SO(3)$**
$$\exp([\omega]_\times) = I + \sin\theta\,[\hat\omega]_\times + (1-\cos\theta)\,[\hat\omega]_\times^2.$$

## Curvature — what makes a space "curved" in the first place

The formulas above look unrelated (trig for the sphere, `arcosh`/hyperbolic for the ball) but
they're two instances of *one* number: the **sectional curvature** $K$. For a 2-D patch, $K$
governs how a family of geodesics starting parallel at $x$ spreads apart, via the **Jacobi
equation** for a normal geodesic-variation field $J(t)$:
$$J''(t) + K\,J(t) = 0.$$

- $K=0$ (flat $\mathbb{R}^n$): $J(t) = J(0) + t J'(0)$ — parallel geodesics stay parallel forever.
- $K=+1$ (unit sphere): $J(t)=\cos(t)J(0)+\sin(t)J'(0)$ — geodesics **converge** (they all meet
  at the antipode) — the same $\cos/\sin$ that appears in $\exp_x$ above.
- $K=-\kappa$ (Poincaré ball): $J(t)=\cosh(\sqrt\kappa t)J(0)+\ldots$ — geodesics **diverge
  exponentially** — the same hyperbolic functions ($\operatorname{arcosh}$/$\tanh$) that appear
  in $d(x,y)$ and $\exp_x$ above. This exponential spreading is *exactly* why hyperbolic space
  fits trees (lesson 04): distance between siblings grows exponentially with depth, matching a
  tree's branching factor.

Curvature is **intrinsic** (Gauss's *Theorema Egregium*): you never need to look at how $M$ sits
in an ambient $\mathbb{R}^N$ to measure it, only the metric $g_x$. The cleanest way to *see* it
without ever writing a Christoffel symbol is **holonomy**: parallel-transport a tangent vector
around a closed loop and measure how much it rotates. Gauss–Bonnet says that rotation angle
equals $\int_{\text{loop}} K\,dA$ — curvature integrated over the enclosed area. Section 7 of
`tutorial/code/riemannian_pytorch.py` transports a vector around a right-angle spherical
triangle (area $\pi/2$, unit sphere $K=1$) and gets back a vector rotated by **exactly 90°**
$= K \cdot \text{area}$ — curvature, made numerically visible.

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
Section 7 runs the holonomy/curvature demo above.

## Check yourself

1. Why does $K=+1$ make the sphere's `exp` map use $\cos/\sin$, while $K=-\kappa$ makes the
   Poincaré ball's use `arcosh`/`tanh`? *(They're both solutions of the same Jacobi equation
   $J''+KJ=0$; the sign of $K$ flips trigonometric solutions into hyperbolic ones — the exact
   same equation, analytically continued.)*
2. If you shrink a spherical triangle's enclosed area toward 0, what happens to its holonomy
   angle? *(It shrinks to 0 too, since holonomy $= K\cdot\text{area} \to 0$ — locally, any
   smooth curved space looks flat, which is exactly why charts/tangent spaces (lesson 01) work.)*
3. On a *flat* torus ($K=0$ everywhere), what holonomy would you expect after transporting a
   vector around any closed loop? *(Zero rotation — flat spaces have trivial holonomy
   regardless of the loop's shape or area, unlike the sphere.)*

---
**Next:** [03 — Groups & equivariance](03_groups_equivariance.md)
