# Continuous Normalizing Flows on Manifolds

- **Authors:** Luca Falorsi
- **Venue/Year:** arXiv preprint, 2021
- **arXiv:** https://arxiv.org/abs/2104.14959
- **Category:** flows_odes

## One-paragraph TL;DR
FFJORD-style continuous normalizing flows are defined by integrating a vector field in flat
$\mathbb{R}^n$; this breaks the moment data itself lives on a manifold (rotations, spheres,
tori, hyperbolic space). This line of work extends CNFs to Riemannian manifolds by defining the
vector field as a genuine **tangent vector field** $f(h,t) \in T_hM$ and integrating it with a
manifold-respecting numerical scheme (retraction or exponential map at every step) instead of
naive ambient-space Euler steps — so the flow never leaves $M$, by construction rather than by
post-hoc projection.

## The problem
Naively running Euler/RK4 in the ambient embedding coordinates ($\mathbb{R}^{n+1}$ for $S^n$, or
$\mathbb{R}^{n\times n}$ for $SO(n)$/SPD) treats a tangent vector field as if it lived in flat
space: each step $h_{t+1} = h_t + \Delta t\, f(h_t,t)$ walks in a *straight line*, which
immediately leaves the curved manifold (a straight line off a sphere is not on the sphere).
Renormalizing back onto $M$ after each step is a crude, uncontrolled approximation whose error
compounds and whose renormalization map isn't even well-defined for manifolds without a natural
embedding-space projection (e.g. $SO(n)$).

## Key idea(s)
- Define the flow's velocity field intrinsically: $f(h,t)\in T_hM$ (a genuine tangent vector at
  $h$, not an ambient-space vector), the direct manifold generalization of Neural ODE's
  $\dot h = f(h,t)$.
- Integrate with a **manifold-aware solver**: each Euler/RK sub-step retracts via
  $\operatorname{Exp}_h(\Delta t\, f(h,t))$ (or a cheaper retraction) instead of ambient addition
  — every intermediate state is *exactly* on $M$, not approximately, at every step.
- The instantaneous change-of-variables formula generalizes using the **Riemannian divergence**
  of $f$ (trace of the covariant derivative $\nabla f$ in an orthonormal frame) in place of the
  flat-space trace of the Jacobian.

## The mathematics
On the sphere $S^{n-1}\subset\mathbb{R}^n$ with a tangent field $f(x,t)\in T_xS^{n-1}$ (i.e.
$\langle x,f(x,t)\rangle=0$), the exact geodesic flow step is
$$x_{t+\Delta t} = \operatorname{Exp}_x(\Delta t\, f(x,t)) = \cos(\|\Delta t\,f\|)\,x +
\sin(\|\Delta t\,f\|)\,\frac{f}{\|f\|}$$
— the closed form used throughout this reading group's `sphere_exp`. Naive ambient Euler,
$x_{t+\Delta t} = x_t + \Delta t\,f(x_t,t)$, satisfies $\|x_{t+\Delta t}\| = \sqrt{1+\Delta
t^2\|f\|^2} > 1$ **strictly** whenever $f\ne 0$ — it leaves the sphere on *every single step*, with
error accumulating $O(\Delta t^2)$ per step and compounding over the trajectory.

## Method / architecture
Every solver step in the CNF is replaced by: (1) evaluate the tangent vector field $f(h,t)$ from a
neural net constrained to output tangent vectors (e.g. project a raw net output onto $T_hM$), (2)
retract via $\operatorname{Exp}_h$ or a cheap first-order retraction, (3) accumulate the log-density
correction via the Riemannian divergence. This composes with FFJORD's Hutchinson trick for the
divergence term at scale.

## Code
See `code/manifold_cnf_drift.py` — integrates the same tangent vector field on $S^2$ with (a) naive
ambient Euler (no retraction) and (b) retraction-based Euler using `sphere_exp` from
`manifold_ops.py`, and measures $|\,\|x_t\|-1\,|$ (distance off the manifold) over the trajectory
for both.

## Why it matters
This closes the loop between Neural ODEs, FFJORD, and the rest of this reading group's geometric
material: once you have `exp`/`log`/tangent projections for a manifold (sphere, Poincaré ball, SPD,
$SO(n)$), *any* Neural-ODE/CNF machinery transfers directly, giving manifold-valued generative
models (e.g. generating rotations, directions, or hierarchies) with exact on-manifold guarantees
instead of post-hoc projection hacks.

## Reading questions / discussion
1. Why does naive ambient Euler *always* increase $\|x\|$ above 1 for a nonzero tangent step on
   $S^{n-1}$ (hint: Pythagoras — $x$ and $f(x)$ are orthogonal)? Does the same argument apply to
   the Poincaré ball, or does that manifold's boundary behave differently?
2. The retraction-based step costs one `sphere_exp` evaluation (a couple of trig calls) per solver
   step, vs. one vector add for naive Euler. When is that overhead negligible, and when might it
   dominate (hint: think about the dimension and curvature of $M$)?
3. How would you constrain a neural net's raw output to be a genuine tangent vector at $x$ (i.e.
   $\langle x, f(x)\rangle = 0$) rather than an arbitrary ambient vector?
