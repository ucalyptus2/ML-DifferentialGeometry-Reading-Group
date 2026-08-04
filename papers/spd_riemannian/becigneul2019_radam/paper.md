# Riemannian Adaptive Optimization Methods

- **Authors:** Gary Bécigneul, Octavian-Eugen Ganea
- **Venue/Year:** ICLR 2019
- **arXiv:** https://arxiv.org/abs/1810.00760
- **Category:** spd_riemannian

## One-paragraph TL;DR
Adaptive optimizers like Adam and Adagrad rescale each coordinate's gradient by an estimate of its
own curvature (a diagonal Euclidean metric), which makes no sense once parameters live on a curved
manifold (Poincaré ball embeddings, SPD covariance features, Stiefel/rotation constraints). This
paper generalizes Adam/Adagrad/Amsgrad to *any* Riemannian manifold by replacing the update rule's
vector-space additions with the manifold's exponential map (or a retraction), and its moment
statistics with parallel-transported accumulators — giving Riemannian Adam a principled way to
"stay adaptive" without ever leaving the manifold.

## The problem
Naive fixes for optimizing on a manifold (e.g. run ordinary Adam in the ambient coordinates, then
renormalize back onto the manifold after each step — "project and hope") don't actually track the
manifold's own moment statistics, and can behave erratically near the boundary of bounded manifolds
like the Poincaré ball, where the metric itself blows up ($\lambda_x\to\infty$ as $\|x\|\to$
boundary). A step-size that looks small in Euclidean coordinates can be enormous in the Riemannian
metric near such regions, or vice-versa.

## Key idea(s)
- Replace $\theta \leftarrow \theta - \eta\, \hat m/\sqrt{\hat v}$ with
  $\theta \leftarrow \operatorname{Exp}_\theta\big(-\eta\, \hat m / \sqrt{\hat v}\big)$ (or a
  cheaper retraction $\operatorname{Retr}_\theta$) — the update becomes a *tangent vector*
  transported onto the manifold, never an ambient-coordinate subtraction.
- The first/second-moment accumulators $m,v$ from step to step live in *different* tangent spaces
  ($T_{\theta_t}M \ne T_{\theta_{t+1}}M$), so they must be **parallel-transported**,
  $m_{t+1} \leftarrow \tau_{\theta_{t+1}\leftarrow\theta_t}(\beta_1 m_t) + (1-\beta_1)g_t$, before
  being combined with the new gradient.
- Gradients used are **Riemannian gradients** $\operatorname{grad} f(\theta) = g_\theta^{-1}
  \nabla f(\theta)$ (the Euclidean gradient raised by the inverse metric), not raw Euclidean ones.

## The mathematics
On the Poincaré ball $\mathbb{B}^n_\kappa$ with metric $g_x=\lambda_x^2 I$,
$\lambda_x=\frac{2}{1-\kappa\|x\|^2}$, the Riemannian gradient of a Euclidean-differentiable loss
$f$ is $\operatorname{grad}f(x) = \lambda_x^{-2}\nabla f(x)$ (the inverse metric rescales the raw
gradient), and the update retracts via the exponential map
$$x_{t+1} = \operatorname{Exp}_{x_t}\!\big(-\eta_t\,\widehat{m}_t/\sqrt{\widehat v_t}\big),$$
where $\operatorname{Exp}$ is the closed-form Poincaré exponential map (Möbius addition of a
scaled tangent vector). Because $\operatorname{Exp}_x$ always lands strictly inside the ball for
any finite tangent vector, the update is *provably* trapped inside $\mathbb{B}^n_\kappa$ no matter
how large the step — unlike Euclidean coordinate updates, which have no such guarantee.

## Method / architecture
Riemannian Adam / Amsgrad wrap any manifold that exposes `exp`, `log` (or a retraction) and
parallel transport `tau`. The paper proves convergence guarantees analogous to Euclidean Adam
under geodesic convexity assumptions, and shows large empirical gains for hyperbolic embeddings
and SPD-manifold classifiers trained near the boundary/singular region where naive optimizers
misbehave.

## Code
See `code/riemannian_adam_demo.py` — runs a toy loss on the Poincaré ball with (a) **naive
Euclidean Adam** (ambient-coordinate update, no retraction) and (b) **Riemannian Adam** (same
moment-estimation logic, but the final step uses `poincare_exp` as the retraction), and shows the
naive version eventually pushes the iterate's norm past the ball boundary while the Riemannian
version never does.

## Why it matters
This is the optimizer counterpart to SPDNet / hyperbolic-NN layers: it's not enough to build
manifold-respecting *architectures* if the *optimizer* silently assumes flat space. Riemannian
Adam is now the default choice for training hyperbolic embeddings and any model with SPD/Stiefel
parameters.

## Reading questions / discussion
1. Why is $\lambda_x^{-2}\nabla f(x)$ (the Riemannian gradient) the *right* rescaling of the
   Euclidean gradient on the Poincaré ball, rather than, say, $\lambda_x^{-1}$?
2. Parallel-transporting the moment accumulator $m_t$ costs an extra `tau` evaluation every step.
   What would go wrong (concretely) if you skipped it and just reused $m_t$ as-is in the new
   tangent space?
3. Near the ball's boundary $\lambda_x\to\infty$. What does that imply about the *Euclidean* size
   of a fixed-Riemannian-norm step as training approaches the boundary — and why does that make
   "project back onto the ball after an ambient Adam step" a bad approximation there?
