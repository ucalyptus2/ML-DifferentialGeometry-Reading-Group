# FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models

- **Authors:** Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, David Duvenaud
- **Venue/Year:** ICLR 2019
- **arXiv:** https://arxiv.org/abs/1810.01367
- **Category:** flows_odes

## One-paragraph TL;DR
Continuous normalizing flows (CNFs) transform a simple base density into a complex data density by
following a Neural-ODE flow, with a change-of-variables formula whose Jacobian-log-determinant term
becomes a *trace* (instead of a full determinant, as in discrete normalizing flows) — but that
trace still costs $O(d^2)$ to compute exactly for a $d$-dimensional flow. FFJORD replaces the exact
trace with an unbiased **Hutchinson stochastic trace estimator**, making the whole layer $O(d)$ per
step and removing all architectural restrictions (unlike RealNVP/Glow's coupling layers) — the
vector field can be an arbitrary neural net.

## The problem
A normalizing flow needs $\log p_1(x) = \log p_0(f^{-1}(x)) - \log|\det J_{f^{-1}}(x)|$. Discrete
flows (RealNVP, Glow) restrict the architecture (coupling layers, autoregressive structure) purely
so this determinant is cheap. In the continuous-time (ODE) formulation, the log-density obeys
$\frac{d\log p(h(t))}{dt} = -\operatorname{tr}\!\big(\frac{\partial f}{\partial h}(h(t),t)\big)$ —
already much simpler than a full determinant — but the trace of the $d\times d$ Jacobian still
needs $d$ backward passes ($d$ vector-Jacobian products) to compute exactly, which is prohibitive
for high-dimensional data.

## Key idea(s)
- **Instantaneous change of variables**: for $\dot h = f(h,t)$, the log-density evolves by
  $\frac{d\log p}{dt} = -\operatorname{tr}(\partial f/\partial h)$ — a *linear* ODE for
  $\log p$ that can be solved jointly with $h(t)$ by the same ODE solver.
- **Hutchinson's trace estimator**: for any matrix $M$ and random vector $\epsilon$ with
  $\mathbb{E}[\epsilon]=0,\ \operatorname{Cov}(\epsilon)=I$ (e.g. Rademacher $\pm1$ entries),
  $\mathbb{E}_\epsilon\big[\epsilon^\top M\epsilon\big] = \operatorname{tr}(M)$ — an unbiased
  estimate computable with a *single* vector-Jacobian product
  $\epsilon^\top(\partial f/\partial h)\epsilon = \epsilon^\top \nabla_h(f\cdot\epsilon)$, i.e.
  one reverse-mode autodiff call instead of $d$.
- This removes every architectural restriction from the flow's vector field $f$ — any neural net
  works, since we never need $f$'s Jacobian in closed form, only cheap vector-Jacobian products.

## The mathematics
The instantaneous change-of-variables formula is the continuous limit of the standard normalizing
flow identity: as the discrete map $f$ becomes $h+\varepsilon\,\dot h$, $\log|\det J_f|\to
\varepsilon\operatorname{tr}(\partial f/\partial h)+O(\varepsilon^2)$ (using
$\det(I+\varepsilon M)=1+\varepsilon\operatorname{tr}(M)+O(\varepsilon^2)$), giving the ODE for
$\log p$ above. Hutchinson's identity itself is a one-line proof:
$\mathbb{E}[\epsilon^\top M\epsilon] = \mathbb{E}[\sum_{ij}\epsilon_i M_{ij}\epsilon_j] =
\sum_i M_{ii}\mathbb{E}[\epsilon_i^2] = \operatorname{tr}(M)$ using independence/unit variance of
$\epsilon$'s entries.

## Method / architecture
Augment the ODE state with a scalar $\log p(t)$: solve
$\big(\dot h,\ \dot{\log p}\big) = \big(f(h,t),\ -\hat\epsilon^\top(\partial f/\partial h)\hat
\epsilon\big)$ jointly with any black-box solver (identical machinery to Neural ODEs), sampling a
*fresh* $\epsilon$ per solver call (or fixing it per forward pass, trading variance for
determinism). Training maximizes exact log-likelihood via this unbiased estimator.

## Code
See `code/hutchinson_trace.py` — for a small MLP's Jacobian: (1) confirms
$\mathbb{E}_\epsilon[\epsilon^\top J\epsilon]\to\operatorname{tr}(J)$ via Monte-Carlo averaging
over many $\epsilon$ samples (both Rademacher and Gaussian), and (2) integrates a tiny CNF's
log-density change using exact trace vs. single-sample Hutchinson trace and compares final
log-densities.

## Why it matters
FFJORD showed that the *trace* trick removes the architectural bottleneck that had constrained
every prior normalizing flow, at the cost of a controllable variance/compute trade-off — the same
trick (stochastic trace estimation for a Jacobian you can't afford to materialize) reappears
throughout scalable generative modeling and score-based diffusion.

## Reading questions / discussion
1. Why does Hutchinson's estimator need $\operatorname{Cov}(\epsilon)=I$ specifically (not just
   zero mean)? What breaks in the derivation $\mathbb{E}[\epsilon^\top M\epsilon]=
   \sum_i M_{ii}\mathbb{E}[\epsilon_i^2]$ if the entries were correlated?
2. Exact trace costs $O(d)$ VJPs; Hutchinson costs $O(1)$ VJPs but is *unbiased, not exact* per
   sample. How does averaging over $k$ samples trade compute for variance, and when is $k=1$
   (as FFJORD uses per solver step) still practical?
3. Rademacher ($\pm1$) vs Gaussian $\epsilon$: both give an unbiased trace estimator — which has
   lower variance in general, and why might that matter for training stability?
