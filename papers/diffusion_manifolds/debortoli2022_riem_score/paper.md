# Riemannian Score-Based Generative Modelling

- **Authors:** Valentin De Bortoli, James Thornton, Jeremy Hutchinson, Arnaud Doucet
- **Venue/Year:** NeurIPS 2022
- **arXiv:** https://arxiv.org/abs/2202.02763
- **Category:** diffusion_manifolds

## One-paragraph TL;DR
The authors extend score-based generative modelling (Song et al. 2021) from flat $\mathbb{R}^d$ to a
complete Riemannian manifold $(\mathcal{M}, g)$. The forward noising process is **Riemannian Brownian
motion**, realised numerically by the Eells–Elworthy–Malliavin construction — at each step sample
isotropic Gaussian noise in the tangent space, project onto $T_x\mathcal{M}$, and retract via the
exponential map. The score $\nabla_{\mathcal{M}}\log p_t$ is learned with a denoising score-matching
objective, and sampling runs the time-reversed SDE (Anderson 1982) backwards from the uniform
distribution. Under an isometric-embedding assumption the reverse process is provably close to the
true time-reversal, recovering the data distribution on manifolds such as the sphere, $\mathrm{SO}(n)$,
and the SPD cone.

## The problem
Euclidean score-based diffusion models corrupt data with additive Gaussian noise and learn to reverse
that corruption. On a manifold there is no canonical "add Gaussian noise" operation — addition is not
even defined — and the Laplace–Beltrami operator (the natural generator of diffusion) does not have a
trivial closed-form SDE. The challenge is to (a) define a tractable forward diffusion whose marginal
law is computable, (b) derive a reverse SDE that stays on the manifold, and (c) train a score network
whose output lives in the correct tangent space.

## Key idea(s)
- **Brownian motion via the stochastic exponential map.** Riemannian Brownian motion with generator
  $\tfrac{1}{2}\Delta_{\mathcal{M}}$ is simulated by the Stratonovich SDE
  $$\mathrm{d}X_t = P_{X_t}\,\circ\,\mathrm{d}W_t,$$
  where $P_x:\mathbb{R}^d\to T_x\mathcal{M}$ is the orthogonal projection onto the tangent space (under
  an isometric embedding $\mathcal{M}\hookrightarrow\mathbb{R}^d$). One Euler step is
  $X_{t+\Delta t}=\exp_{X_t}\!\bigl(\sqrt{\Delta t}\,P_{X_t}\xi\bigr)$, $\xi\sim\mathcal{N}(0,I)$.
- **Reverse SDE on the manifold.** Anderson's time reversal gives
  $$\mathrm{d}X_t = -\tfrac{1}{2}\,\nabla_{\mathcal{M}}\log p_t(X_t)\,\mathrm{d}t + P_{X_t}\,\circ\,\mathrm{d}\bar W_t,$$
  discretised as $X_{t-\Delta t}=\exp_{X_t}\!\bigl(\sqrt{\Delta t}\,P_{X_t}\xi - \tfrac{\Delta t}{2}\,s_\theta(X_t,t)\bigr)$,
  where $s_\theta\approx\nabla_{\mathcal{M}}\log p_t$.
- **Denoising score matching on manifolds.** The score is trained with
  $\mathcal{L}=\mathbb{E}\bigl[\|s_\theta(X_t,t)-\nabla_{\mathcal{M}}\log p_{t|0}(X_t\mid X_0)\|^2_g\bigr]$.
  For small $t$ the Brownian transition is approximately Gaussian in $T_{X_0}\mathcal{M}$, yielding the
  closed-form target $-\tfrac{1}{t}\log_{X_t}(X_0)$, where $\log$ is the Riemannian logarithmic map.

## The mathematics
On the unit sphere $\mathbb{S}^{d-1}\subset\mathbb{R}^d$ the tangent space at $x$ is
$T_x\mathbb{S}^{d-1}=\{v\in\mathbb{R}^d:v\cdot x=0\}$ and the projection is
$P_x(v)=v-(v\cdot x)\,x$. The exponential and logarithmic maps are
$$\exp_x(v)=\cos\|v\|\,x+\frac{\sin\|v\|}{\|v\|}\,v,\qquad
  \log_x(y)=\frac{\arccos(x\cdot y)}{\sqrt{1-(x\cdot y)^2}}\,(y-(x\cdot y)x).$$

**Forward (heat) step.** Given $X_0=x_0$, simulate $n$ steps of size $\Delta t=T/n$:
$$X_{k+1}=\exp_{X_k}\!\bigl(\sqrt{\Delta t}\,P_{X_k}\xi_k\bigr),\qquad \xi_k\sim\mathcal{N}(0,I_d).$$
The marginal density $p_t$ satisfies the heat equation $\partial_t p_t=\tfrac{1}{2}\Delta_{\mathbb{S}}\,p_t$ and
converges to the uniform measure as $t\to\infty$.

**Denoising target.** For the Gaussian-in-tangent approximation, the conditional score is
$$\nabla_{\mathcal{M}}\log p_{t|0}(x_t\mid x_0)\;\approx\;-\frac{1}{t}\,\log_{x_t}(x_0),$$
a tangent vector at $x_t$ pointing back toward $x_0$, scaled by $1/t$.

**Reverse step.** Let $s_\theta(x,t)$ be a network whose output is projected to $T_x\mathbb{S}^{d-1}$.
Then
$$X_{k-1}=\exp_{X_k}\!\bigl(\sqrt{\Delta t}\,P_{X_k}\xi_k-\tfrac{\Delta t}{2}\,s_\theta(X_k,t_k)\bigr),$$
with $t_k=T-k\Delta t$, initialised at uniform $\mathrm{Unif}(\mathbb{S}^{d-1})$.

## Method / architecture
- **Score network.** An MLP $f_\theta:(x,t)\mapsto v\in\mathbb{R}^d$ with the output projected to the
  tangent space: $s_\theta(x,t)=P_x\,f_\theta(x,t)$. Time $t$ is fed as an extra scalar input.
- **Training.** For each batch: sample $X_0$ from data, run $k$ forward steps to obtain $X_t$,
  compute the denoising target $-\tfrac{1}{t}\log_{X_t}(X_0)$, and minimise the squared tangent-norm loss.
- **Sampling.** Initialise from $\mathrm{Unif}(\mathbb{S}^{d-1})$ (the $t\to\infty$ marginal) and integrate
  the reverse SDE for $n$ steps.
- **Extensions in the paper.** The framework applies to any complete Riemannian manifold with a computable
  exponential map; the authors also treat $\mathrm{SO}(n)$ and SPD manifolds, and give convergence proofs
  under a finite-distortion isometric-embedding assumption.

## Code
See `code/riemann_score_sphere.py` — implements batched `exp`/`log`/tangent-projection on $\mathbb{S}^2$,
the forward heat (Brownian via exponential map), a small tangent-projected score network trained by
denoising score matching, and the reverse SDE sampler. The demo verifies that generated samples
concentrate near a two-mode target (mean geodesic distance to the nearest mode $\ll\pi/2$, the
uniform-sphere baseline).

## Why it matters
This was the first work to put score-based diffusion on a rigorous Riemannian footing, unifying the
Euclidean theory with the geometry of the data manifold. The Eells–Elworthy–Malliavin / exponential-map
discretisation became the standard recipe for manifold diffusion, and the denoising target
$-\tfrac{1}{t}\log_{x_t}(x_0)$ via the Riemannian $\log$ map is now the canonical training signal. It
opened the door to diffusion on Lie groups, SPD matrices, spheres, and product manifolds used in
molecular and directional statistics applications.

## Reading questions / discussion
1. The forward SDE uses $P_x\,\circ\,\mathrm{d}W_t$ (Stratonovich). What goes wrong if you use the Itô
   form directly, and why does the Stratonovich–Itô correction involve the trace of the connection?
2. The denoising target $-\tfrac{1}{t}\log_{x_t}(x_0)$ is exact only for small $t$ (Gaussian-in-tangent
   approximation). How does the paper handle the bias at larger $t$, and what manifold-specific
   correction does curvature introduce?
3. On the sphere the uniform measure is the $t\to\infty$ limit of Brownian motion. On a compact manifold
   with boundary (or a non-compact manifold) what plays the role of the "prior" for reverse sampling?
4. The convergence proof requires an isometric embedding $\mathcal{M}\hookrightarrow\mathbb{R}^d$ with
   bounded second fundamental form. Which manifolds of practical interest violate this, and how might one
   still extend the method?
