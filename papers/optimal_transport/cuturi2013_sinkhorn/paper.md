# Sinkhorn Distances: Lightspeed Computation of Optimal Transport Distances

- **Authors:** Marco Cuturi
- **Venue/Year:** NeurIPS 2013
- **arXiv:** https://arxiv.org/abs/1306.0895
- **Category:** optimal_transport

## One-paragraph TL;DR
Exact optimal transport (OT) between two discrete distributions is a linear program — polynomial but
$\mathcal{O}(n^3\log n)$ and non-differentiable, making it impractical for large-scale ML. Cuturi adds an
**entropic regularisation** term to the Kantorovich problem, turning the LP into a strictly convex
program whose unique optimum has the scaling form $P=\operatorname{diag}(u)\,K\,\operatorname{diag}(v)$
with $K=\exp(-C/\lambda)$. The optimal scaling vectors $(u,v)$ are found by **Sinkhorn–Knopp
iterations** — alternating normalisations of rows and columns — in $\mathcal{O}(n^2)$ per step, yielding
a smooth, differentiable "Sinkhorn distance" that can be used as a loss in neural networks.

## The problem
The discrete OT (Kantorovich) problem seeks the cheapest coupling between source measure
$\alpha=\sum_i a_i\delta_{x_i}$ and target $\beta=\sum_j b_j\delta_{y_j}$:
$$\mathrm{OT}(\alpha,\beta)=\min_{P\in U(a,b)}\langle P,C\rangle,\qquad
  U(a,b)=\{P\ge 0:\,P\mathbf{1}=a,\;P^\top\mathbf{1}=b\},$$
where $C_{ij}=c(x_i,y_j)$ is the ground cost. Solving this LP with interior-point methods costs
$\mathcal{O}(n^3\log n)$, the solution is a sparse (non-differentiable) vertex of the transport
polytope, and scaling to $n>10^4$ is infeasible. Machine learning needs a fast, smooth surrogate.

## Key idea(s)
- **Entropic regularisation.** Replace the objective with
  $$\min_{P\in U(a,b)}\;\langle P,C\rangle - \frac{1}{\lambda}\,H(P),\qquad
    H(P)=-\sum_{ij}P_{ij}\log P_{ij},$$
  where $\lambda>0$ controls the strength. The $-H$ term is strictly concave in $P$, so the objective is
  strictly convex and the optimum is **unique** and **fully dense** (positive everywhere).
- **Scaling form.** The KKT conditions force the optimum into the form
  $$P^*=\operatorname{diag}(u)\,K\,\operatorname{diag}(v),\qquad K=\exp(-\lambda^{-1}C),$$
  where $u,v>0$ are dual scaling vectors (the exponentials of the dual potentials).
- **Sinkhorn–Knopp iterations.** The marginal constraints $P\mathbf{1}=a$ and $P^\top\mathbf{1}=b$ are
  enforced by the fixed-point iteration
  $$v \leftarrow \frac{b}{K^\top u},\qquad u \leftarrow \frac{a}{K v},$$
  initialised with $u=\mathbf{1}$. Each step is two matrix–vector products — $\mathcal{O}(n^2)$, fully
  parallelisable, and differentiable.
- **Sinkhorn distance.** The final transport cost is $d_\lambda=\langle P^*,C\rangle$. As
  $\lambda\to 0$, $d_\lambda\to\mathrm{OT}(\alpha,\beta)$; for finite $\lambda$ it is a smoothed,
  biased-but-fast approximation.

## The mathematics
**Lagrangian derivation.** The Lagrangian of the entropic problem (with multipliers $f$ for row
constraints, $g$ for column constraints) is
$$\mathcal{L}(P,f,g)=\langle P,C\rangle-\frac{1}{\lambda}H(P)-f^\top(P\mathbf{1}-a)-g^\top(P^\top\mathbf{1}-b).$$
Setting $\partial\mathcal{L}/\partial P_{ij}=0$ gives
$$P_{ij}=\exp\!\bigl(-\lambda(C_{ij}-f_i-g_j)\bigr)=u_i\,K_{ij}\,v_j,$$
with $u_i=e^{f_i/\lambda}$, $v_j=e^{g_j/\lambda}$, $K_{ij}=e^{-C_{ij}/\lambda}$. Substituting into the
marginal constraints yields the Sinkhorn fixed point.

**Convergence.** Sinkhorn iterations are a contraction in the Hilbert projective metric; the log-domain
error $\max_i|\log u_i^{(k)}-\log u_i^{*}|$ decreases geometrically with rate depending on the
"scaling" of $K$. In practice $\sim100$ iterations suffice for machine-precision marginals on small
problems.

**Sinkhorn divergence (bonus).** Cuturi also defines the symmetric, positive-definite Sinkhorn
divergence $S_\lambda(\alpha,\beta)=d_\lambda(\alpha,\beta)-\tfrac{1}{2}d_\lambda(\alpha,\alpha)
-\tfrac{1}{2}d_\lambda(\beta,\beta)$, which vanishes iff $\alpha=\beta$.

## Method / architecture
- **Input.** Marginals $a,b\in\Delta_n$ (probability simplices) and cost matrix $C\in\mathbb{R}^{n\times m}$.
- **Kernel.** Precompute $K=\exp(-C/\lambda)$ once.
- **Iterate.** Run $L$ rounds of $v\leftarrow b/(K^\top u)$, $u\leftarrow a/(Kv)$.
- **Output.** Coupling $P=\operatorname{diag}(u)K\operatorname{diag}(v)$ and Sinkhorn cost
  $\langle P,C\rangle$.
- **Differentiability.** The entire pipeline is a composition of smooth ops, so $\nabla_\theta d_\lambda$
  flows through backprop — the basis for using Sinkhorn as a loss in Wasserstein-style training and
  barycentre computation.

## Code
See `code/sinkhorn.py` — implements Sinkhorn–Knopp iterations (with a numerically stable log-domain
option), computes the Sinkhorn distance, and verifies that (a) the recovered coupling satisfies the
marginal constraints to high precision and (b) the cost converges to the exact OT cost (computed by
brute-force assignment enumeration) as $\lambda\to 0$.

## Why it matters
Sinkhorn distances made optimal transport tractable in machine learning. Before Cuturi, OT was a
theoretical curiosity in ML; after, it became a differentiable loss function (Wasserstein training,
barycentres, domain adaptation, mini-batch OT) running in milliseconds. The entropic-regularisation trick
is now the default in virtually all applied OT — from the Wasserstein GAN gradient penalty to
transport-based cell-trajectory inference — and the Sinkhorn–Knopp iteration remains the computational
engine underneath.

## Reading questions / discussion
1. The entropic term makes the optimum **dense** (every $P_{ij}>0$). When is this a feature (smoothness)
   and when a bug (spurious mass on expensive routes)? How does the choice of $\lambda$ trade these off?
2. The Sinkhorn distance $d_\lambda$ is biased: $d_\lambda>\mathrm{OT}$ for all $\lambda>0$. Why does the
   *divergence* $S_\lambda$ remove the self-transport bias, and what does "positive definite" mean here?
3. Log-domain stabilisation replaces $K=\exp(-C/\lambda)$ by an online absorption of scaling factors.
   What numerical failure does this prevent, and at what computational cost?
4. The complexity is $\mathcal{O}(n^2 L)$ for $L$ iterations. For $n\sim10^6$ (e.g. large-scale matching)
   even this is too slow. What approximations (slicing, low-rank, Nyström) preserve the Sinkhorn spirit?
