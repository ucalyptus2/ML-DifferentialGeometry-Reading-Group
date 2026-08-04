# A Riemannian Network for SPD Matrix Learning

- **Authors:** Zhiwu Huang, Luc Van Gool
- **Venue/Year:** AAAI 2017
- **arXiv:** https://arxiv.org/abs/1608.04233
- **Category:** spd_riemannian

## One-paragraph TL;DR
SPDNet builds a deep network whose *features at every layer* are symmetric positive-definite (SPD)
matrices — e.g. covariance descriptors of video/skeleton data — instead of flattening them into
vectors first. Three purpose-built layers (BiMap, ReEig, LogEig) replace the standard
linear/ReLU/flatten stack so that every intermediate representation stays a valid point on the SPD
manifold until the very last step, when `LogEig` maps to the tangent space at the identity for a
normal Euclidean classifier.

## The problem
Region-covariance and other SPD descriptors are common in vision (video, pose, EEG), but a
naive deep net flattens $A\in\mathbb{S}^n_{++}$ into $\mathbb{R}^{n(n+1)/2}$ and treats it as flat
data. That destroys the manifold structure: a linear combination or ReLU applied entrywise can
easily produce a matrix that is no longer SPD (not even symmetric, in general), and Euclidean
distance between covariance matrices is known to be a poor proxy for their true (affine-invariant)
geometric distance.

## Key idea(s)
- **BiMap** layer: $A_{k+1} = W_k A_k W_k^\top$ with $W_k \in \mathbb{R}^{n_{k+1}\times n_k}$
  ($n_{k+1}\le n_k$) **row full-rank** — a *bilinear map* that keeps the output SPD by
  construction (congruence transforms preserve positive-definiteness as long as $W_k$ has full
  row rank), the direct analogue of a linear layer.
- **ReEig** layer: eigen-decompose $A=U\Sigma U^\top$ and rectify small eigenvalues,
  $\operatorname{ReEig}(A) = U\max(\Sigma,\epsilon I)U^\top$ — the SPD analogue of ReLU, keeping
  the result strictly positive-definite (never singular) instead of merely non-negative.
- **LogEig** layer: $\operatorname{LogEig}(A) = U\log(\Sigma)U^\top$ — the Riemannian logarithm at
  the identity, flattening the manifold into its tangent space so a standard (Euclidean) fully
  connected + softmax head can be applied at the very end.
- Backprop through eigendecomposition uses a matrix-valued chain rule (Ionescu et al.) so gradients
  flow correctly through $U,\Sigma$.

## The mathematics
SPD matrices $\mathbb{S}^n_{++}=\{A=A^\top: x^\top A x>0\ \forall x\ne 0\}$ form a Riemannian
manifold under the affine-invariant metric $g_A(X,Y)=\operatorname{tr}(A^{-1}XA^{-1}Y)$, with
$$\operatorname{Exp}_A(V)=A^{1/2}\exp(A^{-1/2}VA^{-1/2})A^{1/2},\qquad
\operatorname{Log}_A(B)=A^{1/2}\log(A^{-1/2}BA^{-1/2})A^{1/2}.$$
`LogEig` is exactly $\operatorname{Log}_I(A) = \log(A) = U\log(\Sigma)U^\top$ — the logarithmic map
*at the identity matrix* $I$, which linearizes the whole manifold into a single tangent space
$T_I\mathbb{S}^n_{++}$ (the space of symmetric matrices) where ordinary Euclidean layers are valid.
`BiMap`'s congruence action $A\mapsto WAW^\top$ is precisely the natural $GL(n)$ group action under
which the affine-invariant metric is *invariant*: $d(WAW^\top, WBW^\top) = d(A,B)$ for any
invertible $W$ — so BiMap is a distance-preserving (up to $W$'s condition number) map on the
manifold, not an arbitrary matrix product.

## Method / architecture
Stack of `[BiMap -> ReEig] x K -> LogEig -> FC -> softmax`, trained end-to-end with matrix
backprop. Weight matrices $W_k$ are optimized on the Stiefel manifold (orthogonality constraint) via
Riemannian SGD so that BiMap stays full-rank throughout training.

## Code
See `code/spdnet_layers.py` — implements `bimap`, `reeig`, `logeig`, chains them into a tiny
2-layer SPDNet, and checks (a) every intermediate activation stays SPD (Cholesky succeeds,
eigenvalues $>0$) and (b) `LogEig` is the exact inverse of `spd_exp` at the identity, reusing
`spd_log` from `code/manifold_ops.py`.

## Why it matters
SPDNet is the template for "manifold-valued deep learning": design each layer as a map that
provably preserves the data's manifold, and only linearize (via `Log`) at the point where you need
a standard head. The same recipe (congruence layers + eigenvalue rectification + tangent-space
readout) reappears in later SPD/Grassmann/hyperbolic architectures.

## Reading questions / discussion
1. Why does `BiMap` require $W_k$ to have full row rank? What breaks in the congruence argument
   if $W_k$ is rank-deficient?
2. `ReEig` clips small eigenvalues instead of negative ones like ReLU — why can an SPD matrix's
   eigenvalues never be negative in the first place, and what does clipping *near zero* protect
   against numerically?
3. `LogEig` is only exact at the identity ($\operatorname{Log}_I$). What would change if the last
   layer instead computed $\operatorname{Log}_{\bar A}$ at the *Fréchet mean* $\bar A$ of the batch
   instead of at $I$?
