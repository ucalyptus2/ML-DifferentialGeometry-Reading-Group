# Hyperbolic Neural Networks

- **Authors:** Octavian-Eugen Ganea, Gary Bécigneul, Thomas Hofmann
- **Venue/Year:** NeurIPS 2018
- **arXiv:** https://arxiv.org/abs/1805.09112
- **Category:** hyperbolic

## One-paragraph TL;DR
This paper ported the core building blocks of deep learning—linear layers, feed-forward networks, and recurrent GRUs—from Euclidean $\mathbb{R}^n$ to the Poincaré ball model of hyperbolic space. The key primitive is the **Möbius matrix-vector product**, which wraps a Euclidean linear map between the logarithmic and exponential maps at the origin, turning $Wx + b$ into $\exp_0(W\log_0(x)) \oplus b$. Because hyperbolic space encodes hierarchical structure exponentially more efficiently than flat space, the resulting hyperbolic networks achieve gains on entailment detection and taxonomy embedding tasks where data is tree-like.

## The problem
Standard neural networks operate in flat Euclidean space. When the data has latent hierarchical structure—taxonomies, ontologies, entailment graphs—the optimal representation space is tree-like, and Euclidean embeddings require exponentially many dimensions to faithfully capture the branching. Poincaré embeddings (Nickel & Kiela 2017) showed that individual points in the Poincaré ball encode hierarchy, but they did not define *neural network layers* that natively operate in hyperbolic space. Without such layers, one cannot build deep architectures (RNNs, classifiers) that keep representations in the manifold across layers.

## Key idea(s)
- **Möbius linear layer.** Replace the Euclidean affine map $x \mapsto Wx + b$ with a hyperbolic analogue. First map the ball point to the tangent space at the origin via $\log_0$, apply the Euclidean matrix $W$, then map back via $\exp_0$, and add the bias via Möbius addition: $H_{\mathbf{W}}^{\mathbf{b}}(x) = \exp_0(W\log_0(x)) \oplus_c b$.
- **Möbius addition as the vector-space replacement.** In the Poincaré ball, the group operation that replaces $+$ is the Möbius addition $\oplus_c$, which is non-commutative and non-associative but satisfies $\exp_x(v) = x \oplus_c \tanh(\lambda_x\|v\|/2)\,v/\|v\|$.
- **Möbius GRU.** The GRU's gate computations $W h + U x$ are replaced by Möbius linear combinations $(W \otimes h) \oplus (U \otimes x)$, and element-wise Hadamard products (used in gating) are performed in the tangent space, mapped out via $\exp$.

## The mathematics
The Poincaré ball of curvature $-c$ is $\mathbb{B}_c^n = \{x \in \mathbb{R}^n : c\|x\|^2 < 1\}$ with the Riemannian metric $g_x = \lambda_x^2\, g_E$ where $\lambda_x = \frac{2}{1 - c\|x\|^2}$.

**Möbius addition:**
$$x \oplus_c y = \frac{(1 + 2c\langle x,y\rangle + c\|y\|^2)\,x + (1 - c\|x\|^2)\,y}{1 + 2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}$$

**Maps at the origin** (where $\lambda_0 = 2$):
$$\exp_0(v) = \tanh(\sqrt{c}\,\|v\|)\,\frac{v}{\sqrt{c}\,\|v\|}, \qquad \log_0(y) = \operatorname{artanh}(\sqrt{c}\,\|y\|)\,\frac{y}{\sqrt{c}\,\|y\|}$$

**Möbius matrix-vector product:**
$$M_{\mathbf{W}} \otimes_c x \;=\; \exp_0\!\big(W\,\log_0(x)\big)$$

This is the natural hyperbolic generalization of $Wx$: the map $\log_0$ identifies the tangent space at the origin with $\mathbb{R}^n$ (isometrically up to the conformal factor), the Euclidean linear map $W$ acts on the tangent vector, and $\exp_0$ retracts back into the ball.

**Möbius linear layer:**
$$H_{\mathbf{W}}^{\mathbf{b}}(x) = \big(M_{\mathbf{W}} \otimes_c x\big) \oplus_c b = \exp_0\!\big(W\,\log_0(x)\big) \oplus_c b$$

where $b \in \mathbb{B}_c^n$ is a hyperbolic bias. The Möbius GRU replaces every $W h + U x$ in the standard GRU with $(M_W \otimes_c h) \oplus_c (M_U \otimes_c x)$, and the Hadamard product $r \odot h$ (gating) is computed as $\exp_0(\log_0(r) \odot \log_0(h))$—i.e., multiply in the tangent space, then map back.

## Method / architecture
- **Möbius linear layer** as above; stacked with a hyperbolic nonlinearity $\sigma_\otimes(x) = \exp_0(\sigma(\log_0(x)))$ for a Euclidean activation $\sigma$.
- **Möbius GRU:** standard GRU cell with all arithmetic replaced by Möbius operations. The update and reset gates use Möbius-linear transforms of the hidden state and input; the candidate state uses a Möbius Hadamard product for gating.
- **Training:** Riemannian Adam (RAdam) or RSGD, with the Euclidean gradient projected to the tangent space via the conformal factor $\lambda_x^2$, and retraction via $\exp$ (or projection back into the ball).
- Evaluated on entailment detection (WordNet hypernymy) and complex symbolic data (DeductionDB), where hyperbolic GRUs outperform Euclidean baselines.

## Code
See `code/mobius_linear.py` — implements $\log_0$, $\exp_0$, the Möbius matrix-vector product, and the full `MobiusLinear` layer (with Möbius bias addition), plus a forward/backward demo verifying that inputs and outputs stay inside the ball and that gradients flow. Official implementation: https://github.com/dalab/hyperbolic_nn

## Why it matters
This paper established that one can build *general-purpose* neural architectures natively in hyperbolic space, not just embeddings. The $\log_0$/$\exp_0$ wrapping pattern became the standard recipe: any Euclidean layer can be "hyperbolized" by moving to the tangent space, applying the Euclidean operation, and mapping back. Every subsequent hyperbolic deep learning paper—HGCN (Chami et al. 2019), hyperbolic attention, hyperbolic VAEs—builds on these primitives.

## Reading questions / discussion
1. The Möbius linear layer applies $W$ in the tangent space at the *origin*. What is lost by always using the origin as the base point, versus applying $W$ in the tangent space at $x$ itself?
2. Möbius addition is non-associative: $(x \oplus y) \oplus z \neq x \oplus (y \oplus z)$ in general. How does this affect the Möbius GRU's gating, and what numerical workarounds are needed?
3. The Hadamard product is performed via $\exp_0(\log_0(r) \odot \log_0(h))$. Is this the *only* sensible hyperbolic generalization of element-wise gating, or are there alternatives (e.g., using the gyrovector space structure)?
4. For data that is *not* hierarchical, would you expect the Möbius layer to hurt or help? What diagnostic would tell you whether hyperbolic geometry is the right inductive bias?