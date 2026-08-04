# Geometric GAN

- **Authors:** Jae Hyun Lim, Jong Chul Ye, Yeong Jun Kang, Se Young Park
- **Venue/Year:** arXiv 2017
- **arXiv:** https://arxiv.org/abs/1705.02894
- **Category:** optimal_transport

## One-paragraph TL;DR
Geometric GAN reframes adversarial training as a **maximum-margin separation** problem: the
discriminator learns a hyperplane (in data or feature space) that separates real from generated samples
with the widest possible margin, exactly as a support vector machine does for two-class classification.
The resulting **hinge loss** for the discriminator — $\max(0,1-D(x))$ for real, $\max(0,1+D(G(z)))$ for
fake — replaces the Jensen–Shannon / Wasserstein objectives of earlier GANs, and the generator pushes
its samples across the margin via a one-sided hinge. The paper shows this geometric view unifies and
stabilises GAN training, connects to optimal transport (the separator approximates the OT boundary), and
outperforms DCGAN and WGAN on image benchmarks.

## The problem
The original GAN (Goodfellow et al. 2014) frames generation as a two-player game with a
Jensen–Shannon-divergence objective. The JS divergence saturates when the discriminator wins (vanishing
gradients for the generator), is constant when supports are disjoint (mode collapse), and provides no
geometric control over how the two distributions are separated. Wasserstein GAN (Arjovsky et al. 2017)
fixes the gradient issue with a Lipschitz-1 critic but requires weight clipping or gradient penalty and
its loss has no margin interpretation. The question: is there a GAN objective with a clean geometric
meaning (a separating margin) that is both stable and competitive?

## Key idea(s)
- **Discriminator as a maximum-margin separator.** Instead of a probabilistic classifier
  $D(x)=\sigma(f_\theta(x))$, the Geometric GAN discriminator $f_\theta:\mathbb{R}^d\to\mathbb{R}$ outputs
  an unnormalised score whose sign classifies real ($f>0$) vs. fake ($f<0$). The hinge loss
  $$\mathcal{L}_D=\mathbb{E}_{x\sim p_{\text{data}}}[\max(0,\,1-f_\theta(x))]+\mathbb{E}_{z\sim p_z}[\max(0,\,1+f_\theta(G_\phi(z)))]$$
  is exactly the soft-margin SVM objective: it penalises any real sample inside the $1$-margin band
  ($f<1$) and any fake sample inside the $-1$-margin band ($f>-1$), and is zero otherwise.
- **One-sided hinge generator.** The generator minimises
  $$\mathcal{L}_G=\mathbb{E}_{z\sim p_z}[\max(0,\,1-f_\theta(G_\phi(z)))],$$
  pushing fake samples past the $+1$ margin (into "real" territory). The gradient vanishes once fakes
  cross the margin — a built-in stopping criterion that prevents the generator from overshooting.
- **Geometric / OT connection.** The separating hyperplane $\{x:f_\theta(x)=0\}$ approximates the
  boundary of the optimal transport map between $p_{\text{data}}$ and $p_G$. The margin $1$ plays the
  role of the OT "transport distance": the wider the margin, the farther fakes must travel to be
  classified as real, analogous to minimising a transport cost.

## The mathematics
**Hinge loss as margin maximisation.** For a linear discriminator $f(x)=w^\top x+b$, the SVM hard-margin
problem is $\min_{w,b}\tfrac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i+b)\ge 1$ with labels $y_i\in\{+1,-1\}$.
The soft-margin relaxation replaces constraints with penalties $\max(0,1-y_i f(x_i))$, which is precisely
the hinge. Geometric GAN uses a *nonlinear* $f_\theta$ (a neural net) but keeps the same hinge penalty,
so the discriminator maximises a data-dependent margin between the real and fake distributions.

**Separator and OT.** When $p_{\text{data}}$ and $p_G$ are absolutely continuous, the optimal transport
map $T^*$ satisfies $T^*(x)-x\propto\nabla f^*(x)$ where $f^*$ is a Kantorovich potential. The GAN
discriminator score $f_\theta$ approximates this potential; the hinge margin enforces that the
approximation has a non-trivial "gap" (the $1$-margin), stabilising the game by preventing the
discriminator from becoming arbitrarily confident.

**Generator dynamics.** The one-sided hinge gradient is
$$\nabla_\phi\mathcal{L}_G=-\mathbb{E}\bigl[\mathbf{1}[f_\theta(G_\phi(z))<1]\,\nabla_\phi f_\theta(G_\phi(z))\bigr],$$
which is active only for fakes still on the wrong side of the margin. This is the geometric analogue of
"push until you cross the boundary, then stop."

## Method / architecture
- **Discriminator** $f_\theta$: a CNN or MLP mapping data to a scalar (no sigmoid). Trained with the
  two-sided hinge $\mathcal{L}_D$ above, optionally with a spectral-normalisation or weight-clip
  regulariser to control the margin geometry (the paper uses a relaxed Lipschitz constraint).
- **Generator** $G_\phi$: maps latent $z\sim\mathcal{N}(0,I)$ to data space. Trained with the one-sided
  hinge $\mathcal{L}_G$.
- **Alternating optimisation.** $k_D$ discriminator updates per generator update (as in WGAN), but the
  hinge loss provides bounded gradients on both sides, avoiding the saturation of JS-GAN and the
  unconstrained growth of the WGAN critic.
- **Loss landscape.** Unlike the JS divergence (constant on disjoint supports) or the Wasserstein
  distance (unbounded), the hinge loss is piecewise linear with a fixed margin, giving a Lipschitz,
  non-vanishing gradient wherever samples are within the margin band.

## Code
See `code/geometric_gan.py` — implements the hinge discriminator and one-sided-hinge generator losses on
a toy 2D target (a Gaussian cluster), trains a small MLP generator + discriminator, and verifies that
generated samples migrate toward the data cluster (mean displacement from the data centre shrinks
dramatically from the initial random-spread baseline).

## Why it matters
Geometric GAN gave the GAN objective a clean margin-based interpretation, bridging adversarial training
with the well-understood theory of support vector machines and optimal transport. The hinge-GAN
formulation it popularised became one of the standard loss functions in modern conditional and
unconditional image generation (including BigGAN-style architectures), prized for its training stability
and bounded, non-saturating gradients. It also planted the idea that the discriminator is learning a
*transport potential*, influencing later work on OT-based generative models.

## Reading questions / discussion
1. The generator's one-sided hinge has **zero gradient** once $f_\theta(G(z))\ge 1$. Compare this to the
   non-saturating GAN loss $-\log D(G(z))$ and the WGAN loss $-f(G(z))$: which gives the most useful
   gradient signal when the generator is already "winning," and what failure modes does each have?
2. The margin constant $1$ is arbitrary. What happens geometrically as you scale it? Does it correspond
   to a Lipschitz constraint on $f_\theta$, and how does this relate to WGAN-GP's gradient penalty?
3. The paper claims the discriminator approximates a Kantorovich potential. Under what conditions on
   $p_{\text{data}}$ and $p_G$ is the SVM separator a good approximation of the OT boundary, and when does
   it fail?
4. Unlike WGAN, the hinge-GAN discriminator is not required to be $1$-Lipschitz. What regularisation
   (spectral norm, weight clipping) is needed in practice, and what does its absence cause?
