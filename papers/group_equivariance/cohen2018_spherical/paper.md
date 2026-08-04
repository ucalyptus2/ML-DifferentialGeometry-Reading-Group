# Spherical CNNs

- **Authors:** Taco S. Cohen, Mario Geiger, Jonas Köhler, Max Welling
- **Venue/Year:** ICLR 2018
- **arXiv:** https://arxiv.org/abs/1801.10130
- **Category:** group_equivariance

## One-paragraph TL;DR
Generalize convnets to data on the **sphere** $S^2$ (omnidirectional vision, climate, molecular
surfaces) by lifting to the rotation group $SO(3)$: a spherical convolution is the cross-correlation
of a band-limited spherical filter with the signal, computed in the **spherical-harmonic** spectral
domain. The resulting features are **$SO(3)$-equivariant** — rotate the input and the features rotate
identically — which is impossible with planar CNNs.

## The problem
A planar CNN is only *translation*-equivariant. Data living on a sphere (whole-sky images, global
climate fields) has $SO(3)$ symmetry, not $\mathbb{R}^2$ translation. Naively patching a planar CNN
onto a latitude–longitude grid breaks under rotation (the grid itself is anisotropic).

## Key idea(s)
- Use the **spherical Fourier transform**: expand $f \in L^2(S^2)$ as $f(\theta,\phi)=\sum_{\ell,m} \hat f_{\ell m} Y_\ell^m$.
- A spherical *convolution* is a pointwise product in the harmonic domain: $\widehat{(f * \psi)}_{\ell m} = \hat f_{\ell m}\, \hat\psi_{\ell m}$ (since $Y_\ell^m$ diagonalize the Laplacian).
- Lifting to $SO(3)$: the cross-correlation of a filter $\psi$ with $f$ lives on $SO(3)$, parameterized by Euler angles $(\alpha,\beta,\gamma)$, so feature maps become functions on the group.

## The mathematics
Spherical harmonics $Y_\ell^m$ (degree $\ell$, order $m$, $|m|\le\ell$) satisfy
$$\int_{S^2} Y_\ell^m \,\overline{Y_{\ell'}^{m'}}\,d\omega = \delta_{\ell\ell'}\delta_{mm'}.$$
The rotation operator $D^\ell(g)$ (Wigner-D matrices) acts on the harmonic coefficients by
$\hat f \mapsto D^\ell(g)\,\hat f$. Because convolution is pointwise in the spectral domain and the
Wigner-D matrices give the **representation of $SO(3)$**, the lifted feature
$$[f * \psi](g) = \int_{S^2} f(x)\,\overline{\psi(g^{-1}x)}\,d\omega(x)$$
is $SO(3)$-equivariant: $[R_h f * \psi](g) = [f * \psi](h^{-1}g)$.

## Method / architecture
- Compute spherical FFT of the input (truncated at band-limit $L$).
- Multiply by learnable per-degree filters (only $\ell$ matters for isotropic kernels; general
  kernels learn all $\hat\psi_{\ell m}$).
- Inverse transform to get $SO(3)$-valued features; pool; repeat; a final invariant pooling
  (e.g. sum over the group) gives a rotation-*invariant* classifier.

## Code
`code/spherical_conv.py` — a tiny real spherical-harmonic convolution on a grid of points on $S^2$,
verified to be **$SO(3)$-equivariant** numerically (rotate input by $g$ ≡ rotate output by $g$).

## Why it matters
It established the **spectral / harmonic** route to equivariance, generalizing Cohen & Welling's
group-CNNs to continuous groups and seeding the line of spherical / $SO(3)$ / $SE(3)$ networks used
in weather, molecular ML, and 3D vision.

## Reading questions / discussion
1. Why is "convolution on the sphere" *not* just convolution on a latitude–longitude image? Where
   does the grid anisotropy bite?
2. The Wigner-D matrices are a **unitary representation** of $SO(3)$. How does that make the
   spectral-domain convolution automatically equivariant?
3. What is the computational cost of going to band-limit $L$, and why does that motivate fast
   spherical transforms?
