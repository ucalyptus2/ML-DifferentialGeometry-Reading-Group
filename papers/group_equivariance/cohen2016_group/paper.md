# Group Equivariant Convolutional Networks

- **Authors:** Taco Cohen, Max Welling
- **Venue/Year:** ICML, 2016
- **arXiv:** https://arxiv.org/abs/1602.07576
- **Category:** group_equivariance

## One-paragraph TL;DR
Standard CNNs are equivariant to *translations* only, because the convolution operator commutes with the translation group acting on the plane. Cohen & Welling observe that any discrete group can be handled the same way: define a **lifting convolution** that maps planar features to features on the group $\mathbb{G}$ (a *feature map on a group*), and a **group convolution** that operates directly on the group. The resulting Group Equivariant Convolutional Network (G-CNN) is equivariant under the *full* group $\mathbb{G}$ — e.g. $p4$ (translations + $90^\circ$ rotations) or $p4m$ (those plus reflections) — which dramatically reduces the number of weights and improves sample efficiency and accuracy. For ML×diff-geometry this is the canonical example of building symmetry into architectures via the action of a group on a homogeneous space.

## The problem
Image recognition features (edges, corners, textures) are roughly rotationally symmetric in natural images, but a translation-equivariant CNN must *re-learn* every rotated copy of a feature from data. This wastes capacity and data. The question: how do we construct a network whose feature maps transform predictably, in a *known* way, under the full symmetry group — not just translations?

## Key idea(s)
- A group action on the input space induces an action on feature maps; we want layers that **commute** with that action (equivariance), so weights are **shared** across the whole orbit of the group.
- **Lifting** expresses a planar feature map $f: \mathbb{Z}^2 \to \mathbb{R}^K$ as a feature map on the group $\mathbb{G}$ (both spatial translation *and* orientation), so that planar rotations become coordinate shifts on the group.
- **Group convolution** generalizes ordinary convolution: instead of sliding a kernel over $\mathbb{Z}^2$, slide it over the group $\mathbb{G}$; this composition law is exactly what makes equivariance hold.

## The mathematics
Let $\mathbb{G}$ be a discrete group acting on the plane. A **feature map** is a function $f: \mathbb{G} \to \mathbb{R}^K$ (all channels processed identically, so drop $K$).

The group $p4$ is the **semi-direct product** of translations and the rotation group:
$$\mathbb{G}=(\mathbb{Z}^2)\rtimes C_4, \qquad (t,r)\cdot (t',r') = (t + r\,t',\, rr'),$$
where $C_4=\{e,r,r^2,r^3\}$ are $90^\circ$ rotations acting on $\mathbb{Z}^2$. $p4m$ adds a reflection $\sigma$ (the dihedral group $D_4$): $\mathbb{G}=(\mathbb{Z}^2)\rtimes D_4$.

A planar feature map $f:\mathbb{Z}^2\to\mathbb{R}$ is **lifted** to the group by, for each group element $g$, correlating the planar kernel $\psi$:
$$[f \star \psi](g) = \sum_{y\in\mathbb{Z}^2} f(y)\,\psi\big(g^{-1} y\big),\qquad g\in\mathbb{G}.$$
This produces a function on the group. Then the **group convolution** of two group feature maps $f,\psi: \mathbb{G}\to\mathbb{R}$ is
$$[f \star \psi](g) = \sum_{h\in\mathbb{G}} f(h)\,\psi\big(g^{-1}h\big).$$

**Equivariance** (the central theorem) states that applying a symmetry $u$ to the input then convolving equals convolving then applying the *left group action* $\mathcal{L}_u$ to the output:
$$\mathcal{L}_u\big([f\star\psi]\big) = \big[\mathcal{L}_u f \star \psi\big], \qquad \mathcal{L}_u[f](g) = f(u^{-1}g).$$
This holds precisely because the group convolution respects the group law $g^{-1}h$. Translation equivariance is the special case where $\mathbb{G}=\mathbb{Z}^2$.

## Method / architecture
- A canonical G-CNN stacks: lifting convolution → nonlinearity (ReLU) → group convolution (several layers) → pooling → fully connected.
- For $p4$: first layer is a **lifting layer** (produces 4 rotated convolutional feature maps), subsequent layers are **group convolutional** (each output channel is a feature map on the group).
- Weight sharing across group orbits means the network explicitly models rotated features rather than rediscovering them.
- Since the group feature maps are functions on a finite group, all sums are finite and the implementation is a modest extension of ordinary conv.

## Code
`code/g4_conv.py` implements, in bare PyTorch: the cyclic rotation $C_4$ acting on the grid, the **lifting convolution** (planar → group), and the **group convolution** (group → group), with correct periodic boundary handling. The `__main__` demo verifies numerically that the group convolution is **equivariant** under left group actions (rotations and translations): it applies a symmetry to the input and checks the output equals the symmetrically-rotated output, within float tolerance. Run with `python code/g4_conv.py`.

## Why it matters
This paper launched the modern line of *equivariant deep learning*. It made explicit that convolution is the special case of **grouplets** / weight-sharing over a group action, and directly motivated the later equivariant steering literature (Cohen–Welling 2017, spherical CNNs, gauge CNNs) that this reading group surveys. The core mathematical move — lifting a feature map on a homogeneous space to a feature map on the group and then convolving there — recurs throughout all of equivariant ML.

## Reading questions / discussion
1. Why does lifting to the group turn a *planar rotation* into a *coordinate shift*? Trace the group law through the lifting formula.
2. The group convolution uses the left action $\mathcal{L}_u f(g)=f(u^{-1}g)$. What changes if we instead use the right action? When do the two coincide?
3. G-CNNs reduce the number of parameters by sharing over group orbits — but they also require group-structured weight sharing. In what sense is this an inductive bias, and when would it hurt?
4. How does the lifting–convolve architecture generalize to a *continuous* group like $\mathrm{SO}(3)$ (see the spherical paper), and what new integration difficulties appear?
