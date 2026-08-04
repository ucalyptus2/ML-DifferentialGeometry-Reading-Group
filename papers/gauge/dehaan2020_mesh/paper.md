# Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs

- **Authors:** Pim de Haan, Maurice Weiler, Taco Cohen, Max Welling
- **Venue/Year:** ICLR 2021
- **arXiv:** https://arxiv.org/abs/2003.05425
- **Category:** gauge

## One-paragraph TL;DR
Makes the gauge-equivariant framework concrete for **triangle meshes**: features are tangent
**vector/tensor fields** on the surface, and convolution between neighbouring vertices requires
**parallel-transporting** the neighbour's feature into the local frame before applying an
*anisotropic* (direction-dependent) kernel. The result is a CNN that respects the surface's gauge
symmetry and can use direction information (unlike isotropic graph CNNs).

## The problem
On a mesh, each vertex has its own tangent frame (a gauge). A vector feature at vertex $j$ cannot
be added to one at vertex $i$ directly — they live in different frames. Ignoring this (as a plain
GCN does) destroys directional information and breaks gauge-equivariance.

## Key idea(s)
- Each directed edge $(i \leftarrow j)$ carries a **parallel-transport** map
  $\tau_{i \leftarrow j} \in G$ (for $G = SO(2)$, a rotation by the angle between the frames).
- The convolution is
  $$ (f * K)(i) = \sum_{j \sim i} K\!\big(\tau_{i \leftarrow j}\big)\, \tau_{i \leftarrow j}\, f(j), $$
  where $K: G \to \mathrm{Hom}(V,V)$ is an **anisotropic kernel** (depends on the transport angle).
- Gauge-equivariance: under a frame flip $g_i$ at $i$, $f(i) \mapsto g_i f(i)$ and
  $\tau_{i \leftarrow j} \mapsto g_i \tau_{i \leftarrow j} g_j^{-1}$; the kernel transforms
  compatibly so $(f*K)(i) \mapsto g_i (f*K)(i)$ — the output is a genuine section.

## The mathematics
For $G = SO(2)$, decompose the kernel into angular Fourier modes
$$ K(\theta) = \sum_k \hat K_k\, e^{i k \theta}, $$
with $\hat K_k$ an intertwiner between representation weights $m$ and $m+k$ (the selection rule
$m_{\text{out}} = m_{\text{in}} + k$). This is exactly the gauge-equivariant counterpart of a
spherical-harmonic filter, but **local** to each vertex's tangent space.

## Method / architecture
- Build a mesh with per-edge transport angles (from the local frames).
- Gauge-equivariant convolution layer: transport neighbour features, apply angular kernel, sum.
- Stack layers + gauge-invariant pooling for classification.

## Code
`code/gauge_mesh_conv.py` — a toy gauge-equivariant mesh conv on a small triangle mesh: per-edge
$SO(2)$ parallel-transport angles, an anisotropic (angle-dependent) kernel applied to transported
vector features, and a **gauge-equivariance check** under local frame rotations.

## Why it matters
It's the first practically usable gauge-equivariant model on real 3D meshes, showing that
directional/anisotropic filters — impossible for plain GCNs — become available once you respect
the gauge structure. It directly instantiates the "2-categorical" side of the geometric-DL blueprint.

## Reading questions / discussion
1. Why does an *isotropic* (angle-independent) kernel reduce to ordinary graph convolution, and
   what expressivity is lost?
2. The parallel-transport angle is data (the mesh geometry). What happens to equivariance if the
   transport is approximated (e.g. shortest-path rotation)?
3. Compare this to E(n)-EGNN: EGNN uses invariant distances and avoids transport; gauge mesh CNNs
   use transport and gain anisotropy. When is each the right tool?
