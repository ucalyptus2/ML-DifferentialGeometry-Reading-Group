# Surface Networks

- **Authors:** Ilya Kostrikov, Zhongshi Jiang, Daniele Panozzo, Denis Zorin, Joan Bruna
- **Venue/Year:** CVPR 2018
- **arXiv:** https://arxiv.org/abs/1705.10819
- **Category:** geodesics_shape

## One-paragraph TL;DR
This paper proposes using intrinsic **differential operators on surfaces**—specifically the Laplace–Beltrami operator $\Delta$ and the intrinsic Dirac operator $\mathcal{D}$—as the aggregation matrices in graph-neural-network-style layers on triangle meshes. Where a standard GCN uses the adjacency matrix to propagate information, Surface Networks use $\Delta$ (which encodes intrinsic geometry and is invariant to isometry) or $\mathcal{D}$ (which additionally encodes extrinsic curvature via the spin connection). The paper shows that these operators give better shape analysis results than adjacency-based or spectral GCNs, especially for tasks where curvature matters.

## The problem
Shape analysis on triangle meshes requires network layers that are *intrinsic*—invariant to how the mesh is triangulated and to isometric deformations of the surface. Standard GCNs use the adjacency matrix, which encodes only combinatorial connectivity, not geometry. Spectral GCNs use the eigendecomposition of the graph Laplacian, which captures intrinsic geometry but is expensive to compute and not localized. The paper asks: can we build network layers from the *actual* differential operators of the surface (Laplacian, Dirac) rather than from combinatorial approximations?

## Key idea(s)
- **Laplacian layer.** Replace the GCN propagation $\hat{A} H W$ with a Laplacian-based diffusion $(I - t\Delta) H W$, where $\Delta$ is the discrete Laplace–Beltrami operator (cotangent or uniform weights). This makes the layer intrinsic (isometry-invariant).
- **Dirac layer.** Use the intrinsic Dirac operator $\mathcal{D}$, a first-order differential operator that squares to the Laplacian ($\mathcal{D}^2 = \Delta$) and encodes extrinsic curvature through the spin connection. This captures bending information that the Laplacian alone misses.
- **Both as drop-in propagation matrices.** The operators are precomputed from the mesh and used as fixed (non-trainable) propagation matrices, with trainable weight matrices $W$ applied as in GCN.

## The mathematics
Let $M$ be a smooth surface (2-manifold) embedded in $\mathbb{R}^3$ with Riemannian metric $g$ induced by the embedding.

**Laplace–Beltrami operator.** The intrinsic Laplacian on $(M, g)$ is:
$$\Delta_g f = \frac{1}{\sqrt{|g|}} \partial_i\!\left(\sqrt{|g|}\, g^{ij}\, \partial_j f\right)$$
On a triangle mesh, the discrete Laplace–Beltrami operator (cotangent formula) is:
$$(\Delta f)_i = \frac{1}{2A_i} \sum_{j \in \mathcal{N}(i)} (\cot \alpha_{ij} + \cot \beta_{ij})\,(f_j - f_i)$$
where $A_i$ is the Voronoi area of vertex $i$, and $\alpha_{ij}, \beta_{ij}$ are the angles opposite edge $(i,j)$. The combinatorial (uniform-weight) approximation uses $\Delta = D - A$.

**Dirac operator.** The intrinsic Dirac operator on a surface is a first-order operator acting on spinors (sections of the spinor bundle). In the discrete setting, it can be approximated via the connection Laplacian or through the squared Dirac operator $\mathcal{D}^2 = \Delta$ plus extrinsic curvature terms. Key property:
$$\mathcal{D}^2 = \Delta + \tfrac{1}{2} K$$
where $K$ is the Gaussian curvature, linking the Dirac operator to extrinsic bending.

**Surface Network layer.** Given node features $H \in \mathbb{R}^{n \times d}$ and a surface operator $S$ (either $\Delta$ or $\mathcal{D}$):
$$H^{(\ell+1)} = \sigma\!\left(S\, H^{(\ell)}\, W^{(\ell)}\right)$$
or, for diffusion-style layers:
$$H^{(\ell+1)} = \sigma\!\left((I - t\,\Delta)\, H^{(\ell)}\, W^{(\ell)}\right)$$
where $t > 0$ is a diffusion time parameter and $W^{(\ell)}$ is a trainable weight matrix. The spectral variant diagonalizes $\Delta = U \Lambda U^\top$ and applies a learnable filter $g(\Lambda)$ in the spectral domain:
$$H^{(\ell+1)} = \sigma\!\left(U\, g(\Lambda)\, U^\top\, H^{(\ell)}\, W^{(\ell)}\right)$$

## Method / architecture
- **Input:** Triangle mesh (vertices + faces) with per-vertex features (e.g. SHOT, GPS, or coordinates).
- **Precompute operators:** Build $\Delta$ (cotangent or uniform Laplacian) and/or $\mathcal{D}$ (Dirac operator) from the mesh geometry.
- **Stack Surface Network layers** using $\Delta$, $\mathcal{D}$, or their spectral decompositions as propagation matrices, with trainable $W^{(\ell)}$ and nonlinearity $\sigma$.
- **Tasks:** Shape classification (FAUST, SHREC), correspondence, segmentation.
- The paper compares Laplacian vs. Dirac variants and shows that the Dirac operator's extrinsic curvature sensitivity helps when bending matters.

## Code
See `code/surface_network.py` — builds a UV-sphere mesh, computes the combinatorial graph Laplacian $L = D - A$, implements a `DiffusionLayer` $H' = (I - tL) H W$ and a `SpectralLayer` $H' = U\,g(\Lambda)\,U^\top H W$ with a learnable spectral filter, and runs a heat-diffusion demo plus a gradient check. Uses `sphere_log` from the shared library to verify tangent-space directions on the unit-sphere mesh. Official implementation: https://github.com/3DVision-Ident/Surface-Networks

## Why it matters
Surface Networks formalized the idea that the *right propagation matrix* for a mesh GNN is a differential operator of the surface, not the adjacency matrix. This directly influenced:
- **Diffusion nets** and **heat-kernel GNNs**, which use $e^{-t\Delta}$ as a localized, multi-scale filter;
- **Directional CNNs** and **spin-equivariant mesh networks**, which generalize the Dirac operator to gauge-equivariant settings;
- The broader principle that geometric priors should enter network layers through *operators*, not just through input features.

The Laplacian-as-propagation idea also connects surface networks to spectral GCNs (Defferrard et al. 2016) and to the manifold neural ODE framework, where $\Delta$ generates continuous-time diffusion.

## Reading questions / discussion
1. The Laplacian $\Delta$ is intrinsic (isometry-invariant) but the Dirac operator $\mathcal{D}$ encodes extrinsic curvature. For which tasks would you prefer one over the other?
2. The diffusion layer $(I - t\Delta)$ is one explicit Euler step of the heat equation $\partial_t f = -\Delta f$. What happens for large $t$, and how does this relate to the spectral filter $g(\Lambda) = e^{-t\Lambda}$?
3. The cotangent Laplacian requires accurate angles and areas. How robust is the Surface Network to mesh degradation (sliver triangles, non-uniform sampling)?
4. The Dirac operator acts on *spinors*, not scalar functions. How would you adapt the feature representation to handle spinor-valued features, and what does this imply about the network architecture?