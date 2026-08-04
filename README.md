# ML × Differential Geometry Reading Group

> **A place to read papers and discuss Machine Learning *on* and *with* Differential Geometry & Shape Analysis —**
> with running code and a full tutorial, not just a reading list.

Machine learning increasingly lives on curved spaces: data often lies on manifolds (poses, SPD
matrices, shapes, hierarchical structures), and modern architectures are built to be **equivariant**
under the symmetries encoded by those geometries. This repository is a living reading group that
covers **the intersection of ML and differential geometry** — from the mathematical foundations
(tangent spaces, Riemannian metrics, geodesics, Lie groups, gauge structures) to state-of-the-art
methods (neural ODEs, hyperbolic embeddings, E(n)-equivariant and gauge-equivariant networks,
Riemannian flows and diffusion models, optimal transport on matrices).

Every paper has:
- a **`paper.md`** — a structured, math-precise summary plus reading-group discussion questions;
- **`code/`** — a runnable PyTorch and/or JAX implementation of the *geometric core* of the paper.

There is also a full **`tutorial/`** that teaches the underlying differential geometry with
executable code, and a hosted site: **<https://ucalyptus2.github.io/ML-DifferentialGeometry-Reading-Group/>**.

---

## 🌍 What is "ML × Differential Geometry"?

The math that powers "AI on curved data" rests on a few ideas that recur across almost every paper
here. If you are new, **start with `tutorial/`**:

| Concept | One-liner | Where it appears |
|---|---|---|
| **Manifold** | A space that locally looks like ℝⁿ | pose/rotation spaces, SPD matrices, shapes, data clouds |
| **Tangent space** `T_x M` | The best linear approximation to `M` at `x` | optimizers, flows, logits on manifolds |
| **Riemannian metric** | A smooth inner product on each tangent space; turns `M` into a metric space | geodesics, "Riemannian" learning |
| **Geodesic / exp & log maps** | Shortest paths and their local coordinates | hyperbolic nets, neural ODEs, diffusion on spheres |
| **Lie group / group action** | A group of symmetries acting on a space | G-CNNs, spherical CNNs, E(n)-equivariant nets |
| **Gauge structure / G-structure** | A consistent choice of local frames; parallel transport | gauge-equivariant mesh CNNs |
| **Equivariance** | `f(g·x) = g·f(x)` — the model respects symmetry | geometric deep learning blueprint |
| **Curvature** | How much the space deviates from flat | hyperbolic (negative κ), sphere (positive κ) |

---

## 📚 The reading list (22 papers, verified references)

### 0. Foundations & the geometric deep learning blueprint
- **Geometric deep learning: going beyond Euclidean data** — Bronstein, Bruna, LeCun, Szlam, Vandergheynst (2017) · *arXiv:1611.08097* → [`papers/foundations/bronstein2017`](papers/foundations/bronstein2017)
- **Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges** — Bronstein, Bruna, Cohen, Veličković (2021) · *arXiv:2104.13478* → [`papers/foundations/bronstein2021`](papers/foundations/bronstein2021)

### 1. Group- & gauge-equivariant networks
- **Group Equivariant Convolutional Networks** — Cohen & Welling (2016) · *arXiv:1602.07576* → [`papers/group_equivariance/cohen2016_group`](papers/group_equivariance/cohen2016_group)
- **Spherical CNNs** — Cohen, Geiger, Koehler, Welling (2018) · *arXiv:1801.10130* → [`papers/group_equivariance/cohen2018_spherical`](papers/group_equivariance/cohen2018_spherical)
- **E(n) Equivariant Graph Neural Networks** — Satorras, Hoogeboom, Welling (2021) · *arXiv:2102.09844* → [`papers/group_equivariance/satorras2021_egnn`](papers/group_equivariance/satorras2021_egnn)
- **Gauge Equivariant Convolutional Networks and the Icosahedral CNN** — Cohen, Weiler, Kicanaoglu, Welling (2019) · *arXiv:1902.04615* → [`papers/gauge/cohen2019_icosahedral`](papers/gauge/cohen2019_icosahedral)
- **Gauge Equivariant Mesh CNNs** — de Haan, Weiler, Cohen, Welling (2021) · *arXiv:2003.05425* → [`papers/gauge/dehaan2020_mesh`](papers/gauge/dehaan2020_mesh)

### 2. Geodesics & shape analysis
- **Geodesic convolutional neural networks on Riemannian manifolds** — Masci, Boscaini, Bronstein, Vandergheynst (2015) · *arXiv:1501.06297* → [`papers/geodesics_shape/masci2015_gcnn`](papers/geodesics_shape/masci2015_gcnn)
- **Surface Networks** — Kostrikov, Jiang, Panozzo, Zorin, Bruna (2018) · *arXiv:1705.10819* → [`papers/geodesics_shape/kostrikov2018_surface`](papers/geodesics_shape/kostrikov2018_surface)

### 3. Hyperbolic geometry & non-Euclidean embeddings
- **Poincaré Embeddings for Learning Hierarchical Representations** — Nickel & Kiela (2017) · *arXiv:1705.08039* → [`papers/hyperbolic/nickel2017_poincare`](papers/hyperbolic/nickel2017_poincare)
- **Hyperbolic Neural Networks** — Ganea, Bécigneul, Hofmann (2018) · *arXiv:1805.09112* → [`papers/hyperbolic/ganea2018_hyperbolic_nn`](papers/hyperbolic/ganea2018_hyperbolic_nn)
- **Hyperbolic Graph Convolutional Neural Networks** — Chami, Ying, Ré, Leskovec (2019) · *arXiv:1910.12933* → [`papers/hyperbolic/chami2019_hgcn`](papers/hyperbolic/chami2019_hgcn)
- **Hyperbolic Entailment Cones for Learning Hierarchical Embeddings** — Ganea, Bécigneul, Hofmann (2018) · *arXiv:1804.01882* → [`papers/hyperbolic/ganea2018_entailment`](papers/hyperbolic/ganea2018_entailment)

### 4. Riemannian optimization & SPD manifolds
- **A Riemannian Network for SPD Matrix Learning** — Huang & Van Gool (2017) · *arXiv:1608.04233* → [`papers/spd_riemannian/huang2017_spdnet`](papers/spd_riemannian/huang2017_spdnet)
- **Riemannian Adaptive Optimization Methods** — Bécigneul & Ganea (2019) · *arXiv:1810.00760* → [`papers/spd_riemannian/becigneul2019_radam`](papers/spd_riemannian/becigneul2019_radam)

### 5. Continuous flows, ODEs & diffusion on manifolds
- **Neural Ordinary Differential Equations** — Chen, Rubanova, Bettencourt, Duvenaud (2018) · *arXiv:1806.07366* → [`papers/flows_odes/chen2018_node`](papers/flows_odes/chen2018_node)
- **FFJORD** — Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud (2019) · *arXiv:1810.01367* → [`papers/flows_odes/grathwohl2018_ffjord`](papers/flows_odes/grathwohl2018_ffjord)
- **Continuous Normalizing Flows on Manifolds** — Falorsi (2021) · *arXiv:2104.14959* → [`papers/flows_odes/falorsi2021_manifold_nf`](papers/flows_odes/falorsi2021_manifold_nf)
- **E(n) Equivariant Normalizing Flows** — Satorras, Hoogeboom, Fuchs, Posner, Welling (2021) · *arXiv:2105.09016* → [`papers/flows_odes/satorras2021_enf`](papers/flows_odes/satorras2021_enf)
- **Riemannian Score-Based Generative Modelling** — De Bortoli, Mathieu, Hutchinson, Thornton, Teh, Doucet (2022) · *arXiv:2202.02763* → [`papers/diffusion_manifolds/debortoli2022_riem_score`](papers/diffusion_manifolds/debortoli2022_riem_score)

### 6. Optimal transport & information geometry
- **Sinkhorn Distances: Lightspeed Computation of Optimal Transport** — Cuturi (2013) · *arXiv:1306.0895* → [`papers/optimal_transport/cuturi2013_sinkhorn`](papers/optimal_transport/cuturi2013_sinkhorn)
- **Geometric GAN** — Lim, Park, Kang, Ye (2017) · *arXiv:1705.02894* → [`papers/optimal_transport/lim2017_geometric_gan`](papers/optimal_transport/lim2017_geometric_gan)

---

## 🧭 The tutorial (start here if you're new)

The [`tutorial/`](tutorial) folder teaches the geometry behind all of these papers, with
executable PyTorch and JAX code:

1. **Why geometry matters for ML** — `00_why_geometry.md`
2. **Manifolds & tangent spaces** — `01_manifolds_tangent_spaces.md`
3. **Riemannian metrics & geodesics (exp/log maps)** — `02_riemannian_metric_geodesics.md`
4. **Groups & equivariance** — `03_groups_equivariance.md`
5. **Hyperbolic geometry (the Poincaré ball)** — `04_hyperbolic_geometry.md`
6. **Matrix manifolds: SPD and retractions** — `05_matrix_manifolds_spd.md`
7. **The geometric deep learning blueprint** — `06_geometric_deep_learning_blueprint.md`

Runnable code lives in [`tutorial/code/`](tutorial/code).

---

## 🌐 The site

A hosted HTML site for this reading group (papers, concepts, and code pointers) lives at
**<https://ucalyptus2.github.io/ML-DifferentialGeometry-Reading-Group/>** (source in `docs/`).

---

## 📥 Contributing

Open to everyone. Read [`CONTRIBUTING_TEMPLATE.md`](CONTRIBUTING_TEMPLATE.md) — the exact template
every `paper.md` and `code/` follows — then open a PR adding a paper, fixing a proof, or improving
a tutorial lesson.

## 📄 License

MIT. See [`LICENSE`](LICENSE).



