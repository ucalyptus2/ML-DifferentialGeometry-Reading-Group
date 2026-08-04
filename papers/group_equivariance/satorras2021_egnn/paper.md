# E(n) Equivariant Graph Neural Networks

- **Authors:** Victor Garcia Satorras, Emiel Hoogeboom, Max Welling
- **Venue/Year:** ICML 2021
- **arXiv:** https://arxiv.org/abs/2102.09844
- **Category:** group_equivariance

## One-paragraph TL;DR
A graph neural network whose features are **coordinates** $x_i \in \mathbb{R}^n$ plus invariant
node states $h_i$, built so that the whole update is **E(n)-equivariant** (rotations, reflections,
translations, and node permutations) *without* any expensive spherical-harmonic or
group-representation machinery. The trick: messages depend only on **invariant distances**
$\|x_i - x_j\|^2$, and coordinate updates are relative vectors rescaled by learned scalars.

## The problem
Many tasks (molecular dynamics, point clouds, N-body simulation) live on unordered sets of points
in $\mathbb{R}^n$ and are physically symmetric under the Euclidean group $E(n) = \mathbb{R}^n
\rtimes O(n)$. Most GNNs ignore this symmetry and must *learn* it from data.

## Key idea(s)
- Node feature $h_i$ is **invariant** (a scalar vector, transforms trivially).
- Coordinate $x_i$ is **equivariant**: under $g \in E(n)$, $x_i \mapsto R x_i + t$.
- Message $m_{ij} = \phi_e\!\big(h_i, h_j, \|x_i - x_j\|^2, a_{ij}\big)$ uses only the **squared
  distance**, which is $O(n)$-invariant.
- Coordinate update: $x_i \leftarrow x_i + \sum_{j \ne i} (x_i - x_j)\,\phi_x(m_{ij})$ — a sum of
  **relative vectors** scaled by scalars, which is automatically translation- and rotation-equivariant.

## The mathematics
Let $g = (R, t) \in E(n)$ act by $x_i \mapsto R x_i + t$. Then:
- $\|x_i - x_j\|^2 \mapsto \|R(x_i - x_j)\|^2 = \|x_i - x_j\|^2$ (invariant — used in messages);
- $(x_i - x_j) \mapsto R(x_i - x_j)$, so $\sum_j (x_i - x_j)\phi_x(m_{ij}) \mapsto
  R \sum_j (x_i - x_j)\phi_x(m_{ij})$ because $\phi_x(m_{ij})$ is invariant — hence **equivariant**;
- translations cancel because only *differences* $x_i - x_j$ appear.

## Method / architecture
EGNN layer: aggregate messages $\to$ update $h_i$ (any GNN update on invariant features) $\to$
update $x_i$ via the relative-vector sum above. Stack layers; final readout can be invariant
(e.g. sum of $h_i$) or equivariant (return final coordinates).

## Code
`code/egnn_layer.py` — the EGNN layer in PyTorch plus an **E(n)-equivariance check**: rotate +
translate + permute the input and verify outputs transform identically.

## Why it matters
EGNN gives strong equivariance "for free" with the cost of an ordinary GNN, beating heavier
spherical-harmonic equivariant models on N-body prediction and molecular property tasks. It's now
a default backbone for equivariant graph learning.

## Reading questions / discussion
1. Why does using $\|x_i - x_j\|^2$ (not the vector $x_i - x_j$) inside the message *guarantee*
   invariance, and why is the vector only used *outside* in the coordinate update?
2. Where would EGNN break equivariance if you naively added a bias term to the coordinate update?
3. Compare the expressivity of EGNN to a full $E(n)$-steerable net: what geometric quantities can
   the latter represent that EGNN cannot?
