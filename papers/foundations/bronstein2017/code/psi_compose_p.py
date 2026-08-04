"""
Bronstein et al. (2017) — "Going beyond Euclidean data".

Implements the paper's central building blocks:
    h(x) = psi( P_f( N(x) ) ),
where
  * N(x)  = the local neighbourhood of node x in a (possibly non-Euclidean) domain;
  * P     = a permutation-invariant local AGGREGATOR (mean / max / sum) that pools
            the features of neighbours -- this encodes symmetry under re-labelling;
  * psi   = a learnable SYMMETRIC function (an MLP) applied to the pooled statistic.

The point: any convolutional architecture on grids, graphs, groups or manifolds is a
special case of this template. This file demonstrates the graph case (a domain where
there is no global coordinate system and neighbour order is arbitrary).

Author: reading-group contributor. Self-contained, no external deps beyond torch.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class Aggregate(nn.Module):
    """Permutation-invariant local aggregator P over each node's neighbourhood.

    Given a graph with adjacency matrix A, for node i we collect features of its
    neighbours j (j != i with A[i,j]=1), and return a symmetric statistic of them:
        'mean' : (1/|N(i)|) * sum_j f_j
        'max'  : elementwise max over neighbours
        'sum'  : sum_j f_j
    Because sum is linear, we implement it with torch.index_add_; mean and max follow
    by reweighting / scatter-reduction. The result is independent of neighbour order.
    """

    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ("mean", "max", "sum"), f"unknown aggregate mode {mode}"
        self.mode = mode

    def _neighbour_edges(self, adj: torch.Tensor) -> tuple:
        """Return source/destination edge indices for incoming neighbours, plus the
        out-degree per node (needed to normalise the mean)."""
        src, dst = torch.nonzero(adj.t(), as_tuple=True)  # src -> dst (incoming)
        deg = adj.sum(dim=1)                              # [n_nodes]
        return src, dst, deg

    def forward(self, f: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """f: [n_nodes, C] node features. adj: [n_nodes, n_nodes] 0/1.
        Returns the pooled statistic per node, shape [n_nodes, C], invariant to
        arbitrary re-labelling of each node's neighbours."""
        n, c = f.shape
        src, dst, deg = self._neighbour_edges(adj)  # edge src -> dst (incoming)
        if self.mode == "sum":
            out = torch.zeros_like(f)
            out.index_add_(0, dst, f[src])
            return out
        if self.mode == "mean":
            out = torch.zeros_like(f)
            out.index_add_(0, dst, f[src])
            return out / deg.clamp(min=1).unsqueeze(1)
        # max: scatter-reduce
        out = torch.full_like(f, float("-inf"))
        out = torch.index_reduce_(out, 0, dst, f[src], reduce="amax",
                                  include_self=True)
        out[deg == 0] = 0.0  # isolated nodes: keep finite
        return out


class SymmetricFn(nn.Module):
    """Learnable symmetric function psi: after pooling, any permutation of the
    neighbour features would have produced the same pooled vector, so an arbitrary
    MLP applied on top is automatically symmetric in the neighbours."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.mlp(pooled)


class PsiComposeP(nn.Module):
    """One geometric-conv layer of the 2017 paper: h = psi(P(N(x)))."""

    def __init__(self, in_dim: int, out_dim: int, mode: str = "mean"):
        super().__init__()
        self.aggregate = Aggregate(mode)
        self.psi = SymmetricFn(in_dim, in_dim * 2, out_dim)

    def forward(self, f: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        pooled = self.aggregate(f, adj)     # P over N(x)
        return self.psi(pooled)             # psi applied symmetrically


# ----------------------------------------------------------------------------
def make_random_graph(n: int = 12, p: float = 0.35, seed: int = 1) -> torch.Tensor:
    """Random undirected 0/1 adjacency matrix (no self-loops)."""
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).float()
    A = torch.maximum(A, A.t())   # symmetrise
    A.fill_diagonal_(0)           # no self-loops
    return A


def check_permutation_invariance():
    """Re-labelling neighbours must not change the pooled statistic."""
    A = make_random_graph(seed=3)
    f = torch.randn(A.shape[0], 4)
    layer = PsiComposeP(4, 6, mode="mean")

    perm = torch.randperm(A.shape[0])          # relabel node indices
    App, fpp = A[perm][:, perm], f[perm]

    with torch.no_grad():
        out1 = layer.aggregate(f, A)
        out2 = layer.aggregate(fpp, App)
    err = (out1[perm] - out2).abs().max().item()
    print(f"  permutation-invariance error of Aggregator P = {err:.2e}")
    assert err < 1e-5, "aggregator is NOT permutation invariant!"

    with torch.no_grad():
        o1, o2 = layer(f, A), layer(fpp, App)
    err2 = (o1[perm] - o2).abs().max().item()
    print(f"  permutation-equivariance error of psi(P(..))  = {err2:.2e}")
    assert err2 < 1e-5


def train_demo():
    """Learn a cheap graph signal f -> neighbour-mean with a psi-comp-P stack."""
    A = make_random_graph(n=16, seed=5)
    x = torch.randn(16, 8)
    deg = A.sum(dim=1).clamp(min=1)
    y_true = (A @ x[:, :1]) / deg.unsqueeze(1)   # [16,1] = mean of neighbour features

    model = PsiComposeP(8, 1, mode="mean")
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    lossfn = nn.MSELoss()

    for step in range(600):
        loss = lossfn(model(x, A), y_true)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 150 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    print(f"  final R2 = {1 - loss.item() / y_true.var().item():.4f}")


if __name__ == "__main__":
    print("== Psi-compose-P (Bronstein et al. 2017) ==")
    print("-- correctness: permutation invariance of the geometric core --")
    check_permutation_invariance()
    print("-- training demo: does psi(P) learn to aggregate? --")
    train_demo()
    print("All checks passed.")

