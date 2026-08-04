"""
Bronstein, Bruna, Cohen, Velickovic (2021) - "Geometric Deep Learning: Grids,
Groups, Graphs, Geodesics, and Gauges" (the Blueprint paper).

Implements the two categorical blueprints:

  1-CATEGORICAL (kinematic):
      (Phi f)(i) = phi( f(i),  (+)_{j in N(i)}  psi(f(i), f(j)) )
     -- POSITIVE scalar fields + a permutation-invariant NEIGHBOUR AGGREGATE.
        Works when data only needs to be *carried* along the domain (directions
        between neighbours are irrelevant).

  2-CATEGORICAL (dynamic / gauge):
      (Phi f)(i) = phi( f(i),  (+)_{j} K g_{ji}  transp_{j->i}( ... ) )
     -- EQUIVARIANT (vector) fields + PARALLEL TRANSPORT g_{ji} between local
        frames + an SO(2)-EQUIVARIANT kernel K.  Needed when the relative
        DIRECTION between points matters (non-kinematic tasks).

The mathematical point demonstrated here: K must COMMUTE with every rotation of
the structure group for the layer to be gauge-equivariant. We verify this with a
global SO(2) equivariance test and show that a generic (non-equivariant) linear
kernel VIOLATES it.

Author: reading-group contributor. Deps: torch, numpy.
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)


def rot2(alpha):
    """2D rotation matrix by angle alpha."""
    c, s = np.cos(alpha), np.sin(alpha)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1-CATEGORICAL BLUEPRINT : positive scalar fields + neighbour aggregate
# ---------------------------------------------------------------------------
class ScalarMessagePassing(nn.Module):
    """(Phi f)(i) = phi( f(i), (+)_{j in N(i)} psi(f(i), f(j)) ).

    f is a positive scalar field (here C channels). The aggregator (+) is a SUM
    over neighbours, which is permutation-invariant, so the layer is equivariant
    to re-labelling nodes. psi is a shared message MLP, phi a shared update MLP.
    """

    def __init__(self, in_c: int, hid: int = 24, out_c: int = 16):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * in_c, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.upd = nn.Sequential(
            nn.Linear(in_c + hid, hid), nn.ReLU(), nn.Linear(hid, out_c))

    def forward(self, f: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """f: [n, C]; edges: [E, 2] of (src,dst). Returns [n, out_c]."""
        if edges.numel() == 0:
            return self.upd(torch.cat([f, torch.zeros_like(f)], dim=-1))
        src, dst = edges[:, 0], edges[:, 1]
        m = self.msg(torch.cat([f[src], f[dst]], dim=-1))   # [E, hid]
        agg = torch.zeros(f.shape[0], m.shape[1], device=f.device)
        agg.index_add_(0, dst, m)                           # (+) sum over N(i)
        return self.upd(torch.cat([f, agg], dim=-1))


# ---------------------------------------------------------------------------
# 2-CATEGORICAL BLUEPRINT : equivariant fields + transport + gauge kernel
# ---------------------------------------------------------------------------
class GaugeEquivariantLayer(nn.Module):
    """Gauge (dynamic) layer on a 2D tangent patch.

    Each node i carries an SO(2)-equivariant VECTOR feature v_i in R^2 plus a
    local FRAME described by a rotation angle theta_i. A message from j to i is
    first parallel-transported into i's frame by g_{ji} = R(theta_j - theta_i),
    then read by a kernel K.  The layer is gauge-equivariant iff K is
    SO(2)-equivariant, i.e. K commutes with every R(alpha). On the irrep R^2
    the only such maps are K = lambda * I (one scalar weight per irrep block).
    """

    def __init__(self, weight: float = 1.0, equivariant: bool = True):
        super().__init__()
        self.equivariant = equivariant
        self.lam = nn.Parameter(torch.tensor(weight))      # equivariant scalar
        self.full = nn.Parameter(torch.eye(2) + 0.5 * torch.randn(2, 2))  # control

    def kernel(self):
        if self.equivariant:
            return self.lam * torch.eye(2)     # commutes with all rotations
        return self.full                       # generic 2x2 (not equivariant)

    def forward(self, v: torch.Tensor, theta: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        """v: [n,2] vector features; theta: [n] frame angles.
        Returns transported + aggregated + filtered output [n,2]."""
        K = self.kernel()
        out = torch.zeros_like(v)
        for (i, j) in edges.tolist():
            g = rot2((theta[j] - theta[i]).item())  # transport j's frame -> i's
            out[i] = out[i] + K @ (g @ v[j])        # aggregate over neighbours
        return out


def check_scalar_invariance():
    torch.manual_seed(7)
    n = 10
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 5],
                          [5, 2], [0, 6], [6, 7], [7, 1]], dtype=torch.long)
    f = torch.randn(n, 3)
    layer = ScalarMessagePassing(3)
    perm = torch.randperm(n)
    pmap = {int(old): int(new) for new, old in enumerate(perm.tolist())}
    e2 = torch.tensor([[pmap[int(a)], pmap[int(b)]] for a, b in edges.tolist()])
    with torch.no_grad():
        out1 = layer(f, edges)
        out2 = layer(f[perm], e2)
    err = (out1[perm] - out2).abs().max().item()
    print(f"  1-cat | relabel-equivariance error of psi/aggregate = {err:.2e}")
    assert err < 1e-4, "1-categorical layer is NOT relabel-equivariant!"


def check_gauge_equivariance():
    torch.manual_seed(11)
    # a tiny triangular manifold patch (4 nodes, 5 directed edges)
    edges = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]], dtype=torch.long)
    n = 4
    theta = torch.tensor([0.2, -0.5, 1.0, 2.0])   # arbitrary local frames
    v = torch.randn(n, 2)

    for eq in (True, False):
        layer = GaugeEquivariantLayer(weight=0.8, equivariant=eq)
        with torch.no_grad():
            out = layer(v, theta, edges)
            alpha = 0.7
            Ra = rot2(alpha)
            v_rot = (Ra.unsqueeze(0) @ v.unsqueeze(-1)).squeeze(-1)  # rotate fields
            theta_rot = theta + alpha                                # rotate frames
            out_rot = layer(v_rot, theta_rot, edges)
            expected = (Ra.unsqueeze(0) @ out.unsqueeze(-1)).squeeze(-1)
            err = (out_rot - expected).abs().max().item()

        label = "equivariant K (lambda*I)" if eq else "generic 2x2 K (NOT equiv.)"
        print(f"  2-cat | {label:28s}  SO(2) equivariance error = {err:.2e}")
        if eq:
            assert err < 1e-4, "equivariant kernel broke SO(2) equivariance!"
        else:
            print("    -> generic kernel violates the gauge constraint (expected)")


if __name__ == "__main__":
    print("== Blueprint layers (Bronstein et al. 2021) ==")
    print("-- 1-categorical: positive scalar fields + neighbour aggregate --")
    check_scalar_invariance()
    print("-- 2-categorical: equivariant fields + parallel transport + gauge K --")
    check_gauge_equivariance()
    print("All checks passed.")

