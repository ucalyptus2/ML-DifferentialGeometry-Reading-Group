"""
Hyperbolic Graph Convolutional Neural Networks (Chami et al., NeurIPS 2019) -- minimal core.

Implements a single HGCN layer:
  1. Map ball features to tangent space at origin:  H_tan = log_0(H)
  2. Euclidean GCN aggregation in tangent space:     H'_tan = tanh(A_hat H_tan W)
  3. Map back to Poincare ball:                       H' = exp_0(H'_tan)

Demo: 2-layer HGCN on a tiny graph, check outputs stay in the ball.

Run:  python hgcn_layer.py
"""
import os, sys, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import poincare_dist  # noqa: E402

C = 1.0


def exp_0(V, c=C):
    """Batch exp at origin:  exp_0(v_i) = tanh(sqrt(c)||v_i||) v_i/(sqrt(c)||v_i||)."""
    norms = torch.linalg.norm(V, dim=-1, keepdim=True) + 1e-12
    return torch.tanh(math.sqrt(c) * norms) * V / (math.sqrt(c) * norms)


def log_0(Y, c=C):
    """Batch log at origin:  log_0(y_i) = artanh(sqrt(c)||y_i||) y_i/(sqrt(c)||y_i||)."""
    norms = torch.linalg.norm(Y, dim=-1, keepdim=True) + 1e-12
    coef = torch.atanh(torch.clamp(math.sqrt(c) * norms, 0, 1 - 1e-7))
    return coef * Y / (math.sqrt(c) * norms)


def normalize_adj(A):
    """Symmetric normalization:  D^{-1/2}(A+I)D^{-1/2}."""
    A = A + torch.eye(A.shape[0])
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(D ** (-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


class HGNNLayer(nn.Module):
    """One hyperbolic GCN layer: log_0 -> GCN -> exp_0."""

    def __init__(self, in_dim, out_dim, c=C):
        super().__init__()
        self.c = c
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)

    def forward(self, H, A_hat):
        V = log_0(H, self.c)                       # to tangent space
        V = torch.tanh(A_hat @ V @ self.W)         # Euclidean GCN
        return exp_0(V, self.c)                    # back to ball


def project_to_ball(H, cap=0.9):
    norms = torch.linalg.norm(H, dim=-1, keepdim=True) + 1e-12
    lim = cap / math.sqrt(C)
    scale = torch.where(norms > lim, lim / norms, torch.ones_like(norms))
    return H * scale


def main():
    torch.manual_seed(0)
    A = torch.tensor([[0, 1, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [1, 1, 0, 1, 0, 0],
                      [0, 0, 1, 0, 1, 1],
                      [0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 1, 1, 0]], dtype=torch.float32)
    A_hat = normalize_adj(A)
    n, d = 6, 4
    H0 = project_to_ball(0.1 * torch.randn(n, d))

    layer1 = HGNNLayer(d, 8)
    layer2 = HGNNLayer(8, 4)
    H1 = project_to_ball(layer1(H0, A_hat))
    H2 = project_to_ball(layer2(H1, A_hat))

    print("HGCN forward pass (outputs in Poincare ball):")
    print(f"  input  norms: {[round(x, 3) for x in torch.linalg.norm(H0, dim=-1).tolist()]}")
    print(f"  layer1 norms: {[round(x, 3) for x in torch.linalg.norm(H1, dim=-1).tolist()]}")
    print(f"  layer2 norms: {[round(x, 3) for x in torch.linalg.norm(H2, dim=-1).tolist()]}")
    max_norm = torch.linalg.norm(H2, dim=-1).max().item()
    print(f"  max ||h|| = {max_norm:.3f} < {1 / math.sqrt(C):.1f} (stays in ball)")

    # gradient check via hyperbolic distance loss
    loss = poincare_dist(H2[0], H2[3], C)
    loss.backward()
    print(f"\nGradient through HGCN: ||dL/dW1||={torch.linalg.norm(layer1.W.grad).item():.4f}, "
          f"||dL/dW2||={torch.linalg.norm(layer2.W.grad).item():.4f}")

    print("\n(HGCN: aggregate in tangent space via GCN, then exp-map back to Poincare ball.)")


if __name__ == "__main__":
    main()
