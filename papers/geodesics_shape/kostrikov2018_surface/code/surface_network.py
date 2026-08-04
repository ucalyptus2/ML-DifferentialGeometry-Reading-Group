"""
Surface Networks (Kostrikov et al., CVPR 2018) -- minimal core.

Implements:
  - Graph Laplacian on a triangle mesh (uniform / combinatorial weights)
  - Laplacian diffusion layer:  H' = (I - t * Delta) H W   (one heat-diffusion step)
  - Spectral layer via eigendecomposition of the Laplacian
  - Demo: heat diffusion on a sphere mesh + trainable layer gradient check.

Run:  python surface_network.py
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import sphere_log  # noqa: E402


def build_sphere(n_lat=12, n_lon=24):
    """UV-sphere mesh: vertices on unit sphere + triangle faces."""
    lats = np.linspace(0, np.pi, n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    V = np.array([[np.sin(la) * np.cos(lo), np.sin(la) * np.sin(lo), np.cos(la)]
                  for la in lats for lo in lons])
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = i * n_lon + j, (i + 1) * n_lon + j
            a2, b2 = i * n_lon + (j + 1) % n_lon, (i + 1) * n_lon + (j + 1) % n_lon
            faces.extend([[a, a2, b], [b, a2, b2]])
    return V, np.array(faces)


def graph_laplacian(V, faces):
    """Combinatorial graph Laplacian L = D - A from mesh edges."""
    n = len(V)
    A = np.zeros((n, n))
    for f in faces:
        for u, v in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            A[u, v] = A[v, u] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


class DiffusionLayer(nn.Module):
    """Heat-diffusion surface-network layer: H' = (I - t * Delta) H W."""

    def __init__(self, L, in_dim, out_dim, t=0.3):
        super().__init__()
        n = L.shape[0]
        self.register_buffer("L", torch.tensor(L, dtype=torch.float32))
        self.register_buffer("I", torch.eye(n))
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        self.t = t

    def forward(self, H):
        return (self.I - self.t * self.L) @ H @ self.W


class SpectralLayer(nn.Module):
    """Spectral surface-network layer: H' = U g(Lambda) U^T H W."""

    def __init__(self, L, in_dim, out_dim, k=16):
        super().__init__()
        eigvals, eigvecs = np.linalg.eigh(L)
        k = min(k, len(eigvals))
        self.register_buffer("U", torch.tensor(eigvecs[:, :k], dtype=torch.float32))
        self.register_buffer("Lam", torch.tensor(eigvals[:k], dtype=torch.float32))
        self.theta = nn.Parameter(torch.ones(k))
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)

    def forward(self, H):
        filt = self.theta * torch.exp(-self.Lam)          # learnable spectral filter
        Hsp = self.U.T @ H                                 # to spectral domain
        Hsp = filt.unsqueeze(1) * Hsp                      # filter
        return (self.U @ Hsp) @ self.W                     # back to spatial + linear


def main():
    torch.manual_seed(0)
    V, faces = build_sphere(12, 24)
    n = len(V)
    print(f"  sphere mesh: {n} vertices, {len(faces)} faces")

    L = graph_laplacian(V, faces)

    # sanity: sphere_log gives tangent direction on unit sphere
    north = int(np.argmax(V[:, 2]))
    v0 = torch.tensor(V[north], dtype=torch.float32)
    v1 = torch.tensor(V[north + 24], dtype=torch.float32)
    tan = sphere_log(v0, v1)
    print(f"  sphere_log(v0, v1) ||tangent||={torch.linalg.norm(tan).item():.3f}")

    # heat diffusion demo (fixed operator, no learning)
    f0 = torch.zeros(n, 1)
    f0[north, 0] = 1.0
    Lt = torch.tensor(L, dtype=torch.float32)
    I = torch.eye(n)
    H = f0.clone()
    for _ in range(5):
        H = (I - 0.3 * Lt) @ H
    top5 = torch.argsort(H.squeeze(), descending=True)[:5]
    print(f"  heat after 5 steps, top-5: {top5.tolist()} (source={north})")

    # trainable diffusion layer
    layer = DiffusionLayer(L, 1, 4, t=0.3)
    out = layer(f0)
    out[north].sum().backward()
    print(f"  DiffusionLayer grad ||dW||={torch.linalg.norm(layer.W.grad).item():.4f}")

    # spectral layer
    slayer = SpectralLayer(L, 1, 4, k=16)
    sout = slayer(f0)
    print(f"  SpectralLayer output shape: {tuple(sout.shape)}")
    print("\n(Surface Networks: Laplacian diffusion + spectral filtering as network layers.)")


if __name__ == "__main__":
    main()
