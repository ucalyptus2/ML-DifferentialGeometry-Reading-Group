"""
Gauge Equivariant CNNs (Cohen et al. 2019) — minimal core.

A toy gauge-equivariant convolution with a Z2 gauge group (the local frame can
be flipped by +/-1). Features are 1D "vector" sections; parallel transport along
an edge multiplies by a sign sigma_ij in {+1,-1}; the kernel is the Z2-intertwiner.

Gauge equivariance: flipping the local frame at every node (g_i in {+1,-1}) must
leave the convolved scalar feature invariant (rho trivial output) and transform
the vector feature by the local flip.

Run:  python gauge_conv.py
"""
import os, sys, math
import torch


def gauge_conv(f, edges, sigma):
    """f: [N] gauge-dependent features (rho = sign rep of Z2).
    edges: list of (i,j). sigma: dict edge -> +/-1 (parallel-transport sign).
    Output is a gauge-invariant scalar field (rho_out trivial).
    """
    N = f.shape[0]
    out = torch.zeros(N)
    for (i, j) in edges:
        # transport feature at j to frame of i: f_j * sigma_{i<-j}; kernel is the intertwiner
        # (for Z2, scalar-gauge output, the intertwiner is just summing transported neighbours)
        out[i] = out[i] + sigma[(i, j)] * f[j]
    return out


def flip_frames(f, g):
    """Apply gauge flips g[i] in {+1,-1} to the feature f (rho = sign rep)."""
    return f * torch.tensor(g)


def flip_transport(sigma, g):
    """Under frame flips g, the transport sign becomes g_i * sigma_{ij} * g_j."""
    return {(i, j): g[i] * s * g[j] for (i, j), s in sigma.items()}


def main():
    torch.manual_seed(0)
    N = 6
    edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    # random parallel-transport signs (a Z2 connection)
    sigma = {(i, j): (1 if (i + j) % 2 == 0 else -1) for (i, j) in edges}
    f = torch.randn(N)

    out = gauge_conv(f, edges, sigma)

    # apply a gauge transformation: flip frames at random nodes
    g = [1 if torch.rand(1).item() > 0.5 else -1 for _ in range(N)]
    f_flipped = flip_frames(f, g)
    sigma_flipped = flip_transport(sigma, g)
    out_flipped = gauge_conv(f_flipped, edges, sigma_flipped)

    # The kernel is a Z2-intertwiner (sign rep -> sign rep), so the output transforms
    # by the SAME representation: out_flipped[i] == g[i] * out[i].
    expected = out * torch.tensor(g)
    err = (out_flipped - expected).abs().max().item()
    print(f"Gauge equivariance (Z2): max|conv(flip(f)) - g*conv(f)| = {err:.6e}")
    assert err < 1e-5, "gauge conv must be equivariant under local frame flips (Z2)"
    print("PASS: convolution is gauge-equivariant -- output transforms covariantly with the frame.")


if __name__ == "__main__":
    main()
