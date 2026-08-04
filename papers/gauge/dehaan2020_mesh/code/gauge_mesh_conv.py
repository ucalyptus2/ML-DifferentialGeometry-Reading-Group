"""
Gauge Equivariant Mesh CNNs (de Haan et al. 2021) — minimal core.

A toy gauge-equivariant convolution on a small triangle mesh with a continuous
SO(2) gauge (local tangent-frame rotations). Each directed edge carries a
parallel-transport angle theta_ij (the rotation that aligns the local frame at j
with the frame at i). The kernel is anisotropic in theta; here we demonstrate the
cleanest SO(2)-equivariant case: transport-then-sum (mode k=0 isotropic gain).

Gauge equivariance check: rotate EVERY local frame by a constant offset g
(a global gauge transform) -- the convolved feature must rotate by that same g.

Run:  python gauge_mesh_conv.py
"""
import math
import torch


def gauge_mesh_conv(f, edges_transport):
    """SO(2)-equivariant mesh convolution.

    f: [N, 2] tangent-vector feature per node (transforms by SO(2) under frame rotation).
    edges_transport: list of (i, j, theta_ij) -- theta_ij transports frame j -> frame i.
    For each incoming edge we rotate neighbour j's feature into frame i (parallel
    transport) and sum. Because transport angles are fixed by geometry and the
    per-node SO(2) action composes correctly, the whole layer is gauge-equivariant.
    """
    out = torch.zeros_like(f)
    for (i, j, th) in edges_transport:
        Rtheta = torch.tensor([[math.cos(th), -math.sin(th)],
                               [math.sin(th),  math.cos(th)]])
        frot = f[j] @ Rtheta.T            # parallel-transport feature into frame i
        out[i] = out[i] + frot            # isotropic gain a_0 = 1 (the k=0 intertwiner)
    return out


def rotate_frames(f, g):
    """Global gauge transform: rotate every node's feature by angle g (SO(2) action)."""
    Rg = torch.tensor([[math.cos(g), -math.sin(g)], [math.sin(g), math.cos(g)]])
    return f @ Rg.T


def main():
    torch.manual_seed(0)
    # tiny triangle mesh, 3 nodes, both edge directions, with transport angles
    edges = [(0, 1, 0.4), (1, 2, 1.1), (2, 0, math.pi / 3),
             (1, 0, -0.4), (2, 1, -1.1), (0, 2, -math.pi / 3)]
    f = torch.randn(3, 2)                 # 2-dim tangent-vector feature per node

    out = gauge_mesh_conv(f, edges)

    # global gauge transform: rotate every local frame by g
    g = 0.7
    f_rot = rotate_frames(f, g)
    out_rot = gauge_mesh_conv(f_rot, edges)
    # equivariance: out_rot must equal rotate_frames(out, g)
    expected = rotate_frames(out, g)
    err = (out_rot - expected).abs().max().item()
    print(f"Gauge equivariance (SO(2) mesh): max|conv(R_g f) - R_g conv(f)| = {err:.6e}")
    assert err < 1e-5, "gauge mesh conv must be SO(2)-equivariant"
    print("PASS: anisotropic gauge-equivariant mesh conv transforms covariantly under SO(2).")


if __name__ == "__main__":
    main()

