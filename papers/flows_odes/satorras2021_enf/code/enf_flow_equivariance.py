"""
E(n) Equivariant Normalizing Flows (Satorras et al. 2021) — minimal core.

Integrates a multi-step Euler flow dx/dt = EGNN(h,x) (reusing the single EGNN layer
from papers/group_equivariance/satorras2021_egnn/code/egnn_layer.py as the vector
field) and checks the WHOLE trajectory -- not just one layer -- commutes with
rotation + translation + permutation, i.e. the entire flow is E(n)- and
permutation-equivariant.

Run:  python enf_flow_equivariance.py
"""
import os, sys, math
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "group_equivariance", "satorras2021_egnn", "code"))
from egnn_layer import EGNNLayer   # noqa: E402


def integrate_flow(layer, h0, x0, edges, steps=6, dt=0.15):
    """Euler-integrate dx/dt = EGNN(h,x); h is carried along unchanged in shape."""
    h, x = h0, x0
    for _ in range(steps):
        h_delta, x_new = layer(h, x, edges)
        h = h + dt * (h_delta - h)     # Euler step toward the layer's proposed h
        x = x + dt * (x_new - x)       # Euler step toward the layer's proposed x
    return h, x


def main():
    torch.manual_seed(0)
    N, hid, n = 5, 16, 2
    h0 = torch.randn(N, hid)
    x0 = torch.randn(N, n)
    edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    layer = EGNNLayer(hid, n)

    h_out, x_out = integrate_flow(layer, h0, x0, edges, steps=6, dt=0.15)

    # E(n) transform: x -> R x + t, plus a node permutation
    theta = 0.6
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
    t = torch.tensor([2.0, -1.0])
    perm = torch.randperm(N)
    h0_p = h0[perm]
    x0_p = x0[perm] @ R.T + t
    edges_p = [(int(torch.where(perm == i)[0]), int(torch.where(perm == j)[0]))
               for (i, j) in edges]

    h_out_p, x_out_p = integrate_flow(layer, h0_p, x0_p, edges_p, steps=6, dt=0.15)

    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N)
    h_match = h_out[inv]
    x_match = x_out[inv]

    h_err = (h_out_p - h_match).abs().max().item()
    x_err = (x_out_p - (x_match @ R.T + t)).abs().max().item()

    print(f"6-step EGNN flow E(n)-equivariance:  "
          f"max|h mismatch| = {h_err:.6e},  max|x mismatch| = {x_err:.6e}")
    assert h_err < 1e-4 and x_err < 1e-4, \
        "the whole multi-step flow must be E(n)- and permutation-equivariant, not just one layer"
    print("PASS: the full multi-step flow -- not just a single EGNN layer -- is "
          "E(n)-equivariant and permutation-equivariant end to end.")


if __name__ == "__main__":
    main()
