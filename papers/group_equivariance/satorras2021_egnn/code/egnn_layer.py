"""
E(n) Equivariant Graph Neural Networks (Satorras et al. 2021) — minimal core.

Implements one EGNN layer and checks E(n)-equivariance numerically:
rotate + translate + permute the input and verify outputs transform identically.

Run:  python egnn_layer.py
"""
import os, sys, math
import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    def __init__(self, hid=16, n=2):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hid + 1, hid), nn.SiLU(), nn.Linear(hid, hid), nn.SiLU())
        self.coord_mlp = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1))
        self.node_mlp = nn.Sequential(nn.Linear(2 * hid, hid), nn.SiLU(), nn.Linear(hid, hid))

    def forward(self, h, x, edges):
        # h: [N, hid] invariant node features; x: [N, n] coordinates; edges: list of (i,j)
        N = h.shape[0]
        msgs = torch.zeros(N, h.shape[1])
        coord_update = torch.zeros_like(x)
        for (i, j) in edges:
            d2 = (x[i] - x[j]).pow(2).sum().unsqueeze(0)        # invariant squared distance
            m = self.edge_mlp(torch.cat([h[i], h[j], d2]))      # message (invariant)
            msgs[i] = msgs[i] + m
            coord_update[i] = coord_update[i] + (x[i] - x[j]) * self.coord_mlp(m)
        h_new = h + self.node_mlp(torch.cat([h, msgs], dim=1))
        x_new = x + coord_update
        return h_new, x_new


def main():
    torch.manual_seed(0)
    N, hid, n = 5, 16, 2
    h = torch.randn(N, hid)
    x = torch.randn(N, n)
    edges = [(i, j) for i in range(N) for j in range(N) if i != j]
    layer = EGNNLayer(hid, n)
    h_out, x_out = layer(h, x, edges)

    # E(n) transform: x -> R x + t  (and permute nodes)
    theta = 0.6
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
    t = torch.tensor([2.0, -1.0])
    perm = torch.randperm(N)
    h_p = h[perm]
    x_p = x[perm] @ R.T + t
    edges_p = [(int(torch.where(perm == i)[0]), int(torch.where(perm == j)[0]))
               for (i, j) in edges]
    h_out_p, x_out_p = layer(h_p, x_p, edges_p)

    # invert the permutation to compare
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N)
    h_match = h_out[inv]
    x_match = x_out[inv]
    h_err = (h_out_p - h_match).abs().max().item()
    x_err = (x_out_p - (x_match @ R.T + t)).abs().max().item()
    print(f"EGNN E(n)-equivariance:  max|h mismatch| = {h_err:.6e},  max|x mismatch| = {x_err:.6e}")
    assert h_err < 1e-4 and x_err < 1e-4, "EGNN layer must be E(n)-equivariant"
    print("PASS: h is invariant and x transforms equivariantly under E(n) + permutation.")


if __name__ == "__main__":
    main()
