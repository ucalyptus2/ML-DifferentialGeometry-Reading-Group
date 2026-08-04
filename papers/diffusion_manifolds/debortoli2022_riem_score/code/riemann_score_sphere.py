"""
Riemannian Score-Based Generative Modelling (De Bortoli et al., NeurIPS 2022) -- minimal core.

Implements on the unit sphere S^2:
  1. Riemannian Brownian motion (forward heat) via the Eells-Elworthy-Malliavin
     construction: sample tangent Gaussian, project, retract via exp map.
  2. A tangent-projected score network trained by denoising score matching
     with target  -(1/t) log_{x_t}(x_0).
  3. The reverse-time SDE sampler (Anderson reversal).

Demo: learn a two-mode target on S^2 and verify that reverse-sampled points
concentrate near the modes (mean geodesic distance << pi/2, the uniform baseline).

Run:  python riemann_score_sphere.py
"""
import os, sys, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import sphere_exp, sphere_log  # noqa: E402  (single-vector refs)

D = 3  # S^2


def proj_tan(x, v):
    """Project v onto tangent space at x on the sphere: v - (v.x) x."""
    return v - (x * v).sum(-1, keepdim=True) * x


def batch_exp(x, v):
    """Batch exp on S^{d-1}: x (B,d), v (B,d) tangent -> (B,d)."""
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return torch.cos(n) * x + torch.sin(n) / (n + 1e-12) * v


def batch_log(x, y):
    """Batch log on S^{d-1}: return tangent vector at x pointing to y."""
    xy = (x * y).sum(-1, keepdim=True)
    xy = torch.clamp(xy, -1 + 1e-7, 1 - 1e-7)
    coef = torch.acos(xy) / torch.sqrt(torch.clamp(1 - xy * xy, 1e-7, 1.0))
    return coef * (y - xy * x)


class ScoreNet(nn.Module):
    """MLP (x, t) -> tangent vector at x (output projected to T_x S^2)."""

    def __init__(self, dim=D, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        t = t.reshape(-1, 1) if t.dim() else t.reshape(1, 1).expand(x.shape[0], 1)
        return proj_tan(x, self.net(torch.cat([x, t], -1)))


def forward_heat(x0, n_steps, dt):
    """Riemannian Brownian motion: return list of (B,d) snapshots."""
    traj = [x0]
    x = x0
    for _ in range(n_steps):
        xi = proj_tan(x, torch.randn_like(x))
        x = batch_exp(x, math.sqrt(dt) * xi)
        traj.append(x)
    return traj


def main():
    torch.manual_seed(0)
    # --- target: two small caps on S^2 ---
    c1 = torch.tensor([0.0, 0.0, 1.0])
    c2 = torch.tensor([1.0, 0.0, 0.0])
    N = 256
    half = N // 2
    x0 = torch.cat([c1.expand(half, D), c2.expand(half, D)]) + 0.08 * torch.randn(N, D)
    x0 = x0 / x0.norm(dim=-1, keepdim=True)

    # --- forward heat ---
    T, dt = 0.6, 0.012
    n_fwd = int(T / dt)
    fwd = torch.stack(forward_heat(x0, n_fwd, dt))  # (n+1, N, D)

    # --- train score net by denoising score matching ---
    net = ScoreNet(D)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    idx_all = torch.arange(N)
    for it in range(2500):
        step = torch.randint(1, n_fwd + 1, (N,))
        t = (step.float() * dt).unsqueeze(1)
        x_t = fwd[step, idx_all]
        target = -(1.0 / (t + 1e-3)) * batch_log(x_t, x0)
        pred = net(x_t, t)
        loss = ((pred - target) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            print(f"  iter {it:4d}  dsm loss = {loss.item():.4f}")

    # --- reverse SDE from uniform on S^2 ---
    x = torch.randn(N, D)
    x = x / x.norm(dim=-1, keepdim=True)
    for i in range(n_fwd):
        tk = T - i * dt
        s = net(x, torch.full((N, 1), tk))
        xi = proj_tan(x, torch.randn_like(x))
        x = batch_exp(x, math.sqrt(dt) * xi - 0.5 * dt * s)

    # --- verification: geodesic distance to nearest mode ---
    with torch.no_grad():
        d1 = torch.acos(torch.clamp(x @ c1, -1 + 1e-7, 1 - 1e-7))
        d2 = torch.acos(torch.clamp(x @ c2, -1 + 1e-7, 1 - 1e-7))
        gen_d = torch.minimum(d1, d2).mean().item()
        uni = torch.randn(2000, D); uni = uni / uni.norm(dim=-1, keepdim=True)
        u1 = torch.acos(torch.clamp(uni @ c1, -1 + 1e-7, 1 - 1e-7))
        u2 = torch.acos(torch.clamp(uni @ c2, -1 + 1e-7, 1 - 1e-7))
        uni_d = torch.minimum(u1, u2).mean().item()

    print(f"\nGenerated mean geodesic dist to nearest mode : {gen_d:.3f} rad")
    print(f"Uniform-on-sphere baseline (same metric)      : {uni_d:.3f} rad")
    print(f"Concentration ratio (baseline / generated)    : {uni_d / gen_d:.2f}x")
    assert gen_d < uni_d * 0.6, "samples did not concentrate near target"
    print("\nPASS: reverse-SDE samples concentrate near the two-mode target.")


if __name__ == "__main__":
    main()
