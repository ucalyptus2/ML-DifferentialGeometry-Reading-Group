"""
Continuous Normalizing Flows on Manifolds (Falorsi 2021) — minimal core.

Integrates a tangent vector field on S^2 two ways:
  (a) naive ambient Euler: x <- x + dt*f(x)               (leaves the sphere every step)
  (b) retraction-based Euler: x <- sphere_exp(x, dt*f(x))  (stays exactly on the sphere)
and measures ||x|| - 1 (distance off the manifold) along both trajectories.

Run:  python manifold_cnf_drift.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import sphere_exp   # noqa: E402


def tangent_field(x):
    """A smooth tangent vector field on S^2: project a fixed rotation-like ambient
    field onto T_x S^2 so it is a genuine tangent vector everywhere."""
    w = torch.tensor([0.3, -0.5, 0.8])          # arbitrary "axis" direction
    raw = torch.linalg.cross(w, x)              # ambient vector, already tangent (w x x . x = 0)
    return raw


def integrate_naive(x0, steps, dt):
    x = x0.clone()
    offs = [abs(torch.linalg.norm(x).item() - 1.0)]
    for _ in range(steps):
        x = x + dt * tangent_field(x)
        offs.append(abs(torch.linalg.norm(x).item() - 1.0))
    return x, offs


def integrate_retraction(x0, steps, dt):
    x = x0.clone()
    offs = [abs(torch.linalg.norm(x).item() - 1.0)]
    for _ in range(steps):
        x = sphere_exp(x, dt * tangent_field(x))
        offs.append(abs(torch.linalg.norm(x).item() - 1.0))
    return x, offs


def main():
    torch.manual_seed(0)
    x0 = torch.tensor([1.0, 0.0, 0.0])
    steps, dt = 40, 0.05

    x_naive, offs_naive = integrate_naive(x0, steps, dt)
    x_retr, offs_retr = integrate_retraction(x0, steps, dt)

    print("naive ambient Euler   ||x||-1 over trajectory:",
          [round(o, 5) for o in offs_naive[::8]])
    print("retraction (sphere_exp) ||x||-1 over trajectory:",
          [round(o, 5) for o in offs_retr[::8]])

    print(f"\nfinal drift off sphere:  naive = {offs_naive[-1]:.5f},  "
          f"retraction = {offs_retr[-1]:.2e}")

    assert offs_naive[-1] > 0.03, "naive ambient Euler must drift measurably off the sphere"
    assert offs_retr[-1] < 1e-5, "retraction-based Euler must stay exactly on the sphere"
    print("\nPASS: naive Euler drifts off the manifold; retraction-based integration "
          "stays on it to numerical precision.")


if __name__ == "__main__":
    main()
