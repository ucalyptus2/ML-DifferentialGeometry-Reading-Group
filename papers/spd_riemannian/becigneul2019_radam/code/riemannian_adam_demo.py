"""
Riemannian Adam (Becigneul & Ganea 2019) — minimal core.

Compares naive Euclidean Adam (ambient-coordinate update, no retraction) against
Riemannian Adam (same moment estimates, but the step retracts via poincare_exp) on
a toy loss that pulls a point on the Poincare ball toward the boundary. The naive
version eventually produces an iterate outside the ball; the Riemannian version
never does, by construction of the exponential-map retraction.

Run:  python riemannian_adam_demo.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import poincare_exp   # noqa: E402

C = 1.0
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8


def loss_grad(x):
    """Toy loss f(x) = -||x||^2 (pulls x toward the boundary); Euclidean gradient."""
    x = x.clone().requires_grad_()
    f = -torch.dot(x, x)
    return torch.autograd.grad(f, x)[0].detach()


def euclidean_adam_step(x, m, v, t, lr=0.1):
    """Naive: treat x as flat R^n, no retraction — plain Adam update."""
    g = loss_grad(x)
    m = BETA1 * m + (1 - BETA1) * g
    v = BETA2 * v + (1 - BETA2) * g * g
    mhat = m / (1 - BETA1 ** t)
    vhat = v / (1 - BETA2 ** t)
    x_new = x - lr * mhat / (torch.sqrt(vhat) + EPS)
    return x_new, m, v


def riemannian_adam_step(x, m, v, t, lr=0.1):
    """Riemannian: Euclidean gradient rescaled by inverse metric, step via poincare_exp."""
    g = loss_grad(x)
    lam = 2.0 / (1 - C * torch.dot(x, x))
    rgrad = g / (lam ** 2)                      # Riemannian gradient = g_x^{-1} . Euclidean grad
    m = BETA1 * m + (1 - BETA1) * rgrad
    v = BETA2 * v + (1 - BETA2) * rgrad * rgrad
    mhat = m / (1 - BETA1 ** t)
    vhat = v / (1 - BETA2 ** t)
    step = -lr * mhat / (torch.sqrt(vhat) + EPS)
    x_new = poincare_exp(x, step, C)            # retraction: always lands inside the ball
    return x_new, m, v


def run(step_fn, steps=60, lr=0.1, dtype=torch.float32):
    x = torch.tensor([0.3, 0.0], dtype=dtype)
    m, v = torch.zeros_like(x), torch.zeros_like(x)
    norms = [torch.linalg.norm(x).item()]
    for t in range(1, steps + 1):
        x, m, v = step_fn(x, m, v, t, lr)
        norms.append(torch.linalg.norm(x).item())
    return norms


def main():
    torch.manual_seed(0)
    eu_norms = run(euclidean_adam_step, steps=60, lr=0.1)
    ri_norms = run(riemannian_adam_step, steps=60, lr=0.02, dtype=torch.float64)

    print("Euclidean Adam:   ||x|| over training ->", [round(n, 3) for n in eu_norms[::10]])
    print("Riemannian Adam:  ||x|| over training ->", [round(n, 3) for n in ri_norms[::10]])

    eu_escaped = max(eu_norms) >= 1.0 / (C ** 0.5)
    ri_inside = max(ri_norms) < 1.0 / (C ** 0.5)

    print(f"\nEuclidean Adam max ||x|| = {max(eu_norms):.4f}  (boundary at "
          f"{1.0 / (C ** 0.5):.4f}) -> escaped ball: {eu_escaped}")
    print(f"Riemannian Adam max ||x|| = {max(ri_norms):.4f} -> stayed strictly inside: "
          f"{ri_inside}")

    assert eu_escaped, "naive Euclidean Adam should overshoot the Poincare ball boundary"
    assert ri_inside, "Riemannian Adam must never leave the manifold (Exp always lands inside)"
    print("\nPASS: naive Adam escapes the ball; Riemannian Adam (Exp-retraction) never does.")


if __name__ == "__main__":
    main()
