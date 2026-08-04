"""
Hyperbolic Neural Networks (Ganea, Becigneul, Hofmann, NeurIPS 2018) -- minimal core.

Implements the Mobius linear layer, the fundamental building block of
hyperbolic NNs:
  - Mobius matrix-vector multiplication  M_W (x)_c = exp_0(W log_0(x))
  - Mobius linear layer  H(x) = (M_W (x)_c) (+)_c b
  - Demo: forward + backward pass on points in the Poincare ball.

Run:  python mobius_linear.py
"""
import os, sys, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import mobius_add  # noqa: E402

C = 1.0


def exp_0(v, c=C):
    """Exponential map at origin: exp_0(v) = tanh(sqrt(c)||v||) v/(sqrt(c)||v||)."""
    vn = torch.linalg.norm(v) + 1e-12
    return torch.tanh(math.sqrt(c) * vn) * v / (math.sqrt(c) * vn)


def log_0(y, c=C):
    """Logarithmic map at origin: log_0(y) = artanh(sqrt(c)||y||) y/(sqrt(c)||y||)."""
    yn = torch.linalg.norm(y) + 1e-12
    return torch.atanh(torch.clamp(math.sqrt(c) * yn, 0, 1 - 1e-7)) * y / (math.sqrt(c) * yn)


def mobius_matvec(W, x, c=C):
    """Mobius matrix-vector product: exp_0(W log_0(x))."""
    return exp_0(W @ log_0(x, c), c)


class MobiusLinear(nn.Module):
    """Mobius linear layer: H(x) = exp_0(W log_0(x)) (+)_c b."""

    def __init__(self, in_dim, out_dim, c=C):
        super().__init__()
        self.c = c
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        out = mobius_matvec(self.W, x, self.c)
        return mobius_add(out, self.b, self.c)


def project_to_ball(x, cap=0.9):
    n = torch.linalg.norm(x)
    lim = cap / math.sqrt(C)
    return x * (lim / (n + 1e-12)) if n > lim else x


def main():
    torch.manual_seed(0)
    layer = MobiusLinear(4, 3)
    xs = [project_to_ball(0.3 * torch.randn(4)) for _ in range(5)]

    print("Mobius linear layer forward pass:")
    for i, x in enumerate(xs):
        y = layer(x)
        print(f"  x{i} ||x||={torch.linalg.norm(x).item():.3f} -> "
              f"||y||={torch.linalg.norm(y).item():.3f}")

    # gradient check
    x0 = project_to_ball(0.2 * torch.randn(4))
    y0 = layer(x0)
    loss = torch.linalg.norm(y0)
    loss.backward()
    print(f"\nGradient check: ||dL/dW||={torch.linalg.norm(layer.W.grad).item():.4f}, "
          f"||dL/db||={torch.linalg.norm(layer.b.grad).item():.4f}")

    # verify exp_0 . log_0 = id
    x = project_to_ball(0.5 * torch.randn(4))
    approx = exp_0(log_0(x))
    print(f"\nexp_0(log_0(x)) ~ x:  max|diff|={torch.max(torch.abs(approx - x)).item():.2e}")

    print("\n(Mobius linear: Euclidean matvec wrapped by log_0/exp_0; bias added via Mobius addition.)")


if __name__ == "__main__":
    main()
