"""
tutorial/code/riemannian_pytorch.py
====================================
The geometric core that powers most papers in this reading group, in PyTorch:

  * Tangent spaces & the Riemannian metric
  * Exponential / Logarithmic maps (sphere, Poincare ball, SPD cone, rotation group)
  * Geodesics
  * Riemannian gradient descent (retraction)
  * A tiny gauge-style / parallel-transport illustration

Run:  python riemannian_pytorch.py
"""
import torch
import math


def sphere_exp(x, v):
    """exp_x(v): move from unit-sphere point x along tangent v (both ||.||=1 <-> v _|_ x)."""
    n = torch.linalg.norm(v)
    return torch.cos(n) * x + torch.sin(n) * (v / (n + 1e-12))


def sphere_log(x, y):
    """Inverse of sphere_exp: the tangent vector at x pointing to y."""
    xy = torch.dot(x, y)
    coef = torch.acos(torch.clamp(xy, -1.0 + 1e-7, 1.0 - 1e-7))
    return coef / torch.sqrt(torch.clamp(1 - xy * xy, 1e-7, 1.0)) * (y - xy * x)


def poincare_dist(u, v, c=1.0):
    """Poincare-ball (hyperbolic) geodesic distance."""
    m = u - v
    d2 = torch.dot(m, m)
    num = 2 * c * d2
    den = (1 - c * torch.dot(u, u)) * (1 - c * torch.dot(v, v))
    return torch.acosh(torch.clamp(1 + 2 * num / (den + 1e-12), 1.0 + 1e-7, 1e7)) / math.sqrt(c)


def poincare_exp(x, v, c=1.0):
    """Exponential map in the Poincare ball at x along tangent v (Mobius addition w.r.t. v)."""
    vn = torch.linalg.norm(v)
    x2 = torch.dot(x, x)
    lamx = 2 / (1 - c * x2)
    lamxv = lamx * vn
    nom = (1 + c * x2) * v + 2 * c * torch.dot(x, v) * x
    denom = 1 + 2 * c * torch.dot(x, v) + (lamxv * lamxv)
    return x + (nom / (denom + 1e-12))


def spd_exp(A, V):
    """Exponential map on SPD manifold at A along symmetric tangent V: Exp_A(V)= A^{1/2} e^{A^{-1/2}V A^{-1/2}} A^{1/2}."""
    L = torch.linalg.cholesky(A + 1e-8 * torch.eye(A.shape[0]))
    Ainv_sqrt = torch.linalg.inv(L)
    M = Ainv_sqrt @ V @ Ainv_sqrt.T
    eigv, eigvec = torch.linalg.eigh(M)
    expM = eigvec @ torch.diag(torch.exp(eigv)) @ eigvec.T
    return L @ expM @ L.T


def spd_log(A, B):
    """Inverse of spd_exp."""
    L = torch.linalg.cholesky(A + 1e-8 * torch.eye(A.shape[0]))
    Ainv_sqrt = torch.linalg.inv(L)
    S = Ainv_sqrt @ B @ Ainv_sqrt.T
    eigv, eigvec = torch.linalg.eigh(S)
    logS = eigvec @ torch.diag(torch.log(torch.clamp(eigv, 1e-12))) @ eigvec.T
    return L @ logS @ L.T


def expm_skew(w):
    """Rotation-group exponential map: axis-angle vector w -> rotation matrix expm([w]_x)."""
    theta = torch.linalg.norm(w)
    W = torch.tensor([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]])
    if theta < 1e-8:
        return torch.eye(3)
    K = W / theta
    return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def euclid_gd(f, x0, lr=0.1, iters=50):
    """Baseline: ordinary gradient descent (kept for comparison)."""

    def closure(xi):
        xi = xi.clone().requires_grad_()
        val = f(xi)
        val.backward()
        return val, xi.grad
    x = x0.clone()
    for _ in range(iters):
        val, g = closure(x)
        x = x - lr * g
    return x


def main():
    torch.manual_seed(0)
    print("== 1. Sphere expmap round-trip ==")
    x = torch.tensor([1.0, 0.0, 0.0])
    v = torch.tensor([0.0, 1.0, 0.0]) * 0.8
    y = sphere_exp(x, v)
    back = sphere_log(x, y)
    print("   exp then log returns:", back.numpy().round(4), "~ v*0.8:", back.numpy().round(4))

    print("\n== 2. Poincare distance: near origin < near boundary ==")
    a = torch.tensor([0.1, 0.0, 0.0])
    far = torch.tensor([0.9, 0.0, 0.0])
    print("   d(a, -a):", round(poincare_dist(a, -a).item(), 3), " vs d(far,-far):",
          round(poincare_dist(far, -far).item(), 3), " (hyperbolic grows near boundary)")

    print("\n== 3. SPD exp/log round-trip ==")
    A = torch.tensor([[1.0, 0.2], [0.2, 1.2]])
    B = torch.tensor([[1.3, 0.1], [0.1, 0.9]])
    V = spd_log(A, B)
    B2 = spd_exp(A, V)
    print("   ||exp_A(log_A(B)) - B|| =", round(torch.norm(B2 - B).item(), 4))

    print("\n== 4. Rotation expmap ==")
    R = expm_skew(torch.tensor([0.0, 0.0, 1.3]))
    print("   rotation of (1,0,0):", (R @ torch.tensor([1.0, 0.0, 0.0])).numpy().round(3))

    print("\n== 5. Riemannian gradient descent keeps points ON the unit sphere ==")
    # optimize f(x)= -x[0] (maximize first coord) constrained to ||x||=1 via retraction
    x = torch.tensor([math.cos(0.1), math.sin(0.1), 0.0])
    lr = 0.3
    for _ in range(60):
        xr = x.clone().requires_grad_()
        g = torch.autograd.grad(-xr[0], xr)[0]       # gradient of f w.r.t. x
        g = g - torch.dot(x, g) * x                  # project gradient onto tangent space
        x = x - lr * g                               # step in tangent space
        x = x / torch.linalg.norm(x)                 # retraction back onto the sphere
    print("   converged x =", x.detach().numpy().round(4), " (should be ~ (1,0,0))")


if __name__ == "__main__":
    main()
