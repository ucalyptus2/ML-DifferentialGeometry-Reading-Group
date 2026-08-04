"""
code/manifold_ops.py
====================
Shared, tested geometric primitives used across the reading-group paper
implementations.  Import like:

    from manifold_ops import (sphere_exp, sphere_log, poincare_dist,
                              poincare_exp, poincare_log, mobius_add,
                              spd_exp, spd_log, expm_skew)

All ops are implemented in PyTorch and operate on the last dimension(s).
"""
import math
import torch


# ---------------------------- Sphere S^n ---------------------------- #
def sphere_exp(x, v):
    """exp_x(v) on the unit sphere; v must be tangent (v . x = 0)."""
    n = torch.linalg.norm(v)
    return torch.cos(n) * x + torch.sin(n) * (v / (n + 1e-12))


def sphere_log(x, y):
    """log_x(y) on the unit sphere."""
    xy = torch.dot(x, y)
    coef = torch.acos(torch.clamp(xy, -1.0 + 1e-7, 1.0 - 1e-7))
    return coef / torch.sqrt(torch.clamp(1 - xy * xy, 1e-7, 1.0)) * (y - xy * x)


# ----------------------- Poincare ball B^n -------------------------- #
def mobius_add(x, y, c=1.0):
    """Möbius addition in the Poincare ball of curvature -c."""
    xy = torch.dot(x, y)
    x2, y2 = torch.dot(x, x), torch.dot(y, y)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    return num / (den + 1e-12)


def poincare_dist(u, v, c=1.0):
    """Geodesic distance in the Poincare ball."""
    m = u - v
    d2 = torch.dot(m, m)
    den = (1 - c * torch.dot(u, u)) * (1 - c * torch.dot(v, v))
    return torch.acosh(torch.clamp(1 + 2 * c * d2 / (den + 1e-12), 1.0 + 1e-7, 1e7)) / math.sqrt(c)


def poincare_exp(x, v, c=1.0):
    """Exponential map on the Poincare ball at x along tangent v."""
    vn = torch.linalg.norm(v) + 1e-12
    x2 = torch.dot(x, x)
    lamx = 2.0 / (1 - c * x2)
    second = torch.tanh(math.sqrt(c) * lamx * vn / 2) * v / (math.sqrt(c) * vn)
    return mobius_add(x, second, c)


def poincare_log(x, y, c=1.0):
    """Logarithmic map on the Poincare ball."""
    add = mobius_add(-x, y, c)
    addn = torch.linalg.norm(add) + 1e-12
    x2 = torch.dot(x, x)
    lamx = 2.0 / (1 - c * x2)
    coef = 2.0 / (math.sqrt(c) * lamx) * torch.atanh(torch.clamp(math.sqrt(c) * addn, 0, 1 - 1e-7))
    return coef * add / addn


# --------------------------- SPD manifold --------------------------- #
def spd_exp(A, V):
    """Affine-invariant Exp_A(V) = A^{1/2} e^{A^{-1/2} V A^{-1/2}} A^{1/2}."""
    n = A.shape[0]
    L = torch.linalg.cholesky(A + 1e-8 * torch.eye(n))
    S = torch.linalg.inv(L)
    M = S @ V @ S.T
    eigv, eigvec = torch.linalg.eigh(M)
    expM = eigvec @ torch.diag(torch.exp(eigv)) @ eigvec.T
    return L @ expM @ L.T


def spd_log(A, B):
    """Inverse of spd_exp."""
    n = A.shape[0]
    L = torch.linalg.cholesky(A + 1e-8 * torch.eye(n))
    S = torch.linalg.inv(L)
    M = S @ B @ S.T
    eigv, eigvec = torch.linalg.eigh(M)
    logM = eigvec @ torch.diag(torch.log(torch.clamp(eigv, 1e-12))) @ eigvec.T
    return L @ logM @ L.T


def spd_dist(A, B):
    """Affine-invariant distance on SPD."""
    n = A.shape[0]
    S = torch.linalg.inv(torch.linalg.cholesky(A + 1e-8 * torch.eye(n)))
    M = S @ B @ S.T
    eigv, _ = torch.linalg.eigh(M)
    return torch.norm(torch.log(torch.clamp(eigv, 1e-12)))


# --------------------------- SO(3) ---------------------------------- #
def expm_skew(w):
    """exp: so(3) -> SO(3); w is a 3-vector (axis * angle)."""
    theta = torch.linalg.norm(w)
    if theta < 1e-8:
        return torch.eye(3)
    W = torch.tensor([[0.0, -w[2], w[1]],
                      [w[2], 0.0, -w[0]],
                      [-w[1], w[0], 0.0]])
    K = W / theta
    return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
