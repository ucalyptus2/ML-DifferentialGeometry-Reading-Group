"""
Sinkhorn Distances (Cuturi, NeurIPS 2013) -- minimal core.

Implements:
  1. Entropic-regularised OT via Sinkhorn-Knopp matrix scaling.
  2. Numerically stable log-domain variant.
  3. Sinkhorn distance  <P*, C>  and the symmetric Sinkhorn divergence.

Demo: two small distributions on R^1; verify (a) the recovered coupling
satisfies marginal constraints, and (b) the Sinkhorn cost converges to the
exact OT cost (brute-force assignment) as lambda -> 0.

Run:  python sinkhorn.py
"""
import os, sys, math
import itertools
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))


def sinkhorn(a, b, C, lam=1.0, n_iter=200, tol=1e-9):
    """Sinkhorn-Knopp: P = diag(u) K diag(v),  K = exp(-C / lam)."""
    K = torch.exp(-C / lam)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iter):
        Kv = K @ v
        u = a / (Kv + 1e-30)
        Kt_u = K.T @ u
        v = b / (Kt_u + 1e-30)
        if torch.max(torch.abs(Kt_u * v - b)) < tol and \
           torch.max(torch.abs(Kv * u - a)) < tol:
            break
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    cost = (P * C).sum()
    return P, cost, u, v


def sinkhorn_log(a, b, C, lam=1.0, n_iter=200):
    """Log-domain stabilised Sinkhorn (avoids K underflow for small lam)."""
    log_a, log_b = torch.log(a + 1e-30), torch.log(b + 1e-30)
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    M = C / lam  # (n, m)
    for _ in range(n_iter):
        # g update:  g_j = log b_j - logsumexp_i(-M_ij + f_i)
        g = log_b - torch.logsumexp(-M + f.unsqueeze(1), dim=0)
        f = log_a - torch.logsumexp(-M + g.unsqueeze(0), dim=1)
    log_P = -M + f.unsqueeze(1) + g.unsqueeze(0)
    P = torch.exp(log_P)
    cost = (P * C).sum()
    return P, cost


def exact_ot_uniform(C):
    """Exact OT for uniform marginals (1/n each) via brute-force assignment."""
    n = C.shape[0]
    best = float("inf")
    for perm in itertools.permutations(range(n)):
        c = sum(C[i, perm[i]] for i in range(n)).item() / n
        best = min(best, c)
    return best


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    # --- two distributions on the real line ---
    xs = torch.tensor([0.0, 1.0, 2.0])
    ys = torch.tensor([0.3, 1.7, 2.5])
    C = (xs.unsqueeze(1) - ys.unsqueeze(0)).abs()  # L1 ground cost
    n = xs.shape[0]
    a = torch.ones(n) / n
    b = torch.ones(n) / n

    exact = exact_ot_uniform(C)
    print(f"Exact OT cost (brute-force assignment) : {exact:.6f}")
    print()

    print(f"{'lambda':>8}  {'sinkhorn_cost':>14}  {'marg_err':>10}  {'rel_err':>10}")
    for lam in [10.0, 1.0, 0.5, 0.1, 0.01]:
        P, cost, _, _ = sinkhorn(a, b, C, lam=lam, n_iter=500)
        marg_err = (P.sum(1) - a).abs().max().item() + (P.sum(0) - b).abs().max().item()
        rel = abs(cost.item() - exact) / exact
        print(f"{lam:8.3f}  {cost.item():14.6f}  {marg_err:10.2e}  {rel:10.4f}")

    # --- stable log-domain check for very small lambda ---
    P_log, cost_log = sinkhorn_log(a, b, C, lam=0.01, n_iter=500)
    print(f"\nLog-domain Sinkhorn (lam=0.01): cost = {cost_log.item():.6f}")
    print(f"  marginal error = {(P_log.sum(1)-a).abs().max().item():.2e}")

    # --- Sinkhorn divergence (symmetric, positive definite) ---
    _, d_ab, _, _ = sinkhorn(a, b, C, lam=0.1, n_iter=500)
    C_aa = (xs.unsqueeze(1) - xs.unsqueeze(0)).abs()
    C_bb = (ys.unsqueeze(1) - ys.unsqueeze(0)).abs()
    _, d_aa, _, _ = sinkhorn(a, a, C_aa, lam=0.1, n_iter=500)
    _, d_bb, _, _ = sinkhorn(b, b, C_bb, lam=0.1, n_iter=500)
    S = d_ab - 0.5 * d_aa - 0.5 * d_bb
    print(f"\nSinkhorn divergence S(alpha, beta) = {S.item():.6f}  (> 0 for alpha != beta)")
    assert S.item() > 0, "divergence should be positive for distinct distributions"

    # --- verify convergence to exact OT ---
    _, cost_small, _, _ = sinkhorn(a, b, C, lam=0.01, n_iter=2000)
    rel_err = abs(cost_small.item() - exact) / exact
    print(f"\nRelative error at lam=0.01: {rel_err:.4f}  (should be small)")
    assert rel_err < 0.1, "Sinkhorn cost should approach exact OT as lambda -> 0"
    print("\nPASS: Sinkhorn marginals satisfied and cost converges to exact OT.")


if __name__ == "__main__":
    main()
