"""
Poincare Embeddings (Nickel & Kiela 2017) — minimal core.

Learns embeddings in the Poincare ball for a tiny hierarchy using a Riemannian
gradient step (project -> retract -> renormalize into the ball).

Run:  python poincare_embeddings.py
"""
import os, sys, math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import poincare_dist, poincare_exp   # noqa: E402

C = 1.0
DIM = 4

# a tiny 4-node tree: 0 -> {1,2}, 1 -> 3   (0 is root)
POS = [(0, 1), (0, 2), (1, 3)]
NEG = [(0, 3), (2, 3), (2, 1)]


def rgrad(x, y, eps=1e-5):
    """Riemannian gradient of the Poincare distance d(x,y) at x (tangent vector)."""
    x = x.clone().requires_grad_()
    d = poincare_dist(x, y, C)
    g = torch.autograd.grad(d, x)[0]
    lam = 2.0 / (1 - C * torch.dot(x.detach(), x.detach()))
    rg = lam * g                              # metric-scaled gradient (conformal -> one factor)
    n = torch.linalg.norm(rg)
    if n > 1.0:                               # clip to keep steps stable
        rg = rg / n
    return rg


def project_to_ball(x, max_norm=0.9):
    """Retraction onto the Poincare ball: clamp norm strictly below 1/sqrt(c)."""
    n = torch.linalg.norm(x)
    cap = max_norm / math.sqrt(C)
    if n > cap:
        return x * (cap / (n + 1e-12))
    return x


def step(emb, lr=0.01):
    for u, v in POS:
        du, dv = rgrad(emb[u], emb[v]), rgrad(emb[v], emb[u])
        emb[u] = project_to_ball(poincare_exp(emb[u].detach(), -lr * du, C))
        emb[v] = project_to_ball(poincare_exp(emb[v].detach(), -lr * dv, C))
    for u, v in NEG:
        gu = rgrad(emb[u], emb[v])
        emb[u] = project_to_ball(poincare_exp(emb[u].detach(), lr * 0.02 * gu, C))


def main():
    torch.manual_seed(0)
    emb = [0.01 * torch.randn(DIM) for _ in range(4)]
    for _ in range(400):
        step(emb, lr=0.05)
    print("Poincare distances after training:")
    for u, v in POS + NEG:
        rel = "ancestor" if (u, v) in POS else "unrelated"
        print(f"  d({u},{v}) = {poincare_dist(emb[u], emb[v], C).item():.3f}  ({rel})")
        norms = [torch.linalg.norm(e).item() for e in emb]
    print("  ||x|| per node (root stays near centre):", [round(n, 3) for n in norms])
    print("\n(toy 4-node demo: ancestor pairs are pulled together in Poincare distance;")
    print(" with a larger tree the embeddings spread toward the boundary to encode depth.)")

    # ---- geometric self-check: Poincare distance grows exponentially with radius ----
    # d(0, x) = 2*atanh(||x||)/sqrt(c). As ||x|| -> 1 boundary, d -> infinity.
    rs = [0.1, 0.5, 0.9, 0.99, 0.999]
    ds = [poincare_dist(torch.tensor([r, 0.0, 0.0, 0.0]), torch.zeros(4), C).item() for r in rs]
    print("\nPoincare ball model check  d(0, [r,0,0,0]) vs radius r:")
    for r, d in zip(rs, ds):
        print(f"  r={r:<6}  d={d:.3f}")
        # the key property: atanh is convex, so equal Euclidean steps near the boundary
    # produce arbitrarily large Poincaré steps (exponential separation of hierarchy).
    assert ds[0] < ds[1] < ds[2] < ds[3] < ds[4], "d must be monotonically increasing in r"
    assert ds[4] > 3 * ds[1], "d must grow super-linearly near the boundary (atanh convexity)"
    print("\nPASS: Poincare distance grows exponentially with Euclidean radius "
          "(hierarchical geometry).")



if __name__ == "__main__":
    main()
