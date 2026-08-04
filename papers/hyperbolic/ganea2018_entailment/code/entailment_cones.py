"""
Hyperbolic Entailment Cones (Ganea, Becigneul, Hofmann, ICML 2018) -- minimal core.

Implements:
  - Entailment cone half-aperture  alpha(u) = arcsin(sin(a0)*(1-c||u||^2)/(2*sqrt(c)*||u||))
  - Cone membership test:  v in C(u)  iff  angle_origin(u,v) <= alpha(u)
  - Hinge entailment loss for positive (ancestor) and negative pairs.

Run:  python entailment_cones.py
"""
import os, sys, math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import poincare_dist  # noqa: E402

C = 1.0


def cone_aperture(u, alpha0, c=C):
    """Half-aperture of the entailment cone at u.

    alpha(u) = arcsin( sin(a0) * (1 - c||u||^2) / (2*sqrt(c)*||u||) )
    Deeper points (larger ||u||) get narrower cones.
    """
    r = torch.linalg.norm(u) + 1e-8
    arg = math.sin(alpha0) * (1 - c * r ** 2) / (2 * math.sqrt(c) * r)
    return torch.arcsin(torch.clamp(arg, -1.0, 1.0))


def angle_at_origin(u, v):
    """Euclidean angle at origin between u and v (conformal model preserves angles)."""
    cos_a = torch.dot(u, v) / (torch.linalg.norm(u) * torch.linalg.norm(v) + 1e-12)
    return torch.arccos(torch.clamp(cos_a, -1.0 + 1e-7, 1.0 - 1e-7))


def in_cone(u, v, alpha0, c=C):
    """Check whether v lies in the entailment cone C(u)."""
    return angle_at_origin(u, v) <= cone_aperture(u, alpha0, c)


def entailment_loss(emb, pos, neg, alpha0, c=C, margin=0.05):
    """Hinge loss: pull descendants into cones, push negatives out."""
    loss = torch.tensor(0.0)
    for u, v in pos:
        ang = angle_at_origin(emb[u], emb[v])
        ap = cone_aperture(emb[u], alpha0, c)
        loss = loss + torch.relu(ang - ap + margin)    # v should be inside
    for u, v in neg:
        ang = angle_at_origin(emb[u], emb[v])
        ap = cone_aperture(emb[u], alpha0, c)
        loss = loss + torch.relu(ap - ang + margin)    # v should be outside
    return loss


def project_to_ball(x, cap=0.95):
    n = torch.linalg.norm(x)
    lim = cap / math.sqrt(C)
    return x * (lim / (n + 1e-12)) if n > lim else x


def main():
    torch.manual_seed(42)
    DIM = 8
    # tiny tree: 0(root) -> {1,2}, 1 -> {3,4}
    pos = [(0, 1), (0, 2), (1, 3), (1, 4)]
    neg = [(1, 0), (2, 1), (3, 1), (4, 0)]
    alpha0 = math.radians(30)

    # structured init: deeper nodes further from origin, clustered by subtree
    base = torch.randn(DIM)
    base = base / (torch.linalg.norm(base) + 1e-12)
    d1 = base + 0.15 * torch.randn(DIM)
    d1 = d1 / (torch.linalg.norm(d1) + 1e-12)
    d2 = base + 0.15 * torch.randn(DIM)
    d2 = d2 / (torch.linalg.norm(d2) + 1e-12)
    emb = [
        project_to_ball(0.12 * base),                       # root: near centre
        project_to_ball(0.35 * d1),                          # child 1
        project_to_ball(0.35 * d2),                          # child 2
        project_to_ball(0.55 * (d1 + 0.1 * torch.randn(DIM))),  # grandchild 3
        project_to_ball(0.55 * (d1 + 0.1 * torch.randn(DIM))),  # grandchild 4
    ]
    lr = 0.03
    for epoch in range(800):
        ev = [e.clone().requires_grad_(True) for e in emb]
        loss = entailment_loss(ev, pos, neg, alpha0)
        loss.backward()
        for i in range(len(emb)):
            emb[i] = project_to_ball(emb[i] - lr * ev[i].grad)

    print("Entailment cones after training:")
    for u, v in pos:
        ic = in_cone(emb[u].detach(), emb[v].detach(), alpha0).item()
        ang = angle_at_origin(emb[u].detach(), emb[v].detach()).item()
        ap = cone_aperture(emb[u].detach(), alpha0).item()
        print(f"  ancestor ({u}->{v}): in_cone={bool(ic)}  angle={ang:.3f}  aperture={ap:.3f}")
    for u, v in neg:
        ic = in_cone(emb[u].detach(), emb[v].detach(), alpha0).item()
        ang = angle_at_origin(emb[u].detach(), emb[v].detach()).item()
        ap = cone_aperture(emb[u].detach(), alpha0).item()
        print(f"  negative ({u}->{v}): in_cone={bool(ic)}  angle={ang:.3f}  aperture={ap:.3f}")
    norms = [round(torch.linalg.norm(e).item(), 3) for e in emb]
    print(f"\n  ||u|| per node (depth proxy): {norms}")
    final = entailment_loss([e.detach() for e in emb], pos, neg, alpha0).item()
    print(f"  Final hinge loss: {final:.4f}")
    print("\n(Entailment cones: deeper nodes get narrower apertures; descendants must fall inside.)")


if __name__ == "__main__":
    main()
