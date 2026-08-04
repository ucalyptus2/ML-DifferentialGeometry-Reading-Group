"""
p4 Group Equivariant Convolution (Cohen & Welling, ICML 2016).

Implements the geometric core of a G-CNN on the group  p4 = (Z^2) x C4 :
  * the cyclic rotation group C4 acting on the grid (90-degree rotations),
  * the LIFTING convolution  (planar feature map -> group feature map),
  * the GROUP convolution    (group feature map -> group feature map),
and verifies numerically that the group convolution is equivariant under the
left group action (translations and 90-degree rotations).

Only torch is required.  Run:
    python g4_conv.py
"""

import torch
import torch.nn.functional as F

G = 4  # order of C4 (90-degree rotations)


def rotate_grid(x: torch.Tensor, r: int) -> torch.Tensor:
    """Rotate a [..., H, W] tensor by r*90 degrees CCW (the C4 action)."""
    return torch.rot90(x, k=r % G, dims=(-2, -1))


# ---------------------------------------------------------------------------
# p4 = (Z^2) x C4.  A group feature map f: p4 -> R is a tensor of shape
#   [N, C, G, H, W]  (critically the G axis indexes the rotation/orientation).
# ---------------------------------------------------------------------------
def lifting_conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Planar feature map -> group feature map:  f(g=(t,r)) = (R_r x) * psi_r.

    x: [N, C, H, W]    w: [G, C, K, K]  (a scalar conv kernel per rotation)
    Returns [N, G, OH, OW]: the lifted, orientation-enriched feature map,
    with the C4 orientation as a dedicated axis (translation = OH,OW).
    """
    out = torch.stack([F.conv2d(rotate_grid(x, r), w[r].unsqueeze(0)).squeeze(1)
                       for r in range(G)], dim=1)
    return out


def group_conv(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Group -> group convolution:  out(g) = sum_h f(h) psi(g^{-1} h).

    f: [N, C, G, H, W]    w: [C_out, C, G, K, K]
    With g=(t,r), h=(t',r'), and  g^{-1}h = ( r^{-1}(t'-t), r^{-1}r' ):
        out(t,r) = sum_{r''} [ f[:,:, rr'']  CROSS-CORR  (R_{r^{-1}} flip psi_{r''}) ](t)
    The spatial term is a standard cross-correlation (torch conv2d), and the
    kernel is the r^{-1}-rotated, flipped copy of psi_{r''}.
    """
    N, C, Gc, H, W = f.shape
    C_out, _, _, K, _ = w.shape
    OH, OW = H - K + 1, W - K + 1
    acc = torch.zeros(N, C_out, Gc, OH, OW, dtype=f.dtype, device=f.device)
    for r in range(Gc):                 # output rotation index
        for rpp in range(Gc):           # r'' = r^{-1} r'  kernel index
            kernel = rotate_grid(w[:, :, rpp].flip(-2, -1), (Gc - r) % Gc)
            corr = F.conv2d(f[:, :, (r * rpp) % Gc], kernel)  # cross-correlation
            acc[:, :, r] += corr
    return acc


# ---------------------------------------------------------------------------
# Left group action used in the equivariance check:  (L_u f)(g) = f(u^{-1} g).
# For u = (t, r): orient the planes by r_u^{-1}, cyclically shift the C4 axis
# by -r, rotate the pixel grid by r, then translate by t.
# ---------------------------------------------------------------------------
def left_action(f: torch.Tensor, t: tuple, r: int) -> torch.Tensor:
    out = f.roll(-r, dims=2)                       # orientation shift on C4 axis
    out = torch.stack([rotate_grid(out[:, :, j], r) for j in range(G)], dim=2)
    # translation by t in pixel space (circular roll; exact with valid conv)
    out = out.roll(shifts=(-t[0], -t[1]), dims=(-2, -1))
    return out


def _demo():
    torch.manual_seed(0)
    H = W = 9
    K = 3
    C_in, C_out = 3, 2

    x = torch.randn(2, C_in, H, W)
    wl = torch.randn(G, C_in, K, K)
    lifted = lifting_conv(x, wl)
    assert lifted.shape == (2, G, H - K + 1, W - K + 1)

    Hf = Wf = 7
    f = torch.randn(2, C_in, G, Hf, Wf)
    wg = torch.randn(C_out, C_in, G, K, K)
    out = group_conv(f, wg)
    assert out.shape == (2, C_out, G, Hf - K + 1, Wf - K + 1)

    tests = [((0, 0), 0), ((1, 0), 0), ((0, -2), 0), ((0, 0), 1), ((1, 1), 2)]
    ok = True
    for (t, r) in tests:
        lhs = group_conv(left_action(f, t, r), wg)   # act-then-conv
        rhs = left_action(out, t, r)                 # conv-then-act
        err = (lhs - rhs).abs().max().item()
        tol = 1e-3
        if err >= tol:
            ok = False
        print(f"  L_{{({list(t)},{r})}} : max|conv o act - act o conv| = {err:.2e}  "
              f"{'OK' if err < tol else 'FAIL'}")

    print("\np4 group convolution is LEFT-EQUIVARIANT:", ok)


if __name__ == "__main__":
    _demo()
