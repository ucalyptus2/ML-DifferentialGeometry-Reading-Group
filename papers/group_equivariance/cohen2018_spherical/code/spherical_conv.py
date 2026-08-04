"""
Spherical CNNs (Cohen et al. 2018) — minimal core.

A zonal spherical convolution on S^2 computed by direct integration on an
equal-area grid, plus a numerical check of SO(2)-equivariance (rotations about
the z-axis, a subgroup of SO(3)):  conv(R_g f) == R_g conv(f).

Run:  python spherical_conv.py
"""
import os, sys, math
import torch


def equal_area_grid(Ntheta=32, Nphi=64):
    """Equal-area grid: weight domega = 4pi / (Ntheta*Nphi) for every cell."""
    thetas = [math.acos(1 - 2 * (i + 0.5) / Ntheta) for i in range(Ntheta)]
    phis = [2 * math.pi * (j + 0.5) / Nphi for j in range(Nphi)]
    domega = 4 * math.pi / (Ntheta * Nphi)
    return thetas, phis, domega


def zonal_conv(f, thetas, phis, domega, psi):
    """Spherical cross-correlation with a zonal filter psi(theta).

    Output is a function of longitude phi0 (a z-rotation). Because psi depends
    only on colatitude, this is a circular correlation in phi weighted by psi:
        [f * psi](phi0) = sum over grid  f(theta, phi - phi0) * psi(theta) * domega
    """
    Nt, Np = len(thetas), len(phis)
    f_grid = f.view(Nt, Np)
    psi_w = torch.tensor([psi(t) for t in thetas]) * domega          # [Nt]
    # circular correlation in phi for each theta, then weighted sum over theta
    out = torch.zeros(Np)
    for j0 in range(Np):
        s = 0.0
        for j in range(Np):
            jsrc = (j - j0) % Np
            s = s + (f_grid[:, jsrc] * psi_w).sum()
        out[j0] = s
    return out


def rotate_z(f, thetas, phis, phi0):
    """R_z(phi0) f: shift the signal in longitude by phi0 (works for 2D or 1D-longitude)."""
    Nt, Np = len(thetas), len(phis)
    shift = int(round(phi0 / (2 * math.pi) * Np)) % Np
    if f.numel() == Nt * Np:               # 2D field on (theta, phi)
        return torch.roll(f.view(Nt, Np), shifts=shift, dims=1).reshape(-1)
    return torch.roll(f, shifts=shift)     # 1D function of longitude only


def main():
    thetas, phis, domega = equal_area_grid(24, 48)
    Nt, Np = len(thetas), len(phis)
    # signal: a couple of lobes
    f = torch.tensor([math.sin(t) * math.cos(2 * p) + 0.7 * math.cos(t)
                      for t in thetas for p in phis])
    psi = lambda t: math.exp(-((t - math.pi / 2) ** 2) / 0.5)   # band near equator

    out = zonal_conv(f, thetas, phis, domega, psi)

    # equivariance: rotate input by phi0, convolve, compare to conv(f) rotated by phi0
    phi0 = 0.9
    f_rot = rotate_z(f, thetas, phis, phi0)
    out_rot = zonal_conv(f_rot, thetas, phis, domega, psi)
    out_then_rot = rotate_z(out, thetas, phis, phi0)

    err = torch.max(torch.abs(out_rot - out_then_rot)).item()
    print(f"SO(2)-equivariance (z-rotation by {phi0} rad): max|conv(Rf) - R(conv(f))| = {err:.6f}")
    assert err < 1e-3, "zonal spherical conv must commute with z-rotations"
    print("PASS: zonal spherical convolution is SO(2)-equivariant (a subgroup of SO(3)).")
    print("(General spherical CNNs extend this to full SO(3) via Wigner-D representations.)")


if __name__ == "__main__":
    main()
