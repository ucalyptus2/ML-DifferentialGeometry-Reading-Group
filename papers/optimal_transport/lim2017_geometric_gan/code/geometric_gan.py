"""
Geometric GAN (Lim et al., 2017) -- minimal core.

Implements the hinge-loss discriminator (SVM-style maximum-margin separator) and
the one-sided-hinge generator loss on a toy 2D distribution.

Discriminator loss:  L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]
Generator loss:      L_G = E[max(0, 1 - D(G(z)))]   (one-sided hinge)

Demo: real data = 2D Gaussian at (2, 2).  Train G and D alternately; verify
the generator's output mean converges toward the data centre and the
inter-distribution distance shrinks vs. the pre-training baseline.

Run:  python geometric_gan.py
"""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))


class Generator(nn.Module):
    def __init__(self, z_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Unnormalised score f(x) -> R; sign classifies real (+) / fake (-)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def hinge_d_loss(d_real, d_fake):
    """SVM-style two-sided hinge: real >= +1, fake <= -1."""
    return torch.relu(1.0 - d_real).mean() + torch.relu(1.0 + d_fake).mean()


def hinge_g_loss(d_fake):
    """One-sided hinge: push fakes past the +1 margin."""
    return torch.relu(1.0 - d_fake).mean()


def main():
    torch.manual_seed(42)
    data_mean = torch.tensor([2.0, 2.0])
    data_std = 0.3

    def sample_real(n):
        return torch.randn(n, 2) * data_std + data_mean

    G, D = Generator(), Discriminator()
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999))
    bs, k_d = 128, 5

    # --- baseline: generator output before training ---
    with torch.no_grad():
        z0 = torch.randn(500, 4)
        gen0 = G(z0)
        d0 = (gen0.mean(0) - data_mean).norm().item()

    for step in range(3000):
        # --- discriminator update (k_d steps) ---
        for _ in range(k_d):
            xr = sample_real(bs)
            zf = torch.randn(bs, 4)
            with torch.no_grad():
                xf = G(zf)
            d_loss = hinge_d_loss(D(xr), D(xf))
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        # --- generator update ---
        zf = torch.randn(bs, 4)
        xf = G(zf)
        g_loss = hinge_g_loss(D(xf))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        if step % 1000 == 0:
            with torch.no_grad():
                xr = sample_real(500); xf = G(torch.randn(500, 4))
                shift = (xf.mean(0) - data_mean).norm().item()
            print(f"  step {step:4d}  D_loss={d_loss.item():.3f}  G_loss={g_loss.item():.3f}"
                  f"  ||mean(G)-mean(data)||={shift:.3f}")

    # --- verification ---
    with torch.no_grad():
        gen = G(torch.randn(1000, 4))
        real = sample_real(1000)
        gen_mean = gen.mean(0)
        real_mean = real.mean(0)
        shift = (gen_mean - real_mean).norm().item()
        gen_std = gen.std(0).mean().item()

    print(f"\nPre-training  ||mean(G)-mean(data)|| : {d0:.3f}")
    print(f"Post-training ||mean(G)-mean(data)|| : {shift:.3f}")
    print(f"Real data mean  : {real_mean.tolist()}")
    print(f"Generated mean  : {gen_mean.tolist()}")
    print(f"Generated std   : {gen_std:.3f}  (real std ~{data_std:.3f})")
    assert shift < 0.5, "generator did not converge toward data"
    assert shift < d0 * 0.3, "insufficient improvement over baseline"
    print("\nPASS: generator moved toward the data distribution (margin-based hinge).")


if __name__ == "__main__":
    main()
