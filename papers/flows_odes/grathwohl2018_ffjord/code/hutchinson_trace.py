"""
FFJORD (Grathwohl et al. 2019) — minimal core.

1. Confirms Hutchinson's identity E_eps[eps^T J eps] = tr(J) for a small MLP's Jacobian,
   via Monte-Carlo averaging (Rademacher and Gaussian eps).
2. Integrates a tiny CNF's log-density change dlogp/dt = -tr(df/dh) over a few Euler
   steps using (a) the exact trace and (b) a K-sample-averaged Hutchinson estimator,
   and checks the two log-density changes agree closely.

Run:  python hutchinson_trace.py
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class TinyMLP(nn.Module):
    def __init__(self, d, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))

    def forward(self, h):
        return self.net(h)


def exact_trace(f, h):
    """tr(df/dh) via d separate vector-Jacobian products (the thing FFJORD avoids)."""
    d = h.shape[0]
    h = h.clone().requires_grad_()
    y = f(h)
    trace = torch.zeros(())
    for i in range(d):
        grad_i = torch.autograd.grad(y[i], h, retain_graph=True)[0]
        trace = trace + grad_i[i]
    return trace.detach()


def hutchinson_estimate(f, h, eps):
    """Single-sample Hutchinson estimate eps^T J eps via one VJP."""
    h = h.clone().requires_grad_()
    y = f(h)
    vjp = torch.autograd.grad((y * eps).sum(), h)[0]
    return torch.dot(eps, vjp)


def check_hutchinson_convergence():
    d = 12
    f = TinyMLP(d)
    h = torch.randn(d)
    tr = exact_trace(f, h)

    for n_samples, kind in [(20, "rademacher"), (2000, "rademacher"), (2000, "gaussian")]:
        ests = []
        for _ in range(n_samples):
            if kind == "rademacher":
                eps = (torch.randint(0, 2, (d,)).float() * 2 - 1)
            else:
                eps = torch.randn(d)
            ests.append(hutchinson_estimate(f, h, eps).item())
        mean_est = sum(ests) / n_samples
        err = abs(mean_est - tr.item())
        print(f"  {kind:>10} n={n_samples:<5} mean(eps^T J eps)={mean_est:8.4f}  "
              f"tr(J)={tr.item():8.4f}  |err|={err:.4f}")

    # a large-sample average must land close to the exact trace (law of large numbers)
    ests = [hutchinson_estimate(f, h, torch.randint(0, 2, (d,)).float() * 2 - 1).item()
            for _ in range(4000)]
    mean_est = sum(ests) / len(ests)
    err = abs(mean_est - tr.item())
    assert err < 0.1, "Hutchinson's estimator must average to the exact trace"
    print(f"\nPASS: 4000-sample Hutchinson average matches exact trace (|err|={err:.4f}).\n")


def check_cnf_logdensity():
    d = 6
    f = TinyMLP(d, hidden=10)
    h0 = torch.randn(d)
    steps, dt = 8, 0.05

    # exact: integrate dh/dt = f(h), dlogp/dt = -tr(df/dh), via Euler
    h = h0.clone()
    delta_logp_exact = 0.0
    for _ in range(steps):
        delta_logp_exact -= exact_trace(f, h).item() * dt
        h = h.detach() + dt * f(h).detach()

    # Hutchinson: same trajectory, replace exact trace with a K-sample average per step
    K = 300
    h = h0.clone()
    delta_logp_hutch = 0.0
    for _ in range(steps):
        ests = [hutchinson_estimate(f, h, torch.randint(0, 2, (d,)).float() * 2 - 1).item()
                for _ in range(K)]
        delta_logp_hutch -= (sum(ests) / K) * dt
        h = h.detach() + dt * f(h).detach()

    err = abs(delta_logp_exact - delta_logp_hutch)
    print(f"CNF delta log p:  exact = {delta_logp_exact:.4f},  "
          f"{K}-sample Hutchinson = {delta_logp_hutch:.4f}  (abs. err = {err:.4f})")
    # exact delta-logp can be near zero (trace can nearly cancel across steps), so an
    # absolute tolerance is the meaningful check here, not a relative one.
    assert err < 0.03, "K-sample-averaged Hutchinson CNF log-density must track the exact one"
    print("PASS: Hutchinson-estimated log-density change tracks the exact instantaneous "
          "change-of-variables integral.")


def main():
    print("== 1. Hutchinson's trace estimator converges to the exact Jacobian trace ==")
    check_hutchinson_convergence()
    print("== 2. CNF log-density change: exact trace vs Hutchinson estimator ==")
    check_cnf_logdensity()


if __name__ == "__main__":
    main()
