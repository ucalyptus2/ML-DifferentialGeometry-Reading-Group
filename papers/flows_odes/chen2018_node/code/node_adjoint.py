"""
Neural ODEs (Chen et al. 2018) — minimal core.

Two checks on the linear ODE dh/dt = A h:
  1. An RK4 solver matches the closed-form flow map torch.matrix_exp(A*T) @ h0
     (dh/dt = Ah is a one-parameter Lie-group flow: h(t) = e^{tA} h(0)).
  2. A hand-coded discrete adjoint method (integrate a companion "adjoint" state
     backward and accumulate a parameter-gradient integral) reproduces the exact
     gradient dL/dA that autograd computes by backpropagating through the unrolled
     Euler solver — the same discretization on both sides makes the two agree to
     near machine precision, which is the discrete analogue of Neural ODE's
     continuous adjoint sensitivity method.

Run:  python node_adjoint.py
"""
import torch

torch.manual_seed(0)


# ---------------- 1. RK4 solver vs closed-form matrix exponential ---------------- #
def rk4_integrate(A, h0, t0, t1, steps):
    dt = (t1 - t0) / steps
    h = h0
    for _ in range(steps):
        k1 = A @ h
        k2 = A @ (h + dt / 2 * k1)
        k3 = A @ (h + dt / 2 * k2)
        k4 = A @ (h + dt * k3)
        h = h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return h


def check_flow_map():
    n = 3
    A = torch.randn(n, n) * 0.5
    h0 = torch.randn(n)
    T = 1.0

    h_rk4 = rk4_integrate(A, h0, 0.0, T, steps=50)
    h_exact = torch.matrix_exp(A * T) @ h0

    err = (h_rk4 - h_exact).abs().max().item()
    print(f"RK4(dh/dt=Ah, T={T}) vs exp(AT)@h0:  max|diff| = {err:.3e}")
    assert err < 1e-4, "RK4 integration of a linear ODE must match its matrix-exponential flow"
    print("PASS: the ODE solver reproduces the closed-form Lie-group flow e^{tA}.\n")


# ---------------- 2. discrete adjoint method vs autograd ---------------- #
def euler_forward(A, h0, steps, dt):
    """Euler-unrolled forward pass; returns final state and the full history."""
    h = h0
    hist = [h]
    for _ in range(steps):
        h = h + dt * (A @ h)
        hist.append(h)
    return h, hist


def adjoint_grad(A, h0, target, steps, dt):
    """Hand-coded discrete adjoint: replay the Euler steps backward, carrying an
    adjoint state a and accumulating dL/dA, without ever building an autograd graph."""
    _, hist = euler_forward(A.detach(), h0, steps, dt)
    hT = hist[-1]
    a = hT - target                          # dL/dh(T) for L = 1/2 ||h(T) - target||^2
    gradA = torch.zeros_like(A)
    for i in range(steps - 1, -1, -1):
        h_i = hist[i]
        gradA = gradA + dt * torch.outer(a, h_i)   # dL/dA contribution from step i
        a = a + dt * (A.T @ a)                     # discrete adjoint step (backward)
    return gradA


def check_adjoint():
    n = 3
    steps, dt = 40, 0.02
    A = torch.randn(n, n) * 0.5
    h0 = torch.randn(n)
    target = torch.randn(n)

    # ground truth: autograd through the unrolled Euler solver
    A_ag = A.clone().requires_grad_()
    hT, _ = euler_forward(A_ag, h0, steps, dt)
    loss = 0.5 * ((hT - target) ** 2).sum()
    loss.backward()
    grad_autograd = A_ag.grad

    grad_adjoint = adjoint_grad(A, h0, target, steps, dt)

    err = (grad_adjoint - grad_autograd).abs().max().item()
    print(f"Adjoint dL/dA vs autograd dL/dA:  max|diff| = {err:.3e}")
    assert err < 1e-4, "the discrete adjoint method must reproduce autograd's gradient exactly"
    print("PASS: backward-in-time adjoint integration matches autograd, "
          "without ever storing the forward computation graph.")


def main():
    print("== 1. ODE solver reproduces the exact flow map ==")
    check_flow_map()
    print("== 2. Adjoint method reproduces autograd's gradient in O(1) graph memory ==")
    check_adjoint()


if __name__ == "__main__":
    main()
