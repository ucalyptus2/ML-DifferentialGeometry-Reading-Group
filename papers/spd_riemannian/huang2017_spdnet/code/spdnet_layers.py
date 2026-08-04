"""
SPDNet (Huang & Van Gool 2017) — minimal core.

Implements BiMap / ReEig / LogEig, chains them into a tiny 2-layer SPDNet, and
checks: (1) every intermediate activation stays SPD, (2) LogEig is exactly the
Riemannian log map at the identity (Log_I), matching manifold_ops.spd_log(I, A).

Run:  python spdnet_layers.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "code"))
from manifold_ops import spd_log, spd_exp   # noqa: E402


def bimap(A, W):
    """BiMap layer: A -> W A W^T. Preserves SPD if W (out_dim x in_dim, out_dim <= in_dim)
    has full row rank."""
    return W @ A @ W.T


def reeig(A, eps=1e-4):
    """ReEig layer: rectify small eigenvalues, keep strictly SPD (analogue of ReLU)."""
    eigv, eigvec = torch.linalg.eigh(A)
    eigv = torch.clamp(eigv, min=eps)
    return eigvec @ torch.diag(eigv) @ eigvec.T


def logeig(A):
    """LogEig layer: A -> U log(Sigma) U^T, the Riemannian log map at the identity."""
    eigv, eigvec = torch.linalg.eigh(A)
    return eigvec @ torch.diag(torch.log(torch.clamp(eigv, 1e-12))) @ eigvec.T


def is_spd(A, tol=1e-6):
    """Check symmetry + strict positive-definiteness via Cholesky."""
    if (A - A.T).abs().max().item() > tol:
        return False
    try:
        torch.linalg.cholesky(A + tol * torch.eye(A.shape[0]))
        eigv = torch.linalg.eigvalsh(A)
        return bool((eigv > 0).all())
    except RuntimeError:
        return False


class TinySPDNet(torch.nn.Module):
    """[BiMap -> ReEig] x 2 -> LogEig, dims n0 -> n1 -> n2."""

    def __init__(self, n0=6, n1=4, n2=3):
        super().__init__()
        # random row-full-rank weights: Q from QR of an (n_in x n_out) matrix has
        # orthonormal COLUMNS of shape (n_in, n_out); W = Q^T is (n_out, n_in) with
        # orthonormal ROWS, i.e. full row rank, as BiMap requires.
        W1 = torch.linalg.qr(torch.randn(n0, n1))[0].T
        W2 = torch.linalg.qr(torch.randn(n1, n2))[0].T
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(W2)

    def forward(self, A):
        A1 = reeig(bimap(A, self.W1))
        A2 = reeig(bimap(A1, self.W2))
        return logeig(A2), (A1, A2)


def random_spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g)
    return M @ M.T + n * torch.eye(n)


def main():
    torch.manual_seed(0)
    net = TinySPDNet(n0=6, n1=4, n2=3)
    A0 = random_spd(6, seed=1)

    out, (A1, A2) = net(A0)

    print("Input SPD:            ", is_spd(A0))
    print("After BiMap+ReEig #1: ", is_spd(A1))
    print("After BiMap+ReEig #2: ", is_spd(A2))
    assert is_spd(A0) and is_spd(A1) and is_spd(A2), \
        "every SPDNet activation must remain SPD"
    print("PASS: every intermediate activation is a valid SPD matrix.\n")

    # LogEig must equal Log_I(A2) = spd_log(I, A2) from manifold_ops.
    n2 = A2.shape[0]
    ref = spd_log(torch.eye(n2), A2)
    err = (out - ref).abs().max().item()
    print(f"LogEig vs manifold_ops.spd_log(I, A2):  max|diff| = {err:.3e}")
    assert err < 1e-4, "LogEig must equal the Riemannian log map at the identity"
    print("PASS: LogEig is exactly Log_I.\n")

    # round-trip: spd_exp(I, LogEig(A2)) must recover A2 (Log/Exp are mutual inverses at I).
    recon = spd_exp(torch.eye(n2), out)
    rt_err = (recon - A2).abs().max().item()
    print(f"spd_exp(I, LogEig(A2)) vs A2:            max|diff| = {rt_err:.3e}")
    assert rt_err < 1e-3, "Exp_I and Log_I must be mutual inverses"
    print("PASS: LogEig / Exp_I round-trip is exact.")


if __name__ == "__main__":
    main()
