"""
tutorial/code/riemannian_jax.py
===============================
Same geometric core as riemannian_pytorch.py but in JAX (jax.numpy + jax.scipy).

Run:  python riemannian_jax.py   (requires: pip install jax)
"""
import jax
import jax.numpy as jnp
import math


def sphere_exp(x, v):
    n = jnp.linalg.norm(v)
    return jnp.cos(n) * x + jnp.sin(n) * (v / (n + 1e-12))


def sphere_log(x, y):
    xy = jnp.dot(x, y)
    coef = jnp.arccos(jnp.clip(xy, -1.0 + 1e-7, 1.0 - 1e-7))
    return coef / jnp.sqrt(jnp.clip(1 - xy * xy, 1e-7, 1.0)) * (y - xy * x)


def poincare_dist(u, v, c=1.0):
    m = u - v
    d2 = jnp.dot(m, m)
    num = 2 * c * d2
    den = (1 - c * jnp.dot(u, u)) * (1 - c * jnp.dot(v, v))
    return jnp.arccosh(jnp.clip(1 + 2 * num / (den + 1e-12), 1.0 + 1e-7, 1e7)) / math.sqrt(c)


def spd_exp(A, V):
    """Exp_A(V) on SPD manifold (affine-invariant) : A^{1/2} e^{A^{-1/2} V A^{-1/2}} A^{1/2}."""
    S = jnp.linalg.inv(jnp.linalg.cholesky(A + 1e-8 * jnp.eye(A.shape[0])))
    M = S @ V @ S.T
    return jnp.linalg.cholesky(A + 1e-8 * jnp.eye(A.shape[0])) @ jax.scipy.linalg.expm(M) @ S


def spd_log(A, B):
    S = jnp.linalg.inv(jnp.linalg.cholesky(A + 1e-8 * jnp.eye(A.shape[0])))
    M = S @ B @ S.T
    return S.T @ jax.scipy.linalg.logm(M) @ S


def main():
    print("== sphere exp/log round trip ==")
    x = jnp.array([1.0, 0.0, 0.0])
    y = sphere_exp(x, jnp.array([0.0, 1.0, 0.0]) * 0.8)
    print("  log(exp(v)) =", sphere_log(x, y))

    print("\n== Poincare distance grows near boundary ==")
    near = jnp.array([0.05, 0.0, 0.0])
    far = jnp.array([0.85, 0.0, 0.0])
    print("  d(near,-near)=", round(float(poincare_dist(near, -near)), 3),
          " d(far,-far)=", round(float(poincare_dist(far, -far)), 3))

    print("\n== SPD exp/log round trip ==")
    A = jnp.array([[1.0, 0.2], [0.2, 1.2]])
    B = jnp.array([[1.3, 0.1], [0.1, 0.9]])
    B2 = spd_exp(A, spd_log(A, B))
    print("  ||exp_A(log_A(B)) - B|| =", round(float(jnp.linalg.norm(B2 - B)), 4))

    print("\n== Riemannian GD on the unit sphere (jax.grad) ==")

    def retract_gd(n_iter=60, lr=0.3):
        x = jnp.array([math.cos(0.1), math.sin(0.1), 0.0])
        for _ in range(n_iter):
            g = jax.grad(lambda z: -z[0])(x)
            g = g - jnp.dot(x, g) * x     # project onto tangent space
            x = x - lr * g
            x = x / jnp.linalg.norm(x)     # retraction
        return x

    print("  converged x =", retract_gd())


if __name__ == "__main__":
    main()
