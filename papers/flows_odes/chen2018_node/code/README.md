# Neural ODEs — code

```bash
python node_adjoint.py
```

1. Integrates `dh/dt = A h` with RK4 and checks it matches the closed-form flow
   `matrix_exp(A*T) @ h0` (the ODE is a one-parameter Lie-group flow).
2. Hand-codes the discrete adjoint method (backward-in-time state replay + gradient
   accumulation) and checks it reproduces autograd's gradient through the unrolled Euler
   solver exactly — the same idea as Neural ODE's O(1)-memory continuous adjoint.
