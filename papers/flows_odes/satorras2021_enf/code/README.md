# E(n) Equivariant Normalizing Flows — code

```bash
python enf_flow_equivariance.py
```

Integrates a 6-step Euler flow using the EGNN layer from
`papers/group_equivariance/satorras2021_egnn/code/egnn_layer.py` as the vector field, and checks
the *whole trajectory* — not just one layer — commutes with rotation + translation + permutation.
