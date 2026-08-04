# SPDNet — code

```bash
python spdnet_layers.py
```

Runs a tiny 2-layer `[BiMap -> ReEig] x 2 -> LogEig` network on a random 6x6 SPD input and checks:
1. every intermediate activation is still a valid SPD matrix,
2. `LogEig` matches `manifold_ops.spd_log(I, A)` exactly (it *is* the log map at the identity),
3. `spd_exp(I, LogEig(A))` round-trips back to `A`.
