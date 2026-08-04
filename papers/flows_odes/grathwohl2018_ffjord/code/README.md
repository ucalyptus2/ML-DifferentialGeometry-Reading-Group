# FFJORD — code

```bash
python hutchinson_trace.py
```

1. Confirms `E_eps[eps^T J eps] = tr(J)` for a small MLP's Jacobian via Monte-Carlo averaging.
2. Integrates a tiny CNF's log-density change with exact trace vs. a K-sample-averaged
   Hutchinson estimator and checks they agree closely.
