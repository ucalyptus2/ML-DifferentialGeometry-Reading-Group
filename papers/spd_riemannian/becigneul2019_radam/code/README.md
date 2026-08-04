# Riemannian Adam — code

```bash
python riemannian_adam_demo.py
```

Runs a toy "pull toward the boundary" loss on the Poincare ball with naive Euclidean Adam
(ambient-coordinate update) vs Riemannian Adam (same moment estimates, retracted via
`poincare_exp`). Naive Adam's iterate norm blows past 1.0 (leaves the ball); Riemannian Adam's
norm approaches but never reaches 1.0, since the exponential map always lands strictly inside.
