# Manifold CNFs — code

```bash
python manifold_cnf_drift.py
```

Integrates the same tangent vector field on $S^2$ with naive ambient Euler (no retraction) vs
retraction-based Euler (`sphere_exp`), and measures $|\,\|x\|-1\,|$ — naive Euler drifts off the
sphere monotonically; the retraction-based integrator stays on it to numerical precision.
