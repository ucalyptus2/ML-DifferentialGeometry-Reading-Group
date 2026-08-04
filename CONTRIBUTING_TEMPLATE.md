# Authoring template for ML × Differential Geometry reading group

Every paper lives at `papers/<category>/<slug>/` containing:

- `paper.md` — a structured markdown summary (see template below)
- `code/` — a **working, runnable** minimal core implementation (PyTorch and/or JAX),
  plus a short `README.md` inside code/ explaining how to run it.

## paper.md template

```markdown
# <Paper Title>

- **Authors:** <authors>
- **Venue/Year:** <venue, year>
- **arXiv:** <https://arxiv.org/abs/XXXX.XXXXX>
- **Category:** <category>

## One-paragraph TL;DR
<3-4 sentence summary of what the paper does and why it matters for ML×diff-geometry>

## The problem
<what problem, why it's hard, how geometry of the data matters>

## Key idea(s)
- <key idea 1>
- <key idea 2>

## The mathematics
<core formulas in LaTeX: manifolds, metric, exp/log map, group action, equivariance, geodesics, etc.
FOCUS on the differential-geometric content: tangent spaces, Riemannian metrics, #exp_m/log maps,
Lie groups/algebras, gauge/G-structure, E(n) actions, curvature, geodesics.>

## Method / architecture
<what they actually built: network layers, training objective, how geometry is injected>

## Code
<what code is provided here + link to official implementation if one exists>

## Why it matters
<impact, follow-ups, what to take away>

## Reading questions / discussion
<2-4 discussion prompts for the reading group>
```

## code/ rules
- Must be self-contained and runnable (import torch and/or jax; no missing funcs).
- Include a `if __name__ == "__main__":` demo/tiny numerical check demonstrating the core op works.
- Prefer small, correct implementations of the *geometric core* (e.g. Poincaré distance+exp/log,
  Riemannian GD on SPD, group-equivariant conv, neural ODE stepping, Sinkhorn, gauge parallel
  transport) rather than a full reproduction of the paper's entire architecture.
- Keep each file under ~200 lines.

## Style
- Clear, pedagogical, math-precise but accessible.
- Use `$...$` / `$$...$$` for math.
- Do NOT include papers' full text or figures you don't have; write original summaries and code.
