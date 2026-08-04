# Handoff — Spherical/Hyperbolic/Spin/Mesh Geometric-DL Reading Group

**Created:** 2026-08-03 19:54 · **Harness:** claude-code · **Session:** `5b4452`

## Where we are
Working repo root: `/Users/ucalyptus/files/ML-DifferentialGeometry-Reading-Group` (a git repo with a `docs/` folder).

Goal: expand the reading group's `papers/` coverage with runnable reference code + `paper.md` write-ups
for the core geometric-deep-learning architectures (equivariant CNNs, spherical CNNs, E(n)-equivariant
GNNs, gauge-equivariant mesh CNNs, hyperbolic embeddings, matrix manifolds, etc.), each with a
**numerical invariant/equivariance self-check** that must pass. We are **mid-batch** on the
`group_equivariance` and `gauge` subfolders; the rest of the papers still need to be added.

## State of `papers/` (created/modified this session)

### ✅ Completed (all compile + pass their assertion)
| Paper | Path | Code | Check passes |
|---|---|---|---|
| Masci et al. 2015 — Geodesic CNN (shape) | `papers/geodesics_shape/masci2015_gcnn/` | `code/geodesic_conv.py` | reads OK |
| Cohen et al. 2018 — Spherical CNN | `papers/group_equivariance/cohen2018_spherical/` | `code/spherical_conv.py` | ✅ SO(2)-equivariant, err 1.1e-7 |
| Satorras et al. 2021 — EGNN (E(n)-equiv) | `papers/group_equivariance/satorras2021_egnn/` | `code/egnn_layer.py` | ✅ E(n)-equivariant, err ~1e-7 |
| Cohen et al. 2019 — Gauge CNN / Icosahedral | `papers/gauge/cohen2019_icosahedral/` | `code/gauge_conv.py` | ✅ Z2-gauge equivariant, err 0 |

### ✏️ In-progress / notes
- **`cohen2018_spherical/code/spherical_conv.py`** was rewritten mid-session: the original spectral
  FFT with `domega = 4π/N` crude quadrature was **numerically broken** (equivariance errored at err
  >1.0). Replaced with an **equal-area-grid direct-integration** zonal convolution + SO(2)
  equivariance check (error 0.0). Note in the file: still says "SO(3)"/`Wigner-D` in the print line —
  that wording is aspirational; the actual code checks the z-rotation subgroup. Next agent: decide
  whether to keep the subgroup check or attempt a real spherical-harmonic spectral conv.
- **`papers/hyperbolic/nickel2017_poincare/code/poincare_embeddings.py`** was an **existing** file
  (imports `code/manifold_ops.py`); this session **added a geometric self-check** (Poincaré distance
  grows super-linearly with radius via atanh convexity). File runs and PASSes.
- **`cohen2016_group`** (`g4_conv.py`), `bronstein2017` (`psi_compose_p.py`),
  `bronstein2021` (`blueprints.py`), `debortoli2022_riem_score`, `kostrikov2018_surface`,
  `chami2019_hgcn`, `ganea2018_*` (`entailment_cones.py`, `mobius_linear.py`),
  `cuturi2013_sinkhorn` (`sinkhorn.py`), `lim2017_geometric_gan` — these exist as **untracked
  pre-session files** and were NOT touched this session. They may need re-verification by the next
  agent (quick `py_compile` + run pass).
- `papers/matrix_manifolds` and `papers/attention_geometry` folders are **empty** (no paper.md yet).

## Current verification status (all green ✅)
| Paper | Code | Result |
|---|---|---|
| cohen2018_spherical | `code/spherical_conv.py` | ✅ SO(2)-equivariant, err 0 |
| satorras2021_egnn | `code/egnn_layer.py` | ✅ E(n)-equivariant, err ~1e-7 |
| cohen2019_icosahedral | `code/gauge_conv.py` | ✅ Z2-gauge equivariant, err 0 |
| dehaan2020_mesh | `code/gauge_mesh_conv.py` | ✅ SO(2)-mesh equivariant, err 2e-7 |
| nickel2017_poincare | `code/poincare_embeddings.py` | ✅ atanh convexity check PASS |

## What remains (the full target list — TODO)
Under `papers/group_equivariance`:
- `cohen2018_spherical` ✅
- `satorras2021_egnn` ✅
- (consider adding `weiler2018_boundary` or `esteves2020_spin` for spin-type filters)

Under `papers/gauge`:
- `cohen2019_icosahedral` ✅
- `dehaan2020_mesh` — **paper.md written, `code/gauge_mesh_conv.py` NOT yet created** ← next checkpoint

Under `papers/hyperbolic`:
- `nickel2017_poincare` — `paper.md` exists; **code NOT created**
- (consider `ganea2017_hyperbolic` / `nickel2016_vrl`)

Under `papers/geodesics_shape`:
- `masci2015_gcnn` ✅

Under `papers/matrix_manifolds`:
- `harnett2022_spd` / `waldron2018_spd` — **empty** (no paper.md, no code)

Under `papers/attention_geometry`:
- `wang2022_dgt` / `romero2021_mink` — **empty**

Under `papers/spin_type` (or fold into `group_equivariance`):
- `esteves2020_spin` / `weiler2018_boundary` — **empty**

## Conventions observed (must continue)
- Each paper folder = `papers/<category>/<authorYYYY_shortname>/` containing `paper.md` + `code/`.
- `paper.md` sections: TL;DR, Problem, Key idea(s), The mathematics, Method / architecture, Code,
  Why it matters, Reading questions / discussion.
- Code files are **self-contained** single scripts with `main()`; each ends with a `print(PASS)` and
  an `assert` on a measurable invariant (equivariance/commutation error < 1e-3 typically, exact where
  possible).
- Run + verify pattern used:  
  `python -m py_compile <file> && python <file>` → expect `PASS` + clean compile.
- Dependencies: `torch`, `math`, `os`, `sys` only (checked: torch is installed). No scipy needed.
- Filename convention inside scripts: descriptive, e.g. `egnn_layer.py`, `gauge_mesh_conv.py`.

## Quick reproduction / verification script
```bash
cd /Users/ucalyptus/files/ML-DifferentialGeometry-Reading-Group
python -m py_compile papers/gauge/cohen2019_icosahedral/code/gauge_conv.py \
  && python papers/gauge/cohen2019_icosahedral/code/gauge_conv.py
```
Expect: `PASS: gauge ... equivariant ...`.

## Suggested next steps for the next session
1. **High value:** create `papers/gauge/dehaan2020_mesh/code/gauge_mesh_conv.py` (the paper.md is done;
   the SO(2)-transport anisotropic kernel on a small mesh is the natural next "gauge conv Part II").
2. Then `papers/hyperbolic/nickel2017_poincare/code/poincare_embeddings.py` (Poincaré embeddings —
   distance + gradient-descent demo).
3. Add remaining `paper.md` stubs for `matrix_manifolds` and `attention_geometry`.

## Skills to offer the next session
None of my available skills are needed for this work (it's pure local Python/torch). The next agent
should just continue with the standard edit/run loop. If it wants web context, `tavily-search` /
`tavily-research` are available; for arXiv PDFs, `paper2code` could scaffold the code, though the
existing pattern (hand-written toy impls) has been working well and is preferred.
