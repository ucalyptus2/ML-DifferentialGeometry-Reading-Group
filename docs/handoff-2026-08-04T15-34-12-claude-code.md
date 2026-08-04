# Handoff — ML × Differential Geometry Reading Group

## Current state

The reading list is now complete: all 22 papers referenced in `README.md` have a `papers/<category>/<slug>/paper.md` and a runnable, self-checking `code/` folder. The six that were previously missing were authored this session:

- `papers/spd_riemannian/huang2017_spdnet` — SPDNet (BiMap/ReEig/LogEig layers)
- `papers/spd_riemannian/becigneul2019_radam` — Riemannian Adam vs naive Euclidean Adam on the Poincaré ball
- `papers/flows_odes/chen2018_node` — Neural ODEs (RK4-vs-matrix-exponential + discrete adjoint vs autograd)
- `papers/flows_odes/grathwohl2018_ffjord` — FFJORD (Hutchinson trace estimator convergence + CNF log-density check)
- `papers/flows_odes/falorsi2021_manifold_nf` — manifold CNFs (naive ambient Euler drifts off $S^2$; retraction-based Euler doesn't)
- `papers/flows_odes/satorras2021_enf` — E(n)-equivariant flows (multi-step EGNN Euler flow stays E(n)-equivariant end to end)

Every `code/*.py` file in the repo — old and new — was re-run this session and exits 0 with `PASS` printed. No new packages were installed; everything uses the PyTorch already present in this environment (no `jax`, no venvs).

The `tutorial/` (7 lessons) got a depth pass: every lesson now has a "Check yourself" exercise section, lesson 02 gained a curvature/holonomy section (Jacobi equation, Gauss–Bonnet), lesson 06 gained a connections/parallel-transport section, and `tutorial/README.md` gained a notation glossary reconciling symbols across all 7 lessons. `tutorial/code/riemannian_pytorch.py` gained two new sections: an SPD-retraction first-order correctness check, and a sphere-holonomy demo (parallel-transport a vector around a right-angle spherical triangle, verify it comes back rotated by exactly 90° = $K\cdot\text{area}$).

**A real bug was found and fixed along the way**: lesson 05's stated SPD retraction formula, $\operatorname{Retr}_A(V) = \exp(A^{-1}V)\,A$, does not satisfy the retraction's first-order condition $\frac{d}{dt}\operatorname{Retr}_A(tV)|_0 = V$ in general (its derivative works out to $A^{-1}VA \ne V$ unless $V$ commutes with $A$). The correct formula has $A$ on the **left**: $\operatorname{Retr}_A(V) = A\,\exp(A^{-1}V)$. Fixed in both the lesson text and the new numerical check in `riemannian_pytorch.py` (which is what caught it — the finite-difference check failed against the old formula and passed against the corrected one). Independently re-derived by an independent verifier pass; confirmed correct.

## Independent verification

Ran a flavor-neo/flavor-forge-style leader/worker/verifier audit this session. The verifier step (`codex:codex-rescue`) could not run — the Codex CLI is not installed on this machine — so verification fell back to a `general-purpose` agent on Opus (a different model than the Sonnet doing the authoring, though not a different vendor; note this for anyone wanting a stricter cross-vendor check later). It re-derived the SPD retraction fix and the sphere-parallel-transport closed form from first principles by hand, ran every new `code/*.py` file itself, byte-diffed a fresh pandoc regeneration against the committed `docs/tutorial/*.html`, and checked all 22 README paper links against the filesystem. **Verdict: PASS, no findings** — no math errors, no template non-conformance, no broken/stale links. (One housekeeping note it raised — a leftover prior-session handoff doc sitting in `docs/` — was intentionally left in place; this repo's convention is to keep old handoff docs as historical record rather than delete them, per the existing `docs/handoff-2026-08-03T19-54-claude-code-5b4452.md`.)

## Important workflow

`tutorial/*.md` remains the canonical lesson source. `docs/tutorial/*.html` is generated output committed for GitHub Pages. The exact pandoc regeneration command (and the reason lesson 06's `papers/` link must stay an absolute GitHub URL, not a repo-relative one) is now documented in **`docs/README.md`** — read it before regenerating, since it also explains a regression this session hit and fixed: `docs/tutorial/../papers` does not exist (papers/ is never published under `docs/`), so lesson 06's `.md` source itself now uses the absolute GitHub tree URL. Previously that URL was only hand-patched into the generated HTML (not the `.md` source), so a routine regeneration from `.md` silently reverted it — this is now fixed at the source, so it won't happen again.

## Worktree and credentials

`master` is one commit ahead of `origin/master` after this session's commit (see below). The only pre-existing uncommitted change (`.gitignore` adding `.omx/`) is included in this commit. A repository-owner PAT exists in `/Users/ucalyptus/files/.env` as `UCALYPTUS2_PAT`; used for the push per `AGENTS.md`, never printed.

## Recommended next session

- There is still no CI/test configuration — every `code/*.py` file's correctness currently depends on someone manually running it. A lightweight CI step that runs all `papers/*/*/code/*.py` and `tutorial/code/riemannian_pytorch.py` and checks for exit 0 would catch regressions automatically.
- If a genuinely cross-vendor verifier is wanted for a future flavor-neo/flavor-forge pass, install the Codex CLI (`npm install -g @openai/codex`, then `/codex:setup`) — it wasn't available this session, so verification used a same-vendor (Opus) fallback instead.
- `tutorial/code/riemannian_jax.py` still isn't exercised in this environment (no `jax` installed, and per this session's explicit instruction, no new packages were installed to fix that) — flagged in `tutorial/README.md` as a caveat for readers who have JAX.
