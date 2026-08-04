# Handoff — ML × Differential Geometry Reading Group

## Current state

Working tree is clean. `master` is up to date with `origin/master` at commit `0029f04` ("Author 6 missing papers, deepen tutorial, fix SPD retraction bug"). All work requested this session (author the 6 missing papers, deepen the tutorial, run a flavor-neo/flavor-forge audit with independent verification, commit, push, write a handoff) is done — there is nothing in flight.

For full detail on what changed and why (the 6 new papers, the tutorial depth pass, the SPD retraction bug that was found and fixed, the flavor-neo/flavor-forge audit results, and credentials/workflow notes), read **`docs/handoff-2026-08-04T15-34-12-claude-code.md`** — it is authoritative for this repo's current content state and is not repeated here. Commit `0029f04` (`git show --stat 0029f04`) has the full diff.

## Recommended next session

Carried forward unchanged from `docs/handoff-2026-08-04T15-34-12-claude-code.md`'s "Recommended next session" section:
- Add CI that runs every `papers/*/*/code/*.py` and `tutorial/code/riemannian_pytorch.py` and checks for exit 0 — there is still no automated test/CI configuration in this repo.
- If a genuinely cross-vendor verifier is wanted for a future flavor-neo/flavor-forge pass, install the Codex CLI first (`npm install -g @openai/codex`, then `/codex:setup`) — it was unavailable this session.
- `tutorial/code/riemannian_jax.py` is still unexercised (no `jax` installed in this environment, and no new packages were installed this session per explicit user instruction).

## Suggested skills for the next session

- `flavor-neo` / flavor-forge (already invoked this session; useful again if another audit-and-repair pass is needed, e.g. once CI exists to add as a fourth worker lane).
- `handoff` (this skill) at the end of the next session.
