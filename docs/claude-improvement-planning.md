# Plan: Critical review of `columnarCL-fabricPC-experiments/docs/migration-plan.md`

## Context

The user has a migration plan (`columnarCL-fabricPC-experiments/docs/migration-plan.md`, 422 lines, 9 phases, written by codex) that proposes moving Split-CIFAR-10 columnar-CL experiments out of `cFabricPC` into a new repo `columnarCL-fabricPC-experiments` while using upstream `FabricPC` as a dependency. They want a critical, augmented, improved version saved as `claude-crit-review-migration-plan.md` in the same directory.

I dispatched three parallel Explore agents to inventory the three repos. Findings below are ground-truth (not the plan's claims).

## Ground-truth deltas vs. the plan's assumptions

**Things the plan gets right** (no augmentation needed for these):
- The four flagged "current best" CLI args (`--prototype_memory_per_class`, `--shared_feature_sep_weight`, `--shared_local_decorr_weight`, `--memory_readout_steps`) ARE in the current `cifar10_colba_split_disjoint.py`.
- The cross-seed numbers (seed 42 = 0.6569, seed 7 = 0.6662, seed 99 = 0.6460) ARE in `summary.jsonl`.
- `fabricpc.continual` does NOT exist in upstream FabricPC — the disjoint plan to keep CL code experiment-local is correct.
- The boundary policy "keep CL code in experiment repo unless clearly general-purpose" is sound.
- Phase 8 patch discipline is sound.

**Things the plan misses, in order of severity:**

1. **CRITICAL — namespace mismatch beyond `fabricpc.continual`.** The plan treats `fabricpc.builder` and `fabricpc.graph` as if they exist upstream. They don't.
   - cFabricPC's `fabricpc.builder.{Edge, TaskMap, graph}` does NOT exist upstream. Upstream has: `fabricpc.graph_assembly.{TaskMap, graph}` and `fabricpc.core.topology.Edge`.
   - cFabricPC's `fabricpc.graph.{initialize_params, state_initializer.initialize_graph_state}` does NOT exist upstream. Upstream has `fabricpc.graph_initialization.{initialize_params, state_initializer.initialize_graph_state}`.
   - The plan's Phase 1 "verify these exports exist" → they DON'T. So Phase 1's deliverable is wrong; the right deliverable is an explicit import-rewrite decision: do we rename imports in the experiment, or add a thin compatibility shim `fabricpc/builder/__init__.py` + `fabricpc/graph/__init__.py` upstream re-exporting under the old names? The plan calls this "compatibility exports" without specifying which.

2. **CRITICAL — venv shadowing already in place.** `/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -c "import fabricpc"` currently resolves to `cFabricPC/fabricpc/__init__.py`, NOT upstream. Phase 2's `pip install -e .../FabricPC` will collide with the existing editable install. The smoke test `python -c "import fabricpc"` will pass for the wrong reason until cFabricPC is uninstalled. Plan must add an explicit `pip uninstall -y fabricpc` step before Phase 2 reinstall, OR recommend a fresh venv.

3. **HIGH — Phase 3 / Phase 4 ordering creates a broken-tree window.** Phase 3 moves the experiment scripts and rewrites their imports to point at `columnar_cl_fabricpc.continual.*`. Phase 4 moves the modules that satisfy those imports. Between these phases the repo doesn't run. Fix: combine Phase 3 + Phase 4 into a single atomic migration, or do them in the reverse order.

4. **HIGH — only 4 of ~14 experiment scripts are listed for migration.** cFabricPC/examples/ contains many more experiment scripts (`single_head_ladder_sh0.py`, `_sh6_bias.py`, `_task_probe.py`, `_teacher_distill.py`, `prototype_geometry_report.py` at 1618 lines, plus `_lwf`, `_microcols`, `_psi`, `_split.py` baseline, `aggregate_multi_seed.py`, `cifar10_disjoint_calibrate.py`, `cifar10_disjoint_joint_ceiling.py`). The plan needs to declare migration scope: (a) all of them, (b) only the "current best" set, or (c) staged migration with deprecation note. Right now they're invisible in the plan.

5. **HIGH — sweep `summary.csv` doesn't exist; only `summary.jsonl` does.** Phase 5 lists `summary.csv` as a thing to move. It will fail. Either generate the CSV from JSONL during migration, or drop the CSV reference.

6. **MEDIUM — cFabricPC's OTHER fork-only additions are unclassified.** Beyond `fabricpc/continual/`, cFabricPC adds `fabricpc/builder/`, `fabricpc/graph/`, `fabricpc/nodes/conv.py`, `fabricpc/utils/data/cifar_raw.py`. The plan addresses none of these. Each needs a verdict: upstream-able (open PR to FabricPC), experiment-local (move to `columnar_cl_fabricpc/*`), or vestigial (delete).

7. **MEDIUM — source branch unspecified.** cFabricPC's active branch is `cifar10`, not `main`. The recent commits ("beginning again with codex, from the ground up", "fixing bad code from prior that i asked to have purged by codex") indicate substantial rewrites. The plan should pin which branch + commit is being migrated FROM, so the migration is deterministic.

8. **MEDIUM — `trainer.py` (SequentialTrainer, 2149 lines) and `gradient_protection.py` are imported by the experiment but not listed in Phase 4's "modules to move".** Phase 4 lists only `config.py`, `data.py`, `data_cifar.py`, `nodes.py`. The actual dependency closure is larger.

9. **MEDIUM — acceptance criterion "close to old baseline" is undefined.** What's "close"? The plan should specify a numeric tolerance band (e.g., seed-7 final memory readout within ±0.01 of 0.6662; forgetting within ±0.02 of 0.1649). Without this, reproduction is unfalsifiable.

10. **LOW — FabricPC commit pin is not specified.** Phase 9 says "expected FabricPC commit hash" but never says which one. Upstream is at `10fbe2e` (branch `feature/columnar_CC_adapter`) — this should be baked into pyproject and into README.

11. **LOW — license attribution.** Target repo LICENSE is MIT © charlie derr 2026; FabricPC LICENSE is MIT © Matthew Behrend. Both MIT, compatible, but worth a one-line `NOTICE` or README note clarifying that re-distributed FabricPC files retain their original copyright.

12. **LOW — no linter/formatter/type-checker discipline.** Phase 9's CI is "compile + import + smoke test." No ruff/black/mypy. Plan should at minimum raise the question (defer is fine).

13. **LOW — no checkpoint policy beyond `.gitignore`.** If an experiment depends on a checkpoint to reproduce a result, what's the storage strategy? Git LFS? External URL? Document-only? Phase 5 only says "gitignore large binary artifacts."

## What I'll write into `claude-crit-review-migration-plan.md`

The output file will be structured as an **augmented review**, NOT a from-scratch replacement. It will explicitly reference the original plan section-by-section and point at the deltas. Sections:

1. **How to read this document** — relationship to the original plan; what's kept, what's amended, what's added.
2. **Ground-truth snapshot** — what the three repos actually contain right now, including the namespace facts that the original plan got wrong.
3. **Issues found, ranked by severity** — the 13 items above, restated with concrete file paths and remediation.
4. **Augmented migration phases** — Phase-by-phase amendments to the original. Where the original is right, say so and move on. Where it's wrong, give the corrected version. New Phase 0 (clean slate / pin source branch) and an atomic Phase 3+4 merge.
5. **Concrete import-rewrite table** — every `fabricpc.*` import in `cifar10_colba_split_disjoint.py`, mapped to (a) the upstream FabricPC path it resolves to today, (b) the FINAL import path after migration, (c) whether this requires an upstream shim PR. This is the table the original plan should have had.
6. **FabricPC patch payload (concrete)** — the actual file additions for the `experiment-support/columnarcl` branch (the two thin `__init__.py` shims, optional viz lazy-import, packaging metadata if any), not just policy.
7. **Reproducibility threshold table** — pass/fail bands for seed-7, seed-42, seed-99 reproduction.
8. **Risk register** — venv shadowing, namespace shims vs. import rewrites, branch drift on cFabricPC's `cifar10` branch, sweep harness hardcoded paths.
9. **Updated acceptance criteria** — replacing the under-specified "close to baseline" with numeric tolerances; adding "all experiment scripts compile and import cleanly" (not just the four canonical ones).
10. **Deferred / out-of-scope** — explicit list of things the augmented plan is NOT doing yet (lint/format choice, checkpoint storage policy, CI provider choice).
11. **Suggested immediate next steps** — replacing the original's 9-step list with a 6-step list that resolves the critical-severity items first (uninstall cFabricPC from venv → pin source branch → decide shim-vs-rewrite → write pyproject → atomic move → smoke).

Length target: ~500-700 lines of markdown. Tone: direct, factual, no padding. Cites concrete paths.

## Files

- READ-ONLY:
  - `/home/ni/repos/fpc/columnarCL-fabricPC-experiments/docs/migration-plan.md` (already read)
  - `/home/ni/repos/fpc/FabricPC/**` (inventoried via Explore)
  - `/home/ni/repos/fpc/cFabricPC/**` (inventoried via Explore)
- TO CREATE:
  - `/home/ni/repos/fpc/columnarCL-fabricPC-experiments/docs/claude-crit-review-migration-plan.md`

No edits to any other file. No commits. No package installs. No FabricPC PRs.

## Verification

After writing the file:

1. `wc -l` the new file to confirm it's in the 500–700 line range.
2. `grep -c '^## '` to confirm section count matches the outline.
3. Read back the file once and check the import-rewrite table is internally consistent (no row left blank, all paths resolve to inventory-verified locations).

No execution, no scripts, no env changes — this is a documentation deliverable.
