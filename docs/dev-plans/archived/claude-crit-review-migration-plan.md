# Critical Review and Augmentation of `migration-plan.md`

**Reviewer:** Claude (Opus 4.7)
**Reviewed file:** `columnarCL-fabricPC-experiments/docs/migration-plan.md`
**Companion file:** this document; lives alongside the original.

---

## How to read this document

This is an **augmented review**, not a replacement. The original `migration-plan.md` is largely sound at the level of intent and overall phase structure — keep it. What this document does:

- **Section 2 — Ground truth.** What the three repos actually contain right now, verified by direct inventory. Some of the original plan's import assumptions are factually wrong; you need this section before trusting any phase that touches imports.
- **Section 3 — Issues found, ranked by severity.** Thirteen concrete deltas between plan and reality, with remediation for each.
- **Sections 4–8 — Augmented material.** Phase-by-phase amendments, a concrete import-rewrite table, the actual FabricPC patch payload, a reproducibility threshold table, a risk register.
- **Section 9 — Updated acceptance criteria.**
- **Sections 10–11 — Deferred items and a 6-step immediate-next-steps list** that supersedes the original plan's 9-step list (which under-orders the critical-severity work).

Where the original is correct, this document says so and moves on. Where it's wrong or under-specified, this document corrects it concretely (with file paths and code).

---

## 2. Ground truth: what the three repos actually contain

This subsection is the load-bearing fact base. Subsequent sections cite it.

### 2.1 Upstream `FabricPC` at `/home/ni/repos/fpc/FabricPC`

- Branch: `feature/columnar_CC_adapter`, commit `10fbe2e`. Public remote `git@github.com:trueagi-io/FabricPC.git`. MIT © Matthew Behrend. v0.3.1 (alpha).
- Package layout — the names that matter for this migration:

  | name | exists upstream? | path |
  |---|---|---|
  | `fabricpc.continual` | **NO** | — |
  | `fabricpc.builder` | **NO** | — |
  | `fabricpc.graph` (as a package) | **NO** | — |
  | `fabricpc.graph_assembly.{graph, TaskMap}` | yes | `fabricpc/graph_assembly/graph_construction.py` |
  | `fabricpc.graph_initialization.initialize_params` | yes | `fabricpc/graph_initialization/params_initializer.py` |
  | `fabricpc.graph_initialization.state_initializer.initialize_graph_state` | yes | `fabricpc/graph_initialization/state_initializer.py` |
  | `fabricpc.core.topology.Edge` | yes | `fabricpc/core/topology.py` |
  | `fabricpc.core.activations.{ReLUActivation, SoftmaxActivation}` | yes | `fabricpc/core/activations.py` |
  | `fabricpc.core.energy.{CrossEntropyEnergy, GaussianEnergy}` | yes | `fabricpc/core/energy.py` |
  | `fabricpc.core.inference.InferenceSGDNormClip` | yes | `fabricpc/core/inference.py` |
  | `fabricpc.core.initializers.{NormalInitializer, XavierInitializer}` | yes | `fabricpc/core/initializers.py` |
  | `fabricpc.core.types.{GraphParams, NodeParams}` | yes | `fabricpc/core/types.py` |
  | `fabricpc.nodes.{IdentityNode, Linear}` | yes | `fabricpc/nodes/__init__.py` |
  | `fabricpc.utils.helpers.set_jax_flags_before_importing_jax` | **NO** | (upstream has the helper only at repo root: `FabricPC/jax_setup.py`) |
- `pyproject.toml`: name `fabricpc`, version `0.3.1`, Python ≥ 3.10. Hard deps: `jax`, `jaxlib`, `optax`, `orbax-checkpoint`, `flax`, `chex`, `jaxtyping`, `numpy`, `tqdm`, `optuna`. `aim` is gated to the `[viz]` extra (already lazy). Extras: `dev`, `tfds`, `experiments`, `viz`, `cpu`, `cuda12`, `cuda13`, `all`. CUDA is not hard-coded.

### 2.2 Fork `cFabricPC` at `/home/ni/repos/fpc/cFabricPC`

- Branch: `cifar10` (not `main`). Recent commits show heavy codex-driven rewrites ("beginning again with codex, from the ground up", "fixing bad code from prior that i asked to have purged by codex"). Working tree clean.
- Fork-only additions under `fabricpc/`:
  - `fabricpc/continual/` — 19 modules, ~14 kLOC, the CL machinery.
  - `fabricpc/builder/` — namespace wrapper around `core.topology.Edge` + `graph_assembly.{graph, TaskMap}`; this is why the disjoint script can write `from fabricpc.builder import Edge, TaskMap, graph`.
  - `fabricpc/graph/` — namespace wrapper around `graph_initialization.*`; this is why the disjoint script can write `from fabricpc.graph import initialize_params` and `from fabricpc.graph.state_initializer import initialize_graph_state`.
  - `fabricpc/nodes/conv.py` — custom convolutional node not in upstream.
  - `fabricpc/utils/data/cifar_raw.py` — CIFAR raw-loader.
  - `fabricpc/utils/helpers.py` — has the in-package `set_jax_flags_before_importing_jax` (upstream's helper lives at the repo root, not in the package).
- `examples/`: the four scripts named in the plan are present, but there are ~14 experiment-scoped Python files in total. Roll-call below.
- The "current best" CLI flags (`--prototype_memory_per_class`, `--shared_feature_sep_weight`, `--shared_local_decorr_weight`, `--memory_readout_steps`) ARE defined on the current disjoint script.
- `results/sweeps/cifar10_colba_disjoint/summary.jsonl` exists and contains the three cited seed numbers (42 = 0.6569, 7 = 0.6662, 99 = 0.6460). `summary.csv` does **not** exist.
- `results/` total disk: 1.2 MB. No large checkpoint payloads currently in tree.

### 2.3 Target repo `columnarCL-fabricPC-experiments` at `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`

- Branch `main`, single commit `f36aeec` "Initial commit". Remote `github.com:charliederr/columnarCL-fabricPC-experiments`.
- Three files: `README.md` (33 bytes; just the title), `LICENSE` (MIT © charlie derr 2026), `docs/migration-plan.md`. `docs/` is untracked.
- No `pyproject.toml`, no `.gitignore`, no Python package skeleton.

### 2.4 Existing venv

- `/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -c "import fabricpc; print(fabricpc.__file__)"` resolves to `/home/ni/repos/fpc/cFabricPC/fabricpc/__init__.py`. **The venv already has an editable install pointing at the fork, not upstream.** This is the primary venv referenced throughout the original plan.

---

## 3. Issues found, ranked by severity

Numbered for reference in later sections.

### CRITICAL

**[C-1] Namespace mismatches beyond `fabricpc.continual`.** The plan's Phase 1 says "verify these exports exist" for a list that includes `fabricpc.builder.{Edge, TaskMap, graph}` and `fabricpc.graph.{initialize_params, state_initializer.initialize_graph_state}`. Per §2.1, **none of these namespace paths exist in upstream FabricPC.** They are fork-only conveniences in cFabricPC. The plan's "small compatibility exports" mention is in the right direction but does not name what's missing, so Phase 1's deliverable understates the work. Remediation: §5 import-rewrite table + §6 FabricPC patch payload.

**[C-2] `fabricpc.utils.helpers.set_jax_flags_before_importing_jax` also doesn't exist upstream** — upstream has the helper at repo root (`FabricPC/jax_setup.py`), not inside the package. Same class of issue as C-1 but the plan doesn't mention it at all. Cleanest fix: copy this small helper into the experiment repo (`columnar_cl_fabricpc/utils/helpers.py`), so the experiment doesn't depend on either FabricPC's root-level convention or a shim. Alternative: add it to upstream `fabricpc.utils.helpers` in the support PR.

**[C-3] Venv shadowing already in place.** Per §2.4 the `ffpc7` venv resolves `import fabricpc` to the cFabricPC fork. The plan's Phase 2 says `pip install -e .../FabricPC`; this will silently collide with the existing editable install (whose package name is also `fabricpc`). The Phase 6 smoke test `python -c "import fabricpc; import columnar_cl_fabricpc"` will **pass for the wrong reason** until cFabricPC is uninstalled. Remediation in Phase 0 (new) below.

### HIGH

**[H-1] Phase 3 / Phase 4 ordering creates a broken-tree window.** Phase 3 moves the experiment scripts and rewrites their imports to `columnar_cl_fabricpc.continual.*`. Phase 4 moves the modules that those imports require. Between the phases the repo cannot import its own scripts. Fix: merge into a single atomic Phase 3+4 ("move scripts and their dependency closure together"), or reverse the order (move modules first, then scripts).

**[H-2] Only ~4 of ~14 experiment scripts are listed for migration.** The plan names `cifar10_colba_split_disjoint.py`, `sweep_cifar10_colba_disjoint.py`, `cifar10_colba_joint.py`, `single_head_ladder_metrics.py`. Inventory shows the experiment fork also contains: `cifar10_colba_split.py`, `cifar10_colba_split_lwf.py`, `cifar10_colba_split_microcols.py`, `cifar10_colba_split_psi.py`, `single_head_ladder_sh0.py`, `single_head_ladder_sh6_bias.py`, `single_head_ladder_task_probe.py`, `single_head_ladder_teacher_distill.py`, `prototype_geometry_report.py` (1618 lines), `aggregate_multi_seed.py`, `cifar10_disjoint_calibrate.py`, `cifar10_disjoint_joint_ceiling.py`. The plan needs to declare migration scope explicitly: (a) all of them, (b) only the "current best" four, (c) staged. Either is fine; silence is not.

**[H-3] Sweep `summary.csv` doesn't exist; only `summary.jsonl` does.** Phase 5 lists both as artifacts to move. Either generate the CSV from JSONL during migration (~5 lines of pandas) or drop the CSV reference. If pinning CSV as the canonical analyst-facing summary, also commit the generator script that produced it.

### MEDIUM

**[M-1] Other cFabricPC fork-only additions are unclassified.** Beyond `fabricpc/continual/`, the fork adds `fabricpc/builder/`, `fabricpc/graph/`, `fabricpc/nodes/conv.py`, `fabricpc/utils/data/cifar_raw.py`, `fabricpc/utils/helpers.py`. The plan addresses none of these. Each needs an explicit verdict — upstream PR, experiment-local move, or drop. Suggested mapping in §6.

**[M-2] Source branch unspecified.** cFabricPC is on `cifar10`, not `main`. Given the codex rewrites visible in recent commit messages, the migration should pin the exact commit being migrated FROM. Otherwise reruns of the migration may pick up later rewrites with different semantics. Add to Phase 0.

**[M-3] Phase 4's module list is too short.** Phase 4 lists `config.py`, `data.py`, `data_cifar.py`, `nodes.py` for the move. The actual dependency closure observed in the disjoint script also touches `trainer.py` (SequentialTrainer, 2149 lines) and `gradient_protection.py` (601 lines, used implicitly via `make_config`). The Phase 4 deliverable should be derived from `python -m modulegraph` or a `grep`-based closure, not a hand-list. See §5 for the actual closure.

**[M-4] Acceptance criterion "close to old baseline" is undefined.** The plan asks for "a result close to the current cFabricPC seed-7 baseline." Without numeric tolerance bands this is unfalsifiable. §7 supplies thresholds.

### LOW

**[L-1] FabricPC commit pin not specified.** Phase 9 says "expected FabricPC commit hash" but never picks one. Per §2.1 upstream is at `10fbe2e` on `feature/columnar_CC_adapter`. Bake this into `pyproject.toml`'s URL pin and into `README.md`.

**[L-2] License attribution not addressed.** Target LICENSE is MIT © charlie derr 2026. Re-distributed FabricPC files would retain MIT © Matthew Behrend. Adding a one-line NOTICE or README paragraph clarifying this avoids future confusion when FabricPC modules are vendored (if any).

**[L-3] No lint/format/type-check policy.** Phase 9 CI is "compile + import + smoke." No `ruff`, `black`, or `mypy`. Plan should at minimum raise the question. Recommendation: defer, but add a `pyproject.toml` `[tool.ruff]` section as a hook for later.

**[L-4] No checkpoint policy beyond `.gitignore`.** If a result requires a saved checkpoint (`.npz` or otherwise) to reproduce, what's the storage model? Phase 5 says "gitignore large binary artifacts" but doesn't address reproducibility. Recommendation in §10: document-only initially (cite checkpoint size + how to regenerate); Git LFS or external storage deferred.

---

## 4. Augmented migration phases

Phase-by-phase amendments to the original. Where the original is fine, this section says so and moves on.

### Phase 0 (NEW) — Pin sources and clear the venv

Resolves [C-3] and [M-2]. **Must precede Phase 1.**

```bash
# 0.1 Pin the source commit being migrated FROM.
cd /home/ni/repos/fpc/cFabricPC
git rev-parse HEAD                   # record this; it goes into the README of the new repo

# 0.2 Pin the upstream FabricPC commit being migrated TO.
cd /home/ni/repos/fpc/FabricPC
git rev-parse HEAD                   # expected: 10fbe2e (record this)

# 0.3 Remove the conflicting editable install from the venv.
/home/ni/repos/fpc/virt-envs/ffpc7/bin/pip uninstall -y fabricpc
/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -c "import fabricpc" 2>&1   # must now fail

# 0.4 Decide: shim path or rewrite path. (See §5 for the choice tree.)
```

**Deliverable:** A `MIGRATION_SOURCE.md` (or section in the migration PR) containing the cFabricPC commit being migrated and the upstream FabricPC commit being targeted.

### Phase 1 — Inventory + dependency map (AMENDED)

Original plan calls for listing imports and classifying them. The deliverable is fine; replace the original's example bullet list with **§5's complete import-rewrite table**, which already does the classification.

Critical correction from [C-1] [C-2]: do not assume `fabricpc.{builder, graph}` and `fabricpc.utils.helpers.set_jax_flags_before_importing_jax` exist upstream — they don't.

### Phase 2 — Package the experiment repo (LIGHTLY AMENDED)

Original `pyproject.toml` skeleton is fine, but pin FabricPC to a specific commit per [L-1]:

```toml
[project]
name = "columnarcl-fabricpc-experiments"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fabricpc @ git+https://github.com/trueagi-io/FabricPC.git@10fbe2e",
  "jax",
  "jaxlib",
  "optax>=0.1.7",
  "numpy>=1.24.0",
  "pandas>=2.0.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["columnar_cl_fabricpc*"]

# Hook for future linter adoption (see L-3); empty for now.
# [tool.ruff]
# line-length = 100
```

For local development, the install order matters because of [C-3]:

```bash
# Order matters: fresh install of upstream, then experiment repo.
/home/ni/repos/fpc/virt-envs/ffpc7/bin/pip install -e /home/ni/repos/fpc/FabricPC
/home/ni/repos/fpc/virt-envs/ffpc7/bin/pip install -e /home/ni/repos/fpc/columnarCL-fabricPC-experiments

# Verify which fabricpc was loaded:
/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -c \
  "import fabricpc; print(fabricpc.__file__)"   # must print a path inside FabricPC, not cFabricPC
```

Add `.gitignore`:

```
__pycache__/
*.egg-info/
build/
dist/
results/**/checkpoints/
results/**/*.npz
*.log
.venv/
```

### Phase 3 + Phase 4 — Atomic merge (FIX FOR [H-1])

**Do these together, not sequentially.** A single migration PR (or branch) should:

1. Move modules from `cFabricPC/fabricpc/continual/` → `columnarCL-fabricPC-experiments/columnar_cl_fabricpc/continual/`. Use the closure in §5, not the original plan's 4-module list (per [M-3]).
2. Move the experiment scripts from `cFabricPC/examples/` → `columnarCL-fabricPC-experiments/examples/`. See [H-2] for the scope decision the user needs to make.
3. Move `cifar_raw.py` and (a small) `helpers.py` (with just `set_jax_flags_before_importing_jax`) into `columnar_cl_fabricpc/utils/`. Per [C-2] and [M-1].
4. Rewrite every import per the §5 table in the same PR.
5. Run `python -m py_compile examples/*.py` and `python -c "import columnar_cl_fabricpc"` to verify the tree is consistent before opening the PR.

The merged phase ends in a runnable tree, not a broken-tree window.

### Phase 5 — Results + docs migration (AMENDED)

- Move `docs/plans/codex.md` and `docs/plans/claude_strategy.md` from cFabricPC. The original plan only names `codex.md`; `claude_strategy.md` is also present and is the parallel strategy log — keep both, or pick one explicitly.
- Move `summary.jsonl`. **Do not list `summary.csv` (it doesn't exist).** If you want a CSV, add a tiny `scripts/jsonl_to_csv.py` generator and commit both the script and the generated CSV. Per [H-3].
- Move selected representative `.log` files for the seed 42/7/99 best runs only. Bulk-moving every sweep log is anti-pattern.
- Add `results/README.md` describing which summaries are canonical (the plan already recommends this — confirm).

### Phase 6 — Smoke test (FIX FOR [C-3])

After Phase 0's uninstall step, the smoke test sequence is meaningful. Add **two checks** the original plan misses:

```bash
# 1. Confirm fabricpc resolves to upstream, not the fork (catches C-3 regression).
/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -c \
  "import fabricpc, os; \
   assert 'FabricPC' in os.path.realpath(fabricpc.__file__), fabricpc.__file__"

# 2. Compile every migrated script, not just the canonical four (catches H-2 fallout).
find examples scripts columnar_cl_fabricpc -name "*.py" -print0 | \
  xargs -0 /home/ni/repos/fpc/virt-envs/ffpc7/bin/python -m py_compile

# 3. Tiny smoke run (original plan's command; tolerant of missing CIFAR data).
/home/ni/repos/fpc/virt-envs/ffpc7/bin/python examples/cifar10_colba_split_disjoint.py \
  --epochs_per_task 1 \
  --prototype_memory_per_class 20 \
  --memory_readout_steps 5
```

### Phase 7 — Reproduce seed 7 first (AMENDED)

Original plan's order (seed 7 first, then 42 and 99) is correct. Add a numeric pass/fail threshold per [M-4] — see §7.

### Phase 8 — FabricPC patch discipline (KEEP)

Original is sound. Add per [M-1]: the patch payload itself is concrete in §6 below — that's the file content that should land on the `experiment-support/columnarcl` branch.

### Phase 9 — CI and reproducibility (LIGHTLY AMENDED)

- Add the commit hash from Phase 0 to the README per [L-1].
- Add the license attribution note per [L-2].
- Keep the "compile + import + smoke" CI floor.
- Defer lint/format/type-check per [L-3].

---

## 5. Import-rewrite table for `cifar10_colba_split_disjoint.py`

This is the table the original plan should have had. Every `fabricpc.*` or local import in the disjoint script, with:

- **Path now**: what the current cFabricPC script writes.
- **Upstream resolves to**: where this name lives in upstream FabricPC (or "N/A" if it doesn't).
- **Final path after migration**: what the experiment script should write after migration.
- **Action**: rewrite-only, vs. requires-FabricPC-shim, vs. move-to-experiment-repo.

| path now (cFabricPC) | upstream resolves to | final path after migration | action |
|---|---|---|---|
| `from fabricpc.utils.helpers import set_jax_flags_before_importing_jax` | **N/A** (upstream has it at repo root `jax_setup.py`, not in package) | `from columnar_cl_fabricpc.utils.helpers import set_jax_flags_before_importing_jax` | **move helper to experiment repo** |
| `from fabricpc.builder import Edge, TaskMap, graph` | **N/A** as a single import. Upstream: `core.topology.Edge`, `graph_assembly.{graph, TaskMap}` | **Option A:** rewrite to upstream paths; **Option B:** add upstream shim (see §6) and keep this line | **decide shim vs. rewrite** |
| `from fabricpc.core.activations import ReLUActivation, SoftmaxActivation` | `fabricpc.core.activations.{...}` ✓ | unchanged | none |
| `from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy` | `fabricpc.core.energy.{...}` ✓ | unchanged | none |
| `from fabricpc.core.inference import InferenceSGDNormClip` | `fabricpc.core.inference.{...}` ✓ | unchanged | none |
| `from fabricpc.core.initializers import NormalInitializer, XavierInitializer` | `fabricpc.core.initializers.{...}` ✓ | unchanged | none |
| `from fabricpc.core.types import GraphParams, NodeParams` | `fabricpc.core.types.{...}` ✓ | unchanged | none |
| `from fabricpc.graph import initialize_params` | **N/A**. Upstream: `fabricpc.graph_initialization.initialize_params` | **Option A:** rewrite; **Option B:** shim | **decide shim vs. rewrite** |
| `from fabricpc.graph.state_initializer import initialize_graph_state` | **N/A**. Upstream: `fabricpc.graph_initialization.state_initializer.initialize_graph_state` | **Option A:** rewrite; **Option B:** shim | **decide shim vs. rewrite** |
| `from fabricpc.nodes import IdentityNode, Linear` | `fabricpc.nodes.{IdentityNode, Linear}` ✓ | unchanged | none |
| `from fabricpc.continual.config import make_config` | **N/A** | `from columnar_cl_fabricpc.continual.config import make_config` | move module |
| `from fabricpc.continual.data_cifar import build_split_cifar10_loaders` | **N/A** | `from columnar_cl_fabricpc.continual.data_cifar import build_split_cifar10_loaders` | move module |
| `from fabricpc.continual.nodes import (...)` | **N/A** | `from columnar_cl_fabricpc.continual.nodes import (...)` | move module |
| `from cifar10_colba_joint import AugmentedCifar10Loader` | **N/A** (local helper) | `from columnar_cl_fabricpc.experiments.cifar10_colba_joint import AugmentedCifar10Loader` | move helper |
| `from single_head_ladder_metrics import (...)` | **N/A** (local helper) | `from columnar_cl_fabricpc.diagnostics.single_head_ladder_metrics import (...)` | move helper |

### The shim-vs-rewrite decision

There are three lines (`fabricpc.builder`, `fabricpc.graph`, `fabricpc.graph.state_initializer`) for which two valid options exist:

- **Option A — rewrite-only.** Change three lines in the experiment script. Pro: no FabricPC PR needed; experiment repo is self-contained against upstream. Con: leaves the cFabricPC convention behind; if other people had also written code against `fabricpc.builder`, they'd have to follow.
- **Option B — upstream shim.** Add two tiny `__init__.py` files to upstream FabricPC re-exporting the names. Pro: cFabricPC convention survives for anyone else who relied on it. Con: requires opening a PR against upstream and depending on its acceptance; the experiment repo is blocked until then or has to ship with an editable fork.

**Recommendation: Option A (rewrite-only).** The number of touch-points is small (3 lines, plus the helper for [C-2] = 4 lines total). Rewriting them is cheaper than maintaining a separate FabricPC branch. The shim PR can still be opened separately if useful, but does not block the migration.

The remaining lines (continual, local helpers) move regardless.

### Module dependency closure for `columnar_cl_fabricpc/continual/`

Modules the disjoint script transitively requires (per [M-3]):

- `config.py` (805 LOC) — provides `make_config`
- `data.py` (366 LOC) — used by `data_cifar.py`
- `data_cifar.py` (262 LOC) — provides `build_split_cifar10_loaders`
- `nodes.py` (1807 LOC) — provides `ColumnNode` and the ColBa nodes
- `trainer.py` (2149 LOC) — provides `SequentialTrainer` (called from the script)
- `gradient_protection.py` (601 LOC) — used implicitly via `make_config`
- `__init__.py` (204 LOC) — package wiring; review for stale re-exports before copying

**Defer for later (not needed by the disjoint script):** `selector.py`, `optimal_transport.py`, `parity.py`, `microcolumn.py`, `ewc.py`, `augmentation.py`, `utils.py`, `native_nodes.py`, `weight_causal.py`, `transweave.py`, `support.py`, `causal.py`. These are needed only by the other experiment scripts (LwF, microcols, Ψ-selector, etc.) per [H-2]'s scope decision.

---

## 6. FabricPC patch payload (for `experiment-support/columnarcl`)

Only relevant if §5's Option B (shim) is chosen. Per the recommendation, this section is **optional reference**, not required path.

### 6.1 If shipping the shims

Two files, ~5 lines each.

**`fabricpc/builder/__init__.py`:**

```python
"""Convenience re-exports kept for backwards compatibility with prior
columnar-CL experiment code that imported under `fabricpc.builder`.
"""
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import graph, TaskMap

__all__ = ["Edge", "graph", "TaskMap"]
```

**`fabricpc/graph/__init__.py`:**

```python
"""Convenience re-exports kept for backwards compatibility with prior
columnar-CL experiment code that imported under `fabricpc.graph`.
"""
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization import state_initializer

__all__ = ["initialize_params", "state_initializer"]
```

Plus a test that imports them so the PR is verified by CI.

### 6.2 Other fork-only additions — verdicts per [M-1]

| fork-only addition | recommended verdict | rationale |
|---|---|---|
| `fabricpc/builder/` | shim per 6.1 above, OR drop after §5 Option A rewrite | small, optional |
| `fabricpc/graph/` | shim per 6.1 above, OR drop after §5 Option A rewrite | small, optional |
| `fabricpc/nodes/conv.py` | move to `columnar_cl_fabricpc/nodes/conv.py` | not used by upstream demos; experiment-specific |
| `fabricpc/utils/data/cifar_raw.py` | move to `columnar_cl_fabricpc/data/cifar_raw.py` | CIFAR-specific |
| `fabricpc/utils/helpers.py::set_jax_flags_before_importing_jax` | move to `columnar_cl_fabricpc/utils/helpers.py` | per [C-2] |
| `fabricpc/continual/*` | move to `columnar_cl_fabricpc/continual/` | per §5 closure |

**Do not open an upstream PR for the conv node or CIFAR loader.** They're experiment-shaped and would dilute the parent library's scope.

### 6.3 Upstream `aim` laziness

The plan's Phase 8 calls for "optional dashboard imports" laziness in FabricPC. Per §2.1 this is **already done** — `aim` is in the `[viz]` extra and gated behind `python_version`/platform guards in upstream `pyproject.toml`. No upstream PR needed for this item. Strike it from the patch payload.

---

## 7. Reproducibility threshold table (fixes [M-4])

Replaces the original plan's underspecified "close to baseline." Acceptance for Phase 7 reproduction:

| seed | baseline metric | baseline value | accept if within | reject if more than |
|---|---|---|---|---|
| 7 | `final_memory_readout` | 0.6662 | ±0.010 | ±0.020 |
| 7 | `forgetting` | 0.1649 | ±0.020 | ±0.040 |
| 42 | `final_memory_readout` | 0.6569 | ±0.010 | ±0.020 |
| 42 | `forgetting` | 0.1732 | ±0.020 | ±0.040 |
| 99 | `final_memory_readout` | 0.6460 | ±0.010 | ±0.020 |
| 99 | `forgetting` | 0.1824 | ±0.020 | ±0.040 |

**Rationale:** Cross-seed `final_memory_readout` std on the original data is ~0.010; using ±0.010 = 1σ as "accept" and ±0.020 = 2σ as "reject." Mid-band is a "look at it" zone.

**Process:**

1. Reproduce seed 7 first. If accept, proceed; if reject, halt and diagnose (likely culprits in §8).
2. Reproduce seed 42 and seed 99 in parallel. Both must accept for full migration sign-off.
3. Record the three reproduced numbers + the upstream FabricPC commit hash in `results/sweeps/cifar10_colba_disjoint/REPRODUCTION.md`.

---

## 8. Risk register

Risks that the plan should know about but doesn't yet name.

| risk | severity | likelihood | mitigation |
|---|---|---|---|
| Venv shadowing — `import fabricpc` resolves to cFabricPC even after migration | C | high | Phase 0 step 0.3 (uninstall); Phase 6 step 1 (assert path) |
| Namespace drift from upstream — FabricPC moves between migration and PR landing | M | medium | Pin commit `10fbe2e` in pyproject (per [L-1]); track upstream weekly |
| cFabricPC `cifar10` branch keeps rewriting under the migration's feet | M | high | Phase 0 step 0.1 (pin source commit); migrate FROM that commit, not HEAD |
| Sweep harness has hardcoded `cFabricPC/...` paths | M | low-medium | grep `sweep_cifar10_colba_disjoint.py` for `cFabricPC` and absolute paths during Phase 3+4 atomic move; replace with project-relative paths |
| Dependency closure (§5) is incomplete; new ImportError surfaces during smoke | L | medium | Phase 6 step 2 (compile-all) catches at compile time; ImportError at runtime is a still-acceptable signal |
| Checkpoint policy regression — some result requires an `.npz` that isn't in tree | L | low | Document-only policy in `results/README.md` per [L-4]; defer LFS |
| LICENSE attribution drift — re-vendored FabricPC files lose Behrend copyright | L | low | One-line NOTICE in README per [L-2] |
| `pyproject.toml` git-URL pin becomes unreachable | L | low | Mirror FabricPC to a fork if upstream becomes unstable; not needed today |

---

## 9. Updated acceptance criteria

Supersedes the original plan's acceptance list.

Migration is complete when **all** of the following hold:

1. `columnarCL-fabricPC-experiments` installs cleanly via `pip install -e .` in a venv where `fabricpc` is NOT already editable-installed against `cFabricPC`.
2. `python -c "import fabricpc; assert 'FabricPC' in fabricpc.__file__"` succeeds — i.e. the loaded `fabricpc` is upstream, not the fork.
3. `find examples scripts columnar_cl_fabricpc -name "*.py" -print0 | xargs -0 python -m py_compile` passes for **all** migrated Python files, not just the canonical four.
4. `python -c "import columnar_cl_fabricpc"` succeeds.
5. The Phase 6 smoke run completes without error.
6. The three seed reproductions land in the "accept" band of §7's table.
7. README documents:
   - the upstream FabricPC commit pinned (per [L-1]),
   - the current best command and expected per-seed numbers,
   - a one-line LICENSE attribution note (per [L-2]),
   - the source cFabricPC commit migrated from.
8. `MIGRATION_SOURCE.md` (or PR description) records:
   - the cFabricPC commit migrated FROM (per [M-2]),
   - the §5 shim-vs-rewrite decision actually made,
   - which experiment scripts from [H-2] were in/out of scope.

---

## 10. Deferred / explicitly out of scope

Items intentionally **not** addressed in this migration. Recording them here so they don't accidentally creep in or get silently dropped.

- **Linter / formatter / type-checker choice.** Defer. Add `[tool.ruff]` placeholder per §4 Phase 2 as a hook.
- **CI provider choice and config.** Defer. Phase 9 lists the floor (compile + import + smoke); the provider (GitHub Actions vs. local Makefile) is a separate decision.
- **Checkpoint storage policy** (Git LFS vs. external URL vs. document-only). Defer. Start with document-only: `results/README.md` records which `.npz` files are needed, their sizes, and the command to regenerate.
- **Migrating the older experimental scripts** (`single_head_ladder_*`, `_lwf`, `_microcols`, `_psi`, `prototype_geometry_report.py`, etc.). User decision per [H-2]. Default recommendation: migrate `single_head_ladder_*` (current diagnostics), defer the rest.
- **FabricPC upstream PR**. Per §5 recommendation (Option A rewrite), this PR is optional. Open it later if independent users need the `fabricpc.builder` / `fabricpc.graph` convenience.
- **Documentation site / API docs.** Defer.
- **GPU/CUDA version pinning.** Upstream FabricPC handles this via `[cuda12]` / `[cuda13]` extras; the experiment repo inherits it. No additional pinning.

---

## 11. Suggested immediate next steps (replaces original plan's 9-step list)

The original plan's 9-step list ends with "reproduce seed 7 for the current best configuration" — but the first eight steps don't resolve the critical-severity issues (venv shadowing, namespace mismatch, broken-tree window). This 6-step list reorders to fix the critical items first.

1. **Pin and clear.** Run Phase 0 (record cFabricPC HEAD on `cifar10`, record FabricPC HEAD on `feature/columnar_CC_adapter`, `pip uninstall -y fabricpc` in the ffpc7 venv).
2. **Decide shim vs. rewrite.** Choose §5 Option A (rewrite, recommended) or Option B (upstream shim PR).
3. **Author `pyproject.toml`, `.gitignore`, and skeleton package directories.** Per Phase 2's amended layout, with FabricPC pinned to `10fbe2e`.
4. **Atomic Phase 3+4 move.** Copy the §5 dependency-closure modules into `columnar_cl_fabricpc/`, copy the in-scope examples into `examples/`, rewrite imports per §5 in the same step. End the step with `py_compile` + `import columnar_cl_fabricpc` passing.
5. **Reinstall and smoke.** `pip install -e FabricPC` then `pip install -e columnarCL-fabricPC-experiments`, then run the three §4 Phase 6 checks (assert-path, compile-all, tiny smoke).
6. **Reproduce seed 7.** If it lands in the §7 accept band, run seeds 42 and 99 in parallel. If reject, halt and consult §8's risk register.

---

## Appendix A — One-page summary for the migrating engineer

If you read nothing else, read this.

- Upstream FabricPC at `/home/ni/repos/fpc/FabricPC` is at commit `10fbe2e`. Treat it as immutable for this migration.
- cFabricPC fork at `/home/ni/repos/fpc/cFabricPC` is on branch `cifar10`. **Pin its current HEAD before starting.** The codex agent rewrites things; you do not want to migrate from a moving target.
- The ffpc7 venv currently has `fabricpc` editable-installed against the **fork**, not upstream. **Uninstall it before anything else.**
- Three import lines in `cifar10_colba_split_disjoint.py` use namespace names that don't exist upstream (`fabricpc.builder`, `fabricpc.graph`, `fabricpc.graph.state_initializer`). Rewrite them to upstream paths (`fabricpc.core.topology`, `fabricpc.graph_assembly`, `fabricpc.graph_initialization`). One more import (`fabricpc.utils.helpers.set_jax_flags_before_importing_jax`) needs the helper copied into the experiment repo.
- `fabricpc.continual.*` doesn't exist upstream by design; move modules into `columnar_cl_fabricpc.continual.*`.
- The dependency closure for the disjoint script is 7 modules under continual, not 4. Specifically: `config`, `data`, `data_cifar`, `nodes`, `trainer`, `gradient_protection`, `__init__`. Defer the other ~12 modules until their consumer scripts are migrated.
- Do the script move and the module move in the same step. Don't leave the tree broken between two PRs.
- Pass/fail for reproducing the old numbers: ±0.010 on final_memory_readout, ±0.020 on forgetting. Three seeds: 7 first, 42 and 99 in parallel after.
- The sweep `summary.csv` does not exist; only `summary.jsonl`. Generate the CSV if you want it.
- The original plan's nine-step closing list under-orders the work. Use §11's six-step list instead.
