# Migration Plan: columnarCL FabricPC Experiments

## Goal

Move the Split-CIFAR-10 columnar continual-learning experiments out of the
`cFabricPC` fork and into this repository, `columnarCL-fabricPC-experiments`,
while using the original `FabricPC` repository as an independent dependency.

The desired end state is:

- `FabricPC` remains the reusable predictive-coding library.
- `columnarCL-fabricPC-experiments` contains experiment scripts, result
  analysis, sweep tools, plans, and experiment-specific continual-learning
  utilities.
- Changes to `FabricPC` are minimal, general-purpose, and upstreamable.
- The experiment repo can be run against a local editable checkout of
  `FabricPC` or a pinned Git revision.

## Current Local Repositories

- Parent library: `/home/ni/repos/fpc/FabricPC`
- Current fork with experiments: `/home/ni/repos/fpc/cFabricPC`
- New experiment repo: `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`

`FabricPC` currently packages `fabricpc` via `pyproject.toml` and contains the
core graph, node, training, utility, and dashboarding modules.

`cFabricPC` currently contains both upstream-like FabricPC code and
experiment-specific additions, including:

- `examples/cifar10_colba_split_disjoint.py`
- `examples/sweep_cifar10_colba_disjoint.py`
- experiment helper scripts such as `examples/cifar10_colba_joint.py`
- diagnostic helpers such as `examples/single_head_ladder_metrics.py`
- `docs/plans/codex.md`
- sweep outputs under `results/sweeps/cifar10_colba_disjoint/`
- continual-learning support modules under `fabricpc/continual/`

The main migration challenge is that the latest experiment imports
`fabricpc.continual.*`, but the original `FabricPC` checkout does not currently
contain the `fabricpc.continual` package.

## Design Decision

Keep experiment-specific continual-learning code in
`columnarCL-fabricPC-experiments`, not in `FabricPC`, unless a module is clearly
general-purpose library functionality.

Recommended import layout:

- Parent library imports stay as `fabricpc.*`.
- Experiment-specific code moves to a new package, for example:
  - `columnar_cl_fabricpc/continual/`
  - `columnar_cl_fabricpc/data/`
  - `columnar_cl_fabricpc/experiments/`
  - `columnar_cl_fabricpc/diagnostics/`

The migrated experiment should import:

```python
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.graph import initialize_params
from columnar_cl_fabricpc.continual.config import make_config
from columnar_cl_fabricpc.continual.data_cifar import build_split_cifar10_loaders
from columnar_cl_fabricpc.continual.nodes import ColumnNode, create_visual_stem
```

This prevents the experiment repository from pretending to be or shadowing the
`fabricpc` package.

## Minimal FabricPC Changes

Only make changes to `FabricPC` when they are required for the experiment repo
to use it as a normal dependency. Proposed minimal changes:

1. **Compatibility exports**
   Verify that imports used by the experiment are available from upstream
   FabricPC:

   - `fabricpc.builder.Edge`
   - `fabricpc.builder.TaskMap`
   - `fabricpc.builder.graph`
   - `fabricpc.graph.initialize_params`
   - `fabricpc.graph.state_initializer.initialize_graph_state`
   - `fabricpc.nodes.IdentityNode`
   - `fabricpc.nodes.Linear`

   If any are missing in original `FabricPC`, add small compatibility exports
   rather than copying large modules.

2. **CUDA/JAX setup**
   Keep FabricPC CUDA behavior neutral. Avoid hard-coding CUDA 12 or CUDA 13 in
   the library. Experiment scripts can call a local setup helper before JAX
   import if needed.

3. **Optional dashboard imports**
   If importing `fabricpc.utils` requires optional packages such as `aim`, make
   the parent library use lazy optional imports. This is general-purpose and
   appropriate for FabricPC.

4. **Packaging metadata**
   Ensure `FabricPC` remains installable with:

   ```bash
   pip install -e /home/ni/repos/fpc/FabricPC
   ```

   Avoid adding experiment-only dependencies such as CIFAR loaders, pandas
   sweep tools, or result-processing utilities to FabricPC unless they are
   optional extras.

Do not move `cFabricPC` experiment logic into `FabricPC` unless it is needed by
multiple library users independent of these columnar continual-learning
experiments.

## New Repository Structure

Target structure for `columnarCL-fabricPC-experiments`:

```text
columnarCL-fabricPC-experiments/
  README.md
  pyproject.toml
  docs/
    migration-plan.md
    plans/
      codex.md
  examples/
    cifar10_colba_split_disjoint.py
    sweep_cifar10_colba_disjoint.py
  columnar_cl_fabricpc/
    __init__.py
    continual/
      __init__.py
      config.py
      data.py
      data_cifar.py
      nodes.py
      augmentation.py
      support.py
      ...
    diagnostics/
      __init__.py
      single_head_ladder_metrics.py
    experiments/
      __init__.py
      cifar10_colba_joint.py
  results/
    sweeps/
      cifar10_colba_disjoint/
        summary.csv
        summary.jsonl
        *.log
  scripts/
    smoke_test_split_cifar10.py
```

Some `fabricpc/continual` modules in `cFabricPC` may be unused by the current
best experiment. During migration, move only the modules required to run the
current experiment first, then add additional modules only when needed.

## Migration Phases

### Phase 1: Inventory and Dependency Mapping

1. List imports used by:

   - `examples/cifar10_colba_split_disjoint.py`
   - `examples/sweep_cifar10_colba_disjoint.py`
   - `examples/cifar10_colba_joint.py`
   - `examples/single_head_ladder_metrics.py`

2. Classify each import as:

   - provided by upstream `FabricPC`
   - experiment-local helper
   - third-party dependency
   - currently only available in `cFabricPC`

3. Create an import map that defines the new module path for every
   `cFabricPC`-only dependency.

Deliverable: a short `docs/import-map.md` or a section in the migration PR.

### Phase 2: Package the Experiment Repo

Create a `pyproject.toml` for this repo. Recommended baseline:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "columnarcl-fabricpc-experiments"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fabricpc",
  "jax",
  "jaxlib",
  "optax>=0.1.7",
  "numpy>=1.24.0",
  "pandas>=2.0.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["columnar_cl_fabricpc*"]
```

For local development, install both repos editably:

```bash
pip install -e /home/ni/repos/fpc/FabricPC
pip install -e /home/ni/repos/fpc/columnarCL-fabricPC-experiments
```

If using the existing environment:

```bash
/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -m pip install -e /home/ni/repos/fpc/FabricPC
/home/ni/repos/fpc/virt-envs/ffpc7/bin/python -m pip install -e /home/ni/repos/fpc/columnarCL-fabricPC-experiments
```

### Phase 3: Move Experiment-Local Code

Copy the current experiment scripts from `cFabricPC` into the new repo:

- `examples/cifar10_colba_split_disjoint.py`
- `examples/sweep_cifar10_colba_disjoint.py`
- `examples/cifar10_colba_joint.py`
- `examples/single_head_ladder_metrics.py`

Then refactor imports:

- `from cifar10_colba_joint import ...`
  becomes `from columnar_cl_fabricpc.experiments.cifar10_colba_joint import ...`
- `from single_head_ladder_metrics import ...`
  becomes `from columnar_cl_fabricpc.diagnostics.single_head_ladder_metrics import ...`
- `from fabricpc.continual...`
  becomes `from columnar_cl_fabricpc.continual...`

Keep `examples/` thin. Larger reusable helpers should live under
`columnar_cl_fabricpc/`.

### Phase 4: Move Required Continual-Learning Modules

Start by moving only the modules imported by the current best Split-CIFAR-10
experiment:

- `fabricpc/continual/config.py`
- `fabricpc/continual/data.py`
- `fabricpc/continual/data_cifar.py`
- `fabricpc/continual/nodes.py`
- any directly imported support modules required by those files

Place them under:

```text
columnar_cl_fabricpc/continual/
```

Update internal imports so they refer either to:

- parent library APIs via `fabricpc.*`, or
- experiment-local modules via `columnar_cl_fabricpc.*`.

Avoid copying the whole `fabricpc` package into this repo.

### Phase 5: Results and Documentation Migration

Move or copy the key experiment artifacts:

- `cFabricPC/docs/plans/codex.md` to `docs/plans/codex.md`
- `cFabricPC/results/sweeps/cifar10_colba_disjoint/summary.csv`
- `cFabricPC/results/sweeps/cifar10_colba_disjoint/summary.jsonl`
- selected important logs for reproduced best runs

Recommended result policy:

- Keep compact summaries in Git.
- Keep very large logs/checkpoints out of Git unless intentionally curated.
- Add `results/**/checkpoints/` and large binary artifacts to `.gitignore`.
- Consider adding `results/README.md` describing which summaries are canonical.

### Phase 6: Smoke Test

Add a fast smoke path before attempting full CIFAR runs. Options:

1. Add a `--quick_smoke` flag to the experiment script.
2. Add a separate `scripts/smoke_test_split_cifar10.py`.
3. Run a minimal compile/import test if data is unavailable.

Required smoke checks:

```bash
python -m py_compile examples/cifar10_colba_split_disjoint.py
python -m py_compile examples/sweep_cifar10_colba_disjoint.py
python -c "import fabricpc; import columnar_cl_fabricpc"
```

Then run one short experiment path if practical:

```bash
python examples/cifar10_colba_split_disjoint.py \
  --epochs_per_task 1 \
  --prototype_memory_per_class 20 \
  --memory_readout_steps 5
```

### Phase 7: Reproduce Current Best

The current best known configuration from `cFabricPC` is:

```bash
python examples/cifar10_colba_split_disjoint.py \
  --prototype_memory_per_class 2000 \
  --shared_feature_sep_weight 0.01 \
  --shared_local_decorr_weight 0.01 \
  --memory_readout_steps 200
```

Known cross-seed memory-readout results from the old repo:

- seed 42: `0.6569`, forgetting `0.1732`
- seed 7: `0.6662`, forgetting `0.1649`
- seed 99: `0.6460`, forgetting `0.1824`
- mean: `0.6564`

After migration, rerun at least seed 7 first. Then rerun seeds 42 and 99 only
after the seed-7 result is close enough to the old baseline to rule out import
or dependency drift.

### Phase 8: FabricPC Patch Discipline

Keep a separate branch in `FabricPC` for minimal library support. Suggested
branch name:

```bash
experiment-support/columnarcl
```

Rules for changes to `FabricPC`:

- No experiment scripts.
- No CIFAR split experiment policy.
- No sweep-result logic.
- No `columnarCL` naming.
- Only compatibility exports, packaging fixes, optional import laziness, or
  general reusable graph/node improvements.

Every FabricPC change should have a small test or import check.

### Phase 9: CI and Reproducibility

Add lightweight checks to the experiment repo:

- Python compile checks for scripts and package modules.
- Import test with editable FabricPC installed.
- Optional smoke test marked as slow or data-dependent.

Add reproducibility metadata to the README:

- expected FabricPC commit hash
- expected Python version
- JAX/CUDA notes
- current best command
- current best cross-seed table

## Open Questions

1. Should `columnar_cl_fabricpc.continual` keep all historical continual modules
   from `cFabricPC`, or only the modules required by the current Split-CIFAR-10
   experiment?

   Recommendation: move only required modules first.

2. Should results logs be fully migrated?

   Recommendation: migrate `summary.csv`, `summary.jsonl`, and selected logs
   for best or decision-point runs. Avoid bulk-moving every LLM log and every
   failed experimental log unless there is a specific audit requirement.

3. Should `FabricPC` be a Git submodule?

   Recommendation: no at first. Use an editable local install for development
   and a pinned Git URL or commit in documentation. A submodule can be added
   later if reproducibility requires it.

4. Should the experiment repo depend on a released FabricPC version?

   Recommendation: not initially. Pin to a commit hash from the local
   `FabricPC` repo once the minimal compatibility changes are made.

## Acceptance Criteria

Migration is complete when:

- `columnarCL-fabricPC-experiments` installs as a package.
- `FabricPC` installs independently and does not contain experiment scripts.
- The migrated experiment imports `fabricpc` from the parent repo, not from a
  copied package.
- `python -m py_compile` passes for migrated scripts.
- A smoke run completes.
- The migrated seed-7 best command produces a result close to the current
  `cFabricPC` seed-7 baseline.
- The README documents setup, the current best command, and expected results.

## Suggested Immediate Next Steps

1. Add `pyproject.toml` and `.gitignore` to this repo.
2. Create `columnar_cl_fabricpc/` package directories.
3. Copy the current experiment scripts into `examples/`.
4. Move the required `fabricpc.continual` modules into
   `columnar_cl_fabricpc.continual`.
5. Refactor imports.
6. Install `FabricPC` and this repo editably in `ffpc7`.
7. Run compile/import checks.
8. Run a tiny smoke experiment.
9. Reproduce seed 7 for the current best configuration.
