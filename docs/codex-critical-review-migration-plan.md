# Codex Critical Review: Migration Plan

## Executive Summary

The existing migration plan is directionally sound: the columnar continual-learning
experiments should move out of `cFabricPC` and into
`columnarCL-fabricPC-experiments`, while the upstream `FabricPC` repository remains
the reusable predictive-coding library.

Several parts of the plan are now stale or need tightening before implementation:

- The target Python environment is now
  `/home/ni/repos/fpc/virt-envs/fpcpy3.12`, not `ffpc7`.
- Migration acceptance should no longer be based on old replay/prototype-memory
  runs.
- Imports should be aligned to upstream FabricPC APIs rather than preserving
  `cFabricPC` compatibility shims.
- CIFAR/data-loader workarounds should stay experiment-local and optional unless
  they are clearly general-purpose FabricPC improvements.
- The dependency list should be kept minimal; avoid hard dependencies added only
  to work around loader issues.

This review should supersede the older review notes where they reference the
old environment, old replay baseline, or old source assumptions.

## Current Ground Truth

The docs directory currently contains:

- `migration-plan.md`
- `claude-crit-review-migration-plan.md`
- `claude-improvement-planning.md`

The target repository currently has only documentation and minimal top-level
metadata; it does not yet contain a Python package skeleton, `pyproject.toml`,
or migrated experiment code.

Local repositories:

- Upstream library: `/home/ni/repos/fpc/FabricPC`
- Current experiment fork: `/home/ni/repos/fpc/cFabricPC`
- New experiment repo:
  `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`

Current commits observed during review:

- `FabricPC`: branch `feature/columnar_CC_adapter`, commit `10fbe2e`
- `cFabricPC`: branch `cifar10`, commit `1b836b0`

The target virtual environment is:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python
```

That environment is clean:

- Python `3.12.13`
- only `pip` installed
- `fabricpc` is not currently importable

This is a good migration starting point because it avoids the old `ffpc7`
shadowing problem, where `fabricpc` resolved to the `cFabricPC` fork instead
of upstream FabricPC.

## Critical Issues

### 1. Old Replay Baselines Are No Longer Valid Targets

The original migration plan and the existing Claude review reference the old
prototype-memory / memory-readout baseline, including commands using:

```bash
--prototype_memory_per_class
--memory_readout_steps
```

Those should no longer be used as migration acceptance criteria. The current
project direction explicitly forbids replay-like stored exemplar memory and
calibration memory.

The migration target should instead be the current no-replay Split-CIFAR path.
Known seed-42 reference points from the no-replay architecture are:

- Control: final native ten-way accuracy `0.1207`
- Best tested no-replay tuning:

  ```bash
  --current_batch_feature_sep_weight 0.03 \
  --current_batch_feature_sep_temperature 0.10
  ```

  final native ten-way accuracy `0.2039`

The migration should preserve these no-replay behaviors before optimizing
further.

### 2. Upstream FabricPC Import Paths Differ From cFabricPC

The active cFabricPC experiment imports several fork-only convenience paths:

```python
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
```

Upstream FabricPC does not expose these namespaces. The migrated experiment
should use upstream-native imports:

```python
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
```

This avoids needing compatibility shims in FabricPC and keeps the experiment
repo aligned with the parent library.

### 3. `fabricpc.continual` Must Become Experiment-Local

Upstream FabricPC does not contain `fabricpc.continual`. That is appropriate:
the continual-learning code is experiment-specific and should move under this
repository, for example:

```text
columnar_cl_fabricpc/continual/
```

The migrated experiment should import:

```python
from columnar_cl_fabricpc.continual.config import make_config
from columnar_cl_fabricpc.continual.data_cifar import build_split_cifar10_loaders
from columnar_cl_fabricpc.continual.nodes import ...
```

Do not copy a top-level `fabricpc` package into the experiment repo.

### 4. Data Loader Workarounds Should Not Become FabricPC Core Dependencies

cFabricPC currently includes CIFAR fallback code under:

```text
fabricpc/utils/data/cifar_raw.py
```

That fallback can require optional packages such as `pyarrow` and `Pillow` for
the HuggingFace parquet mirror. These were useful as workarounds, but they
should not become hard dependencies of FabricPC or of the experiment repo unless
they are truly required.

Recommended policy:

- Prefer upstream FabricPC's standard data path when available.
- Keep CIFAR fallback code experiment-local if it remains necessary.
- Make fallback dependencies optional extras, not base dependencies.
- Do not add `tensorflow`, `tensorflow-datasets`, `pyarrow`, or `Pillow` as hard
  requirements in the first migration pass.

### 5. Source State Must Be Pinned Before Copying

cFabricPC is still evolving. Before migration work starts, record the exact
source commit being migrated from and the FabricPC commit being targeted.

At review time:

```text
cFabricPC source: 1b836b0 on branch cifar10
FabricPC target: 10fbe2e on branch feature/columnar_CC_adapter
```

If either repo changes before implementation, the migration notes should be
updated to record the actual source commits used.

## Recommended Migration Strategy

### Phase 0: Pin and Prepare

Use the new Python 3.12 environment:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -V
```

Record source commits:

```bash
git -C /home/ni/repos/fpc/cFabricPC rev-parse HEAD
git -C /home/ni/repos/fpc/FabricPC rev-parse HEAD
```

Install FabricPC editably into the clean environment:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pip install -e /home/ni/repos/fpc/FabricPC
```

Verify it resolves to upstream FabricPC:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -c \
  "import fabricpc, os; print(os.path.realpath(fabricpc.__file__))"
```

The printed path must be inside:

```text
/home/ni/repos/fpc/FabricPC/
```

### Phase 1: Create Minimal Experiment Package

Add a package skeleton under the experiment repo:

```text
columnarCL-fabricPC-experiments/
  pyproject.toml
  columnar_cl_fabricpc/
    __init__.py
    continual/
      __init__.py
    diagnostics/
      __init__.py
    experiments/
      __init__.py
    data/
      __init__.py
    utils/
      __init__.py
  examples/
  scripts/
```

Keep the base dependency list minimal:

```toml
dependencies = [
  "fabricpc",
  "jax",
  "jaxlib",
  "optax>=0.1.7",
  "numpy>=1.24.0",
]
```

Add `pandas` only if committed summary tooling requires it. Keep CIFAR fallback
packages as an optional extra if needed.

### Phase 2: Migrate Only the Active No-Replay Dependency Closure

Migrate the active Split-CIFAR experiment first:

```text
examples/cifar10_colba_split_disjoint.py
examples/sweep_cifar10_colba_disjoint.py
```

Move required helpers into package modules:

```text
columnar_cl_fabricpc/experiments/cifar10_colba_joint.py
columnar_cl_fabricpc/diagnostics/single_head_ladder_metrics.py
```

Move only the required continual-learning modules first. The likely initial
closure is:

```text
continual/config.py
continual/data.py
continual/data_cifar.py
continual/nodes.py
continual/gradient_protection.py
continual/trainer.py
```

Inspect imports during implementation and move additional modules only if the
active no-replay scripts require them.

Historical scripts such as LwF, microcolumn, psi-selector, and older ladder
variants should be deferred unless they are still part of the no-replay
architecture.

### Phase 3: Rewrite Imports To Parent FabricPC Standard

Use upstream-native FabricPC imports:

| cFabricPC import | migrated import |
|---|---|
| `fabricpc.builder.Edge` | `fabricpc.core.topology.Edge` |
| `fabricpc.builder.TaskMap` | `fabricpc.graph_assembly.TaskMap` |
| `fabricpc.builder.graph` | `fabricpc.graph_assembly.graph` |
| `fabricpc.graph.initialize_params` | `fabricpc.graph_initialization.initialize_params` |
| `fabricpc.graph.state_initializer.initialize_graph_state` | `fabricpc.graph_initialization.state_initializer.initialize_graph_state` |
| `fabricpc.continual.*` | `columnar_cl_fabricpc.continual.*` |
| local `cifar10_colba_joint` | `columnar_cl_fabricpc.experiments.cifar10_colba_joint` |
| local `single_head_ladder_metrics` | `columnar_cl_fabricpc.diagnostics.single_head_ladder_metrics` |

For JAX setup, do not depend on `fabricpc.utils.helpers` unless upstream adds
that helper. Prefer one of these:

- call upstream root module `jax_setup.py` if it is intentionally supported, or
- keep a tiny experiment-local helper in `columnar_cl_fabricpc.utils.jax_setup`.

Do not modify FabricPC just to preserve `cFabricPC` import spellings.

### Phase 4: Keep Replay Removed

The migrated scripts should not expose or document replay-like options as valid
experiment paths.

The following should be absent or hard-disabled:

```text
--prototype_memory_per_class > 0
--prototype_gate_holdout_per_class > 0
--memory_readout_steps > 0
--pair_readout_steps > 0
stored exemplar memory
calibration-memory readouts
ReplayBuffer sampling
```

If legacy flags are retained for backwards CLI compatibility, they must fail
immediately when set to replay-enabling values.

### Phase 5: Results and Docs

Do not bulk-copy old result logs. Keep only compact summaries and carefully
selected reference logs.

Recommended result docs:

```text
results/README.md
results/sweeps/cifar10_colba_disjoint/summary.jsonl
```

If a CSV summary is desired, generate it from JSONL during migration rather
than assuming it already exists.

The README should document:

- target FabricPC commit
- source cFabricPC commit
- Python version and environment path
- no-replay baseline command
- tuned no-replay command
- expected seed-42 reference results

## Minimal Dependency Plan

Base dependencies should be as small as practical:

```text
fabricpc
jax
jaxlib
optax
numpy
```

Potential optional extras:

```toml
[project.optional-dependencies]
analysis = ["pandas>=2.0.0"]
cifar-fallback = ["pyarrow>=14", "Pillow>=10"]
```

Do not make these hard dependencies unless a migrated script cannot run without
them:

- `tensorflow`
- `tensorflow-datasets`
- `pyarrow`
- `Pillow`
- `pandas`
- `aim`
- `plotly`
- `kaleido`

FabricPC already owns its own optional extras. The experiment repo should not
broaden FabricPC's base install footprint to support one experiment.

## No-Replay Acceptance Criteria

Migration is complete when:

1. `FabricPC` installs editably into `fpcpy3.12`.
2. `columnarCL-fabricPC-experiments` installs editably into `fpcpy3.12`.
3. `import fabricpc` resolves to `/home/ni/repos/fpc/FabricPC`.
4. `import columnar_cl_fabricpc` succeeds.
5. All migrated Python files compile.
6. A smoke run completes when CIFAR data is available.
7. A full no-replay seed-42 control run reproduces the current control behavior
   within a practical tolerance.
8. A full no-replay seed-42 tuned run reproduces the current tuned behavior
   within a practical tolerance.
9. No documented command relies on replay, stored exemplar memory, or
   calibration-memory readout.

Suggested initial numeric targets:

| run | expected final native ten-way |
|---|---:|
| control seed 42 | `0.1207` |
| feature-separation tuned seed 42 | `0.2039` |

Because the no-replay architecture is still unstable, use these as migration
equivalence checks rather than scientific performance goals. A difference of
roughly `0.02` should trigger inspection for import, dependency, data-order, or
JAX-version drift.

## Immediate Next Steps

1. Pin the exact source and target commits in a `MIGRATION_SOURCE.md` or README
   section.
2. Add the experiment package skeleton and minimal `pyproject.toml`.
3. Install FabricPC and the experiment repo into `fpcpy3.12`.
4. Move only the active no-replay Split-CIFAR dependency closure.
5. Rewrite imports to upstream-native FabricPC paths.
6. Keep CIFAR fallback code experiment-local and optional.
7. Compile all migrated Python files.
8. Run an import smoke test that proves `fabricpc` resolves to upstream
   FabricPC.
9. Reproduce the seed-42 no-replay control and tuned runs.

## Final Recommendation

Proceed with the migration, but update the plan before implementation:

- Treat the old replay/prototype-memory baseline as deprecated.
- Use `fpcpy3.12` as the clean target environment.
- Rewrite imports to upstream FabricPC APIs instead of adding compatibility
  shims.
- Keep experiment-specific continual-learning code under
  `columnar_cl_fabricpc`.
- Keep the first migration narrow: active no-replay Split-CIFAR only.
- Keep additional pip dependencies optional unless they are proven necessary.

