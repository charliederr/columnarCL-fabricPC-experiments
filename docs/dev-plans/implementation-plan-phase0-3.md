# Implementation Plan: Phases 0–3 (Current)

This document captures the actual implementation steps taken, updated as work
proceeds. It supersedes `implementation-plan-phase0-2.md` which was the initial
plan before implementation began.

## Source Documents

- `codex-even-more-minimal-fabricpc-hooks-plan.md`: proposes reducing FabricPC
  changes to one item (external custom-node contract) and building everything
  else in the experiment repo.
- `claude-crit-review-even-more-minimal-hooks-plan.md`: augments the plan with
  corrections based on what actually exists in FabricPC.

## Implementation Progress

### Phase 0: Venv Hygiene

**Status: COMPLETE**

Installed upstream FabricPC editable into `/home/ni/repos/fpc/py3`:
```
/home/ni/repos/fpc/py3/bin/pip install -e /home/ni/repos/fpc/FabricPC
```

Verified:
```
python -c "import fabricpc; print(fabricpc.__file__)"
# /home/ni/repos/fpc/FabricPC/fabricpc/__init__.py
```

### Phase 1: FabricPC External Custom Node Test

**Status: COMPLETE**

Added `FabricPC/tests/test_external_custom_node.py` with:

- `ScaledSumNode(NodeBase)` subclass defined in the test file (simulates
  external package)
- Implements `get_slots()`, `initialize_params()`, `forward()` following the
  six-step contract from `NodeBase.forward`
- 10 tests covering: instantiation, slots, graph placement, param init, state
  init, forward pass, multiple inputs, activation, JIT compatibility, gradients

All 10 tests pass.

### Phase 2: Experiment Repo Skeleton

**Status: COMPLETE**

```
columnarCL-fabricPC-experiments/
  pyproject.toml
  columnar_cl_fabricpc/
    __init__.py
    columns/
      __init__.py
      example_node.py            # ExampleColumnNode(NodeBase)
    data/
      __init__.py
    experiments/
      __init__.py
    utils/
      __init__.py
      metadata.py                # NodeMetadataRegistry
  tests/
    __init__.py
    conftest.py                  # JAX env config
    test_imports.py              # 14 smoke tests
```

**Result:** All 14 tests pass. Package installs successfully. FabricPC resolves
to upstream.

### Phase 3: First Columnar Experiment

**Status: NOT STARTED**

Will be planned after Phase 2 completes. See critical review for numeric
acceptance thresholds.

## Decisions Made During Implementation

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package name | `columnar_cl_fabricpc` | Unambiguous in cross-repo grep |
| Initial subpackages | 5 (columns, data, experiments, utils, tests) | Minimal |
| JAX setup | No dedicated module | Env vars in conftest.py and script tops suffice |
| Metadata registry | `NodeMetadataRegistry` class | Graph-construction-time only; jit-incompatibility documented |
| Training loop | Use `train_pcn` with callbacks | Callbacks already exist in FabricPC |

## Deviations from Original Plan

1. **Removed `utils/jax_setup.py`**: Originally planned a dedicated module for
   JAX environment configuration. Decided against it because env vars can be
   set in `conftest.py` for tests or at the top of experiment scripts. A
   dedicated module adds indirection without clear benefit.

## Files Created

**In FabricPC:**
- `tests/test_external_custom_node.py` — 10 tests for external custom node contract

**In columnarCL-fabricPC-experiments:**
- `pyproject.toml`
- `columnar_cl_fabricpc/__init__.py`
- `columnar_cl_fabricpc/columns/__init__.py`
- `columnar_cl_fabricpc/columns/example_node.py`
- `columnar_cl_fabricpc/data/__init__.py`
- `columnar_cl_fabricpc/experiments/__init__.py`
- `columnar_cl_fabricpc/utils/__init__.py`
- `columnar_cl_fabricpc/utils/metadata.py`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/test_imports.py` — 14 smoke tests
- `docs/dev-plans/implementation-plan-phase0-2.md` (initial plan)
- `docs/dev-plans/implementation-plan-phase0-3.md` (this file)
