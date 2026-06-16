# Implementation Plan: Phases 0–2

This document captures the concrete implementation steps being taken, derived from
`codex-even-more-minimal-fabricpc-hooks-plan.md` and its critical review
`claude-crit-review-even-more-minimal-hooks-plan.md`.

## Source Documents

- `codex-even-more-minimal-fabricpc-hooks-plan.md`: proposes reducing FabricPC
  changes to one item (external custom-node contract) and building everything
  else in the experiment repo.
- `claude-crit-review-even-more-minimal-hooks-plan.md`: augments the plan with
  corrections based on what actually exists in FabricPC (training callbacks
  already exist, custom node contract mostly documented, venv hygiene needed).

## Key Findings from Critical Review

1. **Phase 1 is verification, not creation.** The custom-node extension contract
   already exists in `fabricpc/nodes/base.py:NodeBase` with documentation in
   `docs/user_guides/06_custom_nodes.md` and a working example in
   `examples/custom_node.py`. The missing piece is a test that exercises a node
   defined outside the `fabricpc` package.

2. **Training callbacks already exist.** `train_pcn` accepts `epoch_callback` and
   `iter_callback` parameters. The experiment repo should use these rather than
   re-implementing a training loop.

3. **Venv hygiene required.** The `py3/` venv at `/home/ni/repos/fpc/py3` must
   have upstream FabricPC installed (not cFabricPC) before any other work.

4. **Reduce skeleton to 5 subpackages.** The original plan proposed 10; the
   critical review recommends `columns/`, `data/`, `experiments/`, `utils/`,
   `tests/` initially.

## Implementation Steps

### Phase 0: Venv Hygiene

**Status: COMPLETE**

1. Install upstream FabricPC editable into `/home/ni/repos/fpc/py3`:
   ```
   /home/ni/repos/fpc/py3/bin/pip install -e /home/ni/repos/fpc/FabricPC
   ```

2. Verify resolution:
   ```
   python -c "import fabricpc; print(fabricpc.__file__)"
   # Should print: /home/ni/repos/fpc/FabricPC/fabricpc/__init__.py
   ```

### Phase 1: FabricPC External Custom Node Test

**Status: IN PROGRESS**

Add `tests/test_external_custom_node.py` to FabricPC. The test:

1. Defines a `ScaledSumNode(NodeBase)` subclass in the test file itself,
   simulating an external package.
2. Implements `get_slots()`, `initialize_params()`, `forward()` following the
   six-step contract from `NodeBase.forward` docstring.
3. Places the node in a graph with `Edge`, `TaskMap`, `InferenceSGD`.
4. Runs `initialize_params()` and `initialize_graph_state()`.
5. Runs one forward pass and verifies outputs.
6. Verifies JIT compatibility and gradient computation.

No MNIST or dataset dependencies. The test exercises the contract with synthetic
data only.

**Acceptance criteria:**
- `pytest tests/test_external_custom_node.py` passes.
- No new files added to `fabricpc/` package (test lives in `tests/`).

### Phase 2: Experiment Repo Skeleton

**Status: PENDING**

Create the experiment repo package structure:

```
columnarCL-fabricPC-experiments/
  pyproject.toml
  columnar_cl_fabricpc/
    __init__.py
    columns/
      __init__.py
      example_node.py      # Minimal ColumnNode(NodeBase) for smoke test
    data/
      __init__.py
    experiments/
      __init__.py
    utils/
      __init__.py
      jax_setup.py         # Clean re-implementation of JAX env config
      metadata.py          # Node-name-keyed registry (graph-construction-time only)
  tests/
    __init__.py
    test_imports.py        # Smoke test: imports succeed, custom node works
```

**Subpackages deferred until needed:**
- `bayes/` — when Bayesian routing is implemented
- `causal/` — when causal coding diagnostics are implemented
- `diagnostics/` — when experiment diagnostics beyond callbacks are needed
- `routing/` — when explicit routing logic is implemented
- `shells/` — when concentric shell organization is implemented
- `training/` — if `train_pcn` callbacks prove insufficient

**pyproject.toml:**
- Name: `columnar_cl_fabricpc`
- Dependencies: inherit FabricPC floors (`jax`, `jaxlib`, `optax>=0.1.7`,
  `numpy>=1.24.0`)
- Editable dependency on FabricPC: `fabricpc @ file:///home/ni/repos/fpc/FabricPC`

**utils/jax_setup.py:**
- Clean re-implementation (not import from FabricPC root `jax_setup.py`)
- Sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_GPU_DETERMINISTIC_OPS=true`,
  `XLA_GPU_AUTOTUNE_LEVEL=1`
- Called before any JAX import

**utils/metadata.py:**
- Dict-of-dicts registry keyed by node name
- Document constraint: graph-construction-time only, not usable inside
  `jax.jit`-compiled forward

**columns/example_node.py:**
- Minimal `ExampleColumnNode(NodeBase)` that sums inputs
- Used as smoke test, not a real column implementation

**tests/test_imports.py:**
- `import columnar_cl_fabricpc` succeeds
- `import fabricpc` resolves to upstream
- `ExampleColumnNode` can be placed in a 3-node graph with one forward step

**Acceptance criteria:**
- `pip install -e columnarCL-fabricPC-experiments` succeeds in `py3/`
- `pytest columnarCL-fabricPC-experiments/tests/` passes
- `import fabricpc` still resolves to upstream FabricPC

## Phases 3–4 (Future)

Not implemented in this document. Phase 3 (first columnar experiment) and
Phase 4 (HiBaCaML-inspired additions) will be planned after Phases 1–2 complete.

See the critical review for numeric acceptance thresholds (reproduce prior
cFabricPC baseline accuracy within stated bands).

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package name | `columnar_cl_fabricpc` | Unambiguous in cross-repo grep, matches repo name |
| Initial subpackages | 5 (columns, data, experiments, utils, tests) | Minimal; others added when first module written |
| JAX setup | Clean re-implementation in experiment repo | Avoids path-dependency on FabricPC root file |
| Metadata registry | External dict, graph-construction-time only | No FabricPC changes needed; jit-incompatibility documented |
| Training loop | Use `train_pcn` with callbacks | Callbacks exist; no re-implementation unless proven insufficient |

## Alternatives Considered

See `claude-crit-review-even-more-minimal-hooks-plan.md` section "Alternatives
the plan does not enumerate" for full enumeration.
