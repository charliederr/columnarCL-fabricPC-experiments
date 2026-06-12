# Even More Minimal FabricPC Hooks Plan

## Feasibility Answer

Yes. We can reduce the required FabricPC changes to only the custom-node
extension contract described in Section 2 of
`codex-minimal-fabricpc-hooks-plan.md`.

The other proposed FabricPC changes can be avoided:

- Graph metadata can live in the experiment repo as an external registry keyed
  by node name or graph role.
- Training observation can live in experiment-owned training loops and wrappers.
- JAX setup can live in the experiment repo and be called before importing JAX.

This is the cleaner reset: make FabricPC only prove that external custom nodes
can integrate with its graph/init/forward machinery, then build all columnar
causal-coding and HiBaCaML-inspired behavior in
`columnarCL-fabricPC-experiments`.

## Only FabricPC Work

FabricPC should receive exactly one small class of changes:

### External Custom Node Contract

Add or document the minimum contract needed for an external package to define a
node that works with FabricPC.

This should include:

- which base class or protocol a custom node should follow
- required fields such as `name`, `shape`, slots, and parameter initialization
  behavior
- how the node participates in graph construction
- how params and state are initialized
- how forward execution should expose outputs
- a tiny test or example showing an external node used in a FabricPC graph

This is general-purpose library work. It does not mention columns, tasks,
CIFAR, continual learning, HiBaCaML, Bayesian routing, or causal coding.

Preferred deliverable in FabricPC:

```text
tests/test_external_custom_node.py
docs or examples/custom_node_minimal.py
```

If the current FabricPC node APIs are already sufficient, this may be only a
test and documentation change. Do not refactor core internals unless the test
exposes a real blocker.

## No Other FabricPC Changes

Do not add these to FabricPC for the initial reset:

- graph metadata fields
- training observer/callback hooks
- package-level JAX setup helper
- columnar nodes
- shell abstractions
- support/routing APIs
- Bayesian priors or posterior utilities
- causal coding utilities
- continual-learning package
- CIFAR loaders or Split-CIFAR helpers
- experiment dependencies
- compatibility shims for `fabricpc.builder` or `fabricpc.graph`

All of those can live in the experiment repo for now.

## How The Experiment Repo Replaces The Dropped Hooks

### Metadata Without FabricPC Changes

Keep metadata outside FabricPC in experiment-owned structures:

```python
node_roles = {
    "stem": {"role": "visual_stem"},
    "col_00": {"role": "shared_column", "shell": 0},
    "col_07": {"role": "local_column", "task": 1, "shell": 1},
}
```

Use node names as stable keys. This is enough for:

- column ids
- shell ids
- support groups
- local/shared roles
- diagnostic tags
- readout roles

If this later becomes painful, a FabricPC metadata field can be proposed with
concrete evidence. It is not needed up front.

### Training Observation Without FabricPC Changes

Own the training loop in `columnarCL-fabricPC-experiments`.

The experiment loop can call FabricPC graph initialization and forward/state
functions, then collect:

- activations
- route/support usage
- per-node losses
- calibration drift
- uncertainty summaries
- causal attribution estimates

This avoids putting experiment-specific observer semantics into FabricPC.

### JAX Setup Without FabricPC Changes

Put a tiny helper in the experiment repo:

```text
columnar_cl_fabricpc/utils/jax_setup.py
```

Every experiment entrypoint should call it before importing JAX:

```python
from columnar_cl_fabricpc.utils.jax_setup import configure_jax

configure_jax()

import jax
```

The helper should remain backend-neutral and should not hard-code a particular
CUDA version. Environment selection can be controlled by shell variables or
experiment CLI policy.

### Bayesian / HiBaCaML Functionality

Put all of this in the experiment repo:

```text
columnar_cl_fabricpc/
  bayes/
  causal/
  columns/
  diagnostics/
  routing/
  shells/
```

FabricPC should see these as ordinary external nodes, parameter pytrees, and
training code.

## Proposed New Experiment Repo Shape

```text
columnarCL-fabricPC-experiments/
  pyproject.toml
  README.md
  columnar_cl_fabricpc/
    __init__.py
    bayes/
      __init__.py
    causal/
      __init__.py
    columns/
      __init__.py
      nodes.py
    data/
      __init__.py
      cifar.py
    diagnostics/
      __init__.py
    experiments/
      __init__.py
    routing/
      __init__.py
    shells/
      __init__.py
    training/
      __init__.py
    utils/
      __init__.py
      jax_setup.py
  examples/
    split_cifar10_minimal.py
  tests/
    test_imports.py
    test_custom_column_node.py
```

The first experiment should be deliberately small:

- no replay
- no old exemplar memory
- no pair-label leakage
- ten-way CIFAR output
- fixed support structure
- simple spatial columns
- explicit diagnostics

## Minimal Dependency Policy

Base dependencies should stay small:

```text
fabricpc
jax
jaxlib
optax
numpy
```

Optional extras can be added later:

```toml
[project.optional-dependencies]
analysis = ["pandas"]
cifar-fallback = ["Pillow"]
```

Avoid hard dependencies on:

- TensorFlow
- TensorFlow Datasets
- pyarrow
- pandas
- aim
- plotly

Only add one when a concrete experiment module requires it and there is no
simpler alternative.

## Implementation Order

### Phase 1: FabricPC Custom Node Proof

In FabricPC:

1. Add a minimal external custom-node example or test.
2. Verify the example can be imported from outside the FabricPC package.
3. Verify the node can be placed in a graph.
4. Verify params/state initialize.
5. Verify a forward pass works.

Stop there. Do not add metadata, callbacks, JAX helpers, or columnar APIs.

### Phase 2: Experiment Repo Skeleton

In `columnarCL-fabricPC-experiments`:

1. Add `pyproject.toml`.
2. Add package directories.
3. Add experiment-local `utils/jax_setup.py`.
4. Add experiment-local metadata registry helpers.
5. Add a tiny custom column node.
6. Add import and smoke tests.

### Phase 3: First Clean Columnar Experiment

Build a minimal no-replay Split-CIFAR experiment:

1. Load data.
2. Build a FabricPC graph with external column nodes.
3. Track metadata externally by node name.
4. Train through experiment-owned loop.
5. Report ten-way accuracy and forgetting.
6. Add diagnostics only in the experiment repo.

### Phase 4: Add HiBaCaML-Inspired Ideas Incrementally

After the minimal experiment works:

1. Add shared/local column roles.
2. Add concentric shell organization.
3. Add causal coding diagnostics.
4. Add Bayesian route priors.
5. Add uncertainty-aware support scoring.
6. Add calibrated readout experiments.

None of these require additional FabricPC changes until a concrete limitation is
encountered.

## Acceptance Criteria

FabricPC is sufficiently prepared when:

- an external custom node test passes
- no columnar or continual-learning code has been added to FabricPC
- no new hard dependencies have been added to FabricPC
- upstream-native imports remain the public graph API

The experiment repo is ready when:

- it installs in `/home/ni/repos/fpc/virt-envs/fpcpy3.12`
- `import fabricpc` resolves to `/home/ni/repos/fpc/FabricPC`
- `import columnar_cl_fabricpc` succeeds
- a custom column node can be used in a FabricPC graph
- metadata and diagnostics are managed experiment-locally
- no replay or stored-exemplar path exists

## Bottom Line

We can reduce FabricPC work to one thing: prove and document that external
custom nodes are supported.

Everything else needed for columnar causal coding, Bayesian routing, shells,
metadata, diagnostics, JAX setup, and no-replay continual-learning experiments
can and should be built in `columnarCL-fabricPC-experiments` first.

