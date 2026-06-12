# Minimal FabricPC Hooks Plan for Columnar Causal Coding

## Purpose

This plan intentionally steps away from migrating the old `cFabricPC`
experiments wholesale. The goal is to make the smallest useful changes to
upstream `FabricPC` so a new, cleaner implementation of columnar causal coding
with Bayesian / HiBaCaML-inspired ideas can be built in
`columnarCL-fabricPC-experiments`.

The old `cFabricPC` code should remain available as reference material, not as
the architecture to preserve. The new experiment repository should be free to
rebuild the columnar system with better boundaries, clearer diagnostics, and no
replay mechanisms.

## Guiding Principles

- Keep `FabricPC` a general predictive-coding library.
- Put columnar continual-learning policy in `columnarCL-fabricPC-experiments`.
- Add only stable extension hooks to `FabricPC`; do not add Split-CIFAR policy,
  HiBaCaML-specific routing, replay, prototype memory, or experiment scripts.
- Prefer upstream-native FabricPC imports over cFabricPC compatibility shims.
- Keep dependency additions minimal. New experiment dependencies belong in the
  experiment repo, not FabricPC, unless they are broadly reusable.
- Treat `/home/ni/repos/fpc/virt-envs/fpcpy3.12` as the target environment.

## Minimal FabricPC Changes

### 1. Public Import Stability

Use current upstream FabricPC paths as the stable public surface:

```python
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear
```

Recommended FabricPC work:

- Document these paths as the supported graph-construction API.
- Add lightweight import tests so these paths do not drift.
- Do not add `fabricpc.builder` or `fabricpc.graph` compatibility packages just
  to preserve cFabricPC import spelling.

### 2. Node Extension Contract

The experiment repo needs to define new nodes such as:

- truncated visual backbones
- spatial residual columns
- support-conditioned composers
- Bayesian or uncertainty-aware routing modules
- calibrated readout nodes

FabricPC should only ensure custom nodes can be added cleanly.

Recommended FabricPC work:

- Add a short developer guide or test showing how an external package defines a
  node compatible with FabricPC graph initialization and forward execution.
- Ensure the public base node/type interfaces are importable and documented.
- Avoid adding column, shell, route, task, or CIFAR-specific node classes to
  FabricPC.

### 3. Graph Metadata Hook

Columnar causal coding will likely need graph-level metadata that is not part of
ordinary tensor edges, for example:

- column id
- shell id
- support group
- local/shared role
- readout role
- diagnostic tags

Recommended FabricPC work:

- If current `Node` / `Edge` objects already support metadata, document that
  convention.
- If they do not, add a minimal optional `metadata: dict` field or equivalent
  non-invasive tag mechanism.
- Metadata must be ignored by core training unless explicitly consumed by
  external experiment code.

This gives the experiment repo enough structure to build column/shell
diagnostics without baking the columnar architecture into FabricPC.

### 4. Training Callback / Observation Hook

HiBaCaML-style development will need diagnostics during training:

- per-node activations
- route/support usage
- local losses
- calibration drift
- uncertainty summaries
- causal attribution estimates

Recommended FabricPC work:

- Provide or document a minimal callback/observer hook around training steps.
- The hook should receive enough context to inspect params, state, batch, loss,
  and step metadata.
- The hook should be optional and no-op by default.

Do not add experiment-specific metrics to FabricPC. The experiment repo should
own all columnar diagnostics.

### 5. JAX Setup Helper

The new experiment repo needs a stable way to set JAX flags before importing
JAX, especially for CUDA backend selection.

Recommended FabricPC work:

- Either document the existing root-level `jax_setup.py` as public, or expose a
  tiny package-level helper such as:

  ```python
  from fabricpc.utils.jax_setup import set_jax_flags_before_importing_jax
  ```

- Keep this helper backend-neutral. It should not hard-code CUDA 12 or CUDA 13.

This is a general utility and may reasonably live in FabricPC if kept small.

### 6. Packaging Extras Only If Broadly Useful

FabricPC already has backend extras such as `cuda12` and `cuda13`. That is the
right place for JAX backend selection.

Recommended FabricPC work:

- Keep FabricPC base dependencies unchanged unless a general library feature
  truly needs another package.
- Do not add CIFAR fallback dependencies such as `pyarrow` or `Pillow` to the
  FabricPC base install.
- Do not add experiment analysis dependencies such as `pandas`, `plotly`, or
  `aim` outside optional extras.

## What Belongs In The Experiment Repo

The following should be built in `columnarCL-fabricPC-experiments`, not
FabricPC:

- Split-CIFAR task construction and experiment policy
- columnar support layout
- shared/local column division
- concentric shell definitions
- causal coding / HiBaCaML-specific update rules
- Bayesian route priors or posterior approximations
- route-stability objectives
- calibrated readout experiments
- all no-replay continual-learning diagnostics
- sweep runners and result summarizers
- optional CIFAR fallback data loaders

The experiment repo should import FabricPC as a normal dependency and expose its
own package, for example:

```text
columnar_cl_fabricpc/
  columns/
  causal/
  bayes/
  diagnostics/
  experiments/
  data/
```

## Explicit Non-Goals

- Do not migrate all historical `cFabricPC` algorithms as a first step.
- Do not preserve replay or stored-exemplar memory paths.
- Do not add `fabricpc.continual` to upstream FabricPC.
- Do not add cFabricPC namespace shims unless there is an independent upstream
  reason.
- Do not make FabricPC depend on CIFAR-specific packages.
- Do not make FabricPC aware of Split-CIFAR, task pairs, or column supports.

## Suggested Implementation Order

### Phase 1: Stabilize FabricPC API Surface

In FabricPC:

1. Add import tests for the upstream graph-construction and initialization API.
2. Add a short external-node example or test.
3. Add or document a metadata convention for nodes/edges.
4. Add or document a no-op training observer/callback hook.
5. Add or document a package-level JAX setup helper.

Each change should be small, independently reviewable, and useful outside this
one experiment.

### Phase 2: Scaffold The Experiment Repo

In `columnarCL-fabricPC-experiments`:

1. Create `pyproject.toml` with minimal dependencies.
2. Install with `/home/ni/repos/fpc/virt-envs/fpcpy3.12`.
3. Verify `fabricpc` resolves to `/home/ni/repos/fpc/FabricPC`.
4. Create package directories for columns, causal logic, Bayesian routing, data,
   diagnostics, and experiments.
5. Add smoke tests that import FabricPC and instantiate a tiny custom node.

### Phase 3: Build The New Architecture Incrementally

Start from a small, clean experiment rather than the old full script:

1. Joint CIFAR sanity test with a truncated visual stem and simple columns.
2. Sequential Split-CIFAR with fixed no-leak supports and ten-way readout.
3. Add shared/local columns and shell metadata.
4. Add causal coding diagnostics.
5. Add Bayesian route priors or uncertainty-aware support scoring.
6. Add calibrated readout and no-replay retention objectives.

At each step, keep the run no-replay and ten-way.

## Acceptance Criteria

FabricPC is ready for the new experiment repo when:

- It installs cleanly in `fpcpy3.12`.
- The documented public imports work.
- An external custom node can be defined and used without modifying FabricPC.
- Optional metadata can be attached without changing training behavior.
- A training observer/callback can inspect state without becoming required.
- JAX setup is documented or exposed in a stable package-level location.
- No experiment-specific dependencies or Split-CIFAR policy have been added.

The experiment repo is ready to begin architecture work when:

- It installs cleanly in `fpcpy3.12`.
- It imports upstream FabricPC, not cFabricPC.
- It has a minimal custom-node smoke test.
- It has no replay, stored-exemplar, or calibration-memory path.
- It has a clean first experiment script that can run a tiny no-replay
  Split-CIFAR smoke test.

## Bottom Line

The right reset is not to port the old cFabricPC architecture into a new
package. The right reset is to give FabricPC a few stable, general hooks and
then build a new columnar causal-coding experiment stack around those hooks.

Keep FabricPC small. Keep columnar causal learning in the experiment repo. Use
cFabricPC as a reference, not as the template.

