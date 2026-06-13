# Improved Migration Plan — Columnar Causal Coding on Upstream FabricPC

Successor to `codex-even-more-minimal-fabricpc-hooks-plan.md`, written as a
standalone executable plan that incorporates findings from the critical
review (`claude-crit-review-even-more-minimal-hooks-plan.md`) and from the
earlier migration-plan review (`claude-crit-review-migration-plan.md`).

This plan:

- Pins repositories and branches explicitly so every step is unambiguous.
- Re-scopes the FabricPC-side work from "add" to "verify" where the work is
  already done in upstream.
- Proposes concrete fixes for training observation (existing
  `train_pcn` callbacks plus a state-capture re-forward when needed).
- Proposes concrete fixes for metadata that must be readable inside a
  `jax.jit`-compiled forward (graph-construction-time decisions baked into
  graph structure plus per-node static config via `node_info.node_config`).
- Enumerates alternatives with pros/cons for every load-bearing decision, per
  the experiment repo's `AGENTS.md` planning rule.
- States reproducibility thresholds tied to the prior `cFabricPC`
  multi-seed numbers.

## Repos and branches

These are pinned for the duration of this plan. Any change to a pin requires
a new revision of the plan.

| Role                | Repo path                                          | Branch                          | Commit pin (as of this writing) |
|---------------------|----------------------------------------------------|----------------------------------|----------------------------------|
| Upstream library    | `/home/ni/repos/fpc/FabricPC`                      | `feature/columnar_CC_adapter`    | `10fbe2e`                        |
| Legacy fork (reference only) | `/home/ni/repos/fpc/cFabricPC`            | `cifar10`                        | use as read-only reference; no migration of files |
| New experiment repo | `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`| `main`                           | repo is currently nearly empty (only `AGENTS.md`, `CLAUDE.md`, `LICENSE`, `README.md`, `docs/`) |

Repo-by-repo writeable scope:

- **FabricPC, branch `feature/columnar_CC_adapter`**: writeable for the
  contract-test and documentation work in Phase 1. No files added under
  paths that name columns, tasks, CIFAR, continual learning, HiBaCaML,
  Bayesian routing, or causal coding. No new mandatory dependencies in
  `pyproject.toml`.
- **cFabricPC, branch `cifar10`**: read-only reference. No files migrated as
  files; concepts and numerical baselines are referenced.
- **columnarCL-fabricPC-experiments, branch `main`**: writeable for
  everything in Phase 2 onward.

## Goal

Build a clean implementation of columnar causal coding (and selected
HiBaCaML-inspired additions) in `columnarCL-fabricPC-experiments` against
upstream `FabricPC` as a normal installed dependency, without porting the
`cFabricPC` fork's tree wholesale. Reproduce the prior multi-seed cFabricPC
baselines as the floor of Phase 3 acceptance; then extend.

The boundary policy is unchanged from the codex plans:

- FabricPC stays a general predictive-coding library.
- Columnar continual-learning policy lives in
  `columnarCL-fabricPC-experiments`.

## FabricPC-side work — re-scope from "add" to "verify"

Several Phase 1 deliverables proposed by the codex plan already exist in
upstream FabricPC. The work is verification, hardening, and one new test —
not new feature work.

### What already exists in FabricPC

These exist on `feature/columnar_CC_adapter` at `10fbe2e`:

- `fabricpc/nodes/base.py:NodeBase` — explicit `ABC` with three abstract
  `@staticmethod`s (`get_slots`, `initialize_params`, `forward`). The
  module docstring (lines 8–32) shows the exact subclassing pattern
  external packages use. The `forward` docstring (lines 304–355) enumerates
  the six steps every subclass must perform.
- `examples/custom_node.py` — `Conv2DNode(NodeBase)` defined end-to-end and
  trained on MNIST through `fabricpc.training.train_pcn`. A complete
  external-node demonstration.
- `docs/user_guides/06_custom_nodes.md` — written guide for custom nodes.
- `fabricpc.training.train_pcn` — accepts `epoch_callback` and
  `iter_callback` parameters (`fabricpc/training/train.py:293–294, 313,
  434, 451–460`).
- `fabricpc/utils/dashboarding/{callbacks.py, trackers.py, extractors.py}`
  — callback factories integrated with the Aim experiment tracker.

### The one thing FabricPC still needs

`examples/custom_node.py` is colocated with FabricPC. There is no test that
exercises a `NodeBase` subclass defined in a *different installed package*.
That is the actual upstream gap.

**Phase 1 deliverable in FabricPC:**

1. Add `tests/test_external_custom_node.py`. The test defines a minimal
   `NodeBase` subclass inside the test file itself, places it in a
   `graph(...)` call with `Edge`, `TaskMap`, and built-in nodes
   (`IdentityNode`, `Linear`), runs `initialize_params` and
   `initialize_graph_state`, runs one forward step. No MNIST dependency,
   no `fabricpc.utils.data`, no example-script coupling.
2. Extend `docs/user_guides/06_custom_nodes.md` with a "Packaging an
   external node module" section that shows the `pyproject.toml` shape
   and import pattern an external package uses.
3. Do not add metadata fields, do not add new callback hooks
   (they exist already), do not add columnar code.

### What does not change in FabricPC

- No `fabricpc/continual/` package.
- No `fabricpc/builder/` or `fabricpc/graph/` compatibility shims for the
  cFabricPC fork's namespace.
- No CIFAR-specific or Split-CIFAR-specific code.
- No new mandatory dependencies in `pyproject.toml`.
- No metadata field on `Node` / `Edge` — see "Metadata strategy" below for
  why this is not needed.
- No new training observer hook — see "Training observation strategy"
  below for why the existing one suffices.

## Training observation strategy

The codex even-more-minimal plan removed the proposed training observer hook
on the grounds that the experiment loop should own observation. Upstream
already has the relevant machinery (`iter_callback`, `epoch_callback`,
dashboard trackers). This section states what those callbacks provide, what
they do not, and which gap (if any) needs experiment-side code.

### What `train_pcn` callbacks give

From `fabricpc/training/train.py`:

- `iter_callback(epoch_idx, batch_idx, energy) -> energy`. Called at end of
  each batch. Receives the scalar batch energy. Useful for per-batch
  energy logging and any post-hoc derived metric of that scalar.
- `epoch_callback(epoch_idx, params, structure, config, rng_key) -> Any`.
  Called at end of each epoch. Receives the live `GraphParams`, the
  `GraphStructure`, the run config, and a JAX PRNG key. Sufficient to:
  - Run an evaluation forward pass with a held-out loader.
  - Capture per-node parameter snapshots (the `GraphParams` pytree).
  - Compute per-layer weight statistics.
  - Run a separate forward to extract `GraphState` and diagnose
    activations.

The existing `fabricpc/utils/dashboarding/callbacks.py:create_epoch_callback`
already wires this to an `AimExperimentTracker` with an optional `eval_fn`
and `eval_loader`. Adopting it costs at most a few lines.

### What callbacks do not give

- **Per-inference-step state.** Inside `train_pcn`'s inner loop, the PCN
  inference iterations (`InferenceSGD(eta_infer=..., infer_steps=...)`)
  update `GraphState.z_latent` for each node. The callbacks see the
  *post-step energy* only; they do not see intermediate inference
  trajectories or per-step latent updates.
- **Per-node activations *at the training batch*.** `epoch_callback`
  receives `GraphParams`, not `GraphState`. The state at the last
  training batch is not available; a separate forward is needed to
  recompute it.

### Three solutions, ordered by cost

**Solution A (recommended for Phase 3): use `epoch_callback` with a
state-capture re-forward.**

The experiment writes an `epoch_callback` that:

1. Runs `evaluate_pcn` (or the experiment's equivalent) on a small
   diagnostic batch with a captured `GraphState` return path.
2. Extracts per-node `z_mu`, `z_latent`, `error`, `energy` from that
   state.
3. Computes column / shell / route diagnostics on that state.
4. Returns them to the trainer for logging.

This works because `evaluate_pcn` already runs a forward through the
graph; the captured `GraphState` is the per-node observation surface.

- **Pro:** zero changes to FabricPC. Drop-in compatible with existing
  `train_pcn`. Composable with the Aim tracker if desired.
- **Pro:** observation cost is per-epoch, not per-batch, which matches
  what continual-learning experiments care about anyway (task boundaries,
  forgetting curves).
- **Con:** observations are at epoch boundaries, not per-batch. Phase 3
  is fine with this; Phase 4 (calibration drift curves at sub-epoch
  resolution) may need more.

**Solution B (deferred to Phase 4 if needed): use `iter_callback` plus a
ring buffer of intermediate states.**

If sub-epoch resolution is needed, the experiment writes an
`iter_callback` that records the scalar batch energy and, every K
batches, kicks off a state-capture re-forward as in Solution A. The
re-forward runs on the *current* batch (or a held-out diagnostic batch);
state is appended to a ring buffer.

- **Pro:** still zero changes to FabricPC.
- **Pro:** sub-epoch resolution at the experiment's chosen cost.
- **Con:** doubled forward cost on the K-th batch. Acceptable if K ≥ 50.

**Solution C (only if A and B prove insufficient): own the training
loop in the experiment repo.**

The experiment forks `train_pcn`'s structure into a local
`columnar_cl_fabricpc/training/columnar_train.py` and adds a third
callback (`infer_step_callback`) inside the inference loop.

- **Pro:** maximum flexibility.
- **Pro:** experiment owns its loop and is not blocked on upstream PRs.
- **Con:** duplicates trainer code that may drift with upstream.
- **Con:** "experiment loop owns observation" was the codex
  even-more-minimal plan's reasoning; this review pushes back because
  Solutions A and B suffice for everything in Phase 3 and most of
  Phase 4.

### Recommendation

- **Phase 3:** Solution A. No FabricPC changes.
- **Phase 4 calibration-drift diagnostics:** evaluate Solution B; adopt
  if needed.
- **Solution C:** defer indefinitely; revisit only if A and B are shown
  insufficient on a specific diagnostic.

## Metadata strategy — jit-compatible from the start

The codex even-more-minimal plan proposes "external registry keyed by
node name" — a Python-side `dict` of `{node_name: {"shell": 1, "task": 1,
...}}`. That registry is unusable inside a `jax.jit`-compiled forward,
because the dict is keyed on a Python string that has no place in a
JAX traced computation.

This section gives the two valid surfaces metadata can live on, and a
recipe for which surface each kind of metadata uses.

### Surface 1: graph-construction-time only (Python-side, never inside forward)

For metadata that controls *what graph is built*, not *what the graph
computes per-batch*. Examples:

- Column id, shell id, support-group membership, local-versus-shared role,
  task id at graph-construction time, diagnostic tag, readout role.

This metadata is consulted only when building the graph. The graph
builder reads it and decides:

- Which nodes exist (one column per task-pair → 5 task-local pathways).
- How edges connect them (shell-0 column receives the shared stem; shell-1
  column receives shell-0 plus a residual edge).
- Which weight slots correspond to which roles (shared composer has one
  weight slot per support member).

Once the graph is built, the metadata has done its job. The forward never
reads it. A Python-side `dict` registry works fine.

**Concrete shape:**

```python
# columnar_cl_fabricpc/utils/metadata.py
node_roles = {
    "stem": {"role": "visual_stem"},
    "col_t0_sh0": {"role": "local_column", "task": 0, "shell": 0},
    "col_t0_sh1": {"role": "local_column", "task": 0, "shell": 1},
    # ...
    "agg_t0":   {"role": "aggregator", "task": 0},
    "readout": {"role": "ten_way_readout"},
}
```

This registry is the source of truth for graph construction. Renaming a
node requires updating both the graph builder and the registry; a unit
test enforces that every node in the constructed `GraphStructure` has a
matching registry entry.

### Surface 2: per-node static config (visible inside forward at trace time)

For metadata that the forward *needs to read* — e.g., a shell-aware
energy weighting, a column-specific activation scaling, or a routing
prior tied to the column id.

`NodeBase.__init__` already takes `**extra_config` and stores it in
`self._extra_config` as a `MappingProxyType`. The graph builder threads
this through to `node_info.node_config`, which is a static field on
`NodeInfo`. Inside `forward(params, inputs, state, node_info)`, the
forward can read `node_info.node_config["shell"]` at trace time. The
value is a Python literal at trace time; JAX specializes the compiled
function to that literal. Different shells produce different compiled
forwards — which is what is wanted.

**Concrete shape:**

```python
class ColumnNode(NodeBase):
    def __init__(self, shape, name, *, shell, task, route_prior, **kwargs):
        super().__init__(
            shape=shape, name=name,
            shell=shell, task=task, route_prior=route_prior,
            **kwargs,
        )
    @staticmethod
    def forward(params, inputs, state, node_info):
        shell = node_info.node_config["shell"]          # Python int, traceable as a static literal
        prior = node_info.node_config["route_prior"]    # Python float, same
        # ...energy computation that uses shell and prior...
```

This is jit-safe because `node_info` is part of the structure pytree
that FabricPC's training treats as static between batches; reads on it
specialize the compiled forward.

### Surface 3 (avoid unless forced): runtime-varying metadata

If metadata varies *per batch* — e.g., a task id that switches between
batches — it must be threaded in as a JAX array, not a Python value.
Concretely, an "input clamp" node holds the integer task id as a
`jnp.array([task_id], dtype=jnp.int32)`; downstream nodes read it from
`inputs["task_id"]` and use it inside the forward.

**This surface is avoided in Phase 3 by design.** Phase 3 uses a fixed
support structure that does not switch at runtime; the task being trained
right now is a property of the training loop, not of the graph. If
Phase 4 introduces runtime task-switching, this surface is the right one
to use, and it does not require any FabricPC change.

### Recipe table — which surface for which metadata

| Metadata                                  | Surface             |
|-------------------------------------------|---------------------|
| Column id (stable, never queried at run)  | Surface 1           |
| Shell id (queried inside energy)          | Surface 2           |
| Task id at graph construction             | Surface 1           |
| Task id at runtime (task switching)       | Surface 3           |
| Support-group membership (graph topology) | Surface 1           |
| Route prior (per-column constant)         | Surface 2           |
| Readout role                              | Surface 1           |
| Diagnostic tag                            | Surface 1           |

### What this strategy costs upstream

Nothing. `NodeBase.__init__(**extra_config)` already exists and already
threads to `node_info.node_config`. Surface 2 is a property of upstream
today.

The codex plan's worry that "if this later becomes painful, a FabricPC
metadata field can be proposed" — is unnecessary. The metadata field is
already there in the form of `node_info.node_config`. The fix is just
documenting it as a public surface and using it.

**Phase 1 documentation extension:** add a paragraph to
`docs/user_guides/06_custom_nodes.md` stating that `**extra_config`
becomes `node_info.node_config` and is the supported channel for
trace-time-static per-node configuration.

## Experiment repo structure

Reduced from the codex plan's ten subpackages to the five that have a
concrete consumer in Phase 2 or Phase 3:

```
columnarCL-fabricPC-experiments/   # branch: main
  pyproject.toml
  README.md
  columnar_cl_fabricpc/
    __init__.py
    columns/
      __init__.py
      nodes.py             # ColumnNode(NodeBase) and friends
    data/
      __init__.py
      cifar.py             # Split-CIFAR-10 task construction
    diagnostics/
      __init__.py
      observers.py         # epoch_callback factories (Solution A)
    experiments/
      __init__.py
      split_cifar10_minimal.py
    utils/
      __init__.py
      metadata.py          # Surface-1 registry
  tests/
    test_imports.py
    test_custom_column_node.py
    test_metadata_registry.py
```

Added when their first module is written:

- `bayes/` — when a Bayesian prior module is added in Phase 4.
- `causal/` — when a causal-coding diagnostic is added in Phase 4.
- `routing/` — when a route-conditioned composer is added in Phase 4.
- `shells/` — when shell-specific shared code (not per-node) is added in
  Phase 4.
- `training/` — only if Solution C is forced.

Empty `__init__.py` files for unused packages are not created up front.

## Implementation phases

Phase ordering:

- Phase 1 (FabricPC contract test and docs) and Phase 2 (experiment repo
  skeleton) are independent and run in parallel.
- Phase 3 (first columnar experiment) depends on both.
- Phase 4 (HiBaCaML-inspired additions) depends on Phase 3 reproducing
  baselines.

### Phase 1 — FabricPC contract test (branch `feature/columnar_CC_adapter`)

1. Add `tests/test_external_custom_node.py`. Define a `NodeBase` subclass
   inside the test file, place it in `graph(...)` with `Edge`, `TaskMap`,
   `IdentityNode`, `Linear`. Run `initialize_params`,
   `initialize_graph_state`, one forward step. No MNIST loader, no
   example coupling.
2. Extend `docs/user_guides/06_custom_nodes.md`: add a section
   "Packaging an external node module" with the `pyproject.toml` shape
   and the recommended import pattern.
3. Extend `docs/user_guides/06_custom_nodes.md`: add a paragraph stating
   that `**extra_config` on `NodeBase.__init__` becomes
   `node_info.node_config` and is the supported channel for trace-time
   static per-node configuration.

No metadata field added. No callback hook added. No columnar code added.
No new mandatory dependencies.

### Phase 2 — experiment repo skeleton (branch `main`)

1. Add `pyproject.toml`. Name: `columnar_cl_fabricpc` (chosen — see
   Alternatives section). Pin dependency floors to match upstream
   FabricPC's `pyproject.toml`: `jax`, `jaxlib`, `optax>=0.1.7`,
   `numpy>=1.24.0`, `flax>=0.7.5`, `chex>=0.1.84`, `jaxtyping>=0.2.23`,
   `tqdm>=4.65.0`. Add `fabricpc` itself (path or git pin to
   `feature/columnar_CC_adapter@10fbe2e`).
2. Create the five subpackages listed above (`columns/`, `data/`,
   `diagnostics/`, `experiments/`, `utils/`). Do not create `bayes/`,
   `causal/`, `routing/`, `shells/`, `training/`.
3. Add `utils/metadata.py` with the Surface-1 registry pattern. Add a
   helper `validate_registry_against_structure(registry, structure)`
   that asserts every node in `structure.nodes` has a registry entry.
4. Add `columns/nodes.py` with a minimal `ColumnNode(NodeBase)` that
   accepts `shell`, `task`, `route_prior` as constructor kwargs and
   stores them via `**extra_config`. Forward computation can be a
   placeholder identity for the smoke test.
5. Add `tests/test_imports.py`, `tests/test_custom_column_node.py`,
   `tests/test_metadata_registry.py`. The custom-node test places a
   `ColumnNode` in a 3-node graph and runs one forward. The registry
   test validates a registry against a built structure.
6. Add `diagnostics/observers.py` with a placeholder
   `make_epoch_callback(...)` factory — populated in Phase 3.

### Phase 3 — first columnar experiment (branch `main`)

Constraints (stated up front as research design choices):

- No replay. **Rationale:** the question Phase 3 is testing is whether
  columnar causal coding alone can close the regime-B gap relative to
  joint training. Adding replay confounds the question. If a future
  question asks "how does columns-plus-replay compare to columns-alone",
  the constraint is lifted and replay is added behind a flag.
- No exemplar storage.
- No pair-label leakage at the readout (the ten-way readout sees only the
  image, not which class pair the batch came from).
- Ten-way output.
- Fixed support structure (one local column per task pair, shared stem).
- Single `--seed` CLI flag.

Steps:

1. `data/cifar.py`: implement Split-CIFAR-10 task construction with
   explicit seed. Returns `(train_loader, test_loader)` per task and a
   `task_classes` mapping.
2. `columns/nodes.py`: extend `ColumnNode` to its real form (the
   placeholder identity is replaced with the column's actual prediction
   computation).
3. `experiments/split_cifar10_minimal.py`: build the graph (stem →
   per-task columns → aggregator → ten-way readout), populate a
   Surface-1 registry, validate the registry against the structure,
   call `initialize_params` + `initialize_graph_state`, train each task
   sequentially with `train_pcn`.
4. `diagnostics/observers.py`: implement Solution A. The
   `epoch_callback` runs `evaluate_pcn` on a held-out diagnostic batch,
   captures `GraphState`, extracts per-node `z_mu`/`error`/`energy`,
   computes per-column activation norms and per-task accuracy. Returns a
   dict to be logged.
5. Report final ten-way accuracy after all tasks, plus per-task
   accuracy at end-of-training, plus average forgetting.
6. Write results to `runs/seed_<N>/summary.jsonl`.

### Phase 4 — HiBaCaML-inspired additions, each gated on a numeric improvement

For each addition, the experiment must state the numeric improvement
expected over the Phase 3 baseline before the change lands. If after one
seed of evaluation the improvement is within the noise band, the change
is dropped (or rebuilt), not kept as inert code.

Order proposed (negotiable):

1. Shared / local column roles via Surface-1 registry + Surface-2
   `role` config on each column node.
2. Concentric shell organization. Shell id flows via Surface-2 into
   each column's `forward`; shell-aware energy weighting tested.
3. Causal-coding diagnostics. New `causal/` subpackage with an
   epoch-level diagnostic; uses Solution A or B.
4. Bayesian route priors. New `bayes/` subpackage; priors flow via
   Surface-2 into the route-conditioned composer.
5. Uncertainty-aware support scoring. Evaluates support-set members
   against a posterior; ranks them.
6. Calibrated readout experiments. Compares ten-way readouts under
   different calibration objectives.

Each addition follows the same pattern: read static config from
`node_info.node_config`, register node-name → role in
`utils/metadata.py`, observe via `epoch_callback`.

## Alternatives considered

Per the experiment repo's `AGENTS.md`, every load-bearing decision lists
the alternatives with pros and cons.

### Decision: scope of FabricPC-side work

- **Alt A — full six minimal hooks (the original codex minimal plan):**
  metadata field, training callback (new), packaged JAX setup,
  packaging extras, import stability, node-extension contract.
  - *Pro:* documented infrastructure for all experiments.
  - *Con:* most of it already exists; the new metadata field is
    redundant with `node_info.node_config`; the new training callback is
    redundant with existing `epoch_callback`/`iter_callback`.
- **Alt B — one hook (the even-more-minimal plan):** only the
  node-extension contract.
  - *Pro:* smallest upstream surface.
  - *Con:* misses the documentation gap on `node_info.node_config` as
    a public metadata surface. Leaves experiment authors to discover
    existing callbacks on their own.
- **Alt C (chosen) — one hook (the contract test) plus two
  documentation extensions:** documenting `**extra_config` →
  `node_info.node_config` as the metadata surface, and documenting
  `epoch_callback` / `iter_callback` as the training observation
  surface.
  - *Pro:* no new code in FabricPC. Two documentation paragraphs and
    one test.
  - *Pro:* makes existing surfaces discoverable; future experiments do
    not re-implement what already works.
  - *Con:* requires writing the docs carefully so they do not over-
    promise.

### Decision: metadata representation

- **Alt A — `dict` registry keyed by node name (codex even-more-minimal
  plan):**
  - *Pro:* simplest possible code.
  - *Con:* not consultable inside `jax.jit`; fails the first time a
    routing or shell decision is needed inside `forward`.
- **Alt B (chosen for graph-construction-time) — same `dict` registry,
  but used only at graph construction:**
  - *Pro:* the jit-incompat problem disappears because the registry is
    never consulted at trace time.
  - *Con:* a stricter contract; the registry validator must enforce
    it, otherwise authors will reach for the registry inside forwards
    by habit.
- **Alt C (chosen for trace-time-static) —
  `node_info.node_config` via `NodeBase.__init__(**extra_config)`:**
  - *Pro:* already exists upstream; jit-safe because reads at trace
    time specialize the compiled forward.
  - *Con:* requires documenting the surface so authors know it is the
    intended channel.
- **Alt D — a new `metadata: dict` field on `NodeInfo` upstream:**
  - *Pro:* explicit, dedicated channel.
  - *Con:* duplicates `node_info.node_config` for no gain; adds
    upstream surface to maintain.

Recipe: Alt B for everything that controls graph construction, Alt C for
everything that the forward needs to read. Alt D rejected as redundant
with what already exists.

### Decision: training observation

- **Alt A — Solution A from the "Training observation strategy"
  section (chosen for Phase 3):** `epoch_callback` plus a state-capture
  re-forward.
- **Alt B — Solution B (deferred to Phase 4):** add `iter_callback`
  sampling for sub-epoch resolution.
- **Alt C — Solution C (deferred indefinitely):** own the training
  loop locally; add an `infer_step_callback`.

Per "Training observation strategy" above.

### Decision: experiment package naming

- **Alt A — `columnar_cl_fabricpc` (chosen):** verbose at import sites
  but unambiguous in cross-repo grep.
- **Alt B — `cclfpc`:** four-letter abbreviation; harder to grep,
  easier to type.
- **Alt C — `columnar_cl`:** drops the `fabricpc` suffix; reads
  cleaner but loses the "built on FabricPC" signal.
- **Alt D — `colcc`:** five letters; ambiguous.

`columnar_cl_fabricpc` chosen because cross-repo greppability outweighs
typing cost, and the suffix signals dependency direction to future
readers.

### Decision: experiment scope for Phase 3

- **Alt A — full HiBaCaML pipeline as Phase 3:** Bayesian priors,
  shells, causal diagnostics from day one.
  - *Con:* Phase 3 becomes unverifiable. No clean baseline to gate
    Phase 4 against.
- **Alt B (chosen) — minimal columnar Phase 3, then layered Phase 4
  additions each gated on numeric improvement:**
  - *Pro:* Phase 3 reproduces the cFabricPC baseline as a falsifiable
    floor.
  - *Pro:* Phase 4 additions are individually attributable to specific
    improvements.

## Reproducibility thresholds

Phase 3 acceptance requires reproducing the prior cFabricPC multi-seed
numbers within explicit bands. Baseline numbers are from cFabricPC's
multi-seed validation across seeds 42, 7, 99 (recorded in the prior
plan reviews).

| Metric                                  | cFabricPC value (mean ± std) | Phase 3 acceptance band |
|-----------------------------------------|------------------------------|--------------------------|
| Regime-A (masked-pair, final)           | 0.8242 ± 0.0534 (disjoint)   | reproduce within ± 0.020 of mean |
| Regime-B (ten-way, final)               | 0.3541 ± 0.0362 (disjoint)   | reproduce within ± 0.025 of mean |
| Joint ceiling (all-aggregator readout)  | 0.6424                       | reproduce within ± 0.020         |
| Forgetting (averaged)                   | task-dependent baseline      | not regressed beyond ± 0.020     |

If a Phase 3 run cannot reproduce these on seed 42, the divergence is
the first finding to investigate; Phase 4 does not begin until the
baseline is reproduced.

## Acceptance criteria

### FabricPC (branch `feature/columnar_CC_adapter`)

1. `pytest tests/test_external_custom_node.py` passes from a clean
   environment.
2. `examples/custom_node.py` still runs end-to-end and reaches ≥ 0.90
   MNIST test accuracy.
3. No new files under `fabricpc/continual/`, `fabricpc/builder/`,
   `fabricpc/graph/`, or anywhere that mentions columns, tasks, or
   CIFAR.
4. `pyproject.toml` base dependencies unchanged.
5. `docs/user_guides/06_custom_nodes.md` has the new "Packaging an
   external node module" section and the new
   `node_info.node_config` documentation paragraph.

### Experiment repo (branch `main`)

1. `pytest tests/` passes: imports, custom column node placed in a
   3-node graph with one forward step, metadata-registry validation.
2. `python examples/split_cifar10_minimal.py --seed 42` produces a
   `runs/seed_42/summary.jsonl` with final ten-way accuracy.
3. Numeric thresholds per the reproducibility table above.
4. `python examples/split_cifar10_minimal.py --seed 7` and `--seed 99`
   reproduce the cross-seed bands.
5. No file under `columnar_cl_fabricpc/` contains the substring
   `replay`, `exemplar`, or `memory_buffer` (asserts the no-replay
   constraint at lint time).
6. Every node in the built `GraphStructure` has a matching entry in
   the metadata registry (asserted by a unit test).

## Risk register

| Risk                                                           | Likelihood | Impact                                                      | Mitigation                                                  |
|----------------------------------------------------------------|------------|-------------------------------------------------------------|-------------------------------------------------------------|
| Experiment author reaches for metadata `dict` inside a forward | High       | jit-incompat runtime error; surface confusion               | Registry validator only runs at graph-construction time; documentation makes Surface 2 prominent |
| Experiment author re-implements `train_pcn` callbacks          | Medium     | Duplicate observation code, divergent metrics               | Phase 3 ships Solution A as a reference observer            |
| Renamed node desyncs Surface-1 registry                        | Medium     | Silent bug; diagnostics attribute behavior to wrong node    | `validate_registry_against_structure` asserts in tests      |
| Upstream FabricPC rebases past `10fbe2e`                       | Medium     | Phase 1 work needs re-pinning                               | Plan revision required for any pin change                   |
| `no replay` constraint hides regime-B ceiling cause            | Medium     | Phase 4 spent chasing a research-design-imposed bound       | Constraint is revisitable in Phase 4 with stated rationale  |
| Premature subpackage scaffolding                               | Low        | Dead `__init__.py` files accumulate                         | Five-subpackage cap stated up front                         |
| Phase 4 additions kept as inert code despite no improvement    | Medium     | Codebase entropy                                            | Numeric gate per addition; failed gate ⇒ drop, not keep     |

## Deferred items with explicit revisit triggers

Each deferred item lists the condition under which the decision is
re-opened. Without an explicit trigger, deferral becomes invisible.

| Deferred                                | Revisit when                                                                 |
|-----------------------------------------|-------------------------------------------------------------------------------|
| New FabricPC metadata field             | A forward path needs Surface-3 (runtime-varying) metadata *and* the input-clamp pattern proves insufficient. |
| New FabricPC training observer hook     | Solution A and Solution B both prove insufficient on a specific diagnostic.   |
| Replay / exemplar / memory buffer       | Phase 4 columnar additions plateau below the joint ceiling and the question shifts to "how much of the gap is the no-replay constraint." |
| Lint / formatter / type-checker         | Experiment repo crosses ~2000 lines.                                          |
| Checkpoint storage policy               | A Phase 3 or Phase 4 result takes > 1 hour wall-clock to reproduce.           |
| Migration of additional cFabricPC scripts | A specific cFabricPC script (e.g., `prototype_geometry_report.py`) is needed to validate a Phase 4 addition. |

## One-page summary

What changes vs. the codex even-more-minimal plan:

- Phase 1 deliverable shrinks to one new test plus two documentation
  paragraphs. The custom-node contract, the example, the guide, and the
  training callbacks are already in upstream.
- The metadata strategy is split: graph-construction-time metadata lives
  in a Python-side registry (Surface 1, as the codex plan proposed),
  trace-time-static metadata lives in `node_info.node_config` via
  `NodeBase.__init__(**extra_config)` (Surface 2, already supported
  upstream — needs documenting, not adding). This fixes the
  jit-incompatibility hole in the codex plan.
- The training observation strategy is split: Solution A
  (`epoch_callback` plus state-capture re-forward) for Phase 3,
  Solution B (`iter_callback` sub-epoch sampling) deferred to Phase 4 if
  needed, Solution C (own the loop) deferred indefinitely. This makes
  use of `train_pcn`'s existing callbacks instead of re-implementing them.
- The experiment repo skeleton shrinks from ten subpackages to five.
  Others land when their first module is written.
- Every load-bearing decision lists alternatives with pros/cons, per
  the experiment repo's own `AGENTS.md`.
- Repos and branches are pinned explicitly.
- Reproducibility thresholds tie Phase 3 acceptance to the prior
  cFabricPC multi-seed numbers.
- Phase 4 additions are individually gated on numeric improvement.

What stays the same:

- Boundary policy: FabricPC general, columnar code in the experiment
  repo, no `fabricpc.continual`.
- No-replay constraint as a Phase 3 research design choice.
- Use upstream FabricPC's native imports (`fabricpc.core.topology.Edge`,
  `fabricpc.graph_assembly.{TaskMap, graph}`,
  `fabricpc.graph_initialization.*`); no shims for the cFabricPC fork's
  namespace.
- cFabricPC remains read-only reference, not migrated as files.
