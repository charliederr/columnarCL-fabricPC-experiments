# Critical Review: codex-even-more-minimal-fabricpc-hooks-plan.md

Augmented review of `codex-even-more-minimal-fabricpc-hooks-plan.md`, written
alongside it. The plan proposes reducing FabricPC-side changes to a single
item — an external custom-node extension contract — and pushing everything
else (metadata, training observation, JAX setup, columnar/Bayesian/causal code)
into the experiment repo.

The direction is sound. Several premises in the plan, however, do not match
ground truth in `FabricPC/`, several decisions are not enumerated against
alternatives the way the project's own AGENTS.md requires, and several deferred
items are deferred without falsifiable triggers for revisiting them.

## How to read this document

This review is a section-by-section augmentation, not a rewrite. Where the plan
is right, that is stated and not re-derived. Where the plan is wrong against
the actual state of the three repositories, the correction is given with file
paths. Where the plan is a judgment call without alternatives, the alternatives
and the cost of each are stated explicitly.

The relationship between documents:

- `codex-minimal-fabricpc-hooks-plan.md` proposed six FabricPC hooks
  (import stability, node-extension contract, graph metadata, training observer,
  JAX setup helper, packaging extras).
- `codex-even-more-minimal-fabricpc-hooks-plan.md` keeps only one of those six
  (the node-extension contract) and pushes the other five into the experiment
  repository.
- This review evaluates whether the "even more minimal" reduction holds up
  against what is actually in the upstream FabricPC tree today.

## Ground-truth snapshot

This is what the three repositories contain right now, gathered by direct
inspection. It is necessary because the plan's deliverables in several places
assume things that do not exist, or do not assume things that already exist.

### FabricPC (upstream) — what already exists

The plan treats Phase 1's deliverable ("add a minimal external custom-node
example or test") as new work. Most of it is already in upstream FabricPC:

- `fabricpc/nodes/base.py` defines `NodeBase` as an explicit `ABC`. Three
  abstract `@staticmethod`s — `get_slots`, `initialize_params`, `forward` —
  define the extension contract. The module docstring (lines 8–32) shows the
  exact subclassing pattern an external package would use. The `forward`
  docstring (lines 304–355) enumerates the six steps every subclass must
  perform.
- `examples/custom_node.py` defines `Conv2DNode(NodeBase)` end-to-end —
  `get_slots`, `initialize_params`, `forward`, `get_weight_fan_in` — and trains
  it on MNIST through `fabricpc.training.train_pcn`. This is a working
  external-node demonstration today.
- `docs/user_guides/06_custom_nodes.md` is a written guide for custom nodes.
- `fabricpc.training.train_pcn` already accepts `epoch_callback` and
  `iter_callback` parameters
  (`fabricpc/training/train.py:293–294, 313, 434, 451–460`). The "training
  callback hook" the prior plan asked for and this plan removes
  *already exists*.
- `fabricpc/utils/dashboarding/{callbacks.py, trackers.py, extractors.py}`
  provides callback factories integrated with the Aim experiment tracker.
  Substantial observation infrastructure is already present.
- `fabricpc/utils/data/` provides `MnistLoader` and friends.
- `jax_setup.py` exists at the FabricPC repository root, not as a package
  submodule. It is currently importable only when the cwd contains it, e.g.
  `from jax_setup import set_jax_flags_before_importing_jax` works from inside
  `FabricPC/` but fails from anywhere else. There is no
  `fabricpc.utils.jax_setup` module.
- `pyproject.toml` declares `name = "fabricpc"`, version `0.3.1`, and
  optional extras `[dev, tfds, experiments, viz]`. Base dependencies already
  include `jax`, `jaxlib`, `optax`, `flax`, `orbax-checkpoint`, `chex`,
  `jaxtyping`, `numpy`, `tqdm`, `optuna`. Backend extras `cuda12` and `cuda13`
  are present (per the prior plan).
- `tests/` contains `test_fabricpc.py`, `test_state_initializer.py`,
  `test_initializers.py`, `test_train_backprop.py`, `test_ndim_shapes.py`,
  and others — but no test that exercises a node defined in an external
  package.

### Experiment repo — what exists

- `columnarCL-fabricPC-experiments/` contains only `AGENTS.md`, `CLAUDE.md`,
  `LICENSE`, `README.md`, and `docs/`. No package, no `pyproject.toml`, no
  source files, no tests.
- `docs/` has the two codex plans plus this review, the earlier
  `claude-crit-review-migration-plan.md`, `claude-improvement-planning.md`,
  `codex-critical-review-migration-plan.md`, and `migration-plan.md`.

### Virtual environments

The plan repeatedly targets `/home/ni/repos/fpc/virt-envs/fpcpy3.12`. That venv
exists but currently has no `fabricpc` installed:

```
$ /home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -c "import fabricpc"
ModuleNotFoundError: No module named 'fabricpc'
```

`ffpc7` already has `fabricpc` editable-installed, but it resolves to the
cFabricPC fork at `/home/ni/repos/fpc/cFabricPC/fabricpc/__init__.py`, not to
the upstream tree at `/home/ni/repos/fpc/FabricPC/`. This is the shadowing
hazard the prior migration review flagged; it applies equally here.

## Issues, ranked by severity

Numbered for cross-reference. Severity reflects how badly the issue would
mis-shape the work if left unaddressed.

### Critical

1. **Phase 1 deliverable is largely already done.** The plan's "FabricPC
   Custom Node Proof" lists five steps: minimal example, importable from
   outside, placeable in a graph, params/state initialization, forward pass.
   Items 1, 3, 4, 5 are demonstrated by `examples/custom_node.py` today; item 2
   (importable from a separate installed package) is the only ungrounded one.
   The deliverable should be re-scoped as: (a) add a test that imports a node
   from a package other than `fabricpc` and uses it in a `graph(...)` call,
   (b) cross-link `06_custom_nodes.md` from the README, (c) confirm
   `examples/custom_node.py` can be turned into a test fixture without coupling
   it to `MnistLoader`. Stating Phase 1 as "add" when it is "verify and
   harden" hides that most of the work is acceptance-criterion design, not
   implementation.

2. **The plan removes a training observation hook that already exists in
   upstream.** Section *Training Observation Without FabricPC Changes* says the
   experiment loop will "call FabricPC graph initialization and forward/state
   functions, then collect" diagnostics — implying the experiment must own the
   loop. But `train_pcn` already takes `epoch_callback` and `iter_callback`
   parameters (`train.py:293–294`), and `fabricpc/utils/dashboarding/callbacks.py`
   already wires them to a tracker. The plan should either (a) recommend using
   the existing callbacks and state what they are sufficient for, (b) state
   what they are insufficient for (e.g., per-node activation extraction
   inside the inference loop, not at epoch boundaries), or (c) commit to
   not using `train_pcn` and owning a parallel loop — but it must pick one and
   say why. Silence here will lead to the experiment repo silently
   re-implementing what is already callable.

3. **Target venv is `fpcpy3.12` but the venv that currently resolves
   `fabricpc` is `ffpc7`, and it points at cFabricPC.** Acceptance criterion
   "it installs in `/home/ni/repos/fpc/virt-envs/fpcpy3.12`" is satisfiable
   only if (a) upstream FabricPC is editable-installed there first, and
   (b) no cFabricPC fabricpc package is on the path. Acceptance criterion
   "`import fabricpc` resolves to `/home/ni/repos/fpc/FabricPC`" is
   technically a different test, and both are needed. The plan should add an
   explicit Phase 0: choose between (i) install upstream FabricPC into the
   empty `fpcpy3.12` venv and verify nothing else is shadowing, or
   (ii) uninstall cFabricPC's `fabricpc` from `ffpc7` and install upstream
   there. Both are valid; one must be picked before any other phase.

### High

4. **The plan dismisses the graph-metadata hook without naming the
   alternative's cost.** The substitute is "external registry keyed by node
   name." That works for graph-construction-time queries
   (`node_roles["col_07"]["shell"] == 1`). It does not work inside a
   `jax.jit`-compiled forward pass, because a Python `dict` lookup against a
   traced node identity is not traceable. If any of the proposed columnar
   behavior — support-conditioned composers, route gating, shell-aware energy
   weighting — needs metadata *inside* the forward path, the registry pattern
   is unusable there and the plan will hit this on first contact. The plan
   should commit to one of: (a) all metadata use is graph-construction-time
   only (state the constraint and design columns around it),
   (b) metadata flows through `node_info.node_config` already supported by
   `NodeBase.__init__`'s `**extra_config` passthrough (state how),
   or (c) FabricPC needs a `node_info.metadata` field after all (state the
   falsifiable trigger that will reopen this decision).

5. **The plan does not list alternatives considered.** The repository's own
   `AGENTS.md` reads "Planning documents always list the alternative approaches
   considered, with pros/cons for each, alongside the chosen approach."
   For each of the five hooks the plan removes from FabricPC, the plan should
   state: what was the original motivation, what does removing it cost, what
   condition would justify adding it back. As written, the plan reads as a
   one-way reduction without showing the trade-off.

6. **Pre-built eight-subpackage skeleton contradicts the plan's own "small
   first experiment" rule.** The proposed `columnar_cl_fabricpc/` layout has
   `bayes/`, `causal/`, `columns/`, `data/`, `diagnostics/`, `experiments/`,
   `routing/`, `shells/`, `training/`, `utils/` — ten top-level packages.
   Phase 3 ("First Clean Columnar Experiment") uses at most three of these.
   Empty `__init__.py` files in the other seven are speculative scaffolding
   that locks in a structure before the experiment has told you what it
   needs. Recommend: create only `data/`, `columns/`, `utils/`,
   `experiments/`, `tests/` for Phase 2. Add the others when the first
   module under them is written.

7. **No alternative naming considered for the experiment package.**
   `columnar_cl_fabricpc` is verbose at import sites
   (`from columnar_cl_fabricpc.columns.nodes import ...`). Alternatives —
   `cclfpc`, `cclf`, `columnar_cl`, `colcc` — should be enumerated. The
   relevant trade-off is: long enough to be unambiguous in cross-repo grep,
   short enough to type at every import. A two-line decision note in the
   plan resolves this once.

### Medium

8. **No mention of `fabricpc.utils.dashboarding`.** This package is the most
   substantial piece of existing observation infrastructure in upstream. The
   plan should state explicitly whether the experiment repo will use it,
   wrap it, ignore it, or supersede it with its own diagnostics module.
   Silence means the experiment author has to discover and re-decide this on
   contact.

9. **The plan places JAX setup at `columnar_cl_fabricpc/utils/jax_setup.py`
   without acknowledging the existing upstream helper at
   `FabricPC/jax_setup.py`.** The upstream helper currently lives at the
   repository root, not inside the `fabricpc` package, so it is not
   reachable as `from fabricpc.utils.jax_setup import ...`. The experiment
   repo helper could (a) be a clean re-implementation that does not depend
   on the upstream's path, or (b) be a thin re-export
   `from jax_setup import set_jax_flags_before_importing_jax`. Option (b) is
   fragile (path-dependent); option (a) is a small duplication. The plan
   should pick (a) and state why duplication is acceptable here.

10. **"No replay or stored-exemplar path exists" is treated as architecture
    when it is a research design choice.** The prior `cFabricPC` work showed
    that no-replay continual learning has a measurable accuracy ceiling
    relative to memory-augmented variants (regime B at ~0.30–0.35 vs. joint
    ceiling 0.64+). Stating "no replay" as a property of the experiment repo
    rules out a class of investigations the user may want later. The plan
    should state this as an *explicit Phase 3 constraint* with rationale
    ("we are testing whether columnar causal coding alone can close the
    regime-B gap; adding replay would confound the question"), not as a
    global property of the repo.

11. **Phase ordering: Phase 1 is a precondition for Phase 2, but does not
    need to be.** The custom-node contract is already in upstream FabricPC,
    so the experiment scaffold (Phase 2) can be built in parallel with the
    Phase 1 deliverable (which is mostly test-and-document). Recommend
    re-ordering as: Phase 0 (venv hygiene), Phase 1 and Phase 2 in parallel,
    Phase 3 once both land.

12. **FabricPC source-of-truth commit pin is not specified.** Upstream's
    active branch (per the prior migration review) is
    `feature/columnar_CC_adapter` at commit `10fbe2e`. Pinning is necessary
    so that "Phase 1 work in FabricPC" is reproducible against a known
    starting point. Drop a single line in the plan: "Source pin: FabricPC
    branch `feature/columnar_CC_adapter`, commit `10fbe2e`."

13. **Phase 4 lists six HiBaCaML-inspired additions but no acceptance
    criterion for any.** "Add concentric shell organization", "Add causal
    coding diagnostics", "Add Bayesian route priors" — each needs at least
    a numeric falsifiable threshold (final accuracy band, forgetting band,
    calibration slope, route-stability score). Without these the phase is
    open-ended scope.

### Low

14. **Acceptance criterion "an external custom node test passes" does not
    say where the test lives.** Both repos are valid hosts; the call
    affects who owns the regression risk. Recommend FabricPC hosts the
    contract test (since it owns the contract), and the experiment repo
    hosts a smoke test that uses a column node (since it owns the columns).

15. **No reproducibility / seed policy stated.** The `AGENTS.md` rule
    "Every stochastic entry point takes an explicit seed" needs an entry
    in Phase 3 ("data, model init, training all derive from a single
    `--seed` CLI flag").

16. **No version constraints on the dependency list.** `jax`, `jaxlib`,
    `optax`, `numpy` are listed without floors. Upstream FabricPC pins
    `optax >= 0.1.7`, `numpy >= 1.24.0`, etc. Mismatched floors will cause
    a divergent resolution; the experiment repo should inherit FabricPC's
    floors at minimum.

17. **No checkpoint / artifact storage policy.** Same gap as the prior
    migration plan. If any Phase 3 result requires a saved checkpoint to
    reproduce, the plan should state where it lives (gitignored, LFS,
    external URL, document-only).

18. **The "Bottom Line" section duplicates the "Feasibility Answer" opening
    and the "Only FabricPC Work" header.** Three statements of the same
    idea. Recommend deleting "Bottom Line"; the document already opens with
    the answer.

19. **The plan opens with "Yes" without quoting the question.** The
    referenced question is "can we reduce required FabricPC changes to only
    the custom-node extension contract from Section 2 of the minimal hooks
    plan." Restate it in the first paragraph so the document is readable
    standalone.

20. **No lint / formatter / type-checker discipline stated.** Same gap as
    the prior plans. Defer is fine; the plan should at least raise the
    decision.

## Alternatives the plan does not enumerate

Per the repository's own `AGENTS.md`, plans list alternative approaches
with pros and cons. The plan picks reductionism without showing the menu.

### Alternative A: keep all six minimal hooks, ship them as the contract
- **Pro:** experiment authors get metadata, callbacks, and JAX setup as
  documented API. No re-implementation drift. Future experiments share
  infrastructure.
- **Pro:** FabricPC becomes the natural place for general predictive-coding
  infrastructure (it already has `train_pcn` callbacks; metadata fields
  and a packaged `jax_setup` are small additions).
- **Con:** more upstream churn before any experiment is written.
- **Con:** risk of upstream API coupling to one specific experiment's needs.

### Alternative B: the even-more-minimal plan (chosen)
- **Pro:** smallest upstream surface to maintain.
- **Pro:** experiment repo free to evolve without coordinating PRs.
- **Con:** silently re-implements `train_pcn` callbacks if experiment authors
  do not notice them.
- **Con:** node-name-keyed metadata registry is non-jit-compatible and will
  break the first time a forward path needs to consult it.
- **Con:** duplicates `jax_setup.py` functionality already present upstream.

### Alternative C: middle path — keep 2 of the 6
- Keep: (i) node-extension contract (it is the only one with a real upstream
  gap — the external-package test), and (ii) `fabricpc.utils.jax_setup`
  package-level helper (it is a 20-line change that fixes a real packaging
  bug — `jax_setup.py` is not currently importable from outside FabricPC's cwd).
- Drop: metadata, training callback (already exists), packaging extras
  (already exists).
- **Pro:** fixes one real upstream packaging bug (jax_setup path) without
  taking on experiment-specific scope.
- **Pro:** acknowledges what is already there (callbacks) and what is not
  (importable jax_setup).
- **Con:** very small upstream PR, but at least it is honest about what is
  missing.

The plan as written is closest to B. A defensible final form would be B
plus the jax_setup packaging fix from C, plus an explicit note that
existing `train_pcn` callbacks are the recommended observation path until
they prove insufficient.

## Concrete next-step list (replacing the plan's Phase 1/2/3/4)

### Phase 0 — venv hygiene (required, blocks everything else)

1. Decide between:
   - **(0a)** Install upstream FabricPC editable into the empty
     `/home/ni/repos/fpc/virt-envs/fpcpy3.12`. Verify nothing shadows.
   - **(0b)** Uninstall cFabricPC's `fabricpc` from
     `/home/ni/repos/fpc/virt-envs/ffpc7`, then `pip install -e
     /home/ni/repos/fpc/FabricPC` into ffpc7.
2. Acceptance: `python -c "import fabricpc; print(fabricpc.__file__)"`
   resolves to `/home/ni/repos/fpc/FabricPC/fabricpc/__init__.py` in the
   chosen venv.
3. Pin upstream FabricPC branch and commit hash in a new
   `columnarCL-fabricPC-experiments/README.md` section.

### Phase 1 — FabricPC contract test (in parallel with Phase 2)

Re-scope from "add" to "verify":

1. Add `tests/test_external_custom_node.py` to FabricPC. The test should
   import a `NodeBase` subclass defined in *this test file*, place it in a
   `graph(...)` call with `Edge`, `TaskMap`, `IdentityNode`, `Linear`,
   run `initialize_params` and `initialize_graph_state`, run one forward
   step. No MNIST dependency, no `fabricpc.utils.data`.
2. Add `docs/user_guides/14_external_node_packaging.md` (or extend
   `06_custom_nodes.md`) showing how an external package would declare a
   `pyproject.toml` and ship a `NodeBase` subclass.
3. Decide on the jax_setup packaging fix (Alternative C above): either
   create `fabricpc/utils/jax_setup.py` that re-exports the root helper, or
   skip and document the workaround.
4. Do not add metadata fields, do not add callback hooks (they exist
   already), do not add columnar code.

### Phase 2 — experiment repo skeleton (in parallel with Phase 1)

Reduced subpackage list:

1. Add `pyproject.toml` with name `columnar_cl_fabricpc` (or shorter, after
   the naming decision is made). Pin dependencies to match FabricPC's
   floors.
2. Create only `columnar_cl_fabricpc/{__init__.py, columns/__init__.py,
   data/__init__.py, experiments/__init__.py, utils/__init__.py}`. Add
   `tests/` with `test_imports.py`. No `bayes/`, `causal/`, `routing/`,
   `shells/`, `diagnostics/`, `training/` — those land when their first
   module is written.
3. Add `utils/jax_setup.py` (clean re-implementation, not import from
   FabricPC's root `jax_setup.py`).
4. Add `utils/metadata.py`: a plain dict-of-dicts registry keyed by node
   name. **Document the constraint that this is graph-construction-time
   only — not consultable inside a `jax.jit`-compiled forward.**
5. Add `columns/example_node.py`: a minimal `ColumnNode(NodeBase)` that
   subclasses upstream `NodeBase`. This is the smoke test, not a real
   column.

### Phase 3 — first columnar experiment

1. Define the experiment in `experiments/split_cifar10_minimal.py`.
   Constraints to state up front: no replay, no exemplar storage, no
   pair-label leakage at the readout, ten-way output, fixed support
   structure, single seed CLI flag.
2. Use `train_pcn` with `iter_callback` for batch-level diagnostics and
   `epoch_callback` for per-task accuracy/forgetting. Do *not* roll a
   custom training loop unless `train_pcn`'s callback signature is
   shown to be insufficient.
3. Define numeric acceptance: report final ten-way accuracy after five
   tasks, with explicit seed. State a baseline number (the prior
   cFabricPC work's no-replay ceiling) so the experiment has something to
   improve against.
4. Defer all HiBaCaML-inspired modules (Bayesian priors, shells, causal
   diagnostics) until this minimal experiment is reproducible.

### Phase 4 — HiBaCaML-inspired ideas (numeric gates required)

For each addition (shared/local roles, shells, causal diagnostics,
Bayesian priors, uncertainty-aware support, calibrated readout):

1. State the falsifiable numeric improvement expected over the Phase 3
   baseline.
2. If after one seed of evaluation the improvement is within the noise
   band of the baseline, drop the change or rebuild it; do not keep it
   in the tree as inert code.

## Reproducibility threshold table

The plan has no numeric acceptance criteria. A minimum set, based on
prior cFabricPC numbers (multi-seed validation across seeds 42, 7, 99):

| Metric                                  | Prior cFabricPC value      | Acceptance band for Phase 3 |
|-----------------------------------------|----------------------------|-----------------------------|
| Regime-A (masked pair, final)           | 0.8242 ± 0.0534 (disjoint) | reproduce within ± 0.020    |
| Regime-B (ten-way, final)               | 0.3541 ± 0.0362 (disjoint) | reproduce within ± 0.025    |
| Joint ceiling (all-aggregator readout)  | 0.6424                     | reproduce within ± 0.020    |
| Forgetting (averaged)                   | task-dependent             | not regressed beyond ± 0.020|

These numbers are the targets the experiment repo's first-experiment script
must meet on `seed=42` to be considered to have successfully reproduced the
cFabricPC pre-existing baseline. If they cannot be reproduced, the
divergence is itself the first finding to investigate.

## Risk register

| Risk                                                          | Likelihood | Impact                                                     |
|---------------------------------------------------------------|------------|------------------------------------------------------------|
| `fpcpy3.12` venv ends up with both upstream and cFabricPC on path | High       | `import fabricpc` resolves nondeterministically            |
| Experiment author re-implements `train_pcn` callbacks         | High       | Duplicate observation code, divergent metrics              |
| Node-name-keyed metadata fails inside jit                     | Medium     | Plan revision needed mid-Phase-4                           |
| Renamed node desyncs external registry                        | Medium     | Silent bug; diagnostics attribute behavior to wrong node   |
| No source pin on FabricPC commit                              | Medium     | Phase 1 work irreproducible after upstream rebases        |
| Eight pre-built subpackages outlive their justification       | Low        | Dead `__init__.py` files accumulate                        |
| `no replay` constraint hides regime-B ceiling cause           | Medium     | Time spent in Phase 4 chasing an architectural bound       |

## Items the plan defers without saying so

These are decisions the plan implicitly defers. Each should be deferred
explicitly with a stated condition for revisiting.

1. **Graph metadata as a FabricPC field.** Trigger to revisit: any forward
   path needs to consult column/shell/support id during inference.
2. **Training callback hook in FabricPC.** Already exists; revisit only if
   per-inference-step extraction is needed and `epoch_callback`/`iter_callback`
   are insufficient.
3. **Packaged JAX setup helper in FabricPC.** Revisit if the experiment
   repo's local `utils/jax_setup.py` diverges from FabricPC's root
   `jax_setup.py` in a way that causes a confusing flag-mismatch bug.
4. **Replay mechanisms.** Revisit if Phase 4's columnar additions plateau
   below the joint ceiling and the user wants to test the floor of what
   columns alone can achieve.
5. **Lint / formatter / type-checker discipline.** Revisit at the start
   of Phase 4 when the codebase first crosses ~2000 lines.
6. **Checkpoint storage.** Revisit when Phase 3 produces a result that
   would be expensive to recompute (>1 hour wall-clock).

## Updated acceptance criteria

Replacing the plan's "FabricPC is sufficiently prepared when" and
"experiment repo is ready when" lists with numerically falsifiable items.

### FabricPC

1. `pytest tests/test_external_custom_node.py` passes from a clean venv
   with FabricPC editable-installed and no `cFabricPC` on `sys.path`.
2. `examples/custom_node.py` runs end-to-end and reaches ≥ 0.90 MNIST
   test accuracy.
3. No new files under `fabricpc/continual/`, `fabricpc/builder/`,
   `fabricpc/graph/`, or anywhere that mentions columns, tasks, or CIFAR.
4. `pyproject.toml` base dependencies unchanged (no new mandatory deps).
5. If the jax_setup packaging fix lands: `from fabricpc.utils.jax_setup
   import set_jax_flags_before_importing_jax` works from the
   `fpcpy3.12` venv.

### Experiment repo

1. `pip install -e columnarCL-fabricPC-experiments` succeeds in
   `fpcpy3.12`.
2. `python -c "import columnar_cl_fabricpc; print(columnar_cl_fabricpc.__file__)"`
   resolves to `columnarCL-fabricPC-experiments/columnar_cl_fabricpc/__init__.py`.
3. `python -c "import fabricpc; print(fabricpc.__file__)"`
   resolves to `/home/ni/repos/fpc/FabricPC/fabricpc/__init__.py`.
4. `pytest tests/` passes (smoke: imports, one custom column node placed
   in a 3-node graph with one `Edge`, one forward step).
5. `python examples/split_cifar10_minimal.py --seed 42` produces a
   `summary.jsonl` with final ten-way accuracy. Numeric thresholds per
   the reproducibility table above.
6. No file under `columnar_cl_fabricpc/` contains the substring
   `replay`, `exemplar`, or `memory_buffer` (asserts the no-replay
   constraint at lint time).

## One-page summary

The plan is right that the FabricPC-side change can be reduced to almost
nothing — but for a different reason than the plan states. The reason is
that almost all the proposed hooks *already exist* in upstream:

- The custom-node extension contract is in
  `fabricpc/nodes/base.py:NodeBase` + `examples/custom_node.py` +
  `docs/user_guides/06_custom_nodes.md` today. Phase 1 is verification
  and an external-package test, not new work.
- The training observation hook is in `train_pcn`'s `epoch_callback` and
  `iter_callback` parameters today, plus the dashboarding callback
  factories. The plan's "experiment repo owns its training loop"
  premise should be conditional on these being insufficient.
- The JAX setup helper exists at `FabricPC/jax_setup.py` but is at the
  wrong path to be importable from an external package. This is the one
  real upstream packaging bug the plan could fix; instead it sidesteps
  it by duplicating the helper in the experiment repo.

The plan's reductionism is therefore directionally correct but
empirically mis-targeted. The right minimum is: (a) one upstream test
verifying the external-node contract works from a package other than
`fabricpc`, (b) one upstream packaging fix to make `jax_setup` importable
as `fabricpc.utils.jax_setup`, (c) experiment repo skeleton with five
subpackages (not ten), (d) explicit decision to use `train_pcn` callbacks
rather than re-implement, (e) Phase 0 venv hygiene step, (f) source pin on
upstream FabricPC commit, (g) reproducibility thresholds tied to prior
cFabricPC multi-seed numbers, (h) explicit no-replay constraint stated as
a research design choice, not as a property of the repo.

Sections of the plan to delete or rewrite:

- "Bottom Line" duplicates the opening; delete.
- Phase 1 "add" verbs become "verify"; rewrite.
- Eight-subpackage skeleton reduced to five; rewrite.
- "Training observation can live in experiment-owned training loops" needs
  the "if `train_pcn` callbacks are insufficient" clause; rewrite.
- "External registry keyed by node name" needs the jit-compatibility
  constraint; rewrite.

What is right and should be kept:

- The boundary policy (FabricPC is general; columnar / Bayesian / causal
  lives in the experiment repo).
- The minimal-dependency posture.
- The five-step implementation-order shape (just with the parallelism and
  numeric gates added).
- The acceptance criterion that `import fabricpc` must resolve to upstream,
  not cFabricPC.
