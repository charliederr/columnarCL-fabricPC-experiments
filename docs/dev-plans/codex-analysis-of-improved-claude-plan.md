# Codex Analysis of `claude-improved-migration-plan.md`

## Summary

The improved Claude plan is substantially better than the earlier plans. Its
core direction is sound:

- keep FabricPC general,
- add only an external custom-node contract test plus documentation,
- keep columnar causal-coding implementation in the experiment repo,
- use existing FabricPC callback surfaces before inventing new hooks,
- use `node_info.node_config` for trace-time-static node configuration.

I would treat the plan as close to implementable, but not quite "good to go."
The main issues are in acceptance criteria and dependency scope, not in the
architecture boundary.

## Suggested Improvements

### 1. Replace the Phase 3 reproducibility thresholds

The plan's reproducibility table still targets older cFabricPC regime metrics:

- masked-pair final accuracy,
- ten-way final accuracy around `0.3541`,
- joint ceiling around `0.6424`.

That conflicts with the current reset: no replay, no pair-label leakage, no
commitment to migrating old cFabricPC algorithms. Phase 3 is supposed to build a
clean first implementation, not reproduce the old fork's mature behavior.

Recommended change:

- Make Phase 3 acceptance about mechanical correctness first:
  - no pair-label leakage,
  - ten-way readout only,
  - no replay or exemplar storage,
  - deterministic seeded run,
  - summary file emitted,
  - per-task accuracy and forgetting reported.
- Treat numeric results as recorded baselines for the new repo, not pass/fail
  against cFabricPC.
- Move cFabricPC comparisons to a later "reference comparison" section, not a
  hard Phase 3 gate.

If a numeric smoke threshold is desired, use a very low sanity floor such as
"better than random on the current task after one task" rather than old
multi-seed parity.

### 2. Keep the experiment repo dependency list smaller

Phase 2 proposes matching many FabricPC dependency floors directly:

```text
flax
chex
jaxtyping
tqdm
```

The experiment repo should not repeat FabricPC's transitive dependencies unless
it imports them directly. Since FabricPC already owns those requirements, the
experiment repo should start with only direct imports.

Recommended base dependencies:

```text
fabricpc
jax
jaxlib
optax
numpy
```

Add `tqdm`, `pandas`, `Pillow`, etc. only when a module directly imports them.
This better matches the user's stated goal of keeping additional pip imports to
the absolutely necessary minimum.

### 3. Remove the MNIST accuracy requirement from FabricPC acceptance

The FabricPC acceptance criteria include:

```text
examples/custom_node.py still runs end-to-end and reaches >= 0.90 MNIST test accuracy.
```

That is too expensive and too data-dependent for the minimal upstream hook
work. The new FabricPC work should prove the external custom-node contract, not
revalidate an example training benchmark.

Recommended change:

- Keep `pytest tests/test_external_custom_node.py`.
- Optionally keep `python -m py_compile examples/custom_node.py`.
- Do not require downloading/running MNIST or reaching an accuracy threshold as
  a gate for this migration.

### 4. Replace the substring ban on `replay`

The plan says no file under `columnar_cl_fabricpc/` may contain:

```text
replay
exemplar
memory_buffer
```

That is brittle. It would reject useful guardrail text such as
`ReplayBuffer is intentionally unsupported` or a test asserting that a replay
flag fails.

Recommended change:

- Ban functional replay APIs and CLI paths, not the words themselves.
- Add targeted tests instead:
  - no `ReplayBuffer` class,
  - no stored-example memory collection function,
  - no replay-enabling CLI flag,
  - if a legacy flag exists, setting it raises immediately.

### 5. Clarify `train_pcn` versus experiment-owned training

The plan sensibly prefers existing FabricPC callbacks, but Phase 3 also assumes
the first Split-CIFAR experiment can train through `train_pcn`. That may be
fine, but it should be framed as a hypothesis to verify in the smoke test, not
as a guaranteed fit for the new columnar setup.

Recommended change:

- Phase 3 should first prove that the minimal custom column graph can run one
  forward and one training step through FabricPC's existing training path.
- If the graph needs custom loss composition or task-wise parameter freezing,
  implement a local experiment training loop without changing FabricPC.

This keeps the plan consistent with the "minimal FabricPC changes" boundary.

## Bottom Line

The plan's hook design is good. I would not broaden FabricPC beyond what the
plan proposes.

The plan should be revised before implementation mainly to:

1. stop using old cFabricPC metrics as hard Phase 3 gates,
2. reduce direct experiment dependencies,
3. remove the MNIST accuracy gate from FabricPC acceptance,
4. replace the naive replay-word ban with functional no-replay tests,
5. treat `train_pcn` compatibility as something to verify early.

With those edits, the plan is ready to implement.

