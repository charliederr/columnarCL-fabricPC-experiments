# Re-analysis: the prior session's energy-dominance hypothesis is not supported by its own measurement

Date: 2026-06-21
Machine identifier: this session is the one that prepared `claude-improved-migration-plan.md` and the FabricPC custom-node test PR; the prior columnar work was done on `singbuntu24`.

Scope: this document is a critical re-analysis of `docs/dev_plans/energy_scaling_and_hibacaml_plan.md` and the diagnostic logs at:

- `results/energy_diag_tiny_1epoch_claude_singbuntu24_20260620_184334.log`
- `results/energy_diag_resnet18_1epoch_claude_singbuntu24_20260620_184431.log`
- `results/depth_spanning_tiny_10epoch_claude_singbuntu24_20260620_175116.log`

Per `AGENTS.md`: this is a mechanistic analysis grounded in measured numbers and the architecture in `scripts/train_cifar10_depth_spanning.py`. It does not propose new architectural changes; that is the user's decision.

## The hypothesis under review

From `energy_scaling_and_hibacaml_plan.md`:

> The hypothesis: intermediate nodes (stage taps, columns, combiner) use GaussianEnergy while only the classifier uses CrossEntropyEnergy. As model capacity grows, the Gaussian energies dominate the total loss, causing parameters to minimize intermediate reconstruction rather than discriminative classification.

The prior plan operationalized this as: `E_gauss = sum of (backbone + stage_taps + columns + combiner)` versus `E_ce = classifier`. The expected signature was `E_gauss / E_ce` being noticeably larger for `resnet18` than for `tiny`.

## What the diagnostic actually measured

The diagnostic in `train_cifar10_depth_spanning.py:82-141` was run once per model, **before any training**, on a single sampled batch with `infer_steps=40`. Both runs ran with identical training hyperparameters and `--column_mode all_active`, `--combiner sum`.

Measured per-component sums (pre-training, sample batch, seed 42):

| component | tiny model | resnet18 model |
|-----------|-----------:|---------------:|
| backbone | 0.0001 | 0.0000 |
| stage_taps | 0.0000 | 0.0000 |
| columns | 0.0001 | 0.0000 |
| combiner | 57.2745 | 60.9907 |
| classifier (E_ce) | 160.9080 | 148.8664 |
| **E_gauss (sum of non-classifier)** | **57.2747** | **60.9907** |
| **E_gauss / E_ce** | **0.3559** | **0.4097** |

## Mechanistic reading of the numbers

Two facts in the measured table are load-bearing:

1. `E_gauss / E_ce` is essentially the same for the two models (0.36 vs 0.41). The cross-entropy contribution is the larger one in both cases. The expected signature — `resnet18` showing E_gauss-dominance and `tiny` not — is absent.

2. Of the four GaussianEnergy contributors, three (backbone, stage_taps, columns) measure as effectively zero at init for both models. All of `E_gauss` is concentrated at the **combiner** node. The columns themselves contribute no measurable Gaussian energy at this point.

Why backbone / stage_taps / columns measure as zero, mechanistically: `GaussianEnergy` is `0.5 * Σ (z_latent − μ)²`, where `μ` is the forward prediction from upstream and `z_latent` is the per-step inference state. With 40 inference steps and `InferenceSGDNormClip`, each node's `z_latent` is iteratively pulled toward its upstream prediction `μ`. For a feedforward node with no downstream pressure that disagrees (i.e., no other-side reconstruction loss attached), `z_latent` converges to `μ` and the GaussianEnergy at that node goes to zero. The combiner is the one place where `z_latent` is pulled in two directions: upstream by the sum/attention of column outputs, and downstream by the classifier's cross-entropy gradient flowing back through `column_pool` → `combiner`. That's why the combiner is the only Gaussian node with non-zero residual at convergence.

So the actual energy-balance picture at initialization is:

```
E_total ≈ E_combiner (≈ 60) + E_classifier (≈ 150)
```

not

```
E_total ≈ E_backbone + E_taps + E_columns + E_combiner + E_classifier
```

The cross-entropy is the dominant energy term in both configurations. The data does not match the hypothesis's expected signature.

## What the diagnostic did *not* measure

The diagnostic was run only at step 0 (before training) and at the final epoch's end. It did not log:

1. The trajectory of `E_gauss / E_ce` *during* training. The 10-epoch tiny run shows that `train_pcn`'s reported `energy` evolves (1.79 → 1.5–2.3 range for tiny; 0.20 → 0.08 for resnet18 per the earlier 10-epoch run). The per-component split for those trajectories is unknown.

2. The magnitudes of column **outputs** (i.e., the values flowing forward into the combiner). The plan's claim that columns "converge to trivial constant outputs that minimize Gaussian energy but destroy classification signal" is a statement about output magnitudes, and was not measured. The Gaussian energy at a column node being ~0 (which is what the diagnostic shows) does *not* imply the column's *output* is ~0 — it implies the column's `z_latent` matches its own forward prediction `μ`, which can be at any magnitude.

3. The gradient norms reaching the column parameters from the classifier's cross-entropy. The Phase 5 of `docs/dev-plans/depth-spanning-columnar-architecture.md` proposes this log, but the script does not implement it.

Without those three trajectories, the energy-dominance claim is neither confirmed nor refuted; it is untested in the relevant regime (during training, at the actual node where the prediction is made).

## Structural facts about the two configurations

These are read directly from `train_cifar10_depth_spanning.py:144-158` and the logged graph shapes; they are not hypotheses.

| property | tiny | resnet18 |
|---|---:|---:|
| backbone stage outputs | `(32,32,16)`, `(16,16,32)`, `(8,8,64)` | `(32,32,32)`, `(16,16,64)`, `(8,8,128)`, `(4,4,256)` |
| final stage spatial grid | `(8,8)` | `(4,4)` |
| `target_grid` for stage taps | `(8,8)` | `(4,4)` |
| tokens per column input slot | 64 | 16 |
| parameters | 182K | 2.92M |
| graph nodes / edges | 24 / 41 | 40 / 62 |

The depth-spanning column logic (`build_depth_spanning_graph` in `train_cifar10_depth_spanning.py:265-459`) sets `target_grid` to the final stage's spatial dimensions and adaptively pools all stage taps down to that grid. The resnet18 model therefore feeds **16 tokens** into each column for downstream reasoning, versus 64 for tiny. This is a 4× change in the spatial token budget that the columns and combiner operate over, and it is independent of energy.

## What the 6.68% test accuracy means mechanistically

10-class CIFAR-10 random chance is 10%. The resnet18 column architecture achieved 6.68%. Below random chance is informative: it indicates the model produces predictions that are anti-correlated with the labels on the test set, not just uncorrelated. For an unconstrained 10-class classifier to be reliably below 10%, the model must be putting probability mass on specific wrong classes systematically, not distributing it uniformly. A trivial "predict one class for everything" baseline gets ~10% on a class-balanced test set. The 6.68% number is therefore evidence of structured wrong predictions, not collapse to a uniform output.

## Gaps in the prior plan that this analysis identifies

1. The hypothesis was treated as supported and the plan moved to a six-run scaling experiment matrix without first measuring `E_gauss / E_ce` during training. The pre-training measurement, taken on its own terms, falsifies the expected signature.

2. The "intermediate nodes use GaussianEnergy, classifier uses CrossEntropyEnergy" framing groups three node families that all measure as zero-energy contributors at init. The combiner is the single Gaussian contributor, and its role is not interchangeable with the others; treating the four as a single bucket obscures which node is doing what.

3. The structural difference in token count (64 → 16) between the two model configurations is not addressed by the energy-scaling intervention, and was not raised as a candidate explanation in the prior plan.

4. The "below random chance" character of the 6.68% number is not addressed at all in the prior plan, which treats it interchangeably with "above 10% but below tiny" outcomes.

## What can be measured cheaply to disambiguate

The training script already constructs `extract_node_energies(final_state)` and exposes `epoch_callback`. The trajectories below can be added without changing the architecture:

1. **Per-epoch energy breakdown.** Call `diagnose_energy_breakdown(...)` from inside `epoch_callback` after evaluation. This produces the `E_gauss / E_ce` trajectory the prior plan needed and never logged.

2. **Per-epoch column output statistics.** Add an `extract_node_states` pass that reports mean(|out|), std(out), and per-token zero fraction for the four `col_*` nodes. This is the direct test of the "columns collapse to trivial output" claim.

3. **Same-token-budget control.** Repeat the resnet18 run with `target_grid=(8,8)` instead of `(4,4)`, by pooling stage4 *up* to 8×8 in `build_depth_spanning_graph`. This isolates the 4× token-budget difference from the model-depth difference.

These three measurements are cheap (the first two are zero-cost adds to existing functions; the third is a single argument change plus one bilinear upsample) and would produce data that either confirms the energy-dominance reading, falsifies it, or points to the token-budget reading. None of them commit to a particular architectural fix yet.

## Status

Hypothesis under review: not supported by the data the prior session collected.
Hypothesis status: untested during training; pre-training measurement contradicts the expected signature.
Recommended action: not in scope for this document; the next step is the user's call between (a) running the three cheap measurements above before any architectural change, or (b) proceeding with the Phase 1B energy-scaling experiment as planned and accepting the risk that it intervenes on a non-cause.
