# Path forward: a columnar architecture that performs adequately on plain CIFAR-10

Date: 2026-06-21
Venv: `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python`
References:
- HiBaCaML paper: `/home/ni/repos/fpc/hibacaml_agi26.pdf` (Goertzel, Derr, Taye 2026), particularly §6 "Scaling to Split-CIFAR"
- Prior diagnostic: `docs/dev-plans/2026-06-21-claude-findings-resnet18-vs-tiny-trajectory.md`
- Prior plan being re-scoped: `docs/dev_plans/energy_scaling_and_hibacaml_plan.md`

## Goal restated

Build a columnar architecture that performs at least as well as a non-columnar predictive-coding baseline on plain (single-task) CIFAR-10. Plain CIFAR-10 is a stepping stone: once a columnar architecture beats the PC ResNet baseline on it, the same column connectivity and signal flow can be carried into the Split-CIFAR-10 task-incremental protocol the paper actually targets. The metric for this stepping stone is test accuracy on plain CIFAR-10 vs. the PC ResNet baseline (37.9% at 3 epochs, 42.45% reported in earlier runs).

The non-goal is matching standard supervised-ResNet18 CIFAR-10 accuracy (94%+). The PC framework alone caps the achievable accuracy in a way orthogonal to the columnar question, and chasing that ceiling is a different research thread.

## Constraints the plan must respect

1. **Columns must be present, not bypassed away.** If the final solution is "use PC ResNet without columns", the stepping-stone has not succeeded. The minimum-viable columnar architecture must include columns that contribute non-zero, non-redundant information to the classifier.

2. **The architecture must be a legal HiBaCaML/ColBa instance.** Per the paper's Definition 1 (§3.2), the architecture must contain a family of structurally restricted component learners (columns), a combiner over an active subset, and a routing/support mechanism — even if the routing is trivial (all-active) during the plain-CIFAR-10 phase. Otherwise transferring to Split-CIFAR-10 would require re-architecting.

3. **One thing varies at a time.** The diagnostic in `docs/dev-plans/2026-06-21-claude-findings-resnet18-vs-tiny-trajectory.md` showed how easily a single round of hyperparameter changes can ship a wrong mechanism. Each experiment in this plan changes one structural axis from a frozen baseline. Multi-axis changes only happen after the single-axis result is known.

## What the diagnostic data establishes

From `results/depth_spanning_{tiny,resnet18}_1epoch_diag_*.log`:

- Columns *added in series after a deep backbone* produce a column → combiner pathway whose `z_latent` magnitude is at the noise floor for resnet18 (std 0.0031), causing the classifier to predict the marginal class distribution (test acc 10.00%).
- The same architecture on tiny backbone reaches column std ≈ 1 and combiner std ≈ 0.04, but test accuracy plateaus at 17% — far below the 38% PC ResNet baseline.
- The `E_gauss / E_ce` ratio behaves identically for both configurations (≈ 2.3 after 1 epoch), so it does not separate working from failing — the prior session's intervention target is not the cause.
- The column outputs DO grow during training (300× from init in tiny, ~150× in resnet18), so the parameters move; the limitation is that the column pathway's signal is too small relative to what the classifier needs, and the entire backbone → tap → column → combiner → pool stack is in series with no bypass.

These four facts shape the design space.

## Candidate paths considered

Below are the candidate structural changes. For each: the mechanism, what would have to be true for the change to help, what could fail, and how to falsify in the cheapest possible experiment.

### Path A — Columns as a residual side-branch

**Mechanism:** Add a direct edge from the deepest backbone stage to the classifier in parallel with the columnar pathway. The classifier sees `α · pooled(stage4) + β · pooled(combiner_out)`, with `α` and `β` learnable scalars initialized as `α = 1, β = 0`. If columns produce nothing useful, accuracy ≥ PC ResNet baseline. If columns add discriminative information, `β` grows and accuracy exceeds baseline.

**What must be true:** The columnar pathway must, given enough epochs, produce outputs that are *not collinear* with the backbone's pooled stage4 features. Otherwise `β` stays near zero and the columns are inert decoration.

**Failure mode:** Columns remain effectively zero forever, the architecture trains as a no-column PC ResNet, and the stepping-stone is technically "achieved" with columns as dead weight. This is detectable: monitor `β` (and `|column_pool|` std) across epochs.

**Cost:** Small — three node additions (a parallel pool of stage4, two scalar parameters, an Add node). No restructuring of column internals.

**Pros:** Guarantees non-regression vs. baseline; cleanest possible test of "do columns add information at all"; aligns with the paper's Definition 1 (columnar architecture is preserved, plus a residual term).

**Cons:** May give an unfaithful answer — if `β` stays at 0, we haven't learned whether the columns *could* work given a fair signal regime, only that they don't add information *given the backbone's signal*. A "columns inert" outcome here doesn't tell us whether redesigning the columns would help.

### Path B — Replace the deep backbone with the paper's recommended shallow stem

**Mechanism:** The paper (§6) explicitly proposes `Conv(3, 32, 3×3) → Conv(32, 64, 3×3, stride 2) → Conv(64, 64, 3×3)` followed by "light patch pooling to 64–96 tokens of width 64–96". This stem has on the order of 50K parameters, vs. ResNet18's 2.9M. The deep residual stack is replaced by columnar processing.

**What must be true:** The columns themselves must do the bulk of the feature extraction, since the shallow stem cannot. This is what the paper assumes.

**Failure mode:** Stem + columns underperforms even the tiny baseline because total backbone capacity is too low for plain CIFAR-10. Detectable by direct comparison to tiny-backbone PC ResNet.

**Cost:** Medium — replace `MODEL_CONFIGS["tiny"]` and `MODEL_CONFIGS["resnet18"]` with a `MODEL_CONFIGS["paper_stem"]` that maps to the three convs above; adjust `target_grid` accordingly. The existing depth-spanning column needs at most three stage outputs to skip into, so feed `[conv1_out, conv2_out, conv3_out]` as the three stages.

**Pros:** Faithful to the paper's design; the columns become the main feature processor as intended; result either way is publishable — a positive result validates the paper's recommendation on plain CIFAR-10, a negative result reveals capacity gap.

**Cons:** PC ResNet baseline becomes irrelevant for comparison (different backbone); need a "shallow-stem alone" baseline to know whether columns help; risk of low absolute accuracy that is hard to distinguish "columns don't help" from "stem is too small".

### Path C — Address signal scale: normalization at stage taps + columns

**Mechanism:** Add `LayerNorm` along the embed_dim axis after each stage tap projection and after each column output, before they enter downstream nodes. This forces stage_tap and column outputs to a known scale at all times during training, breaking the variance-drift chain. The vanishing-init signature in `stage4_pool` (std 0.0004 vs 0.021 for tiny) would no longer propagate.

**What must be true:** The failure on resnet18 is dominated by scale, not by structural information loss. If `column_pool` reaches std ≈ 1 with normalization, signal reaches the classifier; if it doesn't, normalization is treating a symptom.

**Failure mode:** Column outputs reach reasonable magnitude but still don't carry discriminative information, because the depth-spanning K/L/B pathways average out diversity. Detectable: per-column correlation matrix at the combiner input. High off-diagonal correlation ⇒ columns are redundant.

**Cost:** Small. FabricPC provides the layer-norm primitive — see "FabricPC normalization primitive" below — so implementation is constructor-flag + a few lines in `StageTapTokenizer.forward` and `DepthSpanningColumnNode.forward`. No new node type is needed.

**Pros:** Direct intervention on the measured failure mode (scale collapse). Compatible with both Path A and Path B. Cheapest to implement.

**Cons:** Treats only the scale axis. If post-normalization columns are still redundant or low-rank, accuracy will not improve. BatchNorm-family normalization complicates PC inference because batch statistics depend on the current batch's `z_latent`, but the chosen primitive (`fabricpc.utils.helpers.layernorm`) computes statistics per-sample along the last axis, so PC inference convergence is unaffected by it in principle — and is empirically fine in the existing transformer nodes that use the same function.

One subtlety to flag: LayerNorm in front of a node renormalizes the input magnitude, which makes the muPC forward-scale of that node a no-op. The FabricPC transformer code notes this at `transformer.py:193` ("Pre-norm LayerNorm absorbs the external muPC forward_scale"). For our purposes this is fine — we are not relying on per-node muPC scaling once the upstream is normalized — but the `MuPCConfig(include_output=False)` line in `build_depth_spanning_graph` should be revisited if LN ends up everywhere.

### Path D — Concat combiner with a wider classifier head

**Mechanism:** Replace the sum-combiner (which lets four column outputs interfere destructively) with a concat-along-embed_dim combiner producing shape `(num_tokens, num_columns × embed_dim)`. The classifier sees a vector of size `num_columns × embed_dim` after pooling, giving each column its own non-interfering output channel.

**What must be true:** The interference among column outputs is currently a real loss of information. If concat-then-classify works substantially better than sum-then-classify on the same column outputs, that confirms interference. If they tie, columns are individually weak and the combiner is not the bottleneck.

**Failure mode:** Concat-and-pool with poor column outputs still gives poor accuracy; meanwhile classifier parameter count grows by `num_columns ×`.

**Cost:** Small — change `MaskedColumnCombinerNode`'s combination mode or add a new combiner; the linear classifier already supports the larger input via `flatten_input=True`.

**Pros:** Tests whether the sum combiner is destroying column diversity, which is a known weakness of additive aggregation when basis functions aren't orthogonal.

**Cons:** Departs from the paper's design (paper uses an attention-style combiner, not concat). If concat works much better than sum, attention is the principled next step rather than concat. Concat is a diagnostic, not a destination.

### Path E — Scale toward the paper's column count (40 columns, sparse activation)

**Mechanism:** Increase `num_columns` to 40, set `column_mode = first_sparse` or `random_sparse` with `active_nonshared = 5` (matching paper's `k_nonshared = 5`, `N_shared = 4`, `k_total = 9`). Columns now act as a redundant pool from which a small subset is selected per example.

**What must be true:** Plain CIFAR-10 contains enough subpopulation structure that 9 specialized columns out of 40 can carve out discriminative features. The paper claims this works for Split-CIFAR-10's task-incremental structure; plain CIFAR-10 with no task boundary may not benefit from sparsity at all.

**Failure mode:** The 9 active columns are functionally equivalent (no specialization signal during training), so the architecture behaves like an all-active 9-column model with extra dead parameters.

**Cost:** Medium — `column_mode` already exists, but exact-search support selection and one-swap teacher (paper's selector mechanism) do not. For the stepping stone, only the all-active and random-sparse modes are needed; the full selector mechanism is deferred to the Split-CIFAR phase.

**Pros:** Most paper-faithful column configuration; sets up directly for Split-CIFAR-10 selector experiments.

**Cons:** On plain CIFAR-10 single task, the sparsity machinery is doing no work — the architecture is overparameterized for the task. Result is unlikely to be informative until the other failure modes are addressed.

## Proposed sequencing

The candidate paths are partially complementary: A and B are alternative *architectural anchors*, while C/D/E are *modifications* compatible with either anchor. The sequencing below picks one experiment at each fork based on cost and expected information.

### Stage 1 — Establish whether columns *can* contribute (Path A + Path C, on tiny backbone)

Reasoning: Path A guarantees non-regression, Path C is the cheapest intervention against the measured failure mode, and tiny is the backbone where columns are already not fully broken. The combination tests whether columns, given proper signal scale and a non-blocking pathway, add information at all.

Concrete change:
- Add `--bypass_columns` flag that wires `stage4_pool` directly to the classifier in parallel with the existing `combiner → column_pool → output` path. Bypass weight `α` initialized to 1, column weight `β` initialized to 0; both trainable scalars.
- Add `--layer_norm_tokens` flag that inserts a LayerNorm-equivalent normalizer along the embed_dim axis after each `stage*_tap` and after each `col_*` output.

Runs (tiny backbone, 10 epochs each, `--diagnose_energy`):

| run | bypass | layer_norm | expected |
|---|---|---|---|
| `tiny_baseline` | off | off | 17% (replicate prior) |
| `tiny_bypass_only` | on | off | ≥ 38% (matches PC ResNet via α=1, β=0) |
| `tiny_bypass_norm` | on | on | ≥ `tiny_bypass_only`; check whether β grows |
| `tiny_norm_only` | off | on | uncertain; isolates the scale fix |

**Go criterion to Stage 2:** `tiny_bypass_norm` shows `β` learning to a non-zero value across training AND test accuracy ≥ `tiny_bypass_only` by ≥ 2 points. This means the columns are contributing *additional* information beyond the backbone bypass.

**No-go branch:** `β` stays near zero or `tiny_bypass_norm` ≤ `tiny_bypass_only`. This means columns add no information given a working backbone; the depth-spanning column design itself is not buying anything, and Stage 2 should pivot to Path B (paper's shallow stem) before more column tuning.

### Stage 2 (go path) — Test on resnet18 backbone

Reasoning: The diagnostic showed the resnet18 failure is dominated by scale collapse through the deeper backbone. If Stage 1's layer-norm intervention solved scale for tiny, the same change applied to resnet18 should produce a similar result; otherwise the depth of the backbone is doing something the normalization doesn't fix.

Concrete change: re-run `bypass_norm` configuration with `--model resnet18`.

| run | expected |
|---|---|
| `resnet18_bypass_only` | ≥ PC ResNet resnet18 baseline (needs to be measured first if not yet known) |
| `resnet18_bypass_norm` | match Stage 1 `tiny_bypass_norm` qualitative behavior: β grows, accuracy improves |

**Go criterion to Stage 3:** `resnet18_bypass_norm` test accuracy ≥ `resnet18_bypass_only` by ≥ 2 points; `β` reaches a non-trivial value.

**No-go branch:** resnet18 specifically resists improvement. Pivot to Path B (replace resnet18 with the paper's shallow stem) and re-run Stage 1's experiment matrix on the paper stem.

### Stage 2 (no-go path) — Test the paper's recommended stem (Path B)

Reasoning: If Stage 1 says columns add nothing useful when the backbone alone is strong, the architecture should be tested in the configuration the paper actually proposes: a backbone too weak to classify on its own, with columns doing the work.

Concrete change: add `MODEL_CONFIGS["paper_stem"]` with `Conv(3,32) → Conv(32,64, stride=2) → Conv(64,64)`. Adjust `build_depth_spanning_graph` to feed `[conv1_out, conv2_out, conv3_out]` as the three stages. Match `target_grid` to `conv3_out`'s spatial size (16×16, giving 256 tokens — within the paper's "64–96 tokens" recommendation only after one further pooling step, which the existing stage_tap code can apply).

| run | expected |
|---|---|
| `paper_stem_alone` | low; serves as the "stem-only" baseline |
| `paper_stem_columns_only` | should exceed `paper_stem_alone` by a meaningful margin |
| `paper_stem_bypass` | guaranteed non-regression vs. `paper_stem_alone` |

**Go criterion to Stage 3:** `paper_stem_columns_only` exceeds `paper_stem_alone` by ≥ 5 points AND reaches ≥ 30% absolute. This shows the paper's recommended configuration is functional on plain CIFAR-10.

**No-go terminal:** If neither the bypass-on-deep-backbone path nor the paper's-shallow-stem path produces columns that beat their own bypass, the depth-spanning column itself needs to be reconsidered. Candidate reconsiderations are deferred to a future plan: revisit K/L/B differentiation, or test the simpler `ColumnarNode` in `columnar_cl_fabricpc/columns/column.py` instead.

### Stage 3 — Scale toward paper-faithful column count

Reasoning: Once a single column actually contributes information, replicating that contribution across a column pool is what the paper specifies for Split-CIFAR-10. Stage 3 is the first stage where the column count and the activation pattern start to matter, because the columns are now structurally useful.

Concrete changes:
- Increase `num_columns` from 4 to 40 in matched runs.
- Add a `num_shared=4`, `active_nonshared=5` configuration matching the paper's recommendation.
- Defer the exact-search and one-swap-teacher selector to Split-CIFAR-10 phase; for plain CIFAR-10 use `column_mode random_sparse` with a fixed mask per epoch.

| run | expected |
|---|---|
| `bypass_norm_40col_all_active` | ≥ `bypass_norm` (4-column) — more capacity is more information |
| `bypass_norm_40col_random_sparse_k9` | should match `all_active` within ~2 points if columns are decorrelated |

**Go criterion (terminal for the plain-CIFAR-10 phase):** A configuration with 40 columns, k_total ≈ 9 active per example, and the bypass + normalization stack achieves test accuracy ≥ 45% (matches the PC ResNet baseline within tolerance and exceeds it slightly, demonstrating that adding columns does not hurt). At this point the stepping stone is complete and the next plan addresses Split-CIFAR-10.

## Cross-cutting measurements

For every run in every stage, log:

1. Per-epoch `E_gauss / E_ce` and per-component breakdown (already wired in `diagnose_energy_breakdown`).
2. Per-epoch column z_latent std distribution across the four (or forty) columns (already wired in `diagnose_column_outputs`).
3. (NEW) Per-epoch bypass-weight `α` and column-weight `β` magnitudes if the bypass is enabled.
4. (NEW) Per-column pairwise correlation of `z_latent` on a fixed diagnostic batch, to detect column redundancy.

The correlation matrix is the new measurement most likely to be informative — Path D's hypothesis (sum-combiner destroys column diversity) hinges on this number, and the prior plan never recorded it.

## Files to add or modify

| file | change | why |
|---|---|---|
| `scripts/train_cifar10_depth_spanning.py` | Add `--bypass_columns` flag; build a parallel `pooled_stage4 → Add ← column_pool → output` topology with learnable scalars. Add `--layer_norm_tokens` flag. | Implements Paths A and C. |
| `columnar_cl_fabricpc/columns/stage_taps.py` | Fuse LayerNorm into `StageTapTokenizer.forward` and `GlobalPoolNode.forward`, gated by an `apply_layer_norm: bool = False` constructor arg. Use `fabricpc.utils.helpers.layernorm` and register `ln_gamma` / `ln_beta` in `initialize_params` (pattern from `fabricpc/nodes/transformer_v2.py:LnMlp1Node`). | Implements Path C upstream of columns. |
| `columnar_cl_fabricpc/columns/depth_spanning_column.py` | Same: fuse LayerNorm into `DepthSpanningColumnNode.forward` on the final K+L+B output, same constructor arg. | Implements Path C on column outputs. |
| `scripts/train_cifar10_depth_spanning.py` | Add `MODEL_CONFIGS["paper_stem"]` per Path B, with three conv layers and explicit stage outputs. | Implements Path B for Stage 2 no-go branch. |
| `scripts/train_cifar10_depth_spanning.py` | Extend `diagnose_column_outputs` to compute the per-column pairwise `z_latent` correlation matrix. | Cross-cutting measurement #4. |
| `docs/dev-plans/2026-06-21-claude-columnar-cifar10-path-forward.md` | (this file) | Plan of record. |

## FabricPC normalization primitive

Investigated directly (`grep -rniE "LayerNorm|GroupNorm" FabricPC/fabricpc`; `python -c "import fabricpc.nodes; dir(fabricpc.nodes)"`). Findings:

- **No standalone `LayerNormNode` exists.** The `fabricpc.nodes` module exports `IdentityNode`, `Linear`, `ConvNode`, `AvgPool`/`MaxPool`, `SkipConnection`, `EmbeddingNode`, and the transformer family, but no normalizer node.
- **A `layernorm` function does exist** at `fabricpc/utils/helpers.py:4` — standard LN along the last axis with learnable `gamma`/`beta`. Already used by `TransformerBlock`, `MhaResidualNode`, `LnMlp1Node`, and `Mlp2ResidualNode` and therefore already validated under `train_pcn`.
- **The idiomatic FabricPC pattern is fusion into the affected node's `forward`,** not a separate node. From `fabricpc/nodes/transformer_v2.py:LnMlp1Node`: register `ln_gamma` (ones, shape `(embed_dim,)`) and `ln_beta` (zeros) in `initialize_params`; call `layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])` on inputs at the top of `forward`.

Action for Path C: gate this behavior on a per-node `apply_layer_norm: bool = False` constructor flag, set from a single `--layer_norm_tokens` CLI flag in the training script. No new node type. The earlier open question about choosing layer-norm vs. group-norm becomes moot since the existing primitive is layer-norm and per-sample, with batch dependence already absent.

## Open questions

1. **What is the resnet18-backbone PC ResNet baseline?** The 37.9% / 42.45% numbers are for the tiny configuration. Resnet18-alone (no columns) accuracy is needed as the Path A bypass baseline for Stage 2. Action: run `train_cifar10_pc_resnet.py --model resnet18 --num_epochs 5` once.

2. **At what scale does the paper's stem start matching standard CIFAR-10 conv baselines?** The stem has ~50K parameters. Standard 3-layer conv on CIFAR-10 hits ~50–60% accuracy with batch norm; without batch norm and with PC inference, the number may be much lower. This baseline ("stem alone, no columns") is needed before Stage 2 no-go can be interpreted.

## Risks

1. **The depth-spanning column may be too elaborate.** The K/L/B/B_pool four-slot design assumes each microcolumn contributes complementary information. If diagnostic measurement #4 (per-column correlation) shows columns are highly correlated, simpler `ColumnarNode` from `columnar_cl_fabricpc/columns/column.py` might be a better testbed. Plan does not currently include this pivot.

2. **PC training with 40 columns may be prohibitively slow on CPU.** This session's tiny 1-epoch run took 4 minutes; resnet18 took 7 minutes. Scaling to 40 columns × 10 epochs is hours per run. GPU availability is open per the prior session's notes (`af3c504 codex finally believes me that the GPU is available`).

3. **The PC ResNet baseline number may be variance-driven.** 37.9% at 3 epochs and 42.45% reported elsewhere differ by ~5 points; without multi-seed runs the "beats baseline" criterion is fuzzy. A two-seed PC ResNet baseline would tighten this comparison. Not currently scoped into Stage 1.

## What this plan does NOT cover

- The full HiBaCaML selector mechanism (exact-search support selection, one-swap teacher). Deferred to the Split-CIFAR-10 phase.
- Internal certificates and shell dynamics. Deferred — these are CL-specific machinery.
- Whether the PC inference loop's hyperparameters (`infer_steps`, `eta_infer`, `infer_max_norm`) are well-tuned for the depth-spanning configuration. The current defaults (40 steps, eta=0.1, max_norm=1.0) are inherited from earlier scripts. A separate inference-tuning experiment is conceivable but not in scope here.
- Data augmentation, longer training schedules, or alternative optimizers. All held fixed at AdamW with the existing schedule. Changes there would confound the columnar-vs-no-columns comparison.

## Decision the user makes next

One choice upstream of any implementation:

1. **Stage 1 anchor**: start with `tiny` backbone (incremental, builds on the 17% known number) or `resnet18` (tests the bypass-fix on the harder case immediately). Recommendation: tiny, because Stage 1's interpretation is cleanest when one factor changes at a time.

(The previous "layer-norm vs group-norm" question is resolved by the FabricPC inventory above — use `fabricpc.utils.helpers.layernorm`.)

After Stage 1 anchor is settled, the matrix can be launched directly. Each cell is a single `python scripts/train_cifar10_depth_spanning.py ...` invocation with the new flags.
