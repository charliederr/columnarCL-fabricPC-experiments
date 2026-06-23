# Plain CIFAR-10 Columnar Architecture Plan

Date: 2026-06-23
Machine identifier: `singbuntu24`
Venv: `/home/ni/repos/fpc/py3`
References:
- HiBaCaML paper: `/home/ni/repos/fpc/hibacaml_agi26.pdf` (Goertzel, Derr, Taye 2026)
- Prior path-forward plan: `docs/dev-plans/2026-06-21-claude-columnar-cifar10-path-forward.md`
- Diagnostic findings: `docs/dev-plans/2026-06-21-claude-findings-resnet18-vs-tiny-trajectory.md`
- Energy reanalysis: `docs/dev-plans/2026-06-21-claude-reanalysis-of-energy-hypothesis.md`

## Goal

Build a columnar architecture that performs at least as well as the PC ResNet baseline (37.9%–42.45%) on plain single-task CIFAR-10. This is a stepping stone: once a columnar architecture matches the PC baseline on plain CIFAR-10, the same architecture can be carried into the Split-CIFAR-10 task-incremental protocol the HiBaCaML paper actually targets.

**Non-goals:**
- Matching supervised-ResNet18 CIFAR-10 accuracy (94%+). The PC framework caps achievable accuracy orthogonally to the columnar question.
- Split-CIFAR-10 continual learning. That is deferred until plain CIFAR-10 is solved.

## Current State

### Baselines

| Configuration | Test Accuracy | Notes |
|---------------|---------------|-------|
| PC ResNet (no columns) | 37.9%–42.45% | `train_cifar10_pc_resnet.py`, leaky_relu, 80 infer steps |
| Random chance | 10% | 10-class uniform |

### Columnar Experiments

| Configuration | Test Accuracy | Diagnosis |
|---------------|---------------|-----------|
| tiny backbone + 4 columns | 16.96%–17.21% | Column outputs reach std ≈ 1–2; combiner z_latent std ≈ 0.04 |
| resnet18 backbone + 4 columns | 6.68%–10.00% | Column outputs remain at std ≈ 0.13; combiner z_latent at noise floor (0.003) |

**Key finding:** Adding columns *reduces* accuracy from 38–42% to 17% (tiny) or 10% (resnet18). The columnar pathway is currently a bottleneck, not an enhancement.

### Diagnosed Failure Mechanisms

From `docs/dev-plans/2026-06-21-claude-findings-resnet18-vs-tiny-trajectory.md`:

1. **Scale collapse through deep backbones.** Resnet18's `stage4_pool` std is 0.0004 vs tiny's 0.021 (50× ratio). The 4-stage residual stack without batch normalization compounds variance drift, and Kaiming initialization on the 256→64 projection shrinks output further.

2. **Token budget reduction.** Resnet18 uses target_grid=(4,4) → 16 tokens. Tiny uses (8,8) → 64 tokens. A 4× reduction in spatial resolution before column processing.

3. **Energy dominance is not the differentiator.** Both models end up with E_gauss/E_ce ≈ 2.3 after one epoch. The energy-dominance hypothesis does not separate working from failing configurations.

4. **Columns in series with no bypass.** The entire backbone → tap → column → combiner → pool stack is in series. If any stage produces near-zero signal, the classifier sees a near-constant input and predicts the marginal distribution.

## HiBaCaML Paper Recommendations for CIFAR-10

From Section 6 "Scaling to Split-CIFAR" (hibacaml_agi26.pdf):

### Recommended Configuration

| Parameter | Paper Value | Current Implementation |
|-----------|-------------|------------------------|
| N_col | 40 | 4 |
| N_shared | 4 | 2 |
| k_active | 9 (4 shared + 5 selected) | 4 (all active) |
| Hard kernel width d_m | 32 | 32 |
| Shell sizes | S¹=10, S²=20, S³=30 | Not implemented |
| Visual stem | `Conv(3,32,3×3) → Conv(32,64,3×3,s2) → Conv(64,64,3×3)` | ResNet-style backbone |
| Tokens | 64–96 tokens of width 64–96 | 16–64 tokens, width 64 |

### Paper's Architectural Insight

The paper's design places the columns *as the main feature processor*, not after a deep backbone. The shallow stem extracts low-level visual primitives; the columns carry the discriminative representation learning. The current implementation inverts this: a deep ResNet backbone does most of the feature extraction, and columns are appended as an afterthought.

## Candidate Approaches

### Approach A: Residual Bypass (Guarantees Non-Regression)

**Mechanism:** Add a direct edge from `stage4_pool` to the classifier in parallel with the columnar pathway. The classifier sees `α·pooled(stage4) + β·pooled(combiner_out)` with learnable scalars initialized α=1, β=0.

**Already implemented:** `--bypass_columns` flag in `train_cifar10_depth_spanning.py`

**Pros:**
- Guarantees non-regression vs. PC ResNet baseline
- Cleanest test of "do columns add information at all"
- If β grows during training, columns are contributing; if β≈0, columns are inert

**Cons:**
- May produce a "columns as dead weight" outcome where the bypass does all the work
- Does not address the underlying signal-scale problem

**Status:** Implemented but not systematically tested.

### Approach B: Paper's Shallow Stem (Faithful to HiBaCaML)

**Mechanism:** Replace the deep ResNet backbone with the paper's recommended 3-conv stem:
```
Conv(3, 32, 3×3) → Conv(32, 64, 3×3, stride=2) → Conv(64, 64, 3×3)
```
This produces (16×16×64) features with ~50K parameters. The columns become the main feature processor.

**Pros:**
- Faithful to the paper's design intent
- Columns are forced to do the discriminative work
- Result either way is informative: positive validates paper, negative reveals capacity gap

**Cons:**
- PC ResNet baseline becomes irrelevant (different backbone)
- Need a new "stem alone" baseline
- Risk of low absolute accuracy that is hard to interpret

**Status:** Not implemented. Requires adding `MODEL_CONFIGS["paper_stem"]`.

### Approach C: LayerNorm at Stage Taps and Columns

**Mechanism:** Insert `LayerNorm` along the embed_dim axis after each stage tap projection and after each column output. Forces signal to known scale through the pipeline.

**Already implemented:** `--layer_norm_tokens` flag in `train_cifar10_depth_spanning.py`

**Pros:**
- Direct intervention on the measured failure mode (scale collapse)
- Compatible with both Approach A and B
- Cheapest to implement

**Cons:**
- Treats only the scale axis
- If post-normalization columns are still redundant, accuracy will not improve

**Status:** Implemented but not systematically tested.

### Approach D: Concat Combiner (Test Column Diversity)

**Mechanism:** Replace sum-combiner with concat-along-embed_dim, producing shape `(tokens, num_columns × embed_dim)`. Each column gets its own non-interfering output channel.

**Pros:**
- Tests whether sum-combiner destroys column diversity
- Diagnostic: if concat >> sum, columns are producing diverse but interfering features

**Cons:**
- Departs from paper's attention-style combiner
- Concat is a diagnostic, not a destination (attention is the principled next step)

**Status:** Not implemented. `MaskedColumnCombinerNode` currently supports only "sum" and "attention".

### Approach E: Scale to Paper's Column Count

**Mechanism:** Increase to 40 columns, k_active=9 (4 shared + 5 selected), with random-sparse selection.

**Pros:**
- Paper-faithful configuration
- Sets up directly for Split-CIFAR-10

**Cons:**
- On plain CIFAR-10 without task boundaries, sparsity machinery does no work
- Computationally expensive (40 columns × 10 epochs)
- Unlikely to help until the underlying signal-flow problems are fixed

**Status:** Partially supported (num_columns configurable, column_mode=random_sparse exists).

## Chosen Path: Systematic Ablation of Implemented Interventions

Rather than implementing more interventions, **first systematically test the interventions already implemented** (`--bypass_columns`, `--layer_norm_tokens`) that were added but never validated.

### Stage 1: Establish Baselines (All on Tiny Backbone)

Tiny backbone is chosen because:
- Columns already produce non-zero signal (std ≈ 1–2)
- Prior 17% accuracy means room for improvement
- Faster iteration (234s per epoch vs 429s for resnet18)

**Runs (10 epochs each, `--diagnose_energy`):**

| Run ID | bypass | layer_norm | Expected Outcome |
|--------|--------|------------|------------------|
| `tiny_baseline` | off | off | ~17% (replicate prior) |
| `tiny_bypass_only` | on | off | ≥38% (matches PC ResNet via α=1, β→0) |
| `tiny_norm_only` | off | on | Uncertain; isolates scale fix |
| `tiny_bypass_norm` | on | on | ≥`tiny_bypass_only`; watch if β grows |

**Measurements:**
- Per-epoch test/val accuracy
- Per-epoch `β` weight magnitude (bypass edge contribution)
- Per-epoch column z_latent std
- Per-epoch column pairwise correlation (already in `diagnose_column_correlation`)

**Go criterion:** `tiny_bypass_norm` shows β > 0.1 by epoch 10 AND test accuracy ≥ `tiny_bypass_only` + 2 points. This means columns are contributing additional information beyond bypass.

**No-go branch:** β stays near zero. Columns add nothing given a working backbone. Proceed to Stage 2B (paper's shallow stem).

### Stage 2A (Go Path): Test on Resnet18 Backbone

If Stage 1 shows columns contributing on tiny, test whether the same interventions fix resnet18.

**Runs:**

| Run ID | Expected Outcome |
|--------|------------------|
| `resnet18_bypass_only` | Baseline; should match PC ResNet resnet18 |
| `resnet18_bypass_norm` | β should grow; accuracy should improve |

**Go criterion:** `resnet18_bypass_norm` shows β > 0.05 and test accuracy ≥ `resnet18_bypass_only` + 2 points.

### Stage 2B (No-Go Path): Paper's Shallow Stem

If Stage 1 shows columns contribute nothing, implement the paper's recommended stem.

**Implementation:**
1. Add `MODEL_CONFIGS["paper_stem"]` with three conv layers
2. Stage outputs become `[conv1_out, conv2_out, conv3_out]`
3. target_grid = (16,16) → 256 tokens (within paper's 64–96 recommendation after one pooling step)

**Runs:**

| Run ID | Expected Outcome |
|--------|------------------|
| `paper_stem_alone` | Low baseline (stem has ~50K params) |
| `paper_stem_columns_bypass` | Non-regression guarantee |
| `paper_stem_columns_only` | Should exceed `paper_stem_alone` by ≥5 points |

**Go criterion:** `paper_stem_columns_only` exceeds `paper_stem_alone` by ≥5 points AND reaches ≥30% absolute.

### Stage 3: Scale to Paper-Faithful Column Count

Once a configuration shows columns contributing information:

1. Increase `num_columns` from 4 to 40
2. Use `column_mode=random_sparse` with `num_shared=4`, `active_nonshared=5`
3. Compare all-active vs. sparse activation

**Target:** Test accuracy ≥45% with 40 columns (matches PC ResNet baseline and demonstrates columns don't hurt).

## Implementation Changes Required

| File | Change | Phase |
|------|--------|-------|
| `scripts/train_cifar10_depth_spanning.py` | Run the Stage 1 experiment matrix | Stage 1 |
| `scripts/train_cifar10_depth_spanning.py` | Add `MODEL_CONFIGS["paper_stem"]` | Stage 2B |
| `columnar_cl_fabricpc/columns/combiner.py` | Add `combination="concat"` mode | Optional (Stage 3) |
| `docs/dev-plans/` | Log results of each stage | Throughout |

## Alternatives Considered

### Alternative 1: Tune Hyperparameters Without Architectural Changes

**Pros:** Zero implementation risk.
**Cons:** Does not address the structural cause. Prior experiments show hyperparameter changes (lr, weight_decay, infer_steps) do not close the gap.
**Decision:** Not sufficient alone.

### Alternative 2: Use Backpropagation Instead of Predictive Coding

**Pros:** Eliminates PC-specific energy dynamics.
**Cons:** Defeats the purpose of the project (PC-based continual learning).
**Decision:** Out of scope.

### Alternative 3: Skip Plain CIFAR-10, Go Directly to Split-CIFAR-10

**Pros:** The paper is designed for continual learning; plain CIFAR-10 may not benefit from columns.
**Cons:** If columns hurt on plain CIFAR-10, they will hurt within-task accuracy on Split-CIFAR-10 too. The paper's claim is that columns *preserve* accuracy while enabling CL, not that they *require* CL to function.
**Decision:** Plain CIFAR-10 first. The stepping stone must be crossed.

### Alternative 4: Implement Full HiBaCaML Mechanisms (Shells, Certificates, One-Swap Teacher)

**Pros:** Most paper-faithful.
**Cons:** Premature optimization. If columns don't contribute basic discriminative information, shells and certificates won't help.
**Decision:** Defer until columns demonstrate positive contribution.

## Success Criteria

### Stage 1 Success
- `tiny_bypass_norm` test accuracy ≥ 38% (matches PC ResNet)
- `β` weight > 0.1 (columns contributing)

### Stage 2 Success
- Either:
  - (Go path) `resnet18_bypass_norm` test accuracy ≥ 38%
  - (No-go path) `paper_stem_columns_only` test accuracy ≥ 30% AND exceeds stem-alone by ≥5 points

### Overall Success
A columnar architecture configuration that:
1. Matches or beats the PC ResNet baseline (38–42%) on plain CIFAR-10
2. Shows evidence that columns contribute (β > 0.1, or columns-only > stem-alone)
3. Can be transferred to Split-CIFAR-10 without re-architecting

## Risks

1. **Bypass may mask whether columns work.** If α → 1 and β → 0, we learn that columns are inert but not *why*. Mitigation: always run columns-only variant alongside bypass.

2. **LayerNorm may break PC inference dynamics.** Normalizing z_latent at each node changes the loss surface. Mitigation: the FabricPC transformer nodes already use LayerNorm; empirically fine.

3. **Paper's shallow stem may have too little capacity.** ~50K params for CIFAR-10 is very small. Mitigation: compare to a stem-alone baseline; if stem-alone is <15%, capacity is the limit, not columns.

4. **40-column runs are slow on CPU.** Mitigation: GPU is available per prior session's notes. Use `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## What This Plan Does NOT Cover

- Split-CIFAR-10 task-incremental protocol (deferred)
- Full HiBaCaML selector mechanism (exact-search support selection, one-swap teacher)
- Internal certificates and shell dynamics
- PC inference hyperparameter tuning (infer_steps, eta_infer)
- Data augmentation or longer training schedules

## Next Action

Run the Stage 1 experiment matrix on tiny backbone. This requires no code changes—only executing `train_cifar10_depth_spanning.py` with the appropriate flags.

```bash
# Baseline (no bypass, no norm)
python scripts/train_cifar10_depth_spanning.py \
    --model tiny --num_epochs 10 --diagnose_energy

# Bypass only
python scripts/train_cifar10_depth_spanning.py \
    --model tiny --num_epochs 10 --diagnose_energy --bypass_columns

# Norm only
python scripts/train_cifar10_depth_spanning.py \
    --model tiny --num_epochs 10 --diagnose_energy --layer_norm_tokens

# Bypass + norm
python scripts/train_cifar10_depth_spanning.py \
    --model tiny --num_epochs 10 --diagnose_energy --bypass_columns --layer_norm_tokens
```

After Stage 1 results are in, the plan branches based on whether β grows.
