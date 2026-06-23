# Energy Scaling and HIBACAML Faithful Reproduction Plan

Machine identifier: `singbuntu24`.

Source: depth-spanning columnar experiments and HIBACAML paper.

Target repository: `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`.

## Problem Statement

The depth-spanning columnar architecture shows scale-dependent failure:

| Model | Parameters | Test Accuracy | Energy Trend |
|-------|------------|---------------|--------------|
| tiny | 182K | 17.21% | stable (1.5–2.3) |
| resnet18 | 2.9M | 6.68% | collapsed (0.2→0.08) |

The tiny model learns meaningful features; resnet18 collapses to worse than random chance. The hypothesis: intermediate nodes (stage taps, columns, combiner) use GaussianEnergy while only the classifier uses CrossEntropyEnergy. As model capacity grows, the Gaussian energies dominate the total loss, causing parameters to minimize intermediate reconstruction rather than discriminative classification.

## Symbol Table

| Symbol | Definition | Architectural Referent |
|--------|------------|------------------------|
| E_total | total energy minimized during PC inference | sum of all node energies in the FabricPC graph |
| E_gauss | sum of Gaussian energies | stage taps + columns + combiner |
| E_ce | CrossEntropy energy | classifier node only |
| α | energy scale factor | multiplier on Gaussian energies (default 1.0) |
| μd | microcolumn dimension | internal width of K/L/B pathways (32 for tiny) |

## Phase 1: Energy Scaling Experiment

### 1A: Diagnosis Validation

Before modifying code, confirm the energy breakdown during training.

**Implementation steps:**

1. Add energy logging to the training loop in `scripts/train_cifar10_depth_spanning.py`:
   - Log per-node-type energy: `E_gauss` (sum of stage taps + columns + combiner) and `E_ce` (classifier).
   - Print ratio `E_gauss / E_ce` every 100 batches.

2. Run tiny model for 1 epoch with energy logging.
   - Expected: `E_gauss / E_ce` should be moderate (columns contribute but don't dominate).

3. Run resnet18 model for 1 epoch with energy logging.
   - Expected: `E_gauss / E_ce` should be much larger (intermediate nodes dominate).

**Acceptance criterion:** If `E_gauss / E_ce` is significantly larger for resnet18 than tiny, the energy dominance hypothesis is supported.

### 1B: Energy Scaling Implementation

Modify the node constructors to accept an energy scale factor α.

**Implementation steps:**

1. Add `energy_scale: float = 1.0` parameter to:
   - `StageTapTokenizer` in `columnar_cl_fabricpc/columns/stage_taps.py`
   - `GlobalPoolNode` in `columnar_cl_fabricpc/columns/stage_taps.py`
   - `DepthSpanningColumnNode` in `columnar_cl_fabricpc/columns/depth_spanning_column.py`
   - `ColumnCombinerNode` in `columnar_cl_fabricpc/columns/combiner.py`

2. Create a `ScaledGaussianEnergy` wrapper (or modify forward pass) that multiplies the Gaussian energy by α:
   ```python
   class ScaledGaussianEnergy(EnergyFunctional):
       def __init__(self, scale: float = 1.0):
           self.scale = scale
           self.base = GaussianEnergy()

       def energy(self, pred, target):
           return self.scale * self.base.energy(pred, target)
   ```

3. Add `--energy_scale` argument to `scripts/train_cifar10_depth_spanning.py`.

4. Wire the scale factor through the graph builder to all intermediate nodes.

### 1C: Scaling Experiments

**Experiment matrix:**

| Model | α (energy scale) | Expected Outcome |
|-------|------------------|------------------|
| tiny | 1.0 | baseline (17.21%) |
| tiny | 0.1 | may improve or stay similar |
| tiny | 0.01 | may improve or become unstable |
| resnet18 | 1.0 | baseline (6.68%) |
| resnet18 | 0.1 | should improve significantly |
| resnet18 | 0.01 | should improve significantly |
| resnet18 | 0.0 | pure CrossEntropy (feedforward columns) |

**Implementation steps:**

1. Run each configuration for 10 epochs.
2. Log results to `results/energy_scaling_<model>_alpha<value>_claude_singbuntu24_<timestamp>.log`.
3. Compare test accuracy and training stability.

**Acceptance criterion:** If α=0.01 or α=0.0 allows resnet18 to achieve >20% test accuracy, energy scaling is a viable path.

### 1D: Alternative — Energy Removal

If scaling works, test complete removal of Gaussian energy from intermediate nodes.

**Implementation steps:**

1. Create `NoEnergy` or use `IdentityEnergy` (zero contribution to total loss).
2. Apply to all intermediate nodes; only classifier contributes to E_total.
3. Run resnet18 for 10 epochs.

**Rationale:** Intermediate nodes would still participate in PC inference (activity propagation) but would not contribute gradient signal. Parameters update only to satisfy the classifier.

## Phase 2: HIBACAML Faithful Reproduction

If Phase 1 yields modest improvements or reveals deeper architectural issues, pivot to faithful HIBACAML reproduction.

### 2A: Paper Audit

**Implementation steps:**

1. Re-read HIBACAML paper (`/home/ni/repos/fpc/hibacaml_agi26.pdf`) focusing on:
   - How energies are structured across the hierarchy
   - Whether intermediate nodes use Gaussian reconstruction or different objectives
   - The role of internal certificates in gradient flow
   - The support mask mechanism and its interaction with energy

2. Document discrepancies between the paper and current implementation:
   - Current: all columns active, no support masking during training
   - Current: K/L/B pathways all use the same energy type
   - Current: no internal certificates
   - Current: no shell/kernel split with differential regularization

3. Create a checklist of missing mechanisms ordered by expected impact on accuracy.

### 2B: Minimal Viable HIBACAML

Start from the conv-to-column baseline described in `docs/dev-plans/2026-06-19-094051-rogdora43-codex-hibacaml-colba-cifar10-accuracy-plan.md` Phase 2.

**Implementation steps:**

1. Use the validated PC ResNet backbone (37.90% at 3 epochs) as the visual stem.
2. Replace the depth-spanning columns with the paper's typed K/L/B microcolumns:
   - K: per-token residual MLP (stable local features)
   - L: 3×3 token-grid convolution (local spatial refinement)
   - B: global pool + broadcast (context integration)
3. Use the masked sparse combiner with:
   - N_shared = 2 always-active columns
   - N_adaptive = 2 selectable columns (reduced from paper's 36 for debugging)
   - k = 2 active non-shared columns
4. Train with all columns active first (k = N_adaptive) to verify the baseline.

**Acceptance criterion:** All-active conv-to-column must match or beat the PC ResNet baseline (37.90%) before sparse selection is added.

### 2C: Shell and Kernel Paths

The paper distinguishes protected hard-kernel paths from exploratory shell paths.

**Implementation steps:**

1. Split each microcolumn's weights into:
   - Hard kernel: 25% of width, low weight decay
   - Shell S1 (reusable): 10% of width
   - Shell S2 (semi-general): 20% of width
   - Shell S3 (exploratory): 45% of width, high weight decay

2. Add diversity energy between same-tier shell outputs across columns.

3. Ablate: compare shell-enabled vs. uniform-width columns.

### 2D: Support Selection

**Implementation steps:**

1. Implement offline support audit:
   - Cache column outputs on validation set
   - Evaluate all support combinations (or one-swap neighbors)
   - Select support S that minimizes validation cross-entropy

2. Add one-swap teacher updates after each epoch.

3. Compare: all-active vs. random fixed support vs. audited support.

**Acceptance criterion:** Exact support selection must improve validation accuracy over all-active columns.

### 2E: Internal Certificates

**Implementation steps:**

1. Compute per-column certificates:
   - Mean column energy on validation set
   - Class selectivity (variance of class-conditional means)
   - Ablation utility (accuracy drop when column removed)

2. Use certificates as input to support scoring.

3. Compare: certificate-aware selection vs. certificate-free.

## Alternatives Considered

### Alternative A: Tune learning rate and weight decay without architectural changes

**Pros:**
- Zero implementation effort
- May partially mitigate energy dominance

**Cons:**
- Does not address the structural cause
- Unlikely to close the gap between tiny (17%) and PC ResNet baseline (38%)

**Decision:** Not sufficient alone; may combine with energy scaling.

### Alternative B: Replace depth-spanning columns with standard ColumnarNode

**Pros:**
- Simpler architecture
- Already implemented in `columnar_cl_fabricpc/columns/column.py`

**Cons:**
- Does not address energy dominance (same GaussianEnergy default)
- Depth-spanning columns are designed for multi-scale features

**Decision:** Test if Phase 1 fails; the issue may be energy, not the column architecture.

### Alternative C: Use backpropagation instead of predictive coding

**Pros:**
- Eliminates energy dominance entirely
- Standard training dynamics

**Cons:**
- Defeats the purpose of the project (PC-based continual learning)
- Loses the PC inference loop's potential for adaptation

**Decision:** Out of scope; this is a PC project.

### Alternative D: Scale CrossEntropy up instead of Gaussian down

**Pros:**
- Equivalent effect on gradient ratios

**Cons:**
- May cause numerical instability (large gradients)
- Harder to interpret energy values across experiments

**Decision:** Prefer scaling Gaussian down; equivalent mathematically but more stable numerically.

## Implementation Order

1. **Phase 1A**: Add energy logging (1 file change)
2. **Phase 1A**: Run diagnostic on tiny and resnet18 (2 runs)
3. **Phase 1B**: Implement ScaledGaussianEnergy (1 new class)
4. **Phase 1B**: Add energy_scale parameter to nodes (4 files)
5. **Phase 1B**: Add CLI argument (1 file)
6. **Phase 1C**: Run scaling experiments (6 runs)
7. **Decision point**: If scaling yields >25% on resnet18, refine Phase 1. Otherwise, proceed to Phase 2.
8. **Phase 2A**: Paper audit (reading, no code)
9. **Phase 2B**: Minimal viable HIBACAML (new script + node modifications)
10. **Phase 2C**: Shell/kernel paths
11. **Phase 2D**: Support selection
12. **Phase 2E**: Internal certificates

## Success Criteria

**Phase 1 success:** resnet18 with energy scaling achieves >30% test accuracy (approaching PC ResNet baseline).

**Phase 2 success:** HIBACAML-faithful architecture matches or beats PC ResNet baseline (38%) on validation accuracy.

**Overall success:** A columnar architecture that:
1. Matches or beats the PC ResNet baseline on CIFAR-10
2. Shows evidence of column specialization (different columns respond to different classes)
3. Demonstrates stable training at the resnet18 scale

## Risks

**Risk:** Energy scaling destabilizes training entirely.
**Mitigation:** Start with α=0.1, not α=0.01. Monitor for NaN or divergence.

**Risk:** Phase 1 partially works but not enough to avoid Phase 2.
**Mitigation:** Phase 2 is designed independently; Phase 1 insights inform but don't block Phase 2.

**Risk:** HIBACAML mechanisms don't map cleanly to FabricPC.
**Mitigation:** The existing plan (`2026-06-19-094051-rogdora43-codex-hibacaml-colba-cifar10-accuracy-plan.md`) already addresses this and specifies local implementations that don't require FabricPC changes.
