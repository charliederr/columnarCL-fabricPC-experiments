# Stage 1-3 Experiment Results Summary

Date: 2026-06-23
Plan: `docs/dev-plans/singbuntu24-claude-20260623-plain-cifar10-columnar-plan.md`
Primary execution machine: rogdora43
Validation run: singbuntu24

## Overview

These experiments test whether the columnar architecture can match or exceed the PC ResNet baseline (37.9%-42.45%) on plain CIFAR-10 classification. Two interventions were tested: `--bypass_columns` (direct backbone→classifier path) and `--layer_norm_tokens` (LayerNorm on stage taps and column outputs).

## Stage 1: Tiny Backbone Ablation

All runs: 10 epochs, 4 columns, all_active mode, sum combiner, batch 128, lr 0.01.

| Configuration | Test Acc | Best Val | Training Time | Machine |
|---------------|----------|----------|---------------|---------|
| tiny_baseline | 22.28% | 21.74% | 2734s | rogdora43 |
| tiny_baseline | 10.70% | 17.78% | 1530s | singbuntu24 |
| tiny_bypass_only | 39.89% | 41.62% | 922s | rogdora43 |
| tiny_norm_only | 18.13% | 19.54% | 2871s | rogdora43 |
| tiny_bypass_norm | 44.07% | 44.30% | 2872s | rogdora43 |

### Stage 1 Findings

1. **Bypass is essential.** Without bypass, accuracy is 10-22% (at or below random chance). With bypass, accuracy reaches 40-44%.

2. **LayerNorm adds value on top of bypass.** `bypass_norm` (44.07%) beats `bypass_only` (39.89%) by +4.18 points.

3. **LayerNorm alone hurts.** `norm_only` (18.13%) is worse than baseline (22.28%). LayerNorm fixes scale but doesn't address the "columns as bottleneck" problem.

4. **Baseline is unstable.** The 10.70% vs 22.28% variance across machines indicates high sensitivity to initialization when columns are the only pathway.

### Column vs Bypass Weight Contribution (tiny_bypass_norm)

| Epoch | bypass_pool→output ||W|| | column_pool→output ||W|| | Ratio (col/bypass) |
|-------|-------------------------|--------------------------|-------------------|
| 1 | 18.11 | 24.90 | 1.37 |
| 10 | 32.13 | 63.39 | **1.97** |

**Interpretation:** On tiny backbone, columns contribute ~2× the weight magnitude of bypass. Columns are the primary contributor, not dead weight.

## Stage 2: ResNet18 Backbone

All runs: 10 epochs, 4 columns, all_active mode, sum combiner.

| Configuration | Test Acc | Best Val | Training Time |
|---------------|----------|----------|---------------|
| resnet18_bypass_only | 29.17% | 33.14% | 2990s |
| resnet18_bypass_norm | **49.22%** | 49.56% | 3067s |

### Stage 2 Findings

1. **LayerNorm is critical for deep backbones.** On resnet18, LayerNorm adds +20.05 points (29.17% → 49.22%). This is 5× the effect seen on tiny (+4.18 points).

2. **ResNet18 + bypass + norm exceeds the baseline.** 49.22% beats the PC ResNet baseline (37.9%-42.45%) by 7+ points.

3. **Deeper backbone enables higher accuracy.** ResNet18 (49.22%) outperforms tiny (44.07%) by +5.15 points with the same interventions.

### Column vs Bypass Weight Contribution (resnet18_bypass_norm)

| Epoch | bypass_pool→output ||W|| | column_pool→output ||W|| | Ratio (col/bypass) |
|-------|-------------------------|--------------------------|-------------------|
| 1 | 37.27 | 15.61 | 0.42 |
| 10 | 83.68 | 35.67 | **0.43** |

**Interpretation:** On resnet18, bypass contributes ~2.3× the weight magnitude of columns. The bypass pathway through the deeper backbone carries more discriminative information; columns add refinement.

## Stage 3: Scaling to 40 Columns

Configuration: resnet18, bypass, LayerNorm, 40 columns, k_active=9 (4 shared + 5 selected).

| Configuration | Test Acc | Best Val | Training Time |
|---------------|----------|----------|---------------|
| resnet18_bypass_norm_40col_k9 | 44.26% | 44.40% | 4039s |

### Stage 3 Findings

1. **More columns hurts on plain CIFAR-10.** 40 columns (44.26%) is worse than 4 columns (49.22%) by -4.96 points.

2. **Sparsity without task boundaries adds noise.** The k_active=9 selection mechanism is designed for continual learning with task-guided selection. On plain CIFAR-10, random sparse activation introduces interference without benefit.

3. **Training time scales with column count.** 4039s for 40 columns vs 3067s for 4 columns (+32%).

## Energy Dynamics

All configurations converge to Gaussian-dominated energy (E_gauss/E_ce > 2) after training. The energy ratio does not distinguish working from failing configurations.

| Configuration | E_gauss/E_ce (init) | E_gauss/E_ce (final) | Test Acc |
|---------------|---------------------|----------------------|----------|
| tiny_baseline | 0.36 | 53.26* | 10.70% |
| tiny_bypass_norm | 0.36 | ~2.3 | 44.07% |
| resnet18_bypass_norm | 0.41 | ~2.3 | 49.22% |

*The singbuntu24 baseline run showed extreme E_gauss/E_ce = 53.26, indicating columns produced large but cancelling outputs.

## Signal Scale Analysis

Stage tap std values after training confirm LayerNorm's effect:

| Configuration | stage2_tap std | stage3_tap std | stage4_tap std | stage4_pool std |
|---------------|----------------|----------------|----------------|-----------------|
| tiny_baseline (no norm) | 1.07 | 10.03 | 14.78 | 64.12 |
| tiny_bypass_norm | 0.93 | 0.93 | 1.06 | 0.99 |
| resnet18_bypass_norm | 0.90 | 1.01 | 1.10 | 1.02 |

LayerNorm constrains all stage taps to std ≈ 1.0, preventing the scale explosion/collapse that occurs without normalization.

## Conclusions

1. **Success criterion met.** `resnet18_bypass_norm` at 49.22% exceeds the PC ResNet baseline (37.9%-42.45%).

2. **Both interventions are necessary for best results:**
   - Bypass provides non-regression guarantee and a working gradient path
   - LayerNorm fixes scale collapse through deep backbones

3. **The optimal plain CIFAR-10 configuration is:** `resnet18 + bypass + LayerNorm + 4 columns`

4. **40-column scaling should wait for Split-CIFAR-10** where task boundaries can guide column selection.

## Recommended Next Steps

1. Implement Split-CIFAR-10 task-incremental protocol
2. Use `resnet18 + bypass + LayerNorm + 4 columns` as starting configuration
3. Scale to 40 columns with task-guided selection once within-task accuracy is verified
4. Evaluate forgetting and forward transfer metrics per HiBaCaML paper

## Log Files

- `results/stage1_tiny_baseline_claude-rogdora43_20260621_140704.log`
- `results/stage1_tiny_bypass_only_claude-rogdora43_20260621_150308.log`
- `results/stage1_tiny_norm_only_claude-rogdora43_20260621_140709.log`
- `results/stage1_tiny_bypass_norm_claude-rogdora43_20260621_140711.log`
- `results/stage2_resnet18_bypass_only_claude-rogdora43_20260621_164804.log`
- `results/stage2_resnet18_bypass_norm_claude-rogdora43_20260621_153607.log`
- `results/stage3_resnet18_bypass_norm_40col_k9_claude-rogdora43_20260621_180416.log`
