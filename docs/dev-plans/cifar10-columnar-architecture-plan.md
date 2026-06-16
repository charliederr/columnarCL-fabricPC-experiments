# CIFAR-10 Columnar Architecture Plan

This plan outlines the implementation of a columnar predictive coding architecture
for CIFAR-10 classification, based on the HiBaCaML/ColBa framework described in
Goertzel et al., "Hierarchical Bayesian Causal Modular Learning: A Two-Level
Columnar Architecture for Continual Learning" (hibacaml_agi26.pdf).

## Goal

Implement and validate the ColBa columnar architecture on standard CIFAR-10
classification (all 10 classes, joint training) before attempting Split-CIFAR-10
continual learning experiments. This staged approach isolates architectural
validation from continual-learning complexity.

## Architecture Overview

The ColBa architecture decomposes into two coupled levels:

1. **Top-level controller (Ψ)**: Selects a sparse subset of columns for each input
2. **Component learners (columns)**: Structurally restricted modules with internal
   probabilistic organization

### Key Insight from HiBaCaML

The architecture is designed so that different kinds of uncertainty are handled
at different structural scales:
- Top level: combinatorial selection problem (which columns to activate)
- Inside each column: structural-hygiene problem (what is reusable vs task-local)

For standard CIFAR-10 (single task), we simplify by:
- Using fixed support selection initially (all columns active, or random subset)
- Deferring internal shell dynamics until continual learning phase
- Focusing on verifying that columnar processing achieves competitive accuracy

## Architecture Components

### 1. Shared Visual Stem

A shallow convolutional stem extracts low-level visual primitives before columnar
processing. This is not a violation of the columnar idea — the columns carry the
modularity burden while the stem provides a better front-end.

From Section 6 of the paper:
```
Conv(3, 32, 3×3) → Conv(32, 64, 3×3, stride 2) → Conv(64, 64, 3×3)
```
Followed by patch pooling to 64–96 tokens of width 64–96.

**Implementation approach**: Build as standard FabricPC nodes (Conv2D or Linear
after flattening patches).

### 2. Column Pool

For CIFAR-10 (paper Section 6 recommendations, adapted for single-task):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Ncol | 40 | Total columns |
| Nshared | 4 | Always-on shared columns |
| Nadaptive | 30 | Columns available for selection |
| Nreserve | 6 | Reserve columns (for continual learning) |
| kactive | 9 | Active columns per example (4 shared + 5 selected) |

For initial CIFAR-10 experiments, we may start simpler:
- All columns always active (no selection), or
- Fixed random subset, or
- Learned soft selection via attention

### 3. Column Internal Structure

Each column contains R=3 typed microcolumns:

| Microcolumn | Role | Description |
|-------------|------|-------------|
| K (kernel) | Stable processing | Kernel-style processing with moderate context |
| L (lateral) | Local refinement | Same-region discriminative refinement |
| B (bridge) | Cross-region context | Integrates information across spatial extent |

Each microcolumn has:
- **Hard kernel**: Width dm=32, protected/non-prunable substrate
- **Shell tiers**: S(1)=10, S(2)=20, S(3)=30 units
  - Inner shells: reusable abstraction
  - Middle shells: semi-general structure
  - Outer shells: task-local residue

For initial CIFAR-10 (single task), shell dynamics are inactive — all parameters
train jointly. Shell semantics become relevant only during continual learning.

### 4. Combiner

An attention-style composer combines active column outputs into the per-token
representation read out by the classification head.

From Section 4.3:
> A small attention-style composer combines the outputs of the five active columns
> into the per-token representation that is then read out by the task-local head.

### 5. Classification Head

For standard CIFAR-10: single 10-way softmax head over combined column outputs.

## Implementation Phases

### Phase A: Infrastructure (Current)

**Status: COMPLETE** (see implementation-plan-phase0-3.md)

- FabricPC installed and verified
- Experiment repo skeleton created
- ExampleColumnNode demonstrates extension contract
- NodeMetadataRegistry available for column role tracking

### Phase B: CIFAR-10 Data Loading

**Status: COMPLETE**

Uses FabricPC's `Cifar10Loader` from `fabricpc.utils.data.dataloader`:
- Requires tensorflow-datasets (via tfds)
- Images normalized per-channel with CIFAR-10 statistics
- Labels one-hot encoded (required for predictive coding)
- Train/val/test splits via tfds slicing syntax

Also created `columnar_cl_fabricpc/data/cifar.py` with direct download option
(no TF dependency) for environments where tensorflow is not available.

### Phase C: Visual Stem Node

**Status: COMPLETE**

Created `columnar_cl_fabricpc/columns/visual_stem.py` with `PatchEmbedNode`:
- Implements option 2 (patch embedding)
- Divides 32×32 image into 4×4 grid of 8×8 patches (16 patches)
- Each patch: 8×8×3 = 192 dims → embed to 96 dims (configurable)
- Includes learnable position embeddings
- Result: 16 tokens of width 96

15 tests verify patch extraction, parameter initialization, and FabricPC integration.

### Phase D: Column Nodes

**Status: COMPLETE**

Created `columnar_cl_fabricpc/columns/column.py` with `ColumnarNode`:
- 3 microcolumns (K, L, B) as internal substructure
- Each microcolumn has input projection (input_dim → microcolumn_dim) and
  output projection (microcolumn_dim → output_dim)
- GELU activation between projections
- Supports "sum", "concat", or "attention" combination modes
- `create_column_pool()` helper creates pools of 40 columns (CIFAR-10 config)

18 tests verify parameter structure, combination modes, and FabricPC integration.

### Phase E: Column Combiner

**Status: COMPLETE**

Created `columnar_cl_fabricpc/columns/combiner.py` with:
- `ColumnCombinerNode`: Combines column outputs via "sum", "concat", or "attention"
- `ClassificationHeadNode`: Pools tokens (mean/max/cls) and projects to class logits
- Uses softmax activation and cross-entropy energy for classification

20 tests verify combiner modes, classification head, and pipeline integration.

### Phase F: Full CIFAR-10 Model

**Status: COMPLETE**

Created `columnar_cl_fabricpc/experiments/cifar10_columnar.py`:
- `create_cifar10_columnar_model()`: Assembles full architecture
- `train_cifar10_columnar()`: Training loop with validation callbacks
- CLI interface for running experiments

Architecture assembled:
```
Input (32×32×3)
  → Patch Embedding (16 tokens × 96 dims)
  → Column Pool (40 columns, all active)
  → Combiner (attention over column outputs)
  → Classification Head (10-way softmax)
```

11 tests verify model assembly, shapes, and forward pass.

### Phase G: Training and Validation

**Status: IN PROGRESS**

Training pipeline verified end-to-end:
- [x] Uses FabricPC's `train_pcn` and `evaluate_pcn` with correct API
- [x] Uses FabricPC's `Cifar10Loader` for data (requires tensorflow-datasets)
- [x] Epoch callbacks for validation accuracy tracking
- [x] Quick test script: `scripts/train_cifar10_quick_test.py`
- [ ] Achieve >85% test accuracy (baseline target)
- [ ] Achieve >90% test accuracy (proceed to continual learning)

**Progress:**
- Initial test (3 columns, 32-dim, 1 epoch): 20.5% accuracy
- Medium test (10 columns, 64-dim, 5 epochs): 18.6% accuracy
- Energy decreases significantly during training (374M → ~1M)
- But classification accuracy does not improve correspondingly

**Diagnosis needed:**
- Energy minimization is happening but not translating to classification accuracy
- Possible causes:
  1. Predictive coding inference hyperparameters (eta_infer, infer_steps) may need tuning
  2. Classification head architecture may need adjustment
  3. Column combination strategy may not be effective
  4. Learning rate or optimizer settings may need tuning

**Next steps:**
- Investigate inference hyperparameters (currently eta_infer=0.1, infer_steps=10)
- Compare with FabricPC MNIST examples to identify architectural differences
- Consider simpler baseline (fewer columns, direct patch→classifier) to isolate issues

### Phase H: Sparse Selection (Optional for CIFAR-10)

**Status: NOT STARTED**

Once Phase G achieves good accuracy, optionally add:
- Soft column selection via learned gates
- Top-k hard selection
- Measure accuracy vs. sparsity tradeoff

This prepares for continual learning but is not required for CIFAR-10 baseline.

## Alternatives Considered

### Column Implementation

| Approach | Pros | Cons |
|----------|------|------|
| Single ColumnarNode per column | Clean, matches paper | Complex node implementation |
| Composition of Linear nodes | Uses existing FabricPC nodes | Graph becomes large |
| Hybrid: ColumnarNode with internal Linear | Manageable complexity | Some custom code |

**Choice**: Hybrid approach. ColumnarNode encapsulates the K/L/B structure but
uses standard operations internally.

### Visual Stem

| Approach | Pros | Cons |
|----------|------|------|
| Conv2D stack | Matches paper exactly | Requires Conv2D node |
| Patch + Linear embedding | Simple, uses existing nodes | Less spatial inductive bias |
| Hybrid: one Conv2D then patches | Good tradeoff | Medium complexity |

**Choice**: Start with patch + Linear embedding (simplest). Add Conv2D stem if
accuracy is insufficient.

### Training

| Approach | Pros | Cons |
|----------|------|------|
| FabricPC predictive coding (train_pcn) | Native, local learning | May need tuning |
| Standard backprop baseline | Well-understood | Loses PC benefits |
| Hybrid: backprop stem, PC columns | Potential best of both | Complex |

**Choice**: Start with full `train_pcn`. If accuracy is poor, diagnose whether
it's architecture or training. Backprop baseline as fallback for comparison.

## Success Criteria

### Phase G (CIFAR-10 Baseline)

- [ ] Model trains without errors
- [ ] Test accuracy >85% (acceptable baseline)
- [ ] Test accuracy >90% (good performance, proceed to continual learning)
- [ ] Training completes in reasonable time (<2 hours on available hardware)

### After CIFAR-10 Success

Once standard CIFAR-10 achieves good accuracy:
1. Implement sparse column selection
2. Implement shell-based parameter organization
3. Implement Split-CIFAR-10 with 5 binary tasks
4. Measure forgetting and compare to baselines

## File Structure (Planned)

```
columnar_cl_fabricpc/
  columns/
    __init__.py
    example_node.py          # (exists)
    visual_stem.py           # Phase C
    column.py                # Phase D
    combiner.py              # Phase E
  data/
    __init__.py
    cifar.py                 # Phase B
  experiments/
    __init__.py
    cifar10_columnar.py      # Phase F
  utils/
    __init__.py
    metadata.py              # (exists)
```

## Dependencies

No new dependencies beyond what FabricPC already requires. CIFAR-10 loading
options:
- torchvision (add as optional dependency)
- Direct download + numpy (no new deps)

Prefer direct download to avoid torch dependency unless user already has it.

## Timeline

Not specified (per CLAUDE.md: no time estimates). Phases are ordered by
dependency, not calendar time.

## References

- Goertzel et al., "Hierarchical Bayesian Causal Modular Learning" (hibacaml_agi26.pdf)
- Section 6: "Scaling to Split-CIFAR" — CIFAR-10 configuration
- Section 4: "The ColBa Architecture for Split-MNIST" — column structure
