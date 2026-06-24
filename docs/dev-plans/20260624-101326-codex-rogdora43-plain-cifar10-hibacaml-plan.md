# Plain CIFAR-10 HiBaCaML Accuracy Plan

Created: 2026-06-24 10:13:26 EDT  
Machine: rogdora43  
Author: Codex  
Repository: columnarCL-fabricPC-experiments  
Branch observed: improve/plain-cifar10

## Goal

The immediate goal is to improve plain CIFAR-10 classification accuracy using predictive-coding mechanisms and columnar ideas from the HiBaCaML paper before moving back to Split-CIFAR. Plain CIFAR-10 should be treated as the substrate-building stage: it should establish that the visual stem, column pathway, support-selection machinery, and readout can classify ten classes robustly before we ask the same machinery to preserve knowledge across task splits.

This plan does not propose porting Split-CIFAR machinery. It does not propose changing the data loader. It also does not propose modifying FabricPC unless a shared upstream limitation is truly required. The expected implementation surface is this repository.

## Current Evidence

The current best live branch is no longer random chance, but it is not yet strong enough to be a stable base for continual learning. The important pattern is that longer runs at learning rate 0.005 are still improving at epoch 20, while the column branch appears weak relative to the bypass branch.

Completed runs seen on rogdora43:

| Run | Commit in log | Learning rate | Epochs | Seed | Best validation accuracy | Best epoch | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `resnet18_bypass_norm_fixedln` | `d60f6120dc8a0543ccdeca0c5ce0ea4ad1e8f9a6` | 0.005 | 20 | 42 | 48.50% | 20 | 47.10% |
| `resnet18_bypass_norm_fixedln` | `d60f6120dc8a0543ccdeca0c5ce0ea4ad1e8f9a6` | 0.005 | 20 | 99 | 46.32% | 20 | 45.23% |

Seed 7 with the same 20-epoch, learning-rate-0.005, no-diagnostics command was in progress when this plan was written. Its visible log had not reached a final summary.

Earlier saved results still matter as targets:

| Run family | Best observed test accuracy | Meaning |
| --- | --- | --- |
| Fixed layer-normalized ResNet-18 with bypass columns, seed 7 | 49.29% | Current work should at least recover this. |
| Fixed layer-normalized ResNet-18 with bypass columns, seed 42 | 48.13% | Current 20-epoch seed 42 is close but still below this. |
| Earlier stage-2 ResNet-18 bypass-normalized run | 49.22% | The current architecture has not yet beaten the simpler older setup. |
| 40-column, 9-active-column experiment | 44.26% | Scaling column count before making each active column useful made accuracy worse. |
| Predictive-coding ResNet baseline | about 42-43% | The current branch is above this baseline but not by enough. |

## Mechanistic Reading Of The HiBaCaML Paper

The HiBaCaML paper is not mainly a claim that more columns improve accuracy. It is a claim that a two-level modular learning system can control interference when the top level selects a sparse set of components and the inside of each component separates reusable structure from task-local residue.

The top-level controller, named `Psi` in this plan, is the mechanism that chooses a sparse active support for a context. A support `S(x)` is the subset of columns active for input or context `x`. In the paper, `Psi` is corrected by teacher-first audits, including exact or local one-swap evaluations of nearby supports.

Each column is internally structured. The paper's ColBa column has typed microcolumns `K`, `L`, and `B`, where `K` is the kernel pathway, `L` is the lateral pathway, and `B` is the bridge pathway. Each microcolumn has a hard kernel plus shells. The hard kernel stores protected reusable structure. Inner shells store reusable abstraction, middle shells store semi-general structure, and outer shells store task-local exploratory structure.

Each active column emits internal certificates. A certificate vector `r_c` is the summary emitted by column `c` that estimates how useful, reusable, saturated, or task-specific that column is. The controller uses these certificates to choose supports better than it could from external validation loss alone.

The Split-CIFAR recommendation in the paper scales the visual substrate and the controller together: a shallow shared visual stem, more columns, sparse active supports, typed microcolumns, teacher-first support audits, and conservative shell dynamics. The paper explicitly keeps the active fraction sparse. It does not recommend making every column active or relying on a dense bypass to carry the task.

## Strategic Diagnosis

The current plain CIFAR-10 implementation contains several pieces that resemble the paper: a ResNet-style visual stem, stage taps, typed depth-spanning column pathways, all-active columns, and a combiner. It does not yet contain the core control mechanisms that make HiBaCaML more than a wider network.

The most important current failure mode is column underuse. Diagnostics from previous runs showed that column outputs had reasonable per-token scale after layer normalization, but the averaged `column_pool` vector had very small standard deviation relative to `bypass_pool`. The classifier therefore appears to learn mainly from the direct stage-4 bypass. In that condition, Split-CIFAR experiments would mostly test whether the backbone and bypass forget, not whether a HiBaCaML-style columnar system works.

This changes the priority order. The next useful work is not to add Split-CIFAR, increase the column count to 40, or tune runtime speed. The next useful work is to make the column pathway carry measurable class information on plain CIFAR-10 while preserving predictive-coding training.

## Symbol Table

| Symbol | Meaning |
| --- | --- |
| `Psi` | The top-level support controller that chooses which columns are active for an input or context. |
| `S(x)` | The active column support selected for input or context `x`. |
| `K`, `L`, `B` | The typed microcolumn pathways inside each column: kernel, lateral, and bridge. |
| `z_col_pool` | The vector produced by averaging the combiner output over the token axis. |
| `z_col_norm` | The normalized column readout vector produced from `z_col_pool` before the classifier. |
| `r_c` | The internal certificate vector emitted by column `c`. |
| `h_c` | The protected hard-kernel feature slice inside column `c`. |
| `s_c,i` | The shell feature slice inside column `c`, where `i=1` is the inner shell, `i=2` is the middle shell, and `i=3` is the outer shell. |
| `d_route` | A directed depth route that projects information between non-adjacent backbone stages or non-adjacent shell depths inside a column. |

## Chosen Direction

The chosen direction is to first make the plain CIFAR-10 column pathway measurable and useful, then add a small teacher-first support mechanism on plain CIFAR-10, then expand toward Split-CIFAR only after those mechanisms pass ablation tests.

### Phase 0: Finish The Three-Seed Baseline

Finish the current seed 7 run and collect a three-seed table for the 20-epoch, learning-rate-0.005, no-diagnostics setting. Record commit, seed, GPU backend, validation trajectory, best validation epoch, and final test accuracy in results.

If all three completed runs peak at epoch 20, extend the same configuration to 30 or 40 epochs for one seed before changing architecture. The mechanism is simple: the validation curve is still rising, so the current training horizon may truncate learning. This should not distract from the column-use problem, but it should set a fair baseline before judging architectural changes.

### Phase 1: Normalize The Column Readout After Pooling

Add a local column readout normalization after `column_pool`. The current `z_col_pool` vector is produced by averaging token features after the combiner. Averaging can shrink feature variance across the token axis. A normalized `z_col_norm` vector should be passed to the classifier so that the column pathway enters the classifier with a scale comparable to the bypass pathway.

This should be implemented in this repository. A small node such as `PooledFeatureNormNode` or a reuse of an existing FabricPC-compatible layer-normalization node is acceptable if it works with the graph and predictive-coding training loop. This is a local architecture change, not a FabricPC change.

Required tests:

- Train with `z_col_norm` and the existing bypass enabled.
- Evaluate the trained model with the bypass contribution removed.
- Evaluate the trained model with the column contribution removed.
- Evaluate the trained model with both contributions present.

The ablation should report validation accuracy for each condition. The direct test is whether removing the column pathway changes accuracy after training. If it does not, the column branch is still not part of the classifier.

### Phase 2: Keep The Bypass, But Stop Letting It Hide Column Failure

The bypass is useful as a non-regression path because it keeps the current model above the predictive-coding ResNet baseline. It should remain available while the column path is repaired. However, every run should report whether the classifier is using the column pathway.

The minimum reporting change is an evaluation-time ablation. A stronger version is to add a learnable or scheduled mixture between `z_col_norm` and `bypass_pool`, but that should come after ablation reporting. A hard zero column gate at initialization should be avoided because it can block the gradient path into the column branch. A small nonzero column scale or scale-normalized readout is safer.

### Phase 3: Add Plain-CIFAR Support Selection Before Split-CIFAR

Once columns carry measurable information, add a small teacher-first support selector on plain CIFAR-10. The context does not need to be a continual-learning task yet. It can be one of:

- The class label during supervised training.
- A feature cluster computed from the shared visual stem.
- An augmentation regime such as crop severity or color perturbation.

For a small column pool, `Psi` should choose `S(x)` by evaluating a limited set of candidate supports and applying one-swap audits. A one-swap audit replaces one active non-shared column with one inactive candidate and accepts the replacement only if the audited validation objective improves. This maps the paper's teacher-first mechanism onto plain CIFAR-10 without introducing split-task machinery.

Start small, for example 8 to 12 total columns with 2 or 3 non-shared active columns. The earlier 40-column experiment was worse, so increasing column count should wait until support selection and readout normalization are working.

### Phase 4: Add Internal Certificates

After support selection exists, add per-column certificate logging. The first certificate vector `r_c` should be simple and directly measurable:

- Mean activation magnitude for column `c`.
- Class-conditioned usefulness of column `c`, measured by validation loss change when column `c` is removed.
- Gradient or predictive-coding update magnitude for column `c`.
- Saturation pressure for column `c`, measured by activation concentration or repeated high-norm use.
- Redundancy for column `c`, measured by correlation with other active columns.

These certificates should first be logged, then used by `Psi`. The paper's Rao-Blackwell argument matters only if the certificate contains information about reuse utility that is not already present in the base score. The implementation should therefore test whether certificates improve support choice on held-out validation batches.

### Phase 5: Add Shell Semantics Only After The Column Path Works

The hard-kernel and shell design is central to the paper, but adding it before the column pathway carries signal creates too many degrees of freedom. Once `z_col_norm`, ablation reporting, and small support selection work, add shell semantics in a minimal form:

- A protected hard-kernel slice that is not pruned.
- Inner, middle, and outer feature slices inside each `K`, `L`, and `B` pathway.
- Stronger inhibition among same-tier slices that represent overlapping features.
- Conservative outer-shell pruning only after warmup.

The plain CIFAR-10 test is whether shell structure improves accuracy or stability across seeds before it is used for continual learning.

The implementation should partition each column's feature width into `h_c` plus `s_c,1`, `s_c,2`, and `s_c,3`. The hard-kernel slice `h_c` is always present in the forward and predictive-coding inference graph. The inner shell `s_c,1` should receive the strongest regularity pressure because it represents reusable structure. The middle shell `s_c,2` should receive moderate regularity pressure because it represents semi-general structure. The outer shell `s_c,3` should receive the weakest stability pressure because it represents task-local exploratory structure.

This does not require pruning at the first implementation step. The first step should only create the slices, route them separately, log their norms, and report their ablation effects. Pruning, promotion, and demotion should wait until the shell slices can be measured.

### Phase 5A: Add Depth-Skip Routes Inside Columns

The current depth-spanning column already consumes multiple backbone stages, but it mostly compresses them into one token stream. A deeper columnar architecture should let a column carry non-adjacent depth information through explicit directed routes. A `d_route` is a learned projection from one depth source to a non-adjacent target inside the same column, for example from the stage-2 token map to the stage-4 shell state, or from an early shell slice to a later shell slice.

The biological analogy should stay secondary to the implementation. In code, the mechanism is a small set of directed graph edges with separate parameters, normalization, and ablation labels. These edges let fine spatial features from shallow stages influence deeper class features without requiring every update to pass through adjacent stage transformations.

Start with three route families:

- Bottom-up depth routes from stage 2 to stage 4 shell states. These routes test whether shallow spatial detail helps class decisions after the normal stage hierarchy has compressed the image.
- Top-down prediction routes from stage 4 shell states to stage 2 token predictions. These routes test whether deeper class features improve predictive-coding inference over shallow token states.
- Cross-shell depth routes from `s_c,3` to `s_c,1` through a gated projection. These routes test whether task-local exploratory structure can contribute to reusable inner-shell structure without directly overwriting `h_c`.

Each route family should have an explicit lesion switch for evaluation. A lesion switch means the route output is set to zero during evaluation while all trained parameters remain unchanged. This creates a direct test of whether the trained model uses that route.

### Phase 5B: Make Concentric Shells Measurable Before Making Them Adaptive

The shell implementation should deepen the experimental design as much as the architecture. Each run should report shell-resolved statistics:

- Norm and standard deviation for `h_c`, `s_c,1`, `s_c,2`, and `s_c,3`.
- Class-conditioned ablation accuracy for each shell slice.
- Route-conditioned ablation accuracy for each `d_route` family.
- Correlation between shell slices in different columns.
- Predictive-coding energy contribution by shell slice when that is available from the graph.

The most important first result is not whether shell pruning improves accuracy. The first result is whether the shells specialize. If `h_c`, `s_c,1`, `s_c,2`, and `s_c,3` have indistinguishable ablation effects and indistinguishable class usage, then the architecture is only a larger dense column. If the shell slices show different ablation profiles, then pruning, promotion, and demotion become meaningful next steps.

### Phase 5C: Deepen Plain-CIFAR Testing

Plain CIFAR-10 should become a richer testbed before Split-CIFAR. The current summary metric, final test accuracy, is necessary but too coarse to judge a columnar architecture. Add a fixed evaluation suite derived from the same CIFAR-10 validation and test images, without changing the data loader:

- Clean validation and test accuracy.
- Per-class accuracy and confusion matrix.
- Column-only, bypass-only, and combined accuracy.
- Hard-kernel-only, inner-shell-only, middle-shell-only, and outer-shell-only accuracy.
- Route-lesioned accuracy for each `d_route` family.
- Accuracy under deterministic occlusion patches, grayscale conversion, and crop severity transforms.
- Calibration summaries such as negative log likelihood and expected calibration error if logits are available.

The deterministic perturbation probes should be generated from the existing CIFAR-10 arrays at evaluation time. They should use fixed seeds and fixed transform settings. This keeps the loader stable while testing whether columnar mechanisms help with nuisance variation.

The expected pattern for a useful shell architecture is not that every shell improves every metric. A better pattern would be that `h_c` and `s_c,1` support clean accuracy and stable class identity, while `s_c,3` and selected `d_route` paths contribute more under occlusion, crop shifts, or class-specific confusions. That would give the later Split-CIFAR work a concrete reason to preserve inner structure and adapt outer structure.

### Phase 6: Move Back To Split-CIFAR

Return to Split-CIFAR only when plain CIFAR-10 passes these conditions:

- Three-seed median test accuracy improves over the older 49% target, or there is clear evidence that the architecture change improves validation accuracy and column contribution even before matching that target.
- Removing the column pathway after training lowers validation accuracy by a measurable amount.
- Removing the bypass pathway after training still leaves column-only accuracy well above chance.
- Lesioning shell slices or depth routes produces interpretable accuracy changes rather than near-zero changes everywhere.
- The support selector's one-swap audit finds and applies local improvements on plain CIFAR-10 contexts.
- Runs are reproducible from a recorded config, seed, commit, and backend.

At that point, the Split-CIFAR work should add task heads and continual-learning accounting around a column pathway that already works. It should not use Split-CIFAR to discover whether the plain classifier can learn.

## Alternatives Considered

### Continue Learning-Rate And Epoch Sweeps

Pros:

- It is low risk and gives cleaner baselines.
- The 20-epoch runs peak at epoch 20, so longer runs are justified.

Cons:

- It can improve the bypass-dominant classifier without repairing the column pathway.
- It does not test the HiBaCaML mechanisms that matter for later continual learning.

Decision: keep a limited longer-run check, but do not make sweeps the main work.

### Scale Immediately To The Paper's 40-Column CIFAR Configuration

Pros:

- It is closer to the Split-CIFAR configuration recommended in the paper.
- It creates enough columns for sparse support selection.

Cons:

- A previous 40-column, 9-active-column run was worse than the smaller model.
- More columns do not help if the pooled column readout is scale-suppressed.
- All-active or poorly selected large column pools can increase interference.

Decision: postpone large column counts until small-column ablations prove that columns carry class information.

### Port Split-CIFAR Machinery Now

Pros:

- It moves directly toward the eventual benchmark.
- It would exercise support selection under real task contexts.

Cons:

- It mixes classification weakness with continual-learning weakness.
- It would likely measure the backbone and bypass more than the columnar mechanism.
- It violates the current goal of plain CIFAR-10 classification first.

Decision: do not port split machinery now.

### Implement Full Hard-Kernel And Shell Semantics Now

Pros:

- It is the most faithful implementation of the paper's internal column structure.
- It creates the substrate needed for conservative pruning and reuse certificates.
- It makes non-adjacent depth routes and shell-specific certificates possible.

Cons:

- It introduces many new parameters before the current readout bottleneck is fixed.
- It makes failed runs harder to diagnose because pooling, selection, shells, and certificates would all change at once.

Decision: implement readout normalization and ablations first, then support selection, then shell semantics.

### Add Unrestricted Non-Adjacent Connections Throughout The Graph

Pros:

- It could increase representational capacity quickly.
- It might recover useful shallow spatial information that the stage hierarchy loses.

Cons:

- It would be hard to distinguish a columnar mechanism from a dense residual network with extra edges.
- It would make predictive-coding energy harder to interpret because many unstructured routes could explain the same target.
- It would weaken the later Split-CIFAR test because there would be no clear inner-shell, outer-shell, or route-level unit to preserve or adapt.

Decision: add a small number of named `d_route` families inside columns, each with evaluation-time lesion reporting.

### Normalize Column Readout And Add Ablations First

Pros:

- It directly targets the observed column-scale failure.
- It preserves the current data loader and predictive-coding training path.
- It creates a clear yes-or-no test for whether columns matter.

Cons:

- It is less ambitious than the full HiBaCaML design.
- It may reveal that the current typed column implementation needs deeper changes.

Decision: this is the chosen next step.

## Immediate Implementation Checklist

1. Add a local normalized column readout from `column_pool` to the classifier.
2. Add evaluation-time ablation reporting for column-only, bypass-only, and combined validation accuracy.
3. Run seed 42 with the 20-epoch, learning-rate-0.005, no-diagnostics configuration and record the result.
4. If seed 42 improves and ablation shows column contribution, run seeds 99 and 7.
5. If column contribution is still near zero, test an attention or concatenation readout before changing column count.
6. After column contribution is measurable, implement a small plain-CIFAR support selector with one-swap audits.
7. After support selection works, split each column into `h_c`, `s_c,1`, `s_c,2`, and `s_c,3` and add shell-resolved logging.
8. Add a small set of named depth-skip route families inside columns and report route-lesioned accuracy.
9. Add the deterministic CIFAR-10 evaluation suite: per-class metrics, shell lesions, route lesions, and fixed perturbation probes.
10. Add certificate logging and only then decide whether pruning, promotion, and demotion should be activated.

## Command Discipline

Every new experiment command should write its own log under `results/` and include commit, backend, full arguments, and final summary. Prefer the repository runner script over hand-pasted multi-line shell commands so the command is robust to console indentation.

The training loop should remain predictive-coding based. The goal is not to replace the mechanism with ordinary backpropagation. The relevant controls remain inference steps, inference step size, predictive-coding energy behavior, architectural support, and column readout behavior.
