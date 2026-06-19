# HiBaCaML Columnar CIFAR-10 Accuracy Plan

Local timestamp: 2026-06-19 09:40:51 EDT -0400.

Machine identifier: `rogdora43`.

Source document: `/home/ni/repos/fpc/hibacaml_agi26.pdf`.

Target repository: `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`.

Target branch: `improve/plain-cifar10`.

## Objective

Improve plain CIFAR-10 10-class predictive-coding classification accuracy using columnar ideas from the HiBaCaML and ColBa paper.

Speed is not an optimization target in this phase. Accuracy is the target. Expensive predictive-coding inference, exact support audits, larger column pools, and longer training runs are acceptable if they improve validation and test accuracy.

The current plain CIFAR-10 predictive-coding reference point is the ResNet-style graph in `scripts/train_cifar10_pc_resnet.py`. The best completed two-epoch result is `leaky_relu` with 80 inference steps: 42.45% test accuracy and 42.26% validation accuracy.

## Scope

In scope:

- Plain CIFAR-10 with one 10-way classifier.
- Predictive-coding training through FabricPC `train_pcn` and FabricPC graph nodes.
- Upstream FabricPC convolution components from the `feature/convolution` branch.
- The existing FabricPC `Cifar10Loader`.
- Local columnar code in `columnarCL-fabricPC-experiments`.

Out of scope:

- Split-CIFAR task sequencing.
- Replay buffers.
- Task masks.
- Task-local heads.
- The cFabricPC data loader.
- Plain backpropagation training.
- Edits to FabricPC unless a missing upstream hook is proven necessary.

## Paper Mechanisms To Use

The paper defines a two-level architecture. The top level selects a sparse subset of columns for each context. The inside of each column tracks which internal structure is reusable, semi-general, task-local, or stale. The two levels communicate through internal certificates, and local counterfactual teachers audit nearby support changes.

For plain CIFAR-10, the adaptation is:

- A context is an image, an image cluster, or a mini-batch, not a task id.
- The support controller selects columns for one 10-class task.
- Internal certificates estimate column utility for current CIFAR-10 classes, not old-task retention.
- The support teacher optimizes validation cross-entropy and balanced class accuracy, not continual-learning retention.

## Symbol Table

`H` means the full HiBaCaML-style CIFAR-10 learner in this repo.

`N` means the set of column nodes in `H`.

`N_col` means the number of columns in `N`.

`N_shared` means the number of always-active shared columns.

`N_adaptive` means the number of non-shared columns available for support selection.

`S` means the active support set, which is the subset of non-shared columns used by a forward and predictive-coding inference pass.

`k` means the number of non-shared columns in `S`.

`C` means the combiner node that maps active column outputs to the representation consumed by the classifier.

`phi_c` means the internal certificate vector emitted by column `c`.

`psi` means the top-level support controller that proposes `S` from image context and column certificates.

`T` means the local counterfactual teacher that audits proposed support changes.

`J(S)` means the audited validation objective for support set `S`.

## Diagnosis Of The Current Columnar Implementation

The repository already contains a first columnar CIFAR-10 graph:

- `columnar_cl_fabricpc/experiments/cifar10_columnar.py` builds a 40-column model.
- `columnar_cl_fabricpc/columns/column.py` implements `ColumnarNode` with K, L, and B microcolumns.
- `columnar_cl_fabricpc/columns/combiner.py` implements attention, sum, and concat combiners.
- `columnar_cl_fabricpc/columns/visual_stem.py` includes a local convolutional stem and patch embedding nodes.

The current implementation is structurally close to the paper but misses the accuracy-critical mechanisms:

- All columns are active in the initial graph, so the top-level sparse support idea is inactive.
- K, L, and B microcolumns are named differently but currently run the same input projection, activation, and output projection pattern.
- The hard-kernel and shell split is documented but not represented as separate trainable paths.
- Internal certificates are not computed.
- The one-swap support teacher is not implemented.
- The strongest current accuracy result comes from the PC ResNet-style graph, not from the current columnar graph.

The plan should therefore graft the missing columnar mechanisms onto the validated convolutional predictive-coding baseline instead of replacing the working convolutional backbone with a weaker raw patch pipeline.

## Chosen Architecture

Use a shared convolutional visual stem built from upstream FabricPC `ConvNode` components, then add a sparse ColBa-style column layer before the classifier.

The first target architecture is:

```text
CIFAR-10 image
  -> ConvNode stem and residual feature stages
  -> tokenization of feature map
  -> N_col column nodes with typed K/L/B microcolumns
  -> masked sparse combiner C using support S
  -> one 10-way softmax cross-entropy classifier
```

This keeps the working convolutional mechanism that produced 42.45% test accuracy and adds the paper's columnar support-selection machinery where it can improve representation specialization.

Initial column-pool configuration:

- `N_col = 40`.
- `N_shared = 4`.
- `N_adaptive = 36`, using the paper's 30 adaptive columns plus 6 reserve columns as one selectable non-shared pool for plain CIFAR-10.
- `k = 5`, so each support uses 4 shared columns and 5 selected non-shared columns.
- Candidate support count is `choose(36, 5) = 376,992`, where `choose(36, 5)` is the number of 5-column subsets of the 36 non-shared columns.

Because speed is not the target, this candidate count is acceptable for offline support audits with cached column outputs or batched evaluation.

## Phase 1: Accuracy Anchor

Keep the current PC ResNet-style graph as the accuracy floor.

Run a longer baseline with the current best activation:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --activation leaky_relu --num_epochs 5 --batch_size 256 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

Then run an accuracy-only inference sweep:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --activation leaky_relu --num_epochs 5 --batch_size 256 --infer_steps 120 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --activation leaky_relu --num_epochs 5 --batch_size 256 --infer_steps 160 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

This tests whether the predictive-coding state settles better with more inference steps. The previous 40-step run lost accuracy, so shorter inference is not a candidate baseline.

Acceptance rule: the columnar architecture must beat the best plain PC ResNet validation accuracy before it replaces the baseline.

## Phase 2: Conv-To-Column Graph

Create a new experiment script, tentatively `scripts/train_cifar10_colba_accuracy.py`.

Use upstream FabricPC `ConvNode`, `SkipConnection`, and `AvgPool` for the shared visual stem. Do not use the cFabricPC loader. Do not change FabricPC.

Start from the validated ResNet-style graph and replace the direct `AvgPool -> Linear` classifier path with:

```text
last convolutional feature map
  -> token projection
  -> column pool
  -> masked sparse combiner
  -> classifier
```

The token projection should preserve enough spatial structure for the typed microcolumns:

- 16 tokens from a 4 by 4 grid for the first implementation.
- 64 tokens from an 8 by 8 grid if 16 tokens loses too much spatial detail.
- Token width 128 or 256 to match the later ResNet feature stages.

This phase should keep all 40 columns active first. The goal is to verify that adding columns does not reduce the validated convolutional baseline before sparse selection is introduced.

Acceptance rule: all-active conv-to-column accuracy should match or beat the PC ResNet baseline at the same epoch count and inference-step count. If it does not, fix the typed column computations before adding support selection.

## Phase 3: Make K/L/B Roles Operational

Replace the current symmetric K/L/B implementation with distinct pathways.

K microcolumn:

- Input: token sequence from the visual stem.
- Operation: per-token residual MLP with a protected hard-kernel path.
- Role: stable local feature transformation.

L microcolumn:

- Input: token grid from the visual stem.
- Operation: local token-neighborhood mixing, such as 3 by 3 grid convolution over tokens.
- Role: same-region discriminative refinement.

B microcolumn:

- Input: token sequence from the visual stem.
- Operation: global token pooling and broadcast, or low-rank token attention if local implementation stays inside the columnar repo.
- Role: cross-region context integration.

The current `ColumnarNode` applies the same projection pattern to K, L, and B. That makes the typed microcolumn idea mostly nominal. This phase makes the paper's typed roles visible in the actual computation.

Acceptance rule: K-only, K+L, K+B, and K+L+B ablations should show that L and B improve validation accuracy over K-only.

## Phase 4: Add Hard Kernels And Shell Paths

Implement a new local node, tentatively `ShellColumnarNode`, rather than changing FabricPC.

Each microcolumn has four paths:

- Hard kernel: protected path with width 32.
- Inner shell `S1`: reusable-abstraction path with width 10.
- Middle shell `S2`: semi-general path with width 20.
- Outer shell `S3`: exploratory path with width 30.

For plain CIFAR-10, shell paths are not pruned and are not task-specific. They are used as structured capacity and as a way to measure internal certificates.

Accuracy-oriented regularization:

- Apply lower weight decay to the hard kernel than to the shells.
- Apply diversity energy between same-tier shell outputs from different columns.
- Apply mild within-shell inhibition to reduce duplicate column responses.
- Do not prune shells in this phase.

The mechanism is that the hard kernel carries stable transformations, while shell paths provide additional class-sensitive capacity. The diversity energy reduces the chance that all columns learn the same representation.

Acceptance rule: shell-enabled columns should improve validation accuracy over the typed K/L/B columns without shells.

## Phase 5: Masked Sparse Combiner

Implement a local masked combiner node, tentatively `MaskedColumnCombinerNode`.

The graph keeps edges from all columns to the combiner. A support mask `m` selects which columns contribute to `C`, where `m_c = 1` means column `c` is active and `m_c = 0` means column `c` is inactive.

The first mask is global:

- The four shared columns are always active.
- The five selected non-shared columns come from one global support `S`.

This keeps graph topology static while allowing support audits to change `S`.

Combiner modes to test:

- Masked sum.
- Masked learned attention over active columns.
- Masked concat followed by projection.

Accuracy rule: use whichever combiner produces the best validation accuracy. Speed is not part of this decision.

## Phase 6: Exact Support Audit

Implement an offline support-audit harness.

For a trained or partially trained model, evaluate `J(S)` for candidate supports. `J(S)` is the audited validation objective for support `S`, defined as validation cross-entropy plus a class-balance penalty. The class-balance penalty is included so a support cannot improve mean loss by ignoring hard classes.

Start with global exact search:

- Evaluate all `choose(36, 5)` non-shared supports if cached column outputs make this feasible.
- If exact full evaluation is too memory-heavy, use exact search on a validation shard and confirm the best supports on the full validation set.
- Because speed is not the target, do not switch to a learned router before exact audits establish which supports work.

Then add one-swap teacher updates:

- A one-swap neighbor replaces one column in `S` with one inactive non-shared column.
- The teacher `T` accepts a one-swap only if it improves `J(S)`.
- After each epoch, run one-swap audits from the current support.
- Periodically run full exact audits to detect whether one-swap search is stuck.

Acceptance rule: exact or one-swap support selection must improve validation accuracy over all-active columns and over a random fixed support.

## Phase 7: Context-Conditioned Support

After global support improves accuracy, add context-conditioned support.

Context source:

- Use the shared visual stem's pooled feature vector for each image.
- Optionally include internal certificates `phi_c` from each column.

Controller:

- `psi` maps the image context to scores over the 36 non-shared columns.
- The selected support `S` is the top-5 non-shared columns plus the 4 shared columns.
- Train `psi` from exact teacher labels using predictive-coding supervision, not plain backprop.

Teacher labels:

- For each validation image or image cluster, audit a manageable candidate set of supports.
- The support with the lowest audited loss becomes the teacher target for `psi`.
- Since speed is not the target, use a large candidate set.

Inference:

- Use `psi` to propose a support.
- Optionally run test-time one-swap refinement using unlabeled predictive-coding energy and classifier confidence. This is a later accuracy experiment because the unlabeled objective may not match true accuracy.

Acceptance rule: context-conditioned support should beat the best global support on validation accuracy and per-class accuracy.

## Phase 8: Internal Certificates

Compute a certificate vector `phi_c` for each column `c`.

Initial certificate fields:

- Mean column energy: the average predictive-coding energy of column `c` on the validation set.
- Class selectivity: the variance of column `c` output means across the 10 CIFAR-10 classes.
- Ablation utility: the validation accuracy drop when column `c` is removed from the support.
- Similarity signature: the cosine similarity between column `c` class-mean outputs and other columns' class-mean outputs.
- Saturation pressure: the ratio between column output variance and shell capacity.

Use certificates in two places:

- Support scoring: `psi` receives `phi_c` as part of each column's score input.
- Teacher audit filtering: exact search can prioritize columns with high ablation utility and low redundancy.

The paper's Rao-Blackwell argument says that useful internal information should improve reuse-utility estimation. In this plain CIFAR-10 adaptation, reuse utility means expected contribution to 10-class accuracy.

Acceptance rule: support selection using certificates should beat support selection using validation fit scores alone.

## Phase 9: Hierarchical Bias Heads

Add hierarchical bias only after sparse column selection improves accuracy.

The paper uses quadrant and global auxiliary targets to bias local-to-global composition. For plain CIFAR-10, implement this as a local node in the columnar repo:

```text
column tokens
  -> quadrant pooled logits
  -> global pooled logits
  -> combined 10-way output
```

The node computes one main cross-entropy energy and optional auxiliary cross-entropy energies from quadrant pooled features. The target is still the same 10-way CIFAR-10 label. There are no task-local heads.

If FabricPC `TaskMap` cannot represent multiple supervised energies cleanly, keep the auxiliary energies inside one local classification node. Do not modify FabricPC unless this proves impossible in local code.

Acceptance rule: hierarchical bias heads should improve validation accuracy without reducing per-class balance.

## Phase 10: Experimental Protocol

Use the same data loader throughout:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m columnar_cl_fabricpc.experiments.cifar10_columnar --epochs 20
```

The command above is the existing columnar script. The new accuracy script should expose equivalent arguments plus support-selection controls.

Required runs:

- PC ResNet baseline with `leaky_relu`, 80 inference steps, 5 epochs.
- PC ResNet baseline with `leaky_relu`, 120 inference steps, 5 epochs.
- Conv-to-column all-active, 80 inference steps, 5 epochs.
- Conv-to-column all-active, best inference-step count from baseline, 5 epochs.
- Typed K/L/B ablations.
- Shell ablations.
- All-active versus random fixed support versus exact global support.
- Global support versus context-conditioned support.
- Certificate-free support scoring versus certificate-aware support scoring.

Metrics:

- Validation accuracy.
- Test accuracy for selected validation winners only.
- Validation cross-entropy.
- Per-class validation accuracy.
- Support-set usage counts.
- Column ablation utility.
- Mean predictive-coding energy by node group.

Decision rule:

- Choose models by validation accuracy and per-class validation accuracy.
- Report test accuracy only after selecting a configuration.
- Do not select by runtime.

## Implementation Order

1. Add the conv-to-column accuracy script using upstream FabricPC convolution nodes and the existing `Cifar10Loader`.
2. Add typed K/L/B column computations.
3. Add hard-kernel and shell paths.
4. Add masked sparse combiner support.
5. Add the offline exact support audit harness.
6. Add one-swap support teacher updates.
7. Add context-conditioned support controller `psi`.
8. Add internal certificates `phi_c`.
9. Add hierarchical bias heads.
10. Run the accuracy protocol and update the dev record with results.

## Risks And Checks

Risk: the columnar layer reduces the already modest PC ResNet accuracy.

Check: all-active conv-to-column must match or beat the PC ResNet baseline before sparse support work starts.

Risk: all columns collapse to similar representations.

Check: track class-mean column-output similarity and add diversity energy between same-tier shell outputs.

Risk: support search overfits the validation set.

Check: use train-audit, validation-audit, and final test splits. Pick support rules on validation only, then run the selected configuration once on test.

Risk: context-conditioned support labels are noisy.

Check: first prove global exact support improves accuracy, then train `psi` from exact support labels.

Risk: auxiliary hierarchy heads conflict with the main classifier.

Check: gate auxiliary energy weight from 0.0 to small values and select by validation accuracy.

## Alternatives Considered

Alternative 1: keep using the PC ResNet graph and tune only inference steps, learning rate, and weight decay.

- Pros: lowest implementation risk and already above the paper's reported two-epoch quick-run reference.
- Cons: does not test the paper's columnar support-selection mechanisms and may plateau without representation specialization.
- Decision: keep as the accuracy floor, but do not stop there.

Alternative 2: use the existing raw patch columnar graph and tune it directly.

- Pros: already implemented and close to the paper's high-level diagram.
- Cons: it discards the validated convolutional PC feature extractor, and CIFAR-10 needs stronger low-level visual abstraction than raw 8 by 8 patch projection.
- Decision: use the validated convolutional stem before the columnar layer.

Alternative 3: train all 40 columns active and rely on the attention combiner.

- Pros: simpler and fully differentiable through the predictive-coding graph.
- Cons: it bypasses the paper's sparse support mechanism and can let columns duplicate each other.
- Decision: use all-active only as a diagnostic baseline.

Alternative 4: implement a learned router first.

- Pros: closer to a deployable dynamic architecture.
- Cons: the paper's audit found learned selection can be suboptimal, and exact teacher data is the cleaner first source of support labels.
- Decision: implement exact support audit and one-swap teacher before training `psi`.

Alternative 5: add split-CIFAR mechanisms from cFabricPC now.

- Pros: connects to the strongest historical result found so far.
- Cons: it changes the problem from plain 10-class CIFAR-10 to continual learning.
- Decision: defer split-CIFAR until plain CIFAR-10 accuracy is strong.

Alternative 6: add data augmentation or a custom loader wrapper immediately.

- Pros: likely improves CIFAR-10 accuracy.
- Cons: the current instruction is to keep the upstream FabricPC loader, and this plan is specifically testing columnar mechanisms.
- Decision: do not change the loader in this phase.

## Expected First Milestone

The first meaningful milestone is not a final target accuracy. It is a validation result showing that the conv-to-column graph with all columns active can match or beat the `leaky_relu`, 80-step PC ResNet baseline.

The second milestone is a validation result showing that exact sparse support selection beats all-active columns.

Only after those two mechanisms work should the plan move to context-conditioned support and internal certificates.
