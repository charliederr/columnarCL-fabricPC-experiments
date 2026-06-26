# CIFAR-10 HiBaCaML Work Log

Created: 2026-06-24  
Machine: rogdora43  
Author: Codex  
Repository: columnarCL-fabricPC-experiments  
Branch observed: improve/plain-cifar10

## Phase 1 Start

Phase 0 is complete per the user. Phase 1 implements a normalized column readout after `column_pool`, where `column_pool` is the vector produced by global averaging over the combiner token axis. The purpose is to pass a scale-stabilized column vector to the classifier and then measure whether the classifier uses that column path.

Implementation direction:

- Add a local graph node that computes `z_col_norm`, the layer-normalized readout of `column_pool`.
- Route `z_col_norm` rather than raw `column_pool` into the classifier.
- Keep `bypass_pool`, the global average-pooled stage-4 backbone vector, available for non-regression.
- Add validation and test ablations by zeroing named classifier input-edge weights for evaluation only.
- Keep training on FabricPC predictive-coding `train_pcn`; do not introduce ordinary backpropagation.

## Phase 1 Implementation

Implemented files:

- `columnar_cl_fabricpc/columns/accuracy_nodes.py`
- `columnar_cl_fabricpc/columns/__init__.py`
- `scripts/train_cifar10_depth_spanning.py`
- `tests/test_pooled_readout_norm.py`

Mechanism:

- Added `PooledFeatureNormNode`, a local FabricPC-compatible node that takes a pooled vector with shape `(batch, feature_dim)` and applies LayerNorm along `feature_dim`.
- Added `column_readout_norm`, the normalized column readout node after `column_pool`.
- Changed the classifier path from `column_pool -> output` to `column_pool -> column_readout_norm -> output`.
- Kept `bypass_pool -> output` when `--bypass_columns` is present.
- Added `evaluate_readout_ablations`, which evaluates copied parameter trees with selected output input-edge matrices zeroed.
- The ablation cases are `combined`, `column_only`, and `bypass_only` when the bypass edge exists.
- The ablation helper requires the `column_readout_norm` edge. This avoids silently evaluating the old raw `column_pool` path.

The training path still uses FabricPC predictive-coding `train_pcn`. The ablation path only changes output-node weight matrices in copied parameters during evaluation.

## Verification

Commands run:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/accuracy_nodes.py columnar_cl_fabricpc/columns/__init__.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: `5 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --quick --bypass_columns --layer_norm_tokens --fix_ln_gamma
```

Result: passed. The quick smoke test completed training and printed validation and test readout ablation tables. The quick run is intentionally too short to interpret for accuracy.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py
```

Result: `121 passed`.

Full `pytest -q` was also run. It produced 129 passes plus CIFAR data-test failures because the restricted session could not create `/home/ni/.local/share/columnar_cl_fabricpc` for the data cache. The failing tests were in `tests/test_cifar_data.py`; they failed before loading data because the home data-cache directory was read-only in this environment.

## Next Experiment Command

Use the repository runner or the direct command with the same training settings as the completed Phase 0 runs. The output now includes readout ablation tables at the end.

```bash
./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 nodiag 20
```

The first result to inspect is not only combined test accuracy. Inspect whether `column_only` validation accuracy rises above chance and whether `bypass_only` differs from `combined`.

## Phase 1 Seed 42 Result

Result log:

- `results/codex_resnet18_bypass_norm_fixedln_seed42_lr0p005_ep20_nodiag_rogdora43_20260624_170421.log`

Run metadata from the log:

- Commit at run start: `7af3fbee25c415559fcde67996176a3324102c70`
- Seed: 42
- Learning rate: 0.005
- Epochs: 20
- Graph: 42 nodes, 65 edges
- Parameters: 2,924,634
- JAX devices: `[CudaDevice(id=0)]`
- JAX backend: `gpu`

Validation trajectory:

| Epoch | Validation accuracy |
| --- | --- |
| 1 | 26.98% |
| 2 | 33.30% |
| 3 | 36.56% |
| 4 | 38.20% |
| 5 | 39.44% |
| 6 | 35.16% |
| 7 | 16.08% |
| 8 | 21.52% |
| 9 | 9.30% |
| 10-20 | 9.84% each epoch |

Best validation epoch was epoch 5. Best-validation test accuracy was 39.45%.

Readout ablations at the selected epoch:

| Split | Combined | Column-only | Bypass-only |
| --- | --- | --- | --- |
| Validation | 39.44% | 9.98% | 33.84% |
| Test | 39.45% | 10.00% | 34.74% |

Interpretation grounded in this run:

- `column_only` is chance on validation and test, so the normalized column readout is not independently class-informative.
- `bypass_only` is below the prior Phase 0 combined seed-42 result, so inserting `column_readout_norm` did not merely fail to help the columns; it also changed training enough to weaken the bypass path.
- The validation curve collapses after epoch 5 and stays at chance from epoch 9 onward. This points to a training stability problem introduced by the Phase 1 graph change, not just a weak final classifier.

Direction change:

The next sub-goal should not be another 20-epoch seed run of the same graph. The next sub-goal should isolate whether the extra predictive-coding node energy from `column_readout_norm` is destabilizing training. The specific test is to compare the current graph against a variant that normalizes the column readout for the classifier without adding a separate Gaussian-energy latent node. If that variant preserves the prior bypass result while still exposing column-only ablation, then the failure was the added predictive-coding node. If it also fails, then feature normalization itself is disrupting the classifier input geometry.

## Single-Node Readout Isolation Implementation

Implemented the isolation variant requested after the negative Phase 1 seed-42 run.

Changed files:

- `columnar_cl_fabricpc/columns/accuracy_nodes.py`
- `columnar_cl_fabricpc/columns/__init__.py`
- `scripts/train_cifar10_depth_spanning.py`
- `tests/test_pooled_readout_norm.py`

Mechanism:

- Replaced the failed two-node path `combiner -> column_pool -> column_readout_norm -> output`.
- Added `GlobalAvgPoolNormNode`, a local node that first globally averages over every non-batch, non-feature axis and then applies LayerNorm along the feature axis.
- The depth-spanning graph now uses `combiner -> column_readout_norm -> output`.
- The raw `column_pool` node is not present in this graph.
- Readout ablations still use the `column_readout_norm` edge for `column_only` and the `bypass_pool` edge for `bypass_only`.

This tests whether the collapse was caused by adding a second predictive-coding latent on the column readout path. The graph now has the same readout-path node count as the pre-Phase-1 raw-pooling graph, but the classifier still receives a normalized column vector.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/accuracy_nodes.py columnar_cl_fabricpc/columns/__init__.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: `5 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --quick --bypass_columns --layer_norm_tokens --fix_ln_gamma
```

Result: passed. The quick graph reported 23 nodes and 33 edges, one fewer node and one fewer edge than the failed two-node quick graph. The run printed validation and test readout ablation tables. The quick run is too short to interpret for accuracy.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py
```

Result: `121 passed`.

Next full isolation run:

```bash
./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 nodiag 20
```

Interpretation rule:

- If validation no longer collapses and `bypass_only` recovers near the prior Phase 0 seed-42 level, then the added second readout latent was the destabilizing change.
- If validation still collapses or `bypass_only` remains far below Phase 0, then normalizing the column readout itself is disrupting the classifier input geometry or the classifier's local predictive-coding update.

## Single-Node Readout Isolation Result

Result log:

- `results/codex_resnet18_bypass_norm_fixedln_seed42_lr0p005_ep20_nodiag_rogdora43_20260624_225259.log`

Run metadata from the log:

- Commit at run start: `e392e9b6c12635c1b0369cfaf965372b8ac0d1d7`
- Seed: 42
- Learning rate: 0.005
- Epochs: 20
- Graph: 41 nodes, 64 edges
- Parameters: 2,924,634
- JAX devices: `[CudaDevice(id=0)]`
- JAX backend: `gpu`

Validation trajectory:

| Epoch | Validation accuracy |
| --- | --- |
| 1 | 29.16% |
| 2 | 34.26% |
| 3 | 37.48% |
| 4 | 41.16% |
| 5 | 43.20% |
| 6 | 40.94% |
| 7 | 42.44% |
| 8 | 44.46% |
| 9 | 38.78% |
| 10 | 42.68% |
| 11 | 41.08% |
| 12 | 43.40% |
| 13 | 40.42% |
| 14 | 32.94% |
| 15 | 31.26% |
| 16 | 27.34% |
| 17 | 27.22% |
| 18 | 26.20% |
| 19 | 27.54% |
| 20 | 27.80% |

Best validation epoch was epoch 8. Best-validation test accuracy was 43.39%.

Readout ablations at the selected epoch:

| Split | Combined | Column-only | Bypass-only |
| --- | --- | --- | --- |
| Validation | 44.46% | 10.00% | 18.16% |
| Test | 43.39% | 10.00% | 17.41% |

Interpretation grounded in this run:

- Removing the extra readout latent avoided the complete chance-level collapse seen in the two-node readout graph.
- The run still underperformed the Phase 0 raw-pooling seed-42 baseline, which reached 48.50% validation accuracy and 47.10% test accuracy.
- `column_only` is chance on validation and test. The normalized column feature is not an independently class-informative representation under this training setup.
- `bypass_only` is far below the combined result. The output node is using a joint sum of the normalized-column edge logits and bypass edge logits, but neither edge is useful after the other edge is removed.
- `combined` being higher than both ablated cases means the normalized column edge contributes to the trained logits. It does not show that the column branch has learned a reusable HiBaCaML-style column representation.

Direction change:

The next sub-goal should not be seed 99 or seed 7 for the single-node normalized-readout graph. Readout normalization has now been tested in two forms. The two-node form collapsed to chance, and the single-node form avoided the hardest collapse but still lost accuracy relative to raw pooling and left `column_only` at chance.

The next sub-goal should implement a more faithful HiBaCaML column mechanism while keeping the stable raw-pooling readout for measurement. The concrete target is a shell-structured typed column:

- Partition each depth-spanning column into typed feature slices for hard-kernel, inner-shell, middle-shell, and outer-shell pathways.
- Keep the current predictive-coding training path.
- Keep the upstream FabricPC CIFAR-10 loading path.
- Keep the raw pooled readout and bypass path initially, so shell implementation is not confounded with the normalized-readout failure mode.
- Add shell lesion diagnostics that zero one typed slice at a time and report combined accuracy, column-path accuracy, and bypass-path accuracy.
- Add shell norm diagnostics that report the L2 norm of each typed slice, where the L2 norm is the square root of the sum of squared feature activations over that slice.

This next step follows the HiBaCaML architecture direction more directly than further readout experiments. It tests whether structured column internals can produce useful class evidence before adding split-CIFAR-10 continual-learning machinery.

## Shell-Typed Column Implementation

Implemented the shell-structured column step after the normalized-readout result showed that readout normalization was not the right next direction.

Changed files:

- `columnar_cl_fabricpc/columns/depth_spanning_column.py`
- `columnar_cl_fabricpc/columns/__init__.py`
- `scripts/train_cifar10_depth_spanning.py`
- `scripts/run_codex_cifar10_depth_spanning.sh`
- `tests/test_depth_spanning_column.py`
- `tests/test_pooled_readout_norm.py`

Mechanism:

- Added `SHELL_NAMES`, the ordered shell names on the column feature axis: `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`.
- Added `compute_shell_sizes(output_dim)`, where `output_dim` is the width of the column output feature axis. The default proportions are 32:10:20:30, matching the paper's CIFAR hard-kernel and shell-width ratio. For `output_dim=64`, the shell widths are 22, 7, 14, and 21.
- Replaced the old global `path_scale` with `shell_path_scale`, a 4 by 3 matrix. The rows are the four shells. The columns are the K pathway, L pathway, and B pathway.
- Applied a fixed structural mask to `shell_path_scale`: `hard_kernel` can use K only, `inner_shell` can use K and L, `middle_shell` can use K, L, and B, and `outer_shell` can use L and B. The active entries are learnable, but the masked zero entries keep the pathway semantics fixed.
- Restored the stable raw readout path: `combiner -> column_pool -> output`. This replaces the experimental normalized readout path in the active depth-spanning graph.
- Added `--diagnose_shells`. This logs shell L2 norms before and after training and prints shell readout ablation tables.
- Added shell readout lesions by masking rows of the `column_pool -> output` classifier matrix. This leaves the predictive-coding graph and trained parameters otherwise unchanged during evaluation.

Interpretation of the shell lesion:

- `combined_without_hard_kernel` means the model is evaluated with the hard-kernel feature rows of the `column_pool -> output` matrix set to zero, while the bypass edge remains present.
- `column_without_hard_kernel` means the bypass edge is removed and the hard-kernel feature rows are set to zero.
- `column_hard_kernel_only` means the bypass edge is removed and only hard-kernel feature rows remain on the column readout edge.
- The same naming applies to `inner_shell`, `middle_shell`, and `outer_shell`.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/depth_spanning_column.py tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: `21 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py
```

Result: `123 passed`.

Quick smoke command run in the sandbox:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --quick --bypass_columns --layer_norm_tokens --fix_ln_gamma --diagnose_shells
```

Result: passed. The sandbox reported `CUDA_ERROR_NO_DEVICE`, so this smoke run used CPU. The quick run trained only four batches and is not an accuracy result. It verified graph construction, shell norm logging, raw readout ablations, and shell readout ablation tables.

Quick smoke details:

- Tiny quick graph: 23 nodes, 33 edges.
- Tiny quick shell widths: hard kernel 11, inner shell 4, middle shell 7, outer shell 10.
- The shell tables printed for validation and test.

Next full experiment for rogdora43:

```bash
./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20
```

This command writes its output to `results/`, records commit and backend, runs the shell-typed depth-spanning architecture with raw `column_pool` readout, and enables shell diagnostics without per-epoch energy diagnostics.

## 2026-06-25 Column Teacher Direction

Timestamp and machine: 2026-06-25 08:45:36 EDT on `rogdora43`.

Starting commit before the change: `20b075c`.

Latest result ingested:

- Result file: `results/codex_resnet18_bypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260625_061924.log`
- Main combined validation accuracy: 49.64% at epoch 20.
- Main combined test accuracy: 47.85%.
- `column_only` accuracy through the main output readout: 10.00% on validation and test.
- `bypass_only` accuracy through the main output readout: 49.16% on validation and 47.85% on test.
- Shell readout lesions had almost no effect on combined accuracy.
- Shell norms changed substantially: the hard-kernel slice grew while the middle-shell and outer-shell slices shrank.

Interpretation:

`column_pool` is the global average pooled output of the depth-spanning HiBaCaML-style columns. `bypass_pool` is the global average pooled backbone path from ResNet stage 4. `output` is the main CIFAR-10 classifier that receives both `column_pool` and optional `bypass_pool`. The shell-typed column changed its internal feature allocation, but the CIFAR-10 class error still reached the supervised output mostly through `bypass_pool`. The mechanism is visible in the ablations: `column_only` stayed at chance while `bypass_only` matched the combined classifier on test.

Chosen next mechanism:

Add `column_teacher_output`, a second terminal predictive-coding classifier fed only by `column_pool`. Add `column_y`, a training task key that carries the same one-hot CIFAR-10 label tensor as `y`, where `y` is the class target for the main combined output. FabricPC clamps every batch key that appears in the graph `TaskMap` during `train_pcn`, so `column_y` makes `column_teacher_output` a supervised CE node during predictive-coding inference and parameter learning. CE means cross entropy, the energy used by the class-label terminal node.

This keeps the HiBaCaML direction intact. The column branch now receives direct class evidence while the main classifier still measures the combined backbone and column readout. It does not replace the upstream CIFAR loader. It wraps each training batch after loading so the training task dictionary contains `x`, `y`, and `column_y`.

Alternatives considered:

- Remove `bypass_pool`. This would force the main output to use columns, but it would also discard the comparison path that tells us whether columns help beyond the backbone.
- Increase columns or switch the combiner to attention. This is architecturally relevant, but the previous result shows that more column capacity can still be unused if class error has an easier path through `bypass_pool`.
- Add shell-specific regularization before the teacher head. This could make the shell norms look more balanced, but it would optimize a branch whose class readout is still at chance.

Implemented files:

- `scripts/train_cifar10_depth_spanning.py`
- `scripts/run_codex_cifar10_depth_spanning.sh`
- `tests/test_pooled_readout_norm.py`

Implemented mechanism:

- Added `COLUMN_TEACHER_TARGET = "column_y"` and `COLUMN_TEACHER_NODE = "column_teacher_output"`.
- Added `column_teacher_output`, a softmax linear classifier with CE energy.
- Connected `column_pool -> column_teacher_output`.
- Added `TaskMap(x=image, y=output, column_y=column_teacher_output)`.
- Added `ColumnTeacherTargetLoader`, which wraps the existing CIFAR loader and duplicates the one-hot `y` tensor into `column_y`.
- Added `evaluate_output_node`, which evaluates any named classifier node by temporarily pointing the evaluation `y` task at that node.
- Added final validation and test metrics for `column_teacher_output`.
- Updated the runner log path to include `column_teacher`.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: `8 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: `23 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py
```

Result: `125 passed`.

Next CUDA-backed experiment for `rogdora43`:

```bash
./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20
```

What to inspect after the run:

- Main combined test accuracy.
- `column_only` test accuracy through the main `output` readout.
- `column_teacher_output` validation and test accuracy.
- Shell readout lesions.
- Shell L2 norms before and after training, where L2 norm is the square root of the sum of squared activations over a shell feature slice.

## 2026-06-25 Weighted Column Teacher Follow-Up

Timestamp and machine: 2026-06-25 15:22:41 EDT on `rogdora43`.

Starting commit before this follow-up: `e4cd180`.

Completed result ingested:

- Result file: `results/codex_resnet18_column_teacher_bypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260625_085006.log`
- Main combined validation accuracy peaked at 36.70% at epoch 3.
- Main combined test accuracy at the selected epoch was 38.08%.
- The main validation metric collapsed to 9.84% from epoch 10 through epoch 20.
- `column_only` test accuracy through the main `output` readout was 10.10%.
- `bypass_only` test accuracy through the main `output` readout was 38.07%.
- `column_teacher_output` test accuracy was 11.78%.
- Training energy rose to roughly 14 by epoch 20, while the previous no-teacher shell run finished with very low reported training energy.

Interpretation:

`column_teacher_output` is the auxiliary classifier attached only to `column_pool`. `column_y` is the same CIFAR-10 one-hot class target as `y`, clamped to `column_teacher_output` during training. Giving `column_teacher_output` full weight made the auxiliary CE target as strong as the main class target. The column-only teacher did not become class-informative, and its error dominated the shared column latents enough to destabilize the main output after a few epochs. This is a scale mismatch in the auxiliary predictive-coding target, not a reason to remove the column teacher mechanism.

Chosen next mechanism:

Keep the same column-only teacher head, but make its CE energy explicitly weighted. `w_teacher` is the scalar multiplier on the `column_teacher_output` CE energy. The main output keeps weight 1.0. The next default is `w_teacher = 0.1`, so the teacher head remains in the predictive-coding graph but contributes a smaller class-error signal to the shared column pathway.

Alternatives considered:

- Remove the teacher head. This restores the previous baseline behavior but abandons the class-target pressure on the column pathway.
- Increase `w_teacher`. The full-weight run already showed high-energy collapse, so increasing the same objective is not supported by the result.
- Move immediately to shell-local targets. This is likely the next architectural step if a low-weight teacher still stays at chance, but it is a larger change. The weighted teacher first tests whether the previous failure was target scale rather than target placement.

Implemented files:

- `columnar_cl_fabricpc/columns/label_smoothed_ce.py`
- `columnar_cl_fabricpc/columns/__init__.py`
- `scripts/train_cifar10_depth_spanning.py`
- `scripts/run_codex_cifar10_depth_spanning.sh`
- `tests/test_pooled_readout_norm.py`

Implemented mechanism:

- Added `WeightedLabelSmoothedCrossEntropyEnergy`, where `z_latent` is the one-hot class target, `z_mu` is the softmax prediction, and `weight` multiplies the per-sample CE energy and latent gradient.
- Changed `make_classifier_energy(label_smoothing, weight)` to use the weighted CE implementation for classifier heads.
- Kept the main `output` classifier at weight 1.0.
- Set `column_teacher_output` to `args.column_teacher_weight`, defaulting to 0.1.
- Added `--column_teacher_weight` to the training script.
- Added a fifth runner argument for the teacher weight and recorded it in the result filename and log metadata.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/label_smoothed_ce.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: `9 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: `24 passed`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py
```

Result: `126 passed`.

Next CUDA-backed experiment for `rogdora43`:

```bash
./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.1
```

What to inspect after the run:

- Main combined validation trajectory. If it still collapses to chance, the teacher target is still disrupting shared latents.
- `column_teacher_output` validation and test accuracy. If it stays near chance while the main classifier recovers, the teacher target is too high-level for the current column representation.
- `column_only` through the main `output` readout. If this rises above chance, the combined classifier has begun using column class evidence.

## 2026-06-25 Weighted Teacher 0.1 Result

Timestamp and machine: 2026-06-25 19:04:28 EDT on `rogdora43`.

Current commit while recording this result: `340a56c`.

Completed result ingested:

- Result file: `results/codex_resnet18_column_teacher0p1_bypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260625_165740.log`
- `w_teacher = 0.1`, where `w_teacher` is the scalar multiplier on the `column_teacher_output` cross-entropy energy.
- Main combined validation accuracy peaked at 43.86% at epoch 6.
- Main combined test accuracy at the selected epoch was 43.05%.
- `column_only` test accuracy through the main `output` readout was 15.85%.
- `bypass_only` test accuracy through the main `output` readout was 43.23%.
- `column_teacher_output` test accuracy was 12.81%.
- Training energy ended around 0.028, so the full-weight teacher collapse was avoided.

Validation trajectory:

| Epoch | Validation accuracy |
| --- | --- |
| 1 | 28.92% |
| 2 | 37.02% |
| 3 | 39.84% |
| 4 | 42.84% |
| 5 | 40.86% |
| 6 | 43.86% |
| 7 | 41.40% |
| 8 | 38.02% |
| 9 | 27.92% |
| 10 | 33.68% |
| 11 | 34.58% |
| 12 | 29.48% |
| 13 | 37.46% |
| 14 | 37.36% |
| 15 | 31.10% |
| 16 | 31.72% |
| 17 | 31.06% |
| 18 | 33.62% |
| 19 | 35.62% |
| 20 | 37.26% |

Shell norms at the selected run endpoint:

| Shell | Width | Mean L2 norm | Standard deviation |
| --- | ---: | ---: | ---: |
| `hard_kernel` | 22 | 4.1763 | 1.5187 |
| `inner_shell` | 7 | 2.3080 | 0.7533 |
| `middle_shell` | 14 | 3.6695 | 1.2343 |
| `outer_shell` | 21 | 4.4774 | 0.8757 |

Interpretation:

The low-weight teacher head changed the column pathway in the intended direction but did not yet improve the main classifier. `column_only` rose from chance to 15.85% test, which means `column_pool` now carries some class information. `column_teacher_output` stayed weak at 12.81% test, which means the teacher head itself is not yet a strong classifier. `bypass_only` remained slightly above the combined output, which means the main `output` classifier still relies on the ResNet stage-4 bypass path rather than using the columns as useful additive evidence.

Compared with the no-teacher shell run, `w_teacher = 0.1` traded main accuracy for a weak column signal. Compared with the full-weight teacher run, it avoided high-energy collapse. The useful result is that the target scale matters and that a nonzero teacher can make `column_pool` non-random.

Next direction:

Run one lower teacher-weight experiment before changing architecture. The next value is `w_teacher = 0.05`. This tests whether the model can retain the new above-chance column signal while recovering closer to the no-teacher main classifier. If `w_teacher = 0.05` still hurts main accuracy and keeps `column_teacher_output` near chance, the next architectural step should move supervision closer to the HiBaCaML shell structure instead of increasing the global teacher. That means shell-local or stage-local auxiliary predictive targets attached to the hard-kernel and shell slices, with each target receiving a small energy weight.

Next CUDA-backed experiment for `rogdora43`:

```bash
./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.05
```

What to inspect after the run:

- Main combined test accuracy. Recovery toward the no-teacher 47.85% test result means the teacher is no longer disrupting the main path as strongly.
- `column_only` test accuracy. Staying above chance means the lower teacher still trains class evidence into `column_pool`.
- `column_teacher_output` test accuracy. If it remains near chance, global column-pool supervision is too coarse and the next target should be shell-local.

## 2026-06-25 Weighted Teacher 0.05 Result And Sweep Plan

Timestamp and machine: 2026-06-25 21:29:52 EDT on `rogdora43`.

Current commit while recording this result: `d993ab2`.

Completed result ingested:

- Result file: `results/codex_resnet18_column_teacher0p05_bypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260625_190650.log`
- `w_teacher = 0.05`, where `w_teacher` is the scalar multiplier on the `column_teacher_output` cross-entropy energy.
- Main combined validation accuracy peaked at 45.74% at epoch 7.
- Main combined test accuracy at the selected epoch was 46.16%.
- `column_only` test accuracy through the main `output` readout was 11.72%.
- `bypass_only` test accuracy through the main `output` readout was 46.17%.
- `column_teacher_output` test accuracy was 10.00%.
- Training energy ended around 0.028, so the full-weight teacher collapse remained avoided.

Validation trajectory:

| Epoch | Validation accuracy |
| --- | --- |
| 1 | 29.74% |
| 2 | 35.94% |
| 3 | 40.04% |
| 4 | 42.94% |
| 5 | 41.30% |
| 6 | 44.18% |
| 7 | 45.74% |
| 8 | 44.94% |
| 9 | 40.76% |
| 10 | 36.60% |
| 11 | 35.20% |
| 12 | 34.90% |
| 13 | 35.80% |
| 14 | 33.04% |
| 15 | 39.26% |
| 16 | 39.42% |
| 17 | 42.04% |
| 18 | 42.82% |
| 19 | 42.82% |
| 20 | 44.10% |

Shell norms at the selected run endpoint:

| Shell | Width | Mean L2 norm | Standard deviation |
| --- | ---: | ---: | ---: |
| `hard_kernel` | 22 | 5.5465 | 2.0897 |
| `inner_shell` | 7 | 2.6608 | 1.4975 |
| `middle_shell` | 14 | 2.3047 | 0.9329 |
| `outer_shell` | 21 | 2.9055 | 2.1983 |

Interpretation:

The `w_teacher = 0.05` run recovered much of the main classifier accuracy relative to `w_teacher = 0.1`, but the global teacher signal became too weak to train a useful teacher head. `column_teacher_output` was exactly chance on test. `column_only` was only slightly above chance at 11.72% test. The main classifier still routes through `bypass_pool`, because `bypass_only` matched the combined test accuracy.

This gives a clearer target-scale picture:

- `w_teacher = 0.0` has not yet been measured in the current code path, but the earlier no-teacher shell run reached 47.85% test and had chance-level `column_only`.
- `w_teacher = 0.05` reached 46.16% test and had 11.72% `column_only`.
- `w_teacher = 0.1` reached 43.05% test and had 15.85% `column_only`.
- `w_teacher = 1.0` collapsed the main validation metric to chance after epoch 9.

Next direction:

Run a single-machine sequential sweep before changing the architecture. The sweep should measure `w_teacher = 0.0`, `0.025`, `0.075`, and `0.125` on seed 42. `w_teacher = 0.0` is the same-code baseline with the teacher head present but zero energy. `w_teacher = 0.025` tests whether a weaker teacher preserves main accuracy while creating any column signal. `w_teacher = 0.075` tests the middle of the useful range between `0.05` and `0.1`. `w_teacher = 0.125` tests whether column utility continues to rise above `0.1` or whether main-path disruption dominates.

If this sweep does not produce a weight with both strong main accuracy and a clearly above-chance column pathway, the next architectural step should be shell-local or stage-local predictive targets instead of a stronger global `column_pool` teacher. That means attaching small-weight auxiliary predictive-coding heads to shell slices or stage-specific column outputs, so the hard-kernel, inner-shell, middle-shell, and outer-shell feature subspaces receive more localized class pressure.

Implemented sweep script:

- `scripts/run_codex_teacher_weight_sweep.sh`

Default sweep command:

```bash
./scripts/run_codex_teacher_weight_sweep.sh
```

Default sweep parameters:

- `SEED=42`
- `LR=0.005`
- `DIAGNOSE_MODE=shells`
- `NUM_EPOCHS=20`
- `WEIGHTS="0.0 0.025 0.075 0.125"`

The script writes one master log under `results/` and calls `scripts/run_codex_cifar10_depth_spanning.sh` for each weight, so each run also gets its own per-experiment result log.

## 2026-06-26 Teacher Weight Sweep Result

Timestamp and machine: 2026-06-26 07:41:30 EDT on `rogdora43`.

Current commit while recording this result: `fb64007`.

Completed sweep ingested:

- Master sweep log: `results/codex_teacher_weight_sweep_seed42_rogdora43_20260625_213438.log`
- Sweep start: 2026-06-25 21:34:38 EDT.
- Sweep finish: 2026-06-26 05:36:19 EDT.
- Seed: 42.
- Learning rate: 0.005.
- Epochs per run: 20.
- Diagnostics: shell diagnostics enabled.

Sweep summary:

| `w_teacher` | Best validation | Best epoch | Test accuracy | `column_only` test | `bypass_only` test | `column_teacher_output` test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 48.46% | 11 | 48.16% | 10.00% | 48.02% | 10.00% |
| 0.025 | 48.80% | 19 | 48.24% | 12.14% | 48.27% | 16.17% |
| 0.05 | 45.74% | 7 | 46.16% | 11.72% | 46.17% | 10.00% |
| 0.075 | 42.36% | 4 | 42.23% | 10.00% | 42.20% | 15.18% |
| 0.1 | 43.86% | 6 | 43.05% | 15.85% | 43.23% | 12.81% |
| 0.125 | 42.44% | 4 | 42.26% | 18.73% | 41.99% | 19.42% |
| 1.0 | 36.70% | 3 | 38.08% | 10.10% | 38.07% | 11.78% |

Interpretation:

`w_teacher` is the scalar multiplier on the `column_teacher_output` cross-entropy energy. The sweep shows a tradeoff rather than a useful scalar optimum. `w_teacher = 0.0` and `w_teacher = 0.025` preserve the main classifier, but the main `output` still routes through `bypass_pool`, because `bypass_only` matches the combined test accuracy. `w_teacher = 0.025` is the best scalar-teacher setting so far: it reaches 48.24% test, keeps `bypass_only` at 48.27%, and raises `column_teacher_output` to 16.17%. The main classifier still does not use the column pathway in a useful way, because `column_only` is only 12.14%.

Higher teacher weights create stronger column or teacher signals but damage the main classifier. `w_teacher = 0.125` raises `column_only` to 18.73% and `column_teacher_output` to 19.42%, but drops main test accuracy to 42.26%. `w_teacher = 1.0` collapses the training trajectory. This means a single global class target attached to `column_pool` is not enough. The global target can make the column pathway encode some class information, but that information is not compatible with the main combined readout at useful accuracy.

Shell lesion details from the stronger-teacher runs support a shell-local next step. At `w_teacher = 0.125`, `column_middle_shell_only` reached 15.29% test and `column_outer_shell_only` reached 16.38% test, while hard-kernel-only stayed at chance. This suggests that the middle-shell and outer-shell feature slices carry the most visible class signal under teacher pressure, but the signal is not organized well enough for the combined classifier.

Decision:

Stop scalar global-teacher sweeps for now. Keep `w_teacher = 0.025` as the best same-code scalar-teacher baseline when a scalar teacher is needed. The next implementation should move the auxiliary predictive-coding target closer to the HiBaCaML shell structure.

Chosen next mechanism:

Implement shell-local auxiliary predictive heads. Each shell-local head receives only one shell slice of `column_pool`, where `column_pool` is the global average pooled output of the depth-spanning columns. A shell slice is the contiguous feature range for one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`. Each shell-local head is a softmax classifier with a small weighted cross-entropy energy clamped to the CIFAR-10 label during training. The main `output` classifier and `bypass_pool` remain unchanged for evaluation.

The initial shell-local target should use small weights so the total auxiliary energy stays below the disruptive global-teacher regime. A conservative starting point is:

- `w_hard_kernel = 0.005`, where `w_hard_kernel` is the weighted cross-entropy multiplier on the hard-kernel auxiliary head.
- `w_inner_shell = 0.005`, where `w_inner_shell` is the weighted cross-entropy multiplier on the inner-shell auxiliary head.
- `w_middle_shell = 0.01`, where `w_middle_shell` is the weighted cross-entropy multiplier on the middle-shell auxiliary head.
- `w_outer_shell = 0.01`, where `w_outer_shell` is the weighted cross-entropy multiplier on the outer-shell auxiliary head.

The middle-shell and outer-shell weights are slightly larger because the sweep showed above-chance shell-only readouts there. The hard-kernel and inner-shell weights remain nonzero because the architecture should not only optimize the slices that already showed a weak signal.

Alternatives considered:

- Continue scalar global-teacher tuning. This is low effort, but the sweep already maps the useful range and shows the tradeoff.
- Use `w_teacher = 0.025` as the next main line without architecture changes. This is the best scalar result, but it still leaves `column_only` near chance and does not make the main classifier use the columns.
- Remove the bypass path during training. This would force column use, but it would also remove the diagnostic path that tells us whether columns add useful evidence beyond the backbone.
- Add a teacher schedule that starts high and decays. This may be useful later, but the current result points more directly at target placement than time schedule.

Next implementation target:

Add shell-slice nodes and shell-local classifier heads in `scripts/train_cifar10_depth_spanning.py`, with tests that verify:

- Each shell-local head receives only its shell slice from `column_pool`.
- Each shell-local target key is present in `TaskMap`.
- The training batch wrapper duplicates the CIFAR-10 label into each shell target key.
- The main `output` readout and the existing readout ablations remain unchanged.

First experiment after implementation:

Run seed 42, learning rate 0.005, 20 epochs, shell diagnostics, global `w_teacher = 0.0`, and shell-local weights `0.005, 0.005, 0.01, 0.01`. The evaluation criteria are main test accuracy, `column_only` test accuracy through the main `output`, each shell-local head accuracy, and shell lesion effects.
