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

## 2026-06-27 Shell-Local Teacher Implementation

Timestamp and machine: 2026-06-27 05:56:37 EDT on `rogdora43`.

Current commit while implementing: `bcc3922`.

Implemented mechanism:

`column_pool` is the global average pooled output of the depth-spanning columns. The implementation now optionally exposes each shell slice of `column_pool` as a separate predictive-coding node through `FeatureSliceNode`. A shell slice is the contiguous final-axis feature range assigned by `get_shell_slices(embed_dim)` to one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`.

Each enabled shell-local head receives only its own `FeatureSliceNode` output. Each head is a `Linear` softmax classifier with a weighted CIFAR-10 cross-entropy energy. The training loader duplicates the same one-hot CIFAR-10 label tensor from `y` into the enabled shell target keys, so FabricPC clamps those targets during predictive-coding training. The main `output`, the bypass path, the global `column_teacher_output`, and the existing readout ablations remain in place.

Shell-local control:

- `w_hard_kernel` is the cross-entropy multiplier on `hard_kernel_teacher_output`.
- `w_inner_shell` is the cross-entropy multiplier on `inner_shell_teacher_output`.
- `w_middle_shell` is the cross-entropy multiplier on `middle_shell_teacher_output`.
- `w_outer_shell` is the cross-entropy multiplier on `outer_shell_teacher_output`.

The command-line argument is `--shell_teacher_weights`, with values ordered as `hard_kernel,inner_shell,middle_shell,outer_shell`. The default is `0,0,0,0`, which omits the shell-local heads.

Files changed:

- `columnar_cl_fabricpc/columns/accuracy_nodes.py`: added `FeatureSliceNode`, a no-parameter predictive-coding node that predicts a configured final-axis feature slice.
- `columnar_cl_fabricpc/columns/__init__.py`: exported `FeatureSliceNode`.
- `scripts/train_cifar10_depth_spanning.py`: added shell-local teacher-head graph construction, shell target-key plumbing, shell teacher evaluation, diagnostic accounting, and `--shell_teacher_weights`.
- `scripts/run_codex_cifar10_depth_spanning.sh`: added a sixth positional argument for shell teacher weights and records the value in the log path and log header.
- `tests/test_pooled_readout_norm.py`: added tests for `FeatureSliceNode`, shell weight parsing, shell-local graph wiring, and shell target duplication.

Verification run locally:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/accuracy_nodes.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_pooled_readout_norm.py -q
```

Result: 13 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: 28 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest -q --ignore=tests/test_cifar_data.py
```

Result: 130 passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_teacher_weight_sweep.sh
```

Result: passed.

Next experiment for GPU run:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && ./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.0 0.005,0.005,0.01,0.01
```

This run tests the first shell-local target setting with the global `column_teacher_output` energy disabled by `w_teacher = 0.0`. The main observations to extract from the result log are combined test accuracy, `column_only` test accuracy, `bypass_only` test accuracy, the four shell-local teacher-head accuracies, and the shell lesion table.

## 2026-06-27 Shell-Local Teacher Result

Timestamp and machine: 2026-06-27 08:08:46 EDT on `rogdora43`.

Current commit while recording this result: `bcc3922`.

Completed result ingested:

- Result file: `results/codex_resnet18_column_teacher0p0_shell0p005_0p005_0p01_0p01_bypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260627_055814.log`
- `w_teacher = 0.0`, where `w_teacher` is the scalar multiplier on the global `column_teacher_output` cross-entropy energy.
- Shell-local teacher weights were `0.005,0.005,0.01,0.01` in `hard_kernel,inner_shell,middle_shell,outer_shell` order.
- Main combined validation accuracy peaked at 44.78% at epoch 7.
- Main combined test accuracy at the selected epoch was 44.72%.
- `column_only` test accuracy through the main `output` readout was 10.00%.
- `bypass_only` test accuracy through the main `output` readout was 44.69%.
- `column_teacher_output` test accuracy was 10.00%.
- All four shell-local teacher heads were exactly 10.00% on test.

Validation trajectory:

| Epoch | Validation accuracy |
| --- | --- |
| 1 | 30.98% |
| 2 | 37.02% |
| 3 | 40.06% |
| 4 | 42.84% |
| 5 | 42.60% |
| 6 | 42.50% |
| 7 | 44.78% |
| 8 | 37.62% |
| 9 | 30.92% |
| 10 | 37.12% |
| 11 | 35.24% |
| 12 | 38.28% |
| 13 | 35.12% |
| 14 | 28.16% |
| 15 | 39.20% |
| 16 | 38.66% |
| 17 | 42.36% |
| 18 | 41.74% |
| 19 | 43.20% |
| 20 | 44.24% |

Test readout and shell diagnostics:

| Metric | Test accuracy |
| --- | ---: |
| `combined` | 44.72% |
| `column_only` | 10.00% |
| `bypass_only` | 44.69% |
| `column_teacher_output` | 10.00% |
| `hard_kernel_teacher_output` | 10.00% |
| `inner_shell_teacher_output` | 10.00% |
| `middle_shell_teacher_output` | 10.00% |
| `outer_shell_teacher_output` | 10.00% |
| `column_without_hard_kernel` | 14.12% |
| `column_hard_kernel_only` | 10.00% |
| `column_without_inner_shell` | 9.90% |
| `column_inner_shell_only` | 10.00% |
| `column_without_middle_shell` | 10.00% |
| `column_middle_shell_only` | 10.00% |
| `column_without_outer_shell` | 10.02% |
| `column_outer_shell_only` | 10.00% |

Shell norms after training:

| Shell | Width | Mean L2 norm | Standard deviation |
| --- | ---: | ---: | ---: |
| `hard_kernel` | 22 | 5.9773 | 2.0310 |
| `inner_shell` | 7 | 2.4327 | 0.7582 |
| `middle_shell` | 14 | 2.8183 | 1.3970 |
| `outer_shell` | 21 | 1.8849 | 2.0367 |

Interpretation:

The first shell-local teacher run did not create class-informative shell heads. The main `output` still routes through `bypass_pool`, because `bypass_only` matched `combined` on test. `column_only` stayed at chance, and every shell-local teacher head stayed at chance. The small shell-local energies therefore acted as extra predictive-coding constraints without creating usable class evidence in `column_pool`.

This result means the current pooled-shell placement is not enough. The shell heads are attached after the four columns have already been combined and globally averaged into `column_pool`. That placement is closer to a shell-sliced readout of a global pooled representation than to shell-local class pressure inside each column. It tests one useful mechanism, but it is not yet a faithful local-column target.

Recommended next diagnostic:

Run one no-bypass experiment before adding another architectural mechanism. In this diagnostic, `output` receives only `column_pool`, so the main CIFAR-10 target must train through the depth-spanning columns. This distinguishes two mechanisms:

- If no-bypass training reaches useful accuracy, the columns can classify CIFAR-10 and the problem is competition from the bypass edge.
- If no-bypass training stays near chance, the current column pathway itself is not carrying class information, and the next implementation should move teacher heads upstream to per-column shell targets before the combiner.

Runner update:

`scripts/run_codex_cifar10_depth_spanning.sh` now accepts a seventh positional argument, `readout_mode`. `readout_mode = bypass` keeps the existing `bypass_pool -> output` edge. `readout_mode = nobypass` omits that edge so the main classifier reads only `column_pool`. Existing commands keep `bypass` as the default.

Next command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && ./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.0 0,0,0,0 nobypass
```

This next run disables both global and shell-local auxiliary teacher energies. The only class target is the main `output` node, and the only readout source is `column_pool`. The result should be interpreted as a direct test of whether the current depth-spanning HiBaCaML-style column pathway can classify plain CIFAR-10 when it cannot use the ResNet stage-4 bypass path.

## 2026-06-27 No-Bypass Column-Only Result

Timestamp and machine: 2026-06-27 12:51:57 EDT on `rogdora43`.

Current commit while recording this result: `2b00a3f`.

Completed result ingested:

- Result file: `results/codex_resnet18_column_teacher0p0_shell0_0_0_0_nobypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260627_081028.log`
- `readout_mode = nobypass`, where `readout_mode` selects whether the main `output` classifier receives the direct `bypass_pool` input. In this run, `output` received only `column_pool`.
- `w_teacher = 0.0`, where `w_teacher` is the scalar multiplier on the global `column_teacher_output` cross-entropy energy.
- Shell-local teacher weights were all zero.
- Main validation accuracy peaked at 20.16% at epoch 18.
- Main test accuracy at the selected epoch was 20.53%.
- `column_only` test accuracy was also 20.53%, because no bypass edge existed.
- `column_teacher_output` test accuracy was 8.98%.

Validation trajectory:

| Epoch | Validation accuracy |
| --- | ---: |
| 1 | 10.72% |
| 2 | 10.22% |
| 3 | 10.30% |
| 4 | 11.90% |
| 5 | 13.04% |
| 6 | 15.60% |
| 7 | 9.40% |
| 8 | 16.48% |
| 9 | 11.98% |
| 10 | 14.60% |
| 11 | 16.44% |
| 12 | 10.50% |
| 13 | 18.78% |
| 14 | 16.12% |
| 15 | 19.82% |
| 16 | 18.44% |
| 17 | 18.70% |
| 18 | 20.16% |
| 19 | 19.46% |
| 20 | 19.60% |

Stability-first interpretation:

This was not a hard collapse. The validation accuracy began near chance, then climbed above 20% by epoch 18. The final training progress line reported energy near 0.0427, and the shell norms remained nonzero after training. The run is low-accuracy, but it shows that the current depth-spanning column pathway can learn a weak CIFAR-10 classifier when the bypass path is removed.

The more important failure mode is shell imbalance. `hard_kernel` grew from mean L2 norm 5.5781 before training to 6.5873 after training. `outer_shell` shrank from 3.1733 to 0.9883. The shell lesion results match this magnitude pattern:

| Metric | Test accuracy |
| --- | ---: |
| `combined` | 20.53% |
| `combined_without_hard_kernel` | 14.62% |
| `column_hard_kernel_only` | 17.44% |
| `combined_without_inner_shell` | 20.56% |
| `column_inner_shell_only` | 14.38% |
| `combined_without_middle_shell` | 18.00% |
| `column_middle_shell_only` | 10.00% |
| `combined_without_outer_shell` | 20.19% |
| `column_outer_shell_only` | 10.00% |

`hard_kernel` is the only shell that carries a strong independent signal. `inner_shell` carries a weaker independent signal. `middle_shell` contributes in combination but is not sufficient by itself. `outer_shell` is effectively unused by the classifier. This is the next collapse-like issue to address: not a whole-network collapse to chance, but a shell participation collapse where one shell dominates and another shell loses magnitude.

Mechanism-level diagnosis:

`DepthSpanningColumnNode` combines shell-specific K/L/B pathway outputs, then applies LayerNorm across the full `output_dim` feature axis when `apply_layer_norm` is enabled. `output_dim` is the full column feature width, and the shell slices are contiguous subranges inside that width. Full-width normalization pins the whole column vector but does not pin each shell slice. The hard-kernel slice can therefore carry most of the class-useful norm while the outer-shell slice shrinks. This matches the no-bypass result.

Recommended next code change:

Treat the shell layout as a stability boundary inside the shared column node. This means changing column output normalization from full-width LayerNorm to shell-wise LayerNorm inside `DepthSpanningColumnNode`. Shell-wise LayerNorm normalizes each shell slice independently after its K/L/B mixture and before concatenating the four shell slices back into the column output. This keeps the predictive-coding node structure intact while preventing the full column feature axis from hiding shell-level magnitude collapse.

This is a shared infrastructure change rather than a local workaround. The scope shift is: the shell layout is now part of the supported stability contract of `DepthSpanningColumnNode`, so normalization must respect shell boundaries. Existing callers that request column layer normalization through `apply_layer_norm` should use the shell-wise behavior. The stage taps and global pools can keep their existing feature-axis normalization because they are not shell-typed nodes.

Alternatives considered:

- Run more seeds of the current no-bypass graph. This would measure variance, but it would not address the observed shell participation collapse.
- Add pooled shell-local teacher heads to the no-bypass graph. This is a no-code experiment, but the shell heads attach after column combining and global pooling. It is less faithful than fixing shell stability inside the column node.
- Add per-column shell teacher heads immediately. This is architecturally relevant, but adding more class-error paths before stabilizing shell magnitudes risks another auxiliary-target collapse.
- Restore the bypass path and tune teacher weights again. The bypass path hides column weakness, and the scalar-teacher sweep already showed the tradeoff.

Next implementation target:

Implement shell-wise LayerNorm in `DepthSpanningColumnNode` when `apply_layer_norm` is true. Keep the parameter interface compatible with the existing `ln_gamma` and `ln_beta` vectors when those parameters are learnable. Add tests that verify each shell slice is normalized independently. Then rerun the same no-bypass experiment:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && ./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.0 0,0,0,0 nobypass
```

Primary success criteria for that run:

- Validation should not collapse to chance after initially rising.
- Training energy should remain bounded.
- All shell mean L2 norms should remain meaningfully nonzero after training, especially `outer_shell`.
- Shell lesion results should show broader shell participation than the current hard-kernel-dominated pattern.
- Accuracy should be considered only after those stability criteria are met.

## 2026-06-27 Shell-Wise LayerNorm Implementation

Timestamp and machine: 2026-06-27 13:02:35 EDT on `rogdora43`.

Current commit while implementing: `359245d`.

Implemented stability mechanism:

`DepthSpanningColumnNode` now treats the shell layout as a normalization boundary. When `apply_layer_norm` is true, the node normalizes each shell slice independently after the shell-specific K/L/B mixture and before concatenating the four shell outputs. `K` is the kernel pathway, `L` is the lateral pathway, and `B` is the bridge pathway. Each shell is the contiguous feature range returned by `get_shell_slices(output_dim)`.

The parameter interface remains compatible with the previous column LayerNorm. When `fix_ln_gamma` is false, `ln_gamma` and `ln_beta` remain full-width vectors of shape `(output_dim,)`. The forward pass slices those vectors to match each shell. When `fix_ln_gamma` is true, the non-learnable scalar gamma and beta are applied per shell.

Files changed:

- `columnar_cl_fabricpc/columns/depth_spanning_column.py`: added `_shellwise_layernorm` and replaced full-width column LayerNorm with shell-wise LayerNorm.
- `tests/test_depth_spanning_column.py`: added tests for learnable parameter shape compatibility, direct shell-wise normalization, and graph-level column output normalization.

Verification run locally:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile columnar_cl_fabricpc/columns/depth_spanning_column.py tests/test_depth_spanning_column.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_depth_spanning_column.py -q
```

Result: 18 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: 31 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest -q --ignore=tests/test_cifar_data.py
```

Result: 133 passed.

```bash
git diff --check
```

Result: passed.

## 2026-06-28 Overnight Shell Readout Sweep Result

Timestamp and machine: 2026-06-28 12:21:40 EDT on `rogdora43`.

Current commit while analyzing: `4222596`.

Master log:

`results/codex_shell_readout_overnight_sweep_rogdora43_20260627_220509.log`

The sweep completed at 2026-06-28 08:56:33 EDT.

Result table:

| Case | `column_shell_teacher_weights` | Test accuracy | Best validation accuracy | Best validation epoch | `column_only` test | `column_shell_readout_only` test | `column_shell_readout_outer_shell_only` test | `column_shell_readout_without_outer_shell` test |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed 42, no per-column shell teachers | `0,0,0,0` | 27.52% | 27.76% | 18 | 10.00% | 25.87% | 10.00% | 22.75% |
| Seed 42, heavier outer-shell teacher | `0.001,0.001,0.002,0.006` | 24.73% | 25.90% | 7 | 13.04% | 24.00% | 10.00% | 23.04% |
| Seed 99, current direct shell readout | `0.001,0.001,0.002,0.002` | 26.58% | 27.50% | 16 | 10.17% | 26.10% | 10.00% | 24.16% |
| Seed 7, current direct shell readout | `0.001,0.001,0.002,0.002` | 27.32% | 28.06% | 20 | 11.46% | 27.32% | 10.00% | 25.30% |

Shell norm stability:

- Seed 42 with no per-column shell teachers ended with `hard_kernel = 4.6902`, `inner_shell = 2.6445`, `middle_shell = 3.7396`, and `outer_shell = 4.5794`.
- Seed 42 with heavier outer-shell teacher ended with `hard_kernel = 4.6900`, `inner_shell = 2.6451`, `middle_shell = 3.7400`, and `outer_shell = 4.5792`.
- Seed 99 current direct shell readout ended with `hard_kernel = 4.6902`, `inner_shell = 2.6449`, `middle_shell = 3.7394`, and `outer_shell = 4.5806`.
- Seed 7 current direct shell readout ended with `hard_kernel = 4.6903`, `inner_shell = 2.6453`, `middle_shell = 3.7407`, and `outer_shell = 4.5816`.

Interpretation:

The strongest result is that disabling per-column shell teacher heads improved seed 42 from the previous 24.11% test accuracy to 27.52% test accuracy. The direct `(column, shell)` readout is robust across seeds, and the old `column_pool` path remains near chance as an isolated readout.

The heavier outer-shell teacher did not solve the `outer_shell` collapse. In the heavier-teacher run, all four outer-shell teacher heads were 10.00% on the test set, and `column_shell_readout_outer_shell_only` was also 10.00%. This means direct label pressure on `outer_shell` is not the right mechanism.

The `outer_shell` feature norm is stable and nonzero in every run, and removing `outer_shell` from the direct shell readout consistently reduces accuracy. For example, seed 7 drops from 27.32% to 25.30% when `outer_shell` is removed. The failure is not absence of activation. The failure is that `outer_shell` is not linearly class-readable by itself.

Next mechanism:

Add a shell-to-shell bridge that routes all four pooled shells from one column through a shared Gaussian predictive-coding latent before the main classifier. This mechanism keeps the concentric shell structure, avoids local per-shell classifier heads, and gives `outer_shell` a way to contribute through a column-local inter-shell prediction path rather than requiring it to classify alone.

Alternatives considered:

- Increase the outer-shell teacher weight again. The 0.006 run did not move outer-shell teacher heads above chance, so another scalar increase is unlikely to address the mechanism.
- Keep direct shell readout and run longer. This may improve accuracy, but it does not add the missing inter-shell predictive path.
- Remove `outer_shell` from the architecture. This conflicts with the goal of faithful columnar-shell architecture because `outer_shell` has stable magnitude and contributes in combination.

## 2026-06-28 Shell Bridge Implementation

Timestamp and machine: 2026-06-28 12:21:40 EDT on `rogdora43`.

Implemented mechanism:

The graph now has an optional per-column shell bridge. A shell bridge is a Gaussian `Linear` latent named `columnXX_shell_bridge`. It receives all pooled shell vectors from one active column: `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`. The bridge then connects to the main `output` classifier.

The bridge is enabled with `--column_shell_bridge`. It is disabled by default. When bridge mode is enabled, shell slice and pool nodes are created even if per-column shell teacher heads and direct shell readout are both disabled.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`: added `--column_shell_bridge`, per-column shell bridge nodes, bridge readout ablations, bridge shell-input ablations, and a generic input-edge masking helper.
- `scripts/run_codex_cifar10_depth_spanning.sh`: added a tenth positional argument for shell bridge mode. Accepted values are `on`, `true`, or `shellbridge` to enable it, and `off`, `false`, or `noshellbridge` to disable it.
- `scripts/run_codex_shell_bridge_sweep.sh`: added a four-run sequential sweep for the next mechanism test.
- `tests/test_pooled_readout_norm.py`: added graph and masking tests for the shell bridge.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_pooled_readout_norm.py -q
```

Result: 18 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: 36 passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result: passed.

```bash
bash -n scripts/run_codex_shell_bridge_sweep.sh
```

Result: passed.

```bash
git diff --check
```

Result: passed.

Next sweep:

The next sweep runs four 20-epoch experiments. The first case tests bridge-only readout on seed 42. The second case tests bridge plus direct `(column, shell)` readout on seed 42. The third and fourth cases test bridge-only readout on seeds 99 and 7.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_sweep.sh
```

Next experiment:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && ./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.0 0,0,0,0 nobypass
```

Primary readout:

- Treat validation stability as the first criterion.
- Compare post-training shell mean L2 norms against the previous no-bypass run, especially `outer_shell = 0.9883`.
- Compare shell lesion results against the previous hard-kernel-dominated pattern.
- Treat test accuracy as secondary until shell participation remains stable.

## 2026-06-27 Shell-Wise LayerNorm No-Bypass Result

Timestamp and machine: 2026-06-27 16:14:30 EDT on `rogdora43`.

Current commit while analyzing: `11cae64`.

Experiment log:

`results/codex_resnet18_column_teacher0p0_shell0_0_0_0_nobypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260627_131114.log`

Experiment configuration:

- `readout_mode = nobypass`, so the classifier used only the columnar predictive-coding path.
- `column_teacher_weight = 0.0`, so the pooled column teacher head was diagnostic only.
- `shell_teacher_weights = 0,0,0,0`, so shell-local teacher heads were diagnostic only.
- Shell-wise LayerNorm was active inside each depth-spanning column.

Result:

- Best validation accuracy: 21.20% at epoch 7.
- Test accuracy: 21.85%.
- Previous no-bypass baseline before shell-wise LayerNorm: 20.16% best validation accuracy and 20.53% test accuracy.
- The small accuracy gain is useful, but the more important result is the stability change.

Stability readout:

- Before shell-wise LayerNorm, the no-bypass run ended with shell mean L2 norms of `hard_kernel = 6.5873`, `inner_shell = 2.9472`, `middle_shell = 2.8263`, and `outer_shell = 0.9883`.
- With shell-wise LayerNorm, the no-bypass run ended with shell mean L2 norms of `hard_kernel = 4.6898`, `inner_shell = 2.6448`, `middle_shell = 3.7410`, and `outer_shell = 4.5823`.
- `outer_shell` is the widest nonlocal shell. Its norm no longer collapses relative to the other shells.
- The training energy remained bounded and decayed from roughly `2.07` early in training to roughly `0.0434` at the end.

Shell lesion readout:

- Test accuracy with all shells: 21.85%.
- Test accuracy without `inner_shell`: 23.06%, which is higher than the full column readout.
- Test accuracy with `inner_shell` only: 10.00%, which is chance for CIFAR-10.
- Test accuracy without `outer_shell`: 21.99%, which is slightly higher than the full column readout.
- Test accuracy with `outer_shell` only: 10.00%, which is chance for CIFAR-10.
- Test accuracy with `middle_shell` only: 17.16%.
- Test accuracy with `hard_kernel` only: 11.81%.

Interpretation:

The collapse target has moved. Shell-wise LayerNorm appears to solve the magnitude collapse of the nonlocal shell activations. The remaining failure is semantic participation: `inner_shell` and `outer_shell` now have stable magnitude, but their slices are not independently predictive of CIFAR-10 class labels and can add noise to the final readout.

This means shell-wise LayerNorm should stay in the baseline. Reverting it would restore the older failure where `outer_shell` lost magnitude. The next mechanism should add class pressure at the column shell boundary, not at the pooled readout alone.

Recommended next implementation:

Add optional per-column shell-local teacher heads before the column combiner. For each active column, slice the column output into `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`, average each shell over spatial tokens, and attach a tiny CIFAR-10 teacher head to each shell slice. The teacher head should contribute a weighted predictive-coding energy term during training and should default to zero weight so the current baseline remains unchanged unless an experiment explicitly enables it.

Mechanism:

Let `h_{j,s}` mean the token tensor emitted by column `j` for shell `s`, where `j` indexes a depth-spanning column and `s` is one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`. The per-column shell-local teacher should pool `h_{j,s}` over the token axis, project the pooled shell vector to 10 CIFAR-10 logits, and add a small supervised energy term for that shell. This gives every shell inside every column a local class target before the column combiner can wash out or ignore that shell.

Initial experiment after implementation:

Use no bypass and very small teacher weights because there are four columns. The first candidate is `hard_kernel = 0.001`, `inner_shell = 0.001`, `middle_shell = 0.002`, and `outer_shell = 0.002`. The primary criterion is not peak accuracy. The primary criterion is whether `inner_shell` and `outer_shell` improve above chance in shell-only lesion readouts without forcing validation accuracy back to chance.

Alternatives considered:

- Increase pooled shell teacher weights using the current code. This is cheaper, but it applies class pressure after column outputs are already pooled together. It is less faithful to the columnar architecture because it does not force each column shell to carry its own class-relevant signal.
- Restore the bypass and optimize accuracy from the combined path. This would likely raise headline accuracy, but it would hide the column-only failure that matters for the HibacaML-style architecture.
- Add more shells or skip connections now. That is architecturally important, but the current shells are not yet class-participating. Adding more routing depth before local shell supervision would make the failure harder to diagnose.

Next action:

Implement per-column shell-local teacher heads in the experiment repo only, keep all new weights disabled by default, add focused tests for parameter shape and loss contribution, then ask for a no-bypass GPU run with small shell-local teacher weights.

## 2026-06-27 Per-Column Shell Teacher Implementation

Timestamp and machine: 2026-06-27 16:22:15 EDT on `rogdora43`.

Current commit while implementing: `c85ea10`.

Implemented mechanism:

The experiment graph now supports per-column shell-local teacher heads. A per-column shell-local teacher head is an auxiliary CIFAR-10 classifier attached to one shell slice from one depth-spanning column before the column combiner. The new path for column `j` and shell `s` is `col_j -> column{j}_{s}_slice -> column{j}_{s}_pool -> column{j}_{s}_teacher_output`. The slice node exposes only shell `s` from column `j`. The pool node averages over spatial tokens. The teacher output node receives the CIFAR-10 label through a weighted predictive-coding cross-entropy energy.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`: added `--column_shell_teacher_weights`, node-name helpers, graph wiring for active-column shell teacher heads, auxiliary target wiring, diagnostics, and validation/test reporting for the new heads.
- `scripts/run_codex_cifar10_depth_spanning.sh`: added an eighth positional argument for `column_shell_teacher_weights` and records those weights in the result log filename and header.
- `tests/test_pooled_readout_norm.py`: added graph tests proving that per-column shell heads attach before the combiner and that inactive columns do not receive shell-local supervision.

Default behavior:

`--column_shell_teacher_weights` defaults to `0,0,0,0`. With that default, no per-column shell teacher nodes are added. This preserves the previous baseline unless an experiment explicitly enables the new mechanism.

Verification run locally:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_pooled_readout_norm.py -q
```

Result: 15 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: 33 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest -q --ignore=tests/test_cifar_data.py
```

Result: 135 passed.

No CIFAR-10 training run was started locally. The next GPU experiment should be run on `rogdora43`.

Next experiment:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && ./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.0 0,0,0,0 nobypass 0.001,0.001,0.002,0.002
```

Readout criteria:

- Primary: shell magnitudes should remain nonzero, especially `outer_shell`.
- Primary: per-column shell teacher heads should rise above chance for `inner_shell` and `outer_shell` without causing validation accuracy to collapse.
- Secondary: shell lesion readouts should stop improving when `inner_shell` or `outer_shell` is removed.
- Secondary: final test accuracy should improve after the collapse and participation criteria are satisfied.

## 2026-06-27 Per-Column Shell Teacher Result

Timestamp and machine: 2026-06-27 18:58:27 EDT on `rogdora43`.

Current commit while analyzing: `59260e1`.

Experiment log:

`results/codex_resnet18_column_teacher0p0_shell0_0_0_0_colshell0p001_0p001_0p002_0p002_nobypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260627_162358.log`

Experiment configuration:

- `readout_mode = nobypass`, so the classifier used only the columnar predictive-coding path.
- `column_teacher_weight = 0.0`, so the pooled column teacher head was diagnostic only.
- `shell_teacher_weights = 0,0,0,0`, so pooled shell teacher heads were disabled.
- `column_shell_teacher_weights = 0.001,0.001,0.002,0.002`, so each active column received shell-local class pressure before the combiner.

Result:

- Best validation accuracy: 20.50% at epoch 17.
- Test accuracy: 20.94%.
- Previous no-bypass shell-wise LayerNorm baseline without per-column shell teachers: 21.20% best validation accuracy and 21.85% test accuracy.
- The main readout became slightly worse, but the shell participation diagnostics improved.

Stability readout:

- Final shell mean L2 norms were `hard_kernel = 4.6903`, `inner_shell = 2.6452`, `middle_shell = 3.7408`, and `outer_shell = 4.5818`.
- These match the previous stable shell-wise LayerNorm run closely. The per-column shell teachers did not reintroduce magnitude collapse.
- End-of-training energy was roughly `0.0681`, compared with roughly `0.0434` in the no-teacher shell-wise LayerNorm baseline. The extra supervised shell energy raised the total energy but did not destabilize training.

Per-column shell teacher readout:

- Mean test accuracy across the four `hard_kernel` teacher heads was roughly 22.24%.
- Mean test accuracy across the four `inner_shell` teacher heads was roughly 21.53%.
- Mean test accuracy across the four `middle_shell` teacher heads was roughly 21.64%.
- Mean test accuracy across the four `outer_shell` teacher heads was roughly 19.46%.
- This is the first run where `inner_shell` and `outer_shell` show clear above-chance local class information before the combiner.

Final readout shell lesions:

- Test accuracy with all shells: 20.94%.
- `column_inner_shell_only` improved from 10.00% in the previous shell-wise LayerNorm baseline to 15.16%.
- `column_hard_kernel_only` improved from 11.81% to 18.41%.
- `column_middle_shell_only` remained similar, moving from 17.16% to 17.61%.
- `column_outer_shell_only` stayed at 10.00%, even though the per-column `outer_shell` teacher heads were around 19.46% mean test accuracy.
- Removing `inner_shell` still slightly improved the main readout to 21.17%.
- Removing `outer_shell` was nearly neutral at 20.97%.

Interpretation:

The per-column shell teachers achieved the immediate stability-first goal. They put class information into `inner_shell`, `middle_shell`, and `outer_shell` without collapsing shell magnitudes. The remaining failure is not that `outer_shell` cannot learn class information. The failure is that the current sum combiner and pooled readout lose that information.

Mechanism:

Let `h_{j,s}` mean the token tensor emitted by column `j` for shell `s`, where `s` is one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`. The teacher head attached to `h_{j,s}` can read class information from that shell. The main classifier does not read `h_{j,s}` directly. It reads a pooled feature after the combiner sums columns featurewise. If different columns encode class evidence in different feature directions, then featurewise summation can cancel or dilute the shell evidence before the main classifier sees it.

Recommended next implementation:

Add an optional column-preserving shell readout path. For each active column and shell, reuse the already-created shell pool `pool_{j,s}` or create it when shell teachers are disabled. Connect those pooled shell vectors directly to the main `output` node through `Linear` input edges, while keeping the existing `column_pool` edge for ablation. This keeps the predictive-coding classifier energy on the main CIFAR-10 output, but it lets the readout see column identity and shell identity instead of forcing all evidence through a featurewise sum first.

The initial experiment should keep the per-column shell teacher weights at `0.001,0.001,0.002,0.002`, keep `readout_mode = nobypass`, and enable the column-preserving shell readout. The key diagnostic should compare the existing `column_pool` edge against the new shell-readout edges. The success criterion is that `outer_shell` and `inner_shell` no longer disappear when information reaches the final classifier.

Alternatives considered:

- Increase `column_shell_teacher_weights`. This might strengthen the local teacher heads, but it does not address the observed mismatch where `outer_shell` is locally predictive before the combiner and chance after the combiner.
- Use the bypass path. This would likely improve headline accuracy, but it would hide whether the columnar route can classify CIFAR-10.
- Replace the combiner immediately. This may be necessary later, but a column-preserving shell readout is a narrower test of whether the loss happens at the sum combiner.

## 2026-06-27 Column-Preserving Shell Readout Implementation

Timestamp and machine: 2026-06-27 19:04:22 EDT on `rogdora43`.

Current commit while implementing: `59260e1`.

Implemented mechanism:

The experiment graph now supports an optional column-preserving shell readout path. A column-preserving shell readout path connects the token-pooled shell vector from each active `(column, shell)` pair directly to the main `output` classifier. The existing `column_pool` edge remains in the graph, so readout ablations can separate the old featurewise-sum path from the new direct shell path.

The flag is `--column_shell_readout`. It defaults to disabled. When disabled, the graph behaves as before. When enabled, each active column and shell creates or reuses `column{j}_{s}_slice` and `column{j}_{s}_pool`, then connects `column{j}_{s}_pool` to `output`.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`: added `--column_shell_readout`, direct `(column, shell)` pool edges to `output`, readout-source ablations, and per-shell direct-readout lesion tables.
- `scripts/run_codex_cifar10_depth_spanning.sh`: added a ninth positional argument for column shell readout mode. Accepted values are `on`, `true`, or `shellreadout` to enable it, and `off`, `false`, or `noshellreadout` to disable it.
- `tests/test_pooled_readout_norm.py`: added a graph test that proves per-column shell readout edges reach `output` without creating teacher heads when teacher weights are zero.

Verification run locally:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_pooled_readout_norm.py -q
```

Result: 16 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: 34 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/pytest -q --ignore=tests/test_cifar_data.py
```

Result: 136 passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result: passed.

No CIFAR-10 training run was started locally.

Next experiment:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && ./scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 shells 20 0.0 0,0,0,0 nobypass 0.001,0.001,0.002,0.002 on
```

Readout criteria:

- Primary: shell norms should remain stable and nonzero.
- Primary: `column_shell_readout_only` should outperform the previous chance-level `column_outer_shell_only` result.
- Primary: `column_shell_readout_outer_shell_only` should rise above chance if the outer-shell information was being lost only at the sum combiner.
- Secondary: combined test accuracy should improve over the previous 20.94% direct-teacher run.
- Secondary: removing `inner_shell` or `outer_shell` from the direct shell readout should no longer improve accuracy.

## 2026-06-27 Column-Preserving Shell Readout Result

Timestamp and machine: 2026-06-27 21:57:39 EDT on `rogdora43`.

Current commit while analyzing: `59260e1`.

Experiment log:

`results/codex_resnet18_column_teacher0p0_shell0_0_0_0_colshell0p001_0p001_0p002_0p002_colshellreadouton_nobypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260627_190528.log`

Experiment configuration:

- `readout_mode = nobypass`, so the classifier used only the columnar predictive-coding route.
- `column_shell_readout = on`, so each active `(column, shell)` pool connected directly to the main output classifier.
- `column_shell_teacher_weights = 0.001,0.001,0.002,0.002`, so per-column shell teacher heads were still present.
- Pooled shell teacher heads remained disabled with `shell_teacher_weights = 0,0,0,0`.

Result:

- Best validation accuracy: 24.92% at epoch 20.
- Test accuracy: 24.11%.
- Previous per-column shell teacher run without direct shell readout: 20.50% best validation accuracy and 20.94% test accuracy.
- The validation trajectory was still rising at epoch 20, which suggests longer runs may be informative.

Stability readout:

- Final shell mean L2 norms were `hard_kernel = 4.6902`, `inner_shell = 2.6453`, `middle_shell = 3.7407`, and `outer_shell = 4.5810`.
- Shell magnitudes remained stable and nonzero.
- End-of-training energy was roughly `0.0339`, lower than the prior direct-teacher run's roughly `0.0681`.

Readout path interpretation:

- `combined` test accuracy was 24.11%.
- `column_shell_readout_only` test accuracy was 23.35%.
- `column_pool_plus_shell_readout` test accuracy was 24.11%.
- `column_only` through the old `column_pool` edge was 10.00%.

The accuracy gain comes from the direct `(column, shell)` readout path. The old summed `column_pool` path collapsed to chance as an independent readout.

Per-shell direct readout:

- `column_shell_readout_hard_kernel_only` test accuracy was 18.52%.
- `column_shell_readout_inner_shell_only` test accuracy was 20.65%.
- `column_shell_readout_middle_shell_only` test accuracy was 19.37%.
- `column_shell_readout_outer_shell_only` test accuracy was 10.00%.
- `column_shell_readout_without_outer_shell` test accuracy was 21.78%, lower than the full direct shell readout at 23.35%.

`outer_shell` is not class-predictive by itself, but it appears to help in combination with other shells.

Per-column shell teacher heads:

- Mean test accuracy across the four `hard_kernel` teacher heads was roughly 22.71%.
- Mean test accuracy across the four `inner_shell` teacher heads was roughly 19.05%.
- Mean test accuracy across the four `middle_shell` teacher heads was roughly 20.35%.
- All four `outer_shell` teacher heads were 10.00%.

This is a regression from the previous no-readout teacher run, where `outer_shell` teacher heads averaged roughly 19.46%. Direct readout improved the main classifier but allowed the outer-shell local teacher heads to collapse back to chance.

Interpretation:

The direct shell readout confirmed that the column-preserving route is useful. It improved CIFAR-10 accuracy and recovered useful signal from hard, inner, and middle shells. The outer shell remains the central stability problem. Its magnitude is stable, and it helps the multi-shell direct readout in combination, but it is not independently class-predictive after this training configuration.

Next experimental objective:

Run an overnight sweep that consumes roughly 6 to 8 hours and answers three questions:

- Robustness: does direct shell readout work across seeds?
- Mechanism: does direct shell readout still work if per-column shell teacher heads are disabled?
- Outer-shell pressure: does increasing the outer-shell teacher weight restore outer-shell class participation without collapsing the rest of the model?

Planned overnight runs:

- Seed 99, direct shell readout on, teacher weights `0.001,0.001,0.002,0.002`.
- Seed 7, direct shell readout on, teacher weights `0.001,0.001,0.002,0.002`.
- Seed 42, direct shell readout on, teacher weights `0,0,0,0`.
- Seed 42, direct shell readout on, teacher weights `0.001,0.001,0.002,0.006`.

Each 20-epoch run has recently taken about 100 to 102 minutes. Four runs should take about 6.7 hours plus overhead.

## 2026-06-27 Overnight Shell Readout Sweep Script

Timestamp and machine: 2026-06-27 22:00:15 EDT on `rogdora43`.

Added script:

`scripts/run_codex_shell_readout_overnight_sweep.sh`

The script runs four 20-epoch CIFAR-10 experiments sequentially and writes a master log to `results/codex_shell_readout_overnight_sweep_<host>_<timestamp>.log`. Each individual run still writes its normal per-run log through `scripts/run_codex_cifar10_depth_spanning.sh`.

Run order:

1. `seed42_no_column_shell_teachers`: direct `(column, shell)` readout enabled with `column_shell_teacher_weights = 0,0,0,0`. This tests whether the main readout path can learn without local shell teacher heads.
2. `seed42_outer_shell_teacher_heavier`: direct `(column, shell)` readout enabled with `column_shell_teacher_weights = 0.001,0.001,0.002,0.006`. This tests whether stronger outer-shell pressure restores class participation in `outer_shell`.
3. `seed99_current_shell_readout_replicate`: current best direct shell readout setting on a second seed.
4. `seed7_current_shell_readout_replicate`: current best direct shell readout setting on a third seed.

The diagnostic priority is collapse first and accuracy second. For these runs, collapse means a shell-specific readout or teacher head stays at roughly chance accuracy while its feature norm remains nonzero. The most important checks are whether `outer_shell` rises above chance under heavier teacher pressure and whether the direct shell readout remains stable across seeds.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_readout_overnight_sweep.sh
```

Verification:

```bash
bash -n scripts/run_codex_shell_readout_overnight_sweep.sh
```

Result: passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result: passed.

```bash
git diff --check
```

Result: passed.

## 2026-06-28 Current Next Action

Timestamp and machine: 2026-06-28 12:21:40 EDT on `rogdora43`.

The completed overnight sweep showed that direct `(column, shell)` readout is robust, that disabling per-column shell teacher heads was better than using them on seed 42, and that heavier outer-shell teacher pressure did not make `outer_shell` independently class-readable.

The implemented next mechanism is `--column_shell_bridge`. A shell bridge is a Gaussian predictive-coding latent that receives all four pooled shells from one column and connects to the main CIFAR-10 output classifier. This tests whether `outer_shell` can participate through inter-shell predictive coupling rather than through a local classifier head.

The next command to run is:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_sweep.sh
```

The expected runtime is roughly 6 to 8 hours. The master log will be written under `results/codex_shell_bridge_sweep_<host>_<timestamp>.log`.
