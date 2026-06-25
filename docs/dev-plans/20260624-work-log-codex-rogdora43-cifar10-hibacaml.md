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
