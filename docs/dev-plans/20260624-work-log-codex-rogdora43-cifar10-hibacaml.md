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

## 2026-07-07 Shell Learning-Rate Profile Results and Context-Teacher Follow-Up

Timestamp and machine: 2026-07-07 20:15:24 EDT on `rogdora43`.

Completed shell learning-rate profile sweep:

- Master log: `results/codex_shell_lr_profile_outer_context_10col3shared_rogdora43_20260706_180917.log`.
- The run finished at 2026-07-07 16:13:01 EDT.
- The tested architecture used 10 columns, 3 shared columns, 7 active non-shared columns, `combiner=shell_attention`, `outer_shell_context=on`, no backbone bypass, no shell teacher heads, no per-column shell teacher heads, and learning rate 0.005 for 20 epochs.
- The baseline for comparison is the prior `shell_lr_multipliers=1,1.5,2,3` run. The shell learning-rate multiplier is the scalar applied to AdamW's parameter update for one shell-local parameter group, in `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` order.

Profile results:

| Profile | `shell_lr_multipliers` | Seed | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Gentler outward schedule | `1,1.25,1.75,2.5` | 42 | 28.62% | 17 | 29.35% |
| Gentler outward schedule | `1,1.25,1.75,2.5` | 99 | 28.56% | 18 | 28.72% |
| Gentler outward schedule | `1,1.25,1.75,2.5` | 7 | 23.58% | 10 | 23.46% |
| Stronger outward schedule | `1,2,3,4` | 42 | 27.30% | 20 | 27.32% |
| Stronger outward schedule | `1,2,3,4` | 99 | 27.04% | 19 | 27.25% |
| Stronger outward schedule | `1,2,3,4` | 7 | 24.70% | 19 | 23.77% |

Aggregate comparison:

| Setting | Mean test accuracy | Across-seed sample standard deviation |
| --- | ---: | ---: |
| Flat control, `1,1,1,1` | 28.06% | 2.03 percentage points |
| Current baseline, `1,1.5,2,3` | 31.07% | 1.31 percentage points |
| Gentler profile, `1,1.25,1.75,2.5` | 27.18% | 3.23 percentage points |
| Stronger profile, `1,2,3,4` | 26.11% | 2.03 percentage points |

Per-seed comparison against the current baseline:

| Seed | Current baseline test accuracy | Gentler profile test accuracy | Stronger profile test accuracy |
| ---: | ---: | ---: | ---: |
| 42 | 32.41% | 29.35% | 27.32% |
| 99 | 29.79% | 28.72% | 27.25% |
| 7 | 31.00% | 23.46% | 23.77% |

Interpretation:

- `shell_lr_multipliers=1,1.5,2,3` remains the best tested shell plasticity profile. Both neighboring profiles reduced mean accuracy and increased collapse risk.
- The stronger profile is not supported. Its `outer_shell_context_only` test accuracy was exactly 10.00% in all three seeds, while the combined classifier fell below the current baseline in every seed.
- The gentler profile is also not supported. It improved seed 99 relative to the flat control but badly hurt seed 7. Its mean accuracy was below the flat control.
- Composer attention was usually sharp in the new runs, so attention sharpness alone is not the missing mechanism.
- The repeated failure mode is that the outer-shell context route is useful as part of the combined classifier, but it often remains weak or chance-level when evaluated by itself. This suggests that the next change should strengthen the context route directly rather than continue scalar shell learning-rate sweeps.

Implemented next mechanism:

- Added `OUTER_SHELL_CONTEXT_TEACHER_NODE = "outer_shell_context_teacher_output"`.
- Added `OUTER_SHELL_CONTEXT_TEACHER_TARGET = "outer_shell_context_y"`.
- Added `--outer_shell_context_teacher_weight`, defaulting to `0.0`.
- When `--outer_shell_context_teacher_weight` is positive, the graph adds one context-only softmax classifier fed by all active `columnXX_outer_shell_context` nodes. The context-only classifier has a weighted cross-entropy energy and is clamped to the same CIFAR-10 label tensor as the main output during predictive-coding training.
- The context teacher requires `--outer_shell_context`; passing a positive context-teacher weight without the context path raises an error.
- This differs from prior per-column outer-shell teacher heads. The prior teacher heads supervised raw per-column shell pools and did not align with the main route. The new head supervises the combined route that already feeds the main classifier.
- Updated `scripts/run_codex_cifar10_depth_spanning.sh` with a seventeenth positional argument for `outer_shell_context_teacher_weight`. Existing calls keep the default value `0.0`.

Added next sweep script:

- `scripts/run_codex_outer_context_teacher_10col3shared_sweep.sh`.

Planned context-teacher sweep:

| Case | `outer_shell_context_teacher_weight` | Seeds | Purpose |
| --- | ---: | --- | --- |
| Very small context teacher | 0.0005 | 42, 99, 7 | Test whether a weak route-local class target makes the context route informative without disrupting the main path. |
| Small context teacher | 0.001 | 42, 99, 7 | Test whether a slightly stronger route-local class target improves the context route or begins to reproduce teacher-driven disruption. |

Shared settings for the new sweep:

- `shell_lr_multipliers=1,1.5,2,3`.
- `outer_shell_context=on`.
- `column_teacher_weight=0.0`.
- `shell_teacher_weights=0,0,0,0`.
- `column_shell_teacher_weights=0,0,0,0`.
- `num_columns=10`, `num_shared=3`, and `active_nonshared=7`.
- `combiner=shell_attention`.
- No backbone bypass.
- 20 epochs.
- Learning rate 0.005.
- Diagnostic mode `composer_shells`.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_outer_context_teacher_10col3shared_sweep.sh
```

Verification completed:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_outer_context_teacher_10col3shared_sweep.sh
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: passed, 35 tests.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py -q
```

Result: passed, 20 tests.

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

## 2026-06-29 Six-Column Capacity Sweep Result

Timestamp and machine: 2026-06-29 22:44:33 EDT on `rogdora43`.

Completed command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh
```

Master log:

`results/codex_shell_bridge_readout_column_capacity_sweep_rogdora43_20260629_133952.log`

The tested configuration used six active columns, direct per-column shell readout, per-column shell bridge readout, no backbone bypass, and zero class-energy weight on column and shell teacher heads. `column_shell_paths_only` is the readout path that keeps direct pooled `(column, shell)` edges plus per-column shell bridge edges and removes the legacy `column_pool` edge.

Results:

| Seed | Columns | Test accuracy | Best validation accuracy | Best validation epoch | `column_shell_paths_only` test | `without_outer_shell` test | `outer_shell_only` test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 6 | 27.36% | 28.14% | 7 | 27.35% | 25.34% | 10.00% |
| 42 | 6 | 22.86% | 21.26% | 3 | 22.86% | 25.21% | 10.00% |

Comparison against four active columns:

| Seed | Four-column test | Six-column test | Change |
| --- | ---: | ---: | ---: |
| 99 | 24.60% | 27.36% | +2.76 points |
| 42 | 29.89% | 22.86% | -7.03 points |

Additional readout diagnostics:

| Seed | Direct shell readout only | Shell bridge only | Direct plus bridge |
| --- | ---: | ---: | ---: |
| 99 | 12.44% | 21.45% | 27.35% |
| 42 | 18.31% | 22.12% | 22.86% |

Interpretation:

Six active columns are not a clean capacity improvement. Seed 99 improved, but seed 42 degraded sharply. Both runs selected early validation checkpoints, epoch 7 for seed 99 and epoch 3 for seed 42. The mechanism looks less stable than the four-column direct-plus-bridge runs, whose best validation epochs were 17, 18, and 18.

The collapse criterion also worsened for seed 42. With four columns, removing `outer_shell` from the combined shell path dropped seed-42 test accuracy from 30.01% to 16.29%. With six columns, removing `outer_shell` increased seed-42 test accuracy from 22.86% to 25.21%. That means the larger graph made `outer_shell` harmful in the combined path for that seed, even though `outer_shell` still had stable nonzero norm. Seed 99 still used `outer_shell` constructively, but only weakly: removing it dropped the shell-path test from 27.35% to 25.34%.

Next experimental objective:

Separate active-column count from learning-rate instability. The next sweep tests five active columns at the previous learning rate and six active columns at a lower learning rate. This keeps the same predictive-coding shell bridge plus direct shell readout mechanism and avoids reverting to non-columnar shortcuts.

Added script:

`scripts/run_codex_shell_bridge_readout_capacity_lr_followup.sh`

Planned runs:

1. Seed 99, five active columns, learning rate `0.005`.
2. Seed 42, five active columns, learning rate `0.005`.
3. Seed 99, six active columns, learning rate `0.0025`.
4. Seed 42, six active columns, learning rate `0.0025`.

The decision rule is:

- If five columns improves seed 99 without damaging seed 42, continue capacity search around five columns.
- If lower learning rate rescues six columns on seed 42, keep six columns and retest seed 7.
- If neither condition holds, return to four active columns and make the next architectural change inside the shell bridge rather than adding columns.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_readout_capacity_lr_followup.sh
```

Verification:

```bash
bash -n scripts/run_codex_shell_bridge_readout_capacity_lr_followup.sh
```

Result: passed.

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

## 2026-06-29 Shell Bridge Sweep Results

Timestamp and machine: 2026-06-29 04:02:20 EDT on `rogdora43`.

Completed command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_sweep.sh
```

Master log:

`results/codex_shell_bridge_sweep_rogdora43_20260628_122628.log`

The tested `column_shell_bridge` is one Gaussian predictive-coding latent per active column. It receives that column's pooled `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` vectors and connects to the main CIFAR-10 output classifier. The direct `(column, shell)` readout is a separate classifier edge from each pooled shell vector to the main output classifier. CIFAR-10 chance accuracy is 10 percent.

Results:

| Case | Test accuracy | Best validation accuracy | Best validation epoch | Main diagnostic |
| --- | ---: | ---: | ---: | --- |
| Seed 42, bridge only, no shell teachers | 26.98% | 26.26% | 17 | Bridge path is above chance; `outer_shell` alone is chance-level, but removing it from the bridge drops test bridge accuracy from 26.87% to 21.72%. |
| Seed 42, bridge plus direct shell readout, no shell teachers | 28.84% | 29.04% | 19 | Best run so far. Direct readout alone is 11.26%, bridge alone is 20.87%, and direct plus bridge is 28.91%. |
| Seed 99, bridge only, no shell teachers | 26.94% | 27.94% | 20 | Bridge path is above chance; removing `outer_shell` from the bridge drops test bridge accuracy from 26.75% to 20.15%. |
| Seed 7, bridge only, no shell teachers | 28.76% | 28.94% | 19 | Bridge path is above chance; removing `outer_shell` from the bridge drops test bridge accuracy from 28.76% to 17.23%. |

Interpretation:

The collapse criterion improved. `outer_shell` remains not independently class-readable: `column_shell_bridge_outer_shell_only` is roughly chance on all seeds. But `outer_shell` is no longer merely unused activation mass. In the bridge-only runs, removing `outer_shell` from the shell bridge consistently damages the classifier, with test drops of 5.15, 6.60, and 11.53 percentage points for seeds 42, 99, and 7. The mechanism is that `outer_shell` contributes through a coupled bridge latent even though the isolated `outer_shell` route cannot classify by itself.

The seed-42 bridge plus direct shell readout run showed a stronger combined effect. The direct shell readout path and bridge path were weak when isolated, but their combined classifier edges recovered 28.91% test accuracy. This suggests that the main output classifier used complementary shell evidence from the two per-column routes. The current logs do not yet show which shell matters inside the full combined direct-plus-bridge route, because the prior ablations masked direct readout and bridge readout separately.

Implemented diagnostic change:

Added `evaluate_column_shell_path_ablations` in `scripts/train_cifar10_depth_spanning.py`. `column_shell_paths_only` keeps only direct per-column shell output edges and shell bridge output edges. `column_shell_paths_without_<shell>` drops one shell from both direct readout edges and bridge input edges. `column_shell_paths_<shell>_only` keeps one shell in both routes. This is an evaluation-only change and does not change training, loaders, the predictive-coding graph, or upstream FabricPC.

Added test coverage in `tests/test_pooled_readout_norm.py` for `mask_column_shell_path_inputs`, verifying that the combined shell-path mask zeroes both direct output edges and shell bridge input edges for the selected shell.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result: 19 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest --ignore=tests/test_cifar_data.py
```

Result: 139 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest
```

Result: 147 passed, 5 failed, 9 errors. The failures are confined to `tests/test_cifar_data.py`; the sandbox cannot create `/home/ni/.local/share/columnar_cl_fabricpc`, and the CIFAR data directory was not present under the workspace for a redirected run. No training or data-loader code was changed.

Next experimental objective:

Replicate the seed-42 bridge plus direct shell readout configuration across seeds and rerun seed 42 with the new combined shell-path ablations. This answers whether the best configuration is seed-stable and whether `outer_shell` contributes through the full direct-plus-bridge shell pathway. The column teacher head remains in the graph with weight `0.0`, so it contributes no class energy during training.

Added script:

`scripts/run_codex_shell_bridge_readout_replicate_sweep.sh`

Planned runs:

1. Seed 42, bridge plus direct shell readout, zero column and shell teacher energy.
2. Seed 99, bridge plus direct shell readout, zero column and shell teacher energy.
3. Seed 7, bridge plus direct shell readout, zero column and shell teacher energy.

Expected runtime is roughly 7 to 8 hours based on the 100 to 102 minute runtime per 20-epoch run in the completed bridge sweep.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_readout_replicate_sweep.sh
```

Verification:

```bash
bash -n scripts/run_codex_shell_bridge_readout_replicate_sweep.sh
```

Result: passed.

```bash
git diff --check
```

Result: passed.

## 2026-06-29 Bridge Plus Direct Shell Readout Replicate Results

Timestamp and machine: 2026-06-29 13:28:06 EDT on `rogdora43`.

Completed command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_readout_replicate_sweep.sh
```

Master log:

`results/codex_shell_bridge_readout_replicate_sweep_rogdora43_20260629_040553.log`

The tested configuration used four active columns, direct per-column shell readout, per-column shell bridge readout, no backbone bypass, and zero class-energy weight on column and shell teacher heads. `column_shell_paths_only` means the output classifier receives only direct pooled `(column, shell)` edges plus per-column shell bridge edges. `column_shell_paths_without_outer_shell` means the same shell-path readout with `outer_shell` removed from both the direct pooled shell edges and the shell bridge input edges.

Results:

| Seed | Test accuracy | Best validation accuracy | Best validation epoch | `column_shell_paths_only` test | `without_outer_shell` test | `outer_shell_only` test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 29.89% | 30.34% | 17 | 30.01% | 16.29% | 10.00% |
| 99 | 24.60% | 25.56% | 18 | 24.66% | 19.02% | 10.00% |
| 7 | 33.62% | 33.72% | 18 | 33.67% | 24.93% | 10.00% |

Additional readout diagnostics:

| Seed | Direct shell readout only | Shell bridge only | Direct plus bridge |
| --- | ---: | ---: | ---: |
| 42 | 10.00% | 23.33% | 30.01% |
| 99 | 15.09% | 19.13% | 24.66% |
| 7 | 15.17% | 28.66% | 33.67% |

Interpretation:

The collapse criterion improved again. `outer_shell` is still not independently class-readable, because `column_shell_paths_outer_shell_only` is at CIFAR-10 chance accuracy. But `outer_shell` contributes contextually through the combined shell pathway. Removing `outer_shell` from both direct shell readout and shell bridge input drops test accuracy by 13.72 points on seed 42, 5.64 points on seed 99, and 8.74 points on seed 7.

The mechanism is now consistent with the columnar target. Individual shells are weak as isolated classifiers, but the coupled direct-plus-bridge path is substantially stronger than either direct shell readout or bridge readout alone. The weak seed 99 is not a full collapse. It still reaches 24.66% through the shell paths, and removing `outer_shell` still damages the model. The current weakness is seed-to-seed reliability and total classification strength.

Next experimental objective:

Increase active column capacity while preserving the same predictive-coding shell mechanism. This tests whether the weak seed is a capacity and specialization problem. The next script uses six active columns instead of four, with the same `embed_dim = 64`, `microcolumn_dim = 32`, direct per-column shell readout, per-column shell bridge readout, zero teacher energy, and no backbone bypass.

Added script:

`scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh`

Planned runs:

1. Seed 99, six active columns, bridge plus direct shell readout, zero column and shell teacher energy.
2. Seed 42, six active columns, bridge plus direct shell readout, zero column and shell teacher energy.

Seed 99 is first because it was the weak case. Seed 42 is second because it gives a direct comparison against a stable middle case. If six columns improves seed 99 without damaging seed 42, the next follow-up should test seed 7 and then consider eight columns.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh
```

Verification:

```bash
bash -n scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh
```

Result: passed.

```bash
git diff --check
```

Result: passed.

## 2026-06-29 Intra-Column Shell Promotion Implementation

Timestamp and machine: 2026-06-29 22:51:49 EDT on `rogdora43`.

Reason for changing direction:

The six-column capacity run showed that simply adding columns is not a clean next mechanism. Seed 99 improved, but seed 42 degraded, and the six-column seed 42 ablation showed that `outer_shell` could become harmful in the combined shell path. The next architectural change should therefore improve within-column shell coupling before adding more columns.

Implemented mechanism:

Each depth-spanning column now promotes evidence through the ordered shells:

1. `hard_kernel` projects into `inner_shell`.
2. `inner_shell` projects into `middle_shell`.
3. `middle_shell` projects into `outer_shell`.

Each promotion is a learned signed matrix from the source shell feature slice to the target shell feature slice. The promotion output is added before shell-wise LayerNorm, so the target shell receives lower-shell evidence while its feature norm remains pinned by the existing normalization path. The promotion matrices are initialized with small normal weights. The three promotion gains are initialized to 0.05, one for each adjacent shell pair.

This is now part of the `DepthSpanningColumnNode` architecture rather than a command-line option. The scope shift is that a depth-spanning column now means a shell-typed column with an intra-column shell cascade, not four independent shell slices mixed only from K, L, and B pathways.

Files changed:

- `columnar_cl_fabricpc/columns/depth_spanning_column.py`: added `SHELL_PROMOTION_PAIRS`, promotion matrix and bias parameters, promotion gains, and the forward-pass cascade.
- `columnar_cl_fabricpc/columns/__init__.py`: exported `SHELL_PROMOTION_PAIRS`.
- `scripts/train_cifar10_depth_spanning.py`: records `Shell promotion: enabled` in experiment output.
- `scripts/run_codex_cifar10_depth_spanning.sh`: records `shell_promotion: on` and includes `shellpromotionon` in result log filenames.
- `scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh`: records `shell_promotion: on` in case logs if the superseded capacity script is rerun.
- `scripts/run_codex_shell_bridge_readout_capacity_lr_followup.sh`: records `shell_promotion: on` in case logs if the capacity learning-rate follow-up is used later.
- `tests/test_depth_spanning_column.py`: added parameter-shape coverage and a forward test proving that shell promotion changes the column output.
- `scripts/run_codex_shell_promotion_replicate_sweep.sh`: added the next sequential experiment script.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py
```

Result: 38 passed.

```bash
bash -n scripts/run_codex_shell_promotion_replicate_sweep.sh scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh scripts/run_codex_shell_bridge_readout_capacity_lr_followup.sh
```

Result: passed.

Next experiment:

Run a three-seed replicate of the previous four-column shell bridge plus direct shell readout configuration with shell promotion active inside every column. Seed 99 runs first because it was the weak four-column replicate case. Seed 42 runs second because it was the stable middle case. Seed 7 runs third because it was the strongest replicate case.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_promotion_replicate_sweep.sh
```

The capacity learning-rate follow-up script is no longer the recommended next run. It remains useful later if shell promotion improves stability and we return to capacity scaling.

## 2026-06-30 Shell Promotion Sweep Result and HiBaCaML Paper Recheck

Timestamp and machine: 2026-06-30 EDT on `rogdora43`.

Completed command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_promotion_replicate_sweep.sh
```

Master log:

`results/codex_shell_promotion_replicate_sweep_rogdora43_20260629_225631.log`

The tested configuration used four active columns, direct per-column shell readout, per-column shell bridge readout, no backbone bypass, zero class-energy weight on column and shell teacher heads, and the newly implemented intra-column shell cascade. CIFAR-10 chance accuracy is 10 percent.

Results:

| Seed | Test accuracy | Best validation accuracy | Best validation epoch | `column_shell_paths_only` test | `without_outer_shell` test | `outer_shell_only` test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 27.91% | 29.32% | 19 | 28.13% | 23.34% | 10.00% |
| 42 | 29.31% | 28.86% | 17 | 29.03% | 16.27% | 10.00% |
| 7 | 28.17% | 28.20% | 10 | 27.87% | 22.05% | 10.00% |

Comparison against the previous four-column bridge plus direct shell readout replicate:

| Setting | Mean test accuracy | Test accuracy range |
| --- | ---: | ---: |
| No shell cascade | 29.37% | 9.02 points |
| Shell cascade | 28.46% | 1.40 points |

Interpretation:

The shell cascade reduced seed-to-seed spread substantially, but it also lowered mean accuracy by 0.91 points and removed the high seed-7 result. This is a stability gain and an accuracy tradeoff rather than a collapse. `outer_shell` remained contextually useful in the combined shell pathway: removing `outer_shell` dropped `column_shell_paths_only` by 4.79 points on seed 99, 12.76 points on seed 42, and 5.82 points on seed 7. `outer_shell` was still not independently class-readable because `column_shell_paths_outer_shell_only` stayed at CIFAR-10 chance accuracy on all three seeds.

Paper recheck:

Re-reading `hibacaml_agi26.pdf` changes the architectural interpretation. The paper defines shell semantics radially: inner shells hold reusable abstraction, middle shells hold stabilizing semi-general structure, and outer shells hold task-local exploratory residue. It also says that promotion moves outer material inward during consolidation, while demotion moves stale material outward during controlled forgetting. The implemented shell cascade ran in the opposite direction: `hard_kernel -> inner_shell -> middle_shell -> outer_shell`. That is an outward evidence cascade, not the paper's consolidation dynamic.

The current code is still useful as an experiment because it showed that coupling shells can stabilize seed behavior and preserve contextual `outer_shell` use. But it should not be described as faithful HiBaCaML shell promotion. The next implementation should replace outward cascade thinking with ColBa-style shell dynamics:

1. Make shell interaction semantics explicit in each typed microcolumn K, L, and B rather than only after the K/L/B outputs are mixed into one feature axis.
2. Implement same-tier inhibition, where features within the same shell tier compete so overlapping causal footprints are not double-counted.
3. Implement inward consolidation as rare outer-to-middle-to-inner movement, gated by evidence that a feature is reusable rather than merely useful for the current task.
4. Implement outward demotion as the path for stale or over-specialized inner material to move toward the outer shell.
5. Keep pruning conservative for CIFAR-10: outer shell only after warmup, middle shell only at boundaries, inner shell not at all in the initial regime.

Recommendation:

Do not tune the current outward cascade gain as the next main experiment. The next main step should be to correct the shell dynamics toward the paper: typed microcolumn shell tiers plus same-tier inhibition first, then rare inward consolidation. For plain CIFAR-10, where there is not yet multi-task evidence, the most faithful first mechanism is same-tier inhibition and shell precision, not active promotion. Promotion should become meaningful after there is either multi-task evidence or a carefully defined proxy for reusable evidence across augmentations and epochs.

## 2026-06-30 Folded Shell Dynamics Implementation

Timestamp and machine: 2026-06-30 08:07:43 EDT on `rogdora43`.

Direction update:

The useful part of the previous outward shell cascade is shell-to-shell communication and skip-like evidence flow. The paper-faithful correction is that this path should not be called consolidation or promotion. In ColBa terminology, promotion is outer-to-inner movement under evidence that material has become reusable. The outward path is now treated as an auxiliary evidence cascade, while the next paper-aligned mechanism is same-tier inhibition inside each typed K, L, and B pathway.

Implemented mechanism:

- Renamed the outward `hard_kernel -> inner_shell -> middle_shell -> outer_shell` mechanism to `shell_evidence_cascade`.
- Added `shell_evidence_cascade_scale`, a three-value initial gain for adjacent outward shell pairs. The default remains `0.05,0.05,0.05` so the prior experiment's useful stabilizing path remains present.
- Added `shell_inhibition_strengths`, a four-value same-tier inhibition vector in hard-kernel, inner-shell, middle-shell, and outer-shell order. The default is `0,0.35,0.22,0.10`, matching the CIFAR shell-dynamics values in the paper with no inhibition applied to the protected hard kernel.
- Applied same-tier inhibition separately inside each K, L, and B pathway before shell-specific K/L/B mixing. This is closer to the paper than applying inhibition only after K, L, and B have already been merged.
- Removed `scripts/run_codex_shell_promotion_replicate_sweep.sh` because the name is now misleading. The completed historical experiment remains documented above.
- Added `scripts/run_codex_shell_dynamics_replicate_sweep.sh` for the next experiment.

Mechanism details:

For one shell tier, the inhibition step computes each feature's magnitude and subtracts `gamma` times the mean magnitude of the other features in the same shell tier, where `gamma` is the shell's inhibition strength. The feature sign is preserved and negative magnitudes are clipped to zero. This creates same-tier competition without ordinary backprop-specific machinery and keeps the implementation inside the predictive-coding graph.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile columnar_cl_fabricpc/columns/depth_spanning_column.py scripts/train_cifar10_depth_spanning.py tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py
```

Result: 41 passed.

```bash
bash -n scripts/run_codex_shell_dynamics_replicate_sweep.sh scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_shell_bridge_readout_column_capacity_sweep.sh scripts/run_codex_shell_bridge_readout_capacity_lr_followup.sh
```

Result: passed.

Next experiment:

Run the same three-seed four-column bridge plus direct shell readout replicate, now with outward evidence cascade plus paper-aligned same-tier inhibition. The key comparison is against both the no-cascade replicate and the outward-cascade replicate:

- Collapse check: `column_shell_paths_only` should remain above chance on every seed.
- `outer_shell` check: removing `outer_shell` from `column_shell_paths_only` should still damage accuracy.
- Stability check: the test accuracy range should stay closer to the shell-cascade run than to the no-cascade run.
- Accuracy check: mean test accuracy should recover some of the 0.91-point drop from the shell-cascade run.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_dynamics_replicate_sweep.sh
```

## 2026-06-30 Shell Dynamics Replicate Result

Timestamp and machine: 2026-06-30 18:32:13 EDT on `rogdora43`.

Completed command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_dynamics_replicate_sweep.sh
```

Master log:

`results/codex_shell_dynamics_replicate_sweep_rogdora43_20260630_081110.log`

The tested configuration used four active columns, no backbone bypass, direct per-column shell readout, per-column shell bridge readout, zero class-energy weight on column and shell teacher heads, outward shell evidence cascade with `shell_evidence_cascade_scale = 0.05,0.05,0.05`, and same-tier inhibition with `shell_inhibition_strengths = 0,0.35,0.22,0.10`. CIFAR-10 chance accuracy is 10 percent.

Results:

| Seed | Test accuracy | Best validation accuracy | Best validation epoch | `column_shell_paths_only` test | `without_outer_shell` test | `outer_shell_only` test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 26.90% | 27.48% | 18 | 26.96% | 17.67% | 10.00% |
| 42 | 29.22% | 29.46% | 17 | 29.03% | 24.87% | 10.00% |
| 7 | 30.14% | 30.62% | 18 | 30.30% | 14.61% | 10.00% |

Additional readout diagnostics:

| Seed | Direct shell readout only | Shell bridge only | Direct plus bridge |
| --- | ---: | ---: | ---: |
| 99 | 13.40% | 19.09% | 26.96% |
| 42 | 14.24% | 24.83% | 29.03% |
| 7 | 10.65% | 25.32% | 30.30% |

Comparison:

| Setting | Mean test accuracy | Test accuracy range |
| --- | ---: | ---: |
| No shell cascade | 29.37% | 9.02 points |
| Outward shell cascade, no same-tier inhibition | 28.46% | 1.40 points |
| Outward shell cascade plus same-tier inhibition | 28.75% | 3.24 points |

Mechanistic interpretation:

The latest run shows coupled shell evidence, not standalone shell classifiers. Direct shell readout stayed weak at 13.40 percent, 14.24 percent, and 10.65 percent. Shell bridge readout was stronger at 19.09 percent, 24.83 percent, and 25.32 percent. Keeping both direct shell-pool edges and bridge edges gave the full shell-path values of 26.96 percent, 29.03 percent, and 30.30 percent.

`K`, `L`, and `B` are the three column pathways in `columnar_cl_fabricpc/columns/depth_spanning_column.py`: `K` combines multi-depth stage inputs, `L` applies local depthwise convolution to the deepest stage input, and `B` projects pooled deepest-stage context back over the token grid. Same-tier inhibition applies separately inside `K`, `L`, and `B` before the shell-specific pathway outputs are mixed. The inhibition operation computes each feature magnitude and subtracts the shell strength times the mean magnitude of other features in the same shell tier, then clips the result at zero while preserving sign.

The direct shell readout values come from `column_shell_pool -> output` edges. The bridge-only values come from `column_shell_pool -> columnXX_shell_bridge -> output` paths. The combined values keep both sets of edges active at `output`. `column_shell_paths_without_outer_shell` masks `outer_shell` direct readout sources and masks `outer_shell` inputs into each shell bridge. `column_shell_paths_outer_shell_only` keeps only `outer_shell` direct sources and only `outer_shell` bridge inputs.

Removing `outer_shell` from the combined shell paths dropped accuracy by 9.29 points on seed 99, 4.16 points on seed 42, and 15.69 points on seed 7. `outer_shell` alone stayed at chance on all three seeds. The current data therefore supports the mechanism that `outer_shell` contributes contextual evidence through the coupled direct-plus-bridge shell path, while `outer_shell` is not independently class-readable.

What the data does not yet explain:

Same-tier inhibition changed the outward-cascade-only result by -1.01 points on seed 99, -0.09 points on seed 42, and +1.97 points on seed 7. The reported accuracies also do not explain why removing `outer_shell` leaves seed 42 at 24.87 percent but leaves seed 7 at 14.61 percent. Answering that needs shell-resolved edge weights, bridge weights, or gradient diagnostics.

Falsifier checked:

The current interpretation would be falsified if `column_shell_paths_without_outer_shell` matched `column_shell_paths_only` while `column_shell_paths_outer_shell_only` stayed at chance. That observation is not present. Removing `outer_shell` damages the shell path on every seed.

Next diagnostic:

Halve same-tier inhibition to `0,0.175,0.11,0.05` while keeping the same graph, outward cascade scale, shell bridge path, readout paths, and zero teacher weights. This tests whether the full inhibition strengths suppress useful shell feature magnitudes too strongly before K/L/B shell mixing and outward shell evidence cascade.

Added script:

`scripts/run_codex_shell_inhibition_half_replicate_sweep.sh`

Planned runs:

1. Seed 99, half same-tier inhibition, outward shell evidence cascade, bridge plus direct shell readout.
2. Seed 42, half same-tier inhibition, outward shell evidence cascade, bridge plus direct shell readout.
3. Seed 7, half same-tier inhibition, outward shell evidence cascade, bridge plus direct shell readout.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_inhibition_half_replicate_sweep.sh
```

Verification:

```bash
bash -n scripts/run_codex_shell_inhibition_half_replicate_sweep.sh
```

Result: passed.

## 2026-07-01 Half Same-Tier Inhibition Result

Timestamp and machine: 2026-07-01 06:06:16 EDT on `rogdora43`.

Completed command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_inhibition_half_replicate_sweep.sh
```

Master log:

`results/codex_shell_inhibition_half_replicate_sweep_rogdora43_20260630_183953.log`

Per-seed logs:

- `results/codex_resnet18_shelldynamicson_halfinhib_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadouton_colshellbridgeon_nobypass_norm_fixedln_seed99_lr0p005_ep20_shells_rogdora43_20260630_183953.log`
- `results/codex_resnet18_shelldynamicson_halfinhib_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadouton_colshellbridgeon_nobypass_norm_fixedln_seed42_lr0p005_ep20_shells_rogdora43_20260630_214411.log`
- `results/codex_resnet18_shelldynamicson_halfinhib_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadouton_colshellbridgeon_nobypass_norm_fixedln_seed7_lr0p005_ep20_shells_rogdora43_20260701_004751.log`

The tested configuration used four active columns, no backbone bypass, direct per-column shell readout, per-column shell bridge readout, zero class-energy weight on column and shell teacher heads, outward shell evidence cascade with `shell_evidence_cascade_scale = 0.05,0.05,0.05`, and half-strength same-tier inhibition with `shell_inhibition_strengths = 0,0.175,0.11,0.05`. CIFAR-10 chance accuracy is 10 percent.

Results:

| Seed | Test accuracy | Best validation accuracy | Best validation epoch | `column_shell_paths_only` test | `without_outer_shell` test | `outer_shell_only` test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 22.02% | 22.64% | 5 | 22.01% | 21.61% | 10.00% |
| 42 | 24.50% | 25.04% | 17 | 24.51% | 19.75% | 10.00% |
| 7 | 30.46% | 30.38% | 18 | 30.33% | 19.67% | 10.00% |

Additional readout diagnostics:

| Seed | Direct shell readout only | Shell bridge only | Direct plus bridge |
| --- | ---: | ---: | ---: |
| 99 | 14.43% | 19.56% | 22.01% |
| 42 | 10.33% | 17.97% | 24.51% |
| 7 | 16.10% | 27.48% | 30.33% |

Comparison:

| Setting | Mean test accuracy | Test accuracy range |
| --- | ---: | ---: |
| No shell cascade | 29.37% | 9.02 points |
| Outward shell cascade, no same-tier inhibition | 28.46% | 1.40 points |
| Outward shell cascade plus full same-tier inhibition | 28.75% | 3.24 points |
| Outward shell cascade plus half same-tier inhibition | 25.66% | 8.44 points |

Direct comparison against full same-tier inhibition:

| Seed | Full inhibition test | Half inhibition test | Change | Full `column_shell_paths_only` | Half `column_shell_paths_only` | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 99 | 26.90% | 22.02% | -4.88 points | 26.96% | 22.01% | -4.95 points |
| 42 | 29.22% | 24.50% | -4.72 points | 29.03% | 24.51% | -4.52 points |
| 7 | 30.14% | 30.46% | +0.32 points | 30.30% | 30.33% | +0.03 points |

`outer_shell` contribution inside the combined shell path:

| Seed | Full inhibition drop when `outer_shell` is removed | Half inhibition drop when `outer_shell` is removed |
| --- | ---: | ---: |
| 99 | 9.29 points | 0.40 points |
| 42 | 4.16 points | 4.76 points |
| 7 | 15.69 points | 10.66 points |

Conclusions:

Half-strength same-tier inhibition is a negative diagnostic in this implementation. It did not produce a collapse to chance, but it lowered mean test accuracy from 28.75 percent to 25.66 percent and widened the three-seed range from 3.24 points to 8.44 points. The seed 99 and seed 42 results degraded by nearly five points each, while seed 7 was unchanged at the combined shell-path level.

The seed 99 validation trace shows an early peak rather than no learning: validation accuracy reached 22.64 percent at epoch 5, then dropped as low as 9.68 percent at epoch 11 before recovering partially to 19.36 percent at epoch 20. This points to a training instability in that seed under half inhibition, not a failure to form any class signal.

The `outer_shell` conclusion changed sharply for seed 99. Under full inhibition, removing `outer_shell` from the combined shell path cost 9.29 points. Under half inhibition, the same removal cost only 0.40 points. Since `outer_shell_only` stayed at chance in both runs, the change is not that `outer_shell` became independently class-readable. The change is that the coupled direct-plus-bridge path stopped using `outer_shell` as helpful contextual evidence for seed 99.

For seed 7, half inhibition preserved the final accuracy but did not explain the broader behavior, because `outer_shell` still contributed 10.66 points and the shell bridge alone improved from 25.32 percent to 27.48 percent. For seed 42, half inhibition reduced bridge-only accuracy from 24.83 percent to 17.97 percent and reduced combined shell-path accuracy from 29.03 percent to 24.51 percent. The same coefficient change therefore did not act uniformly across seeds.

The diagnostic answer is that the full same-tier inhibition coefficients are better supported than the half-strength coefficients for the current graph and training loop. This does not prove the full coefficients are optimal. It does show that simply allowing more same-tier shell feature magnitude through the K, L, and B pathways does not improve CIFAR-10 classification or stability here.

Pause point:

No follow-up experiment is proposed here. The next step is to inspect and discuss the architecture and code mechanisms before choosing another direction.

## 2026-07-01 Column-Shell Composer Implementation

Timestamp and machine: 2026-07-01 08:25:17 EDT on `rogdora43`.

Direction update:

The recent result pattern showed that the main whole-column `combiner -> column_pool -> output` route stayed at chance, while the side route that preserved `(column, shell)` identity through direct per-column shell pools and per-column shell bridges carried useful class signal. The next implementation therefore changed the main composer mechanism instead of tuning inhibition coefficients again.

Implemented mechanism:

- Added `ColumnShellComposerNode` in `columnar_cl_fabricpc/columns/accuracy_nodes.py`.
- Exported `ColumnShellComposerNode` from `columnar_cl_fabricpc/columns/__init__.py`.
- Added `--combiner shell_attention` to `scripts/train_cifar10_depth_spanning.py`.
- When `--combiner shell_attention` is selected, the graph keeps the node name `combiner`, but uses `ColumnShellComposerNode` rather than `MaskedColumnCombinerNode`.
- Updated `scripts/run_codex_cifar10_depth_spanning.sh` with an optional eleventh positional argument for combiner mode. Existing calls default to `sum`.
- Added `scripts/run_codex_shell_composer_replicate_sweep.sh` for a three-seed shell-composer experiment.

Mechanism details:

`ColumnShellComposerNode` receives the active column latents, where each column latent has shape `(batch, tokens, embed_dim)`. It slices the final feature axis into `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` using the same shell layout as `DepthSpanningColumnNode`. For each column index `c` and shell name `s`, it applies a learned projection:

```text
projected(c, s) = column(c)[..., shell_slice(s)] @ W_col{c}_{s} + b_col{c}_{s}
```

It also learns `component_attention`, a matrix with shape `(num_columns, 4)`. The row index is the column index and the shell index follows `SHELL_NAMES`. The support mask zeros inactive columns by setting their logits to a large negative value before softmax. The composer output is:

```text
combiner_z_mu = sum over active c and all shells s of attention(c, s) * projected(c, s)
```

This output remains a FabricPC predictive-coding node with Gaussian energy. Its latent is inferred during predictive-coding inference, then `column_pool` averages it and feeds the main `output` classifier. This keeps the main architectural path inside predictive coding rather than routing around it with classifier-only side edges.

The implementation deliberately does not implement promotion or demotion. The HiBaCaML paper treats promotion as rare and dependent on multi-task evidence. Plain CIFAR-10 does not provide that evidence. The current change instead fixes the composer mechanism that was losing column and shell identity before readout.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile columnar_cl_fabricpc/columns/accuracy_nodes.py columnar_cl_fabricpc/columns/__init__.py scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_shell_composer_replicate_sweep.sh
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py tests/test_depth_spanning_column.py
```

Result: 43 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -k 'not cifar_data'
```

Result: 145 passed, 22 deselected.

Full-suite note:

Running the full suite without excluding data tests reached `tests/test_cifar_data.py` and failed because the sandbox could not write to `/home/ni/.local/share/columnar_cl_fabricpc`. The non-data tests passed. This failure is not specific to the column-shell composer implementation.

Next experiment:

Run the three-seed shell-composer replicate. The primary success criterion is that the `column_only` ablation, which now means `ColumnShellComposerNode -> column_pool -> output`, rises clearly above chance. The secondary criteria are that `column_shell_readout_plus_bridge` does not collapse and that seed-to-seed range does not widen relative to the full-inhibition shell-dynamics run.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_composer_replicate_sweep.sh
```

## 2026-07-01 Shell-Composer Diagnostic Instrumentation

Timestamp and machine: 2026-07-01 19:22:48 EDT on `rogdora43`.

Direction update:

The three-seed shell-composer run showed a stable combined classifier but did not satisfy the primary criterion that `ColumnShellComposerNode -> column_pool -> output` should become independently class-informative. The next change therefore adds diagnostic logging and evaluation around the existing shell composer rather than changing the architecture.

Topology note:

No graph nodes or edges changed. The ASCII architecture diagram in `scripts/train_cifar10_depth_spanning.py` remains accurate. The new code only inspects learned parameters and evaluates copied parameter trees with selected composer projections zeroed.

Implemented diagnostics:

- Added `--diagnose_composer` to `scripts/train_cifar10_depth_spanning.py`.
- Added composer attention logging. `attention[c, s]` is the learned softmax weight inside `ColumnShellComposerNode`, where `c` is the active column index and `s` is one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`.
- Added composer projection norm logging. Each value is the Frobenius norm of the learned projection from one `(column, shell)` slice into the shared composer output feature axis.
- Replaced edge-key classifier norm reporting with source-name reporting. The output now names architectural routes such as `column_pool`, `column00_hard_kernel_pool`, and `column00_shell_bridge`.
- Added validation composer component lesions. Each lesion keeps the classifier restricted to `column_pool -> output`, then zeros one active composer projection such as `(col_00, outer_shell)` in a copied parameter tree before evaluation.
- Added `scripts/run_codex_shell_composer_diagnostic_comparison.sh`, which runs a two-case comparison: shell composer with direct shell readout plus bridge, then shell composer without those side routes.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/accuracy_nodes.py
```

Result: passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_shell_composer_diagnostic_comparison.sh scripts/run_codex_shell_composer_replicate_sweep.sh
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result: 26 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -k 'not cifar_data'
```

Result: 148 passed, 22 deselected.

Next diagnostic run:

The recommended next run is a one-seed, two-case shell-composer comparison. It uses seed 42, learning rate 0.005, 8 epochs, no bypass, zero teacher weights, shell diagnostics, and composer diagnostics. The first case preserves the direct per-column shell readout and shell bridge. The second case removes those side routes so `ColumnShellComposerNode -> column_pool -> output` is the only classifier route.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_composer_diagnostic_comparison.sh
```

## 2026-07-02 Shell-Preserving Composer Fix

Timestamp and machine: 2026-07-02 05:49:12 EDT on `rogdora43`.

Direction update:

The shell-composer diagnostic comparison showed that the composer-only case reached 29.65 percent test accuracy at seed 42, but the learned `component_attention` concentrated almost all mass on one hard-kernel component. The old composer projected every `(column, shell)` input into the full `embed_dim` output. That allowed a hard-kernel input slice to write into dimensions later treated as `inner_shell`, `middle_shell`, and `outer_shell` during shell readout ablations.

Mechanism change:

`ColumnShellComposerNode` in `columnar_cl_fabricpc/columns/accuracy_nodes.py` is now shell-preserving.

- `hard_kernel` inputs project only into the `hard_kernel` output slice.
- `inner_shell` inputs project only into the `inner_shell` output slice.
- `middle_shell` inputs project only into the `middle_shell` output slice.
- `outer_shell` inputs project only into the `outer_shell` output slice.
- `component_attention[c, s]` is now normalized over active columns for each shell `s`, where `c` is the column index and `s` is the shell name. Before this change, attention was normalized globally over all `(column, shell)` components.

For an input shell slice with width `w_s`, the learned projection now has shape `(w_s, w_s)`. Before this change, it had shape `(w_s, embed_dim)`. The composer still has the same graph node name, `combiner`, and the same input and output graph edges. This means the top-level ASCII graph in `scripts/train_cifar10_depth_spanning.py` remains accurate.

Diagnostic update:

`diagnose_composer_attention()` in `scripts/train_cifar10_depth_spanning.py` now applies the same per-shell attention normalization as `ColumnShellComposerNode`. Composer projection norm diagnostics still report one norm for each `(column, shell)` projection, but those projections now refer to shell-local matrices.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile columnar_cl_fabricpc/columns/accuracy_nodes.py scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result: 27 passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_shell_composer_diagnostic_comparison.sh scripts/run_codex_shell_composer_replicate_sweep.sh
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -k 'not cifar_data'
```

Result: 149 passed, 22 deselected.

Next diagnostic run:

Run the same short two-case diagnostic comparison again. The primary check is whether the composer-only case avoids chance-level collapse after preventing cross-shell writes. The second check is whether composer lesions now show any non-hard-kernel shell participation.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_composer_diagnostic_comparison.sh 42 8 0.005
```

## 2026-07-02 Shell-Preserving Composer Replicate Prepared

Timestamp and machine: 2026-07-02 10:07:49 EDT on `rogdora43`.

Direction update:

The corrected shell-preserving composer avoided chance-level collapse in the seed-42 composer-only diagnostic. The next test is a three-seed, 20-epoch replicate of the clean composer-only path before adding any new outer-shell mechanism.

Prepared run script:

- Added `scripts/run_codex_shell_preserving_composer_only_replicate_sweep.sh`.
- The script runs seeds 42, 99, and 7 sequentially.
- Each run uses `--combiner shell_attention`, no bypass, no direct per-column shell readout, no per-column shell bridge, zero teacher weights, shell diagnostics, and composer diagnostics.
- Each child run writes its own result log through `scripts/run_codex_cifar10_depth_spanning.sh`.
- The sweep writes a master log in `results/`.

Verification:

```bash
bash -n scripts/run_codex_shell_preserving_composer_only_replicate_sweep.sh
```

Result: passed.

Pasteable command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_preserving_composer_only_replicate_sweep.sh
```

## 2026-07-02 Shell-Preserving Composer Replicate Results

Timestamp and machine: 2026-07-02 16:37:57 EDT on `rogdora43`.

Run script:

- `scripts/run_codex_shell_preserving_composer_only_replicate_sweep.sh`

Master log:

- `results/codex_shell_preserving_composer_only_replicate_sweep_rogdora43_20260702_100835.log`

Child logs:

- `results/codex_resnet18_shelldynamicson_combinershell-attention_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadoutoff_colshellbridgeoff_nobypass_norm_fixedln_seed42_lr0p005_ep20_composer_shells_rogdora43_20260702_100835.log`
- `results/codex_resnet18_shelldynamicson_combinershell-attention_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadoutoff_colshellbridgeoff_nobypass_norm_fixedln_seed99_lr0p005_ep20_composer_shells_rogdora43_20260702_122627.log`
- `results/codex_resnet18_shelldynamicson_combinershell-attention_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadoutoff_colshellbridgeoff_nobypass_norm_fixedln_seed7_lr0p005_ep20_composer_shells_rogdora43_20260702_144346.log`

Run settings:

- The model used `shell_attention`, where the composer node combines the four column shell slices into the column-token representation used by the classifier.
- The run used no backbone bypass, no direct per-column shell readout, no per-column shell bridge, and zero teacher weights.
- The shell evidence cascade was enabled with scale `0.05,0.05,0.05`.
- Shell inhibition strengths were `hard_kernel=0`, `inner_shell=0.35`, `middle_shell=0.22`, and `outer_shell=0.10`.
- Diagnostics were `composer_shells`, which evaluates readout ablations, shell-only ablations, and composer component lesions after training.

Summary:

| Seed | Status | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | --- | ---: | ---: | ---: |
| 42 | Complete | 28.00% | 17 | 27.14% |
| 99 | Complete | 32.46% | 19 | 32.36% |
| 7 | Incomplete | Not recorded | Not recorded | Not recorded |

Seed 7 was killed after the 20-epoch training progress reached the end, before the script wrote validation selection, test accuracy, or diagnostic tables. The log ends with the shell runner reporting `Killed` for the Python process at `scripts/run_codex_cifar10_depth_spanning.sh: line 151`. I am not treating seed 7 as a failed accuracy result, because the result file does not contain the evaluation metrics needed to compare it with seeds 42 and 99.

Completed-seed aggregate:

| Aggregate over complete seeds | Test accuracy |
| --- | ---: |
| Mean of seeds 42 and 99 | 29.75% |
| Minimum complete seed | 27.14% |
| Maximum complete seed | 32.36% |
| Range across complete seeds | 5.22 percentage points |

Mechanism observations:

- The shell-preserving composer did not collapse to chance in the two complete 20-epoch runs.
- `hard_kernel`, the innermost shell slice in each column, is still the strongest classifier source. Removing all hard-kernel shell contribution reduced test accuracy to 11.04% for seed 42 and 16.81% for seed 99.
- `middle_shell`, the intermediate shell slice intended to carry a deeper contextual representation, now contributes measurable signal. The `middle_shell`-only test readout reached 19.13% for seed 42 and 21.65% for seed 99. Removing `middle_shell` reduced test accuracy from 27.14% to 21.16% for seed 42 and from 32.36% to 26.53% for seed 99.
- `inner_shell`, the shell slice between `hard_kernel` and `middle_shell`, is not independently class-readable in these runs. The `inner_shell`-only test readout stayed at 10.00% in both complete seeds, while removing `inner_shell` caused only a small drop.
- `outer_shell`, the most contextual shell slice, is still not independently class-readable. The `outer_shell`-only test readout stayed at 10.00% in both complete seeds. Removing `outer_shell` reduced test accuracy from 27.14% to 23.53% for seed 42 and from 32.36% to 31.94% for seed 99, so it may help the composed representation in some cases without carrying a stable class signal by itself.

Composer attention and projection pattern:

- Seed 42 concentrated `hard_kernel` attention on column 0 and `middle_shell` attention on column 1. The largest projection norms were `col_00.hard_kernel=46.28` and `col_01.middle_shell=23.14`.
- Seed 99 concentrated `hard_kernel` attention on column 2 and `middle_shell` attention on column 1. The largest projection norms were `col_02.hard_kernel=59.52` and `col_01.middle_shell=30.91`.
- This is qualitatively better than the pre-fix composer behavior, because the corrected composer no longer lets one shell write into every other shell slice. The model now finds seed-dependent column identities for the hard and middle shells while preserving shell-local output slices.

Conclusion:

The corrected shell-preserving composer is directionally useful for avoiding immediate chance-level collapse, but it has not yet produced robust CIFAR-10 accuracy. The complete runs show that `hard_kernel` and `middle_shell` carry the useful class signal. The central unresolved architectural issue is that `outer_shell` participates weakly and is not independently class-readable. That matters because the HiBaCaML-inspired direction wants context-bearing shell dynamics rather than only an innermost class path plus a smaller middle-shell contribution.

Recommended next step:

Before adding another architectural mechanism, recover the missing seed 7 evaluation or rerun seed 7 alone with the same configuration. The run was terminated after expensive training had already completed, so the immediate engineering weakness is that long experiments can lose their final metrics if post-training diagnostics are interrupted. After that result is recovered, the next architectural target should be outer-shell participation. The most likely faithful direction is to add a shell-local predictive objective or a shell-preserving context pathway that gives `outer_shell` a direct training signal while keeping classification routed through the predictive-coding graph rather than adding ordinary backpropagation.

## 2026-07-02 10-Column Composer Replicate Prepared

Timestamp and machine: 2026-07-02 16:56:20 EDT on `rogdora43`.

Direction update:

The next run repeats the shell-preserving composer-only replicate with higher column capacity. The only intended architectural capacity change is from 4 columns with 2 shared columns to 10 columns with 3 shared columns. The run uses `active_nonshared=7`, where `active_nonshared` is the number of non-shared columns selected by sparse column modes. This sweep still uses `column_mode=all_active`, so all 10 columns participate in every case.

Prepared run script:

- Added `scripts/run_codex_shell_preserving_composer_only_replicate_sweep_10col3shared.sh`.
- The script runs seeds 42, 99, and 7 sequentially.
- Each run uses `num_columns=10`, `num_shared=3`, and `active_nonshared=7`.
- Each run keeps the previous shell-preserving composer-only settings: `--combiner shell_attention`, no backbone bypass, no direct per-column shell readout, no per-column shell bridge, zero teacher weights, shell diagnostics, and composer diagnostics.
- Each child log name includes `10col_3shared_7active` so it cannot be confused with the prior 4-column logs.

Verification:

```bash
bash -n scripts/run_codex_shell_preserving_composer_only_replicate_sweep_10col3shared.sh
```

Result: passed.

Launch command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_preserving_composer_only_replicate_sweep_10col3shared.sh
```

## 2026-07-03 10-Column Composer Replicate Results

Timestamp and machine: 2026-07-03 02:41:20 EDT on `rogdora43`.

Completed host-run master log:

- `results/codex_shell_preserving_composer_only_replicate_sweep_10col3shared_rogdora43_20260702_170009.log`

Child logs:

- `results/codex_resnet18_shelldynamicson_10col_3shared_7active_combinershell-attention_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadoutoff_colshellbridgeoff_nobypass_norm_fixedln_seed42_lr0p005_ep20_composer_shells_rogdora43_20260702_170009.log`
- `results/codex_resnet18_shelldynamicson_10col_3shared_7active_combinershell-attention_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadoutoff_colshellbridgeoff_nobypass_norm_fixedln_seed99_lr0p005_ep20_composer_shells_rogdora43_20260702_195807.log`
- `results/codex_resnet18_shelldynamicson_10col_3shared_7active_combinershell-attention_column_teacher0p0_shell0_0_0_0_colshell0_0_0_0_colshellreadoutoff_colshellbridgeoff_nobypass_norm_fixedln_seed7_lr0p005_ep20_composer_shells_rogdora43_20260702_225528.log`

Run settings:

- `num_columns=10`, where `num_columns` is the number of depth-spanning column nodes connected to the shell-preserving composer.
- `num_shared=3`, where `num_shared` is the number of columns that would always be active under sparse column modes.
- `active_nonshared=7`, where `active_nonshared` is the number of non-shared columns selected by sparse column modes.
- `column_mode=all_active`, so all 10 columns participated in every seed despite the shared/non-shared labels.
- `combiner=shell_attention`, no backbone bypass, no direct per-column shell readout, no per-column shell bridge, and zero teacher weights.
- JAX reported `[CudaDevice(id=0)]` and backend `gpu` for all three completed child runs.
- Each child run built a 47-node, 93-edge graph with 3,061,694 parameters.

Summary:

| Seed | Status | Best validation accuracy | Best validation epoch | Test accuracy | Training time |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Complete | 22.82% | 6 | 21.53% | 6694.0 s |
| 99 | Complete | 22.06% | 5 | 20.68% | 6673.4 s |
| 7 | Complete | 26.90% | 19 | 25.79% | 6671.8 s |

Aggregate:

| Aggregate over complete seeds | Test accuracy |
| --- | ---: |
| Mean of seeds 42, 99, and 7 | 22.67% |
| Minimum complete seed | 20.68% |
| Maximum complete seed | 25.79% |
| Range across complete seeds | 5.11 percentage points |

Comparison to the previous 4-column shell-preserving composer-only run:

- The 4-column completed seeds had a two-seed mean test accuracy of 29.75% from seed 42 at 27.14% and seed 99 at 32.36%.
- The 10-column version reduced seed 42 from 27.14% to 21.53%.
- The 10-column version reduced seed 99 from 32.36% to 20.68%.
- Seed 7 completed this time at 25.79%, which fixes the missing-result problem from the earlier 4-column sweep but is still below the better 4-column complete seeds.

Shell readout ablations on the test split:

| Seed | Combined | Without hard kernel | Hard kernel only | Without inner shell | Inner shell only | Without middle shell | Middle shell only | Without outer shell | Outer shell only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 21.53% | 12.13% | 18.47% | 20.13% | 15.88% | 19.02% | 13.84% | 21.14% | 10.00% |
| 99 | 20.68% | 15.97% | 20.85% | 20.94% | 16.64% | 20.86% | 10.00% | 20.78% | 10.00% |
| 7 | 25.79% | 15.79% | 22.03% | 24.52% | 13.84% | 25.56% | 16.69% | 24.45% | 10.00% |

Mechanism observations:

- Increasing the column count did not cause chance-level collapse, but it did reduce accuracy substantially.
- `hard_kernel`, the innermost shell slice, remains the load-bearing source. Removing it caused the largest drop in every seed.
- `inner_shell`, the shell slice between `hard_kernel` and `middle_shell`, became more independently class-readable than in the 4-column run. Its test-only readout was 13.84% to 16.64%, versus chance-level behavior in the prior complete 4-column seeds.
- `middle_shell`, the intermediate context shell, became less reliable than it was in the 4-column run. Seed 99's `middle_shell`-only readout was at chance, and removing `middle_shell` barely changed seed 99 or seed 7.
- `outer_shell`, the most contextual shell slice, remains the clearest unsolved problem. Its test-only readout was exactly 10.00% in all three seeds. Removing it had small effects compared with removing `hard_kernel`.

Composer attention pattern:

- Seed 42 concentrated `hard_kernel` attention on column 1, `inner_shell` attention on column 2, and `middle_shell` attention on column 6. `outer_shell` attention stayed nearly uniform across all columns.
- Seed 99 concentrated both `hard_kernel` and `inner_shell` attention on column 2. `middle_shell` attention was distributed across several columns, especially columns 0, 8, and 9. `outer_shell` attention stayed nearly uniform.
- Seed 7 concentrated `hard_kernel` attention on column 3, `inner_shell` attention on column 5, and `middle_shell` attention on column 9. `outer_shell` attention was mostly uniform, with column 9 somewhat higher than the others.

Conclusion:

The 10-column expansion is not an immediate improvement over the 4-column shell-preserving composer. The added capacity appears to diffuse or destabilize the useful middle-shell contribution more than it improves classification. It does, however, show that the composer can select different columns per shell when capacity is available: hard-kernel, inner-shell, and middle-shell attention often specialize to different columns. The remaining architecture gap is still not raw column count. It is the lack of a mechanism that makes the outer shell carry useful class-relevant contextual evidence while preserving predictive-coding structure.

Recommended next step:

Do not continue expanding columns until the shell dynamics are stronger. The next change should target outer-shell participation directly. The conservative HiBaCaML-aligned direction is to add a shell-local predictive objective or shell-preserving context pathway for `outer_shell`, then test it at 4 columns first. Four columns are faster and had better accuracy, so they are the better diagnostic setting for mechanism work. Once the outer shell becomes measurably useful at 4 columns, retest 10 columns to see whether extra capacity helps rather than diluting the signal.

## 2026-07-03 Outer-Shell Context Path Prepared

Timestamp and machine: 2026-07-03 02:56:56 EDT on `rogdora43`.

Direction update:

The next experiment stays with the user's requested 10-column setting while targeting `outer_shell`, the shell slice that remained at chance as an independent readout in the prior 10-column sweep. The implementation combines two graph-native mechanisms:

- A shell-local predictive objective on each active column's `outer_shell` slice.
- A shell-preserving context pathway whose output has only the `outer_shell` width.

Implemented mechanism:

`outer_shell_context` is a new optional per-column Gaussian latent. For active column `c`, `columncc_outer_shell_context` receives that column's pooled `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` vectors. It emits a vector with width equal to the `outer_shell` slice. With `embed_dim=64`, this width is 21 feature dimensions. The context latent connects to the main `output` classifier.

This path differs from the older `column_shell_bridge` path. `column_shell_bridge` emits a full `embed_dim` vector, so it can mix back into all shell-sized regions at readout. `outer_shell_context` emits only an outer-shell-width vector, so the pathway is context-using but shell-preserving at its output.

The initial experiment also sets `column_shell_teacher_weights=0,0,0,0.002`. The value `0.002` is the cross-entropy energy multiplier on each per-column `outer_shell` teacher head. The other per-column shell teachers are omitted. This keeps the auxiliary target local to `outer_shell` and avoids adding direct label pressure to `hard_kernel`, `inner_shell`, or `middle_shell`.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`
- `scripts/run_codex_cifar10_depth_spanning.sh`
- `scripts/run_codex_outer_shell_context_10col3shared_sweep.sh`
- `tests/test_pooled_readout_norm.py`

Training-script changes:

- Added `--outer_shell_context`.
- Added `outer_shell_context_node_name(c)`, where `c` is the column index.
- Added `is_outer_shell_context_node(name)`.
- Added per-column outer-shell context node construction inside `build_depth_spanning_graph`.
- Added context nodes to energy diagnostics and latent diagnostics.
- Added readout ablations for `outer_shell_context_only` and `column_pool_plus_outer_shell_context`.
- Added context-input lesions that keep or drop one shell input at a time.
- Added validation and test tables named `Outer-Shell Context Ablations`.

Runner changes:

- `scripts/run_codex_cifar10_depth_spanning.sh` now accepts:
  - argument 12: `outer_shell_context_mode`, with `on` or `off`.
  - argument 13: `num_columns`.
  - argument 14: `num_shared`.
  - argument 15: `active_nonshared`.
- Result filenames now include the column capacity and `outercontexton/off`, so 4-column and 10-column logs are distinguishable.

Prepared sweep:

- `scripts/run_codex_outer_shell_context_10col3shared_sweep.sh`
- Seeds: 42, 99, and 7.
- `num_columns=10`, `num_shared=3`, `active_nonshared=7`.
- `combiner=shell_attention`.
- No backbone bypass.
- No direct per-column shell readout.
- No full-width per-column shell bridge.
- `outer_shell_context=on`.
- `column_shell_teacher_weights=0,0,0,0.002`.
- Shell diagnostics and composer diagnostics enabled.

Verification:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_outer_shell_context_10col3shared_sweep.sh scripts/run_codex_shell_preserving_composer_only_replicate_sweep_10col3shared.sh
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: 29 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py tests/test_pooled_readout_norm.py -q
```

Result: 49 passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -k 'not cifar_data' -q
```

Result: 151 passed, 22 deselected.

```bash
git diff --check
```

Result: passed.

Next experiment command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_outer_shell_context_10col3shared_sweep.sh
```

Primary readouts to inspect after the run:

- Combined test accuracy.
- `outer_shell_context_only`, the classifier accuracy when only the new context path reaches `output`.
- `column_pool_plus_outer_shell_context`, the accuracy when the original composer path and new context path are kept together.
- `outer_shell_context_outer_shell_only`, the context-path accuracy when each context node receives only its own outer-shell pool.
- `outer_shell_context_without_outer_shell`, the context-path accuracy when outer-shell inputs are removed from the context nodes.
- The regular `column_outer_shell_only` shell readout, to see whether the raw composed outer-shell slice becomes class-readable.

## 2026-07-03 Outer-Shell Context Seed-42 Result

Timestamp and machine: 2026-07-03 07:50:18 EDT on `rogdora43`.

Master log:

- `results/codex_outer_shell_context_10col3shared_sweep_rogdora43_20260703_030239.log`

Run status:

The sweep produced a complete seed-42 result, but it did not continue to seed 99 or seed 7. The shared runner attempted to create a child log whose filename exceeded the filesystem filename limit:

```text
tee: ... File name too long
```

The child training output still reached the master log, so the seed-42 result is usable. The failed child `tee` returned a nonzero status after Python finished, so the outer sweep stopped before the next planned cases.

Seed-42 run settings:

- `num_columns=10`, `num_shared=3`, `active_nonshared=7`.
- `combiner=shell_attention`.
- No backbone bypass.
- No direct per-column shell readout.
- No full-width per-column shell bridge.
- `outer_shell_context=on`.
- `column_shell_teacher_weights=0,0,0,0.002`.
- Graph size: 147 nodes and 233 edges.
- Parameter count: 3,079,644.
- JAX reported `[CudaDevice(id=0)]` and backend `gpu`.

Main result:

| Setting | Seed | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| 10 columns, no outer context | 42 | 22.82% | 6 | 21.53% |
| 10 columns, outer context plus outer teacher `0.002` | 42 | 24.54% | 7 | 23.04% |

The outer-shell context mechanism improved seed-42 test accuracy by 1.51 percentage points compared with the previous 10-column shell-preserving composer-only seed-42 run. This is a useful directional result, but it is not yet robust because only one seed completed.

Readout details:

| Readout | Test accuracy |
| --- | ---: |
| Combined | 23.04% |
| `column_only` | 12.75% |
| `outer_shell_context_only` | 11.26% |
| `column_pool_plus_outer_shell_context` | 23.04% |
| `column_outer_shell_only` | 10.00% |

The context path is not independently sufficient. `outer_shell_context_only` is slightly above chance, but most of the useful signal still requires the combined classifier. The main combined accuracy equals `column_pool_plus_outer_shell_context`, which is expected because there is no bypass, no direct per-column shell readout, and no full-width bridge in this run.

Outer-shell context input lesions on the test split:

| Context-path lesion | Test accuracy |
| --- | ---: |
| `outer_shell_context_only` | 11.26% |
| `outer_shell_context_without_hard_kernel` | 10.00% |
| `outer_shell_context_hard_kernel_only` | 13.81% |
| `outer_shell_context_without_inner_shell` | 17.80% |
| `outer_shell_context_inner_shell_only` | 10.00% |
| `outer_shell_context_without_middle_shell` | 14.60% |
| `outer_shell_context_middle_shell_only` | 10.00% |
| `outer_shell_context_without_outer_shell` | 12.12% |
| `outer_shell_context_outer_shell_only` | 10.00% |

This table shows that the context path mostly learned to route hard-kernel signal through an outer-shell-shaped latent. Removing hard-kernel inputs collapses the context-only path to chance. Keeping only outer-shell inputs is still chance. The context mechanism is therefore not yet solving the intended outer-shell problem, even though it modestly improved the combined classifier on seed 42.

Per-column outer-shell teacher heads on the test split:

| Head | Test accuracy |
| --- | ---: |
| `column00_outer_shell_teacher_output` | 16.12% |
| `column01_outer_shell_teacher_output` | 10.00% |
| `column02_outer_shell_teacher_output` | 10.00% |
| `column03_outer_shell_teacher_output` | 13.60% |
| `column04_outer_shell_teacher_output` | 10.00% |
| `column05_outer_shell_teacher_output` | 11.46% |
| `column06_outer_shell_teacher_output` | 10.00% |
| `column07_outer_shell_teacher_output` | 16.23% |
| `column08_outer_shell_teacher_output` | 10.08% |
| `column09_outer_shell_teacher_output` | 10.00% |

Some per-column outer-shell teacher heads are above chance, especially columns 0 and 7. This is better than the raw `column_outer_shell_only` readout, but it is still weak and not consistently distributed across columns.

Composer attention pattern:

- `hard_kernel` attention concentrated on column 0 at 99.63%.
- `inner_shell` attention concentrated on column 0 at 97.76%.
- `middle_shell` attention concentrated on column 2 at 84.19%.
- `outer_shell` attention stayed distributed, with column 0 at 19.73% and column 5 at 19.30%.

Conclusion:

The outer-shell context path is a modest improvement for seed 42, but the mechanism did not make the raw outer shell class-readable. The context path used hard-kernel evidence heavily, which means the outer-shell-shaped latent can act as another classifier route without forcing the actual outer shell to carry class evidence. This is still useful because it shows the additional context path can help the combined classifier, but the next step should separate three questions:

- Whether the context path helps without any outer-shell teacher.
- Whether a stronger outer-shell teacher improves or destabilizes the path.
- Whether the `0.002` outer-shell teacher result generalizes to seeds 99 and 7.

Shared runner fix:

The filename-limit failure came from the shared runner after capacity and context labels were added to the result filename. I changed `scripts/run_codex_cifar10_depth_spanning.sh` to use a compact child-log name. The full configuration remains in the log header. The compact filename records only short identifiers for column capacity, combiner, teacher weights, shell readout, shell bridge, outer context, readout mode, seed, learning rate, epoch count, diagnostic mode, hostname, and timestamp.

Verification after the runner fix:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_outer_shell_context_10col3shared_sweep.sh scripts/run_codex_outer_shell_context_followup_10col3shared_sweep.sh
```

Result: passed.

```bash
git diff --check
```

Result: passed.

Next experiment:

Added `scripts/run_codex_outer_shell_context_followup_10col3shared_sweep.sh`.

Planned cases:

| Case | Seed | `column_shell_teacher_weights` | Purpose |
| --- | ---: | --- | --- |
| `seed42_outer_context_no_outer_teacher` | 42 | `0,0,0,0` | Isolate the context path without any per-column outer-shell class target. |
| `seed42_outer_context_outer_teacher0p006` | 42 | `0,0,0,0.006` | Test whether the seed-42 result improves with a stronger local outer-shell objective. |
| `seed99_outer_context_outer_teacher0p002` | 99 | `0,0,0,0.002` | Check whether the seed-42 gain generalizes. |
| `seed7_outer_context_outer_teacher0p002` | 7 | `0,0,0,0.002` | Check whether the seed-42 gain generalizes. |

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_outer_shell_context_followup_10col3shared_sweep.sh
```

## 2026-07-04 Outer-Shell Context Follow-Up Results

Timestamp and machine: 2026-07-04 09:22:59 EDT on `rogdora43`.

Master log:

- `results/codex_outer_shell_context_followup_10col3shared_rogdora43_20260703_075551.log`

Child logs:

- `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_sr0_br0_oc1_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260703_075551.log`
- `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0p006_sr0_br0_oc1_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260703_113257.log`
- `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0p002_sr0_br0_oc1_nobyp_seed99_lr0p005_ep20_composer_shells_rogdora43_20260703_153436.log`
- `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0p002_sr0_br0_oc1_nobyp_seed7_lr0p005_ep20_composer_shells_rogdora43_20260703_193615.log`

All four planned cases completed. Each child log reported `[CudaDevice(id=0)]` and backend `gpu`. The TensorFlow CUDA warning still appeared, but JAX reported the CUDA backend.

Run settings shared by all cases:

- `num_columns=10`, `num_shared=3`, `active_nonshared=7`.
- `combiner=shell_attention`.
- No backbone bypass.
- No direct per-column shell readout.
- No full-width per-column shell bridge.
- `outer_shell_context=on`.
- Shell diagnostics and composer diagnostics enabled.

Summary:

| Case | Seed | `column_shell_teacher_weights` | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | ---: | --- | ---: | ---: | ---: |
| Outer context, no outer teacher | 42 | `0,0,0,0` | 27.48% | 17 | 26.66% |
| Outer context, stronger outer teacher | 42 | `0,0,0,0.006` | 23.92% | 7 | 22.54% |
| Outer context, current outer teacher | 99 | `0,0,0,0.002` | 26.66% | 19 | 25.55% |
| Outer context, current outer teacher | 7 | `0,0,0,0.002` | 24.50% | 18 | 24.01% |

Comparison to prior 10-column no-context baseline:

| Seed | No-context test | Outer context test | Difference |
| ---: | ---: | ---: | ---: |
| 42, no outer teacher | 21.53% | 26.66% | +5.13 pp |
| 42, outer teacher `0.002` | 21.53% | 23.04% | +1.51 pp |
| 42, outer teacher `0.006` | 21.53% | 22.54% | +1.01 pp |
| 99, outer teacher `0.002` | 20.68% | 25.55% | +4.87 pp |
| 7, outer teacher `0.002` | 25.79% | 24.01% | -1.78 pp |

The three-seed mean for the `0.002` outer-teacher setting is 24.20% using seed 42 from the previous partial run plus seed 99 and seed 7 from this run. The earlier 10-column no-context mean was 22.67%. This is a modest mean improvement, but the no-teacher seed-42 result is stronger than any teacher-attached seed-42 result.

Readout and shell ablations on the test split:

| Case | Combined | `column_only` | `outer_shell_context_only` | `column_pool_plus_outer_shell_context` | `column_outer_shell_only` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 42, no outer teacher | 26.66% | 10.00% | 15.58% | 26.66% | 10.00% |
| Seed 42, outer teacher `0.006` | 22.54% | 11.93% | 10.00% | 22.54% | 10.00% |
| Seed 99, outer teacher `0.002` | 25.55% | 10.00% | 10.74% | 25.55% | 10.00% |
| Seed 7, outer teacher `0.002` | 24.01% | 10.00% | 10.00% | 24.01% | 10.00% |

The raw composed outer-shell slice remains at chance in all cases. The no-teacher seed-42 run is the only run where `outer_shell_context_only` is clearly above chance.

Outer-shell context input lesions on the test split:

| Case | Context only | Without hard kernel | Hard kernel only | Without outer shell | Outer shell only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 42, no outer teacher | 15.58% | 11.80% | 10.00% | 11.51% | 10.00% |
| Seed 42, outer teacher `0.006` | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |
| Seed 99, outer teacher `0.002` | 10.74% | 11.86% | 10.00% | 10.00% | 10.00% |
| Seed 7, outer teacher `0.002` | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |

For seed 42 without an outer teacher, the context path uses multiple shell inputs together. Removing hard-kernel inputs or removing outer-shell inputs both reduces the context-only readout, but neither hard-kernel-only nor outer-shell-only is sufficient. With outer-shell teachers enabled, the context path is near chance in isolation.

Per-column outer-shell teacher heads:

- Seed 42 with outer teacher `0.006`: several teacher heads were above chance, including `column00_outer_shell_teacher_output=17.45%`, `column03_outer_shell_teacher_output=15.11%`, `column08_outer_shell_teacher_output=13.16%`, and `column09_outer_shell_teacher_output=16.22%` on test.
- Seed 99 with outer teacher `0.002`: all per-column teacher heads were above chance on test, ranging from 16.45% to 22.39%.
- Seed 7 with outer teacher `0.002`: five teacher heads were above chance on test, with `column05_outer_shell_teacher_output=20.30%` the strongest.

The per-column outer-shell teacher heads can learn class signal, especially for seed 99, but that class signal does not make `column_outer_shell_only` or `outer_shell_context_only` reliably useful. This means the local teacher objective is not aligned with the main classifier route.

Composer attention pattern:

- Seed 42 without an outer teacher selected different columns per shell: `hard_kernel` and `middle_shell` concentrated on column 7, `inner_shell` on column 1, and `outer_shell` on column 3.
- Seed 42 with outer teacher `0.006` concentrated `hard_kernel` on column 5 and concentrated `inner_shell`, `middle_shell`, and `outer_shell` mostly on column 0.
- Seed 99 with outer teacher `0.002` concentrated `hard_kernel` on column 2 and concentrated `inner_shell`, `middle_shell`, and `outer_shell` on column 8.
- Seed 7 with outer teacher `0.002` concentrated `hard_kernel` on column 3, `inner_shell` on column 8, `middle_shell` on column 1, and `outer_shell` on column 6.

Conclusion:

The outer-shell context path is useful, but the outer-shell teacher is not. The best result in this group is seed 42 with `outer_shell_context=on` and no per-column outer-shell teacher. The teacher-attached runs show that outer-shell teacher heads can learn local class labels, but this does not translate into the main predictive-coding readout. The current teacher objective is probably creating local classifiers that are not coordinated with the shell-composer and context routes.

The strongest immediate hypothesis is that the context path should be tested as an architectural mechanism without local class teacher pressure. The next question is whether the no-teacher context result generalizes to seeds 99 and 7. I am not making that change or preparing another run here, per the request to wait for further implementation instructions.

## 2026-07-04 Shell Learning-Rate Plasticity Implementation

Timestamp and machine: 2026-07-04 09:43:20 EDT on `rogdora43`.

Goal:

Test a HiBaCaML-aligned shell plasticity gradient without adding more class-teacher energy. The hard kernel should remain the most stable shell, while the inner shell, middle shell, and outer shell receive progressively larger learned-parameter updates. This changes learned shell plasticity. It does not change predictive-coding inference step size.

Mechanism:

- Added `--shell_lr_multipliers` to `scripts/train_cifar10_depth_spanning.py`.
- The values are ordered as `hard_kernel,inner_shell,middle_shell,outer_shell`.
- The default is `1,1,1,1`, which preserves the previous optimizer behavior.
- The proposed first setting is `1,1.5,2,3`.
- The multiplier is applied after AdamW computes its scheduled update. With base learning rate `0.005`, the effective update multipliers are `0.005`, `0.0075`, `0.010`, and `0.015` for the hard kernel, inner shell, middle shell, and outer shell.

Implemented parameter coverage:

- `DepthSpanningColumnNode` output-sliced matrices and biases: `K_W_out`, `K_b_out`, `L_W_out`, `L_b_out`, `B_W_out`, and `B_b_out` use the shell slice on the final output axis.
- `DepthSpanningColumnNode` learnable shell path weights: `shell_path_scale` uses the explicit shell row.
- `DepthSpanningColumnNode` shell evidence cascade matrices and biases use the target shell. For example, `shell_evidence_cascade_middle_shell_to_outer_shell` uses the outer-shell multiplier.
- `DepthSpanningColumnNode` shell evidence cascade gains use the target shells `inner_shell`, `middle_shell`, and `outer_shell`.
- Learnable shell layer-normalization vectors, when present, use the shell slice on the final feature axis.
- `ColumnShellComposerNode` shell projection matrices and biases use the shell encoded in the parameter name, such as `W_col00_outer_shell`.
- `ColumnShellComposerNode` component attention logits use the shell axis, so outer-shell route selection is more plastic than hard-kernel route selection.
- Per-column `outer_shell_context` nodes use the outer-shell multiplier for all their weights and biases because their output latent is an outer-shell-width context vector.
- Per-column full-width shell bridge nodes, when enabled in another experiment, use the shell slice on their output feature axis.

Parameters intentionally left at the base update rate:

- ResNet backbone parameters.
- Stage-tap projection parameters.
- Hidden shared K, L, and B pathway parameters before the shell-specific output projection, because those tensors are not assigned to one shell.
- Main and auxiliary classifier heads, because the experiment is about shell representation plasticity rather than readout learning-rate changes.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`: added shell learning-rate multiplier parsing, multiplier-tree construction, Optax update scaling, logging, and CLI support.
- `scripts/run_codex_cifar10_depth_spanning.sh`: added positional argument 16 for shell learning-rate multipliers, plus log header and result filename recording.
- `scripts/run_codex_shell_lr_outer_context_10col3shared_sweep.sh`: added a three-seed sweep for the current preferred outer-shell context architecture with no teacher heads.
- `tests/test_pooled_readout_norm.py`: added tests for parsing, parameter multiplier construction, composer/context scaling, and update-tree scaling.

Planned experiment:

Run the current strongest architecture with shell plasticity and no teacher heads:

- `num_columns=10`, `num_shared=3`, `active_nonshared=7`.
- `combiner=shell_attention`.
- No backbone bypass.
- `outer_shell_context=on`.
- `column_teacher_weight=0.0`.
- `shell_teacher_weights=0,0,0,0`.
- `column_shell_teacher_weights=0,0,0,0`.
- `shell_lr_multipliers=1,1.5,2,3`.
- Seeds 42, 99, and 7.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_lr_outer_context_10col3shared_sweep.sh
```

Verification completed:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_shell_lr_outer_context_10col3shared_sweep.sh
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py
```

Result: passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result: passed, 33 tests.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_depth_spanning_column.py -q
```

Result: passed, 20 tests.

```bash
git diff --check
```

Result: passed.

Expected comparison:

The most direct comparison is against the outer-shell context/no-teacher seed-42 result from the previous section, which reached 26.66% test accuracy. The broader question is whether shell plasticity improves or stabilizes the no-teacher outer-context architecture across seeds 99 and 7.

## 2026-07-06 Shell Learning-Rate Plasticity Results and Flat-Control Plan

Timestamp and machine: 2026-07-06 06:03:36 EDT on `rogdora43`.

Completed graded shell learning-rate run:

- Master log: `results/codex_shell_lr_outer_context_10col3shared_rogdora43_20260705_154126.log`.
- Git commit: `6c6244ef303cd310e8dbe0287a0c2f296230715d`.
- `shell_lr_multipliers=1,1.5,2,3`, where each value multiplies AdamW's parameter update for `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` respectively.
- `outer_shell_context=on`, no backbone bypass, no shell teacher heads, and no column teacher energy.
- JAX reported `[CudaDevice(id=0)]` and backend `gpu` in the child logs.

Completed graded shell learning-rate results:

| Seed | Best validation accuracy | Best validation epoch | Test accuracy |
| ---: | ---: | ---: | ---: |
| 42 | 32.82% | 20 | 32.41% |
| 99 | 30.08% | 18 | 29.79% |
| 7 | 31.02% | 18 | 31.00% |

Mean test accuracy was 31.07%. The test range was 29.79% to 32.41%, with sample standard deviation 1.31 percentage points.

Comparison to the prior 10-column, 3-shared, no-context/no-teacher baseline:

| Seed | Prior no-context test accuracy | Graded shell learning-rate test accuracy | Difference |
| ---: | ---: | ---: | ---: |
| 42 | 21.53% | 32.41% | +10.88 percentage points |
| 99 | 20.68% | 29.79% | +9.11 percentage points |
| 7 | 25.79% | 31.00% | +5.21 percentage points |

Mean improvement over the no-context/no-teacher baseline was 8.40 percentage points.

Mechanistic observations:

- The shell composer became sharply shell-specialized. In each seed, each shell routed through a dominant column with attention near 0.99.
- `column_pool` alone remained weak. Test `column_only` accuracy was 13.20% for seed 42, 10.00% for seed 99, and 12.92% for seed 7.
- `outer_shell_context` alone was also weak. Test `outer_shell_context_only` accuracy was 16.51% for seed 42, 12.61% for seed 99, and 10.00% for seed 7.
- The combined classifier required both the pooled column route and the outer-shell context route. The `column_pool_plus_outer_shell_context` ablation matched the combined test accuracy in all three seeds.
- The hard-kernel and outer-shell feature paths were load-bearing under shell lesions. Dropping the hard-kernel slice or outer-shell slice reduced combined accuracy sharply.

Interrupted earlier attempt:

- Master log: `results/codex_shell_lr_outer_context_10col3shared_rogdora43_20260704_094757.log`.
- Git commit: `049aa5d574fb3c859fa8dc5742bff76522d53db0`.
- Seed 42 completed with 27.30% test accuracy.
- Seed 99 completed with 23.09% test accuracy.
- Seed 7 finished training and reached 29.02% validation accuracy, but the run was interrupted before test evaluation completed.
- Because this attempt used an earlier dirty worktree and did not complete, it is secondary evidence only.

Next experiment:

Run the missing flat shell learning-rate control. This keeps the same outer-shell context architecture and sets `shell_lr_multipliers=1,1,1,1`. The control isolates whether the 31.07% three-seed mean comes from the graded shell plasticity itself or from the outer-shell context architecture with no teacher heads.

Added script:

- `scripts/run_codex_shell_lr_flat_outer_context_10col3shared_sweep.sh`.

Planned flat-control cases:

| Case | Seed | `shell_lr_multipliers` |
| --- | ---: | --- |
| Flat shell learning-rate outer context | 42 | `1,1,1,1` |
| Flat shell learning-rate outer context | 99 | `1,1,1,1` |
| Flat shell learning-rate outer context | 7 | `1,1,1,1` |

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_lr_flat_outer_context_10col3shared_sweep.sh
```

## 2026-07-06 Flat Shell Learning-Rate Control Results

Timestamp and machine: 2026-07-06 17:40:33 EDT on `rogdora43`.

Completed flat shell learning-rate control:

- Master log: `results/codex_shell_lr_flat_outer_context_10col3shared_rogdora43_20260706_061152.log`.
- Git commit recorded in the run logs: `8b3eed64ae19be40b1bf82dd4b289c47c3ec7ba9`.
- `shell_lr_multipliers=1,1,1,1`, where the shell learning-rate multiplier is the scalar applied to the AdamW update for the parameter subset belonging to one shell.
- Shell order was `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`.
- `outer_shell_context=on`, no backbone bypass, no shell teacher heads, and no column teacher energy.
- JAX reported `[CudaDevice(id=0)]` and backend `gpu` in the child logs.

Flat shell learning-rate results:

| Seed | Best validation accuracy | Best validation epoch | Test accuracy |
| ---: | ---: | ---: | ---: |
| 42 | 30.44% | 20 | 30.20% |
| 99 | 26.62% | 8 | 26.17% |
| 7 | 28.64% | 19 | 27.82% |

Mean test accuracy was 28.06%. The test range was 26.17% to 30.20%, with sample standard deviation 2.03 percentage points. The sample standard deviation is the across-seed standard deviation computed from the three test accuracies.

Comparison against the graded shell learning-rate run:

| Seed | Flat shell learning-rate test accuracy | Graded shell learning-rate test accuracy | Difference |
| ---: | ---: | ---: | ---: |
| 42 | 30.20% | 32.41% | +2.21 percentage points |
| 99 | 26.17% | 29.79% | +3.62 percentage points |
| 7 | 27.82% | 31.00% | +3.18 percentage points |
| Mean | 28.06% | 31.07% | +3.00 percentage points |

The graded shell learning-rate run used `shell_lr_multipliers=1,1.5,2,3`, so the hard-kernel parameters received the base AdamW update, the inner-shell parameters received 1.5 times that update, the middle-shell parameters received 2 times that update, and the outer-shell parameters received 3 times that update. This improved every seed relative to the flat `1,1,1,1` control.

Comparison against the prior 10-column, 3-shared, no-context/no-teacher baseline:

| Seed | Prior no-context test accuracy | Flat shell learning-rate outer-context test accuracy | Difference |
| ---: | ---: | ---: | ---: |
| 42 | 21.53% | 30.20% | +8.67 percentage points |
| 99 | 20.68% | 26.17% | +5.49 percentage points |
| 7 | 25.79% | 27.82% | +2.03 percentage points |
| Mean | 22.67% | 28.06% | +5.40 percentage points |

Interpretation:

- The outer-shell context architecture improved the no-context/no-teacher baseline even when all four shells used the same parameter-update scale.
- The outward-increasing shell learning-rate schedule added a further mean gain of 3.00 percentage points over the flat control and reduced across-seed variability from 2.03 percentage points to 1.31 percentage points.
- The flat run did not show catastrophic collapse, but seed 99 selected the best validation checkpoint at epoch 8 and finished at only 26.17% test accuracy. That is weaker and less stable than the graded run, where seed 99 selected epoch 18 and reached 29.79% test accuracy.
- The flat seed-99 composer attention left the middle-shell and outer-shell routes diffuse. The largest middle-shell attention values were 0.460526 on column 0 and 0.383664 on column 8, and the largest outer-shell attention value was 0.160646 on column 0. The graded seed-99 run selected hard kernel, inner shell, middle shell, and outer shell sharply, with dominant attentions 0.999368, 0.998577, 0.997867, and 0.987216 respectively.
- The combined classifier continued to require both the pooled column route and the outer-shell context route. In the flat run, `column_pool_plus_outer_shell_context` matched the combined test accuracy for all three seeds, while `column_only` stayed near chance.
- The hard-kernel path remained load-bearing. Removing the hard-kernel slice reduced the flat combined test readout to 11.19% for seed 42, 11.11% for seed 99, and 10.00% for seed 7.

Conclusion:

The flat control supports the shell learning-rate idea. The mechanism is not just that `outer_shell_context` exists. The outward-increasing update schedule makes the outer shells learn more strongly while preserving the hard-kernel route, and that combination produced a consistent three-seed gain.

Recommended next step:

Keep `shell_lr_multipliers=1,1.5,2,3` as the current baseline. The next useful experiment is a profile sweep around that schedule, with no architecture changes yet. The two most informative profiles are:

| Profile name | `shell_lr_multipliers` | Reason |
| --- | --- | --- |
| Gentler outward schedule | `1,1.25,1.75,2.5` | Tests whether the current gain comes from any outward bias while reducing risk of outer-shell overfitting. |
| Stronger outward schedule | `1,2,3,4` | Tests whether the current schedule is still under-training the outer shells. |

The preferred next run is a 3-seed sweep over both profiles, using seeds 42, 99, and 7. This would be six total runs with the same 10-column, 3-shared, outer-context, no-teacher configuration. It should take roughly twice as long as the flat-control sweep.

## 2026-07-06 Shell Learning-Rate Profile Sweep Prepared

Timestamp and machine: 2026-07-06 18:07:48 EDT on `rogdora43`.

Implemented the next sweep wrapper:

- Script: `scripts/run_codex_shell_lr_profile_outer_context_10col3shared_sweep.sh`.
- Purpose: test whether the current outward-increasing shell learning-rate baseline is too weak, too strong, or close to the useful range.
- Baseline retained for comparison: `shell_lr_multipliers=1,1.5,2,3`.
- Flat control retained for comparison: `shell_lr_multipliers=1,1,1,1`.

The shell learning-rate multiplier is the scalar applied to the AdamW parameter update for one shell-local parameter group. The shell order is `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`.

Planned sweep:

| Profile | `shell_lr_multipliers` | Seeds | Mechanistic question |
| --- | --- | --- | --- |
| Gentler outward schedule | `1,1.25,1.75,2.5` | 42, 99, 7 | Tests whether a smaller outward plasticity gradient preserves the benefit while reducing the risk that outer-shell parameters over-specialize. |
| Stronger outward schedule | `1,2,3,4` | 42, 99, 7 | Tests whether the current baseline still under-trains middle-shell and outer-shell parameters. |

Shared configuration for all six runs:

- `num_columns=10`.
- `num_shared=3`.
- `active_nonshared=7`.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- `column_teacher_weight=0.0`.
- `shell_teacher_weights=0,0,0,0`.
- `column_shell_teacher_weights=0,0,0,0`.
- No backbone bypass.
- 20 epochs.
- Learning rate 0.005.
- Diagnostic mode `composer_shells`.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_lr_profile_outer_context_10col3shared_sweep.sh
```

Verification completed:

```bash
bash -n scripts/run_codex_shell_lr_profile_outer_context_10col3shared_sweep.sh
```

Result: passed.

## 2026-07-08 Outer-Shell Context Teacher Results and Next Proposal

Timestamp and machine: 2026-07-08 19:51:47 EDT on `rogdora43`.

Completed outer-shell context teacher sweep:

- Master log: `results/codex_outer_context_teacher_10col3shared_rogdora43_20260707_212806.log`.
- The run completed at 2026-07-08 19:29:52 EDT.
- Shared architecture: 10 columns, 3 shared columns, 7 active non-shared columns, `combiner=shell_attention`, `outer_shell_context=on`, no backbone bypass, no column teacher energy, no shell teacher energy, no per-column shell teacher energy, `shell_lr_multipliers=1,1.5,2,3`, 20 epochs, and learning rate 0.005.
- `w_ctx` means `outer_shell_context_teacher_weight`, the scalar multiplier on the auxiliary cross-entropy energy at `outer_shell_context_teacher_output`.
- `outer_shell_context_teacher_output` is the context-only teacher head fed by all active `columnXX_outer_shell_context` latent nodes.

Results:

| `w_ctx` | Seed | Best validation accuracy | Best validation epoch | Test accuracy | `outer_shell_context_teacher_output` test | `outer_shell_context_only` test | `column_only` test |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0005 | 42 | 28.14% | 20 | 27.47% | 26.76% | 13.46% | 10.00% |
| 0.0005 | 99 | 31.36% | 18 | 30.30% | 29.91% | 10.00% | 10.00% |
| 0.0005 | 7 | 24.28% | 3 | 23.75% | 22.53% | 19.08% | 16.04% |
| 0.0010 | 42 | 30.44% | 17 | 29.89% | 30.34% | 14.86% | 10.00% |
| 0.0010 | 99 | 28.66% | 18 | 27.49% | 26.63% | 10.00% | 16.53% |
| 0.0010 | 7 | 25.64% | 19 | 25.57% | 26.01% | 10.00% | 10.00% |

Aggregate comparison:

| Setting | Mean test accuracy | Across-seed sample standard deviation |
| --- | ---: | ---: |
| No context teacher, `w_ctx = 0.0` | 31.07% | 1.31 percentage points |
| Context teacher, `w_ctx = 0.0005` | 27.17% | 3.29 percentage points |
| Context teacher, `w_ctx = 0.0010` | 27.65% | 2.16 percentage points |

Per-seed comparison against the no-context-teacher baseline:

| Seed | No context teacher test accuracy | `w_ctx = 0.0005` test accuracy | `w_ctx = 0.0010` test accuracy |
| ---: | ---: | ---: | ---: |
| 42 | 32.41% | 27.47% | 29.89% |
| 99 | 29.79% | 30.30% | 27.49% |
| 7 | 31.00% | 23.75% | 25.57% |

Interpretation:

- The context teacher learned class-readable context latents. The teacher head reached 22.53% to 30.34% test accuracy across the six runs.
- The learned teacher signal did not transfer to the main `output` classifier. The main combined test accuracy fell below the no-context-teacher baseline in five of six runs.
- The `outer_shell_context_only` main-output ablation stayed weak. It reached 19.08% only for seed 7 at `w_ctx = 0.0005`, while the same run's combined classifier fell to 23.75%.
- The seed-7 `w_ctx = 0.0005` run selected epoch 3 as the best checkpoint. This is an early instability signal, not a robust improvement.
- The result is not evidence that the context latents lack class information. The separate teacher head could decode class information from those latents. The problem is alignment: the teacher head's classifier weights are separate from the main `output` classifier's context-edge weights.

Conclusion:

Do not increase `w_ctx` and do not continue scalar sweeps on the separate context teacher head. The mechanism taught the auxiliary head to decode the context latents, but it did not make the main predictive-coding output use those latents better.

Recommended next step:

Convert the context teacher from a separate diagnostic classifier into an aligned context-evidence pathway.

Mechanism to consider:

- Add an `outer_shell_context_evidence` latent with shape `(10,)`, where each of the 10 dimensions corresponds to one CIFAR-10 class logit.
- Feed all active `columnXX_outer_shell_context` latents into `outer_shell_context_evidence`.
- Feed `outer_shell_context_evidence` into the main `output` classifier.
- Optionally attach a small cross-entropy energy to `outer_shell_context_evidence` or to a teacher node fed by it.

This differs from the completed context-teacher experiment. In the completed experiment, the class-readable teacher head was separate from the main output route. In the proposed mechanism, the class-shaped context evidence is part of the route that the main classifier receives.

Alternative approaches considered:

| Approach | Advantage | Reason not recommended as the next step |
| --- | --- | --- |
| Increase `w_ctx` above 0.001 | Simple scalar sweep. | Both tested context-teacher weights reduced mean accuracy, and the auxiliary head already learned class signal. A larger weight is likely to increase target competition. |
| Lower `w_ctx` below 0.0005 | Tests whether a smaller auxiliary energy avoids damage. | The current issue is route alignment, not just energy scale. The teacher head learned without helping the main route. |
| Return to shell learning-rate sweeps | No new mechanism. | The best tested profile remains `1,1.5,2,3`; neighboring profiles were worse. |
| Re-enable per-column shell teacher heads | Local to HiBaCaML shell structure. | Previous runs showed local shell teacher heads can learn labels without improving the main route. |

Proposed first experiment after implementation:

- Use `shell_lr_multipliers=1,1.5,2,3`.
- Keep `outer_shell_context=on`.
- Add the aligned `outer_shell_context_evidence` path.
- Test no auxiliary context evidence cross-entropy energy and a very small auxiliary cross-entropy energy, such as `0.0005`.
- Run seeds 42, 99, and 7.

The evaluation criterion should prioritize the main combined test accuracy and collapse resistance. Secondary metrics are `outer_shell_context_evidence` test accuracy, `outer_shell_context_only` test accuracy, `column_only` test accuracy, and shell lesion effects.

## 2026-07-08 Outer-Shell Context Evidence Implementation

Timestamp and machine: 2026-07-08 20:08:41 EDT on `rogdora43`.

Implemented the aligned outer-shell context evidence path.

Mechanism:

- `outer_shell_context_evidence` is a Gaussian predictive-coding latent with shape `(10,)`, where the 10 dimensions are class-width evidence coordinates for CIFAR-10.
- Each active `columnXX_outer_shell_context` latent feeds `outer_shell_context_evidence`.
- `outer_shell_context_evidence` feeds the main `output` classifier, so the class-shaped context signal is available to the route that determines the primary test accuracy.
- `outer_shell_context_evidence_teacher_output` is an optional auxiliary CE classifier fed only by `outer_shell_context_evidence`.
- `outer_shell_context_evidence_teacher_weight` is the scalar multiplier on that optional auxiliary CE energy. Weight `0.0` omits the teacher head while keeping the evidence path active when `--outer_shell_context_evidence` is set.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`.
- `scripts/run_codex_cifar10_depth_spanning.sh`.
- `scripts/run_codex_outer_context_evidence_10col3shared_sweep.sh`.
- `tests/test_pooled_readout_norm.py`.

Graph wiring details:

- `--outer_shell_context_evidence` requires `--outer_shell_context`.
- `--outer_shell_context_evidence_teacher_weight > 0` requires `--outer_shell_context_evidence`.
- The previous raw `columnXX_outer_shell_context -> output` edges remain in place when `--outer_shell_context` is enabled.
- The new path adds `columnXX_outer_shell_context -> outer_shell_context_evidence -> output`.
- If `--outer_shell_context_evidence_teacher_weight` is positive, the graph also adds `outer_shell_context_evidence -> outer_shell_context_evidence_teacher_output`.

Evaluation additions:

- Readout ablations now include `outer_shell_context_evidence_only`, `column_pool_plus_outer_shell_context_evidence`, and `outer_shell_context_plus_evidence`.
- Validation and test reporting now evaluate `outer_shell_context_evidence` directly as a class-width diagnostic.
- Validation and test reporting evaluate `outer_shell_context_evidence_teacher_output` when the optional teacher exists.
- Shell-lesion diagnostics now include `outer_shell_context_evidence_without_{shell}` and `outer_shell_context_evidence_{shell}_only`, where `{shell}` is one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`.

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_outer_context_evidence_10col3shared_sweep.sh
git diff --check
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Results:

- Python compile check: passed.
- Shell syntax check: passed.
- `git diff --check`: passed.
- `tests/test_pooled_readout_norm.py`: 39 passed.
- Full `pytest` run: 169 passed, 5 failed, and 9 errored. All failures/errors were in `tests/test_cifar_data.py` because the sandbox could not create `/home/ni/.local/share/columnar_cl_fabricpc`. The failure was `OSError: [Errno 30] Read-only file system`.

Prepared experiment:

- Script: `scripts/run_codex_outer_context_evidence_10col3shared_sweep.sh`.
- Architecture: 10 columns, 3 shared columns, 7 active non-shared columns, `combiner=shell_attention`, `outer_shell_context=on`, `outer_shell_context_evidence=on`, no backbone bypass, no raw context teacher, no shell teacher heads, no per-column shell teacher heads.
- Shell learning-rate multipliers: `1,1.5,2,3`.
- Learning rate: 0.005.
- Epochs: 20.
- Seeds: 42, 99, and 7.
- Evidence teacher weights: `0.0` and `0.0005`.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_outer_context_evidence_10col3shared_sweep.sh
```

Expected comparison:

- Compare `outer_shell_context_evidence_teacher_weight=0.0` against the current no-evidence baseline mean test accuracy of 31.07%.
- Compare `outer_shell_context_evidence_teacher_weight=0.0005` against both the no-evidence baseline and the no-teacher evidence path.
- Prioritize main `combined` test accuracy and collapse resistance.
- Use `outer_shell_context_evidence_only`, `column_pool_plus_outer_shell_context_evidence`, and evidence shell lesions to determine whether the class-shaped evidence route is being used by the main classifier.

## 2026-07-09 Outer-Context Evidence Result and Shell-Local Correction

Timestamp and machine: 2026-07-09 17:34:51 EDT on `rogdora43`.

The outer-context evidence sweep gave enough evidence to reject the free class-shaped evidence route.

Completed configuration:

- 10 columns, 3 shared columns, 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- `outer_shell_context_evidence=on`.
- `outer_shell_context_evidence_teacher_weight=0.0`.
- No backbone bypass, no column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `shell_lr_multipliers=1,1.5,2,3`.
- Learning rate 0.005 for 20 epochs.

Completed results:

| Seed | Best validation accuracy | Best validation epoch | Test accuracy | `outer_shell_context_evidence` direct test | `outer_shell_context_evidence_only` test |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 26.72% | 17 | 26.39% | 10.00% | 10.00% |
| 99 | 28.84% | 18 | 27.77% | 10.00% | 10.00% |
| 7 | 26.24% | 10 | 25.18% | 10.00% | 10.00% |

Aggregate result:

- Mean test accuracy was 26.45%.
- Across-seed sample standard deviation was 1.30 percentage points.
- The prior 10-column, 3-shared, no-evidence outer-context baseline mean was 31.07%.
- The per-seed accuracy changes versus that baseline were -6.02 percentage points for seed 42, -2.02 percentage points for seed 99, and -5.82 percentage points for seed 7.

The `outer_shell_context_evidence_teacher_weight=0.0005` seed-42 run was killed before final evaluation. I am not treating it as a result.

Mechanistic conclusion:

- The free `outer_shell_context_evidence -> output` route did not become class-readable. `outer_shell_context_evidence` direct accuracy and `outer_shell_context_evidence_only` readout accuracy stayed at chance.
- The combined classifier learned weakly through the larger graph, but the evidence latent did not become an independently useful predictive-coding structure.
- Completing another long teacher-weight run would mostly test whether an auxiliary classifier can decode labels from this latent. It would not address the mechanism that failed in the completed runs.

Correction implemented:

- Added `ShellContextPredictionNode` in `columnar_cl_fabricpc/columns/accuracy_nodes.py`.
- Exported `ShellContextPredictionNode` from `columnar_cl_fabricpc/columns/__init__.py`.
- Added `--outer_shell_context_shell_prediction_weight` to `scripts/train_cifar10_depth_spanning.py`.
- Added positional argument 20 to `scripts/run_codex_cifar10_depth_spanning.sh` for `outer_shell_context_shell_prediction_weight`.
- Added `scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh`.
- Updated `tests/test_pooled_readout_norm.py`.

New mechanism:

- `ShellContextPredictionNode` is a terminal local predictive objective, not a classifier head.
- Its `target` slot receives one pooled shell state, such as `column00_hard_kernel_pool`.
- Its `context` slot receives the same column's outer-context latent, such as `column00_outer_shell_context`.
- It computes a learned linear prediction from the context latent to the target shell width.
- Its energy is `0.5 * w_shell_ctx * sum((target_shell - predicted_shell)^2)`, where `w_shell_ctx` is `outer_shell_context_shell_prediction_weight`, `target_shell` is the pooled shell vector, and `predicted_shell` is the context-derived prediction of that pooled shell vector.
- A positive `outer_shell_context_shell_prediction_weight` disables the direct `columnXX_outer_shell_context -> output` route. In that mode, outer context predicts shell states rather than feeding class logits.
- `outer_shell_context_evidence` no longer feeds `output`. It remains available only as a diagnostic or optional teacher target.
- `ShellContextPredictionNode.forward_and_latent_grads` overrides FabricPC's terminal-node shortcut so the local objective contributes gradients even though the node has no downstream child.
- Shell learning-rate multipliers apply to shell-context predictor parameters according to the target shell. A hard-kernel predictor uses the hard-kernel multiplier, and an outer-shell predictor uses the outer-shell multiplier.

Diagnostics added:

- `diagnose_shell_context_prediction_energies` reports mean local prediction energy by target shell.
- When `--diagnose_shells` is enabled, the script prints shell-context prediction energy before training, after training, and for the selected checkpoint.
- Existing shell-norm and shell-readout ablations still report collapse and class-readout dependence.

Prepared experiment:

- Script: `scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh`.
- Default seeds: `42`.
- Default shell-context weights: `0.00025 0.0005 0.001`.
- Default learning rate: 0.005.
- Default epochs: 20.
- Default diagnostics: `composer_shells`.
- Architecture: 10 columns, 3 shared columns, 7 active non-shared columns, `combiner=shell_attention`, `outer_shell_context=on`, `outer_shell_context_evidence=off`, no backbone bypass, no teacher heads, and `shell_lr_multipliers=1,1.5,2,3`.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh
```

Environment overrides:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && SEEDS="42 99 7" WEIGHTS="0.0005" bash scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh
```

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/accuracy_nodes.py tests/test_pooled_readout_norm.py
bash -n scripts/run_codex_cifar10_depth_spanning.sh
bash -n scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh
git diff --check
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Verification results:

- Python compile check: passed.
- Shell syntax checks: passed.
- `git diff --check`: passed.
- `tests/test_pooled_readout_norm.py`: 42 passed.

## 2026-07-10 Shell-Local Context Prediction Sweep

Timestamp and machine: 2026-07-10 11:58:56 EDT on `rogdora43`.

Completed run:

- Master log: `results/codex_shell_context_prediction_10col3shared_rogdora43_20260709_173846.log`.
- Child logs:
  - `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p00025_ocet0p0_sr0_br0_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260709_173846.log`.
  - `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0005_ocet0p0_sr0_br0_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260709_205752.log`.
  - `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p001_ocet0p0_sr0_br0_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260710_001612.log`.

Shared configuration:

- Seed 42.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- `outer_shell_context_evidence=off`.
- No backbone bypass, no column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `shell_lr_multipliers=1,1.5,2,3`.
- Learning rate 0.005 for 20 epochs.
- `w_shell_ctx` means `outer_shell_context_shell_prediction_weight`, the scalar multiplier on the local Gaussian objective where each column's outer-context latent predicts that column's pooled shell states.

Results:

| `w_shell_ctx` | Best validation accuracy | Best validation epoch | Test accuracy | Test `column_only` | Test `column_teacher_output` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00025 | 21.60% | 15 | 20.05% | 20.05% | 16.01% |
| 0.00050 | 26.46% | 20 | 26.02% | 26.02% | 12.47% |
| 0.00100 | 26.08% | 20 | 24.66% | 24.66% | 15.88% |

Direct seed-42 comparison:

| Condition | Test accuracy | Test `column_only` | Test `outer_shell_context_only` | Output sources |
| --- | ---: | ---: | ---: | --- |
| Prior outer-context baseline, `w_shell_ctx = 0` | 32.41% | 13.20% | 16.51% | `column_pool` plus `columnXX_outer_shell_context` |
| Shell prediction, `w_shell_ctx = 0.00025` | 20.05% | 20.05% | not present | `column_pool` only |
| Shell prediction, `w_shell_ctx = 0.00050` | 26.02% | 26.02% | not present | `column_pool` only |
| Shell prediction, `w_shell_ctx = 0.00100` | 24.66% | 24.66% | not present | `column_pool` only |

Shell-readout diagnostics:

| `w_shell_ctx` | Test without hard kernel | Test hard kernel only | Test without inner shell | Test without middle shell | Test without outer shell | Test outer shell only |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00025 | 10.14% | 18.10% | 19.76% | 19.33% | 20.93% | 9.43% |
| 0.00050 | 15.03% | 22.94% | 25.00% | 21.31% | 25.74% | 8.35% |
| 0.00100 | 11.59% | 20.63% | 22.31% | 21.12% | 21.89% | 10.00% |

Shell norms and local prediction energy:

- Shell L2 norms remained close to the pre-training magnitudes. This run did not show an obvious shell-norm collapse.
- The logged weighted shell-context prediction energies printed as `0.000000` after training and at the selected checkpoint for all shells. The output precision is six decimals, so this only shows that the weighted energy was below the displayed resolution.
- The local objective changed the representation enough to improve the `column_pool` route versus the prior seed-42 `column_only` value of 13.20%, but it did not replace the missing context readout.

Mechanistic interpretation:

- In the prior seed-42 outer-context baseline, the main `output` classifier received both `column_pool` and all `columnXX_outer_shell_context` latents. The combined test accuracy was 32.41%, while `column_only` was 13.20%. The context path was therefore load-bearing for the combined classifier.
- In the shell-prediction sweep, positive `w_shell_ctx` disabled the direct `columnXX_outer_shell_context -> output` edges. The only classifier input source was `column_pool`, so `combined` and `column_only` were identical in every run.
- The local shell-prediction objective improved the class readability of `column_pool` from 13.20% to as high as 26.02% on seed 42. That is useful evidence that the local objective affects the column representation.
- The same local objective reduced total combined accuracy because it removed the direct context evidence used by the best seed-42 baseline. The largest completed shell-prediction result, 26.02%, remained 6.39 percentage points below the 32.41% direct-context seed-42 baseline.
- The hard-kernel slice remained load-bearing. Removing the hard-kernel slice reduced test accuracy to 10.14%, 15.03%, and 11.59% across the three shell-prediction weights.

Conclusion:

The shell-local context-prediction objective should not replace the outer-context readout. The result supports using it as an added local predictive objective while retaining a direct context contribution to the main classifier.

Recommended next step:

Modify the graph so `outer_shell_context_shell_prediction_weight > 0` does not automatically disable `columnXX_outer_shell_context -> output`. Then run the seed-42 comparison with:

- `w_shell_ctx = 0.0005`.
- `outer_shell_context=on`.
- Direct outer-context readout still connected to `output`.
- `outer_shell_context_evidence=off`.
- No teacher heads.
- Same 10-column, 3-shared, shell-attention, `shell_lr_multipliers=1,1.5,2,3`, learning-rate 0.005, 20-epoch configuration.

Alternative approaches considered:

| Approach | Advantage | Reason not recommended as the immediate next step |
| --- | --- | --- |
| Replicate `w_shell_ctx = 0.0005` across seeds with direct context disabled | Tests whether the 26.02% seed-42 result is stable. | It is already below the direct-context seed-42 baseline by 6.39 percentage points, and the mechanism has no context readout path at `output`. |
| Increase `w_shell_ctx` above 0.001 with direct context disabled | Tests stronger local shell prediction. | The trend peaked at 0.0005 and fell at 0.001 on seed 42. A stronger weight would still leave `output` without the context path that made the baseline work. |
| Restore only the previous direct-context baseline | Recovers the best tested seed-42 setup. | It does not use the new evidence that shell-local prediction made `column_pool` more class-readable. |
| Keep direct context readout and add shell-local context prediction | Preserves the load-bearing context path while adding local shell structure. | This is the recommended next mechanism to test. |

## 2026-07-10 Retain Context Readout With Shell Prediction

Timestamp and machine: 2026-07-10 14:18:01 EDT on `rogdora43`.

Implemented the recommended correction from the shell-local context prediction sweep.

Scope change:

- Before this change, a positive `outer_shell_context_shell_prediction_weight` replaced the direct `columnXX_outer_shell_context -> output` readout with shell-local prediction objectives.
- After this change, a positive `outer_shell_context_shell_prediction_weight` adds shell-local prediction objectives while retaining direct `columnXX_outer_shell_context -> output` readout.
- This makes shell-local context prediction an additional HiBaCaML-style local objective, not a substitute for the load-bearing context evidence route.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`.
- `tests/test_pooled_readout_norm.py`.
- `scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh`.
- `docs/dev-plans/20260624-work-log-codex-rogdora43-cifar10-hibacaml.md`.

Graph wiring:

- `columnXX_outer_shell_context` still receives that column's pooled `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell` states.
- `columnXX_outer_shell_context` always feeds `output` when `--outer_shell_context` is enabled.
- When `outer_shell_context_shell_prediction_weight > 0`, each `columnXX_outer_shell_context` also feeds four `ShellContextPredictionNode` objectives, one for each target shell.
- Each `ShellContextPredictionNode` receives one target pooled shell state and the same column's outer-context latent. The weighted local energy is `0.5 * w_shell_ctx * sum((target_shell - predicted_shell)^2)`, where `w_shell_ctx` is `outer_shell_context_shell_prediction_weight`.

Script update:

- `scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh` now defaults to `WEIGHTS=0.0005`.
- The master log prefix is now `codex_shell_context_prediction_with_readout_10col3shared`.
- The script header records `outer_shell_context direct readout: retained`.

Next experiment:

- Seed 42.
- `w_shell_ctx = 0.0005`.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- Direct outer-context readout retained.
- `outer_shell_context_evidence=off`.
- No teacher heads.
- `shell_lr_multipliers=1,1.5,2,3`.
- Learning rate 0.005 for 20 epochs.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh
```

Verification:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
bash -n scripts/run_codex_shell_context_prediction_10col3shared_sweep.sh scripts/run_codex_cifar10_depth_spanning.sh
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Verification results:

- Python compile check: passed.
- Shell syntax checks: passed.
- `tests/test_pooled_readout_norm.py`: 42 passed.

## 2026-07-11 Retained Context Readout Shell-Prediction Result

Timestamp and machine: 2026-07-11 06:45:20 EDT on `rogdora43`.

Completed run:

- Master log: `results/codex_shell_context_prediction_with_readout_10col3shared_rogdora43_20260710_141925.log`.
- Child log: `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0005_ocet0p0_sr0_br0_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260710_141925.log`.

Configuration:

- Seed 42.
- 10 columns, 3 shared columns, and 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- Direct `columnXX_outer_shell_context -> output` readout retained.
- `outer_shell_context_evidence=off`.
- No backbone bypass, no column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `w_shell_ctx = 0.0005`, where `w_shell_ctx` is `outer_shell_context_shell_prediction_weight`, the scalar multiplier on the local objective where each column's outer-context latent predicts that column's pooled shell states.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values scale parameter updates for the hard-kernel, inner-shell, middle-shell, and outer-shell slices.
- Learning rate 0.005 for 20 epochs.
- `diagnose_mode=composer_shells`.

Primary result:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 25.88% |
| Best validation epoch | 5 |
| Test accuracy | 25.93% |
| Parameter count | 3,091,524 |

Validation trajectory:

| Epoch | Validation accuracy |
| ---: | ---: |
| 1 | 10.08% |
| 2 | 17.50% |
| 3 | 20.76% |
| 4 | 25.42% |
| 5 | 25.88% |
| 6 | 24.20% |
| 7 | 21.04% |
| 8 | 13.86% |
| 9 | 18.78% |
| 10 | 22.12% |
| 11 | 20.78% |
| 12 | 14.26% |
| 13 | 22.24% |
| 14 | 22.62% |
| 15 | 21.52% |
| 16 | 22.06% |
| 17 | 24.02% |
| 18 | 23.56% |
| 19 | 25.64% |
| 20 | 25.06% |

Readout ablations:

| Test condition | Accuracy |
| --- | ---: |
| Combined `output` readout | 25.93% |
| `column_pool` only | 11.22% |
| Outer-shell context only | 16.33% |
| `column_pool` plus outer-shell context | 25.93% |
| Column teacher head | 11.66% |

Outer-shell context ablations:

| Test condition | Accuracy |
| --- | ---: |
| Outer-shell context only | 16.33% |
| Context without hard-kernel source | 16.24% |
| Context from hard-kernel source only | 16.65% |
| Context without inner-shell source | 16.65% |
| Context from inner-shell source only | 15.38% |
| Context without middle-shell source | 16.61% |
| Context from middle-shell source only | 16.57% |
| Context without outer-shell source | 15.47% |
| Context from outer-shell source only | 10.00% |

Shell readout ablations:

| Test condition | Accuracy |
| --- | ---: |
| Combined without hard kernel | 21.73% |
| Column path without hard kernel | 10.15% |
| Column path hard kernel only | 17.84% |
| Combined without inner shell | 25.80% |
| Column path without inner shell | 11.04% |
| Column path inner shell only | 10.00% |
| Combined without middle shell | 22.68% |
| Column path without middle shell | 18.58% |
| Column path middle shell only | 10.30% |
| Combined without outer shell | 22.19% |
| Column path without outer shell | 12.29% |
| Column path outer shell only | 10.00% |

Shell norms:

| Shell | Width | Mean L2 before training | Mean L2 after training | Mean L2 at selected checkpoint |
| --- | ---: | ---: | ---: | ---: |
| Hard kernel | 22 | 4.6834 | 4.6897 | 4.6897 |
| Inner shell | 7 | 2.6343 | 2.6454 | 2.6454 |
| Middle shell | 14 | 3.7262 | 3.7409 | 3.7409 |
| Outer shell | 21 | 4.5527 | 4.5810 | 4.5810 |

Composer attention at the selected checkpoint:

| Shell | Dominant column | Attention weight |
| --- | --- | ---: |
| Hard kernel | `col_08` | 0.988349 |
| Inner shell | `col_01` | 0.983834 |
| Middle shell | `col_07` | 0.984583 |
| Outer shell | `col_01` | 0.703865 |

Output edge weight norms at the selected checkpoint:

| Edge source | Weight norm |
| --- | ---: |
| `column00_outer_shell_context` | 16.857279 |
| `column01_outer_shell_context` | 16.961300 |
| `column02_outer_shell_context` | 17.985710 |
| `column03_outer_shell_context` | 15.951418 |
| `column04_outer_shell_context` | 15.154963 |
| `column05_outer_shell_context` | 16.288431 |
| `column06_outer_shell_context` | 17.014477 |
| `column07_outer_shell_context` | 17.685543 |
| `column08_outer_shell_context` | 15.571239 |
| `column09_outer_shell_context` | 17.397055 |
| `column_pool` | 14.395780 |

Comparison to the direct-context baseline:

| Condition | Test accuracy | Test `column_pool` only | Test outer-shell context only | Best validation epoch |
| --- | ---: | ---: | ---: | ---: |
| Direct outer-context baseline with no shell-prediction objective | 32.41% | 13.20% | 16.51% | 20 |
| Shell-prediction objective with direct context readout disabled | 26.02% | 26.02% | not present | 20 |
| Shell-prediction objective with direct context readout retained | 25.93% | 11.22% | 16.33% | 5 |

Interpretation:

- Retaining direct outer-context readout did not recover the earlier direct-context baseline. The retained-readout run reached 25.93% test accuracy, which is 6.48 percentage points below the 32.41% direct-context seed-42 baseline.
- Retaining direct outer-context readout also did not improve over the replacement-path shell-prediction run. The retained-readout result was 25.93%, while the replacement-path result was 26.02%.
- The shell magnitudes did not collapse. The hard-kernel, inner-shell, middle-shell, and outer-shell mean L2 norms all stayed close to their initial magnitudes.
- The failure looks functional rather than magnitude-based. Validation accuracy peaked at epoch 5, then varied between 13.86% and 25.64% instead of climbing toward the direct-context baseline.
- The `column_pool` route became weak in the retained-readout run. It was 11.22%, close to random chance, while the replacement-path shell-prediction run made `column_pool` reach 26.02%.
- The outer-shell context route stayed weak but similar to the previous baseline. It was 16.33% here versus 16.51% in the direct-context baseline.
- The hard-kernel shell stayed load-bearing. Removing it dropped the combined readout from 25.93% to 21.73%, and the hard-kernel-only column path reached 17.84%.
- The outer shell still does not carry class-readable evidence by itself. The outer-shell-only column path and outer-shell-only context path were both 10.00%.
- The composer chose sparse shell ownership. One column dominated the hard kernel, one dominated the inner shell, one dominated the middle shell, and one dominated most of the outer shell. This is an architectural specialization signal, but the specialized components did not combine into a robust classifier.

Mechanistic concern:

`ShellContextPredictionNode` currently computes `error = target - prediction`, where `target` is a pooled shell vector from one column and `prediction` is a linear projection of the same column's outer-context latent. Its `forward_and_latent_grads` method differentiates the energy with respect to the entire input dictionary. The target shell vector and the context vector therefore both receive gradients from the local prediction energy.

That gradient path makes the objective a bidirectional self-consistency pressure. It trains the context pathway to predict shell states, and it also pulls the shell states toward what the current context pathway can reconstruct. Because the outer-context latent is built from the same pooled shell states that it predicts, the current mechanism can reward low-dimensional reconstructability without necessarily preserving class-discriminative shell evidence.

Recommended next step:

Implement a scoped semantic correction to `ShellContextPredictionNode` before another large sweep. The node should apply `jax.lax.stop_gradient` to the target shell vector inside the shell-prediction energy. The context vector, prediction weights, and outer-context pathway would still receive gradients from the local predictive objective, while the shell target would remain a measured target for that objective. Do not preserve the current target-gradient behavior behind a long-lived fallback flag. If a bidirectional shell-prediction objective is reintroduced, it should be implemented with explicit precision-controlled error routing instead of sharing this auxiliary node's measured-target semantics.

This is a diagnostic step, not a final claim about the architecture. A fully bidirectional shell-prediction objective is closer to predictive-coding inference when the prediction error is properly precision-weighted and routed through explicit error units. The current graph lacks that precision control, and the retained-readout result suggests the target-shell gradient may be overpowering class-discriminative shell organization.

Proposed experiment after that change:

- Run the same seed-42 retained-readout configuration.
- Keep `w_shell_ctx = 0.0005`.
- Use the revised measured-target shell-prediction objective.
- Keep direct outer-context readout connected to `output`.
- Keep `outer_shell_context_evidence=off`.
- Keep no teacher heads.
- Keep the 10-column, 3-shared, shell-attention, `shell_lr_multipliers=1,1.5,2,3`, learning-rate 0.005, 20-epoch configuration.

Decision criterion:

- If the measured-target objective recovers toward the 32.41% direct-context baseline while preserving or improving context-only accuracy, then the next architecture should add a precision-controlled version of shell prediction rather than dropping shell prediction.
- If the measured-target objective still stays near 26%, then the failure is probably not just the shell-target gradient. The next mechanism should be leave-one-shell-out context prediction, where the context for a target shell is built from the other shells instead of from the same shell state it is asked to predict.

Alternative approaches considered:

| Approach | Advantage | Reason not recommended as the immediate next step |
| --- | --- | --- |
| Sweep smaller `w_shell_ctx` values with the current bidirectional objective | Tests whether the retained-readout failure is a precision issue. | The displayed shell-prediction energy is already below six-decimal resolution, and the current gradient path still lets the auxiliary objective reshape shell targets. |
| Replicate the retained-readout run across seeds | Tests stability. | The seed-42 result is below both relevant comparisons, so more seeds would mostly measure a mechanism that already failed the direct check. |
| Remove shell prediction and return to the direct-context baseline | Restores the strongest seed-42 result in this family. | It gives up the HiBaCaML-style local predictive objective before testing whether the objective is failing because of gradient routing. |
| Change shell prediction to use a measured target with `jax.lax.stop_gradient` | Separates context-prediction learning from target-shell reshaping. | This is the recommended next diagnostic because it tests the most direct mechanism implicated by the run without adding a persistent fallback mode. |
| Build leave-one-shell-out context prediction immediately | Avoids self-prediction from a context latent that already consumed the target shell. | It is a larger architectural change. The target-gradient diagnostic should be run first because it can identify whether the present failure is caused by target-shell gradients. |

## 2026-07-11 Redirect to Stronger Column-Shell Branch

Timestamp and machine: 2026-07-11 07:35:40 EDT on `rogdora43`.

Direction change:

- The measured-target shell-prediction diagnostic above is paused.
- The next experiment restarts from earlier, stronger no-bypass results instead of continuing from the 25.93% retained-readout shell-prediction branch.
- The goal is to test a more productive way to combine columns and shells without adding another auxiliary prediction objective.

Earlier results being used as anchors:

| Branch | Configuration summary | Best relevant result | Mechanistic note |
| --- | --- | ---: | --- |
| 10-column outer-context baseline | 10 columns, 3 shared columns, 7 active non-shared columns, `combiner=shell_attention`, `outer_shell_context=on`, no bypass, no teacher heads, `shell_lr_multipliers=1,1.5,2,3` | 32.41% on seed 42, three-seed mean about 31.07% | The combined classifier used `column_pool` plus direct `columnXX_outer_shell_context` inputs. |
| Per-column shell bridge and readout branch | 4 columns, shell bridge on, per-column shell readout on, no bypass, no teacher heads | 33.62% on seed 7 | `column_shell_bridge_only` reached 28.66% while `column_pool` alone stayed at 10.00%. |
| Shell-preserving composer branch | 4 columns, shell evidence cascade on, shell inhibition on, `combiner=shell_attention`, no bypass, no bridge/readout | 32.36% on seed 99 | The composer route itself became class-readable, with `column_only` equal to combined accuracy. |

Mechanistic hypothesis:

The 10-column outer-context baseline is the strongest stable no-bypass family. Its weakness is that `column_pool` alone remains weak, while the direct context inputs carry useful evidence only when combined with the composer route. The earlier shell bridge branch showed a different useful route: `column_shell_bridge`, a per-column latent fed by pooled hard-kernel, inner-shell, middle-shell, and outer-shell states, can carry class evidence even when the ordinary `column_pool` route fails.

The new experiment combines these two successful mechanisms. `column_pool` is the shell-attention composer route. `columnXX_outer_shell_context` is the per-column context route from the 10-column baseline. `columnXX_shell_bridge` is the per-column shell bridge route that previously carried useful no-bypass evidence. Direct per-column shell readout is also tested because it gives each pooled shell state a direct classifier error path without adding teacher heads.

Implemented script:

- `scripts/run_codex_outer_context_bridge_readout_10col3shared_sweep.sh`.

The script runs three seed-42 cases sequentially:

| Case | `column_shell_bridge` | `column_shell_readout` | Purpose |
| --- | --- | --- | --- |
| `bridge_only` | on | off | Test whether the shell bridge adds useful per-column evidence to the 10-column outer-context baseline. |
| `readout_only` | off | on | Test whether direct pooled shell readout helps when paired with the 10-column outer-context route. |
| `bridge_plus_readout` | on | on | Test whether the earlier productive bridge/readout combination transfers to the 10-column outer-context baseline. |

Shared configuration:

- Seed 42.
- Learning rate 0.005 for 20 epochs.
- `diagnose_mode=composer_shells`.
- 10 columns, 3 shared columns, and 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- No bypass path.
- No column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `outer_shell_context_teacher_weight=0.0`.
- `outer_shell_context_evidence=off`.
- `outer_shell_context_shell_prediction_weight=0.0`.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values scale parameter updates for hard-kernel, inner-shell, middle-shell, and outer-shell parameters.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_outer_context_bridge_readout_10col3shared_sweep.sh
```

Environment overrides:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && SEED=99 NUM_EPOCHS=20 bash scripts/run_codex_outer_context_bridge_readout_10col3shared_sweep.sh
```

Expected analysis:

- Compare each case against the seed-42 10-column outer-context baseline of 32.41%.
- Use `column_shell_bridge_only` to determine whether the bridge path remains independently useful in the 10-column architecture.
- Use `column_shell_readout_only` to determine whether direct shell readout helps or only adds noisy classifier inputs.
- Use shell lesion diagnostics to determine whether hard-kernel dominance decreases and whether inner-shell, middle-shell, or outer-shell evidence becomes more retained.
- Treat collapse resistance as primary. A case that avoids chance-level `column_pool`, bridge, or context ablations is more important than a tiny combined-accuracy change.

Alternatives considered:

| Approach | Advantage | Reason not chosen now |
| --- | --- | --- |
| Continue with shell-prediction target-gradient correction | Directly addresses the latest failed mechanism. | The user correctly redirected us toward the stronger historical branches before making more changes to the weaker branch. |
| Replicate the 10-column outer-context baseline again | Reconfirms the current best stable no-bypass family. | The three-seed baseline already exists, and the next question is how to add a productive shell/column route to it. |
| Add teacher heads to the stronger branch | Could make local shells more class-readable. | Previous teacher-head results often created local class signal that did not align with the main route. The next test should change routing first, not add another objective. |
| Combine outer context with shell bridge and direct shell readout | Tests two historically useful routing mechanisms together without new auxiliary losses. | This is the chosen experiment. |

## 2026-07-12 Outer-Context Bridge/Readout Sweep Result

Timestamp and machine: 2026-07-12 03:24:59 EDT on `rogdora43`.

Completed run:

- Master log: `results/codex_outer_context_bridge_readout_10col3shared_rogdora43_20260711_074007.log`.
- `bridge_only` child log: `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260711_074007.log`.
- `readout_only` child log: `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr1_br0_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260711_121029.log`.
- `bridge_plus_readout` child log: `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr1_br1_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260711_164133.log`.

Shared configuration:

- Seed 42.
- 10 columns, 3 shared columns, and 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`, where `outer_shell_context` is a per-column latent that receives pooled hard-kernel, inner-shell, middle-shell, and outer-shell states and feeds the main `output` classifier.
- `column_shell_bridge` varied by case, where `column_shell_bridge` is a per-column latent fed by pooled shell states and connected to the main `output` classifier.
- `column_shell_readout` varied by case, where `column_shell_readout` means direct `columnXX_{shell}_pool -> output` edges from each pooled shell state.
- No bypass path.
- No column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `outer_shell_context_teacher_weight=0.0`.
- `outer_shell_context_evidence=off`.
- `outer_shell_context_shell_prediction_weight=0.0`.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values scale optimizer updates for hard-kernel, inner-shell, middle-shell, and outer-shell parameters.
- Learning rate 0.005 for 20 epochs.
- `diagnose_mode=composer_shells`.

Primary results:

| Case | `column_shell_bridge` | `column_shell_readout` | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| Prior 10-column outer-context baseline | off | off | 32.82% | 20 | 32.41% |
| `bridge_only` | on | off | 34.30% | 20 | 33.93% |
| `readout_only` | off | on | 26.98% | 7 | 26.68% |
| `bridge_plus_readout` | on | on | 19.88% | 7 | 20.28% |

Validation trajectory summary:

| Case | Early behavior | Late behavior |
| --- | --- | --- |
| `bridge_only` | Reached 29.94% validation at epoch 8. | Continued improving to 34.30% at epoch 20. |
| `readout_only` | Reached 26.98% validation at epoch 7. | Fell to 10.48% by epoch 20. |
| `bridge_plus_readout` | Reached 19.88% validation at epoch 7. | Stayed unstable and ended at 14.00% by epoch 20. |

Readout ablations on test:

| Case | Combined | `column_pool` only | Shell readout only | Shell bridge only | Outer context only | `column_pool` plus outer context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior 10-column outer-context baseline | 32.41% | 13.20% | not present | not present | 16.51% | 32.41% |
| `bridge_only` | 33.93% | 10.00% | not present | 10.00% | 10.00% | 10.00% |
| `readout_only` | 26.68% | 10.00% | 15.77% | not present | 10.00% | 19.77% |
| `bridge_plus_readout` | 20.28% | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |

Shell lesion tests on the combined readout:

| Case | Without hard kernel | Without inner shell | Without middle shell | Without outer shell |
| --- | ---: | ---: | ---: | ---: |
| Prior 10-column outer-context baseline | 15.56% | 22.12% | 23.93% | 21.74% |
| `bridge_only` | 22.98% | 25.08% | 23.43% | 14.95% |
| `readout_only` | 10.16% | 25.11% | 10.01% | 10.00% |
| `bridge_plus_readout` | 18.92% | 16.30% | 10.16% | 10.57% |

Shell norms:

| Case | Hard-kernel mean L2 after training | Inner-shell mean L2 after training | Middle-shell mean L2 after training | Outer-shell mean L2 after training |
| --- | ---: | ---: | ---: | ---: |
| `bridge_only` | 4.6884 | 2.6428 | 3.7381 | 4.5788 |
| `readout_only` | 4.6892 | 2.6456 | 3.7416 | 4.5824 |
| `bridge_plus_readout` | 4.6899 | 2.6454 | 3.7412 | 4.5820 |

Composer attention at the selected checkpoint:

| Case | Hard-kernel owner | Inner-shell owner | Middle-shell owner | Outer-shell owner |
| --- | --- | --- | --- | --- |
| `bridge_only` | `col_09` at 0.990800 | `col_01` at 0.996494 | `col_03` at 0.998733 | `col_04` at 0.999713 |
| `readout_only` | `col_03` at 0.985626 | `col_07` at 0.989531 | `col_03` at 0.994776 | `col_03` at 0.998020 |
| `bridge_plus_readout` | `col_00` at 0.919298 | `col_02` at 0.890878 | `col_03` at 0.991923 | `col_00` at 0.997665 |

Interpretation:

- `bridge_only` is the best seed-42 result in the current no-bypass 10-column family. It improved over the prior seed-42 10-column outer-context baseline by 1.52 percentage points, from 32.41% to 33.93%.
- `bridge_only` did not show magnitude collapse. The shell L2 norms stayed close to the pre-training values, and validation accuracy improved through epoch 20.
- `bridge_only` did not make the shell bridge independently class-readable. `column_shell_bridge_only`, `outer_shell_context_only`, `column_pool` only, and `column_pool` plus outer context were all at chance on test.
- The useful signal in `bridge_only` appears to be a joint readout interaction among `column_pool`, `columnXX_outer_shell_context`, and `columnXX_shell_bridge`. The current ablations do not yet include pairwise `outer_shell_context plus shell_bridge` or "combined without one readout family" cases, so this mechanism cannot be fully localized from the existing diagnostics.
- Direct shell readout was harmful in this 10-column outer-context setting. `readout_only` peaked at epoch 7 and collapsed toward chance by epoch 20, and `bridge_plus_readout` was worse than either `bridge_only` or `readout_only`.
- The direct shell readout edges created some isolated shell evidence in `readout_only`: `column_shell_readout_hard_kernel_only` reached 19.55% and `column_shell_readout_inner_shell_only` reached 17.85% on test. That isolated evidence did not coordinate with the main combined classifier.
- The outer shell is load-bearing in `bridge_only`. Removing it dropped test accuracy from 33.93% to 14.95%, while removing hard kernel, inner shell, or middle shell left 22.98%, 25.08%, and 23.43% respectively.
- This differs from the prior 10-column outer-context baseline, where removing the hard kernel was the most damaging lesion. The bridge-only result therefore looks like a real shift toward outer-shell participation.

Conclusion:

The productive branch is `column_shell_bridge=on` with `column_shell_readout=off`, not direct pooled-shell readout. The result supports the idea that a per-column shell-preserving bridge can improve the 10-column outer-context architecture, but only as part of the full predictive-coding readout graph. Direct shell readout should not be added to this branch unless its classifier pressure is redesigned.

Recommended next step:

Run a two-seed replicate of the `bridge_only` condition using the existing wrapper, with seeds 99 and 7. This tests whether the 33.93% seed-42 gain is stable across seeds before changing code. The important metrics are combined test accuracy, validation trajectory, `column_shell_bridge_only`, `outer_shell_context_only`, `column_pool` only, and shell lesion effects.

Commands:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_cifar10_depth_spanning.sh 99 0.005 composer_shells 20 0.0 0,0,0,0 nobypass 0,0,0,0 off on shell_attention on 10 3 7 1,1.5,2,3 0.0 off 0.0 0.0
```

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_cifar10_depth_spanning.sh 7 0.005 composer_shells 20 0.0 0,0,0,0 nobypass 0,0,0,0 off on shell_attention on 10 3 7 1,1.5,2,3 0.0 off 0.0 0.0
```

Possible code change after the replicate:

If `bridge_only` is stable across seeds, add readout-family lesion diagnostics rather than changing architecture first. The missing diagnostics are `combined_without_column_pool`, `combined_without_outer_shell_context`, `combined_without_column_shell_bridge`, and the pairwise combinations `column_pool_plus_shell_bridge`, `outer_shell_context_plus_shell_bridge`, and `column_pool_plus_outer_shell_context`. The goal would be to localize the joint interaction that made `bridge_only` work.
