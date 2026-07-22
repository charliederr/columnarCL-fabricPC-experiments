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

## 2026-07-12 Bridge-Only Replicate Runner

Timestamp and machine: 2026-07-12 03:29:25 EDT on `rogdora43`.

Prepared the approved two-seed replicate for the successful `bridge_only` condition.

Script:

- `scripts/run_codex_outer_context_bridge_only_10col3shared_replicate.sh`.

This script does not change the training graph or model code. It runs the existing `scripts/run_codex_cifar10_depth_spanning.sh` wrapper twice, once for seed 99 and once for seed 7.

Configuration:

- 10 columns, 3 shared columns, and 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- `column_shell_bridge=on`.
- `column_shell_readout=off`.
- No bypass path.
- No column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `outer_shell_context_teacher_weight=0.0`.
- `outer_shell_context_evidence=off`.
- `outer_shell_context_shell_prediction_weight=0.0`.
- `shell_lr_multipliers=1,1.5,2,3`.
- Learning rate 0.005 for 20 epochs.
- `diagnose_mode=composer_shells`.

Run command:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_outer_context_bridge_only_10col3shared_replicate.sh
```

Optional single-seed override:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && SEEDS=99 bash scripts/run_codex_outer_context_bridge_only_10col3shared_replicate.sh
```

Expected outputs:

- A master log named `results/codex_outer_context_bridge_only_10col3shared_replicate_<host>_<timestamp>.log`.
- One child training log per seed from `scripts/run_codex_cifar10_depth_spanning.sh`.

Interpretation plan after completion:

- Compare seeds 99 and 7 to the seed-42 `bridge_only` result of 33.93%.
- Check whether validation improves through late epochs or collapses after an early peak.
- Check whether the outer-shell lesion remains large.
- Check whether all single-route readout ablations remain chance-level, which would support the current hypothesis that the useful signal is a joint route interaction.

## 2026-07-12 Bridge-Only Replicate Result

Timestamp and machine: 2026-07-12 12:55 EDT on `rogdora43`.

Completed logs:

- Master log: `results/codex_outer_context_bridge_only_10col3shared_replicate_rogdora43_20260712_033022.log`.
- Seed 99 child log: `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_nobyp_seed99_lr0p005_ep20_composer_shells_rogdora43_20260712_033022.log`.
- Seed 7 child log: `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_nobyp_seed7_lr0p005_ep20_composer_shells_rogdora43_20260712_080455.log`.

Configuration:

- 10 columns, 3 shared columns, 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- `column_shell_bridge=on`.
- `column_shell_readout=off`.
- No bypass path.
- No column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `outer_shell_context_teacher_weight=0.0`.
- `outer_shell_context_evidence=off`.
- `outer_shell_context_shell_prediction_weight=0.0`.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values scale optimizer updates for hard-kernel, inner-shell, middle-shell, and outer-shell parameters.
- Learning rate 0.005 for 20 epochs.

Accuracy results:

| Seed | Best validation accuracy | Best validation epoch | Test accuracy | Validation trajectory |
| ---: | ---: | ---: | ---: | --- |
| 42 | 34.30% | 20 | 33.93% | Improved through epoch 20. |
| 99 | 33.96% | 20 | 33.62% | Improved through epoch 20. |
| 7 | 30.90% | 19 | 29.91% | Weaker than seeds 42 and 99, but did not collapse to chance. |

The three-seed mean test accuracy is 32.49%. The 10-column direct outer-context baseline recorded earlier had a three-seed mean of about 31.07%, so the shell bridge added about 1.42 percentage points on this matched family.

Test readout ablations:

| Seed | Combined | `column_pool` only | `column_shell_bridge` only | `column_pool` plus shell bridge | Outer context only | `column_pool` plus outer context |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 33.93% | 10.00% | 10.00% | not logged | 10.00% | 10.00% |
| 99 | 33.62% | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |
| 7 | 29.91% | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |

The signal remains a full-readout interaction. None of the currently logged single-route or partially combined routes can classify CIFAR-10 above chance. The missing diagnostic pair is `outer_shell_context plus column_shell_bridge`, and the missing family-removal tests are `combined_without_column_pool`, `combined_without_outer_shell_context`, and `combined_without_column_shell_bridge`.

Test shell lesion results:

| Seed | Combined | Without hard kernel | Without inner shell | Without middle shell | Without outer shell |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 33.93% | 22.98% | 25.08% | 23.43% | 14.95% |
| 99 | 33.62% | 11.65% | 17.53% | 12.55% | 23.64% |
| 7 | 29.91% | 13.08% | 24.62% | 13.18% | 10.00% |

Across the three seeds, hard kernel, middle shell, and outer shell are all load-bearing. The mean lesion accuracies are:

| Lesion | Mean accuracy | Mean drop from combined mean |
| --- | ---: | ---: |
| Without hard kernel | 15.90% | 16.59 percentage points |
| Without inner shell | 22.41% | 10.08 percentage points |
| Without middle shell | 16.39% | 16.10 percentage points |
| Without outer shell | 16.20% | 16.29 percentage points |

Shell norms after training:

| Seed | Hard-kernel mean L2 | Inner-shell mean L2 | Middle-shell mean L2 | Outer-shell mean L2 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 4.6884 | 2.6428 | 3.7381 | 4.5788 |
| 99 | 4.6890 | 2.6438 | 3.7375 | 4.5747 |
| 7 | 4.6883 | 2.6426 | 3.7375 | 4.5776 |

The shell magnitudes are almost identical across seeds, so seed 7's weaker accuracy is not a magnitude-collapse failure.

Composer attention owners:

| Seed | Hard-kernel owner | Inner-shell owner | Middle-shell owner | Outer-shell owner |
| ---: | --- | --- | --- | --- |
| 42 | `col_09` at 0.990800 | `col_01` at 0.996494 | `col_03` at 0.998733 | `col_04` at 0.999713 |
| 99 | `col_07` at 0.999668 | `col_04` at 0.999369 | `col_00` at 0.999331 | `col_09` at 0.993876 |
| 7 | `col_03` at 0.999571 | `col_05` at 0.998728 | `col_01` at 0.999654 | `col_02` at 0.999994 |

Interpretation:

- The `bridge_only` branch replicated. Seeds 42 and 99 were almost identical, and seed 7 remained well above chance.
- The branch is still not robust enough for the intended CIFAR-10 goal. A three-seed mean of 32.49% is an improvement over the prior no-bypass 10-column family, but it is far below the earlier bypass result and below ordinary CIFAR-10 classification quality.
- The useful mechanism is still hidden in the full readout graph. `column_pool`, `column_shell_bridge`, and `outer_shell_context` do not classify alone or in the currently tested pairs.
- The shell lesion results are more encouraging than the route ablations. Hard kernel, middle shell, and outer shell all matter across seeds, which means the classifier depends on the radial shell structure rather than using only one slice.
- The composer is effectively choosing one column per shell. This is sparse in practice, but it is not the paper's top-level support controller. The selected shell owners differ by seed, which suggests the model finds different local decompositions without a support teacher or certificate channel.

Conclusion:

`column_shell_bridge=on`, `outer_shell_context=on`, and `column_shell_readout=off` should remain the current best no-bypass 10-column branch. The result supports continuing with shell-preserving routing, but the next change should measure the route interaction explicitly before adding new objectives or changing shell dynamics.

Recommended next step:

Add readout-family lesion diagnostics to `scripts/train_cifar10_depth_spanning.py`:

- `combined_without_column_pool`.
- `combined_without_outer_shell_context`.
- `combined_without_column_shell_bridge`.
- `outer_shell_context_plus_column_shell_bridge`.
- Keep existing `column_pool_plus_shell_bridge` and `column_pool_plus_outer_shell_context`.

This is not an architectural change. It is a measurement change that should reveal whether the class signal is carried by the pair of context and bridge routes or only by the full three-route interaction. After that diagnostic, the next architectural step can be chosen more safely.

## 2026-07-12 Readout-Family Lesion Diagnostics

Timestamp and machine: 2026-07-12 13:10 EDT on `rogdora43`.

Implemented the recommended measurement change. No training graph, loss, optimizer, or model-path topology was changed. The implementation only expands the source sets evaluated by `evaluate_readout_ablations()` after a checkpoint has already been selected.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`.
- `tests/test_pooled_readout_norm.py`.

Mechanism:

- Added `build_readout_ablation_cases(structure)`, where `structure` is the FabricPC `GraphStructure` containing output input edges.
- `build_readout_ablation_cases()` returns named source-node sets for the `output` classifier.
- `evaluate_readout_ablations()` now iterates over that helper and still uses `mask_output_input_sources()` to create copied parameter trees for evaluation.
- `mask_output_input_sources()` zeroes dropped `output` edge weights in the copied parameter tree. It does not remove nodes, remove edges, alter inference, alter training, or alter the selected checkpoint.

New diagnostic cases:

- `combined_without_column_pool`: keeps every `output` input except `column_pool`.
- `combined_without_column_shell_bridge`: keeps every `output` input except all `columnXX_shell_bridge` sources.
- `combined_without_outer_shell_context`: keeps every `output` input except all `columnXX_outer_shell_context` sources.
- `outer_shell_context_plus_column_shell_bridge`: keeps the `columnXX_outer_shell_context` and `columnXX_shell_bridge` source families while dropping `column_pool` and any other output routes.

Existing pairwise cases kept:

- `column_pool_plus_shell_bridge`.
- `column_pool_plus_outer_shell_context`.

Added test:

- `test_build_readout_ablation_cases_includes_family_lesions()` builds a small graph with `column_shell_bridge=on`, `outer_shell_context=on`, and `bypass_columns=False`, then verifies that each new case keeps exactly the intended source-node family.

Validation:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && /home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result:

- 43 passed in 13.28 seconds.

Recommended experiment:

Run the current best diagnostic branch once with seed 42 so the new readout-family cases appear in both validation and test reports:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 composer_shells 20 0.0 0,0,0,0 nobypass 0,0,0,0 off on shell_attention on 10 3 7 1,1.5,2,3 0.0 off 0.0 0.0
```

Interpretation target:

- If `outer_shell_context_plus_column_shell_bridge` is above chance, then the useful signal can flow through the two per-column context/bridge routes without `column_pool`.
- If only `combined` is above chance, then `column_pool`, `columnXX_outer_shell_context`, and `columnXX_shell_bridge` form a three-route interaction.
- If `combined_without_column_pool` is strong but `outer_shell_context_plus_column_shell_bridge` is weak, then another output family is involved and the diagnostic should be expanded before changing the architecture.

## 2026-07-13 Readout-Family Lesion Diagnostic Result

Timestamp and machine: 2026-07-13 00:28 EDT on `rogdora43`.

Completed log:

- `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260712_181700.log`.

Configuration:

- 10 columns, 3 shared columns, 7 active non-shared columns.
- `combiner=shell_attention`.
- `outer_shell_context=on`.
- `column_shell_bridge=on`.
- `column_shell_readout=off`.
- No bypass path.
- No column teacher energy, no shell teacher energy, and no per-column shell teacher energy.
- `outer_shell_context_teacher_weight=0.0`.
- `outer_shell_context_evidence=off`.
- `outer_shell_context_shell_prediction_weight=0.0`.
- `shell_lr_multipliers=1,1.5,2,3`.
- Learning rate 0.005 for 20 epochs.

Main result:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 28.70% |
| Best validation epoch | 8 |
| Test accuracy at best validation checkpoint | 28.99% |

This rerun underperformed the earlier seed-42 `bridge_only` run, which reached 34.30% validation and 33.93% test. The command-line configuration matched, and the diagnostic code should only affect post-training evaluation. The lower score should therefore be treated as training instability or version/state variance rather than evidence that the readout-family diagnostics changed the trained architecture.

Validation trajectory:

| Epoch | Validation accuracy |
| ---: | ---: |
| 1 | 16.28% |
| 4 | 23.58% |
| 8 | 28.70% |
| 12 | 27.56% |
| 15 | 17.14% |
| 20 | 25.14% |

The run did not collapse to chance, but it also did not improve through epoch 20 as the stronger seed-42 and seed-99 runs did.

New readout-family diagnostics:

| Test readout case | Accuracy | Interpretation |
| --- | ---: | --- |
| `combined` | 28.99% | Full `column_pool` plus outer-context plus shell-bridge route. |
| `column_only` | 10.00% | The shell-attention composer route alone is chance. |
| `combined_without_column_pool` | 18.46% | Outer-context plus shell-bridge carries partial class signal without the composer route. |
| `column_shell_bridge_only` | 10.00% | Shell bridge alone is chance. |
| `column_pool_plus_shell_bridge` | 10.00% | Composer plus shell bridge is chance without outer context. |
| `combined_without_column_shell_bridge` | 10.00% | Composer plus outer context is chance without shell bridge. |
| `outer_shell_context_only` | 10.00% | Outer context alone is chance. |
| `column_pool_plus_outer_shell_context` | 10.00% | Composer plus outer context is chance without shell bridge. |
| `combined_without_outer_shell_context` | 10.00% | Composer plus shell bridge is chance without outer context. |
| `outer_shell_context_plus_column_shell_bridge` | 18.46% | Same source family as `combined_without_column_pool`; partial class signal exists in the context/bridge pair. |

Shell lesion diagnostics:

| Test case | Accuracy |
| --- | ---: |
| `combined_without_hard_kernel` | 22.52% |
| `combined_without_inner_shell` | 20.69% |
| `combined_without_middle_shell` | 16.71% |
| `combined_without_outer_shell` | 19.64% |

Every shell lesion damaged the combined classifier. The middle shell was the largest lesion in this run, dropping test accuracy from 28.99% to 16.71%. The hard-kernel, inner-shell, and outer-shell lesions also remained meaningful.

Shell norms after training:

| Shell | Mean L2 |
| --- | ---: |
| Hard kernel | 4.6883 |
| Inner shell | 2.6423 |
| Middle shell | 3.7401 |
| Outer shell | 4.5812 |

The shell magnitudes remained stable, so the weaker score is not a magnitude-collapse result.

Composer attention owners:

| Shell | Owner |
| --- | --- |
| Hard kernel | `col_09` at 0.987025 |
| Inner shell | `col_01` at 0.986865 |
| Middle shell | `col_07` at 0.993859 |
| Outer shell | `col_03` at 0.997998 |

Interpretation:

- The diagnostic result rules out the simplest "only full three-route interaction works" explanation. The pair `columnXX_outer_shell_context` plus `columnXX_shell_bridge` carries partial class signal without `column_pool`.
- The diagnostic also shows that `column_pool` is still useful: adding it to the context/bridge pair raised test accuracy from 18.46% to 28.99% in this run.
- The individual routes and the pairs involving `column_pool` were chance-level. The route interaction is asymmetric: outer context and shell bridge form the minimum useful pair, and the composer route improves that pair only when both are already present.
- This supports treating shell bridge and outer context as a coupled per-column mechanism rather than separate auxiliary routes.
- The underperformance relative to the earlier seed-42 run means we should avoid overfitting the next architectural step to this single rerun's lower accuracy.

Recommended next step:

Add a coupled `outer_shell_context -> column_shell_bridge` pathway and test it against the current best branch. The mechanism should keep the same output families but make the bridge explicitly conditioned on the outer-context latent before the classifier. This is still aligned with the columnar interpretation: the outer shell carries exploratory/contextual residue, and the shell bridge integrates cross-shell column state. The current diagnostic says those two routes are the first pair to become class-informative together.

Implementation constraint:

- Keep `column_shell_readout=off`.
- Keep teacher heads off.
- Do not add a direct bypass.
- Do not change the upstream FabricPC code.
- Make the new pathway optional with a command-line flag so the existing branch remains available for matched ablations.

## 2026-07-13 Outer-Context-to-Bridge Conditioning Implementation

Timestamp and machine: 2026-07-13 00:55 EDT on `rogdora43`.

Implemented the recommended optional pathway. This changes only `columnarCL-fabricPC-experiments`; upstream FabricPC was not modified.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`.
- `scripts/run_codex_cifar10_depth_spanning.sh`.
- `tests/test_pooled_readout_norm.py`.

Mechanism:

- Added `--outer_shell_context_to_bridge`.
- The flag requires both `--outer_shell_context` and `--column_shell_bridge`.
- When enabled, each active column gets an additional edge from its outer-context latent into its shell bridge latent:
  - `columnXX_outer_shell_context -> columnXX_shell_bridge:in`.
- The existing direct output edges remain:
  - `columnXX_outer_shell_context -> output`.
  - `columnXX_shell_bridge -> output`.
  - `column_pool -> output`.
- The path is optional and defaults to off, so previous experiments keep the same graph unless the flag is set.

Architectural intent:

- `columnXX_outer_shell_context` is the outer-shell-width latent that summarizes a column's pooled hard-kernel, inner-shell, middle-shell, and outer-shell states.
- `columnXX_shell_bridge` is the full-width latent that integrates the same column's pooled shell states before reaching the classifier.
- The new edge lets the bridge condition its full-width shell-integrating state on the context latent before classifier readout.
- This directly tests the diagnostic finding that `outer_shell_context_plus_column_shell_bridge` was the first above-chance readout pair.

Diagnostic masking update:

- Updated `mask_column_shell_bridge_inputs()` so shell lesions also mask shell inputs inside any `columnXX_outer_shell_context` node that feeds the bridge.
- Without this fix, a bridge shell lesion could remove a direct pooled-shell input while letting the same shell still reach the bridge through `outer_shell_context`.
- The updated helper keeps the context source connected to the bridge, but masks the context node's own shell inputs according to the same lesion.

Runner change:

- `scripts/run_codex_cifar10_depth_spanning.sh` now accepts positional argument 21 for `outer_shell_context_to_bridge_mode`.
- Accepted values:
  - On: `on`, `true`, `outerbridge`.
  - Off: `off`, `false`, `noouterbridge`.
- The log header records `outer_shell_context_to_bridge`.
- The child log filename includes `ocb1` or `ocb0`.

Tests added:

- `test_depth_spanning_graph_adds_outer_context_to_shell_bridge_edges()`.
- `test_outer_shell_context_to_bridge_requires_context_path()`.
- `test_outer_shell_context_to_bridge_requires_shell_bridge()`.
- `test_mask_column_shell_bridge_inputs_masks_context_conditioning()`.

Validation:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && /home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result:

- Passed.

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && /home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result:

- 47 passed in 12.89 seconds.

Recommended experiment:

Run a seed-42 comparison with the new conditioning edge enabled:

```bash
cd /home/ni/repos/fpc/columnarCL-fabricPC-experiments && bash scripts/run_codex_cifar10_depth_spanning.sh 42 0.005 composer_shells 20 0.0 0,0,0,0 nobypass 0,0,0,0 off on shell_attention on 10 3 7 1,1.5,2,3 0.0 off 0.0 0.0 on
```

Interpretation target:

- Compare the result to the best previous `bridge_only` seed-42 test result of 33.93%.
- Check whether `outer_shell_context_plus_column_shell_bridge` improves above the previous 18.46% diagnostic result.
- Check whether `combined_without_column_pool` improves, which would mean the coupled context/bridge mechanism is carrying more class signal without relying on the composer route.
- Check shell lesions to confirm the new context-to-bridge pathway does not bypass the shell structure.

## 2026-07-13 Outer-Context-to-Bridge Conditioning Result

Recorded on 2026-07-13 13:36 EDT on `rogdora43`.

Run log:

- `results/codex_dspan_10c_3s_7a_shatt_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_ocb1_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260713_063611.log`

Configuration:

- Seed: 42.
- Columns: 10 total, with 3 shared columns and 7 active non-shared columns.
- Learning rate: 0.005.
- Epochs: 20.
- Readout mode: no bypass.
- Combiner: `shell_attention`.
- Shell learning-rate multipliers: hard kernel 1.0, inner shell 1.5, middle shell 2.0, outer shell 3.0.
- Column shell bridge: enabled.
- Direct pooled-shell readout: disabled.
- Outer-shell context: enabled.
- Outer-shell context to shell bridge: enabled.
- Shell evidence cascade: enabled.
- Same-tier shell inhibition strengths: hard kernel 0.0, inner shell 0.35, middle shell 0.22, outer shell 0.10.

Top-line result:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 27.46% |
| Best validation epoch | 20 |
| Test accuracy at best validation checkpoint | 26.29% |

Validation trajectory:

| Epoch | Validation accuracy |
| ---: | ---: |
| 1 | 16.24% |
| 2 | 18.98% |
| 3 | 22.16% |
| 4 | 17.14% |
| 5 | 22.36% |
| 6 | 19.38% |
| 7 | 22.34% |
| 8 | 22.20% |
| 9 | 18.70% |
| 10 | 21.62% |
| 11 | 17.58% |
| 12 | 9.88% |
| 13 | 13.68% |
| 14 | 19.64% |
| 15 | 19.52% |
| 16 | 24.24% |
| 17 | 23.66% |
| 18 | 18.10% |
| 19 | 27.22% |
| 20 | 27.46% |

Readout-family ablations on the test set:

| Readout family | Test accuracy |
| --- | ---: |
| `combined` | 26.29% |
| `column_only` | 10.00% |
| `combined_without_column_pool` | 10.00% |
| `column_shell_bridge_only` | 10.00% |
| `column_pool_plus_shell_bridge` | 10.00% |
| `combined_without_column_shell_bridge` | 10.00% |
| `outer_shell_context_only` | 10.00% |
| `column_pool_plus_outer_shell_context` | 10.00% |
| `combined_without_outer_shell_context` | 10.00% |
| `outer_shell_context_plus_column_shell_bridge` | 10.00% |

Shell lesion ablations on the test set:

| Lesion condition | Test accuracy |
| --- | ---: |
| `combined_without_hard_kernel` | 18.80% |
| `combined_without_inner_shell` | 24.79% |
| `combined_without_middle_shell` | 16.97% |
| `combined_without_outer_shell` | 10.79% |

Shell state norms after training:

| Shell | Width | Mean L2 norm |
| --- | ---: | ---: |
| Hard kernel | 22 | 4.6888 |
| Inner shell | 7 | 2.6440 |
| Middle shell | 14 | 3.7383 |
| Outer shell | 21 | 4.5793 |

Interpretation:

- The direct `columnXX_outer_shell_context -> columnXX_shell_bridge:in` edge did not improve the target mechanism.
- The previous diagnostic run without this edge had `outer_shell_context_plus_column_shell_bridge` at 18.46% test accuracy. With the new conditioning edge enabled, the same readout family dropped to 10.00%.
- The full classifier also fell from the earlier best seed-42 bridge-only result of 33.93% test accuracy to 26.29%.
- The outer shell remains load-bearing inside the combined classifier. Removing outer-shell inputs drops test accuracy from 26.29% to 10.79%.
- The shell state norms did not collapse. The hard-kernel, middle-shell, and outer-shell states all retained substantial L2 norm after training.
- The failure is therefore not simple activation disappearance. The more likely mechanism is that direct context-to-bridge coupling entangles the bridge route with the context route in a way that destroys the class-readable two-route signal observed in the no-conditioning diagnostic.

Conclusion:

The direct context-to-bridge edge should not be the next accuracy path. It is useful to keep it as a controlled flag because it produced a clear negative result, but it should stay off in the next productive experiments.

Recommended next direction:

Replace the direct full-strength context-to-bridge edge with a controlled modulatory version before testing this idea again. A faithful predictive-coding version would let `columnXX_outer_shell_context` influence `columnXX_shell_bridge` through a small residual or precision-like gate rather than as an unconstrained same-strength input.

Concrete implementation candidate:

- Add an optional `--outer_shell_context_bridge_scale` value.
- Keep the direct context-to-bridge path disabled by default.
- When enabled, send the context latent into the bridge through a learned or fixed scale initialized near zero.
- Test fixed scales such as 0.025, 0.05, and 0.10 against the no-conditioning bridge-only baseline.
- Keep the shell lesion diagnostics and readout-family diagnostics unchanged so the mechanism can be compared against the 33.93% bridge-only seed-42 result and the 18.46% no-conditioning context-plus-bridge diagnostic.

This preserves the HiBaCaML-motivated idea that outer context should modulate shell integration, but avoids letting the context latent become an uncontrolled peer input to the shell bridge.

## 2026-07-13 Stage3 Column-Grid / 96-Width Implementation

Recorded on 2026-07-13 13:50 EDT on `rogdora43`.

Direction:

- The latest direct `outer_shell_context -> column_shell_bridge` result is treated as a negative result, not as the next branch to refine.
- The implementation returns to the strongest no-bypass branch:
  - 10 columns.
  - 3 shared columns.
  - 7 active non-shared columns.
  - `outer_shell_context=on`.
  - `column_shell_bridge=on`.
  - `column_shell_readout=off`.
  - `outer_shell_context_to_bridge=off`.
  - `combiner=shell_attention`.
  - `shell_lr_multipliers=1,1.5,2,3`.
- The new change aligns that branch more closely with the HiBaCaML CIFAR guidance by giving the columns a larger token grid and a wider shell substrate before adding more local objectives.

Implemented changes:

- Added `--column_grid` to `scripts/train_cifar10_depth_spanning.py`.
- `column_grid` names the backbone stage whose spatial grid defines the shared token grid for all stage taps and all depth-spanning columns.
- The default is `stage4`, which preserves the historical ResNet-18 CIFAR-10 setting of 4 by 4 tokens.
- The new recommended setting is `stage3`, which gives 8 by 8 tokens on ResNet-18 CIFAR-10.
- Added `resolve_column_target_grid()` in `scripts/train_cifar10_depth_spanning.py`.
- Updated the graph header to print `Column grid`, `Column grid source`, `Target grid`, and token count.
- Updated `columnar_cl_fabricpc/columns/stage_taps.py` so `StageTapTokenizer` supports both exact average-pooling down and linear resizing up.
- This shared tokenizer fix is necessary because `stage4_tap` must map a 4 by 4 source feature map onto the selected 8 by 8 `stage3` token grid.
- Updated `scripts/run_codex_cifar10_depth_spanning.sh` to accept three new trailing arguments:
  - argument 22: `column_grid`.
  - argument 23: `embed_dim`.
  - argument 24: `microcolumn_dim`.
- Added `scripts/run_codex_stage3_96_bridge_best.sh`, a short wrapper for the recommended branch.

Exact shell widths under the recommended diagnostic:

- `embed_dim=96` means the column output feature width is 96.
- The current proportional slicer uses the existing `32:10:20:30` hard-kernel / inner-shell / middle-shell / outer-shell proportions.
- With total width 96, the realized shell widths are:
  - hard kernel: 33.
  - inner shell: 11.
  - middle shell: 21.
  - outer shell: 31.
- This is close to the HiBaCaML CIFAR sketch. If we later want the exact `32,10,20,30` shell widths, the clean implementation should add explicit shell-width control rather than changing the global proportional slicer implicitly.

Validation:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py columnar_cl_fabricpc/columns/stage_taps.py
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_stage3_96_bridge_best.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result:

- 49 passed in 13.57 seconds.

Recommended diagnostic run:

```bash
bash scripts/run_codex_stage3_96_bridge_best.sh 42 10
```

Argument meanings:

- `42` is the seed.
- `10` is the number of epochs.
- The wrapper defaults to learning rate `0.005` and `diagnose_mode=composer_shells`.

Interpretation target:

- The first check is whether the wider 64-token column grid avoids early collapse and improves above the 10-column bridge-only family by epoch 10.
- If the validation trajectory is still improving at epoch 10, rerun the same script with 20 epochs.
- Compare against the earlier 10-column bridge-only seed-42 result of 33.93% test accuracy and the three-seed mean of 32.49%.

## 2026-07-13 Stage3 Column-Grid / 96-Width Diagnostic Result

Recorded on 2026-07-13 20:16 EDT on `rogdora43`.

Run log:

- `results/codex_dspan_10c_3s_7a_shatt_cgstage3_e96_m32_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep10_composer_shells_rogdora43_20260713_135318.log`

Configuration:

- Seed: 42.
- Columns: 10 total, with 3 shared columns and 7 active non-shared columns.
- Column grid: `stage3`.
- Target grid: 8 by 8, for 64 column tokens.
- Column feature width: 96.
- Microcolumn hidden width: 32.
- Realized shell widths: hard kernel 33, inner shell 11, middle shell 21, outer shell 31.
- Learning rate: 0.005.
- Epochs: 10.
- Readout mode: no bypass.
- Combiner: `shell_attention`.
- Column shell bridge: enabled.
- Direct pooled-shell readout: disabled.
- Outer-shell context: enabled.
- Outer-shell context to shell bridge: disabled.
- Shell learning-rate multipliers: hard kernel 1.0, inner shell 1.5, middle shell 2.0, outer shell 3.0.

Top-line result:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 21.96% |
| Best validation epoch | 7 |
| Test accuracy at best validation checkpoint | 22.19% |

Validation trajectory:

| Epoch | Validation accuracy |
| ---: | ---: |
| 1 | 14.60% |
| 2 | 16.50% |
| 3 | 12.78% |
| 4 | 19.82% |
| 5 | 18.34% |
| 6 | 19.78% |
| 7 | 21.96% |
| 8 | 18.36% |
| 9 | 19.98% |
| 10 | 19.72% |

Readout-family ablations on the test set:

| Readout family | Test accuracy |
| --- | ---: |
| `combined` | 22.19% |
| `column_only` | 10.00% |
| `combined_without_column_pool` | 15.01% |
| `column_shell_bridge_only` | 10.00% |
| `column_pool_plus_shell_bridge` | 10.00% |
| `combined_without_column_shell_bridge` | 16.49% |
| `outer_shell_context_only` | 12.51% |
| `column_pool_plus_outer_shell_context` | 16.49% |
| `combined_without_outer_shell_context` | 10.00% |
| `outer_shell_context_plus_column_shell_bridge` | 15.01% |

Shell lesion ablations on the test set:

| Lesion condition | Test accuracy |
| --- | ---: |
| `combined_without_hard_kernel` | 16.48% |
| `combined_without_inner_shell` | 22.27% |
| `combined_without_middle_shell` | 18.62% |
| `combined_without_outer_shell` | 20.33% |

Shell state norms:

| Shell | Before training mean L2 | After training mean L2 |
| --- | ---: | ---: |
| Hard kernel | 5.7322 | 5.7424 |
| Inner shell | 3.3015 | 3.3149 |
| Middle shell | 4.5587 | 4.5805 |
| Outer shell | 5.5252 | 5.5647 |

Interpretation:

- This run underperformed the current best no-bypass 10-column bridge-only branch. The earlier seed-42 bridge-only branch reached 33.93% test accuracy, while this stage3-grid, 96-width diagnostic reached 22.19%.
- The validation curve peaked at epoch 7 and then declined, so simply extending this exact configuration to 20 epochs is not the best next use of compute.
- This was not a magnitude collapse. All four shell state norms remained stable or increased slightly.
- The full classifier still required the outer-shell context route. Removing outer-shell context dropped test accuracy from 22.19% to 10.00%.
- The outer shell itself became weakly load-bearing. Removing the outer shell dropped test accuracy only from 22.19% to 20.33%. In the stronger 33.93% branch, removing the outer shell dropped test accuracy to 14.95%.
- The hard kernel became the most important shell lesion in this run. Removing it dropped test accuracy from 22.19% to 16.48%.
- The shell bridge did not become independently class-readable. `column_shell_bridge_only` and all bridge shell lesions remained at chance.

Mechanistic conclusion:

The stage3 token grid and 96-wide shell substrate are still plausible HiBaCaML-aligned ingredients, but they changed the predictive-coding energy balance. The local Gaussian consistency terms now cover many more latent dimensions than in the stage4, 64-width graph: 64 tokens times 96 features instead of 16 tokens times 64 features. The classifier cross-entropy target still has 10 class dimensions. The expanded local graph therefore gives much more unnormalized precision to latent consistency than to class-discriminative pressure.

Recommended next direction:

Do not run a 20-epoch repeat of this exact stage3/96 configuration yet. First, add an explicit local precision normalization for the expanded columnar graph. The target is not plain backprop. The target is predictive coding with precision-scaled local errors, so increasing token count and feature width does not silently increase the influence of local Gaussian error terms.

Concrete implementation candidate:

- Add a local normalized Gaussian energy for columnar stage taps, depth-spanning columns, the shell-aware combiner, per-column shell bridges, and outer-shell context nodes.
- Normalize each node's Gaussian energy per sample by its non-batch latent size, or add an explicit `--column_gaussian_precision` control whose default preserves existing behavior.
- Use the normalized setting for `column_grid=stage3` and `embed_dim=96`, while preserving the historical default for stage4 experiments unless explicitly changed.
- Rerun the same 10-epoch seed-42 stage3/96 diagnostic after the precision correction.

This is a shared-mechanism fix. It treats the move from 16 tokens to 64 tokens as a predictive-coding precision change, not just a capacity change.

## 2026-07-13 Columnar Gaussian Precision Normalization Implementation

Recorded on 2026-07-13 20:58 EDT on `rogdora43`.

Direction:

- Implemented the shared precision correction proposed after the failed stage3/96 diagnostic.
- Upstream FabricPC was not modified.
- Historical summed-Gaussian behavior remains the default so prior stage4 results remain comparable.
- The stage3/96 wrapper now enables the normalized local precision mode by default.

Mechanism:

- Added `MeanSquaredGaussianEnergy` in `columnar_cl_fabricpc/columns/normalized_gaussian.py`.
- For `e = z_latent - z_mu`, where `z_latent` is the inferred predictive-coding latent and `z_mu` is the node prediction, the new energy computes:
  - `E = 0.5 * precision * sum(e**2) / D`.
  - `D` is the number of non-batch latent elements in that node.
- The corresponding explicit latent gradient helper computes:
  - `dE/dz_latent = precision * e / D`.
- This keeps the local Gaussian predictive-coding objective but prevents larger token grids and wider feature axes from silently increasing local error precision.

Graph wiring:

- Added `--normalize_column_gaussian_energy`.
- Added `--column_gaussian_precision`.
- Added `make_column_gaussian_energy()` in `scripts/train_cifar10_depth_spanning.py`.
- When normalized mode is enabled, the graph uses `MeanSquaredGaussianEnergy`.
- When normalized mode is disabled, the graph uses FabricPC's `GaussianEnergy`.
- The selected local Gaussian energy is applied to:
  - `stage2_tap`, `stage3_tap`, and `stage4_tap`.
  - `stage4_pool`.
  - every `DepthSpanningColumnNode`.
  - `combiner`, including `ColumnShellComposerNode`.
  - `column_pool`.
  - global shell teacher slice nodes when present.
  - per-column shell slice nodes and shell pools.
  - per-column outer-shell context latents.
  - per-column shell bridge latents.
  - the optional class-shaped outer-shell context evidence latent.
- Classifier heads still use cross-entropy energy. The backbone convolution and skip nodes still use their existing FabricPC Gaussian energy.

Shared constructor updates:

- `create_stage_tap()` now accepts an optional `energy` argument.
- `create_global_pool()` now accepts an optional `energy` argument.
- `create_depth_spanning_column()` now accepts an optional `energy` argument.
- `create_depth_spanning_column_pool()` now forwards that optional `energy` argument.

Runner updates:

- `scripts/run_codex_cifar10_depth_spanning.sh` now accepts two new trailing arguments:
  - argument 25: normalized column Gaussian mode, with `on`, `true`, or `mean` enabling normalized local Gaussian energy.
  - argument 26: `column_gaussian_precision`.
- The log header records `normalize_column_gaussian_energy` and `column_gaussian_precision`.
- The child log filename includes `ng1` or `ng0`, plus the precision label.
- `scripts/run_codex_stage3_96_bridge_best.sh` now passes normalized mode `on` and precision `1.0`.

Validation:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py columnar_cl_fabricpc/columns/stage_taps.py columnar_cl_fabricpc/columns/depth_spanning_column.py columnar_cl_fabricpc/columns/normalized_gaussian.py columnar_cl_fabricpc/columns/__init__.py
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_stage3_96_bridge_best.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result:

- 52 passed in 13.65 seconds.

Recommended diagnostic run:

```bash
bash scripts/run_codex_stage3_96_bridge_best.sh 42 10
```

Interpretation target:

- Compare directly against the previous unnormalized stage3/96 diagnostic, which reached 22.19% test accuracy.
- Check whether normalized local precision restores useful outer-shell participation. In the failed unnormalized run, removing outer shell only dropped test accuracy from 22.19% to 20.33%.
- Check whether the validation curve continues rising past epoch 7. If it does, rerun for 20 epochs.
- Do not compare only the top-line number. The route diagnostics should show whether `combined_without_column_pool`, `combined_without_column_shell_bridge`, and `combined_without_outer_shell_context` become less chance-like.

## 2026-07-14 Stage3/96 Full Mean-Normalized Gaussian Result

Recorded on 2026-07-14 04:30 EDT on `rogdora43`.

Run:

- Log: `results/codex_dspan_10c_3s_7a_shatt_cgstage3_e96_m32_ng1_gp1p0_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep10_composer_shells_rogdora43_20260713_211226.log`.
- Git commit recorded by the run: `a999f94d9aaf3bd969a61c3fca65576db7630bd3`.
- Configuration: 10 columns, 3 shared columns, 7 active non-shared columns, `column_grid=stage3`, `embed_dim=96`, `microcolumn_dim=32`, `shell_attention`, `column_shell_bridge=on`, `outer_shell_context=on`, no bypass, `normalize_column_gaussian_energy=on`, `column_gaussian_precision=1.0`.
- The normalized local Gaussian energy was `E = 0.5 * precision * sum(e**2) / D`, where `E` is local predictive-coding energy, `precision` is the scalar error weight, `e` is the node residual `z_latent - z_mu`, and `D` is the number of non-batch latent elements in that node.

Main result:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 18.32% |
| Best validation epoch | 7 |
| Test accuracy | 18.89% |

Validation trajectory:

| Epoch | Validation accuracy |
| ---: | ---: |
| 1 | 15.52% |
| 2 | 13.08% |
| 3 | 15.12% |
| 4 | 13.64% |
| 5 | 15.02% |
| 6 | 18.20% |
| 7 | 18.32% |
| 8 | 16.46% |
| 9 | 16.20% |
| 10 | 15.86% |

Test route diagnostics:

| Route or lesion | Test accuracy |
| --- | ---: |
| Combined classifier | 18.89% |
| Column pool only | 10.00% |
| Combined without column pool | 10.00% |
| Shell bridge only | 10.00% |
| Combined without shell bridge | 10.00% |
| Outer-shell context only | 10.00% |
| Combined without outer-shell context | 10.01% |
| Combined without hard kernel | 10.00% |
| Combined without inner shell | 18.18% |
| Combined without middle shell | 10.00% |
| Combined without outer shell | 13.87% |

Comparison against the preceding stage3/96 run:

| Run | Local Gaussian | Best validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Stage3/96 seed 42, 10 epochs | summed residual energy | 21.96% | 22.19% |
| Stage3/96 seed 42, 10 epochs | full latent mean residual energy, precision 1.0 | 18.32% | 18.89% |

Comparison against the stronger stage4 line:

| Run | Commit recorded by run | Best validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Stage4, 10 columns, seed 42, 20 epochs, 2026-07-11 | `cccaecd58e14e7744fe2f65949ddf510bd4b88ed` | 34.30% | 33.93% |
| Stage4, 10 columns, seed 42, 20 epochs, 2026-07-12 | `6eb04f446e2e60452777e3a920cbd6ca93ac4b65` | 28.70% | 28.99% |
| Stage4, 10 columns, seed 7, 20 epochs, 2026-07-12 | `6eb04f446e2e60452777e3a920cbd6ca93ac4b65` | 30.90% | 29.91% |
| Stage4, 10 columns, seed 99, 20 epochs, 2026-07-12 | `6eb04f446e2e60452777e3a920cbd6ca93ac4b65` | 33.96% | 33.62% |

Interpretation:

- Full latent mean-normalization did not solve the stage3/96 problem. It made the result worse than the unnormalized stage3/96 run by 3.30 percentage points on test accuracy.
- The diagnostic routes became more chance-like than before. The combined classifier reached 18.89%, but removing any of the column pool, shell bridge, or outer-shell context routes dropped the test result to about chance.
- The shell lesion pattern says the combined classifier was carried mainly by the hard kernel, middle shell, and outer shell together. Removing the inner shell barely changed the result, from 18.89% to 18.18%.
- The current `MeanSquaredGaussianEnergy` divides by all latent elements. For a stage3 token node with 64 tokens and 96 feature channels, `D = 6144`. For the historical stage4 token node with 16 tokens and 64 feature channels, `D = 1024`. The new correction therefore weakens the stage3 local predictive-coding residual by a factor of 6144 when `precision=1.0`.
- This appears to underconstrain the local predictive-coding latents. The earlier unnormalized stage3 run likely overconstrained the expanded spatial grid, but the full mean-normalized run went too far in the other direction.

Recommended next direction:

- Do not continue with full latent mean-normalization at `precision=1.0`.
- Keep the stronger stage4 result as the current anchor: the 2026-07-11 seed-42 stage4 run reached 33.93% test accuracy before the later stage3 and normalization changes.
- Replace full latent mean-normalization with spatial-reference Gaussian scaling for columnar token nodes.
- Mechanism: compute `S`, the number of spatial or token sites in the node, and `S_ref`, the historical reference site count. For the ResNet18 stage4 column grid, `S_ref = 16` because the grid is 4 by 4. Scale local Gaussian energy by `max(1, S / S_ref)` rather than by all non-batch latent elements.
- This means a stage4 token node with `S = 16` keeps the historical summed residual energy. A stage3 token node with `S = 64` divides by 4, correcting the extra replicated spatial positions without weakening the feature channels or one-dimensional shell/context nodes.
- This is more aligned with the HiBaCaML direction than the full mean version. It treats more spatial sites as more cortical positions, not as a reason to erase per-feature predictive precision inside each position.

Proposed next implementation, pending confirmation:

- Add a `SpatialReferenceGaussianEnergy` or equivalent local energy class in `columnar_cl_fabricpc/columns/normalized_gaussian.py`.
- Add CLI controls for enabling spatial-reference scaling and setting the reference site count, with `16` as the default reference for the current ResNet18 stage4 anchor.
- Apply it to the same columnar predictive-coding nodes that received the full mean energy.
- Preserve historical stage4 behavior when the scaling mode is off or when the node has 16 or fewer token sites.
- Rerun the stage3/96 seed-42 10-epoch diagnostic and compare against both the 22.19% unnormalized stage3 result and the 33.93% best stage4 anchor.

## 2026-07-14 Spatial-Reference Gaussian Implementation

Recorded on 2026-07-14 04:39 EDT on `rogdora43`.

Implemented after confirmation:

- Added `SpatialReferenceGaussianEnergy` in `columnar_cl_fabricpc/columns/normalized_gaussian.py`.
- Exported `SpatialReferenceGaussianEnergy` from `columnar_cl_fabricpc/columns/__init__.py`.
- Replaced the Boolean training flag with one explicit mode selector:
  - `--column_gaussian_energy_mode sum`.
  - `--column_gaussian_energy_mode mean`.
  - `--column_gaussian_energy_mode spatial_reference`.
- Added `--column_gaussian_reference_sites`, with default `16.0`.
- Updated `make_column_gaussian_energy()` in `scripts/train_cifar10_depth_spanning.py` so every columnar predictive-coding node receives the selected local Gaussian energy.
- Updated `scripts/run_codex_cifar10_depth_spanning.sh` so logs record `column_gaussian_energy_mode`, `column_gaussian_precision`, and `column_gaussian_reference_sites`.
- Updated `scripts/run_codex_stage3_96_bridge_best.sh` to run the stage3/96 diagnostic with `column_gaussian_energy_mode=spatial_reference`, `column_gaussian_precision=1.0`, and `column_gaussian_reference_sites=16`.

Mechanism:

- `S` is the number of spatial or token sites in a node latent.
- `S_ref` is the reference site count whose summed residual precision is preserved.
- For CIFAR-10 ResNet18 stage4 columns, `S_ref = 16`, corresponding to the 4 by 4 stage4 token grid.
- `A = max(1, S / S_ref)` is the spatial precision divisor.
- For the residual `e = z_latent - z_mu`, where `z_latent` is the inferred predictive-coding latent and `z_mu` is the node prediction, the per-sample energy is:
  - `E = 0.5 * precision * sum(e**2) / A`.
- The explicit latent gradient is:
  - `dE/dz_latent = precision * e / A`.
- A stage4 token tensor with `S = 16` keeps the historical summed Gaussian energy because `A = 1`.
- A stage3 token tensor with `S = 64` divides by `4`, correcting for the larger 8 by 8 token grid.
- A pooled shell, shell bridge, outer-shell context, or column-pool tensor has `S = 1`, so `A = 1` and the route is not weakened by feature width.

Validation:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py columnar_cl_fabricpc/columns/normalized_gaussian.py columnar_cl_fabricpc/columns/__init__.py
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_stage3_96_bridge_best.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result:

- 54 passed in 13.66 seconds.

Recommended diagnostic run:

```bash
bash scripts/run_codex_stage3_96_bridge_best.sh 42 10
```

Interpretation target:

- Compare against the unnormalized stage3/96 run at 22.19% test accuracy.
- Compare against the full mean-normalized stage3/96 run at 18.89% test accuracy.
- Check whether `combined_without_column_pool`, `combined_without_column_shell_bridge`, and `combined_without_outer_shell_context` rise above chance.
- Check whether `combined_without_outer_shell` drops more strongly than it did in the unnormalized stage3/96 run. In that run it only dropped from 22.19% to 20.33%, which showed weak outer-shell participation.
- If validation improves past epoch 7 instead of declining, use the same wrapper for a 20-epoch run.

## 2026-07-14 Stage3/96 Spatial-Reference Gaussian Result

Recorded on 2026-07-14 13:21 EDT on `rogdora43`.

Run:

- Log: `results/codex_dspan_10c_3s_7a_shatt_cgstage3_e96_m32_gspref_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep10_composer_shells_rogdora43_20260714_044320.log`.
- Git commit recorded by the run: `5e544e48acb56b7f00df0a1651eed5329aeebad4`.
- Configuration: 10 columns, 3 shared columns, 7 active non-shared columns, `column_grid=stage3`, `embed_dim=96`, `microcolumn_dim=32`, `shell_attention`, `column_shell_bridge=on`, `outer_shell_context=on`, no bypass, `column_gaussian_energy_mode=spatial_reference`, `column_gaussian_precision=1.0`, and `column_gaussian_reference_sites=16`.

Main result:

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 18.34% |
| Best validation epoch | 8 |
| Test accuracy | 19.34% |

Validation trajectory:

| Epoch | Validation accuracy |
| ---: | ---: |
| 1 | 16.32% |
| 2 | 16.94% |
| 3 | 11.80% |
| 4 | 12.28% |
| 5 | 13.76% |
| 6 | 14.92% |
| 7 | 15.28% |
| 8 | 18.34% |
| 9 | 17.80% |
| 10 | 16.00% |

Test route diagnostics:

| Route or lesion | Test accuracy |
| --- | ---: |
| Combined classifier | 19.34% |
| Column pool only | 10.00% |
| Combined without column pool | 10.00% |
| Shell bridge only | 10.00% |
| Combined without shell bridge | 10.00% |
| Outer-shell context only | 10.00% |
| Combined without outer-shell context | 10.00% |
| Outer-shell context plus shell bridge | 10.00% |
| Combined without hard kernel | 10.02% |
| Combined without inner shell | 10.55% |
| Combined without middle shell | 17.05% |
| Combined without outer shell | 9.54% |

Shell state norms:

| Shell | Before training mean L2 | After training mean L2 |
| --- | ---: | ---: |
| Hard kernel | 5.7322 | 5.7423 |
| Inner shell | 3.3015 | 3.3158 |
| Middle shell | 4.5588 | 4.5797 |
| Outer shell | 5.5253 | 5.5624 |

Comparison:

| Run | Local Gaussian mode | Best validation accuracy | Test accuracy |
| --- | --- | ---: | ---: |
| Stage3/96 seed 42, 10 epochs | summed residual energy | 21.96% | 22.19% |
| Stage3/96 seed 42, 10 epochs | full latent mean residual energy | 18.32% | 18.89% |
| Stage3/96 seed 42, 10 epochs | spatial-reference residual energy | 18.34% | 19.34% |
| Stage4 seed 42, 20 epochs, best bridge/context anchor | summed residual energy | 34.30% | 33.93% |

Interpretation:

- Spatial-reference Gaussian scaling did not rescue the stage3/96 branch. It improved over full latent mean-normalization by 0.45 percentage points on test accuracy, but it remained 2.85 percentage points below the unnormalized stage3/96 run and 14.59 percentage points below the best stage4 bridge/context anchor.
- The failure is not a shell-magnitude collapse. All four shell state norms remained stable or increased slightly.
- The route diagnostics are more important than the top-line result. `combined` reached 19.34%, but every major route alone, every pairwise readout family, and every "combined without one route family" case was at chance.
- The shell lesion pattern says the weak stage3 signal depends on hard-kernel, inner-shell, and outer-shell participation together. Removing the middle shell preserved most of the weak signal, from 19.34% to 17.05%.
- The stage3 grid increases spatial capacity, but the added capacity is not becoming a stable class-relevant columnar code. The current stage3 path should not be extended to 20 epochs.

Recommendation:

- Stop the stage3/96 branch for now.
- Return to the stage4 10-column bridge/context anchor because it is the strongest no-bypass branch: seed 42 reached 33.93% test accuracy, seed 99 reached 33.62%, and seed 7 reached 29.91%.
- Continue from the earlier diagnostic finding that the useful stage4 signal is a joint readout interaction among `column_pool`, `columnXX_outer_shell_context`, and `columnXX_shell_bridge`.
- Do not re-enable direct full-strength `outer_shell_context_to_bridge`. That run dropped to 26.29% test accuracy and made `outer_shell_context_plus_column_shell_bridge` chance-level.
- Next implementation should add a controlled, fixed-scale outer-context-to-bridge path rather than a full-strength edge.

Proposed next implementation, pending confirmation:

- Add `--outer_shell_context_bridge_scale`, defaulting to `0.0`.
- When the scale is positive, route `columnXX_outer_shell_context` into `columnXX_shell_bridge` through a fixed scalar multiplier before the bridge receives it.
- Keep teacher heads off.
- Keep `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`, 10 columns, 3 shared columns, 7 active non-shared columns, `shell_attention`, `column_shell_bridge=on`, `outer_shell_context=on`, no bypass, summed Gaussian energy.
- Run a seed-42 scale sweep at `0.025`, `0.05`, and `0.10` for 20 epochs.
- Compare each scale against the best stage4 bridge/context anchor and the failed full-strength context-to-bridge run.

## 2026-07-14 Controlled Outer-Context-to-Bridge Implementation

Recorded on 2026-07-14 13:32 EDT on `rogdora43`.

Implemented after confirmation:

- Added `--outer_shell_context_bridge_scale`, defaulting to `0.0`.
- Kept the older `--outer_shell_context_to_bridge` flag intact for reproducing the failed full-strength edge experiment.
- Added validation so a positive `outer_shell_context_bridge_scale` requires both `--outer_shell_context` and `--column_shell_bridge`.
- Added validation so `--outer_shell_context_bridge_scale` cannot be combined with `--outer_shell_context_to_bridge`.
- Added `outer_shell_context_bridge_scale_node_name(column_idx)`, which names the per-column scaled bridge-conditioning latent as `columnXX_outer_shell_context_bridge_scale`.
- Updated the architecture diagram in `scripts/train_cifar10_depth_spanning.py` to show the optional scaled context latent before the shell bridge.
- Updated the shell-bridge masking helper so shell lesions trace through the scaled context latent and still mask the shell inputs to `columnXX_outer_shell_context`.
- Updated `scripts/run_codex_cifar10_depth_spanning.sh` with a new trailing positional argument for `outer_shell_context_bridge_scale`. Existing runner calls keep the default value `0.0`.
- Added `scripts/run_codex_outer_context_bridge_scale_sweep.sh`, which runs the stage4 10-column bridge/context anchor at fixed scales `0.025`, `0.05`, and `0.10` by default.

Mechanism:

- `columnXX_outer_shell_context` is the learned per-column context latent with width equal to the `outer_shell` shell.
- `columnXX_outer_shell_context_bridge_scale` is a FabricPC `IdentityNode` with no learned weights and fixed `scale = outer_shell_context_bridge_scale`.
- The scaled route is:
  - `columnXX_outer_shell_context -> columnXX_outer_shell_context_bridge_scale -> columnXX_shell_bridge`.
- `columnXX_shell_bridge` still receives the four pooled shell states directly.
- `columnXX_shell_bridge` still feeds the final CIFAR-10 classifier as before.
- The scaled latent has local Gaussian predictive-coding energy, so the path remains inside the predictive-coding graph instead of becoming a plain forward-only multiplier.
- When `outer_shell_context_bridge_scale = 0.0`, the graph omits the scaled latent and preserves the previous bridge/context anchor.

Validation:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_outer_context_bridge_scale_sweep.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py
```

Result:

- 59 passed in 13.85 seconds.

Recommended diagnostic run:

```bash
bash scripts/run_codex_outer_context_bridge_scale_sweep.sh
```

Interpretation target:

- Compare the three scaled runs against the best stage4 bridge/context anchor: seed 42, 20 epochs, 33.93% test accuracy.
- Compare the three scaled runs against the failed full-strength `outer_shell_context_to_bridge` result: 26.29% test accuracy.
- Check whether `outer_shell_context_plus_column_shell_bridge` rises above the previous no-conditioning diagnostic value of 18.46% without reducing the full `combined` accuracy.
- Check whether `combined_without_outer_shell` still drops strongly. In the best stage4 anchor, removing outer shell dropped test accuracy from 33.93% to 14.95%.
- If none of the fixed scales improves the stage4 anchor or the context-plus-bridge diagnostic, stop this context-to-bridge direction and return to shell-preserving readout or composer-side mechanisms.

## 2026-07-15 Controlled Outer-Context-to-Bridge Scale Sweep Result

Recorded on 2026-07-15 06:00 EDT on `rogdora43`.

Completed logs:

- Master log: `results/codex_outer_context_bridge_scale_sweep_seed42_rogdora43_20260714_133407.log`.
- Scale 0.025 child log: `results/codex_dspan_10c_3s_7a_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p025_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260714_133407.log`.
- Scale 0.05 child log: `results/codex_dspan_10c_3s_7a_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p05_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260714_181723.log`.
- Scale 0.10 child log: `results/codex_dspan_10c_3s_7a_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p10_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260714_230027.log`.

Configuration:

- Seed 42.
- 10 columns, 3 shared columns, and 7 active non-shared columns.
- `combiner=shell_attention`.
- `column_grid=stage4`, `embed_dim=64`, and `microcolumn_dim=32`.
- `outer_shell_context=on`, where `outer_shell_context` is a per-column predictive-coding latent whose width matches the column's `outer_shell` shell.
- `column_shell_bridge=on`, where `column_shell_bridge` is a per-column predictive-coding latent that receives the four pooled shell states and feeds the main CIFAR-10 classifier.
- `outer_shell_context_bridge_scale` is the fixed scalar multiplier on the route `columnXX_outer_shell_context -> columnXX_outer_shell_context_bridge_scale -> columnXX_shell_bridge`.
- No bypass path, no shell readout path, no teacher heads, no context evidence path, and no shell-context prediction objective.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values scale optimizer updates for hard-kernel, inner-shell, middle-shell, and outer-shell parameters.
- Learning rate 0.005 for 20 epochs.

Primary results:

| `outer_shell_context_bridge_scale` | Best validation accuracy | Best validation epoch | Test accuracy | Validation trajectory |
| ---: | ---: | ---: | ---: | --- |
| 0.025 | 30.74% | 19 | 30.28% | Improved late, then dipped slightly at epoch 20. |
| 0.05 | 29.84% | 20 | 29.13% | Improved late and selected epoch 20. |
| 0.10 | 30.70% | 20 | 30.08% | Improved late and selected epoch 20. |

Readout-family results on test:

| `outer_shell_context_bridge_scale` | Combined | `combined_without_column_pool` | `outer_shell_context_plus_column_shell_bridge` | `column_shell_bridge_only` | `outer_shell_context_only` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.025 | 30.28% | 13.35% | 13.35% | 10.00% | 10.00% |
| 0.05 | 29.13% | 10.81% | 10.81% | 10.00% | 10.00% |
| 0.10 | 30.08% | 15.37% | 15.37% | 10.00% | 10.00% |

Shell lesion results on test:

| `outer_shell_context_bridge_scale` | Combined | Without hard kernel | Without inner shell | Without middle shell | Without outer shell |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.025 | 30.28% | 17.62% | 20.91% | 18.39% | 18.19% |
| 0.05 | 29.13% | 16.71% | 21.33% | 19.76% | 16.36% |
| 0.10 | 30.08% | 18.48% | 22.37% | 18.82% | 22.86% |

Comparison to anchors:

| Run | Test accuracy | Notes |
| --- | ---: | --- |
| Best stage4 bridge/context anchor, seed 42 | 33.93% | `column_shell_bridge=on`, `outer_shell_context=on`, no context-to-bridge edge. |
| Three-seed stage4 bridge/context mean | 32.49% | Seeds 42, 99, and 7 from the bridge-only replicate. |
| Full-strength `outer_shell_context_to_bridge`, seed 42 | 26.29% | Raw context-to-bridge edge. |
| Best scaled context-to-bridge run | 30.28% | Fixed scale 0.025. |

Interpretation:

- Controlled scaling mitigated the full-strength context-to-bridge failure. The best scaled run reached 30.28% test accuracy, which is 3.99 percentage points above the 26.29% full-strength edge result.
- Controlled scaling did not improve the current stage4 anchor. The best scaled run remained 3.65 percentage points below the 33.93% seed-42 bridge/context anchor and 2.21 percentage points below the 32.49% three-seed bridge/context mean.
- The route diagnostic failed the intended mechanism test. `outer_shell_context_plus_column_shell_bridge` reached at most 15.37% test accuracy, below the previous no-conditioning diagnostic value of 18.46%.
- `column_shell_bridge_only` and `outer_shell_context_only` stayed at chance for all three scales. The scaled bridge path did not make either local route independently class-readable.
- Late validation improvement shows that this was not an early collapse failure. The weak result is a routing and representation issue inside the predictive-coding graph.
- Shell lesions still show that hard kernel, middle shell, and outer shell matter. The full classifier remains shell-dependent, but the scaled context-to-bridge route does not make the shell/context subgraph more useful.

Conclusion:

Stop the context-to-bridge conditioning branch for now. The controlled edge is less harmful than the full-strength edge, but it does not beat the unscaled stage4 bridge/context anchor and it weakens the specific context-plus-bridge route we meant to improve.

Recommended next step:

Return to the unscaled stage4 bridge/context anchor and move toward support-structured column specialization rather than adding more context-to-bridge edges.

The next experiment should use the existing static support machinery before adding a learned support controller. Run the best stage4 bridge/context branch with smaller non-shared supports, such as 3 shared columns plus 1, 3, or 5 non-shared columns, while keeping `column_shell_bridge=on`, `outer_shell_context=on`, `column_shell_readout=off`, no bypass, no teacher heads, `shell_attention`, and `shell_lr_multipliers=1,1.5,2,3`.

Reasoning:

- HiBaCaML emphasizes sparse support selection, while the current best branch uses all 10 columns on every example.
- The current composer already chooses one dominant column per shell at readout, but all columns still train on all examples. Static support sparsity is not the full paper mechanism, but it tests whether reduced column interference improves CIFAR-10 accuracy before we implement a learned support posterior.
- This experiment works from the best branch rather than continuing from the poorer scaled-context result.
- If one sparse support size improves or matches the 32.49% bridge/context mean, the next implementation should make support selection trainable or auditable. If all sparse support sizes fall below the anchor, the next implementation should instead target shell consolidation, because static support reduction would have shown that capacity was not the limiting source of interference.

## 2026-07-15 Static Support-Sparsity Sweep Implementation

Recorded on 2026-07-15 06:31 EDT on `rogdora43`.

Implemented after confirmation:

- Extended `scripts/run_codex_cifar10_depth_spanning.sh` with a trailing positional argument for `column_mode`.
- `column_mode` is the support-mask strategy passed to `scripts/train_cifar10_depth_spanning.py` through `--column_mode`.
- The valid `column_mode` values are `all_active`, `first_sparse`, and `random_sparse`.
- Existing wrapper calls keep `column_mode=all_active` when they omit the new trailing argument.
- The wrapper now records `column_mode` in the log header and includes a compact `cm...` label in new child log filenames.
- Added `scripts/run_codex_support_sparsity_10col3shared_sweep.sh`.

New sweep mechanism:

- The sweep keeps the strongest current no-bypass branch: stage4 column grid, 10 columns, 3 shared columns, `shell_attention`, `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, no teacher heads, no bypass path, and no outer-context-to-bridge edge.
- `num_shared=3` means columns 0, 1, and 2 are always active in sparse modes.
- `active_nonshared` is the number of additional non-shared columns that the support mask activates.
- The default support sweep uses `active_nonshared=1`, `3`, and `5`, corresponding to 4, 6, and 8 total active columns.
- The default `column_mode=first_sparse` selects the first N non-shared columns after the three shared columns. For example, `active_nonshared=3` activates columns 0, 1, 2, 3, 4, and 5.
- `COLUMN_MODE=random_sparse` is available as an environment override if we want a randomized static support control.

Prepared run command:

```bash
bash scripts/run_codex_support_sparsity_10col3shared_sweep.sh
```

Optional overrides:

```bash
SEEDS="42 99 7" ACTIVE_NONSHARED_VALUES="3" bash scripts/run_codex_support_sparsity_10col3shared_sweep.sh
```

```bash
COLUMN_MODE=random_sparse bash scripts/run_codex_support_sparsity_10col3shared_sweep.sh
```

Expected outputs:

- A master log named `results/codex_support_sparsity_10col3shared_<column_mode>_seeds<...>_active<...>_<host>_<timestamp>.log`.
- One child training log per seed and support size from `scripts/run_codex_cifar10_depth_spanning.sh`.

Interpretation plan:

- Compare each sparse support run against the unscaled stage4 bridge/context anchor: 33.93% seed-42 test accuracy and 32.49% three-seed mean test accuracy.
- Prioritize collapse resistance and main `combined` test accuracy.
- Check whether sparse support preserves the shell-lesion pattern from the anchor, where hard kernel, middle shell, and outer shell are all load-bearing.
- Check whether `outer_shell_context_plus_column_shell_bridge` rises above the scaled-context sweep's best value of 15.37% and the earlier no-conditioning diagnostic value of 18.46%.
- If a sparse support size improves or matches the anchor, the next implementation target should be an auditable support controller rather than more static support sweeps.
- If sparse supports all underperform the anchor, the next implementation target should shift toward shell consolidation or shell-promotion mechanisms inside each column.

Verification:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_support_sparsity_10col3shared_sweep.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result:

- 59 passed in 13.24 seconds.

```bash
git diff --check
```

Result:

- Passed.

## 2026-07-16 Static Support-Sparsity Sweep Results

Recorded on 2026-07-16 08:05 EDT on `rogdora43`.

Completed logs:

- First-sparse master log: `results/codex_support_sparsity_10col3shared_first_sparse_seeds42_active1_3_5_rogdora43_20260715_060041.log`.
- Random-sparse master log: `results/codex_support_sparsity_10col3shared_random_sparse_seeds42_active1_3_5_rogdora43_20260715_184731.log`.
- First-sparse child logs:
  - `results/codex_dspan_10c_3s_1a_cmfirst_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260715_060041.log`.
  - `results/codex_dspan_10c_3s_3a_cmfirst_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260715_100517.log`.
  - `results/codex_dspan_10c_3s_5a_cmfirst_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260715_142037.log`.
- Random-sparse child logs:
  - `results/codex_dspan_10c_3s_1a_cmrandom_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260715_184731.log`.
  - `results/codex_dspan_10c_3s_3a_cmrandom_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260715_225101.log`.
  - `results/codex_dspan_10c_3s_5a_cmrandom_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260716_030538.log`.

Configuration:

- Seed 42.
- `active_nonshared` is the number of non-shared columns activated in addition to the three shared columns.
- Total active columns equals `3 + active_nonshared`.
- `first_sparse` activates columns 0, 1, and 2 as shared columns, then activates the first N non-shared columns.
- `random_sparse` activates columns 0, 1, and 2 as shared columns, then samples N non-shared columns using the seed-42 pseudorandom permutation.
- All runs used the current stage4 bridge/context branch: 10 total columns, 3 shared columns, `shell_attention`, `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, no bypass path, no teacher heads, no context-to-bridge edge, and `shell_lr_multipliers=1,1.5,2,3`.

Primary results:

| `column_mode` | `active_nonshared` | Active columns | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | ---: | --- | ---: | ---: | ---: |
| `first_sparse` | 1 | `[0, 1, 2, 3]` | 24.98% | 20 | 24.41% |
| `first_sparse` | 3 | `[0, 1, 2, 3, 4, 5]` | 27.04% | 4 | 26.97% |
| `first_sparse` | 5 | `[0, 1, 2, 3, 4, 5, 6, 7]` | 26.38% | 6 | 27.50% |
| `random_sparse` | 1 | `[0, 1, 2, 7]` | 26.92% | 20 | 26.58% |
| `random_sparse` | 3 | `[0, 1, 2, 5, 7, 8]` | 26.28% | 20 | 26.82% |
| `random_sparse` | 5 | `[0, 1, 2, 5, 6, 7, 8, 9]` | 30.42% | 20 | 30.11% |

Readout-family results on test:

| `column_mode` | `active_nonshared` | Combined | `combined_without_column_pool` | `outer_shell_context_plus_column_shell_bridge` | `column_pool_plus_shell_bridge` | `combined_without_column_shell_bridge` | `combined_without_outer_shell_context` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_sparse` | 1 | 24.41% | 10.29% | 10.29% | 10.00% | 10.03% | 10.00% |
| `first_sparse` | 3 | 26.97% | 17.33% | 17.33% | 10.00% | 10.00% | 10.00% |
| `first_sparse` | 5 | 27.50% | 14.55% | 14.55% | 13.28% | 10.00% | 13.28% |
| `random_sparse` | 1 | 26.58% | 10.00% | 10.00% | 10.00% | 15.46% | 10.00% |
| `random_sparse` | 3 | 26.82% | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |
| `random_sparse` | 5 | 30.11% | 10.00% | 10.00% | 10.00% | 10.00% | 10.00% |

Shell lesion results on test:

| `column_mode` | `active_nonshared` | Combined | Without hard kernel | Without inner shell | Without middle shell | Without outer shell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `first_sparse` | 1 | 24.41% | 10.00% | 16.03% | 14.56% | 10.03% |
| `first_sparse` | 3 | 26.97% | 20.24% | 18.17% | 20.62% | 21.78% |
| `first_sparse` | 5 | 27.50% | 18.61% | 20.31% | 17.10% | 22.12% |
| `random_sparse` | 1 | 26.58% | 10.00% | 17.99% | 13.75% | 13.39% |
| `random_sparse` | 3 | 26.82% | 10.04% | 15.39% | 10.61% | 10.00% |
| `random_sparse` | 5 | 30.11% | 10.24% | 21.01% | 18.14% | 18.07% |

Shell state norms after training:

| `column_mode` | `active_nonshared` | Hard-kernel mean L2 | Inner-shell mean L2 | Middle-shell mean L2 | Outer-shell mean L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `first_sparse` | 1 | 4.6897 | 2.6446 | 3.7408 | 4.5815 |
| `first_sparse` | 3 | 4.6898 | 2.6450 | 3.7401 | 4.5813 |
| `first_sparse` | 5 | 4.6896 | 2.6446 | 3.7390 | 4.5809 |
| `random_sparse` | 1 | 4.6896 | 2.6447 | 3.7395 | 4.5774 |
| `random_sparse` | 3 | 4.6893 | 2.6436 | 3.7391 | 4.5790 |
| `random_sparse` | 5 | 4.6887 | 2.6428 | 3.7396 | 4.5790 |

Comparison to anchors:

| Run | Test accuracy | Notes |
| --- | ---: | --- |
| Best stage4 bridge/context anchor, seed 42 | 33.93% | All 10 columns active. |
| Three-seed stage4 bridge/context mean | 32.49% | Seeds 42, 99, and 7. |
| Best controlled context-to-bridge scale run | 30.28% | Scale 0.025. |
| Best static sparse support run | 30.11% | `random_sparse`, `active_nonshared=5`, 8 total active columns. |

Interpretation:

- Static support sparsity did not recover the stage4 bridge/context anchor. The best sparse run reached 30.11% test accuracy, 3.82 percentage points below the 33.93% seed-42 anchor and 2.38 percentage points below the 32.49% three-seed anchor mean.
- Static support sparsity did not strengthen the context-plus-bridge route. `outer_shell_context_plus_column_shell_bridge` reached 17.33% only in the deterministic six-column support, below the earlier no-conditioning diagnostic value of 18.46%. It was at chance in all random-sparse runs.
- The random eight-column run improved through epoch 20 and produced the best sparse result, but all readout-family removals were at chance. Its useful signal was even more dependent on the full combined readout than the anchor.
- The sparse runs did not collapse by shell magnitude. Hard-kernel, inner-shell, middle-shell, and outer-shell state norms stayed close to the prior stage4 values in every run.
- The random eight-column run made the hard kernel especially load-bearing. Removing the hard kernel dropped test accuracy from 30.11% to 10.24%, while removing inner shell, middle shell, or outer shell left 21.01%, 18.14%, and 18.07% respectively.
- Deterministic sparse supports with 6 or 8 active columns peaked early, at epochs 4 and 6. Random sparse supports selected epoch 20 in all three runs. This suggests support composition affects training stability, but the effect is not sufficient to beat the all-active anchor.

Conclusion:

Do not continue static support-size sweeps as the next main path. The sparse-support idea is architecturally relevant to HiBaCaML, but this static version did not improve CIFAR-10 accuracy or route interpretability. It mostly reduced capacity while leaving the same full-readout dependency.

Recommended next step:

Move to shell consolidation or shell promotion inside each active column, starting from the all-active stage4 bridge/context anchor rather than from the sparse-support result.

Proposed mechanism, pending confirmation:

- Add a small inward shell-promotion predictive objective inside `DepthSpanningColumnNode`.
- The promotion path should move reusable outer-shell evidence inward through learned local predictors, not through a classifier head.
- The first implementation should be conservative:
  - Predict `middle_shell` from `outer_shell` inside each column.
  - Predict `inner_shell` from `middle_shell` inside each column.
  - Keep `hard_kernel` untouched in the first pass to avoid destabilizing the strongest current class-bearing path.
  - Weight the promotion energy very lightly, such as `0.0001` or `0.00025`.
  - Keep the main graph at the stage4 bridge/context anchor: all 10 columns active, `outer_shell_context=on`, `column_shell_bridge=on`, no shell readout, no bypass, no teacher heads, and no context-to-bridge edge.
- The diagnostic should compare no promotion against one or two light promotion weights on seed 42 before any three-seed replicate.

Reasoning:

- The static-support experiment says column count and support composition are not the immediate limiting factor.
- The repeated route-ablation failure says local context and bridge latents are still not independently useful.
- The shell-lesion pattern says shell identity matters, but the model often depends on the full classifier to combine shell evidence.
- A local inward promotion objective is closer to the HiBaCaML consolidation idea than more readout edges, because it gives shell-to-shell structure a learning signal inside each column without adding another class-label head.

## 2026-07-16 Random Support Density Follow-Up Results

Recorded on 2026-07-16 20:32 EDT on `rogdora43`.

This section supersedes the immediate next-step recommendation from the previous static support section. The user correctly pointed out that a support branch should not be dropped merely because it did not beat the previous best result. The follow-up run fills the high-density random-support cases and provides a current-code all-column control.

Completed logs:

- Master log: `results/codex_support_sparsity_10col3shared_random_sparse_seeds42_active6_7_rogdora43_20260716_091144.log`.
- `active_nonshared=6` child log: `results/codex_dspan_10c_3s_6a_cmrandom_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260716_091144.log`.
- `active_nonshared=7` child log: `results/codex_dspan_10c_3s_7a_cmrandom_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260716_135817.log`.

Configuration:

- Seed 42.
- `column_mode=random_sparse`.
- `active_nonshared=6` means three shared columns plus six of the seven non-shared columns, for nine total active columns.
- `active_nonshared=7` means three shared columns plus all seven non-shared columns, for ten total active columns. Its support mask is therefore the same as `all_active`, even though the logged mode is `random_sparse`.
- Both runs used the stage4 bridge/context branch: 10 total columns, 3 shared columns, `shell_attention`, `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, no bypass path, no teacher heads, no context-to-bridge edge, and `shell_lr_multipliers=1,1.5,2,3`.

Primary results:

| `active_nonshared` | Active columns | Best validation accuracy | Best validation epoch | Test accuracy | Validation trajectory |
| ---: | --- | ---: | ---: | ---: | --- |
| 6 | `[0, 1, 2, 3, 5, 6, 7, 8, 9]` | 31.56% | 20 | 30.48% | Improved through epoch 20. |
| 7 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | 32.98% | 20 | 32.47% | Improved through epoch 20. |

Readout-family results on test:

| `active_nonshared` | Combined | `combined_without_column_pool` | `outer_shell_context_plus_column_shell_bridge` | `column_pool_plus_shell_bridge` | `combined_without_column_shell_bridge` | `combined_without_outer_shell_context` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 30.48% | 16.06% | 16.06% | 10.00% | 10.00% | 10.00% |
| 7 | 32.47% | 18.64% | 18.64% | 10.00% | 10.00% | 10.00% |

Shell lesion results on test:

| `active_nonshared` | Combined | Without hard kernel | Without inner shell | Without middle shell | Without outer shell |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 30.48% | 19.10% | 22.98% | 16.53% | 14.45% |
| 7 | 32.47% | 21.09% | 24.31% | 21.02% | 17.95% |

Shell state norms after training:

| `active_nonshared` | Hard-kernel mean L2 | Inner-shell mean L2 | Middle-shell mean L2 | Outer-shell mean L2 |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 4.6892 | 2.6445 | 3.7389 | 4.5793 |
| 7 | 4.6884 | 2.6430 | 3.7394 | 4.5795 |

Composer attention at the selected checkpoint:

| `active_nonshared` | Hard-kernel owner | Inner-shell owner | Middle-shell owner | Outer-shell owner |
| ---: | --- | --- | --- | --- |
| 6 | `col_01` at 0.995160 | `col_07` at 0.998983 | `col_03` at 0.999242 | `col_06` at 0.999657 |
| 7 | `col_09` at 0.997870 | `col_00` at 0.997463 | `col_05` at 0.999196 | `col_00` at 0.999712 |

Comparison to anchors:

| Run | Test accuracy | Notes |
| --- | ---: | --- |
| Best stage4 bridge/context anchor, seed 42 | 33.93% | Earlier all-active high result. |
| Three-seed stage4 bridge/context mean | 32.49% | Seeds 42, 99, and 7 from the earlier bridge/context replicate. |
| Random sparse, `active_nonshared=5` | 30.11% | Best result from the previous sparse sweep. |
| Random sparse, `active_nonshared=6` | 30.48% | Nine total active columns. |
| Random sparse, `active_nonshared=7` | 32.47% | Ten total active columns; current-code all-column control. |

Interpretation:

- The support-density follow-up makes the support branch more important, not less. Accuracy increased as support density approached all columns: 30.11% at eight active columns, 30.48% at nine active columns, and 32.47% at ten active columns.
- The ten-column current-code control did not reproduce the old 33.93% seed-42 high, but it essentially matched the earlier three-seed mean of 32.49%. This reduces concern that recent code changes broke the branch.
- `outer_shell_context_plus_column_shell_bridge` rose to 18.64% in the ten-column run. That slightly exceeds the previous no-conditioning diagnostic value of 18.46%.
- The route remains a full-readout interaction. `column_pool_plus_shell_bridge`, `combined_without_column_shell_bridge`, and `combined_without_outer_shell_context` stayed at chance in both runs.
- The shell norms again show no magnitude collapse.
- The composer continues to choose sparse shell owners even when all columns are active. In the ten-column run, `col_00` owned both inner shell and outer shell, `col_05` owned middle shell, and `col_09` owned hard kernel.

Conclusion:

Support should remain an active research branch. The result does not show that smaller static supports are better than all columns, but it does show that support composition and support density shape training stability and route use. The correct next support step is not another scalar support-size sweep. It is an auditable support-mask experiment that can test specific column subsets without coupling the support mask to the training seed.

Recommended next step:

Add explicit support-mask control, then run a leave-one-non-shared-column-out support audit from the stage4 bridge/context branch.

Proposed mechanism, pending confirmation:

- Add a `--support_mask` CLI argument to `scripts/train_cifar10_depth_spanning.py`.
- `support_mask` should be a comma-separated list of 10 binary values, where value 1 activates the corresponding column and value 0 excludes it from the support mask.
- When `--support_mask` is provided, it should override `--column_mode`, `--num_shared`, and `--active_nonshared` for mask construction.
- Update `scripts/run_codex_cifar10_depth_spanning.sh` with a trailing positional argument for `support_mask`.
- Add a runner for a leave-one-non-shared-column-out audit:
  - all columns active: `1,1,1,1,1,1,1,1,1,1`;
  - drop column 3: `1,1,1,0,1,1,1,1,1,1`;
  - drop column 4: `1,1,1,1,0,1,1,1,1,1`;
  - drop column 5: `1,1,1,1,1,0,1,1,1,1`;
  - drop column 6: `1,1,1,1,1,1,0,1,1,1`;
  - drop column 7: `1,1,1,1,1,1,1,0,1,1`;
  - drop column 8: `1,1,1,1,1,1,1,1,0,1`;
  - drop column 9: `1,1,1,1,1,1,1,1,1,0`.
- Keep the graph otherwise unchanged: stage4 column grid, `shell_attention`, `outer_shell_context=on`, `column_shell_bridge=on`, no shell readout, no bypass, no teacher heads, no context-to-bridge edge, and `shell_lr_multipliers=1,1.5,2,3`.

Reasoning:

- `active_nonshared=6` already tested one leave-one-out mask by excluding column 4. That run reached 30.48%, so column 4 appears useful in that specific comparison.
- The ten-column run's composer used columns 0, 5, and 9 strongly. A leave-one-out audit can test whether the unused columns are neutral, harmful, or necessary through indirect predictive-coding routes.
- Explicit masks are closer to a support controller than the current `first_sparse` and `random_sparse` modes, because the experiment can test named supports and later compare them with learned or audited support proposals.
- If one leave-one-out support beats the all-column control, support auditing becomes the next implementation target. If all leave-one-out supports underperform, then shell promotion remains the next best mechanism.

## 2026-07-16 Explicit Support-Mask Audit Implementation

Recorded on 2026-07-16 21:04 EDT on `rogdora43`.

Implemented after confirmation:

- Added `parse_support_mask(value, num_columns)` to `scripts/train_cifar10_depth_spanning.py`.
- Added `--support_mask` to `scripts/train_cifar10_depth_spanning.py`.
- `support_mask` is a comma-separated list with one binary value per column.
- Value 1 activates the column, and value 0 excludes it from the support mask.
- When `--support_mask` is provided, it overrides `--column_mode`, `--num_shared`, and `--active_nonshared` for support-mask construction.
- The explicit support mask flows through the existing shared `build_support_mask()` path.
- The same mask is used by the combiner, active-column per-shell pools, per-column shell bridges, per-column outer-shell context latents, and active-column diagnostics.
- Added a trailing positional argument 30 to `scripts/run_codex_cifar10_depth_spanning.sh` for `support_mask`.
- The wrapper now records `support_mask` in the log header and includes a compact support-mask label in new child log filenames.
- Added `scripts/run_codex_support_leave_one_out_10col3shared.sh`.

Prepared audit:

- The audit keeps the stage4 bridge/context branch unchanged except for the support mask.
- Shared configuration: seed 42, 10 total columns, 3 shared columns, `shell_attention`, `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, no bypass path, no teacher heads, no context-to-bridge edge, `shell_lr_multipliers=1,1.5,2,3`, learning rate 0.005, and 20 epochs.
- The default audit runs these masks:

| Label | `support_mask` | Meaning |
| --- | --- | --- |
| `all` | `1,1,1,1,1,1,1,1,1,1` | Current-code all-column control. |
| `drop_col03` | `1,1,1,0,1,1,1,1,1,1` | Excludes non-shared column 3. |
| `drop_col04` | `1,1,1,1,0,1,1,1,1,1` | Excludes non-shared column 4. |
| `drop_col05` | `1,1,1,1,1,0,1,1,1,1` | Excludes non-shared column 5. |
| `drop_col06` | `1,1,1,1,1,1,0,1,1,1` | Excludes non-shared column 6. |
| `drop_col07` | `1,1,1,1,1,1,1,0,1,1` | Excludes non-shared column 7. |
| `drop_col08` | `1,1,1,1,1,1,1,1,0,1` | Excludes non-shared column 8. |
| `drop_col09` | `1,1,1,1,1,1,1,1,1,0` | Excludes non-shared column 9. |

Run command:

```bash
bash scripts/run_codex_support_leave_one_out_10col3shared.sh
```

## 2026-07-18 Full Leave-One-Out Support Audit Result

Recorded on 2026-07-18 after the overnight run completed on `rogdora43`.

Run inspected:

- Master log: `results/codex_support_leave_one_out_10col3shared_seed42_rogdora43_20260717_134042.log`.
- Commit recorded by the master log: `38e08fdbe0ab9ae2ab315bda1294e9149b5cad22`.
- Started at `2026-07-17 13:40:42 EDT`.
- Completed at `2026-07-18 06:11:53 EDT`.

Configuration:

- Seed 42, learning rate 0.005, and 20 epochs.
- 10 columns, 3 shared columns, 7 active non-shared columns, and `column_mode=all_active`.
- `support_mask` is the binary vector that selects which of the 10 columns participate in the predictive coding graph for the run. A value of 1 means the column is active. A value of 0 means the column is omitted.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, and `microcolumn_dim=32`.
- `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, `outer_shell_context_to_bridge=off`, and no backbone bypass.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values apply to `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`.
- `diagnose_mode=nodiag`.
- `post_training_diagnostics=core`.

Result:

| Mask label | `support_mask` | Best validation accuracy | Best validation epoch | Test accuracy | Test delta from all columns |
| --- | --- | ---: | ---: | ---: | ---: |
| `all` | `1,1,1,1,1,1,1,1,1,1` | 32.60% | 18 | 31.93% | baseline |
| `drop_col03` | `1,1,1,0,1,1,1,1,1,1` | 26.46% | 5 | 26.61% | -5.32 percentage points |
| `drop_col04` | `1,1,1,1,0,1,1,1,1,1` | 28.78% | 9 | 28.23% | -3.70 percentage points |
| `drop_col05` | `1,1,1,1,1,0,1,1,1,1` | 26.46% | 5 | 26.13% | -5.80 percentage points |
| `drop_col06` | `1,1,1,1,1,1,0,1,1,1` | 29.32% | 18 | 29.02% | -2.91 percentage points |
| `drop_col07` | `1,1,1,1,1,1,1,0,1,1` | 28.56% | 16 | 28.68% | -3.25 percentage points |
| `drop_col08` | `1,1,1,1,1,1,1,1,0,1` | 27.72% | 9 | 27.06% | -4.87 percentage points |
| `drop_col09` | `1,1,1,1,1,1,1,1,1,0` | 28.62% | 20 | 27.89% | -4.04 percentage points |

Infrastructure conclusion:

- The full support audit completed all eight child runs.
- Each child run printed `Results Summary`, `Test Accuracy`, `Best Val Accuracy`, and `Best Val Epoch`.
- The `post_training_diagnostics=core` setting solved the immediate measurement issue by recording the selected checkpoint metrics before any optional detailed diagnostics.

Model interpretation:

- The all-column control was the best result within this audit.
- Every leave-one-out support mask reduced test accuracy relative to the all-column control.
- Dropping column 6 was the least harmful omission in this run, with test accuracy falling from 31.93% to 29.02%.
- Dropping column 5 was the most harmful omission in this run, with test accuracy falling from 31.93% to 26.13%.
- The earlier two-mask recovery run found `drop_col04` nearly tied with the all-column control, but this full audit found `drop_col04` 3.70 percentage points below the all-column control. This difference shows that the single-seed support-mask ranking is noisy.
- These results do not support pruning columns or adding a static support-selection rule as the next accuracy step.

Recommended next step:

- Keep all 10 columns active for the next architectural change.
- Move the next experiment back inside each column rather than changing which columns participate.
- Add a graph-native inward shell-promotion objective. For each active column, the `outer_shell` state should predict the `middle_shell` state, the `middle_shell` state should predict the `inner_shell` state, and the `inner_shell` state should predict the `hard_kernel` state.
- This keeps the experiment within predictive coding dynamics because each promotion pathway is a prediction edge inside the graph, not a separate backpropagation classifier.
- The intended mechanism is to make each wider shell explain the more central shell in the same column, so deeper shell states are trained to consolidate information inward while the class readout still uses the existing shell-preserving column representation.
- The first test should keep the strongest current all-column settings: 10 columns, 3 shared columns, all columns active, `combiner=shell_attention`, stage4 column bridge, outer shell context enabled, column shell bridge enabled, no bypass, no teacher heads, and shell learning-rate multipliers `1,1.5,2,3`.

Optional targeted subset syntax:

```bash
bash scripts/run_codex_support_leave_one_out_10col3shared.sh all:1,1,1,1,1,1,1,1,1,1 drop_col04:1,1,1,1,0,1,1,1,1,1
```

Expected outputs:

- Master log: `results/codex_support_leave_one_out_10col3shared_seed42_<host>_<timestamp>.log`.
- One child training log per support mask from `scripts/run_codex_cifar10_depth_spanning.sh`.

Interpretation plan:

- Compare every leave-one-out mask against the current-code all-column control from the same script.
- Also compare against the previous random `active_nonshared=7` current-code control: 32.47% test accuracy.
- Track `outer_shell_context_plus_column_shell_bridge`; the previous all-column current-code run reached 18.64%.
- Track shell lesions. In the previous all-column current-code run, removing hard kernel, inner shell, middle shell, and outer shell left 21.09%, 24.31%, 21.02%, and 17.95%.
- If one leave-one-out mask beats the all-column control, the next implementation should move toward an auditable support controller.
- If all leave-one-out masks underperform the all-column control, the support branch still remains useful as a diagnostic, but the next implementation should shift to shell promotion inside columns.

Verification:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_support_leave_one_out_10col3shared.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --help | rg -n "support_mask|column_mode"
```

Result:

- `--support_mask` appears in the CLI help beside `--column_mode`.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result:

- 61 passed in 13.63 seconds.

```bash
git diff --check
```

Result:

- Passed.

## 2026-07-17 Explicit Support-Mask Audit Run Did Not Complete

Recorded on 2026-07-17 06:06 EDT on `rogdora43`.

Run inspected:

- Master log: `results/codex_support_leave_one_out_10col3shared_seed42_rogdora43_20260716_204115.log`.
- Child log for the first mask: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_composer_shells_rogdora43_20260716_204115.log`.
- Commit recorded by the master log: `333f6796874eef09e1dab5f0be0acf063b37d19d`.
- `support_mask = 1,1,1,1,1,1,1,1,1,1`, where each value is the binary active-column indicator for one of the 10 columns.

Configuration of the child run:

- 10 columns, 3 shared columns, 7 active non-shared columns, `column_mode=all_active`, and explicit all-column `support_mask`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, and `microcolumn_dim=32`.
- `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, `outer_shell_context_to_bridge=off`, and no backbone bypass.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values apply to `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`.
- `diagnose_mode=composer_shells`.

Observed outcome:

- The audit did not finish the all-column control and did not start the leave-one-out masks.
- The child process reached the end of the 20-epoch training loop, with the progress bar reporting epoch 20 complete and training energy around `0.0035`.
- The process was then killed during a GPU JIT compilation step before `Training time`, `Results Summary`, `Test Accuracy`, `Best Val Accuracy`, or `Best Val Epoch` were written.
- No `val_acc` lines are present in either the master log or the child log.
- The run therefore provides no usable CIFAR-10 classification metric.

Conclusion:

- This is an evaluation/logging infrastructure failure, not a model-quality result.
- The explicit support-mask implementation itself was exercised far enough to build the graph with all 10 columns active: 147 nodes, 273 edges, and 3,125,444 parameters.
- The all-column mask cannot yet be compared with the previous 10-column random-support result or with the planned leave-one-out masks.

Recommended next step:

- Change the shared training script so core metrics are durable before expensive diagnostics.
- Mechanism: compute and print the selected-parameter test accuracy, best validation accuracy, and best validation epoch immediately after training and before composer lesions, readout lesions, shell lesions, teacher-head evaluations, and path ablations.
- Add an explicit post-training diagnostic level if needed, where the main metric pass is always run and the expensive ablation suite is opt-in for long sweeps.
- Then rerun the support-mask audit with a lighter default diagnostic setting. The first rerun should include only the all-column control and one leave-one-out mask to verify that the script writes complete metrics before committing another overnight run.

## 2026-07-17 Durable Core Metrics Implementation

Recorded on 2026-07-17 06:17 EDT on `rogdora43`.

Implemented after confirmation:

- Added `--post_training_diagnostics` to `scripts/train_cifar10_depth_spanning.py`.
- `post_training_diagnostics = core` means the script evaluates the selected checkpoint on the test loader, prints `Results Summary`, and returns before after-training diagnostics.
- `post_training_diagnostics = full` keeps the new early `Results Summary` and then runs the expensive composer, shell, teacher-head, readout, bridge, context, and path ablation diagnostics.
- The selected checkpoint is `best_params` when validation evaluation has selected one; otherwise it is `final_params`.
- The core metric path calls FabricPC `evaluate_pcn(eval_params, structure, test_loader, train_config, summary_key)`, where `eval_params` is the selected parameter tree and `summary_key` is the deterministic JAX pseudo-random key derived from the experiment seed.
- The script now prints `Training time`, `Test Accuracy`, `Best Val Accuracy`, and `Best Val Epoch` before any post-training shell norms, composer diagnostics, or ablation evaluations can start.
- Epoch validation lines now use `flush=True`, so `val_acc` entries are more likely to survive if a later post-training step is killed.
- The previous full diagnostic output is now labeled `Detailed Test Diagnostics` so it is distinct from the early durable `Results Summary`.

Runner changes:

- Added trailing positional argument 31 to `scripts/run_codex_cifar10_depth_spanning.sh`.
- Argument 31 is `post_training_diagnostics`, with valid values `core` and `full`.
- Existing calls keep the default `full`.
- The wrapper records `post_training_diagnostics` in each child log header.
- Updated `scripts/run_codex_support_leave_one_out_10col3shared.sh` to default to `DIAGNOSE_MODE=nodiag`.
- Updated `scripts/run_codex_support_leave_one_out_10col3shared.sh` to default to `POST_TRAINING_DIAGNOSTICS=core`.
- The support-mask audit still accepts `DIAGNOSE_MODE` and `POST_TRAINING_DIAGNOSTICS` environment overrides for targeted full diagnostic runs.

Recommended recovery run:

```bash
bash scripts/run_codex_support_leave_one_out_10col3shared.sh all:1,1,1,1,1,1,1,1,1,1 drop_col04:1,1,1,1,0,1,1,1,1,1
```

Interpretation target:

- Confirm that both child logs print `Results Summary`, `Test Accuracy`, `Best Val Accuracy`, and `Best Val Epoch`.
- Confirm that the master log reaches `completed_at`.
- If this two-mask recovery run completes, run the full default leave-one-out audit:

```bash
bash scripts/run_codex_support_leave_one_out_10col3shared.sh
```

If a specific support mask later looks promising, run diagnostics on that mask only:

```bash
DIAGNOSE_MODE=composer_shells POST_TRAINING_DIAGNOSTICS=full bash scripts/run_codex_support_leave_one_out_10col3shared.sh all:1,1,1,1,1,1,1,1,1,1
```

Verification:

```bash
bash -n scripts/run_codex_cifar10_depth_spanning.sh
```

Result:

- Passed.

```bash
bash -n scripts/run_codex_support_leave_one_out_10col3shared.sh
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py
```

Result:

- Passed.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --help | rg -n "post_training_diagnostics|support_mask|column_mode"
```

Result:

- `--post_training_diagnostics`, `--support_mask`, and `--column_mode` appear in the CLI help.

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q
```

Result:

- 61 passed in 13.67 seconds.

```bash
git diff --check
```

Result:

- Passed.

## 2026-07-17 Two-Mask Support Audit Recovery Result

Recorded on 2026-07-17 13:35 EDT on `rogdora43`.

Run inspected:

- Master log: `results/codex_support_leave_one_out_10col3shared_seed42_rogdora43_20260717_062131.log`.
- All-column child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260717_062131.log`.
- `drop_col04` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111011111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260717_082358.log`.
- Commit recorded by the master log: `38e08fdbe0ab9ae2ab315bda1294e9149b5cad22`.

Configuration:

- Seed 42, learning rate 0.005, 20 epochs.
- 10 columns, 3 shared columns, 7 active non-shared columns, `column_mode=all_active`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, and `microcolumn_dim=32`.
- `outer_shell_context=on`, `column_shell_bridge=on`, `column_shell_readout=off`, `outer_shell_context_to_bridge=off`, and no backbone bypass.
- `shell_lr_multipliers=1,1.5,2,3`, where the four values apply to `hard_kernel`, `inner_shell`, `middle_shell`, and `outer_shell`.
- `diagnose_mode=nodiag`.
- `post_training_diagnostics=core`.

Result:

| Mask label | `support_mask` | Best validation accuracy | Best validation epoch | Test accuracy |
| --- | --- | ---: | ---: | ---: |
| `all` | `1,1,1,1,1,1,1,1,1,1` | 33.66% | 18 | 33.40% |
| `drop_col04` | `1,1,1,1,0,1,1,1,1,1` | 33.92% | 18 | 33.41% |

Validation trajectory:

| Epoch | `all` validation accuracy | `drop_col04` validation accuracy |
| ---: | ---: | ---: |
| 1 | 11.34% | 11.14% |
| 2 | 21.76% | 21.30% |
| 3 | 20.30% | 22.16% |
| 4 | 24.56% | 19.74% |
| 5 | 25.88% | 25.84% |
| 6 | 24.12% | 24.70% |
| 7 | 25.88% | 24.66% |
| 8 | 27.14% | 27.90% |
| 9 | 30.18% | 22.74% |
| 10 | 23.86% | 28.16% |
| 11 | 22.70% | 23.36% |
| 12 | 27.18% | 27.32% |
| 13 | 25.80% | 28.48% |
| 14 | 28.06% | 26.98% |
| 15 | 26.82% | 27.18% |
| 16 | 29.22% | 28.50% |
| 17 | 29.14% | 30.12% |
| 18 | 33.66% | 33.92% |
| 19 | 30.80% | 31.76% |
| 20 | 31.38% | 31.90% |

Infrastructure conclusion:

- The durable core metric path works for this recovery run.
- Both child logs printed `Results Summary`, `Test Accuracy`, `Best Val Accuracy`, and `Best Val Epoch`.
- The master log reached `completed_at`.
- This fixes the immediate measurement failure from the prior support-mask audit.

Model interpretation:

- Dropping column 4 did not materially change test accuracy in seed 42.
- The all-column and `drop_col04` results differ by 0.01 percentage points on test accuracy.
- The best validation score is slightly higher for `drop_col04`, but the difference is small enough that it should be treated as support neutrality until the remaining leave-one-out masks are tested.
- The two completed masks are slightly above the previous 10-column random-support current-code control of 32.47% test accuracy, but this comparison is not definitive because the recovery run used the all-column support construction and only one seed.

Recommended next step:

- Run the full default leave-one-out support audit now that core metrics survive.
- Keep `DIAGNOSE_MODE=nodiag` and `POST_TRAINING_DIAGNOSTICS=core` for this audit.
- After the full audit identifies any promising or harmful masks, run full diagnostics only for the all-column control and the most informative leave-one-out mask.

Run command:

```bash
bash scripts/run_codex_support_leave_one_out_10col3shared.sh
```

## 2026-07-18 Inward Shell-Promotion Implementation

Recorded on 2026-07-18 on `rogdora43`.

Implementation goal:

- Keep the 10-column stage4 bridge/context architecture active.
- Add a HiBaCaML-aligned consolidation pressure inside each active column.
- Keep the objective predictive-coding native by adding Gaussian prediction nodes to the graph, rather than adding class-supervised heads or plain backpropagation losses.

Mechanism:

- `inward_shell_promotion_weight` is the scalar multiplier on the new local Gaussian objective.
- `source_shell` is the wider shell whose pooled latent vector provides the prediction context.
- `target_shell` is the adjacent more central shell whose pooled latent vector is predicted.
- For each active column, the graph now adds three shell-promotion prediction nodes when `inward_shell_promotion_weight > 0`:
  - `outer_shell -> middle_shell`.
  - `middle_shell -> inner_shell`.
  - `inner_shell -> hard_kernel`.
- Each prediction node receives the `target_shell` pooled vector through its `target` slot and the `source_shell` pooled vector through its `context` slot.
- Its energy is `0.5 * inward_shell_promotion_weight * sum((target_shell - predicted_target_shell)^2)`, where `predicted_target_shell` is a learned linear projection of the wider shell vector.
- The node is terminal with respect to the class readout. It contributes predictive-coding energy and gradients to the shell states and projection weights, but it does not connect to `output`.
- Optimizer update scaling uses the target shell. For example, the `middle_shell -> inner_shell` promotion weights use the `inner_shell` multiplier from `shell_lr_multipliers`.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`:
  - Added `--inward_shell_promotion_weight`.
  - Added `INWARD_SHELL_PROMOTION_PAIRS`.
  - Added `inward_shell_promotion_node_name()`, `is_inward_shell_promotion_node()`, `inward_shell_promotion_pair()`, and `inward_shell_promotion_target_shell()`.
  - Added graph construction for the three inward promotion objectives per active column.
  - Added inward promotion nodes to energy categorization and column-output diagnostics.
  - Added `diagnose_inward_shell_promotion_energies()`.
  - Added diagnostic printing for inward promotion energy when `--diagnose_shells` is enabled.
  - Updated the top-level architecture diagram to include the inward shell-promotion path.
- `columnar_cl_fabricpc/columns/accuracy_nodes.py`:
  - Updated `ShellContextPredictionNode` documentation so it covers both the existing outer-context prediction objective and the new wider-shell-to-inner-shell promotion objective.
- `scripts/run_codex_cifar10_depth_spanning.sh`:
  - Added positional argument 32 for `inward_shell_promotion_weight`.
  - Added compact `isp...` labeling in child result filenames.
  - Added header logging and Python CLI forwarding for `--inward_shell_promotion_weight`.
- `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`:
  - Added a sequential sweep script for the current all-column stage4 bridge/context branch.
  - Default weights are `0.0 0.00025 0.0005 0.001`.
  - Default diagnostics are `nodiag` and `post_training_diagnostics=core`.
- `tests/test_pooled_readout_norm.py`:
  - Added graph-construction coverage for inward promotion nodes and their target/context edges.
  - Added optimizer multiplier coverage showing promotion parameters use the target shell learning-rate multiplier.

Verification:

- `bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`: passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py columnar_cl_fabricpc/columns/accuracy_nodes.py`: passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --help | rg -n "inward_shell_promotion|outer_shell_context_shell_prediction|post_training"`: the new CLI flag appears.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q`: 63 passed in 14.70 seconds.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py`: 185 passed in 31.38 seconds.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q`: 193 passed, 5 failed, and 9 errored. The failures/errors are all in `tests/test_cifar_data.py` because the sandbox cannot create `/home/ni/.local/share/columnar_cl_fabricpc/data`.
- `git diff --check` on the edited code, script, test, and work-log files: passed.

Recommended experiment:

- Run the new sweep as a seed-42 scalar-weight test.
- Keep `POST_TRAINING_DIAGNOSTICS=core` so each child records the selected checkpoint accuracy before optional diagnostics.
- If one promotion weight improves or reduces collapse relative to the zero-weight control, run that weight again with `DIAGNOSE_MODE=composer_shells` and `POST_TRAINING_DIAGNOSTICS=full` to inspect route interaction and promotion energy.

Run command:

```bash
bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh
```

## 2026-07-18 Inward Shell-Promotion Sweep Abort

Recorded on 2026-07-18 on `rogdora43`.

Logs inspected:

- `results/codex_inward_shell_promotion_10col3shared_seed42_rogdora43_20260718_081208.log`.
- `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260718_081208.log`.

Configuration attempted:

- Commit: `d825383dadc94058917af15cbd53382d5f01402e`.
- Seed: `42`.
- Learning rate: `0.005`.
- Epochs: `20`.
- Diagnostics: `DIAGNOSE_MODE=nodiag`, `POST_TRAINING_DIAGNOSTICS=core`.
- Model path: 10 columns, 3 shared columns, 7 nonshared active columns, explicit support mask `1,1,1,1,1,1,1,1,1,1`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`, `combiner=shell_attention`, outer-shell context on, column shell bridge on, no bypass readout.
- Inward shell-promotion weights planned: `0.0`, `0.00025`, `0.0005`, `0.001`.

Outcome:

| Inward shell-promotion weight | Status | Last observed training point | Summary metrics |
| --- | --- | --- | --- |
| `0.0` | Killed during epoch 2 | Batch 368 of 7040; epoch 1 validation accuracy was 11.32% | None |
| `0.00025` | Not started | The prior child process exited nonzero | None |
| `0.0005` | Not started | The prior child process exited nonzero | None |
| `0.001` | Not started | The prior child process exited nonzero | None |

Interpretation:

- The sweep did not produce a usable CIFAR-10 classification result.
- The process died during the zero-weight control. Inward shell promotion was inactive in that child run because `inward_shell_promotion_weight=0.0`.
- The child graph had 147 nodes, 273 edges, and 3,125,444 parameters. That matches the previous completed all-column control from `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260717_134042.log`, which reached 31.93% test accuracy and 32.60% best validation accuracy.
- Because the abort happened before any nonzero inward shell-promotion weight ran, this result is evidence about experiment execution only. It is not evidence for or against the new promotion objective.

Recommended next step:

- Do not change architecture based on this run.
- Skip the already-tested zero-weight control and run one active promotion case first.
- Use `inward_shell_promotion_weight=0.0005` as the first active case. This weight is small enough to act as a local consolidation pressure rather than dominating classification energy.
- If the single active case completes, compare it to the prior completed zero-weight control. If it aborts again, add runner-level failure handling and host/GPU memory snapshots before continuing the sweep.

Run command:

```bash
WEIGHTS="0.0005" bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh
```

## 2026-07-18 Single Active Inward Shell-Promotion Result

Recorded on 2026-07-18 on `rogdora43`.

Logs inspected:

- First attempt master log: `results/codex_inward_shell_promotion_10col3shared_seed42_rogdora43_20260718_151803.log`.
- First attempt child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0005_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260718_151803.log`.
- Completed retry master log: `results/codex_inward_shell_promotion_10col3shared_seed42_rogdora43_20260718_163253.log`.
- Completed retry child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0005_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260718_163253.log`.

Execution status:

- The first attempt used commit `d825383dadc94058917af15cbd53382d5f01402e`.
- The first attempt built the active-promotion graph with 177 nodes, 333 edges, and 3,131,334 parameters, then was killed at the first epoch progress line before any validation result.
- The completed retry used commit `e74c2aab0374729ee3bf43ff2383116f6dabadd8`.
- The completed retry reached `completed_at` and printed the core results summary.

Completed retry configuration:

- Seed: `42`.
- Learning rate: `0.005`.
- Epochs: `20`.
- Diagnostics: `DIAGNOSE_MODE=nodiag`, `POST_TRAINING_DIAGNOSTICS=core`.
- Model path: 10 columns, 3 shared columns, 7 active non-shared columns, explicit support mask `1,1,1,1,1,1,1,1,1,1`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`, `combiner=shell_attention`, outer-shell context on, column shell bridge on, no bypass readout.
- Inward shell-promotion weight: `0.0005`.
- `inward_shell_promotion_weight` is the scalar multiplier on the local Gaussian prediction energy added inside each active column.
- `outer_shell -> middle_shell` means the pooled outer-shell vector predicts the pooled middle-shell vector in the same column.
- `middle_shell -> inner_shell` means the pooled middle-shell vector predicts the pooled inner-shell vector in the same column.
- `inner_shell -> hard_kernel` means the pooled inner-shell vector predicts the pooled hard-kernel vector in the same column.

Result:

| Run | Inward shell-promotion weight | Graph | Parameters | Best validation accuracy | Best validation epoch | Test accuracy | Training time |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Prior all-column control | `0.0` | 147 nodes, 273 edges | 3,125,444 | 32.60% | 18 | 31.93% | 7265.5 seconds |
| Active promotion retry | `0.0005` | 177 nodes, 333 edges | 3,131,334 | 30.48% | 18 | 29.95% | 7342.0 seconds |

Validation trajectory:

| Epoch | Prior all-column control | Active promotion retry |
| ---: | ---: | ---: |
| 1 | 10.86% | 10.96% |
| 2 | 21.40% | 22.00% |
| 3 | 19.22% | 20.28% |
| 4 | 21.28% | 22.78% |
| 5 | 24.40% | 23.46% |
| 6 | 25.36% | 22.60% |
| 7 | 24.08% | 25.06% |
| 8 | 28.66% | 26.90% |
| 9 | 26.00% | 16.24% |
| 10 | 25.90% | 23.12% |
| 11 | 22.86% | 20.12% |
| 12 | 25.94% | 22.08% |
| 13 | 27.32% | 26.02% |
| 14 | 28.44% | 28.62% |
| 15 | 27.32% | 24.90% |
| 16 | 31.44% | 28.32% |
| 17 | 31.56% | 28.10% |
| 18 | 32.60% | 30.48% |
| 19 | 31.00% | 28.46% |
| 20 | 31.52% | 28.96% |

Interpretation:

- The active-promotion objective did run. The graph grew by 30 nodes and 60 edges, which matches three promotion prediction nodes per active column across 10 active columns.
- The result is not a collapse. The active-promotion model recovered to 30.48% best validation accuracy and 29.95% test accuracy.
- The result is a meaningful accuracy regression relative to the prior all-column control: test accuracy fell by 1.98 percentage points and best validation accuracy fell by 2.12 percentage points.
- The sharp epoch-9 validation drop in the active-promotion run suggests the new local objective made training less stable even though it later recovered.
- The current implementation promotes all the way into `hard_kernel`. That differs from the earlier conservative plan, which recommended keeping `hard_kernel` untouched in the first pass because it is the strongest central class-bearing shell.

Recommendation:

- Keep inward shell promotion as a research branch, but do not run more full three-pair promotion sweeps yet.
- Change the promotion mechanism so the active pair set is selectable from the experiment runner.
- The next tested pair set should include `outer_shell -> middle_shell` and `middle_shell -> inner_shell`, but exclude `inner_shell -> hard_kernel`.
- Test lower weights first, specifically `0.0001` and `0.00025`, because `0.0005` with the hard-kernel target reduced accuracy.
- Keep all other settings fixed against the completed control: 10 columns, 3 shared columns, all columns active, `shell_attention`, stage4 column grid, outer-shell context on, column shell bridge on, no bypass, no teacher heads, and `shell_lr_multipliers=1,1.5,2,3`.

Alternatives considered:

| Alternative | Advantage | Reason not chosen |
| --- | --- | --- |
| Run lower weights with the existing three-pair objective | No code changes are needed. | It still pushes into `hard_kernel`, which is the part most likely to interfere with the strongest class-bearing shell. |
| Disable inward promotion entirely | Returns to the current all-column control. | It would abandon a HiBaCaML-aligned local consolidation mechanism after only one overly broad active test. |
| Add pair-selective promotion and exclude `inner_shell -> hard_kernel` first | Tests the same idea more faithfully against the earlier conservative plan. | This requires a small implementation change before the next experiment. |

Proposed next step, pending confirmation:

- Add an experiment argument that selects which adjacent shell-promotion pairs are active.
- Add a runner for the conservative pair set `outer_shell -> middle_shell` and `middle_shell -> inner_shell`.
- Run a two-weight sweep with `0.0001` and `0.00025` on seed 42, using core metrics first.

## 2026-07-18 Pair-Selective Inward Shell-Promotion Implementation

Recorded on 2026-07-18 on `rogdora43`.

Implemented after confirmation:

- Added `--inward_shell_promotion_pairs` to `scripts/train_cifar10_depth_spanning.py`.
- `inward_shell_promotion_pairs` is the comma-separated set of adjacent shell-promotion prediction pairs used when `inward_shell_promotion_weight` is positive.
- Valid pair labels are:
  - `outer_to_middle`, where the pooled `outer_shell` vector predicts the pooled `middle_shell` vector in the same column.
  - `middle_to_inner`, where the pooled `middle_shell` vector predicts the pooled `inner_shell` vector in the same column.
  - `inner_to_hard`, where the pooled `inner_shell` vector predicts the pooled `hard_kernel` vector in the same column.
- The value `all` selects all three adjacent pairs and preserves the previous behavior.
- The value `none` selects no pairs and is rejected when `inward_shell_promotion_weight` is positive.
- The conservative pair set is `outer_to_middle,middle_to_inner`, which leaves `hard_kernel` out of the local promotion target path.

Mechanism:

- `parse_inward_shell_promotion_pairs()` parses named pair labels into `(source_shell, target_shell)` tuples.
- `resolve_inward_shell_promotion_pairs()` validates the scalar weight and selected pairs before graph construction.
- `build_depth_spanning_graph()` now creates `ShellContextPredictionNode` promotion objectives only for the selected adjacent pairs.
- Shell-pool construction now uses the selected pair set when promotion is the only reason a shell pool is needed. This prevents an unselected `hard_kernel` pool from being created solely for promotion.
- `train_cifar10_depth_spanning()` prints both `Inward shell promotion weight` and `Inward shell promotion pairs`, so result logs record whether `inner_to_hard` was active.

Runner changes:

- `scripts/run_codex_cifar10_depth_spanning.sh` now accepts positional argument 33 as `inward_shell_promotion_pairs`.
- The wrapper writes `inward_shell_promotion_pairs` to each child log header and forwards it to Python.
- `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh` now accepts the `PROMOTION_PAIRS` environment variable. Its default is `all`.
- Added `scripts/run_codex_conservative_inward_shell_promotion_10col3shared_sweep.sh`.
- The conservative runner sets `PROMOTION_PAIRS=outer_to_middle,middle_to_inner` and defaults `WEIGHTS=0.0001 0.00025`.

Test coverage:

- Added parser tests for `all`, `none`, the conservative pair subset, duplicate labels, and invalid labels.
- Added graph tests showing the default all-pair objective still creates every prior promotion node.
- Added graph tests showing the conservative pair subset creates `outer_shell -> middle_shell` and `middle_shell -> inner_shell`, but does not create `inner_shell -> hard_kernel`.
- Added validation coverage that a positive promotion weight with `inward_shell_promotion_pairs=none` is rejected.

Verification:

- `bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh scripts/run_codex_conservative_inward_shell_promotion_10col3shared_sweep.sh`: passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py tests/test_pooled_readout_norm.py`: passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --help`: passed and showed `--inward_shell_promotion_pairs`.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q`: 66 passed in 13.69 seconds.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py`: 188 passed in 30.39 seconds.

Recommended experiment:

- Run the conservative two-pair sweep on seed 42.
- Keep the same anchor settings used by the completed active-promotion run: 10 columns, 3 shared columns, all columns active, explicit all-column support mask, `shell_attention`, stage4 grid, outer-shell context on, column shell bridge on, no bypass, no teacher heads, and `shell_lr_multipliers=1,1.5,2,3`.
- Compare both runs against the prior zero-promotion all-column control at 31.93% test accuracy and against the full three-pair `0.0005` run at 29.95% test accuracy.
- Interpretation target: if `0.0001` or `0.00025` recovers toward the control while avoiding the epoch-9 drop seen with full three-pair `0.0005`, the next step should tune the conservative pair weights. If both remain below 30%, the next step should inspect promotion energy with full diagnostics before changing the mechanism again.

Run command:

```bash
bash scripts/run_codex_conservative_inward_shell_promotion_10col3shared_sweep.sh
```

## 2026-07-19 Conservative Inward Shell-Promotion Sweep Result

Recorded on 2026-07-19 on `rogdora43`.

Logs inspected:

- Master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_seed42_rogdora43_20260718_194619.log`.
- Weight `0.0001` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0001_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260718_194619.log`.
- Weight `0.00025` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p00025_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260718_215001.log`.

Execution status:

- Commit: `3e4d05e82b08d8e12161af19a7ff9af1070a3b10`.
- Started at `2026-07-18 19:46:19 EDT`.
- Completed at `2026-07-18 23:53:34 EDT`.
- Both child logs printed `Results Summary`, `Test Accuracy`, `Best Val Accuracy`, and `Best Val Epoch`.

Configuration:

- Seed: `42`.
- Learning rate: `0.005`.
- Epochs: `20`.
- Diagnostics: `DIAGNOSE_MODE=nodiag`, `POST_TRAINING_DIAGNOSTICS=core`.
- Model path: 10 columns, 3 shared columns, 7 active non-shared columns, explicit support mask `1,1,1,1,1,1,1,1,1,1`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`, `combiner=shell_attention`, outer-shell context on, column shell bridge on, no bypass readout.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `outer_to_middle` is the local prediction objective where the pooled `outer_shell` vector predicts the pooled `middle_shell` vector in the same column.
- `middle_to_inner` is the local prediction objective where the pooled `middle_shell` vector predicts the pooled `inner_shell` vector in the same column.
- `inner_to_hard` was not active. The pooled `hard_kernel` vector was not a target of the inward shell-promotion objective.

Result:

| Run | Promotion pairs | Inward shell-promotion weight | Graph | Parameters | Best validation accuracy | Best validation epoch | Test accuracy | Training time |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Prior zero-promotion all-column control | none | `0.0` | 147 nodes, 273 edges | 3,125,444 | 32.60% | 18 | 31.93% | 7265.5 seconds |
| Full three-pair promotion | `outer_to_middle,middle_to_inner,inner_to_hard` | `0.0005` | 177 nodes, 333 edges | 3,131,334 | 30.48% | 18 | 29.95% | 7342.0 seconds |
| Conservative promotion | `outer_to_middle,middle_to_inner` | `0.0001` | 167 nodes, 313 edges | 3,129,574 | 30.24% | 17 | 28.45% | 7316.4 seconds |
| Conservative promotion | `outer_to_middle,middle_to_inner` | `0.00025` | 167 nodes, 313 edges | 3,129,574 | 30.22% | 18 | 29.58% | 7308.1 seconds |

Validation trajectory:

| Epoch | Zero-promotion control | Full three-pair `0.0005` | Conservative `0.0001` | Conservative `0.00025` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.86% | 10.96% | 11.22% | 11.42% |
| 2 | 21.40% | 22.00% | 21.36% | 21.48% |
| 3 | 19.22% | 20.28% | 21.14% | 19.06% |
| 4 | 21.28% | 22.78% | 26.54% | 24.32% |
| 5 | 24.40% | 23.46% | 24.32% | 23.30% |
| 6 | 25.36% | 22.60% | 27.28% | 20.08% |
| 7 | 24.08% | 25.06% | 22.36% | 21.76% |
| 8 | 28.66% | 26.90% | 26.32% | 22.52% |
| 9 | 26.00% | 16.24% | 24.32% | 22.88% |
| 10 | 25.90% | 23.12% | 24.68% | 22.88% |
| 11 | 22.86% | 20.12% | 20.94% | 20.20% |
| 12 | 25.94% | 22.08% | 22.14% | 25.46% |
| 13 | 27.32% | 26.02% | 27.86% | 25.96% |
| 14 | 28.44% | 28.62% | 28.80% | 25.38% |
| 15 | 27.32% | 24.90% | 26.14% | 27.52% |
| 16 | 31.44% | 28.32% | 26.72% | 30.00% |
| 17 | 31.56% | 28.10% | 30.24% | 28.46% |
| 18 | 32.60% | 30.48% | 29.58% | 30.22% |
| 19 | 31.00% | 28.46% | 29.46% | 28.50% |
| 20 | 31.52% | 28.96% | 29.08% | 29.14% |

Interpretation:

- The pair selector worked. The conservative graph had 167 nodes and 313 edges, which is 10 fewer nodes and 20 fewer edges than the full three-pair graph. That difference corresponds to leaving out one promotion node per active column and its two incoming edges.
- The conservative objective did not solve the accuracy regression. The best conservative result was 29.58% test accuracy at weight `0.00025`, which is 2.35 percentage points below the zero-promotion control and 0.37 percentage points below the full three-pair `0.0005` run.
- The full three-pair epoch-9 validation drop did not repeat in the conservative runs, but that stability improvement did not recover accuracy.
- The regression is therefore not explained only by `inner_shell -> hard_kernel`. Removing the hard-kernel target reduced the graph pressure on the central shell, but the selected local prediction objective still failed to improve the classifier path.
- The current local objective may be pulling shell states toward cross-shell predictability without guaranteeing that the transferred content is useful for CIFAR-10 class separation. In the current graph, the same local Gaussian prediction energy updates the source shell state, target shell state, and prediction weights.

Conclusion:

- Do not continue scalar weight tuning of the current inward-promotion objective.
- Keep the branch, because it is graph-native and HiBaCaML-aligned in intent, but treat the present local mean-squared prediction form as incomplete.
- The next useful step should inspect or change the mechanism that couples local promotion energy to the class-bearing path, not run more weights of the same objective.

Recommended next step, pending confirmation:

- Add a promotion-target anchoring option before running more overnight experiments.
- The anchoring mechanism should make the local promotion predictor learn to explain the inner target shell without letting the target shell move primarily to satisfy the promotion predictor.
- In code terms, this means adding a `target_gradient_scale` or equivalent precision control to `ShellContextPredictionNode`, where `target_gradient_scale=0.0` leaves the prediction-weight and source-context learning active but removes the promotion objective's direct gradient on the target shell state.
- First test the conservative pair set with `target_gradient_scale=0.0` and weights `0.00025` and `0.0005`.
- This keeps the mechanism predictive-coding native because the graph still contains prediction-error nodes and learned prediction weights, but it changes which state is allowed to absorb the local promotion error.

## 2026-07-19 Promotion Target-Anchoring Implementation

Recorded on 2026-07-19 at `2026-07-19 03:02:23 EDT` on `rogdora43`.

Base commit before these uncommitted changes:

- `8a10f825a4d52ef50f061395e675af5849578c92`.

Goal:

- Implement the confirmed next step from the conservative inward shell-promotion sweep.
- Keep the inward shell-promotion objective in the predictive-coding graph, but stop that local objective from directly moving the target shell state during inference when target anchoring is enabled.

Mechanism:

- `target_gradient_scale` means the scalar multiplier applied to the inference gradient returned through the `target` slot of `ShellContextPredictionNode`.
- `target` means the pooled shell vector that the local prediction objective tries to explain.
- `context` means the pooled shell or context latent that is linearly projected to predict the target shell.
- With `target_gradient_scale=1.0`, `ShellContextPredictionNode` has its previous behavior. The local Gaussian prediction error sends gradients to both the target shell and the context shell.
- With `target_gradient_scale=0.0`, the local Gaussian prediction error still exists, the prediction weights still learn, and the context shell still receives the prediction-error gradient. The target shell does not receive that local objective's inference gradient through the prediction node.

Implementation details:

- Updated `columnar_cl_fabricpc/columns/accuracy_nodes.py`.
- `ShellContextPredictionNode` now requires an explicit `target_gradient_scale` constructor argument.
- `ShellContextPredictionNode.forward_and_latent_grads()` scales only input gradients whose edge key ends with `:target`.
- The target-gradient scale is constrained to `[0, 1]`.
- Updated `scripts/train_cifar10_depth_spanning.py`.
- Added `INWARD_SHELL_PROMOTION_DEFAULT_TARGET_GRADIENT_SCALE=1.0`.
- Added `--inward_shell_promotion_target_gradient_scale`.
- Added validation through `resolve_inward_shell_promotion_target_gradient_scale()`.
- Inward shell-promotion predictors receive the CLI value.
- Outer-shell-context shell predictors now pass `target_gradient_scale=1.0` explicitly, so that existing context-prediction dynamics remain unanchored.
- Updated the architecture docstring to note that inward shell-promotion target-slot gradients can be scaled. The graph topology is unchanged because the same nodes and edges are still created.
- Updated `scripts/run_codex_cifar10_depth_spanning.sh`.
- Positional argument 34 is now `inward_shell_promotion_target_gradient_scale`.
- Child log filenames include `iptg<value>`, where `iptg` means inward-promotion target-gradient scale.
- Child log headers print `inward_shell_promotion_target_gradient_scale`.
- Updated `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`.
- Added the `INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE` environment variable.
- Master log filenames include the target-gradient scale.
- Added `scripts/run_codex_anchored_inward_shell_promotion_10col3shared_sweep.sh`.
- The anchored runner uses `PROMOTION_PAIRS=outer_to_middle,middle_to_inner`, `WEIGHTS=0.00025 0.0005`, and `INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE=0.0` by default.

Test coverage:

- Added `test_shell_context_prediction_node_can_anchor_target_gradient()`.
- This test confirms that `target_gradient_scale=0.0` zeros the target-slot input gradient while keeping the context-slot input gradient nonzero.
- Added `test_depth_spanning_graph_sets_inward_promotion_target_gradient_scale()`.
- This test confirms that graph-built inward-promotion nodes carry the configured scale.
- Added `test_depth_spanning_graph_rejects_invalid_promotion_target_gradient_scale()`.
- This test confirms that values outside `[0, 1]` are rejected.

Verification:

- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py columnar_cl_fabricpc/columns/accuracy_nodes.py tests/test_pooled_readout_norm.py`: passed.
- `bash -n scripts/run_codex_cifar10_depth_spanning.sh scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh scripts/run_codex_conservative_inward_shell_promotion_10col3shared_sweep.sh scripts/run_codex_anchored_inward_shell_promotion_10col3shared_sweep.sh`: passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -q`: 69 passed in 13.94 seconds.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --help`: passed and showed `--inward_shell_promotion_target_gradient_scale`.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest -q --ignore=tests/test_cifar_data.py`: 191 passed in 31.00 seconds.
- `git diff --check`: passed.

Next experiment:

- Run the anchored conservative promotion sweep.
- Compare against the prior zero-promotion all-column control at 31.93% test accuracy and the unanchored conservative promotion results at 28.45% and 29.58% test accuracy.
- The key question is whether target anchoring prevents local promotion error from moving class-bearing target shell states away from the CIFAR-10 classifier objective.

Run command:

```bash
bash scripts/run_codex_anchored_inward_shell_promotion_10col3shared_sweep.sh
```

## 2026-07-19 Anchored Inward Shell-Promotion Sweep Result

Recorded on 2026-07-19 at `2026-07-19 20:43:01 EDT` on `rogdora43`.

Logs inspected:

- Master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed42_rogdora43_20260719_031054.log`.
- Weight `0.00025` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p00025_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260719_031054.log`.
- Weight `0.0005` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0005_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260719_051423.log`.

Execution status:

- Commit: `2eef43e1978ff5b6d79979af23ebe1ab105e93d9`.
- Started at `2026-07-19 03:10:54 EDT`.
- Completed at `2026-07-19 07:17:13 EDT`.
- Both child logs printed `Results Summary`, `Test Accuracy`, `Best Val Accuracy`, and `Best Val Epoch`.

Configuration:

- Seed: `42`.
- Learning rate: `0.005`.
- Epochs: `20`.
- Diagnostics: `DIAGNOSE_MODE=nodiag`, `POST_TRAINING_DIAGNOSTICS=core`.
- Model path: 10 columns, 3 shared columns, 7 active non-shared columns, explicit support mask `1,1,1,1,1,1,1,1,1,1`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`, `combiner=shell_attention`, outer-shell context on, column shell bridge on, no bypass readout.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- `inward_shell_promotion_target_gradient_scale` is the scalar multiplier applied to the inference gradient sent from a shell-promotion prediction node into its target shell. The value `0.0` anchors the target shell against that local objective while leaving prediction error, prediction weights, and source-shell context gradients active.

Result:

| Run | Promotion pairs | Weight | Target-gradient scale | Graph | Parameters | Best validation accuracy | Best validation epoch | Test accuracy | Training time |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Prior zero-promotion all-column control | none | `0.0` | none | 147 nodes, 273 edges | 3,125,444 | 32.60% | 18 | 31.93% | 7265.5 seconds |
| Prior unanchored conservative promotion | `outer_to_middle,middle_to_inner` | `0.00025` | `1.0` | 167 nodes, 313 edges | 3,129,574 | 30.22% | 18 | 29.58% | 7308.1 seconds |
| Anchored conservative promotion | `outer_to_middle,middle_to_inner` | `0.00025` | `0.0` | 167 nodes, 313 edges | 3,129,574 | 29.16% | 18 | 29.46% | 7302.4 seconds |
| Anchored conservative promotion | `outer_to_middle,middle_to_inner` | `0.0005` | `0.0` | 167 nodes, 313 edges | 3,129,574 | 33.54% | 18 | 32.83% | 7267.0 seconds |

Validation trajectory:

| Epoch | Zero-promotion control | Unanchored conservative `0.00025` | Anchored conservative `0.00025` | Anchored conservative `0.0005` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.86% | 11.42% | 10.86% | 10.92% |
| 2 | 21.40% | 21.48% | 21.72% | 20.90% |
| 3 | 19.22% | 19.06% | 20.26% | 20.68% |
| 4 | 21.28% | 24.32% | 25.46% | 24.90% |
| 5 | 24.40% | 23.30% | 24.22% | 24.94% |
| 6 | 25.36% | 20.08% | 26.72% | 27.58% |
| 7 | 24.08% | 21.76% | 19.68% | 25.38% |
| 8 | 28.66% | 22.52% | 25.38% | 28.14% |
| 9 | 26.00% | 22.88% | 24.80% | 29.62% |
| 10 | 25.90% | 22.88% | 26.96% | 26.88% |
| 11 | 22.86% | 20.20% | 20.38% | 20.76% |
| 12 | 25.94% | 25.46% | 23.04% | 28.16% |
| 13 | 27.32% | 25.96% | 22.50% | 28.44% |
| 14 | 28.44% | 25.38% | 24.50% | 29.34% |
| 15 | 27.32% | 27.52% | 24.60% | 30.46% |
| 16 | 31.44% | 30.00% | 25.68% | 33.30% |
| 17 | 31.56% | 28.46% | 28.16% | 30.82% |
| 18 | 32.60% | 30.22% | 29.16% | 33.54% |
| 19 | 31.00% | 28.50% | 27.80% | 32.10% |
| 20 | 31.52% | 29.14% | 28.98% | 33.32% |

Interpretation:

- Target anchoring changed the outcome at `inward_shell_promotion_weight=0.0005`.
- The anchored `0.0005` run reached 32.83% test accuracy, which is 0.90 percentage points above the zero-promotion all-column control and 3.25 percentage points above the best unanchored conservative run at weight `0.00025`.
- The anchored `0.0005` run also avoided the severe epoch-9 validation drop seen in the prior full three-pair unanchored `0.0005` run. Its epoch-9 validation accuracy was 29.62%.
- The anchored `0.00025` run did not improve. Its 29.46% test accuracy was essentially tied with the unanchored `0.00025` conservative result and below the zero-promotion control.
- The result supports the mechanism-level hypothesis that unanchored promotion harmed the classifier path by moving target shell states to satisfy local shell-prediction error. At `target_gradient_scale=0.0`, the target shell remains governed by the rest of the graph while the source shell and predictor weights still receive local predictive-coding pressure.
- The result also suggests that the promotion weight must be large enough to train the source-shell predictor path. Anchoring alone did not make the `0.00025` objective useful.

Conclusion:

- Keep the target-anchored promotion mechanism.
- Treat anchored conservative promotion at weight `0.0005` as the new active candidate for the 10-column, 3-shared, all-active branch.
- Do not add a new mechanism before checking robustness. This is the first promotion variant in this branch that exceeded the zero-promotion all-column control on seed 42.

Recommended next step:

- Run a multi-seed replicate of the anchored conservative `0.0005` setting on seeds `99` and `7`, keeping seed `42` as the completed reference.
- The main question is whether anchored promotion improves the average result and avoids collapse across seeds.
- If seed `99` and seed `7` are stable, then the next mechanism change should tune the anchored promotion family around weight `0.0005`, with candidate weights `0.000375`, `0.00075`, and `0.001`.
- If either replicate collapses, then inspect promotion and classifier energy traces before increasing weight.

## 2026-07-19 Anchored Promotion Replicate Runner Setup

Recorded on 2026-07-19 at `2026-07-19 20:45:56 EDT` on `rogdora43`.

Base commit before these uncommitted setup changes:

- `2eef43e1978ff5b6d79979af23ebe1ab105e93d9`.

Goal:

- Run the robustness check recommended after the positive anchored seed-42 result.
- Keep seed `42` as the completed reference.
- Run seeds `99` and `7` sequentially with the exact active anchored-promotion setting that reached 32.83% test accuracy on seed `42`.

Implementation:

- Added `scripts/run_codex_anchored_inward_shell_promotion_10col3shared_replicate.sh`.
- The script calls `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh` once per seed.
- Default seeds are `99 7`.
- Default weight is `0.0005`.
- Default promotion pairs are `outer_to_middle,middle_to_inner`.
- Default `inward_shell_promotion_target_gradient_scale` is `0.0`.
- The script writes an outer master log named `results/codex_anchored_inward_shell_promotion_10col3shared_replicate_pairsouter_to_middle_middle_to_inner_iptg0p0_<hostname>_<timestamp>.log`.
- Each child call still writes its own seed-specific master log and child training log through the existing inward-promotion sweep runner.

Configuration inherited from the existing sweep runner:

- 10 columns.
- 3 shared columns.
- 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`.
- `column_grid=stage4`.
- `embed_dim=64`.
- `microcolumn_dim=32`.
- Outer-shell context on.
- Column shell bridge on.
- No bypass readout.
- No teacher heads.
- `shell_lr_multipliers=1,1.5,2,3`.
- `POST_TRAINING_DIAGNOSTICS=core`.

Verification:

- `bash -n scripts/run_codex_anchored_inward_shell_promotion_10col3shared_replicate.sh scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh scripts/run_codex_cifar10_depth_spanning.sh`: passed.

Run command:

```bash
bash scripts/run_codex_anchored_inward_shell_promotion_10col3shared_replicate.sh
```

Optional overrides:

- `SEEDS="99 7"` changes the seed list.
- `WEIGHTS="0.0005"` changes the promotion-weight list.
- `INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE="0.0"` changes the target-gradient scale.

## 2026-07-20 Anchored Promotion Replicate Result

Recorded on 2026-07-20 at `2026-07-20 06:08:32 EDT` on `rogdora43`.

Current repository commit after the completed run:

- `ac5bceab9df2728b642dd2646f3c1e6f11213f00`.

Logs analyzed:

- Outer master log: `results/codex_anchored_inward_shell_promotion_10col3shared_replicate_pairsouter_to_middle_middle_to_inner_iptg0p0_rogdora43_20260719_204703.log`.
- Seed 99 master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed99_rogdora43_20260719_204703.log`.
- Seed 99 training log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0005_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed99_lr0p005_ep20_nodiag_rogdora43_20260719_204703.log`.
- Seed 7 master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed7_rogdora43_20260719_225041.log`.
- Seed 7 training log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0005_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed7_lr0p005_ep20_nodiag_rogdora43_20260719_225041.log`.

Execution note:

- Seed `99` ran at commit `2eef43e1978ff5b6d79979af23ebe1ab105e93d9`.
- Seed `7` ran at commit `ac5bceab9df2728b642dd2646f3c1e6f11213f00`.
- The diff between those commits contains the work log, result logs, and the anchored replicate wrapper. It does not contain model or training-code changes, so the two seed results remain comparable for this replicate.

Configuration:

- `inward_shell_promotion_weight` is the scalar multiplier on the shell-context prediction energy that predicts a more central shell state from a more outer shell state inside each column.
- `inward_shell_promotion_weight=0.0005`.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale` is the multiplier on gradient flowing into the target shell state from the local inward-promotion objective.
- `inward_shell_promotion_target_gradient_scale=0.0`, so the promotion objective trains the source/predictor path while the target shell state remains governed by the rest of the predictive-coding graph.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.
- 20 epochs, learning rate `0.005`, seed-specific CIFAR-10 train/validation/test splits.
- Graph size was 167 nodes and 313 edges.
- Parameter count was 3,129,574.

Primary results:

| Seed | Best validation accuracy | Best epoch | Test accuracy | Training time |
| --- | ---: | ---: | ---: | ---: |
| 42 | 33.54% | 18 | 32.83% | 7,267.0s |
| 99 | 26.86% | 8 | 26.38% | 7,311.0s |
| 7 | 32.56% | 19 | 32.35% | 7,306.9s |

Aggregate:

- Mean test accuracy over seeds `42`, `99`, and `7`: 30.52%.
- Sample standard deviation of test accuracy over those seeds: 3.59 percentage points.
- Test accuracy range over those seeds: 6.45 percentage points.
- Mean best validation accuracy over those seeds: 30.99%.
- Sample standard deviation of best validation accuracy over those seeds: 3.61 percentage points.

Validation trajectory:

| Epoch | Seed 42 validation | Seed 99 validation | Seed 7 validation |
| ---: | ---: | ---: | ---: |
| 1 | 10.92% | 11.00% | 13.20% |
| 2 | 20.90% | 18.94% | 23.24% |
| 3 | 20.68% | 21.00% | 20.44% |
| 4 | 24.90% | 19.78% | 22.04% |
| 5 | 24.94% | 24.90% | 26.54% |
| 6 | 27.58% | 23.14% | 22.96% |
| 7 | 25.38% | 24.14% | 21.04% |
| 8 | 28.14% | 26.86% | 25.40% |
| 9 | 29.62% | 17.96% | 24.26% |
| 10 | 26.88% | 9.60% | 23.98% |
| 11 | 20.76% | 14.52% | 26.02% |
| 12 | 28.16% | 17.26% | 25.02% |
| 13 | 28.44% | 14.12% | 24.46% |
| 14 | 29.34% | 15.62% | 28.08% |
| 15 | 30.46% | 16.30% | 24.24% |
| 16 | 33.30% | 18.80% | 27.96% |
| 17 | 30.82% | 17.60% | 30.70% |
| 18 | 33.54% | 20.82% | 30.66% |
| 19 | 32.10% | 22.66% | 32.56% |
| 20 | 33.32% | 19.74% | 31.00% |

Comparison points:

| Condition | Seed 42 test | Seed 99 test | Seed 7 test | Mean test |
| --- | ---: | ---: | ---: | ---: |
| Earlier bridge/context anchor, no inward promotion | 33.93% | 33.62% | 29.91% | 32.49% |
| Current explicit-mask zero-promotion control | 31.93% | not found | not found | not available |
| Anchored inward promotion, `outer_to_middle,middle_to_inner`, weight `0.0005` | 32.83% | 26.38% | 32.35% | 30.52% |

Interpretation:

- The anchored promotion result is not robust across seeds.
- Seed `42` and seed `7` are useful results. They show that anchored inward promotion can operate above 32% test accuracy in the current 10-column, no-bypass family.
- Seed `99` is the important failure. It rose to 26.86% validation accuracy at epoch 8, fell to 17.96% at epoch 9, and reached 9.60% at epoch 10. This is a classification collapse by the current project priority.
- The collapse is not explained by the final epoch alone, because the best validation checkpoint for seed `99` also produced only 26.38% test accuracy.
- The earlier bridge/context anchor reached 33.62% test accuracy on seed `99`, so seed `99` is not inherently unusable for this architecture family.
- There are no exact current-code, explicit-support-mask, zero-promotion control logs for seeds `99` and `7`. This means the current result can compare directly to the seed-42 current control, but it can only compare indirectly to the earlier three-seed bridge/context anchor.

Conclusion:

- Do not sweep higher anchored-promotion weights yet.
- Treat the seed-99 collapse as the primary issue to solve.
- The immediate question is whether the collapse comes from the current training baseline under seed `99`, from the `outer_to_middle` promotion edge, from the `middle_to_inner` promotion edge, or from using both promotion edges at the same time.

Recommended next step:

- Run one targeted seed-99 diagnostic sequence before adding any new mechanism.
- The diagnostic sequence should use 20 epochs, the same current 10-column/3-shared setup, `inward_shell_promotion_target_gradient_scale=0.0`, and no bypass readout.
- Run A: seed `99`, `inward_shell_promotion_weight=0.0`, no effective inward promotion. This establishes the missing current-code seed-99 control.
- Run B: seed `99`, `inward_shell_promotion_weight=0.0005`, `inward_shell_promotion_pairs=outer_to_middle`.
- Run C: seed `99`, `inward_shell_promotion_weight=0.0005`, `inward_shell_promotion_pairs=middle_to_inner`.

Decision rule:

- If Run A is stable and one single-pair run is stable, continue from the stable promotion pair and replicate it on seeds `42` and `7`.
- If Run A is stable and both single-pair runs collapse, reduce the promotion objective or add an epoch schedule so the local shell objective cannot dominate early classifier formation.
- If Run A collapses, return to the current bridge/context baseline before modifying promotion, because the instability would not be promotion-specific.

Alternatives considered:

- Increasing promotion weight was rejected for now because seed `99` already collapsed at weight `0.0005`.
- A lower-weight sweep was deferred because it would not distinguish whether `outer_to_middle` or `middle_to_inner` caused the failure.
- Full per-batch energy logging was deferred until the failing edge is isolated, because the current result already identifies the failing seed and epoch window.

## 2026-07-20 Anchored Promotion Nearby-Weight Sweep Setup

Recorded on 2026-07-20 at `2026-07-20 06:37:22 EDT` on `rogdora43`.

Current repository commit before this uncommitted setup change:

- `1afdcb816d7c8c4f515e7cc0ffa618ca1122ddb0`.

Direction change:

- The prior recommendation was to isolate the seed-99 collapse before sweeping higher weights.
- The user redirected the next step toward nearby-weight exploration, because the immediate priority is finding whether anchored inward promotion has a better operating point near `0.0005`.
- This setup follows that direction and keeps the architecture fixed.

Implementation:

- Added `scripts/run_codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot.sh`.
- The script is a wrapper around `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`.
- It runs the anchored conservative promotion path with `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- It keeps `inward_shell_promotion_target_gradient_scale=0.0`.
- It defaults to seed `42` so the four 20-epoch runs are a practical first pass.
- It defaults to weights `0.0001 0.000375 0.00075 0.001`.
- It writes an outer master log named `results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairsouter_to_middle_middle_to_inner_weights0p0001_0p000375_0p00075_0p001_iptg0p0_<hostname>_<timestamp>.log`.
- Each child call still writes a seed-specific inward-promotion master log and child training logs through the existing sweep runner.

Fixed configuration inherited from the inward-promotion runner:

- 10 columns.
- 3 shared columns.
- 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`.
- `column_grid=stage4`.
- `embed_dim=64`.
- `microcolumn_dim=32`.
- Outer-shell context on.
- Column shell bridge on.
- No bypass readout.
- No teacher heads.
- `shell_lr_multipliers=1,1.5,2,3`.
- `POST_TRAINING_DIAGNOSTICS=core`.

Primary run command:

```bash
bash scripts/run_codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot.sh
```

Optional overrides:

- `SEEDS="42 99 7"` runs the same weight list over all three seeds.
- `WEIGHTS="0.0001 0.000375 0.00075 0.001"` changes the weight list.
- `PROMOTION_PAIRS="outer_to_middle,middle_to_inner"` changes the promotion pairs.
- `INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE="0.0"` changes the target-gradient scale.

Interpretation plan:

- Compare each weight against the completed anchored `0.0005` seed-42 result, which reached 33.54% validation accuracy and 32.83% test accuracy.
- If one nearby weight beats `0.0005` on seed `42`, replicate that weight on seeds `99` and `7`.
- If the best nearby weight is still near `0.0005`, narrow the next search rather than changing architecture.
- If all nearby weights are worse than `0.0005`, keep `0.0005` as the active anchored-promotion setting and consider a schedule or pair-specific weights next.

## 2026-07-20 Anchored Promotion Nearby-Weight Sweep Result

Recorded on 2026-07-20 at `2026-07-20 15:22:26 EDT` on `rogdora43`.

Current repository commit after the completed run:

- `c3da437000307578eb110ad70f8b5c8ee08ff8c1`.

Logs analyzed:

- Outer master log: `results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairsouter_to_middle_middle_to_inner_weights0p0001_0p000375_0p00075_0p001_iptg0p0_rogdora43_20260720_063844.log`.
- Inner master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed42_rogdora43_20260720_063844.log`.
- Weight `0.0001` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0001_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260720_063844.log`.
- Weight `0.000375` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p000375_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260720_084211.log`.
- Weight `0.00075` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p00075_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260720_104543.log`.
- Weight `0.001` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p001_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260720_124918.log`.

Execution note:

- The outer master log started at commit `1afdcb816d7c8c4f515e7cc0ffa618ca1122ddb0`.
- The first child run, weight `0.0001`, also ran at commit `1afdcb816d7c8c4f515e7cc0ffa618ca1122ddb0`.
- The remaining child runs, weights `0.000375`, `0.00075`, and `0.001`, ran at commit `c3da437000307578eb110ad70f8b5c8ee08ff8c1`.
- The diff between those commits contains only the work log, result logs, and the nearby-weight sweep wrapper. It does not contain model or training-code changes.

Configuration:

- `inward_shell_promotion_weight` is the scalar multiplier on the shell-context prediction energy that predicts a more central shell state from a more outer shell state inside each column.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale` is the multiplier on gradient flowing into the target shell state from the local inward-promotion objective.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- Seed `42`, 20 epochs, learning rate `0.005`.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.
- Graph size was 167 nodes and 313 edges.
- Parameter count was 3,129,574.

Primary results:

| Inward shell promotion weight | Best validation accuracy | Best epoch | Test accuracy | Delta from current zero-promotion control | Delta from prior `0.0005` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 current control | 32.60% | 18 | 31.93% | 0.00 pp | -0.90 pp |
| 0.0001 | 33.14% | 18 | 31.72% | -0.21 pp | -1.11 pp |
| 0.000375 | 33.80% | 18 | 33.74% | +1.81 pp | +0.91 pp |
| 0.0005 prior anchored reference | 33.54% | 18 | 32.83% | +0.90 pp | 0.00 pp |
| 0.00075 | 30.42% | 19 | 29.94% | -1.99 pp | -2.89 pp |
| 0.001 | 27.92% | 18 | 28.41% | -3.52 pp | -4.42 pp |

Validation trajectory for the new sweep:

| Epoch | Weight 0.0001 | Weight 0.000375 | Weight 0.00075 | Weight 0.001 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.90% | 11.34% | 11.86% | 11.14% |
| 2 | 22.12% | 20.74% | 22.38% | 22.24% |
| 3 | 20.84% | 20.88% | 20.34% | 20.68% |
| 4 | 26.10% | 26.16% | 25.56% | 21.80% |
| 5 | 24.26% | 24.78% | 24.46% | 20.46% |
| 6 | 26.32% | 27.24% | 26.84% | 20.96% |
| 7 | 24.72% | 24.74% | 22.68% | 22.42% |
| 8 | 26.40% | 26.34% | 25.58% | 23.60% |
| 9 | 28.54% | 24.70% | 23.68% | 17.46% |
| 10 | 28.02% | 25.90% | 22.74% | 22.32% |
| 11 | 21.44% | 21.08% | 20.44% | 20.40% |
| 12 | 26.40% | 26.10% | 25.22% | 23.02% |
| 13 | 28.92% | 29.54% | 27.58% | 22.34% |
| 14 | 28.70% | 31.32% | 28.54% | 21.28% |
| 15 | 29.70% | 29.94% | 26.74% | 24.92% |
| 16 | 30.42% | 27.66% | 28.50% | 24.78% |
| 17 | 30.36% | 32.06% | 28.70% | 25.78% |
| 18 | 33.14% | 33.80% | 29.04% | 27.92% |
| 19 | 30.38% | 31.92% | 30.42% | 27.66% |
| 20 | 31.46% | 32.42% | 30.36% | 27.82% |

Interpretation:

- Weight `0.000375` is the best seed-42 result in this nearby-weight sweep.
- Weight `0.000375` improved test accuracy by 0.91 percentage points over the previous anchored `0.0005` reference and by 1.81 percentage points over the current explicit-mask zero-promotion control.
- Weight `0.0001` did not improve test accuracy over the zero-promotion control, despite reaching a slightly higher best validation accuracy than the control.
- Weights `0.00075` and `0.001` degraded substantially. The higher-weight side of the curve is not promising under this anchored conservative two-pair setup.
- The useful region appears to be below `0.0005`, with `0.000375` currently the best point. This supports the idea that inward shell promotion is useful when it is present but weak enough not to dominate the classifier pathway.
- Post-training diagnostics did not emit additional core metrics beyond the marker line in these child logs.

Recommended next step:

- Replicate weight `0.000375` on seeds `99` and `7` before changing architecture.
- Keep promotion pairs `outer_to_middle,middle_to_inner`, `inward_shell_promotion_target_gradient_scale=0.0`, and the same 10-column/3-shared setup.
- Compare the three-seed mean for weight `0.000375` against the completed weight `0.0005` three-seed mean of 30.52%.
- If weight `0.000375` holds seed `42` and improves either seed `99` or seed `7`, it should replace `0.0005` as the active anchored-promotion baseline.

## 2026-07-20 Anchored Promotion 0.000375 Replicate Result

Recorded on 2026-07-20 at `2026-07-20 21:09:48 EDT` on `rogdora43`.

Current repository commit after the completed run:

- `43ae67c6fa0e88307568bc6871fb60c0ec9b8dce`.

Logs analyzed:

- Outer master log: `results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairsouter_to_middle_middle_to_inner_weights0p000375_iptg0p0_rogdora43_20260720_152604.log`.
- Seed 99 master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed99_rogdora43_20260720_152604.log`.
- Seed 99 child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p000375_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed99_lr0p005_ep20_nodiag_rogdora43_20260720_152604.log`.
- Seed 7 master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed7_rogdora43_20260720_172952.log`.
- Seed 7 child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p000375_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed7_lr0p005_ep20_nodiag_rogdora43_20260720_172952.log`.

Execution note:

- Seed `99` ran at commit `c3da437000307578eb110ad70f8b5c8ee08ff8c1`.
- Seed `7` ran at commit `43ae67c6fa0e88307568bc6871fb60c0ec9b8dce`.
- The diff between those commits contains the work log, result logs, and the nearby-weight sweep wrapper. It does not contain model or training-code changes, so the seed results remain comparable.

Configuration:

- `inward_shell_promotion_weight` is the scalar multiplier on the shell-context prediction energy that predicts a more central shell state from a more outer shell state inside each column.
- `inward_shell_promotion_weight=0.000375`.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale` is the multiplier on gradient flowing into the target shell state from the local inward-promotion objective.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.
- Graph size was 167 nodes and 313 edges.
- Parameter count was 3,129,574.

Primary results:

| Seed | Best validation accuracy | Best epoch | Test accuracy | Training time |
| --- | ---: | ---: | ---: | ---: |
| 42 | 33.80% | 18 | 33.74% | 7,306.9s |
| 99 | 34.60% | 19 | 32.93% | 7,320.9s |
| 7 | 30.84% | 19 | 31.31% | 7,317.1s |

Aggregate:

- Mean test accuracy over seeds `42`, `99`, and `7`: 32.66%.
- Sample standard deviation of test accuracy over those seeds: 1.24 percentage points.
- Test accuracy range over those seeds: 2.43 percentage points.
- Mean best validation accuracy over those seeds: 33.08%.
- Sample standard deviation of best validation accuracy over those seeds: 1.98 percentage points.

Comparison against the prior anchored `0.0005` replicate:

| Seed | Weight 0.0005 test | Weight 0.000375 test | Delta |
| --- | ---: | ---: | ---: |
| 42 | 32.83% | 33.74% | +0.91 pp |
| 99 | 26.38% | 32.93% | +6.55 pp |
| 7 | 32.35% | 31.31% | -1.04 pp |
| Mean | 30.52% | 32.66% | +2.14 pp |

Comparison against the earlier bridge/context anchor:

| Seed | Earlier bridge/context test | Weight 0.000375 test | Delta |
| --- | ---: | ---: | ---: |
| 42 | 33.93% | 33.74% | -0.19 pp |
| 99 | 33.62% | 32.93% | -0.69 pp |
| 7 | 29.91% | 31.31% | +1.40 pp |
| Mean | 32.49% | 32.66% | +0.17 pp |

Validation trajectory:

| Epoch | Seed 42 validation | Seed 99 validation | Seed 7 validation |
| ---: | ---: | ---: | ---: |
| 1 | 11.34% | 10.88% | 14.44% |
| 2 | 20.74% | 18.60% | 22.42% |
| 3 | 20.88% | 19.36% | 20.54% |
| 4 | 26.16% | 19.98% | 22.46% |
| 5 | 24.78% | 25.46% | 25.88% |
| 6 | 27.24% | 23.36% | 24.14% |
| 7 | 24.74% | 27.84% | 24.84% |
| 8 | 26.34% | 30.28% | 24.08% |
| 9 | 24.70% | 29.08% | 24.98% |
| 10 | 25.90% | 24.86% | 25.58% |
| 11 | 21.08% | 26.32% | 17.92% |
| 12 | 26.10% | 20.38% | 24.22% |
| 13 | 29.54% | 26.28% | 23.68% |
| 14 | 31.32% | 21.02% | 23.86% |
| 15 | 29.94% | 27.68% | 23.38% |
| 16 | 27.66% | 29.70% | 27.74% |
| 17 | 32.06% | 30.18% | 28.80% |
| 18 | 33.80% | 33.10% | 29.96% |
| 19 | 31.92% | 34.60% | 30.84% |
| 20 | 32.42% | 32.66% | 30.74% |

Interpretation:

- Weight `0.000375` should replace `0.0005` as the active anchored-promotion baseline.
- The main improvement is robustness. The three-seed mean increased from 30.52% to 32.66%, and the test-accuracy range dropped from 6.45 percentage points to 2.43 percentage points.
- Seed `99` no longer shows the severe collapse seen at weight `0.0005`. Its best validation accuracy rose from 26.86% to 34.60%, and its test accuracy rose from 26.38% to 32.93%.
- Seed `7` is weaker than its weight-`0.0005` result by 1.04 percentage points, but it remains above 31% test accuracy and above the earlier bridge/context seed-7 result.
- Seed `7` did dip to 17.92% validation accuracy at epoch 11 before recovering to 30.84% at epoch 19. This is worth monitoring, but it is not the same failure mode as the seed-99 collapse at weight `0.0005`.
- Relative to the earlier bridge/context anchor, the new `0.000375` anchored-promotion setting is essentially tied on mean test accuracy and is more balanced across seeds.

Recommended next step:

- Treat `0.000375` as the current best conservative anchored-promotion setting.
- Run a narrow seed-42 scalar search around it before adding a new mechanism.
- Candidate weights: `0.0003`, `0.00035`, `0.0004`, and `0.00045`.
- If none beat `0.000375`, keep `0.000375` and move on to pair-specific promotion weights or a weak promotion schedule.
- If one candidate beats `0.000375` by a meaningful margin, replicate that candidate on seeds `99` and `7`.

Suggested command:

```bash
SEEDS="42" WEIGHTS="0.0003 0.00035 0.0004 0.00045" bash scripts/run_codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot.sh
```

## 2026-07-21 Anchored Promotion Narrow Scalar Sweep Result

Recorded on 2026-07-21 at `2026-07-21 13:54:06 EDT` on `rogdora43`.

Current repository commit after the completed run:

- `3076d6c15297382c30b882352df31e2719a849ed`.

Logs analyzed:

- Outer master log: `results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairsouter_to_middle_middle_to_inner_weights0p0003_0p00035_0p0004_0p00045_iptg0p0_rogdora43_20260720_211655.log`.
- Inner master log: `results/codex_inward_shell_promotion_10col3shared_pairsouter_to_middle_middle_to_inner_iptg0p0_seed42_rogdora43_20260720_211655.log`.
- Weight `0.0003` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0003_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260720_211655.log`.
- Weight `0.00035` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p00035_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260720_232024.log`.
- Weight `0.0004` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p0004_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260721_012342.log`.
- Weight `0.00045` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p00045_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260721_032656.log`.

Execution note:

- The outer master log started at commit `43ae67c6fa0e88307568bc6871fb60c0ec9b8dce`.
- The first child run, weight `0.0003`, ran at commit `43ae67c6fa0e88307568bc6871fb60c0ec9b8dce`.
- The remaining child runs, weights `0.00035`, `0.0004`, and `0.00045`, ran at commit `3076d6c15297382c30b882352df31e2719a849ed`.
- The diff between those commits contains docs, result logs, and session-log files. It does not contain model or training-code changes.

Configuration:

- `inward_shell_promotion_weight` is the scalar multiplier on the shell-context prediction energy that predicts a more central shell state from a more outer shell state inside each column.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale` is the multiplier on gradient flowing into the target shell state from the local inward-promotion objective.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- Seed `42`, 20 epochs, learning rate `0.005`.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.
- Graph size was 167 nodes and 313 edges.
- Parameter count was 3,129,574.

Primary results:

| Inward shell promotion weight | Best validation accuracy | Best epoch | Test accuracy | Delta from active `0.000375` seed-42 test | Delta from current zero-promotion control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 32.84% | 18 | 31.38% | -2.36 pp | -0.55 pp |
| 0.00035 | 32.84% | 18 | 31.61% | -2.13 pp | -0.32 pp |
| 0.000375 active reference | 33.80% | 18 | 33.74% | 0.00 pp | +1.81 pp |
| 0.0004 | 29.74% | 19 | 28.73% | -5.01 pp | -3.20 pp |
| 0.00045 | 30.82% | 18 | 30.87% | -2.87 pp | -1.06 pp |

Validation trajectory:

| Epoch | Weight 0.0003 | Weight 0.00035 | Weight 0.0004 | Weight 0.00045 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.36% | 11.48% | 11.08% | 11.06% |
| 2 | 22.14% | 21.90% | 21.68% | 21.34% |
| 3 | 20.36% | 20.76% | 20.18% | 20.12% |
| 4 | 26.16% | 24.64% | 25.18% | 25.30% |
| 5 | 25.00% | 24.04% | 25.08% | 24.34% |
| 6 | 27.96% | 25.82% | 27.00% | 26.54% |
| 7 | 25.80% | 23.20% | 24.84% | 23.94% |
| 8 | 28.54% | 29.80% | 23.80% | 26.88% |
| 9 | 26.32% | 25.82% | 23.30% | 27.82% |
| 10 | 25.54% | 25.68% | 23.26% | 26.18% |
| 11 | 22.46% | 22.16% | 20.16% | 20.76% |
| 12 | 28.32% | 26.36% | 25.24% | 24.54% |
| 13 | 29.38% | 24.60% | 20.88% | 26.96% |
| 14 | 27.16% | 28.02% | 26.42% | 26.66% |
| 15 | 28.02% | 28.36% | 25.18% | 27.60% |
| 16 | 29.74% | 31.88% | 28.08% | 27.70% |
| 17 | 30.86% | 30.20% | 28.70% | 27.92% |
| 18 | 32.84% | 32.84% | 29.42% | 30.82% |
| 19 | 31.60% | 31.48% | 29.74% | 29.72% |
| 20 | 31.54% | 31.76% | 29.54% | 29.48% |

Interpretation:

- None of the narrower seed-42 scalar candidates beat the active `0.000375` anchored-promotion setting.
- The two below-`0.000375` candidates, `0.0003` and `0.00035`, were close in validation accuracy but fell more than two percentage points behind in test accuracy.
- The two above-`0.000375` candidates, `0.0004` and `0.00045`, degraded clearly. This reinforces the pattern from the wider sweep that the useful scalar weight region is narrow and drops off above `0.000375`.
- The active `0.000375` result remains the best seed-42 scalar point in this anchored two-pair promotion family.
- The current scalar sweep does not justify another nearby scalar-only search.

Recommended next step:

- Keep `inward_shell_promotion_weight=0.000375` as the active scalar baseline.
- Move to pair-contribution testing before implementing pair-specific weights.
- Run seed `42` with `outer_to_middle` only and `middle_to_inner` only, both at weight `0.000375` and target-gradient scale `0.0`.
- If one single-pair result approaches or beats the two-pair result, implement pair-specific weights so the stronger edge can be emphasized without strengthening the weaker edge.
- If both single-pair results are below the two-pair result, keep both pairs and consider a weak schedule for the same scalar baseline.

Suggested commands:

```bash
SEEDS="42" WEIGHTS="0.000375" PROMOTION_PAIRS="outer_to_middle" bash scripts/run_codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot.sh
SEEDS="42" WEIGHTS="0.000375" PROMOTION_PAIRS="middle_to_inner" bash scripts/run_codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot.sh
```

## 2026-07-21 Anchored Promotion Single-Pair Results

Question:

- Test whether the current anchored inward shell-promotion gain comes mostly from one shell edge or from the two shell edges acting together.

Mechanism definitions:

- `inward_shell_promotion_weight` is the scalar multiplier on the local predictive-coding energy where a more outer shell state predicts a more inner shell state inside the same column.
- `outer_to_middle` means the column's outer shell state predicts that column's middle shell state.
- `middle_to_inner` means the column's middle shell state predicts that column's inner shell state.
- `inward_shell_promotion_target_gradient_scale` is the multiplier on gradient flowing into the target shell state from this local promotion objective.
- Percentage point means absolute accuracy difference on the CIFAR-10 validation or test percentage.

Configuration:

- Commit: `75bc2292eb9cbf0ac1632a28afff06749d7a17a5`.
- Seed `42`, 20 epochs, learning rate `0.005`.
- `inward_shell_promotion_weight=0.000375`.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- 10 columns, 3 shared columns, 7 active non-shared columns.
- Explicit all-column support mask `1,1,1,1,1,1,1,1,1,1`.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.

Logs:

- `outer_to_middle` master log: `results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairsouter_to_middle_weights0p000375_iptg0p0_rogdora43_20260721_140459.log`.
- `outer_to_middle` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p000375_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260721_140459.log`.
- `middle_to_inner` master log: `results/codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot_pairsmiddle_to_inner_weights0p000375_iptg0p0_rogdora43_20260721_160804.log`.
- `middle_to_inner` child log: `results/codex_dspan_10c_3s_7a_cmall_sm1111111111_shatt_cgstage4_e64_m32_gsum_gp1p0_rs16_ct0p0_sh0_0_0_0_csh0_0_0_0_slr1_1p5_2_3_oct0p0_ocsp0p0_isp0p000375_iptg0p0_ocet0p0_ocbs0p0_sr0_br1_oc1_oce0_ocb0_nobyp_seed42_lr0p005_ep20_nodiag_rogdora43_20260721_160804.log`.

Primary results:

| Promotion pairs | Nodes | Edges | Parameters | Best validation accuracy | Best epoch | Test accuracy | Training time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `outer_to_middle` | 157 | 293 | 3,128,524 | 31.58% | 18 | 31.00% | 7,278.9s |
| `middle_to_inner` | 157 | 293 | 3,126,494 | 31.56% | 18 | 30.65% | 7,284.3s |
| `outer_to_middle,middle_to_inner` active reference | 167 | 313 | 3,129,574 | 33.80% | 18 | 33.74% | 7,306.9s |
| Current zero-promotion control | 147 | 273 | 3,125,444 | 32.60% | 18 | 31.93% | 7,265.5s |

Validation trajectory:

| Epoch | `outer_to_middle` only | `middle_to_inner` only | Two-pair active reference |
| ---: | ---: | ---: | ---: |
| 1 | 11.16% | 11.16% | 11.46% |
| 2 | 21.42% | 20.96% | 21.72% |
| 3 | 20.62% | 20.74% | 20.36% |
| 4 | 25.12% | 23.72% | 26.04% |
| 5 | 24.32% | 24.06% | 25.08% |
| 6 | 27.18% | 26.50% | 27.80% |
| 7 | 22.40% | 25.48% | 26.26% |
| 8 | 24.38% | 27.82% | 30.28% |
| 9 | 24.38% | 24.80% | 27.64% |
| 10 | 25.82% | 24.92% | 26.84% |
| 11 | 20.90% | 20.78% | 22.24% |
| 12 | 26.34% | 27.16% | 27.86% |
| 13 | 27.20% | 27.56% | 29.34% |
| 14 | 26.38% | 27.36% | 28.82% |
| 15 | 26.96% | 28.50% | 28.40% |
| 16 | 28.22% | 25.92% | 29.94% |
| 17 | 29.58% | 28.82% | 32.16% |
| 18 | 31.58% | 31.56% | 33.80% |
| 19 | 30.22% | 31.16% | 32.84% |
| 20 | 30.88% | 31.06% | 33.10% |

Comparisons:

| Run | Test delta from two-pair active reference | Best-validation delta from two-pair active reference | Test delta from zero-promotion control |
| --- | ---: | ---: | ---: |
| `outer_to_middle` only | -2.74 percentage points | -2.22 percentage points | -0.93 percentage points |
| `middle_to_inner` only | -3.09 percentage points | -2.24 percentage points | -1.28 percentage points |
| `outer_to_middle,middle_to_inner` active reference | 0.00 percentage points | 0.00 percentage points | +1.81 percentage points |

Interpretation:

- Neither single promotion edge is enough to reproduce the current best result.
- Both single-edge runs stayed above random chance and improved into the low 30% range by epoch 18, so this does not look like a hard collapse.
- Both single-edge runs finished below the current zero-promotion control, while the two-edge active reference finished above it. This points to an interaction between the two inward shell-promotion edges rather than one edge carrying the benefit alone.
- The result does not support pair-specific weights as the immediate next code change. There is no stronger single edge to emphasize.
- The result does support keeping both inward-promotion pairs active and changing when the shared promotion weight turns on.

Recommended next step:

- Keep `inward_shell_promotion_weight=0.000375` with both `outer_to_middle` and `middle_to_inner` as the active baseline.
- Add a schedule for `inward_shell_promotion_weight`, where the scalar starts at `0.0`, remains off during early classifier formation, then ramps linearly to `0.000375`.
- The first scheduled tests should use seed `42` with two schedules:
  - Warm up for 4 epochs, then ramp linearly to `0.000375` by epoch 12.
  - Warm up for 8 epochs, then ramp linearly to `0.000375` by epoch 16.
- If either schedule beats the static seed-42 result or reduces the epoch-11 validation dip without lowering the final test result, replicate it on seeds `99` and `7`.

Rationale:

- The scalar sweeps show that the useful promotion-weight region is narrow.
- The single-pair results show that the effect depends on both shell edges being present.
- A schedule is the smallest mechanism change that preserves the current HiBaCaML-aligned shell hierarchy while testing whether early promotion pressure is interfering with classifier formation.

## 2026-07-21 Inward Promotion Schedule Implementation

Goal:

- Implement the recommended scheduled inward shell-promotion experiment without changing upstream FabricPC.
- Keep the graph architecture faithful to the current HiBaCaML-aligned column and shell structure.
- Test whether delayed local shell-promotion energy helps the classifier route form before the outer-to-middle and middle-to-inner predictive objectives become active.

Mechanism definitions:

- `inward_shell_promotion_weight` is the final scalar multiplier on the local predictive-coding energy where a more outer shell state predicts a more inner shell state inside the same column.
- `inward_shell_promotion_warmup_epochs` is the number of initial training epochs where the inward shell-promotion objective weight is forced to `0.0`.
- `inward_shell_promotion_ramp_epochs` is the number of epochs after warmup used to linearly increase the inward shell-promotion objective weight from `0.0` to `inward_shell_promotion_weight`.
- `objective_weight` is the per-node scalar stored in the `ShellContextPredictionNode` config and multiplied into that node's Gaussian prediction energy.

Implemented code changes:

- Added schedule validation and epoch-weight calculation in `scripts/train_cifar10_depth_spanning.py`.
- Added `set_inward_shell_promotion_objective_weight`, which returns a copied `GraphStructure` with only inward-promotion nodes' `objective_weight` values changed. The node set, edge set, task map, and node order stay unchanged.
- Added `train_pcn_with_epoch_structures`, a local training wrapper that preserves FabricPC's predictive-coding `train_step` and optimizer update but rebuilds the static JIT closure per epoch with the scheduled graph structure.
- Validation selection now stores both `best_params` and `best_structure`, so a checkpoint selected during a schedule is evaluated with the same inward-promotion objective weight that produced its validation score.
- Final diagnostics use `final_structure` for final-parameter diagnostics and `eval_structure` for selected-checkpoint diagnostics.
- Added command-line flags:
  - `--inward_shell_promotion_warmup_epochs`.
  - `--inward_shell_promotion_ramp_epochs`.
- Extended the shell runner interface with appended positional arguments:
  - Argument 35 is `inward_shell_promotion_warmup_epochs`.
  - Argument 36 is `inward_shell_promotion_ramp_epochs`.
- Added schedule-aware logging to the inward-promotion sweep wrappers.
- Added a new sequential schedule runner: `scripts/run_codex_anchored_inward_shell_promotion_schedule_10col3shared.sh`.

Files changed:

- `scripts/train_cifar10_depth_spanning.py`.
- `scripts/run_codex_cifar10_depth_spanning.sh`.
- `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`.
- `scripts/run_codex_anchored_inward_shell_promotion_10col3shared_weight_sweet_spot.sh`.
- `scripts/run_codex_anchored_inward_shell_promotion_10col3shared_replicate.sh`.
- `scripts/run_codex_anchored_inward_shell_promotion_10col3shared_sweep.sh`.
- `scripts/run_codex_conservative_inward_shell_promotion_10col3shared_sweep.sh`.
- `scripts/run_codex_anchored_inward_shell_promotion_schedule_10col3shared.sh`.
- `tests/test_pooled_readout_norm.py`.

Tests run:

- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py -k "inward_shell_promotion"`: 6 passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_pooled_readout_norm.py`: 72 passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_depth_spanning.py`: passed.
- `bash -n` on the changed runner scripts: passed.
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_depth_spanning.py --help`: passed and showed the new schedule flags.

Experiment command prepared:

```bash
bash scripts/run_codex_anchored_inward_shell_promotion_schedule_10col3shared.sh
```

Default experiment encoded by that command:

- Seed `42`.
- `inward_shell_promotion_weight=0.000375`.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- Schedule A: `inward_shell_promotion_warmup_epochs=4`, `inward_shell_promotion_ramp_epochs=8`. This reaches the final weight at epoch 12.
- Schedule B: `inward_shell_promotion_warmup_epochs=8`, `inward_shell_promotion_ramp_epochs=8`. This reaches the final weight at epoch 16.
- 20 epochs, learning rate `0.005`, 10 columns, 3 shared columns, explicit all-column support mask, `shell_attention`, outer-shell context on, column shell bridge on, no bypass readout, no teacher heads, shell learning-rate multipliers `1,1.5,2,3`.

Expected interpretation:

- If either schedule beats the static seed-42 `0.000375` result of 33.74% test accuracy, replicate that schedule on seeds `99` and `7`.
- If neither schedule beats static `0.000375` but one reduces mid-training validation dips, consider a three-seed stability replicate before rejecting scheduling.
- If both schedules fall below the zero-promotion control at 31.93% seed-42 test accuracy, scheduling is probably not the next useful direction for this mechanism.

## 2026-07-22 Inward Promotion Schedule Result

Question:

- Test whether delayed inward shell promotion improves seed-42 CIFAR-10 classification by letting the classifier pathway form before the local shell-promotion predictive objective becomes fully active.

Mechanism definitions:

- `inward_shell_promotion_weight` is the final scalar multiplier on the local predictive-coding energy where a more outer shell state predicts a more inner shell state inside the same column.
- `warmup_epochs` is the number of initial training epochs with the inward shell-promotion objective weight set to `0.0`.
- `ramp_epochs` is the number of epochs after warmup used to linearly increase the inward shell-promotion objective weight from `0.0` to `inward_shell_promotion_weight`.
- Percentage point means absolute difference between two accuracy percentages.

Execution note:

- The `warmup=4, ramp=8` run started at commit `dfbfe279c105b6b9584bfbfc54844f628b943428` with the schedule implementation present as uncommitted working-tree changes.
- The `warmup=8, ramp=8` run started at commit `6e6d7941e30786be4e8068372c516b8f222a90e2`, which committed the same schedule implementation.
- I am comparing the two runs as intended schedule runs, while noting that the first log's `git_commit` alone does not describe the full executed source state.

Logs:

- Master schedule log: `results/codex_anchored_inward_shell_promotion_schedule_10col3shared_pairsouter_to_middle_middle_to_inner_weights0p000375_schedules4w8_8w8_iptg0p0_rogdora43_20260721_202048.log`.
- `warmup=4, ramp=8` child log: `results/codex_dspan_sched_10c_3s_7a_shatt_isp0p000375_iptg0p0_ipw4_ipr8_seed42_lr0p005_ep20_nodiag_rogdora43_20260721_202048.log`.
- `warmup=8, ramp=8` child log: `results/codex_dspan_sched_10c_3s_7a_shatt_isp0p000375_iptg0p0_ipw8_ipr8_seed42_lr0p005_ep20_nodiag_rogdora43_20260721_231911.log`.

Configuration:

- Seed `42`, 20 epochs, learning rate `0.005`.
- `inward_shell_promotion_weight=0.000375`.
- `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`.
- `inward_shell_promotion_target_gradient_scale=0.0`.
- 10 columns, 3 shared columns, explicit all-column support mask.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.
- Graph size was 167 nodes and 313 edges.
- Parameter count was 3,129,574.

Primary results:

| Run | Warmup epochs | Ramp epochs | Best validation accuracy | Best epoch | Test accuracy | Delta from static `0.000375` test | Delta from zero-promotion control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static active reference | 0 | 0 | 33.80% | 18 | 33.74% | 0.00 pp | +1.81 pp |
| Scheduled promotion | 4 | 8 | 33.46% | 18 | 32.90% | -0.84 pp | +0.97 pp |
| Scheduled promotion | 8 | 8 | 32.18% | 18 | 30.95% | -2.79 pp | -0.98 pp |
| Zero-promotion control | none | none | 32.60% | 18 | 31.93% | -1.81 pp | 0.00 pp |

Validation trajectory:

| Epoch | Static `0.000375` | Warmup 4, ramp 8 | Warmup 8, ramp 8 |
| ---: | ---: | ---: | ---: |
| 1 | 11.46% | 12.04% | 11.10% |
| 2 | 21.72% | 22.12% | 21.62% |
| 3 | 20.36% | 21.68% | 20.26% |
| 4 | 26.04% | 25.96% | 25.28% |
| 5 | 25.08% | 25.68% | 24.44% |
| 6 | 27.80% | 27.36% | 25.82% |
| 7 | 26.26% | 25.76% | 23.56% |
| 8 | 30.28% | 25.34% | 23.42% |
| 9 | 27.64% | 28.06% | 21.08% |
| 10 | 26.84% | 28.28% | 25.92% |
| 11 | 22.24% | 20.16% | 22.42% |
| 12 | 27.86% | 26.76% | 26.82% |
| 13 | 29.34% | 26.98% | 26.98% |
| 14 | 28.82% | 30.18% | 27.24% |
| 15 | 28.40% | 29.64% | 28.72% |
| 16 | 29.94% | 29.30% | 26.90% |
| 17 | 32.16% | 30.74% | 30.18% |
| 18 | 33.80% | 33.46% | 32.18% |
| 19 | 32.84% | 31.74% | 30.56% |
| 20 | 33.10% | 32.78% | 31.24% |

Interpretation:

- The `warmup=4, ramp=8` schedule remained useful. It beat the zero-promotion control by 0.97 percentage points on test accuracy, but it did not beat the static `0.000375` reference.
- The `warmup=8, ramp=8` schedule was too delayed. It finished below the zero-promotion control and trailed static promotion by 2.79 percentage points on test accuracy.
- The schedules did not solve the epoch-11 validation dip. The `warmup=4, ramp=8` run dipped to 20.16% at epoch 11, which was worse than the static run's 22.24%.
- These results suggest that inward shell-promotion pressure is helpful early in training. Delaying it until after early classifier formation does not improve this architecture.
- I would not remove schedule support, because it remains a useful experimental control. I would stop testing long warmup schedules for now.

Recommended next step:

- Keep the static two-pair `0.000375` run as the active baseline.
- Test front-loaded schedules that keep promotion active from epoch 1 but smooth the onset:
  - `warmup=0, ramp=2`.
  - `warmup=0, ramp=4`.
- This checks whether abrupt full-strength promotion is best, or whether a short ramp can preserve early shell alignment while avoiding any initial overconstraint.
- If both front-loaded schedules are below static `0.000375`, stop schedule tuning and move to a different mechanism.

Suggested command:

```bash
SCHEDULES="0:2 0:4" bash scripts/run_codex_anchored_inward_shell_promotion_schedule_10col3shared.sh
```

## 2026-07-22 Front-Loaded Inward Promotion Schedule Result

Context:

- `inward_shell_promotion_weight` is the scalar weight on local predictive objectives where one shell predicts a more inward shell in the same column.
- `warmup` is the number of initial epochs where that scalar is held at zero.
- `ramp` is the number of epochs used to linearly increase the scalar from zero to the requested final value.
- Both runs used the two active promotion pairs `outer_to_middle` and `middle_to_inner`, so the outer shell predicted the middle shell and the middle shell predicted the inner shell.
- Both runs used `inward_shell_promotion_target_gradient_scale=0.0`, so each target shell received no direct gradient from the local promotion objective. The target shell still trained through the rest of the predictive coding graph.

Provenance:

- The `warmup=0, ramp=2` child log recorded commit `6e6d7941e30786be4e8068372c516b8f222a90e2`.
- The `warmup=0, ramp=4` child log recorded commit `e1e9aabc280f925ac5910ed63362dbbd1227d6d5`.
- The only tracked source change between those two commits under `scripts/`, `tests/`, and `docs/dev-plans/` was a work-log update. No training script or runner script changed between the two runs.
- The current commit `d454506` adds the completed front-loaded result logs.

Logs:

- Master schedule log: `results/codex_anchored_inward_shell_promotion_schedule_10col3shared_pairsouter_to_middle_middle_to_inner_weights0p000375_schedules0w2_0w4_iptg0p0_rogdora43_20260722_072648.log`.
- `warmup=0, ramp=2` child log: `results/codex_dspan_sched_10c_3s_7a_shatt_isp0p000375_iptg0p0_ipw0_ipr2_seed42_lr0p005_ep20_nodiag_rogdora43_20260722_072648.log`.
- `warmup=0, ramp=4` child log: `results/codex_dspan_sched_10c_3s_7a_shatt_isp0p000375_iptg0p0_ipw0_ipr4_seed42_lr0p005_ep20_nodiag_rogdora43_20260722_102515.log`.

Configuration:

- Seed `42`, 20 epochs, learning rate `0.005`.
- Final `inward_shell_promotion_weight=0.000375`.
- 10 columns, 3 shared columns, explicit all-column support mask.
- `combiner=shell_attention`, `column_grid=stage4`, `embed_dim=64`, `microcolumn_dim=32`.
- Outer-shell context on, column shell bridge on, no bypass readout, no teacher heads.
- Shell learning-rate multipliers `1,1.5,2,3`.
- Graph size was 167 nodes and 313 edges.
- Parameter count was 3,129,574.

Primary results:

| Run | Warmup epochs | Ramp epochs | Best validation accuracy | Best epoch | Test accuracy | Delta from static `0.000375` test | Delta from zero-promotion control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static active reference | 0 | 0 | 33.80% | 18 | 33.74% | 0.00 percentage points | +1.81 percentage points |
| Scheduled promotion | 4 | 8 | 33.46% | 18 | 32.90% | -0.84 percentage points | +0.97 percentage points |
| Front-loaded schedule | 0 | 2 | 32.68% | 20 | 31.18% | -2.56 percentage points | -0.75 percentage points |
| Front-loaded schedule | 0 | 4 | 30.68% | 18 | 30.50% | -3.24 percentage points | -1.43 percentage points |
| Zero-promotion control | none | none | 32.60% | 18 | 31.93% | -1.81 percentage points | 0.00 percentage points |

Validation trajectory:

| Epoch | Static `0.000375` | Warmup 4, ramp 8 | Warmup 0, ramp 2 | Warmup 0, ramp 4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.46% | 12.04% | 11.42% | 11.08% |
| 2 | 21.72% | 22.12% | 21.82% | 22.18% |
| 3 | 20.36% | 21.68% | 20.80% | 20.92% |
| 4 | 26.04% | 25.96% | 26.10% | 24.90% |
| 5 | 25.08% | 25.68% | 22.84% | 24.36% |
| 6 | 27.80% | 27.36% | 26.42% | 28.36% |
| 7 | 26.26% | 25.76% | 22.16% | 22.54% |
| 8 | 30.28% | 25.34% | 26.04% | 26.94% |
| 9 | 27.64% | 28.06% | 25.76% | 22.24% |
| 10 | 26.84% | 28.28% | 26.08% | 26.10% |
| 11 | 22.24% | 20.16% | 19.92% | 23.30% |
| 12 | 27.86% | 26.76% | 24.42% | 22.88% |
| 13 | 29.34% | 26.98% | 26.76% | 27.30% |
| 14 | 28.82% | 30.18% | 26.88% | 28.58% |
| 15 | 28.40% | 29.64% | 28.84% | 27.06% |
| 16 | 29.94% | 29.30% | 28.22% | 27.66% |
| 17 | 32.16% | 30.74% | 29.76% | 29.70% |
| 18 | 33.80% | 33.46% | 32.06% | 30.68% |
| 19 | 32.84% | 31.74% | 31.52% | 29.62% |
| 20 | 33.10% | 32.78% | 32.68% | 30.54% |

Interpretation:

- Front-loaded scheduling did not improve the active mechanism. The `warmup=0, ramp=2` run was close to the zero-promotion control on validation accuracy, but it trailed the zero-promotion control by 0.75 percentage points on test accuracy.
- The `warmup=0, ramp=4` run was clearly worse. Its test accuracy was 30.50%, which was 3.24 percentage points below the static `0.000375` active reference.
- The best schedule remains `warmup=4, ramp=8`, but that still trails the static `0.000375` run by 0.84 percentage points on test accuracy.
- These results argue against spending more experiment time on the timing of this one scalar. The current local promotion objective is most useful when present at full strength from the beginning.

Recommended next step:

- Keep static `inward_shell_promotion_weight=0.000375`, `inward_shell_promotion_pairs=outer_to_middle,middle_to_inner`, and `inward_shell_promotion_target_gradient_scale=0.0` as the active baseline.
- Stop schedule tuning for now.
- Move to a mechanism change: make promoted shell evidence available to the classifier path, not only to the local predictive energy. The concrete candidate is a promoted-shell bridge that maps source-shell activity into target-shell width, trains with the same local predictive objective, and also exposes its promoted representation to the existing shell-preserving bridge or readout.
- The reason is architectural. In the current implementation, inward promotion can shape local prediction energy, but the readout still primarily sees the ordinary shell states. If promoted shell content is meant to act as a consolidated inward representation, it needs a route into the classification path while preserving the hard-kernel, inner-shell, middle-shell, and outer-shell identities.

Pause point:

- I will wait for confirmation before making code changes for the promoted-shell bridge.

## 2026-07-22 Promoted Shell Bridge Infrastructure

Context:

- `inward_shell_promotion_weight` is the scalar weight on local predictive objectives where one wider shell predicts an adjacent more inward shell in the same column.
- `target_gradient_scale` is the multiplier on the inference gradient returned from a promotion objective into the target shell. The active baseline uses `target_gradient_scale=0.0`, so the target shell is not moved directly by the promotion objective.
- `promoted_shell_bridge` is the new optional route that makes promoted shell predictions available to the classifier path.
- `embed_dim` is the full per-column feature width. It is partitioned into hard-kernel, inner-shell, middle-shell, and outer-shell feature slices by `get_shell_slices(embed_dim)`.

Implementation summary:

- Added `PromotedShellPredictionNode` in `columnar_cl_fabricpc/columns/accuracy_nodes.py`.
  - It receives a `target` shell vector and one or more `context` shell vectors.
  - It computes a prediction from the context vectors.
  - It keeps the local target-prediction energy used by inward shell promotion.
  - It also keeps its own latent anchored to that prediction, so downstream graph edges receive the promoted prediction rather than the target shell.
- Added `PromotedShellBridgeNode` in `columnar_cl_fabricpc/columns/accuracy_nodes.py`.
  - It receives raw pooled shell vectors and promoted prediction latents.
  - A raw `middle_shell` pool writes only to the middle-shell slice of the full bridge output.
  - An `outer_shell_to_middle_shell` promoted prediction writes only to the middle-shell slice of the full bridge output.
  - Inputs in the same target shell are averaged by `1/sqrt(n)`, where `n` is the number of inputs that write to that shell.
  - The output has shape `(embed_dim,)`, so it can feed the existing `output` classifier as a standard predictive-coding latent.
- Added `--promoted_shell_bridge` to `scripts/train_cifar10_depth_spanning.py`.
  - The flag requires positive `--inward_shell_promotion_weight`.
  - When the flag is off, existing inward promotion uses `ShellContextPredictionNode` and prior commands keep the old topology.
  - When the flag is on, inward promotion uses `PromotedShellPredictionNode`, and each active column also gets one `columnXX_promoted_shell_bridge` node feeding `output`.
- Updated shell learning-rate multiplier infrastructure.
  - Promoted prediction parameters use the target shell's multiplier.
  - Promoted bridge weights and biases use the multiplier of the shell slice they write into.
- Updated diagnostics.
  - Energy and latent diagnostics now include promoted-shell bridge nodes.
  - Readout ablations now include `promoted_shell_bridge_only`, `column_pool_plus_promoted_shell_bridge`, and `combined_without_promoted_shell_bridge`.
  - Promoted bridge ablations now mask raw and promoted bridge inputs by target shell.
- Updated runner support.
  - `scripts/run_codex_cifar10_depth_spanning.sh` accepts positional argument 37 for promoted bridge mode.
  - `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh` accepts `PROMOTED_SHELL_BRIDGE=on`.

Current source state:

- Base commit before this work-log entry: `2d656a3f55bb6b56b981e172a4cbc45877c26ff9`.
- Modified local files:
  - `columnar_cl_fabricpc/columns/accuracy_nodes.py`.
  - `columnar_cl_fabricpc/columns/__init__.py`.
  - `scripts/train_cifar10_depth_spanning.py`.
  - `scripts/run_codex_cifar10_depth_spanning.sh`.
  - `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`.
  - `tests/test_pooled_readout_norm.py`.
  - This work log.

Validation:

- `python -m py_compile` passed for `scripts/train_cifar10_depth_spanning.py`, `columnar_cl_fabricpc/columns/accuracy_nodes.py`, and `columnar_cl_fabricpc/columns/__init__.py`.
- `bash -n` passed for `scripts/run_codex_cifar10_depth_spanning.sh` and `scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh`.
- `pytest tests/test_pooled_readout_norm.py` passed with 77 tests.
- `git diff --check` passed.
- A CLI help check confirmed `--promoted_shell_bridge` is registered.

Recommended first experiment:

- Run one direct comparison against the active static-promotion baseline by enabling the promoted bridge while keeping the same 10-column, 3-shared, two-pair, `0.000375` promotion configuration.
- This run intentionally keeps the ordinary `column_shell_bridge` on. The first question is whether the promoted bridge adds useful evidence without removing the previously useful bridge path.

Suggested command:

```bash
PROMOTED_SHELL_BRIDGE=on WEIGHTS="0.000375" PROMOTION_PAIRS="outer_to_middle,middle_to_inner" INWARD_SHELL_PROMOTION_TARGET_GRADIENT_SCALE=0.0 SEED=42 LR=0.005 NUM_EPOCHS=20 POST_TRAINING_DIAGNOSTICS=core bash scripts/run_codex_inward_shell_promotion_10col3shared_sweep.sh
```
