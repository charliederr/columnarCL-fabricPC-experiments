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
