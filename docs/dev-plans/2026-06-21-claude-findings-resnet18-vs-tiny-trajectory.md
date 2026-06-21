# Findings: resnet18 vs tiny — column-output trajectory during training

Date: 2026-06-21
Venv: `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python`
Inputs:
- `results/depth_spanning_tiny_1epoch_diag_20260621_132537.log`
- `results/depth_spanning_resnet18_1epoch_diag_20260621_132547.log`

These two 1-epoch runs use the per-epoch instrumentation added to `scripts/train_cifar10_depth_spanning.py` (functions `diagnose_energy_breakdown` and `diagnose_column_outputs`). All other hyperparameters are identical: `--num_columns 4`, `--column_mode all_active`, `--combiner sum`, batch 128, lr 0.01, infer_steps 40.

## Observed outcomes

| model | test acc | val acc | wall clock |
|---|---:|---:|---:|
| tiny | 16.96% | 17.80% | 234.7 s |
| resnet18 | 10.00% | 9.40% | 428.8 s |

Both miss the PC ResNet baseline (37.9% at 3 epochs / 42.45% reported elsewhere). 10-class random chance is 10%; resnet18 sits at chance, tiny is ~7 points above.

## E_gauss / E_ce trajectory

| | tiny: init | tiny: end of epoch 1 | resnet18: init | resnet18: end of epoch 1 |
|---|---:|---:|---:|---:|
| backbone | 0.0002 | 1.9285 | 0.0000 | 0.0048 |
| stage_taps | 0.0000 | 0.0024 | 0.0000 | 0.0000 |
| columns | 0.0001 | 0.0301 | 0.0000 | 0.0000 |
| combiner | 57.2745 | 20.2232 | 60.9907 | 21.2376 |
| classifier (E_ce) | 160.9080 | 9.1301 | 148.8664 | 9.2436 |
| **E_gauss** | 57.27 | 22.18 | 60.99 | 21.24 |
| **E_gauss / E_ce** | **0.36** | **2.43** | **0.41** | **2.30** |

The pre-training measurement that the prior session's plan was built on — both models showing `E_gauss / E_ce < 0.5` — does not survive a single epoch of training. After one epoch, both models cross over to Gauss-dominated regime. The end-of-epoch ratios are essentially identical for the two models (2.43 vs 2.30); the energy-dominance criterion does not separate the working configuration from the failing one.

Mechanistically, what happens during epoch 1: the cross-entropy energy at the classifier collapses by ~16× in both models (160.9 → 9.1 for tiny, 148.9 → 9.2 for resnet18) as the readout learns. The combiner Gaussian energy also drops but only ~3× (57 → 20 for both). So the cross-over to Gauss-dominance comes from `E_ce` falling fast while `E_combiner` falls slowly, in both models. This is a property of the loss landscape, not of the failure mode.

## Column output magnitudes — where the two models actually differ

`z_latent` standard deviation for the four `col_*` nodes, on the same diagnostic batch:

| | tiny: init | tiny: end of epoch 1 | resnet18: init | resnet18: end of epoch 1 |
|---|---:|---:|---:|---:|
| col_00 std | 0.0065 | 2.0709 | 0.0010 | 0.1249 |
| col_01 std | 0.0057 | 0.9111 | 0.0009 | 0.1367 |
| col_02 std | 0.0063 | 1.4211 | 0.0010 | 0.2385 |
| col_03 std | 0.0059 | 1.2465 | 0.0009 | 0.1307 |

Two facts:

1. **At init, resnet18 columns are ~6× narrower than tiny columns.** `col_*` std is around 0.001 for resnet18 and 0.006 for tiny. Walking up the graph, the stage taps are ~3–5× narrower for resnet18 (e.g. `stage2_tap` std: tiny 0.109 vs resnet18 0.020), and `stage4_pool` std collapses by 50× (tiny 0.021 vs resnet18 0.0004).

2. **After one epoch, resnet18 columns have grown to std ≈ 0.13–0.24, tiny columns have grown to std ≈ 0.9–2.07.** The tiny columns have reached an order of magnitude larger amplitude than the resnet18 ones.

The combiner sums the column outputs (in sum-combiner mode) and converges its z_latent to that sum. The combiner z_latent std for resnet18 after epoch 1 is **0.0031**, essentially unchanged from its init value of 0.0026. For tiny it grew from 0.001 to 0.043. So:

- In tiny, the combiner is delivering a non-zero, structured signal to `column_pool → output`. Classifier sees variation; accuracy can rise above chance.
- In resnet18, the combiner z_latent std is at the noise floor. The signal reaching `column_pool → output` is essentially zero. Classifier sees a near-constant input batch-to-batch and converges to predicting the marginal class distribution.

The 10.00% test accuracy on resnet18 is the marginal-distribution baseline (uniform over 10 classes). The 9.40% val acc is consistent.

## Why resnet18 columns start (and stay) smaller

This is the mechanism question. The structural facts:

- Resnet18 backbone: 4 stages with strides 1/2/2/2, output `(4,4,256)`. Two residual blocks per stage. No batch / layer normalization in the configured `make_residual_block`.
- Tiny backbone: 3 stages with strides 1/2/2, output `(8,8,64)`. One residual block per stage.
- All convs initialized with `MuPCInitializer`. All stage tap projection weights initialized with `KaimingInitializer` (fan_in mode is the default).

The depth difference matters at init for two reasons read directly from the code:

1. **More layers between input and the stage outputs.** Each ConvNode applies `pre_activation → activation`, and Kaiming-style init aims to preserve activation variance through ReLU-family activations *with batch normalization*. Without BN, even Kaiming init does not perfectly preserve variance across a stack of layers; small mismatches compound. The resnet18 backbone has 8 residual blocks (16 convs + skip projections) vs 3 for tiny, so the variance drift compounds longer.

2. **Higher channel count at deeper stages.** The stage taps project from `source_channels` to `embed_dim=64`. Resnet18's stage4 has 256 channels; tiny's has 64. KaimingInitializer at the projection scales weights by `1/sqrt(fan_in)`, which is 1/16 for resnet18 stage4 vs 1/8 for tiny. The projected output's std is `input_std × sqrt(fan_in) / sqrt(fan_in) = input_std` if the input is unit variance — but with the upstream variance already reduced, the projected output is correspondingly smaller. The `stage4_pool` numbers (tiny 0.021 vs resnet18 0.0004, a 50× ratio) are the most extreme example: a `GlobalPool → Linear(256→64)` after a 4-block deep stack produces a tiny output.

Per training step, only the gradient through the classifier reaches the columns. With the column outputs at std ≈ 0.001 and the combiner z_latent essentially zero, the classifier's gradient is small relative to the parameter scale, and the column parameters take 350 batches (one epoch) to reach std ≈ 0.13 — still small in absolute terms. The tiny model's columns reach std ≈ 1–2 in the same number of batches because they started 6× larger and travel along a steeper region of the loss surface.

## What the data does *not* say

It does not say that "Gaussian energy dominates" causes the resnet18 failure. Both models end up Gauss-dominated by the same factor; only one of them performs above chance.

It does not say that depth-spanning columns are intrinsically broken. The tiny configuration is not failing for the same reason resnet18 is: tiny columns reach reasonable magnitudes; the limit there is whatever keeps the architecture at 17% rather than the PC ResNet baseline of 38%, which this diagnostic does not address.

It does not say that 10 epochs would fix resnet18. The 1-epoch run has already saturated the cross-entropy loss reduction (148.87 → 9.24, similar to tiny's 160.91 → 9.13). The combiner z_latent remained at noise floor throughout. Without an architectural change, additional epochs would change the weight magnitudes inside the columns but not deliver more signal to the classifier; the prior session's 10-epoch resnet18 run at 6.68% is consistent with this.

## What this implies for "improve CIFAR-10 accuracy"

The current architecture, run on plain (single-task) CIFAR-10:

- `resnet18` backbone + depth-spanning columns: at chance (10%) at 1 epoch, below chance (6.68%) at 10 epochs.
- `tiny` backbone + depth-spanning columns: 17% at 1 epoch, 17% at 10 epochs.
- PC ResNet (no columns at all): 37.9% at 3 epochs, 42.45% reported elsewhere.

The columns reduce accuracy from 38–42% to 17% on the tiny backbone, and from 38–42% to chance on the resnet18 backbone. On the same task, with the same dataset, adding the columnar pathway hurts. This is consistent with the HiBaCaML paper's framing (`hibacaml_agi26.pdf`, §5): the paper presents the architecture as preparing for *continual* learning (Split-CIFAR) and explicitly says "We treat these results as architectural validation rather than as a continual-learning leaderboard claim." The paper does not predict that adding ColBa columns improves plain CIFAR-10 single-task accuracy over a direct classifier.

The current code path has been running plain CIFAR-10 in `train_cifar10_depth_spanning.py` (single train/val/test split, single-task `train_pcn`). The HiBaCaML setup the paper is designed for is the Split-CIFAR-10 task-incremental protocol with five binary tasks and task-local heads.

## Three options the user may want to choose between

1. **Pursue the architecture as designed by reproducing the Split-CIFAR-10 protocol.** Build a five-binary-task train loop with task-local heads (`columnar_cl_fabricpc/experiments/`), evaluate forgetting and forward transfer, treat columnar machinery as the cross-task scaffold. Plain CIFAR-10 accuracy stops being the success metric; the paper's metrics take over. This aligns the goal with the architecture.

2. **Keep plain CIFAR-10 as the metric, but redesign the column pathway so it adds capacity instead of bottlenecking signal.** Two concrete failures the diagnostic identifies that would need fixing: (a) the stage taps lose 5–50× variance through the resnet18 backbone, so add normalization (layer norm on tokens, or rescale projections by the measured backbone-output std); (b) the combiner in sum mode at `(num_tokens=16, embed_dim=64)` lets four columns interfere destructively, so consider concat-then-project or attention combiner with residual to the stem. Neither is a small change.

3. **Drop the depth-spanning columns from the plain-CIFAR-10 metric and run the PC ResNet (or the paper's recommended shallow stem) longer.** Use the columnar work as a parallel track scoped to Split-CIFAR, and treat plain CIFAR-10 as a backbone-quality question handled by `train_cifar10_pc_resnet.py` and its variants. The 42.45% baseline is the current ceiling there and improving it is a different research question (longer training, BN, augmentation, infer_steps tuning).

These three options are not exhaustive and not ranked. The choice depends on what the user is actually trying to demonstrate, which is not derivable from the code or data alone.
