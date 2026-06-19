# Plain CIFAR-10 PC Work Record

Local timestamp: 2026-06-19 04:33:56 EDT -0400.

Machine identifier: `rogdora43`.

Kernel: `Linux rogdora43 7.0.11-100.fc43.x86_64 #1 SMP PREEMPT_DYNAMIC Mon Jun 1 22:51:40 UTC 2026 x86_64 GNU/Linux`.

CPU: AMD Ryzen 9 8940HX with Radeon Graphics, 16 cores, 32 threads.

GPU hardware visible by PCI: NVIDIA GB206M GeForce RTX 5060 Max-Q / Mobile and AMD Raphael integrated graphics.

GPU software visibility from this environment:

- `nvidia-smi -L` failed with: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`
- `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python` reported `jax.device_count() == 1`, `jax.devices() == [CpuDevice(id=0)]`, and `jax.default_backend() == "cpu"`.
- The JAX CUDA plugin reported `CUDA_ERROR_NO_DEVICE` in sandboxed runs.
- The escalated experiment runs still printed TensorFlow CUDA library loading failures, but this warning did not match the JAX runtime behavior observed from the user's shell.

GPU software visibility from the user's interactive shell:

- `nvidia-smi` succeeded on `rogdora43` at Fri Jun 19 04:30:01 2026.
- NVIDIA driver version: 595.71.05.
- CUDA version reported by the driver: 13.2.
- GPU: NVIDIA GeForce RTX 5060, 8,151 MiB.
- GPU state at that moment: 2 MiB memory used, 11% utilization, no listed compute processes.
- `python -c "import jax; print(jax.devices()); print(jax.default_backend())"` from the activated `fpcpy3.12` environment reported `[CudaDevice(id=0)]` and `gpu`.
- During the activation sweep, `nvidia-smi` at Fri Jun 19 05:23:05 2026 showed `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python` using 4,816 MiB of GPU memory with 100% GPU utilization.

The key distinction is process-local GPU visibility. The machine has working NVIDIA hardware, the user's interactive shell can talk to the driver, and JAX in the user's activated `fpcpy3.12` environment selects the GPU backend. The CUDA library warning observed in command output is therefore not sufficient evidence that the PC experiment is CPU-only. In the completed sweep, `nvidia-smi` confirmed that the Python experiment process was using the NVIDIA GPU.

## Repository State

Columnar repository: `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`.

Columnar branch: `improve/plain-cifar10`.

Columnar pending changes before this note:

- `AGENTS.md` staged as an added file. I did not create or modify it.
- `CLAUDE.md` staged as an added file. I did not create or modify it.
- `columnar_cl_fabricpc.egg-info/SOURCES.txt` modified. This was already dirty before my current implementation work.
- `scripts/train_cifar10_pc_resnet.py` added by me, still untracked at the time of this note.

FabricPC repository: `/home/ni/repos/fpc/FabricPC`.

FabricPC branch: `feature/convolution`, tracking `origin/feature/convolution`.

FabricPC commit: `8781f67 Final?`.

FabricPC status: clean after switching branches. I did not edit FabricPC.

Python environment used: `/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python`.

JAX versions from that environment: `jax 0.10.1`, `jaxlib 0.10.1`.

## History Findings

The newer `columnarCL-fabricPC-experiments` repository has a short local history. The useful high-performing result found earlier was in the separate `cFabricPC` repository, not in this repo.

The best recorded `cFabricPC` result I found was commit `de60ba8`, but it was a split-CIFAR-10 memory-readout result. That is not the current target because the current target is one plain CIFAR-10 task with one 10-class classifier.

The `de60ba8` sweep recorded cross-seed split-CIFAR memory-readout accuracy around 65.64% mean across seeds 42, 7, and 99. I did not port that split-task mechanism after the direction changed.

No cFabricPC loader was ported. The columnar repo continues to use FabricPC's existing `Cifar10Loader`.

## Work Completed

I created branch `improve/plain-cifar10` in `columnarCL-fabricPC-experiments`.

I initially added a backprop mode to `scripts/train_cifar10_conv_stem.py` while adapting the non-split cFabricPC joint experiment. You clarified that the goal is predictive-coding mechanisms, not plain backprop. I removed that backprop change. `scripts/train_cifar10_conv_stem.py` is back to its original tracked content.

I inspected FabricPC branches and found that `origin/feature/convolution` is the upstream branch with the real convolution support:

- `fabricpc.nodes.ConvNode`.
- `fabricpc.nodes.MaxPool`.
- `fabricpc.nodes.AvgPool`.
- `fabricpc.nodes.SkipConnection`.
- ND-aware Xavier and Kaiming initializer tests.
- `examples/resnet18_cifar10_demo.py`, a predictive-coding CIFAR-10 convolutional baseline.

I switched the FabricPC working tree to local branch `feature/convolution` tracking `origin/feature/convolution`. I did not edit FabricPC.

I added `scripts/train_cifar10_pc_resnet.py` in the columnar repo. This script builds a plain CIFAR-10 predictive-coding convolutional graph using upstream FabricPC convolution components:

- `IdentityNode` image input with shape `(32, 32, 3)`.
- `ConvNode` stem and residual-block convolutions.
- `SkipConnection` residual summation points.
- `AvgPool(global_pool=True)` before the classifier.
- One `Linear(shape=(10,), activation=SoftmaxActivation(), energy=CrossEntropyEnergy())` output head.
- `MuPCConfig(include_output=False)` for predictive-coding scaling.
- `InferenceSGDNormClip` for PC inference.
- `train_pcn` and `evaluate_pcn` for training and evaluation.
- `Cifar10Loader` from FabricPC with `tensor_format="NHWC"`.

The script has two graph sizes:

- `--model tiny`: a small smoke-test graph.
- `--model resnet18`: the real default baseline modeled after FabricPC's upstream CIFAR-10 ResNet-18 demo.

It has no split-CIFAR machinery, no task masks, no replay state, no memory readout, and no cFabricPC data loader.

## Verification Completed

Syntax checks passed:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile scripts/train_cifar10_pc_resnet.py scripts/train_cifar10_conv_stem.py
```

Focused tests passed:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m pytest tests/test_visual_stem.py tests/test_column.py tests/test_combiner.py tests/test_cifar10_model.py
```

Result: 64 passed.

Smoke run completed:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --quick --batch_size 512
```

Smoke result:

- Model: `tiny`.
- Epochs: `0.01`.
- Inference steps: `2`.
- One capped training batch.
- Full test evaluation completed.
- Test accuracy: 10.62%.

This is only a pipeline validation result. It verifies graph construction, initialization, data loading, PC training, and PC evaluation. It is not a meaningful CIFAR-10 training result.

Attempted real run:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --num_epochs 2 --batch_size 256 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

The run started and built the expected graph:

- 31 nodes.
- 38 edges.
- 2,795,210 parameters.
- 176 train batches, 20 validation batches, 40 test batches.

It was stopped because this session saw only the CPU backend. After JIT compilation, it was still running too slowly for an interactive check.

Attempted short tiny run:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model tiny --num_epochs 0.1 --batch_size 512 --infer_steps 10 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 0
```

This run was killed by the environment with exit code 137 before producing metrics.

## Experiment Sweep Completed

Sweep output directory: `results/plain_cifar10_pc_sweep_20260619_052201`.

All sweep runs used the same plain CIFAR-10 predictive-coding graph:

- Model: `resnet18`.
- Batch size: 256.
- Epochs: 2.
- Learning rate: 0.01.
- Weight decay: 0.01.
- Predictive-coding inference step size: `eta_infer=0.1`.
- Evaluation: after each epoch and once on the test set.
- Data loader: upstream FabricPC `Cifar10Loader` with NHWC tensors.
- Classifier: one 10-way softmax cross-entropy head.

Activation sweep at `infer_steps=80`:

| Activation | Epoch 1 validation accuracy | Epoch 2 validation accuracy | Test accuracy | Training time | Log |
| --- | ---: | ---: | ---: | ---: | --- |
| `tanh` | 0.3480 | 0.3814 | 0.3751 | 1127.3s | `activation_tanh_infer80.log` |
| `relu` | 0.2880 | 0.3578 | 0.3726 | 1130.9s | `activation_relu_infer80.log` |
| `gelu` | 0.3624 | 0.3968 | 0.3981 | 1130.9s | `activation_gelu_infer80.log` |
| `leaky_relu` | 0.3726 | 0.4226 | 0.4245 | 1129.5s | `activation_leaky_relu_infer80.log` |

Inference-step comparison for the best activation:

| Activation | Inference steps | Epoch 1 validation accuracy | Epoch 2 validation accuracy | Test accuracy | Training time | Log |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `leaky_relu` | 80 | 0.3726 | 0.4226 | 0.4245 | 1129.5s | `activation_leaky_relu_infer80.log` |
| `leaky_relu` | 40 | 0.3306 | 0.3690 | 0.3750 | 601.3s | `infer40_leaky_relu.log` |

The best result in this sweep is `leaky_relu` with 80 predictive-coding inference steps: 42.45% test accuracy after two epochs. This is above the upstream FabricPC convolution demo docstring's reported two-epoch test accuracy of 40.89%.

Reducing the inference loop from 80 steps to 40 steps reduced training time from 1129.5 seconds to 601.3 seconds, but it also reduced test accuracy from 42.45% to 37.50%. Under these settings, the predictive-coding state update needs the longer 80-step inference loop to settle enough before weight updates.

## Recommended Next Steps

Use `leaky_relu` with `infer_steps=80` as the current plain CIFAR-10 predictive-coding baseline.

Run a slightly longer baseline before changing the architecture:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --activation leaky_relu --num_epochs 5 --batch_size 256 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

If five epochs still plateaus early, tune the predictive-coding inference and update hyperparameters one variable at a time:

- `eta_infer`: compare 0.05, 0.1, and 0.2 with `leaky_relu` and `infer_steps=80`.
- `lr`: compare 0.003, 0.01, and 0.03 after choosing `eta_infer`.
- `weight_decay`: compare 0.0, 0.001, and 0.01 after choosing `lr`.

Keep the sweep local to `columnarCL-fabricPC-experiments`. Do not modify FabricPC unless a missing upstream hook is proven necessary.

After the PC convolutional baseline is validated, port the useful mechanism into the columnar graph locally:

- Replace the local manual `ConvStemNode` with a stem assembled from upstream `ConvNode` and `AvgPool` nodes.
- Keep a single 10-class `TaskMap(x=image, y=classifier)`.
- Keep `Cifar10Loader`; do not add a loader wrapper from cFabricPC.
- Do not add split-task masks, replay buffers, memory readout, task-local heads, or support-column selection yet.

Only after plain CIFAR-10 is reliably above chance should the split-CIFAR and continual-learning mechanisms from `cFabricPC` be revisited.

## Alternatives Considered

Shortening predictive-coding inference from 80 steps to 40 steps:

- Pro: reduced two-epoch training time from 1129.5 seconds to 601.3 seconds for `leaky_relu`.
- Con: reduced test accuracy from 42.45% to 37.50%.
- Decision: do not use 40 steps as the baseline.

Porting the `cFabricPC` split-CIFAR memory-readout machinery now:

- Pro: the `de60ba8` result is the best historical result found so far.
- Con: it solves a split-CIFAR memory-readout setup, not the current one-task CIFAR-10 classification target.
- Decision: defer until the plain CIFAR-10 classifier is stronger.

Modifying FabricPC directly:

- Pro: could expose missing convolution hooks upstream if the baseline requires them.
- Con: the needed convolution support already exists on FabricPC branch `feature/convolution`.
- Decision: do not modify FabricPC for this phase.

Using plain backpropagation as a comparison:

- Pro: would be a familiar CIFAR-10 control.
- Con: the current objective is predictive-coding classification, and adding backprop would split the training mechanism being evaluated.
- Decision: do not add a backprop path.
