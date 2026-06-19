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
- The escalated run reported TensorFlow CUDA library loading failures and JAX still executed as CPU from this process.

The laptop fan and very high load average may still reflect heavy CPU execution or another process outside this session. From the commands launched here, JAX did not expose a GPU backend. The machine clearly has NVIDIA hardware, but the active Python process did not have a working driver or CUDA library path.

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

## Recommended Next Steps

First, confirm GPU visibility from the same shell and virtual environment that will run the experiments:

```bash
nvidia-smi -L
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python - <<'PY'
import jax
print(jax.devices())
print(jax.default_backend())
PY
```

The target condition is that JAX reports a CUDA GPU device, not only `CpuDevice(id=0)`.

Once JAX sees the GPU, run the upstream-style PC convolutional baseline from the columnar repo:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --num_epochs 2 --batch_size 256 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

The upstream FabricPC convolution demo docstring reports 40.89% test accuracy for a two-epoch quick run. That is the first sanity threshold to reproduce before changing the columnar architecture.

If the ResNet18 PC baseline reproduces substantially above chance, run a small activation and inference sweep:

```bash
for activation in tanh relu gelu leaky_relu; do
  /home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_pc_resnet.py --model resnet18 --activation "$activation" --num_epochs 2 --batch_size 256 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
done
```

Then compare `infer_steps=40` and `infer_steps=80` for the best activation. The mechanism being tested is whether PC inference has enough latent update steps to settle output and intermediate states before local weight updates are computed.

After the PC convolutional baseline is validated, port the useful mechanism into the columnar graph locally:

- Replace the local manual `ConvStemNode` with a stem assembled from upstream `ConvNode` and `AvgPool` nodes.
- Keep a single 10-class `TaskMap(x=image, y=classifier)`.
- Keep `Cifar10Loader`; do not add a loader wrapper from cFabricPC.
- Do not add split-task masks, replay buffers, memory readout, task-local heads, or support-column selection yet.

Only after plain CIFAR-10 is reliably above chance should the split-CIFAR and continual-learning mechanisms from `cFabricPC` be revisited.
