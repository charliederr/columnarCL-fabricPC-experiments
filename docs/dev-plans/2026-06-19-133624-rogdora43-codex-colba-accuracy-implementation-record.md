# Codex ColBa Accuracy Implementation Record

Local timestamp: 2026-06-19 13:36:24 EDT -0400.

Machine identifier: `rogdora43`.

Kernel: `Linux rogdora43 7.0.11-100.fc43.x86_64 #1 SMP PREEMPT_DYNAMIC Mon Jun 1 22:51:40 UTC 2026 x86_64 GNU/Linux`.

CPU: AMD Ryzen 9 8940HX with Radeon Graphics, 32 logical CPUs.

Repository: `/home/ni/repos/fpc/columnarCL-fabricPC-experiments`.

Branch: `improve/plain-cifar10`.

Python environment: `/home/ni/repos/fpc/virt-envs/fpcpy3.12`.

## Objective

Implement the first plain CIFAR-10 accuracy mechanisms from `docs/dev-plans/2026-06-19-094051-rogdora43-codex-hibacaml-colba-cifar10-accuracy-plan.md`.

The target is predictive-coding CIFAR-10 classification, not split-CIFAR. The implementation uses FabricPC `train_pcn`, FabricPC `ConvNode`, FabricPC `SkipConnection`, FabricPC `AvgPool`, and the upstream FabricPC `Cifar10Loader`. No FabricPC source files were edited.

## Implemented Files

`columnar_cl_fabricpc/columns/accuracy_nodes.py` was added.

It contains three local node types:

- `FeatureTokenizerNode`: maps the final convolutional feature map to a token sequence.
- `TypedColBaColumnNode`: implements K, L, and B microcolumn paths. K is a per-token residual MLP, L is 3 by 3 local token-grid mixing, and B is global token pooling with broadcast back to each token.
- `MaskedColumnCombinerNode`: combines column outputs through a fixed support mask. The support mask `m` has one entry per column, where `m_c = 1` means column `c` contributes to the combiner output and `m_c = 0` means column `c` is excluded from the combiner.

`scripts/train_cifar10_colba_accuracy.py` was added.

It builds a plain CIFAR-10 predictive-coding graph:

```text
image -> FabricPC ResNet-style conv stack -> feature tokens -> typed ColBa columns -> masked combiner -> pooled column vector -> 10-way classifier
```

The first version fed all 16 column tokens directly into the classifier. That version failed at chance. The script was then revised to add a global `AvgPool` after the masked combiner so the classifier receives a 256-dimensional pooled vector, matching the interface shape used by the working PC ResNet baseline.

## Verification Commands

Syntax check:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python -m py_compile columnar_cl_fabricpc/columns/accuracy_nodes.py scripts/train_cifar10_colba_accuracy.py
```

Quick smoke after adding the pooled classifier path:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_colba_accuracy.py --quick --batch_size 512
```

Result: completed with test accuracy 9.89%. This quick mode trains one tiny batch with two inference steps, so it only verifies graph construction, compilation, training, and evaluation.

## Experiment Results

Result directory for the first all-active ColBa run:

`results/plain_cifar10_colba_accuracy_20260619_102323`

Command:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_colba_accuracy.py --model resnet18 --activation leaky_relu --column_activation leaky_relu --num_columns 40 --column_mode all_active --num_epochs 2 --batch_size 128 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

Result:

- Graph: 72 nodes, 118 edges, 7,525,738 parameters.
- Epoch 1 validation accuracy: 10.06%.
- Epoch 2 validation accuracy: 10.06%.
- Test accuracy: 10.00%.
- Exit status: 0.

Interpretation: the unpooled all-active column layer failed the Phase 2 acceptance rule. It did not preserve the PC ResNet baseline behavior.

Result directory for the pooled revision:

`results/plain_cifar10_colba_pooled_accuracy_20260619_132421`

Attempted full pooled command:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_colba_accuracy.py --model resnet18 --activation leaky_relu --column_activation leaky_relu --num_columns 40 --column_mode all_active --num_epochs 2 --batch_size 128 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

Result:

- Graph: 73 nodes, 119 edges, 7,487,338 parameters.
- Interrupted with status 130 by Codex.
- Reason: this Codex process reported `[CpuDevice(id=0)]` and backend `cpu`; steady-state ResNet batch time was about 40 seconds. The same command should be run from a shell where JAX reports `[CudaDevice(id=0)]` and backend `gpu`.

Tiny pooled CPU diagnostic:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_colba_accuracy.py --model tiny --activation leaky_relu --column_activation leaky_relu --num_columns 6 --num_shared 2 --active_nonshared 2 --column_mode all_active --num_tokens 64 --embed_dim 64 --microcolumn_dim 16 --num_epochs 1 --batch_size 512 --infer_steps 20 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

Result:

- Graph: 23 nodes, 30 edges, 173,442 parameters.
- Energy dropped from 1.7185 at batch 1 to about 0.7648 by batch 38.
- The process was killed with exit status 137 before validation and test metrics were produced.

## Current Assessment

The implementation covers Phase 2 and Phase 3 of the plan: conv-to-column graph construction and operational K/L/B column paths.

The unpooled architecture failed at chance. The pooled architecture is the current candidate because it restores the classifier interface used by the working PC ResNet baseline while keeping the column layer between the convolutional stack and classifier.

The pooled ResNet result is not known yet because Codex currently cannot see the GPU from its process environment. The user shell previously confirmed:

```bash
python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

with output:

```text
[CudaDevice(id=0)]
gpu
```

In this Codex process, the same check returned:

```text
[CpuDevice(id=0)]
cpu
```

## Next Commands

Run this first in the user shell:

```bash
python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

If it prints `[CudaDevice(id=0)]` and `gpu`, run the pooled all-active ResNet experiment:

```bash
/home/ni/repos/fpc/virt-envs/fpcpy3.12/bin/python scripts/train_cifar10_colba_accuracy.py --model resnet18 --activation leaky_relu --column_activation leaky_relu --num_columns 40 --column_mode all_active --num_epochs 2 --batch_size 128 --infer_steps 80 --eta_infer 0.1 --lr 0.01 --weight_decay 0.01 --eval_every 1
```

Acceptance check: the pooled all-active ColBa validation accuracy must approach or beat the previous PC ResNet reference point of 42.26% validation accuracy and 42.45% test accuracy before moving to sparse support search.

If the pooled all-active run is still near chance, the next implementation step should reduce the self-energy isolation of the column block. The most direct local change is to add a column residual bypass at the combiner level:

```text
tokens -> typed columns -> masked combiner
tokens -----------------> residual add -> pooled classifier
```

That preserves the already useful convolutional token representation while letting typed columns add refinements. The mechanism is that the classifier keeps a direct predictive-coding path to the convolutional features, while the column paths learn corrections instead of needing to carry the whole representation immediately.

## Known Gaps

Sparse support is only represented in the combiner contribution mask. Inactive columns still exist in the graph and still carry their own Gaussian energies. This is acceptable for the all-active diagnostic, but it is not yet a true inactive-column mechanism.

Hard-kernel and shell paths are not yet implemented.

Exact support audits, one-swap support teacher updates, context-conditioned support, internal certificates, and hierarchical bias heads are not yet implemented.
