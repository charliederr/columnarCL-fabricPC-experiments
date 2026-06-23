#!/usr/bin/env python3
"""
Plain CIFAR-10 accuracy experiment with a ColBa-style column layer.

This keeps the validated predictive-coding convolutional backbone from
``train_cifar10_pc_resnet.py`` and inserts a typed K/L/B column pool before the
single 10-way classifier. The task is still plain CIFAR-10, not Split-CIFAR.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    LeakyReLUActivation,
    ReLUActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import AvgPool, ConvNode, IdentityNode, Linear, SkipConnection
from fabricpc.training import evaluate_pcn, train_pcn
from fabricpc.utils.data.dataloader import Cifar10Loader

from columnar_cl_fabricpc.columns.accuracy_nodes import (
    FeatureTokenizerNode,
    MaskedColumnCombinerNode,
    TypedColBaColumnNode,
)

jax.config.update("jax_default_prng_impl", "threefry2x32")


MODEL_CONFIGS = {
    "tiny": {
        "stem_channels": 16,
        "stages": [(16, 1, 1), (32, 2, 1), (64, 2, 1)],
    },
    "resnet18": {
        "stem_channels": 32,
        "stages": [(32, 1, 2), (64, 2, 2), (128, 2, 2), (256, 2, 2)],
    },
}


def get_activation(name):
    factories = {
        "relu": ReLUActivation,
        "tanh": TanhActivation,
        "gelu": GeluActivation,
        "leaky_relu": lambda: LeakyReLUActivation(alpha=0.1),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation: {name}")
    return factories[name]()


def strided_same_dim(size: int, stride: int) -> int:
    return int(math.ceil(size / stride))


def make_residual_block(prev_node, channels, stride, block_name, weight_init, activation):
    in_h, in_w, in_channels = prev_node.shape
    out_h = strided_same_dim(in_h, stride)
    out_w = strided_same_dim(in_w, stride)

    conv_a = ConvNode(
        shape=(out_h, out_w, channels),
        name=f"{block_name}_conv_a",
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
    )
    conv_b = ConvNode(
        shape=(out_h, out_w, channels),
        name=f"{block_name}_conv_b",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
    )
    skip = SkipConnection(shape=(out_h, out_w, channels), name=f"{block_name}_skip_sum")

    nodes = [conv_a, conv_b, skip]
    edges = [
        Edge(source=prev_node, target=conv_a.slot("in")),
        Edge(source=conv_a, target=conv_b.slot("in")),
        Edge(source=conv_b, target=skip.slot("in")),
    ]

    if stride != 1 or in_channels != channels:
        projection = ConvNode(
            shape=(out_h, out_w, channels),
            name=f"{block_name}_skip_proj",
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
        )
        nodes.append(projection)
        edges.append(Edge(source=prev_node, target=projection.slot("in")))
        edges.append(Edge(source=projection, target=skip.slot("in")))
    else:
        edges.append(Edge(source=prev_node, target=skip.slot("in")))

    return nodes, edges, skip


def build_support_mask(
    mode: str,
    num_columns: int,
    num_shared: int,
    active_nonshared: int,
    seed: int,
) -> Tuple[float, ...]:
    if num_shared > num_columns:
        raise ValueError("num_shared cannot exceed num_columns")

    mask = [0.0] * num_columns
    if mode == "all_active":
        return tuple([1.0] * num_columns)

    for idx in range(num_shared):
        mask[idx] = 1.0

    selectable = list(range(num_shared, num_columns))
    if active_nonshared > len(selectable):
        raise ValueError("active_nonshared exceeds available non-shared columns")

    if mode == "first_sparse":
        selected = selectable[:active_nonshared]
    elif mode == "random_sparse":
        key = jax.random.PRNGKey(seed)
        perm = list(map(int, jax.random.permutation(key, jnp.asarray(selectable))))
        selected = perm[:active_nonshared]
    else:
        raise ValueError(f"Unknown column mode: {mode}")

    for idx in selected:
        mask[idx] = 1.0
    return tuple(mask)


def build_colba_accuracy_graph(args):
    model_config = MODEL_CONFIGS[args.model]
    weight_init = MuPCInitializer()
    conv_activation = get_activation(args.activation)

    image = IdentityNode(shape=(32, 32, 3), name="input")
    stem_channels = model_config["stem_channels"]
    stem = ConvNode(
        shape=(32, 32, stem_channels),
        name="stem",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=conv_activation,
        weight_init=weight_init,
    )

    nodes = [image, stem]
    edges = [Edge(source=image, target=stem.slot("in"))]
    prev = stem

    for stage_idx, (channels, first_stride, num_blocks) in enumerate(
        model_config["stages"], start=1
    ):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            block_nodes, block_edges, prev = make_residual_block(
                prev_node=prev,
                channels=channels,
                stride=stride,
                block_name=f"s{stage_idx}b{block_idx + 1}",
                weight_init=weight_init,
                activation=conv_activation,
            )
            nodes.extend(block_nodes)
            edges.extend(block_edges)

    feature_h, feature_w, _ = prev.shape
    if args.num_tokens != feature_h * feature_w:
        raise ValueError(
            f"Current graph expects --num_tokens {feature_h * feature_w} "
            f"for final feature map {prev.shape}; got {args.num_tokens}"
        )
    grid_size = (feature_h, feature_w)

    tokenizer = FeatureTokenizerNode(
        shape=(args.num_tokens, args.embed_dim),
        name="tokens",
        embed_dim=args.embed_dim,
        num_tokens=args.num_tokens,
    )

    columns = [
        TypedColBaColumnNode(
            shape=(args.num_tokens, args.embed_dim),
            name=f"col_{idx:02d}",
            input_dim=args.embed_dim,
            microcolumn_dim=args.microcolumn_dim,
            grid_size=grid_size,
            hidden_activation=args.column_activation,
            leaky_alpha=0.1,
            residual=True,
        )
        for idx in range(args.num_columns)
    ]

    support_mask = build_support_mask(
        args.column_mode,
        args.num_columns,
        args.num_shared,
        args.active_nonshared,
        args.seed,
    )
    combiner = MaskedColumnCombinerNode(
        shape=(args.num_tokens, args.embed_dim),
        name="masked_combiner",
        num_columns=args.num_columns,
        support_mask=support_mask,
        combination=args.combiner,
    )
    column_pool = AvgPool(
        shape=(args.embed_dim,),
        name="column_pool",
        global_pool=True,
    )
    output = Linear(
        shape=(10,),
        name="output",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=XavierInitializer(),
    )

    nodes.extend([tokenizer] + columns + [combiner, column_pool, output])
    edges.append(Edge(source=prev, target=tokenizer.slot("in")))
    for col in columns:
        edges.append(Edge(source=tokenizer, target=col.slot("in")))
        edges.append(Edge(source=col, target=combiner.slot("in")))
    edges.append(Edge(source=combiner, target=column_pool.slot("in")))
    edges.append(Edge(source=column_pool, target=output.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=image, y=output),
        inference=InferenceSGDNormClip(
            eta_infer=args.eta_infer,
            infer_steps=args.infer_steps,
            max_norm=args.infer_max_norm,
        ),
        scaling=MuPCConfig(include_output=False),
    )
    return structure, support_mask


def train_cifar10_colba_accuracy(args):
    print("=" * 70)
    print("Plain CIFAR-10 PC ColBa Accuracy Experiment")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Conv activation: {args.activation}")
    print(f"Column activation: {args.column_activation}")
    print(f"Columns: {args.num_columns}")
    print(f"Column mode: {args.column_mode}")
    print(f"Shared columns: {args.num_shared}")
    print(f"Active non-shared columns: {args.active_nonshared}")
    print(f"Combiner: {args.combiner}")
    print(f"Tokens: {args.num_tokens}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Microcolumn dim: {args.microcolumn_dim}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Inference steps: {args.infer_steps}")
    print(f"Inference eta: {args.eta_infer}")
    print()

    master_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)

    structure, support_mask = build_colba_accuracy_graph(args)
    params = initialize_params(structure, graph_key)

    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    print(f"Active columns: {active_columns}")
    print(f"Graph: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {total_params:,}")

    print("\nLoading CIFAR-10 with FabricPC Cifar10Loader...")
    train_loader = Cifar10Loader(
        "train[:90%]",
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        tensor_format="NHWC",
    )
    val_loader = Cifar10Loader(
        "train[90%:]",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        tensor_format="NHWC",
    )
    test_loader = Cifar10Loader(
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        tensor_format="NHWC",
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    steps_per_epoch = len(train_loader)
    total_steps = max(1, round(args.num_epochs * steps_per_epoch))
    warmup_steps = min(total_steps - 1, int(0.05 * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    train_config = {"num_epochs": args.num_epochs}

    best_val_acc = -1.0
    best_val_epoch = None
    best_params = None
    final_epoch = math.ceil(args.num_epochs)

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        nonlocal best_val_acc, best_val_epoch, best_params
        epoch_num = epoch_idx + 1
        if args.eval_every <= 0:
            return None
        if epoch_num % args.eval_every != 0 and epoch_num != final_epoch:
            return None

        metrics = evaluate_pcn(params, structure, val_loader, config, eval_key)
        val_acc = float(metrics.get("accuracy", 0.0))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch_num
            best_params = params
        print(f"  Epoch {epoch_num}: val_acc={val_acc:.4f}")
        return metrics

    print(f"\nTraining for {args.num_epochs} epochs...")
    start_time = time.time()
    final_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=False,
        epoch_callback=epoch_callback,
    )
    elapsed = time.time() - start_time

    eval_params = best_params if best_params is not None else final_params
    if best_params is not None:
        print(
            f"\nTraining time: {elapsed:.1f}s"
            f"\nEvaluating best validation params from epoch {best_val_epoch} on test set..."
        )
    else:
        print(f"\nTraining time: {elapsed:.1f}s")
        print("Evaluating final params on test set...")
    test_metrics = evaluate_pcn(
        eval_params, structure, test_loader, train_config, eval_key
    )
    test_acc = float(test_metrics.get("accuracy", 0.0))

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    if best_params is not None:
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Best Val Epoch: {best_val_epoch}")
    return test_acc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="resnet18")
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        default="leaky_relu",
    )
    parser.add_argument(
        "--column_activation",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        default="leaky_relu",
    )
    parser.add_argument("--num_columns", type=int, default=40)
    parser.add_argument("--num_shared", type=int, default=4)
    parser.add_argument("--active_nonshared", type=int, default=5)
    parser.add_argument(
        "--column_mode",
        choices=["all_active", "first_sparse", "random_sparse"],
        default="all_active",
    )
    parser.add_argument("--combiner", choices=["attention", "sum"], default="attention")
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--microcolumn_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--infer_steps", type=int, default=80)
    parser.add_argument("--eta_infer", type=float, default=0.1)
    parser.add_argument("--infer_max_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.model = "tiny"
        args.num_columns = 6
        args.num_shared = 2
        args.active_nonshared = 2
        args.column_mode = "all_active"
        args.num_tokens = 64
        args.embed_dim = 64
        args.microcolumn_dim = 16
        args.num_epochs = 0.01
        args.infer_steps = 2
        args.eval_every = 0
    train_cifar10_colba_accuracy(args)


if __name__ == "__main__":
    main()
