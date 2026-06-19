#!/usr/bin/env python3
"""
Plain CIFAR-10 predictive-coding convolutional baseline.

This experiment uses FabricPC's upstream convolution branch:
ConvNode builds the convolutional layers, SkipConnection builds residual
summation points, AvgPool performs global pooling, and MuPCConfig supplies
predictive-coding scaling for the graph. The task is a single 10-way CIFAR-10
classification problem, with no split-task machinery and no replay state.

Usage:
    python scripts/train_cifar10_pc_resnet.py --model tiny --num_epochs 0.01
    python scripts/train_cifar10_pc_resnet.py --model resnet18 --num_epochs 2
"""

import argparse
import math
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import optax

from fabricpc.nodes import ConvNode, Linear, IdentityNode, SkipConnection, AvgPool
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    GeluActivation,
    LeakyReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import Cifar10Loader

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


def strided_same_dim(size, stride):
    return int(math.ceil(size / stride))


def make_residual_block(prev_node, channels, stride, block_name, weight_init, activation):
    """Create one predictive-coding residual block."""
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


def build_pc_resnet(model_name, activation, infer_steps, eta_infer):
    """Build a PC graph for one 10-class CIFAR-10 task."""
    model_config = MODEL_CONFIGS[model_name]
    weight_init = MuPCInitializer()

    image = IdentityNode(shape=(32, 32, 3), name="input")
    stem_channels = model_config["stem_channels"]
    stem = ConvNode(
        shape=(32, 32, stem_channels),
        name="stem",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
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
                activation=activation,
            )
            nodes.extend(block_nodes)
            edges.extend(block_edges)

    avg_pool = AvgPool(shape=(prev.shape[-1],), name="avgpool", global_pool=True)
    output = Linear(
        shape=(10,),
        name="output",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=XavierInitializer(),
    )

    nodes.extend([avg_pool, output])
    edges.extend(
        [
            Edge(source=prev, target=avg_pool.slot("in")),
            Edge(source=avg_pool, target=output.slot("in")),
        ]
    )

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=image, y=output),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=1.0,
        ),
        scaling=MuPCConfig(include_output=False),
    )


def train_cifar10_pc_resnet(args):
    print("=" * 60)
    print("Plain CIFAR-10 PC Conv Baseline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Activation: {args.activation}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Inference steps: {args.infer_steps}")
    print(f"Inference eta: {args.eta_infer}")
    print()

    master_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)

    activation = get_activation(args.activation)
    structure = build_pc_resnet(
        model_name=args.model,
        activation=activation,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )
    params = initialize_params(structure, graph_key)

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

    best_val_acc = 0.0
    final_epoch = math.ceil(args.num_epochs)

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        nonlocal best_val_acc
        epoch_num = epoch_idx + 1
        if args.eval_every <= 0:
            return None
        if epoch_num % args.eval_every != 0 and epoch_num != final_epoch:
            return None

        metrics = evaluate_pcn(params, structure, val_loader, config, eval_key)
        val_acc = float(metrics.get("accuracy", 0.0))
        best_val_acc = max(best_val_acc, val_acc)
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

    print(f"\nTraining time: {elapsed:.1f}s")
    print("Evaluating on test set...")
    test_metrics = evaluate_pcn(
        final_params, structure, test_loader, train_config, eval_key
    )
    test_acc = float(test_metrics.get("accuracy", 0.0))

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    if best_val_acc > 0.0:
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
    return test_acc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="resnet18")
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        default="tanh",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--infer_steps", type=int, default=40)
    parser.add_argument("--eta_infer", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.model = "tiny"
        args.num_epochs = 0.01
        args.infer_steps = 2
        args.eval_every = 0
    train_cifar10_pc_resnet(args)


if __name__ == "__main__":
    main()
