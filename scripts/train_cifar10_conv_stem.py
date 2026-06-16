#!/usr/bin/env python3
"""
CIFAR-10 Columnar Model with Convolutional Stem.

From HiBaCaML Section 6:
> Shared visual stem. A shallow convolutional stem extracts low-level
> visual primitives before support selection begins:
> Conv(3, 32, 3×3) → Conv(32, 64, 3×3, stride 2) → Conv(64, 64, 3×3),
> followed by light patch pooling to 64–96 tokens of width 64–96.

This implements the recommended convolutional front-end so the columnar
machinery is not overwhelmed by raw-pixel work.

Usage:
    python scripts/train_cifar10_conv_stem.py
    python scripts/train_cifar10_conv_stem.py --num_epochs 20
    python scripts/train_cifar10_conv_stem.py --quick
"""

import os
import argparse
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import Cifar10Loader

from columnar_cl_fabricpc.columns import (
    create_cifar10_conv_stem,
    create_column_pool,
    create_combiner,
    create_classification_head,
)

jax.config.update("jax_default_prng_impl", "threefry2x32")


def create_cifar10_conv_stem_model(
    rng_key,
    num_columns=10,
    num_tokens=64,
    embed_dim=64,
    microcolumn_dim=32,
    combination="attention",
    infer_steps=40,
    eta_infer=0.1,
    max_norm=1.0,
):
    """
    Create CIFAR-10 columnar model with convolutional stem.

    Architecture:
        Image (32×32×3)
        → Conv stem (3 conv layers → 64 tokens × 64 dims)
        → Column pool (N columns, each with K/L/B microcolumns)
        → Combiner (attention over columns)
        → Classification head (10-way softmax)
    """
    # Input node
    image_input = IdentityNode(shape=(32, 32, 3), name="image")

    # Convolutional stem (replaces patch embedding)
    conv_stem = create_cifar10_conv_stem(
        name="conv_stem",
        num_tokens=num_tokens,
        embed_dim=embed_dim,
    )

    # Column pool
    columns = create_column_pool(
        num_columns=num_columns,
        num_tokens=num_tokens,
        input_dim=embed_dim,
        output_dim=embed_dim,
        microcolumn_dim=microcolumn_dim,
        combination="sum",
        prefix="col",
    )

    # Column combiner
    combiner = create_combiner(
        name="combiner",
        num_columns=num_columns,
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        combination=combination,
    )

    # Classification head
    classifier = create_classification_head(
        name="classifier",
        num_classes=10,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        pooling="mean",
    )

    # Build graph
    nodes = [image_input, conv_stem] + columns + [combiner, classifier]

    edges = [
        Edge(source=image_input, target=conv_stem.slot("in")),
    ]

    for col in columns:
        edges.append(Edge(source=conv_stem, target=col.slot("in")))

    for col in columns:
        edges.append(Edge(source=col, target=combiner.slot("in")))

    edges.append(Edge(source=combiner, target=classifier.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=image_input, y=classifier),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=max_norm,
        ),
    )

    params = initialize_params(structure, rng_key)
    return params, structure


def train_cifar10_conv_stem(args):
    """Train with convolutional stem."""
    print("=" * 60)
    print("CIFAR-10 Columnar Model - Convolutional Stem")
    print("=" * 60)
    print(f"Columns: {args.num_columns}")
    print(f"Tokens: {args.num_tokens}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Microcolumn dim: {args.microcolumn_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Inference steps: {args.infer_steps}")
    print()

    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    print("Creating model with conv stem...")
    params, structure = create_cifar10_conv_stem_model(
        rng_key=graph_key,
        num_columns=args.num_columns,
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        microcolumn_dim=args.microcolumn_dim,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )

    print(f"Model: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data
    print("\nLoading data...")
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

    # Optimizer with cosine schedule
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

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

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        nonlocal best_val_acc
        epoch_num = epoch_idx + 1

        if args.eval_every > 0 and (
            epoch_num % args.eval_every == 0 or epoch_num == args.num_epochs
        ):
            val_metrics = evaluate_pcn(
                params=params,
                structure=structure,
                test_loader=val_loader,
                config={},
                rng_key=eval_key,
            )
            val_acc = val_metrics.get("accuracy", 0.0)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            print(f"  Epoch {epoch_num}: val_acc={val_acc:.4f}")
            return val_metrics
        return None

    print(f"\nTraining for {args.num_epochs} epochs...")
    start_time = time.time()

    final_params, epoch_losses, _ = train_pcn(
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
    print(f"\nTraining time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)")

    print("\nEvaluating on test set...")
    test_metrics = evaluate_pcn(
        params=final_params,
        structure=structure,
        test_loader=test_loader,
        config={},
        rng_key=eval_key,
    )
    test_acc = test_metrics.get("accuracy", 0.0)

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")

    if test_acc >= 0.90:
        print("\nSUCCESS: >90% accuracy!")
    elif test_acc >= 0.85:
        print("\nSUCCESS: >85% baseline accuracy")
    else:
        print(f"\nBelow 85% target - current: {test_acc:.1%}")

    return test_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Columnar Model with Conv Stem"
    )
    parser.add_argument("--num_columns", type=int, default=10)
    parser.add_argument("--num_tokens", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--microcolumn_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--infer_steps", type=int, default=40)
    parser.add_argument("--eta_infer", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.num_epochs = 2
        args.eval_every = 1
    train_cifar10_conv_stem(args)


if __name__ == "__main__":
    main()
