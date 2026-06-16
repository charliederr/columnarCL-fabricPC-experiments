#!/usr/bin/env python3
"""
CIFAR-10 Columnar Model Training - Improved Settings.

Based on FabricPC's resnet18_cifar10_demo.py, this applies:
1. More inference steps (40 instead of 10)
2. InferenceSGDNormClip for stability (max_norm=1.0)
3. Data augmentation (random horizontal flip + random crop)
4. Cosine LR schedule with warmup
5. Higher learning rate (0.01) and weight decay (0.01)

Usage:
    python scripts/train_cifar10_improved.py
    python scripts/train_cifar10_improved.py --num_epochs 20
    python scripts/train_cifar10_improved.py --quick  # 2-epoch smoke test
"""

import os
import argparse
import time

# JAX environment configuration
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
    create_cifar10_patch_embed,
    create_column_pool,
    create_combiner,
    create_classification_head,
)

jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Data Augmentation (from resnet18_cifar10_demo.py)
# =============================================================================


class AugmentedCifar10Loader:
    """Wraps Cifar10Loader with random horizontal flip and random crop+pad."""

    def __init__(self, base_loader, seed=42, pad=4):
        self.base_loader = base_loader
        self.seed = seed
        self.pad = pad
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pad = self.pad
        for images, labels in self.base_loader:
            # Random horizontal flip
            flip_mask = rng.random(images.shape[0]) > 0.5
            images[flip_mask] = images[flip_mask, :, ::-1, :]

            # Random crop with padding
            padded = np.pad(
                images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
            )
            B, H, W, C = images.shape
            crop_y = rng.integers(0, 2 * pad + 1, size=B)
            crop_x = rng.integers(0, 2 * pad + 1, size=B)
            for i in range(B):
                images[i] = padded[
                    i, crop_y[i] : crop_y[i] + H, crop_x[i] : crop_x[i] + W, :
                ]

            yield images, labels

    def __len__(self):
        return len(self.base_loader)


# =============================================================================
# Model Creation with Improved Inference
# =============================================================================


def create_cifar10_columnar_model_improved(
    rng_key,
    num_columns=10,
    embed_dim=64,
    microcolumn_dim=32,
    combination="attention",
    infer_steps=40,
    eta_infer=0.1,
    max_norm=1.0,
):
    """
    Create CIFAR-10 columnar model with improved inference settings.

    Key changes from original:
    - Uses InferenceSGDNormClip instead of InferenceSGD
    - More inference steps (40 vs 10)
    - Gradient norm clipping for stability
    """
    num_tokens = 16  # 4×4 grid of patches

    # Input node for images
    image_input = IdentityNode(shape=(32, 32, 3), name="image")

    # Patch embedding (visual stem)
    patch_embed = create_cifar10_patch_embed(
        name="patches",
        embed_dim=embed_dim,
        patch_size=8,
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

    # Build node list
    nodes = [image_input, patch_embed] + columns + [combiner, classifier]

    # Build edges
    edges = [
        Edge(source=image_input, target=patch_embed.slot("in")),
    ]

    for col in columns:
        edges.append(Edge(source=patch_embed, target=col.slot("in")))

    for col in columns:
        edges.append(Edge(source=col, target=combiner.slot("in")))

    edges.append(Edge(source=combiner, target=classifier.slot("in")))

    # Assemble graph with IMPROVED inference settings
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

    # Initialize parameters
    params = initialize_params(structure, rng_key)

    return params, structure


# =============================================================================
# Training
# =============================================================================


def train_cifar10_improved(args):
    """Train with improved settings from ResNet-18 demo."""
    print("=" * 60)
    print("CIFAR-10 Columnar Model - Improved Training")
    print("=" * 60)
    print(f"Columns: {args.num_columns}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Microcolumn dim: {args.microcolumn_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Inference steps: {args.infer_steps}")
    print(f"Inference eta: {args.eta_infer}")
    print(f"Data augmentation: {not args.no_augment}")
    print()

    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    print("Creating model...")
    params, structure = create_cifar10_columnar_model_improved(
        rng_key=graph_key,
        num_columns=args.num_columns,
        embed_dim=args.embed_dim,
        microcolumn_dim=args.microcolumn_dim,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )

    print(f"Model: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data loaders
    print("\nLoading data...")
    base_train_loader = Cifar10Loader(
        "train[:90%]",
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        tensor_format="NHWC",
    )
    if args.no_augment:
        train_loader = base_train_loader
    else:
        train_loader = AugmentedCifar10Loader(base_train_loader, seed=args.seed)

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
    print(f"Train batches: {len(base_train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Cosine LR schedule with warmup (from ResNet-18 demo)
    steps_per_epoch = len(base_train_loader)
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

    # Epoch callback for validation
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

    # Train
    print(f"\nTraining for {args.num_epochs} epochs...")
    start_time = time.time()

    final_params, epoch_losses, epoch_energies = train_pcn(
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

    # Final evaluation
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
        print("\nSUCCESS: Achieved >90% accuracy - ready for continual learning!")
    elif test_acc >= 0.85:
        print("\nSUCCESS: Achieved >85% baseline accuracy")
    else:
        print(f"\nBelow 85% target - current: {test_acc:.1%}")

    return test_acc


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Columnar Model - Improved Training"
    )
    parser.add_argument(
        "--num_columns", type=int, default=10, help="Number of columns (default: 10)"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=64, help="Embedding dimension (default: 64)"
    )
    parser.add_argument(
        "--microcolumn_dim",
        type=int,
        default=32,
        help="Microcolumn dimension (default: 32)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Peak learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=40,
        help="Inference steps (default: 40)",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.1,
        help="Inference learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=2,
        help="Evaluate every N epochs (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 2 epochs, no augmentation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.num_epochs = 2
        args.no_augment = True
        args.eval_every = 1
    train_cifar10_improved(args)


if __name__ == "__main__":
    main()
