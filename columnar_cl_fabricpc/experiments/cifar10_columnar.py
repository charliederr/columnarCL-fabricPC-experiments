"""
CIFAR-10 Columnar Predictive Coding Experiment.

Implements the ColBa columnar architecture from HiBaCaML for CIFAR-10
classification using FabricPC's predictive coding framework.

Architecture::

    CIFAR-10 Image (32×32×3)
            │
            ▼
    ┌───────────────────┐
    │  Patch Embedding  │   32×32×3 → 16 patches × 96 dims
    │   (visual stem)   │   + learnable position embeddings
    └─────────┬─────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      Column Pool                             │
    │                                                              │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐ ┌─────┐        │
    │  │Col 0│ │Col 1│ │Col 2│ │Col 3│ ... │Col38│ │Col39│        │
    │  │ K/L/B│ │ K/L/B│ │ K/L/B│ │ K/L/B│     │ K/L/B│ │ K/L/B│        │
    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘ └──┬──┘        │
    │     │       │       │       │           │       │            │
    │     └───────┴───────┴───┬───┴───────────┴───────┘            │
    │                         │                                    │
    │  40 columns, each with K/L/B microcolumns (dm=32)           │
    │  For initial CIFAR-10: all columns active                    │
    └─────────────────────────┼────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Column Combiner  │   Attention over 40 columns
                    │   (attention)     │   → (batch, 16, 96)
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Classification   │   Mean pool → Linear → Softmax
                    │      Head         │   → (batch, 10)
                    └─────────┬─────────┘
                              │
                              ▼
                      Class Predictions

    Training: FabricPC predictive coding (train_pcn)
    Target: >85% accuracy (baseline), >90% (proceed to continual learning)

Usage:
    python -m columnar_cl_fabricpc.experiments.cifar10_columnar
    python -m columnar_cl_fabricpc.experiments.cifar10_columnar --num_columns 10
    python -m columnar_cl_fabricpc.experiments.cifar10_columnar --epochs 50
"""

import argparse
import os
from typing import Tuple

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
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn

from columnar_cl_fabricpc.columns import (
    create_cifar10_patch_embed,
    create_column_pool,
    create_combiner,
    create_classification_head,
)
from fabricpc.utils.data.dataloader import Cifar10Loader


def create_cifar10_columnar_model(
    rng_key: jax.Array,
    num_columns: int = 40,
    embed_dim: int = 96,
    microcolumn_dim: int = 32,
    combination: str = "attention",
) -> Tuple:
    """
    Create the full CIFAR-10 columnar model.

    Args:
        rng_key: JAX random key for initialization
        num_columns: Number of columns in the pool (default: 40)
        embed_dim: Embedding dimension (default: 96)
        microcolumn_dim: Width of each microcolumn (default: 32)
        combination: Column combination strategy (default: "attention")

    Returns:
        Tuple of (params, structure) for training
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
        combination="sum",  # Internal microcolumn combination
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
        # Image → Patch embedding
        Edge(source=image_input, target=patch_embed.slot("in")),
    ]

    # Patch embedding → All columns
    for col in columns:
        edges.append(Edge(source=patch_embed, target=col.slot("in")))

    # All columns → Combiner
    for col in columns:
        edges.append(Edge(source=col, target=combiner.slot("in")))

    # Combiner → Classifier
    edges.append(Edge(source=combiner, target=classifier.slot("in")))

    # Assemble graph
    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=image_input, y=classifier),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=10),
    )

    # Initialize parameters
    params = initialize_params(structure, rng_key)

    return params, structure


def train_cifar10_columnar(
    num_columns: int = 40,
    embed_dim: int = 96,
    microcolumn_dim: int = 32,
    batch_size: int = 128,
    num_epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train the CIFAR-10 columnar model.

    Args:
        num_columns: Number of columns (default: 40)
        embed_dim: Embedding dimension (default: 96)
        microcolumn_dim: Microcolumn width (default: 32)
        batch_size: Training batch size (default: 128)
        num_epochs: Number of training epochs (default: 20)
        learning_rate: Learning rate (default: 1e-3)
        weight_decay: Weight decay (default: 1e-4)
        seed: Random seed (default: 42)
        verbose: Whether to print progress (default: True)

    Returns:
        Dictionary with training results
    """
    if verbose:
        print("=" * 70)
        print("CIFAR-10 Columnar Predictive Coding")
        print("=" * 70)
        print(f"Columns: {num_columns}")
        print(f"Embed dim: {embed_dim}")
        print(f"Microcolumn dim: {microcolumn_dim}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print()

    # Initialize random key
    rng_key = jax.random.PRNGKey(seed)
    rng_key, model_key, data_key = jax.random.split(rng_key, 3)

    # Create model
    if verbose:
        print("Creating model...")
    params, structure = create_cifar10_columnar_model(
        rng_key=model_key,
        num_columns=num_columns,
        embed_dim=embed_dim,
        microcolumn_dim=microcolumn_dim,
    )

    if verbose:
        n_nodes = len(structure.nodes)
        n_edges = len(structure.edges)
        print(f"Model created: {n_nodes} nodes, {n_edges} edges")

    # Load data using FabricPC's Cifar10Loader
    if verbose:
        print("Loading CIFAR-10 data...")
    train_loader = Cifar10Loader(
        split="train[:90%]",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        tensor_format="NHWC",
    )
    val_loader = Cifar10Loader(
        split="train[90%:]",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        tensor_format="NHWC",
    )
    test_loader = Cifar10Loader(
        split="test",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        tensor_format="NHWC",
    )
    if verbose:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print()

    # Optimizer
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)

    # Training configuration
    train_config = {"num_epochs": num_epochs}

    # Split keys for training and evaluation
    rng_key, train_key, val_key, test_key = jax.random.split(rng_key, 4)

    # Epoch callback for validation
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        nonlocal best_val_acc

        # Evaluate on validation set
        val_metrics = evaluate_pcn(
            params=params,
            structure=structure,
            test_loader=val_loader,
            config={},
            rng_key=val_key,
        )

        val_acc = val_metrics.get("accuracy", 0.0)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{num_epochs}: val_acc={val_acc:.4f}")

    # Train
    if verbose:
        print("Training...")
    final_params, epoch_losses, epoch_energies = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        epoch_callback=epoch_callback,
    )
    history["train_loss"] = epoch_losses

    # Final evaluation on test set
    if verbose:
        print("\nEvaluating on test set...")
    test_metrics = evaluate_pcn(
        params=final_params,
        structure=structure,
        test_loader=test_loader,
        config={},
        rng_key=test_key,
    )
    test_acc = test_metrics.get("accuracy", 0.0)

    if verbose:
        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

    return {
        "params": final_params,
        "structure": structure,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 columnar predictive coding model"
    )
    parser.add_argument(
        "--num_columns",
        type=int,
        default=40,
        help="Number of columns (default: 40)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=96,
        help="Embedding dimension (default: 96)",
    )
    parser.add_argument(
        "--microcolumn_dim",
        type=int,
        default=32,
        help="Microcolumn dimension (default: 32)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    results = train_cifar10_columnar(
        num_columns=args.num_columns,
        embed_dim=args.embed_dim,
        microcolumn_dim=args.microcolumn_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Best Val Accuracy: {results['best_val_accuracy']:.4f}")

    # Success criteria
    if results["test_accuracy"] >= 0.90:
        print("\n✓ Achieved >90% accuracy - ready for continual learning!")
    elif results["test_accuracy"] >= 0.85:
        print("\n✓ Achieved >85% baseline accuracy")
    else:
        print("\n✗ Below 85% accuracy - needs tuning")


if __name__ == "__main__":
    main()
