#!/usr/bin/env python3
"""
Medium-scale CIFAR-10 columnar model training (5 epochs).

Configuration:
- 10 columns (reduced from full 40 for faster iteration)
- 64-dim embeddings
- 16-dim microcolumns
- 5 epochs
- Batch size 128

Usage:
    python scripts/train_cifar10_medium_5epochs.py
"""

import os

# JAX environment configuration
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import optax

from columnar_cl_fabricpc.experiments import create_cifar10_columnar_model
from fabricpc.utils.data.dataloader import Cifar10Loader
from fabricpc.training import train_pcn, evaluate_pcn

# Configuration
NUM_COLUMNS = 10
EMBED_DIM = 64
MICROCOLUMN_DIM = 16
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42


def main():
    print("=" * 60)
    print("CIFAR-10 Columnar Model - Medium Scale (5 epochs)")
    print("=" * 60)
    print(f"Columns: {NUM_COLUMNS}")
    print(f"Embed dim: {EMBED_DIM}")
    print(f"Microcolumn dim: {MICROCOLUMN_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    print("Creating model...")
    rng = jax.random.PRNGKey(SEED)
    params, structure = create_cifar10_columnar_model(
        rng,
        num_columns=NUM_COLUMNS,
        embed_dim=EMBED_DIM,
        microcolumn_dim=MICROCOLUMN_DIM,
    )
    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")

    print("\nLoading data...")
    train_loader = Cifar10Loader(
        split="train[:90%]",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        tensor_format="NHWC",
    )
    val_loader = Cifar10Loader(
        split="train[90%:]",
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        tensor_format="NHWC",
    )
    test_loader = Cifar10Loader(
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        tensor_format="NHWC",
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\nStarting training...")
    optimizer = optax.adamw(LEARNING_RATE)
    rng_key, train_key, val_key, test_key = jax.random.split(rng, 4)

    config = {"num_epochs": NUM_EPOCHS}

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        val_metrics = evaluate_pcn(
            params=params,
            structure=structure,
            test_loader=val_loader,
            config={},
            rng_key=val_key,
        )
        print(
            f"Epoch {epoch_idx + 1}/{NUM_EPOCHS}: "
            f"val_acc={val_metrics.get('accuracy', 0):.4f}"
        )

    final_params, epoch_losses, epoch_energies = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=config,
        rng_key=train_key,
        epoch_callback=epoch_callback,
    )

    print("\nEvaluating on test set...")
    test_metrics = evaluate_pcn(
        params=final_params,
        structure=structure,
        test_loader=test_loader,
        config={},
        rng_key=test_key,
    )
    test_acc = test_metrics.get("accuracy", 0)
    print(f"Test accuracy: {test_acc:.4f}")

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f}")

    if test_acc >= 0.90:
        print("SUCCESS: Achieved >90% accuracy - ready for continual learning!")
    elif test_acc >= 0.85:
        print("SUCCESS: Achieved >85% baseline accuracy")
    else:
        print(f"Below 85% target - needs tuning (current: {test_acc:.1%})")


if __name__ == "__main__":
    main()
