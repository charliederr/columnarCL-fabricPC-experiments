#!/usr/bin/env python3
"""
Quick training test for CIFAR-10 columnar model.

This script runs a minimal training loop to verify the pipeline works.
Use this for debugging and quick sanity checks.

Usage:
    python scripts/train_cifar10_quick_test.py
    python scripts/train_cifar10_quick_test.py --num_columns 5 --epochs 2
"""

import argparse
import os

# JAX environment configuration
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import optax

from columnar_cl_fabricpc.experiments import create_cifar10_columnar_model
from fabricpc.utils.data.dataloader import Cifar10Loader
from fabricpc.training import train_pcn, evaluate_pcn


def parse_args():
    parser = argparse.ArgumentParser(description="Quick CIFAR-10 training test")
    parser.add_argument("--num_columns", type=int, default=3, help="Number of columns")
    parser.add_argument("--embed_dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--microcolumn_dim", type=int, default=8, help="Microcolumn dim")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CIFAR-10 Columnar Model - Quick Training Test")
    print("=" * 60)
    print(f"Columns: {args.num_columns}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Microcolumn dim: {args.microcolumn_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print()

    print("Creating model...")
    rng = jax.random.PRNGKey(args.seed)
    params, structure = create_cifar10_columnar_model(
        rng,
        num_columns=args.num_columns,
        embed_dim=args.embed_dim,
        microcolumn_dim=args.microcolumn_dim,
    )
    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")

    print("\nLoading data (using FabricPC's Cifar10Loader)...")
    train_loader = Cifar10Loader(
        split="train[:90%]",
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        tensor_format="NHWC",
    )
    val_loader = Cifar10Loader(
        split="train[90%:]",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        tensor_format="NHWC",
    )
    test_loader = Cifar10Loader(
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        tensor_format="NHWC",
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\nStarting training...")
    optimizer = optax.adamw(args.lr)
    rng_key, train_key, val_key, test_key = jax.random.split(rng, 4)

    config = {"num_epochs": args.epochs}

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        val_metrics = evaluate_pcn(
            params=params,
            structure=structure,
            test_loader=val_loader,
            config={},
            rng_key=val_key,
        )
        print(
            f"Epoch {epoch_idx + 1}/{args.epochs}: "
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
    print(f"Test accuracy: {test_metrics.get('accuracy', 0):.4f}")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
