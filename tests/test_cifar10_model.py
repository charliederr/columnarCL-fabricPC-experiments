"""
Tests for the full CIFAR-10 columnar model.

Verifies that:
1. Model assembles correctly with all components
2. Forward pass works end-to-end
3. Model is compatible with FabricPC training
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from columnar_cl_fabricpc.experiments import create_cifar10_columnar_model


class TestModelCreation:
    """Test model creation and structure."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_create_model_default(self, rng_key):
        """Test model creation with default parameters."""
        params, structure = create_cifar10_columnar_model(rng_key)

        # Check nodes: image + patches + 40 columns + combiner + classifier
        expected_nodes = 1 + 1 + 40 + 1 + 1  # 44 nodes
        assert len(structure.nodes) == expected_nodes

        # Check key nodes exist
        assert "image" in structure.nodes
        assert "patches" in structure.nodes
        assert "combiner" in structure.nodes
        assert "classifier" in structure.nodes

        # Check columns exist
        for i in range(40):
            assert f"col_{i:02d}" in structure.nodes

    def test_create_model_custom_columns(self, rng_key):
        """Test model creation with custom number of columns."""
        params, structure = create_cifar10_columnar_model(
            rng_key, num_columns=10
        )

        # 1 + 1 + 10 + 1 + 1 = 14 nodes
        assert len(structure.nodes) == 14

        # Check column count
        column_nodes = [n for n in structure.nodes if n.startswith("col_")]
        assert len(column_nodes) == 10

    def test_create_model_custom_dims(self, rng_key):
        """Test model creation with custom dimensions."""
        params, structure = create_cifar10_columnar_model(
            rng_key,
            num_columns=5,
            embed_dim=64,
            microcolumn_dim=16,
        )

        # Verify patch embedding output shape
        patches_node = structure.nodes["patches"]
        assert patches_node.shape == (16, 64)  # 16 tokens, 64 dims

    def test_params_initialized(self, rng_key):
        """Test that all node parameters are initialized."""
        params, structure = create_cifar10_columnar_model(
            rng_key, num_columns=5
        )

        # All nodes should have params
        for node_name in structure.nodes:
            assert node_name in params.nodes

    def test_edges_correct(self, rng_key):
        """Test that edge connectivity is correct."""
        params, structure = create_cifar10_columnar_model(
            rng_key, num_columns=5
        )

        # Count edge types
        edges = structure.edges

        # Should have:
        # 1 edge: image → patches
        # 5 edges: patches → each column
        # 5 edges: each column → combiner
        # 1 edge: combiner → classifier
        # Total: 12 edges
        assert len(edges) == 12


class TestModelForwardPass:
    """Test forward pass through the model."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_forward_pass_shapes(self, rng_key):
        """Test that forward pass produces correct output shapes."""
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        params, structure = create_cifar10_columnar_model(
            rng_key, num_columns=5
        )

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        # Create fake CIFAR-10 images
        images = jax.random.uniform(key1, (batch_size, 32, 32, 3), minval=-1, maxval=1)
        clamps = {"image": images}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Check intermediate shapes
        assert state.nodes["patches"].z_latent.shape == (batch_size, 16, 96)
        assert state.nodes["col_00"].z_latent.shape == (batch_size, 16, 96)
        assert state.nodes["combiner"].z_latent.shape == (batch_size, 16, 96)
        assert state.nodes["classifier"].z_latent.shape == (batch_size, 10)

    def test_classifier_output_sums_to_one(self, rng_key):
        """Test that classifier output (softmax) sums to 1."""
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        params, structure = create_cifar10_columnar_model(
            rng_key, num_columns=5
        )

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        images = jax.random.uniform(key1, (batch_size, 32, 32, 3), minval=-1, maxval=1)
        clamps = {"image": images}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # z_mu should be softmax output
        classifier_z_mu = state.nodes["classifier"].z_mu

        # Each row should sum to 1 (softmax)
        row_sums = jnp.sum(classifier_z_mu, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)


class TestModelScaling:
    """Test model with different configurations."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_small_model(self, rng_key):
        """Test small model configuration."""
        params, structure = create_cifar10_columnar_model(
            rng_key,
            num_columns=3,
            embed_dim=32,
            microcolumn_dim=8,
        )

        assert len(structure.nodes) == 7  # image + patches + 3 cols + combiner + cls

    def test_medium_model(self, rng_key):
        """Test medium model configuration."""
        params, structure = create_cifar10_columnar_model(
            rng_key,
            num_columns=10,
            embed_dim=64,
            microcolumn_dim=16,
        )

        assert len(structure.nodes) == 14

    def test_large_model(self, rng_key):
        """Test large model (full CIFAR-10 config)."""
        params, structure = create_cifar10_columnar_model(
            rng_key,
            num_columns=40,
            embed_dim=96,
            microcolumn_dim=32,
        )

        assert len(structure.nodes) == 44


class TestTaskMap:
    """Test task map configuration."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_input_output_nodes(self, rng_key):
        """Test that input/output nodes are correctly configured."""
        params, structure = create_cifar10_columnar_model(
            rng_key, num_columns=5
        )

        task_map = structure.task_map

        # x should be image, y should be classifier
        assert task_map["x"] == "image"
        assert task_map["y"] == "classifier"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
