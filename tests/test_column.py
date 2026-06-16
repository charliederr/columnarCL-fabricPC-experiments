"""
Tests for the ColumnarNode with K/L/B microcolumns.

Verifies that ColumnarNode correctly:
1. Creates K/L/B microcolumns with proper structure
2. Processes input through all microcolumns
3. Combines outputs using different strategies
4. Integrates with FabricPC graph machinery
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from columnar_cl_fabricpc.columns import (
    ColumnarNode,
    create_column,
    create_column_pool,
)


class TestColumnarNodeBasic:
    """Test basic ColumnarNode creation and configuration."""

    def test_create_node(self):
        """Test basic node creation."""
        node = ColumnarNode(
            shape=(16, 96),
            name="col_00",
            input_dim=96,
            microcolumn_dim=32,
        )

        assert node.name == "col_00"
        assert node.shape == (16, 96)
        assert node.input_dim == 96
        assert node.microcolumn_dim == 32
        assert node.num_microcolumns == 3

    def test_create_column_helper(self):
        """Test the create_column helper function."""
        col = create_column(
            name="test_col",
            num_tokens=16,
            input_dim=96,
            output_dim=64,
            microcolumn_dim=32,
        )

        assert col.name == "test_col"
        assert col.shape == (16, 64)
        assert col.input_dim == 96
        assert col.microcolumn_dim == 32

    def test_create_column_pool(self):
        """Test creating a pool of columns."""
        columns = create_column_pool(num_columns=10, prefix="col")

        assert len(columns) == 10
        assert columns[0].name == "col_00"
        assert columns[9].name == "col_09"

    def test_invalid_shape(self):
        """Shape must be 2D (num_tokens, output_dim)."""
        with pytest.raises(ValueError, match="must be.*num_tokens.*output_dim"):
            ColumnarNode(shape=(16,), name="bad", input_dim=96)

    def test_slots(self):
        """Node should have single multi-input slot."""
        slots = ColumnarNode.get_slots()

        assert "in" in slots
        assert slots["in"].is_multi_input is True


class TestColumnarNodeParams:
    """Test parameter initialization for microcolumns."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_params_structure(self, rng_key):
        """Verify K/L/B microcolumn parameters are created."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (16, 96)}
        config = {
            "input_dim": 96,
            "microcolumn_dim": 32,
            "num_microcolumns": 3,
            "combination": "sum",
        }

        params = ColumnarNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # Check K microcolumn
        assert "K_W_in" in params.weights
        assert "K_W_out" in params.weights
        assert "K_b_in" in params.biases

        # Check L microcolumn
        assert "L_W_in" in params.weights
        assert "L_W_out" in params.weights
        assert "L_b_in" in params.biases

        # Check B microcolumn
        assert "B_W_in" in params.weights
        assert "B_W_out" in params.weights
        assert "B_b_in" in params.biases

    def test_params_shapes(self, rng_key):
        """Verify parameter shapes are correct."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (16, 96)}
        config = {
            "input_dim": 96,
            "microcolumn_dim": 32,
            "num_microcolumns": 3,
            "combination": "sum",
        }

        params = ColumnarNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # Input projection: (input_dim, microcolumn_dim) = (96, 32)
        assert params.weights["K_W_in"].shape == (96, 32)
        assert params.weights["L_W_in"].shape == (96, 32)
        assert params.weights["B_W_in"].shape == (96, 32)

        # Output projection: (microcolumn_dim, output_dim) = (32, 96)
        assert params.weights["K_W_out"].shape == (32, 96)
        assert params.weights["L_W_out"].shape == (32, 96)
        assert params.weights["B_W_out"].shape == (32, 96)

        # Bias: (microcolumn_dim,) = (32,)
        assert params.biases["K_b_in"].shape == (32,)

    def test_concat_combination_params(self, rng_key):
        """Concat combination should add W_combine parameter."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (16, 96)}
        config = {
            "input_dim": 96,
            "microcolumn_dim": 32,
            "num_microcolumns": 3,
            "combination": "concat",
        }

        params = ColumnarNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # W_combine: (3 * 96, 96) = (288, 96)
        assert "W_combine" in params.weights
        assert params.weights["W_combine"].shape == (288, 96)

    def test_attention_combination_params(self, rng_key):
        """Attention combination should add mc_attention parameter."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (16, 96)}
        config = {
            "input_dim": 96,
            "microcolumn_dim": 32,
            "num_microcolumns": 3,
            "combination": "attention",
        }

        params = ColumnarNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert "mc_attention" in params.weights
        assert params.weights["mc_attention"].shape == (3,)


class TestColumnarNodeInGraph:
    """Test ColumnarNode integration with FabricPC graph."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_single_column_in_graph(self, rng_key):
        """Test single column can be placed in FabricPC graph."""
        from fabricpc.nodes import Linear, IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        input_node = IdentityNode(shape=(16, 96), name="input")
        column = create_column(name="col_00", num_tokens=16, input_dim=96, output_dim=96)
        output = Linear(shape=(16, 10), name="output")

        structure = graph(
            nodes=[input_node, column, output],
            edges=[
                Edge(source=input_node, target=column.slot("in")),
                Edge(source=column, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
        )

        assert len(structure.nodes) == 3
        assert "col_00" in structure.nodes

        params = initialize_params(structure, rng_key)
        assert "col_00" in params.nodes

    def test_multiple_columns_parallel(self, rng_key):
        """Test multiple columns processing same input in parallel."""
        from fabricpc.nodes import Linear, IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        input_node = IdentityNode(shape=(16, 96), name="input")
        col_0 = create_column(name="col_00", num_tokens=16, input_dim=96, output_dim=96)
        col_1 = create_column(name="col_01", num_tokens=16, input_dim=96, output_dim=96)
        col_2 = create_column(name="col_02", num_tokens=16, input_dim=96, output_dim=96)
        output = Linear(shape=(16, 10), name="output")

        structure = graph(
            nodes=[input_node, col_0, col_1, col_2, output],
            edges=[
                Edge(source=input_node, target=col_0.slot("in")),
                Edge(source=input_node, target=col_1.slot("in")),
                Edge(source=input_node, target=col_2.slot("in")),
                Edge(source=col_0, target=output.slot("in")),
                Edge(source=col_1, target=output.slot("in")),
                Edge(source=col_2, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
        )

        assert len(structure.nodes) == 5
        params = initialize_params(structure, rng_key)
        assert all(f"col_{i:02d}" in params.nodes for i in range(3))

    def test_forward_pass(self, rng_key):
        """Test forward pass produces correct output shape."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        input_node = IdentityNode(shape=(16, 96), name="input")
        column = create_column(name="col_00", num_tokens=16, input_dim=96, output_dim=64)

        structure = graph(
            nodes=[input_node, column],
            edges=[
                Edge(source=input_node, target=column.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        tokens = jax.random.normal(key1, (batch_size, 16, 96))
        clamps = {"input": tokens}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        assert "col_00" in state.nodes
        col_state = state.nodes["col_00"]
        assert col_state.z_latent.shape == (batch_size, 16, 64)


class TestCombinationModes:
    """Test different microcolumn combination strategies."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_sum_combination(self, rng_key):
        """Test sum combination mode."""
        col = create_column(
            name="col_sum",
            num_tokens=16,
            input_dim=96,
            output_dim=64,
            combination="sum",
        )
        assert col.combination == "sum"

    def test_concat_combination(self, rng_key):
        """Test concat combination mode."""
        col = ColumnarNode(
            shape=(16, 64),
            name="col_concat",
            input_dim=96,
            combination="concat",
        )
        assert col.combination == "concat"

    def test_attention_combination(self, rng_key):
        """Test attention combination mode."""
        col = ColumnarNode(
            shape=(16, 64),
            name="col_attn",
            input_dim=96,
            combination="attention",
        )
        assert col.combination == "attention"


class TestColumnPoolConfiguration:
    """Test column pool creation with various configurations."""

    def test_cifar10_pool_size(self):
        """Test creating CIFAR-10 recommended pool size (40 columns)."""
        columns = create_column_pool(num_columns=40)
        assert len(columns) == 40

    def test_custom_dimensions(self):
        """Test pool with custom dimensions."""
        columns = create_column_pool(
            num_columns=5,
            num_tokens=8,
            input_dim=64,
            output_dim=128,
            microcolumn_dim=16,
        )

        assert len(columns) == 5
        for col in columns:
            assert col.shape == (8, 128)
            assert col.input_dim == 64
            assert col.microcolumn_dim == 16

    def test_pool_naming(self):
        """Test custom prefix naming."""
        columns = create_column_pool(num_columns=3, prefix="shared")

        assert columns[0].name == "shared_00"
        assert columns[1].name == "shared_01"
        assert columns[2].name == "shared_02"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
