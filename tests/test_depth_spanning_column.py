"""
Tests for the depth-spanning columnar node.

Verifies that DepthSpanningColumnNode correctly:
1. Processes inputs from four stage slots (stage2, stage3, stage4, stage4_pool)
2. Implements K, L, B pathways with proper skip connections
3. Combines pathway outputs with learned weights
4. Integrates with FabricPC graph machinery
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from columnar_cl_fabricpc.columns import (
    DepthSpanningColumnNode,
    create_depth_spanning_column,
    create_depth_spanning_column_pool,
    create_cifar10_stage_taps,
    StageTapTokenizer,
    GlobalPoolNode,
)


class TestDepthSpanningColumnNode:
    """Test DepthSpanningColumnNode instantiation and configuration."""

    def test_create_node(self):
        """Test basic node creation."""
        node = DepthSpanningColumnNode(
            shape=(64, 64),
            name="col0",
            input_dim=64,
            microcolumn_dim=32,
            grid_size=(8, 8),
        )
        assert node.name == "col0"
        assert node.shape == (64, 64)

    def test_create_column_helper(self):
        """Test the create_depth_spanning_column helper function."""
        node = create_depth_spanning_column(
            name="test_col",
            input_dim=64,
            output_dim=128,
            microcolumn_dim=32,
            grid_size=(8, 8),
        )
        assert node.shape == (64, 128)  # 64 tokens, 128 output_dim

    def test_invalid_shape_dimensions(self):
        """Shape must be 2D (tokens, output_dim)."""
        with pytest.raises(ValueError, match="must be.*tokens.*output_dim"):
            DepthSpanningColumnNode(
                shape=(64,),
                name="bad",
                input_dim=64,
            )

    def test_invalid_token_count(self):
        """Shape[0] must match grid_size product."""
        with pytest.raises(ValueError, match="grid_size product"):
            DepthSpanningColumnNode(
                shape=(100, 64),  # 100 != 8×8=64
                name="bad",
                input_dim=64,
                grid_size=(8, 8),
            )

    def test_slots(self):
        """Node should have four input slots for multi-stage input."""
        slots = DepthSpanningColumnNode.get_slots()

        assert "stage2" in slots
        assert "stage3" in slots
        assert "stage4" in slots
        assert "stage4_pool" in slots

        # All slots are single-input (one edge per slot)
        for slot_name, slot_spec in slots.items():
            assert slot_spec.is_multi_input is False, f"{slot_name} should be single-input"


class TestDepthSpanningColumnParams:
    """Test parameter initialization."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_initialize_params_shapes(self, rng_key):
        """Verify all parameter shapes are correct."""
        node_shape = (64, 64)  # 64 tokens, 64 output_dim
        input_shapes = {
            "stage2:stage2": (64, 64),
            "stage3:stage3": (64, 64),
            "stage4:stage4": (64, 64),
            "stage4_pool:stage4_pool": (1, 64),
        }
        config = {
            "input_dim": 64,
            "microcolumn_dim": 32,
            "grid_size": (8, 8),
        }

        params = DepthSpanningColumnNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # K pathway
        assert params.weights["K_W_deep"].shape == (64, 32)  # input → μd
        assert params.weights["K_W_mid"].shape == (32 + 64, 32)  # μd + input → μd
        assert params.weights["K_W_out"].shape == (32 + 64, 64)  # μd + input → output

        # L pathway
        assert params.weights["L_W_deep"].shape == (64, 32)  # input → μd
        assert params.weights["L_W_conv_dw"].shape == (3, 3, 1, 32)  # depthwise conv (kh, kw, 1, μd)
        assert params.weights["L_W_out"].shape == (32, 64)  # μd → output

        # B pathway
        assert params.weights["B_W_deep"].shape == (64, 32)  # input → μd
        assert params.weights["B_W_out"].shape == (32, 64)  # μd → output

        # Path scale
        assert params.weights["path_scale"].shape == (3,)

    def test_initialize_params_requires_input_dim(self, rng_key):
        """Should raise if input_dim not in config."""
        node_shape = (64, 64)
        input_shapes = {"stage4:stage4": (64, 64)}
        config = {"microcolumn_dim": 32}  # Missing input_dim

        with pytest.raises(ValueError, match="input_dim"):
            DepthSpanningColumnNode.initialize_params(
                rng_key, node_shape, input_shapes, config=config
            )

    def test_path_scale_initialization(self, rng_key):
        """Path scale should be initialized for equal variance contribution."""
        node_shape = (64, 64)
        input_shapes = {
            "stage2:stage2": (64, 64),
            "stage3:stage3": (64, 64),
            "stage4:stage4": (64, 64),
            "stage4_pool:stage4_pool": (1, 64),
        }
        config = {"input_dim": 64, "microcolumn_dim": 32}

        params = DepthSpanningColumnNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # Should be 1/sqrt(3) for three pathways
        expected_scale = 1.0 / jnp.sqrt(3.0)
        assert jnp.allclose(params.weights["path_scale"], expected_scale)


class TestDepthSpanningColumnPool:
    """Test creating pools of columns."""

    def test_create_pool(self):
        """Test creating a pool of columns."""
        columns = create_depth_spanning_column_pool(
            num_columns=4,
            input_dim=64,
            output_dim=64,
            microcolumn_dim=32,
        )

        assert len(columns) == 4
        assert "col0" in columns
        assert "col1" in columns
        assert "col2" in columns
        assert "col3" in columns

        for name, col in columns.items():
            assert col.shape == (64, 64)

    def test_create_pool_custom_prefix(self):
        """Test pool with custom name prefix."""
        columns = create_depth_spanning_column_pool(
            num_columns=2,
            input_dim=64,
            output_dim=64,
            name_prefix="depth_col",
        )

        assert "depth_col0" in columns
        assert "depth_col1" in columns


class TestDepthSpanningColumnInGraph:
    """Test DepthSpanningColumnNode integration with FabricPC graph."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_in_graph_with_stage_taps(self, rng_key):
        """Test column connected to stage taps in FabricPC graph."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        # Create stage outputs (simulating ResNet)
        stage2 = IdentityNode(shape=(32, 32, 16), name="stage2")
        stage3 = IdentityNode(shape=(16, 16, 32), name="stage3")
        stage4 = IdentityNode(shape=(8, 8, 64), name="stage4")

        # Create stage taps
        taps = create_cifar10_stage_taps(embed_dim=64, target_grid=(8, 8))

        # Create depth-spanning column
        column = create_depth_spanning_column(
            name="col0",
            input_dim=64,
            output_dim=64,
            microcolumn_dim=32,
        )

        nodes = [stage2, stage3, stage4] + list(taps.values()) + [column]
        edges = [
            # Stage outputs to taps
            Edge(source=stage2, target=taps["stage2_tap"].slot("in")),
            Edge(source=stage3, target=taps["stage3_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_pool"].slot("in")),
            # Taps to column
            Edge(source=taps["stage2_tap"], target=column.slot("stage2")),
            Edge(source=taps["stage3_tap"], target=column.slot("stage3")),
            Edge(source=taps["stage4_tap"], target=column.slot("stage4")),
            Edge(source=taps["stage4_pool"], target=column.slot("stage4_pool")),
        ]

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )

        assert "col0" in structure.nodes
        params = initialize_params(structure, rng_key)
        assert "col0" in params.nodes

    def test_forward_pass_shapes(self, rng_key):
        """Test forward pass produces correct output shapes."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        # Create stage outputs
        stage2 = IdentityNode(shape=(32, 32, 16), name="stage2")
        stage3 = IdentityNode(shape=(16, 16, 32), name="stage3")
        stage4 = IdentityNode(shape=(8, 8, 64), name="stage4")

        # Create stage taps
        taps = create_cifar10_stage_taps(embed_dim=64)

        # Create column
        column = create_depth_spanning_column(
            name="col0",
            input_dim=64,
            output_dim=64,
        )

        nodes = [stage2, stage3, stage4] + list(taps.values()) + [column]
        edges = [
            Edge(source=stage2, target=taps["stage2_tap"].slot("in")),
            Edge(source=stage3, target=taps["stage3_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_pool"].slot("in")),
            Edge(source=taps["stage2_tap"], target=column.slot("stage2")),
            Edge(source=taps["stage3_tap"], target=column.slot("stage3")),
            Edge(source=taps["stage4_tap"], target=column.slot("stage4")),
            Edge(source=taps["stage4_pool"], target=column.slot("stage4_pool")),
        ]

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        # Create fake stage outputs
        clamps = {
            "stage2": jax.random.normal(key1, (batch_size, 32, 32, 16)),
            "stage3": jax.random.normal(key2, (batch_size, 16, 16, 32)),
            "stage4": jax.random.normal(key3, (batch_size, 8, 8, 64)),
        }

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key4,
            clamps=clamps,
            params=params,
        )

        # Verify column state has correct shape: (batch, tokens, output_dim)
        assert "col0" in state.nodes
        col_state = state.nodes["col0"]
        assert col_state.z_latent.shape == (batch_size, 64, 64)


class TestMultipleColumns:
    """Test using multiple depth-spanning columns together."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_multiple_columns_share_stage_taps(self, rng_key):
        """Test multiple columns sharing the same stage taps."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        # Create stage outputs
        stage2 = IdentityNode(shape=(32, 32, 16), name="stage2")
        stage3 = IdentityNode(shape=(16, 16, 32), name="stage3")
        stage4 = IdentityNode(shape=(8, 8, 64), name="stage4")

        # Shared stage taps
        taps = create_cifar10_stage_taps(embed_dim=64)

        # Multiple columns
        columns = create_depth_spanning_column_pool(
            num_columns=3,
            input_dim=64,
            output_dim=64,
        )

        nodes = [stage2, stage3, stage4] + list(taps.values()) + list(columns.values())

        edges = [
            # Stage outputs to taps (shared)
            Edge(source=stage2, target=taps["stage2_tap"].slot("in")),
            Edge(source=stage3, target=taps["stage3_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_pool"].slot("in")),
        ]

        # Each column gets the same tap outputs
        for col in columns.values():
            edges.extend([
                Edge(source=taps["stage2_tap"], target=col.slot("stage2")),
                Edge(source=taps["stage3_tap"], target=col.slot("stage3")),
                Edge(source=taps["stage4_tap"], target=col.slot("stage4")),
                Edge(source=taps["stage4_pool"], target=col.slot("stage4_pool")),
            ])

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )

        # All columns should be in the graph
        assert "col0" in structure.nodes
        assert "col1" in structure.nodes
        assert "col2" in structure.nodes

        params = initialize_params(structure, rng_key)

        # Each column has independent parameters
        assert "col0" in params.nodes
        assert "col1" in params.nodes
        assert "col2" in params.nodes

        # Parameters should be different (randomly initialized)
        col0_k_deep = params.nodes["col0"].weights["K_W_deep"]
        col1_k_deep = params.nodes["col1"].weights["K_W_deep"]
        assert not jnp.allclose(col0_k_deep, col1_k_deep)


class TestPathwayContributions:
    """Test that K, L, B pathways contribute to output."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_path_scale_affects_output(self, rng_key):
        """Verify that path_scale affects the output."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        # Create stage outputs
        stage2 = IdentityNode(shape=(32, 32, 16), name="stage2")
        stage3 = IdentityNode(shape=(16, 16, 32), name="stage3")
        stage4 = IdentityNode(shape=(8, 8, 64), name="stage4")

        taps = create_cifar10_stage_taps(embed_dim=64)
        column = create_depth_spanning_column(name="col0", input_dim=64, output_dim=64)

        nodes = [stage2, stage3, stage4] + list(taps.values()) + [column]
        edges = [
            Edge(source=stage2, target=taps["stage2_tap"].slot("in")),
            Edge(source=stage3, target=taps["stage3_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_pool"].slot("in")),
            Edge(source=taps["stage2_tap"], target=column.slot("stage2")),
            Edge(source=taps["stage3_tap"], target=column.slot("stage3")),
            Edge(source=taps["stage4_tap"], target=column.slot("stage4")),
            Edge(source=taps["stage4_pool"], target=column.slot("stage4_pool")),
        ]

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 2
        keys = jax.random.split(rng_key, 5)

        clamps = {
            "stage2": jax.random.normal(keys[0], (batch_size, 32, 32, 16)),
            "stage3": jax.random.normal(keys[1], (batch_size, 16, 16, 32)),
            "stage4": jax.random.normal(keys[2], (batch_size, 8, 8, 64)),
        }

        # Get baseline output
        state1 = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=keys[3],
            clamps=clamps,
            params=params,
        )
        output1 = state1.nodes["col0"].z_mu

        # Modify path_scale to zero out K pathway
        modified_params = params._replace(
            nodes={
                **params.nodes,
                "col0": params.nodes["col0"]._replace(
                    weights={
                        **params.nodes["col0"].weights,
                        "path_scale": jnp.array([0.0, 1.0, 1.0]),  # Zero K
                    }
                ),
            }
        )

        state2 = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=keys[4],
            clamps=clamps,
            params=modified_params,
        )
        output2 = state2.nodes["col0"].z_mu

        # Outputs should be different when path_scale changes
        assert not jnp.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
