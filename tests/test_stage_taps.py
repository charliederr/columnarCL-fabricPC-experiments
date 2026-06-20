"""
Tests for the stage-tapping infrastructure.

Verifies that StageTapTokenizer and GlobalPoolNode correctly:
1. Handle different spatial resolutions from ResNet stages
2. Adaptively pool to target grid size
3. Project to embedding dimension
4. Add position embeddings
5. Integrate with FabricPC graph machinery
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from columnar_cl_fabricpc.columns import (
    StageTapTokenizer,
    GlobalPoolNode,
    create_stage_tap,
    create_global_pool,
    create_cifar10_stage_taps,
)


class TestAdaptivePooling:
    """Test adaptive average pooling logic."""

    def test_pool_same_size(self):
        """No pooling when input matches target."""
        x = jnp.ones((2, 8, 8, 64))
        pooled = StageTapTokenizer._adaptive_avg_pool_2d(x, 8, 8)
        assert pooled.shape == (2, 8, 8, 64)
        assert jnp.allclose(pooled, x)

    def test_pool_2x_downsample(self):
        """Pool 16×16 → 8×8."""
        x = jnp.ones((2, 16, 16, 32))
        pooled = StageTapTokenizer._adaptive_avg_pool_2d(x, 8, 8)
        assert pooled.shape == (2, 8, 8, 32)

    def test_pool_4x_downsample(self):
        """Pool 32×32 → 8×8."""
        x = jnp.ones((2, 32, 32, 16))
        pooled = StageTapTokenizer._adaptive_avg_pool_2d(x, 8, 8)
        assert pooled.shape == (2, 8, 8, 16)

    def test_pool_preserves_channel_values(self):
        """Verify pooling averages correctly."""
        # Create input where each 4×4 block has uniform value
        x = jnp.zeros((1, 32, 32, 1))
        # Top-left 4×4 block (maps to output position [0,0]) = 1.0
        x = x.at[0, 0:4, 0:4, 0].set(1.0)
        # Second 4×4 block (maps to output position [0,1]) = 2.0
        x = x.at[0, 0:4, 4:8, 0].set(2.0)

        pooled = StageTapTokenizer._adaptive_avg_pool_2d(x, 8, 8)

        # With 32→8 pooling, each output cell averages a 4×4 input block
        assert pooled.shape == (1, 8, 8, 1)
        assert jnp.isclose(pooled[0, 0, 0, 0], 1.0)
        assert jnp.isclose(pooled[0, 0, 1, 0], 2.0)


class TestStageTapTokenizer:
    """Test StageTapTokenizer instantiation and configuration."""

    def test_create_node(self):
        """Test basic node creation."""
        node = StageTapTokenizer(
            shape=(64, 64),
            name="stage2_tap",
            source_channels=16,
            target_grid=(8, 8),
        )
        assert node.name == "stage2_tap"
        assert node.shape == (64, 64)

    def test_create_stage_tap_helper(self):
        """Test the create_stage_tap helper function."""
        node = create_stage_tap(
            name="test_tap",
            source_channels=32,
            embed_dim=64,
            target_grid=(8, 8),
        )
        assert node.shape == (64, 64)  # 8×8 tokens, 64 embed_dim

    def test_invalid_shape_dimensions(self):
        """Shape must be 2D (tokens, embed_dim)."""
        with pytest.raises(ValueError, match="must be.*tokens.*embed_dim"):
            StageTapTokenizer(
                shape=(64,),
                name="bad",
                source_channels=16,
            )

    def test_invalid_token_count(self):
        """Shape[0] must match target_grid product."""
        with pytest.raises(ValueError, match="target_grid product"):
            StageTapTokenizer(
                shape=(100, 64),  # 100 != 8×8=64
                name="bad",
                source_channels=16,
                target_grid=(8, 8),
            )

    def test_slots(self):
        """Node should have single multi-input slot."""
        slots = StageTapTokenizer.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True


class TestStageTapParams:
    """Test parameter initialization."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_initialize_params_shapes(self, rng_key):
        """Verify parameter shapes are correct."""
        node_shape = (64, 64)  # 64 tokens, 64 embed_dim
        input_shapes = {"source:in": (16, 16, 32)}
        config = {
            "source_channels": 32,
            "target_grid": (8, 8),
            "add_pos_embed": True,
        }

        params = StageTapTokenizer.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # Projection: (source_channels, embed_dim) = (32, 64)
        assert params.weights["W_proj"].shape == (32, 64)

        # Position embedding: (1, tokens, embed_dim) = (1, 64, 64)
        assert params.weights["pos_embed"].shape == (1, 64, 64)

        # Bias: (embed_dim,) = (64,)
        assert params.biases["b_proj"].shape == (64,)

    def test_initialize_params_no_pos_embed(self, rng_key):
        """Verify no position embedding when disabled."""
        node_shape = (64, 64)
        input_shapes = {"source:in": (8, 8, 64)}
        config = {
            "source_channels": 64,
            "target_grid": (8, 8),
            "add_pos_embed": False,
        }

        params = StageTapTokenizer.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert "pos_embed" not in params.weights

    def test_initialize_params_requires_source_channels(self, rng_key):
        """Should raise if source_channels not in config."""
        node_shape = (64, 64)
        input_shapes = {"source:in": (8, 8, 64)}
        config = {"target_grid": (8, 8)}  # Missing source_channels

        with pytest.raises(ValueError, match="source_channels"):
            StageTapTokenizer.initialize_params(
                rng_key, node_shape, input_shapes, config=config
            )


class TestGlobalPoolNode:
    """Test GlobalPoolNode instantiation and configuration."""

    def test_create_node(self):
        """Test basic node creation."""
        node = GlobalPoolNode(
            shape=(1, 64),
            name="stage4_pool",
            source_channels=64,
        )
        assert node.name == "stage4_pool"
        assert node.shape == (1, 64)

    def test_create_global_pool_helper(self):
        """Test the create_global_pool helper function."""
        node = create_global_pool(
            name="test_pool",
            source_channels=64,
            embed_dim=128,
        )
        assert node.shape == (1, 128)

    def test_invalid_shape_dimensions(self):
        """Shape must be 2D (1, embed_dim)."""
        with pytest.raises(ValueError, match="must be.*1.*embed_dim"):
            GlobalPoolNode(
                shape=(64,),
                name="bad",
                source_channels=64,
            )

    def test_invalid_token_count(self):
        """Shape[0] must be 1."""
        with pytest.raises(ValueError, match="must be 1"):
            GlobalPoolNode(
                shape=(4, 64),  # 4 != 1
                name="bad",
                source_channels=64,
            )

    def test_slots(self):
        """Node should have single multi-input slot."""
        slots = GlobalPoolNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True


class TestGlobalPoolParams:
    """Test GlobalPoolNode parameter initialization."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_initialize_params_shapes(self, rng_key):
        """Verify parameter shapes are correct."""
        node_shape = (1, 128)
        input_shapes = {"source:in": (8, 8, 64)}
        config = {"source_channels": 64}

        params = GlobalPoolNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # Projection: (source_channels, embed_dim) = (64, 128)
        assert params.weights["W_proj"].shape == (64, 128)

        # Bias: (embed_dim,) = (128,)
        assert params.biases["b_proj"].shape == (128,)


class TestCifar10StageTaps:
    """Test the CIFAR-10 stage taps factory function."""

    def test_create_cifar10_stage_taps(self):
        """Test creating all stage taps for CIFAR-10."""
        taps = create_cifar10_stage_taps(embed_dim=64, target_grid=(8, 8))

        assert "stage2_tap" in taps
        assert "stage3_tap" in taps
        assert "stage4_tap" in taps
        assert "stage4_pool" in taps

        # All taps should produce 64 tokens of dimension 64
        assert taps["stage2_tap"].shape == (64, 64)
        assert taps["stage3_tap"].shape == (64, 64)
        assert taps["stage4_tap"].shape == (64, 64)

        # Pool produces single token
        assert taps["stage4_pool"].shape == (1, 64)

    def test_different_embed_dims(self):
        """Test with different embedding dimensions."""
        for embed_dim in [32, 64, 128]:
            taps = create_cifar10_stage_taps(embed_dim=embed_dim)
            for name, tap in taps.items():
                assert tap.shape[-1] == embed_dim


class TestStageTapInGraph:
    """Test StageTapTokenizer integration with FabricPC graph."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_in_graph(self, rng_key):
        """Test node can be placed in FabricPC graph."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        # Simulate ResNet stage output
        stage_output = IdentityNode(shape=(16, 16, 32), name="stage3")
        stage_tap = create_stage_tap(
            name="stage3_tap",
            source_channels=32,
            embed_dim=64,
            target_grid=(8, 8),
        )

        structure = graph(
            nodes=[stage_output, stage_tap],
            edges=[
                Edge(source=stage_output, target=stage_tap.slot("in")),
            ],
            task_map=TaskMap(x=stage_output),
            inference=InferenceSGD(),
        )

        assert len(structure.nodes) == 2
        assert "stage3_tap" in structure.nodes

        params = initialize_params(structure, rng_key)
        assert "stage3_tap" in params.nodes

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

        # Test with stage 2 (32×32×16) → 64 tokens
        stage_output = IdentityNode(shape=(32, 32, 16), name="stage2")
        stage_tap = create_stage_tap(
            name="stage2_tap",
            source_channels=16,
            embed_dim=64,
            target_grid=(8, 8),
        )

        structure = graph(
            nodes=[stage_output, stage_tap],
            edges=[
                Edge(source=stage_output, target=stage_tap.slot("in")),
            ],
            task_map=TaskMap(x=stage_output),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        # Create fake stage output
        stage_data = jax.random.normal(key1, (batch_size, 32, 32, 16))
        clamps = {"stage2": stage_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Verify stage tap state has correct shape
        assert "stage2_tap" in state.nodes
        tap_state = state.nodes["stage2_tap"]
        assert tap_state.z_latent.shape == (batch_size, 64, 64)


class TestGlobalPoolInGraph:
    """Test GlobalPoolNode integration with FabricPC graph."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

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

        # Test with stage 4 (8×8×64) → single context token
        stage_output = IdentityNode(shape=(8, 8, 64), name="stage4")
        pool = create_global_pool(
            name="stage4_pool",
            source_channels=64,
            embed_dim=64,
        )

        structure = graph(
            nodes=[stage_output, pool],
            edges=[
                Edge(source=stage_output, target=pool.slot("in")),
            ],
            task_map=TaskMap(x=stage_output),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        # Create fake stage output
        stage_data = jax.random.normal(key1, (batch_size, 8, 8, 64))
        clamps = {"stage4": stage_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Verify pool state has correct shape: (batch, 1, embed_dim)
        assert "stage4_pool" in state.nodes
        pool_state = state.nodes["stage4_pool"]
        assert pool_state.z_latent.shape == (batch_size, 1, 64)


class TestMultipleStageTaps:
    """Test using multiple stage taps together."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_three_stages_parallel(self, rng_key):
        """Test three stage taps operating in parallel."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        # Create stage nodes (simulating ResNet outputs)
        stage2 = IdentityNode(shape=(32, 32, 16), name="stage2")
        stage3 = IdentityNode(shape=(16, 16, 32), name="stage3")
        stage4 = IdentityNode(shape=(8, 8, 64), name="stage4")

        # Create taps
        taps = create_cifar10_stage_taps(embed_dim=64)

        nodes = [stage2, stage3, stage4] + list(taps.values())
        edges = [
            Edge(source=stage2, target=taps["stage2_tap"].slot("in")),
            Edge(source=stage3, target=taps["stage3_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_tap"].slot("in")),
            Edge(source=stage4, target=taps["stage4_pool"].slot("in")),
        ]

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )

        # Verify all nodes are in the graph
        assert "stage2_tap" in structure.nodes
        assert "stage3_tap" in structure.nodes
        assert "stage4_tap" in structure.nodes
        assert "stage4_pool" in structure.nodes

        params = initialize_params(structure, rng_key)
        assert len(params.nodes) == 7  # 3 stages + 4 taps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
