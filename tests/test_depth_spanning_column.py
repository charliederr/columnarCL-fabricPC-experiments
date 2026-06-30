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

from fabricpc.core.types import NodeParams

from columnar_cl_fabricpc.columns import (
    SHELL_NAMES,
    SHELL_EVIDENCE_CASCADE_PAIRS,
    DepthSpanningColumnNode,
    compute_shell_sizes,
    create_depth_spanning_column,
    create_depth_spanning_column_pool,
    create_cifar10_stage_taps,
    get_shell_slices,
    StageTapTokenizer,
    GlobalPoolNode,
)
from columnar_cl_fabricpc.columns.depth_spanning_column import _shellwise_layernorm


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

    def test_default_shell_sizes_scale_cifar_ratio(self):
        """Default shells use the CIFAR hard/shell width ratio."""
        assert compute_shell_sizes(64) == (22, 7, 14, 21)
        assert get_shell_slices(64) == {
            "hard_kernel": (0, 22),
            "inner_shell": (22, 29),
            "middle_shell": (29, 43),
            "outer_shell": (43, 64),
        }
        assert SHELL_NAMES == (
            "hard_kernel",
            "inner_shell",
            "middle_shell",
            "outer_shell",
        )
        assert SHELL_EVIDENCE_CASCADE_PAIRS == (
            ("hard_kernel", "inner_shell"),
            ("inner_shell", "middle_shell"),
            ("middle_shell", "outer_shell"),
        )


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

        # Shell-specific K/L/B mixer
        assert params.weights["shell_path_scale"].shape == (4, 3)

        # Shell evidence cascade follows the feature widths in SHELL_NAMES order
        assert params.weights[
            "shell_evidence_cascade_hard_kernel_to_inner_shell"
        ].shape == (22, 7)
        assert params.biases[
            "shell_evidence_cascade_b_hard_kernel_to_inner_shell"
        ].shape == (7,)
        assert params.weights[
            "shell_evidence_cascade_inner_shell_to_middle_shell"
        ].shape == (7, 14)
        assert params.biases[
            "shell_evidence_cascade_b_inner_shell_to_middle_shell"
        ].shape == (14,)
        assert params.weights[
            "shell_evidence_cascade_middle_shell_to_outer_shell"
        ].shape == (14, 21)
        assert params.biases[
            "shell_evidence_cascade_b_middle_shell_to_outer_shell"
        ].shape == (21,)
        assert params.weights["shell_evidence_cascade_scale"].shape == (3,)
        assert jnp.all(params.weights["shell_evidence_cascade_scale"] > 0.0)

    def test_initialize_params_requires_input_dim(self, rng_key):
        """Should raise if input_dim not in config."""
        node_shape = (64, 64)
        input_shapes = {"stage4:stage4": (64, 64)}
        config = {"microcolumn_dim": 32}  # Missing input_dim

        with pytest.raises(ValueError, match="input_dim"):
            DepthSpanningColumnNode.initialize_params(
                rng_key, node_shape, input_shapes, config=config
            )

    def test_shell_path_scale_initialization(self, rng_key):
        """Shell path scale should be initialized with typed pathway support."""
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

        expected_scale = jnp.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0), 0.0],
                [1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)],
                [0.0, 1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)],
            ],
            dtype=jnp.float32,
        )
        assert jnp.allclose(params.weights["shell_path_scale"], expected_scale)

    def test_initialize_learnable_layer_norm_params_shapes(self, rng_key):
        """Learnable shell-wise normalization keeps full-width scale vectors."""
        node_shape = (64, 64)
        input_shapes = {
            "stage2:stage2": (64, 64),
            "stage3:stage3": (64, 64),
            "stage4:stage4": (64, 64),
            "stage4_pool:stage4_pool": (1, 64),
        }
        config = {
            "input_dim": 64,
            "microcolumn_dim": 32,
            "apply_layer_norm": True,
            "fix_ln_gamma": False,
        }

        params = DepthSpanningColumnNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert params.weights["ln_gamma"].shape == (64,)
        assert params.biases["ln_beta"].shape == (64,)

    def test_shellwise_layer_norm_normalizes_each_shell(self):
        """Each shell slice is normalized independently on its feature axis."""
        shell_widths = (4, 4, 4, 4)
        shell_slices = get_shell_slices(16, shell_widths)
        shell_outputs = tuple(
            jnp.arange(2 * 3 * width, dtype=jnp.float32).reshape(2, 3, width)
            * float(shell_idx + 1)
            + float(10 * shell_idx)
            for shell_idx, width in enumerate(shell_widths)
        )

        normalized = _shellwise_layernorm(
            shell_outputs,
            shell_slices,
            NodeParams(weights={}, biases={}),
            fix_ln_gamma=True,
        )

        for shell_output in normalized:
            assert jnp.allclose(jnp.mean(shell_output, axis=-1), 0.0, atol=1e-6)
            assert jnp.allclose(jnp.var(shell_output, axis=-1), 1.0, atol=1e-4)


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

    def test_forward_layer_norm_is_shellwise(self, rng_key):
        """A normalized column emits each shell with zero mean and unit variance."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        stage2 = IdentityNode(shape=(4, 16), name="stage2")
        stage3 = IdentityNode(shape=(4, 16), name="stage3")
        stage4 = IdentityNode(shape=(4, 16), name="stage4")
        stage4_pool = IdentityNode(shape=(1, 16), name="stage4_pool")
        column = create_depth_spanning_column(
            name="col0",
            input_dim=16,
            output_dim=16,
            microcolumn_dim=8,
            grid_size=(2, 2),
            shell_proportions=(1, 1, 1, 1),
            apply_layer_norm=True,
            fix_ln_gamma=True,
        )
        structure = graph(
            nodes=[stage2, stage3, stage4, stage4_pool, column],
            edges=[
                Edge(source=stage2, target=column.slot("stage2")),
                Edge(source=stage3, target=column.slot("stage3")),
                Edge(source=stage4, target=column.slot("stage4")),
                Edge(source=stage4_pool, target=column.slot("stage4_pool")),
            ],
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        keys = jax.random.split(rng_key, 5)
        batch_size = 3
        clamps = {
            "stage2": jax.random.normal(keys[0], (batch_size, 4, 16)),
            "stage3": jax.random.normal(keys[1], (batch_size, 4, 16)),
            "stage4": jax.random.normal(keys[2], (batch_size, 4, 16)),
            "stage4_pool": jax.random.normal(keys[3], (batch_size, 1, 16)),
        }

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=keys[4],
            clamps=clamps,
            params=params,
        )
        output = state.nodes["col0"].z_mu

        shell_slices = get_shell_slices(16, (1, 1, 1, 1))
        for shell_name in SHELL_NAMES:
            start, end = shell_slices[shell_name]
            shell = output[..., start:end]
            assert jnp.allclose(jnp.mean(shell, axis=-1), 0.0, atol=1e-5)
            assert jnp.allclose(jnp.var(shell, axis=-1), 1.0, atol=1e-3)

    def test_shell_evidence_cascade_affects_forward_output(self, rng_key):
        """Shell evidence cascade changes later shell outputs inside the column."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        stage2 = IdentityNode(shape=(4, 16), name="stage2")
        stage3 = IdentityNode(shape=(4, 16), name="stage3")
        stage4 = IdentityNode(shape=(4, 16), name="stage4")
        stage4_pool = IdentityNode(shape=(1, 16), name="stage4_pool")
        column = create_depth_spanning_column(
            name="col0",
            input_dim=16,
            output_dim=16,
            microcolumn_dim=8,
            grid_size=(2, 2),
            shell_proportions=(1, 1, 1, 1),
        )
        structure = graph(
            nodes=[stage2, stage3, stage4, stage4_pool, column],
            edges=[
                Edge(source=stage2, target=column.slot("stage2")),
                Edge(source=stage3, target=column.slot("stage3")),
                Edge(source=stage4, target=column.slot("stage4")),
                Edge(source=stage4_pool, target=column.slot("stage4_pool")),
            ],
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        keys = jax.random.split(rng_key, 5)
        batch_size = 2
        clamps = {
            "stage2": jax.random.normal(keys[0], (batch_size, 4, 16)),
            "stage3": jax.random.normal(keys[1], (batch_size, 4, 16)),
            "stage4": jax.random.normal(keys[2], (batch_size, 4, 16)),
            "stage4_pool": jax.random.normal(keys[3], (batch_size, 1, 16)),
        }
        node_params = params.nodes["col0"]
        zero_cascade_params = params._replace(
            nodes={
                **params.nodes,
                "col0": node_params._replace(
                    weights={
                        **node_params.weights,
                        "shell_evidence_cascade_scale": jnp.zeros((3,)),
                    }
                ),
            }
        )
        strong_cascade_weights = {
            **node_params.weights,
            "shell_evidence_cascade_scale": jnp.ones((3,)),
            "shell_evidence_cascade_hard_kernel_to_inner_shell": jnp.full(
                (4, 4), 0.25
            ),
            "shell_evidence_cascade_inner_shell_to_middle_shell": jnp.full(
                (4, 4), 0.25
            ),
            "shell_evidence_cascade_middle_shell_to_outer_shell": jnp.full(
                (4, 4), 0.25
            ),
        }
        strong_cascade_params = params._replace(
            nodes={
                **params.nodes,
                "col0": node_params._replace(weights=strong_cascade_weights),
            }
        )

        state_without_cascade = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=keys[4],
            clamps=clamps,
            params=zero_cascade_params,
        )
        state_with_cascade = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=keys[4],
            clamps=clamps,
            params=strong_cascade_params,
        )

        assert not jnp.allclose(
            state_without_cascade.nodes["col0"].z_mu,
            state_with_cascade.nodes["col0"].z_mu,
        )

    def test_shell_inhibition_strengths_affect_forward_output(self, rng_key):
        """Same-tier shell inhibition changes the column output."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        stage2 = IdentityNode(shape=(4, 16), name="stage2")
        stage3 = IdentityNode(shape=(4, 16), name="stage3")
        stage4 = IdentityNode(shape=(4, 16), name="stage4")
        stage4_pool = IdentityNode(shape=(1, 16), name="stage4_pool")
        column_without_inhibition = create_depth_spanning_column(
            name="col0",
            input_dim=16,
            output_dim=16,
            microcolumn_dim=8,
            grid_size=(2, 2),
            shell_proportions=(1, 1, 1, 1),
            shell_inhibition_strengths=(0.0, 0.0, 0.0, 0.0),
        )
        column_with_inhibition = create_depth_spanning_column(
            name="col1",
            input_dim=16,
            output_dim=16,
            microcolumn_dim=8,
            grid_size=(2, 2),
            shell_proportions=(1, 1, 1, 1),
            shell_inhibition_strengths=(0.0, 0.35, 0.22, 0.10),
        )
        structure = graph(
            nodes=[
                stage2,
                stage3,
                stage4,
                stage4_pool,
                column_without_inhibition,
                column_with_inhibition,
            ],
            edges=[
                Edge(source=stage2, target=column_without_inhibition.slot("stage2")),
                Edge(source=stage3, target=column_without_inhibition.slot("stage3")),
                Edge(source=stage4, target=column_without_inhibition.slot("stage4")),
                Edge(
                    source=stage4_pool,
                    target=column_without_inhibition.slot("stage4_pool"),
                ),
                Edge(source=stage2, target=column_with_inhibition.slot("stage2")),
                Edge(source=stage3, target=column_with_inhibition.slot("stage3")),
                Edge(source=stage4, target=column_with_inhibition.slot("stage4")),
                Edge(
                    source=stage4_pool,
                    target=column_with_inhibition.slot("stage4_pool"),
                ),
            ],
            task_map=TaskMap(x=stage2),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        params = params._replace(
            nodes={
                **params.nodes,
                "col1": params.nodes["col0"],
            }
        )
        keys = jax.random.split(rng_key, 5)
        batch_size = 2
        clamps = {
            "stage2": jax.random.normal(keys[0], (batch_size, 4, 16)),
            "stage3": jax.random.normal(keys[1], (batch_size, 4, 16)),
            "stage4": jax.random.normal(keys[2], (batch_size, 4, 16)),
            "stage4_pool": jax.random.normal(keys[3], (batch_size, 1, 16)),
        }

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=keys[4],
            clamps=clamps,
            params=params,
        )

        assert not jnp.allclose(
            state.nodes["col0"].z_mu,
            state.nodes["col1"].z_mu,
        )


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
    """Test that shell-specific K, L, B pathways contribute to output."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_shell_path_scale_affects_output(self, rng_key):
        """Verify that shell_path_scale affects the output."""
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

        # Modify shell_path_scale to zero out the hard-kernel K contribution
        modified_params = params._replace(
            nodes={
                **params.nodes,
                "col0": params.nodes["col0"]._replace(
                    weights={
                        **params.nodes["col0"].weights,
                        "shell_path_scale": params.nodes["col0"].weights[
                            "shell_path_scale"
                        ]
                        .at[0, 0]
                        .set(0.0),
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

        # Outputs should be different when shell_path_scale changes
        assert not jnp.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
