"""
Tests for the column combiner and classification head.

Verifies that:
1. ColumnCombinerNode correctly combines multiple column outputs
2. ClassificationHeadNode pools tokens and classifies
3. Both integrate with FabricPC graph machinery
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from columnar_cl_fabricpc.columns import (
    ColumnCombinerNode,
    ClassificationHeadNode,
    create_combiner,
    create_classification_head,
)


class TestColumnCombinerBasic:
    """Test basic combiner creation and configuration."""

    def test_create_combiner_node(self):
        """Test basic node creation."""
        node = ColumnCombinerNode(
            shape=(16, 96),
            name="combiner",
            num_columns=5,
            input_dim=96,
            combination="attention",
        )

        assert node.name == "combiner"
        assert node.shape == (16, 96)
        assert node.num_columns == 5
        assert node.combination == "attention"

    def test_create_combiner_helper(self):
        """Test create_combiner helper function."""
        combiner = create_combiner(
            name="test_combiner",
            num_columns=3,
            num_tokens=16,
            embed_dim=64,
            combination="sum",
        )

        assert combiner.name == "test_combiner"
        assert combiner.shape == (16, 64)
        assert combiner.num_columns == 3
        assert combiner.combination == "sum"

    def test_invalid_shape(self):
        """Shape must be 2D."""
        with pytest.raises(ValueError, match="must be.*num_tokens.*output_dim"):
            ColumnCombinerNode(shape=(16,), name="bad", num_columns=5, input_dim=96)

    def test_slots(self):
        """Node should have multi-input slot."""
        slots = ColumnCombinerNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True


class TestCombinerParams:
    """Test combiner parameter initialization."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_attention_params(self, rng_key):
        """Attention combination should have col_attention parameter."""
        node_shape = (16, 96)
        input_shapes = {f"col_{i}:in": (16, 96) for i in range(5)}
        config = {
            "num_columns": 5,
            "input_dim": 96,
            "combination": "attention",
        }

        params = ColumnCombinerNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert "col_attention" in params.weights
        assert params.weights["col_attention"].shape == (5,)
        # Should be initialized to uniform
        assert jnp.allclose(params.weights["col_attention"], 0.2)

    def test_concat_params(self, rng_key):
        """Concat combination should have W_combine parameter."""
        node_shape = (16, 96)
        input_shapes = {f"col_{i}:in": (16, 96) for i in range(5)}
        config = {
            "num_columns": 5,
            "input_dim": 96,
            "combination": "concat",
        }

        params = ColumnCombinerNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # W_combine: (5 * 96, 96) = (480, 96)
        assert "W_combine" in params.weights
        assert params.weights["W_combine"].shape == (480, 96)
        assert "b_combine" in params.biases

    def test_sum_no_extra_params(self, rng_key):
        """Sum combination should have no extra parameters."""
        node_shape = (16, 96)
        input_shapes = {f"col_{i}:in": (16, 96) for i in range(5)}
        config = {
            "num_columns": 5,
            "input_dim": 96,
            "combination": "sum",
        }

        params = ColumnCombinerNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert "col_attention" not in params.weights
        assert "W_combine" not in params.weights


class TestClassificationHeadBasic:
    """Test classification head creation and configuration."""

    def test_create_head_node(self):
        """Test basic node creation."""
        node = ClassificationHeadNode(
            shape=(10,),
            name="classifier",
            input_dim=96,
            num_tokens=16,
            pooling="mean",
        )

        assert node.name == "classifier"
        assert node.shape == (10,)
        assert node.input_dim == 96
        assert node.pooling == "mean"

    def test_create_head_helper(self):
        """Test create_classification_head helper function."""
        head = create_classification_head(
            name="test_cls",
            num_classes=10,
            embed_dim=64,
            num_tokens=16,
            pooling="max",
        )

        assert head.name == "test_cls"
        assert head.shape == (10,)
        assert head.input_dim == 64
        assert head.pooling == "max"

    def test_invalid_shape(self):
        """Shape must be 1D (num_classes,)."""
        with pytest.raises(ValueError, match="must be.*num_classes"):
            ClassificationHeadNode(
                shape=(10, 5), name="bad", input_dim=96, num_tokens=16
            )

    def test_slots(self):
        """Node should have multi-input slot."""
        slots = ClassificationHeadNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True


class TestClassificationHeadParams:
    """Test classification head parameter initialization."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_params_shapes(self, rng_key):
        """Verify parameter shapes."""
        node_shape = (10,)
        input_shapes = {"combiner:in": (16, 96)}
        config = {"input_dim": 96, "num_tokens": 16, "pooling": "mean"}

        params = ClassificationHeadNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # W_cls: (96, 10)
        assert params.weights["W_cls"].shape == (96, 10)
        # b_cls: (10,)
        assert params.biases["b_cls"].shape == (10,)


class TestCombinerInGraph:
    """Test combiner integration with FabricPC graph."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_combiner_in_graph(self, rng_key):
        """Test combiner can be placed in FabricPC graph."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        col_0 = IdentityNode(shape=(16, 96), name="col_0")
        col_1 = IdentityNode(shape=(16, 96), name="col_1")
        col_2 = IdentityNode(shape=(16, 96), name="col_2")
        combiner = create_combiner(
            name="combiner", num_columns=3, num_tokens=16, embed_dim=96
        )

        structure = graph(
            nodes=[col_0, col_1, col_2, combiner],
            edges=[
                Edge(source=col_0, target=combiner.slot("in")),
                Edge(source=col_1, target=combiner.slot("in")),
                Edge(source=col_2, target=combiner.slot("in")),
            ],
            task_map=TaskMap(x=col_0),  # col_0 is input
            inference=InferenceSGD(),
        )

        assert "combiner" in structure.nodes
        params = initialize_params(structure, rng_key)
        assert "combiner" in params.nodes

    def test_full_pipeline(self, rng_key):
        """Test combiner + classification head pipeline."""
        from fabricpc.nodes import IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        # Simulate 3 column outputs
        col_0 = IdentityNode(shape=(16, 96), name="col_0")
        col_1 = IdentityNode(shape=(16, 96), name="col_1")
        col_2 = IdentityNode(shape=(16, 96), name="col_2")

        combiner = create_combiner(
            name="combiner", num_columns=3, num_tokens=16, embed_dim=96
        )
        classifier = create_classification_head(
            name="classifier", num_classes=10, embed_dim=96, num_tokens=16
        )

        structure = graph(
            nodes=[col_0, col_1, col_2, combiner, classifier],
            edges=[
                Edge(source=col_0, target=combiner.slot("in")),
                Edge(source=col_1, target=combiner.slot("in")),
                Edge(source=col_2, target=combiner.slot("in")),
                Edge(source=combiner, target=classifier.slot("in")),
            ],
            task_map=TaskMap(x=col_0, y=classifier),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        # Create fake column outputs
        col_output = jax.random.normal(key1, (batch_size, 16, 96))
        clamps = {"col_0": col_output}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Verify classifier output shape
        assert "classifier" in state.nodes
        cls_state = state.nodes["classifier"]
        assert cls_state.z_latent.shape == (batch_size, 10)


class TestPoolingModes:
    """Test different pooling strategies in classification head."""

    def test_mean_pooling(self):
        """Test mean pooling configuration."""
        head = create_classification_head(pooling="mean")
        assert head.pooling == "mean"

    def test_max_pooling(self):
        """Test max pooling configuration."""
        head = create_classification_head(pooling="max")
        assert head.pooling == "max"

    def test_cls_pooling(self):
        """Test CLS token pooling configuration."""
        head = create_classification_head(pooling="cls")
        assert head.pooling == "cls"


class TestCombinationModes:
    """Test different combination strategies."""

    def test_sum_mode(self):
        """Test sum combination."""
        combiner = create_combiner(combination="sum")
        assert combiner.combination == "sum"

    def test_attention_mode(self):
        """Test attention combination."""
        combiner = create_combiner(combination="attention")
        assert combiner.combination == "attention"

    def test_concat_mode(self):
        """Test concat combination."""
        combiner = create_combiner(combination="concat")
        assert combiner.combination == "concat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
