"""
Tests for the visual stem (patch embedding) node.

Verifies that PatchEmbedNode correctly:
1. Extracts patches from images
2. Projects patches to embedding dimension
3. Adds position embeddings
4. Integrates with FabricPC graph machinery
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from columnar_cl_fabricpc.columns import PatchEmbedNode, create_cifar10_patch_embed


class TestPatchExtraction:
    """Test patch extraction logic."""

    def test_extract_patches_shape(self):
        """Verify extracted patches have correct shape."""
        batch_size = 4
        images = jnp.ones((batch_size, 32, 32, 3))

        patches = PatchEmbedNode.extract_patches(images, patch_size=8)

        # 4×4 grid of 8×8 patches = 16 patches
        # Each patch is 8×8×3 = 192 dims
        assert patches.shape == (batch_size, 16, 192)

    def test_extract_patches_values(self):
        """Verify patches contain correct pixel values."""
        # Create image with distinct values in each patch region
        images = jnp.zeros((1, 32, 32, 3))

        # Set first patch (0:8, 0:8) to all 1s
        images = images.at[0, 0:8, 0:8, :].set(1.0)

        # Set second patch (0:8, 8:16) to all 2s
        images = images.at[0, 0:8, 8:16, :].set(2.0)

        patches = PatchEmbedNode.extract_patches(images, patch_size=8)

        # First patch should be all 1s
        assert jnp.allclose(patches[0, 0, :], 1.0)

        # Second patch should be all 2s
        assert jnp.allclose(patches[0, 1, :], 2.0)

        # Third patch (row 0, col 2) should be 0s
        assert jnp.allclose(patches[0, 2, :], 0.0)

    def test_extract_patches_raster_order(self):
        """Verify patches are in raster (row-major) order."""
        # Create image where each patch has a unique identifying value
        batch = 1
        images = jnp.zeros((batch, 32, 32, 1))

        patch_idx = 0
        for row in range(4):
            for col in range(4):
                r_start, r_end = row * 8, (row + 1) * 8
                c_start, c_end = col * 8, (col + 1) * 8
                images = images.at[0, r_start:r_end, c_start:c_end, 0].set(
                    float(patch_idx)
                )
                patch_idx += 1

        patches = PatchEmbedNode.extract_patches(images, patch_size=8)

        # Each patch should have uniform value equal to its index
        for i in range(16):
            # All elements in patch i should be value i
            assert jnp.allclose(patches[0, i, :], float(i))


class TestPatchEmbedNode:
    """Test PatchEmbedNode instantiation and configuration."""

    def test_create_node(self):
        """Test basic node creation."""
        node = PatchEmbedNode(
            shape=(16, 96),
            name="patch_embed",
            patch_size=8,
            image_size=32,
            in_channels=3,
        )

        assert node.name == "patch_embed"
        assert node.shape == (16, 96)
        assert node.num_patches == 16
        assert node.patch_dim == 192  # 8×8×3

    def test_create_cifar10_helper(self):
        """Test the CIFAR-10 helper function."""
        node = create_cifar10_patch_embed(name="test", embed_dim=96)

        assert node.shape == (16, 96)
        assert node.patch_size == 8
        assert node.image_size == 32
        assert node.in_channels == 3

    def test_invalid_shape_dimensions(self):
        """Shape must be 2D (num_patches, embed_dim)."""
        with pytest.raises(ValueError, match="must be.*num_patches.*embed_dim"):
            PatchEmbedNode(shape=(16,), name="bad")

    def test_invalid_num_patches(self):
        """Shape[0] must match computed num_patches."""
        with pytest.raises(ValueError, match="num_patches"):
            # 32/8 = 4, 4×4 = 16 patches, not 20
            PatchEmbedNode(
                shape=(20, 96),
                name="bad",
                patch_size=8,
                image_size=32,
            )

    def test_slots(self):
        """Node should have single multi-input slot."""
        slots = PatchEmbedNode.get_slots()

        assert "in" in slots
        assert slots["in"].is_multi_input is True


class TestPatchEmbedParams:
    """Test parameter initialization."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_initialize_params_shapes(self, rng_key):
        """Verify parameter shapes are correct."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (32, 32, 3)}
        config = {
            "patch_size": 8,
            "in_channels": 3,
            "add_pos_embed": True,
            "use_bias": True,
        }

        params = PatchEmbedNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        # Projection weights: (192, 96)
        assert params.weights["W_proj"].shape == (192, 96)

        # Position embedding: (1, 16, 96)
        assert params.weights["pos_embed"].shape == (1, 16, 96)

        # Bias: (1, 1, 96)
        assert params.biases["b_proj"].shape == (1, 1, 96)

    def test_initialize_params_no_pos_embed(self, rng_key):
        """Verify no position embedding when disabled."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (32, 32, 3)}
        config = {
            "patch_size": 8,
            "in_channels": 3,
            "add_pos_embed": False,
            "use_bias": True,
        }

        params = PatchEmbedNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert "pos_embed" not in params.weights

    def test_initialize_params_no_bias(self, rng_key):
        """Verify no bias when disabled."""
        node_shape = (16, 96)
        input_shapes = {"source:in": (32, 32, 3)}
        config = {
            "patch_size": 8,
            "in_channels": 3,
            "add_pos_embed": True,
            "use_bias": False,
        }

        params = PatchEmbedNode.initialize_params(
            rng_key, node_shape, input_shapes, config=config
        )

        assert "b_proj" not in params.biases


class TestPatchEmbedInGraph:
    """Test PatchEmbedNode integration with FabricPC graph."""

    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)

    def test_in_graph(self, rng_key):
        """Test node can be placed in FabricPC graph."""
        from fabricpc.nodes import Linear, IdentityNode
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        # Create a simple graph: image -> patch_embed -> linear
        image_node = IdentityNode(shape=(32, 32, 3), name="image")
        patch_embed = create_cifar10_patch_embed(name="patches", embed_dim=96)
        output = Linear(shape=(16, 10), name="output")

        structure = graph(
            nodes=[image_node, patch_embed, output],
            edges=[
                Edge(source=image_node, target=patch_embed.slot("in")),
                Edge(source=patch_embed, target=output.slot("in")),
            ],
            task_map=TaskMap(x=image_node, y=output),
            inference=InferenceSGD(),
        )

        assert len(structure.nodes) == 3
        assert "patches" in structure.nodes

        params = initialize_params(structure, rng_key)
        assert "patches" in params.nodes

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

        image_node = IdentityNode(shape=(32, 32, 3), name="image")
        patch_embed = create_cifar10_patch_embed(name="patches", embed_dim=96)

        structure = graph(
            nodes=[image_node, patch_embed],
            edges=[
                Edge(source=image_node, target=patch_embed.slot("in")),
            ],
            task_map=TaskMap(x=image_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)

        # Create fake CIFAR-10 images
        images = jax.random.normal(key1, (batch_size, 32, 32, 3))
        clamps = {"image": images}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Verify patch embed state has correct shape
        assert "patches" in state.nodes
        patch_state = state.nodes["patches"]
        assert patch_state.z_latent.shape == (batch_size, 16, 96)


class TestPatchEmbedDifferentConfigs:
    """Test PatchEmbedNode with different configurations."""

    def test_different_patch_sizes(self):
        """Test with 4×4 patches (64 patches total)."""
        node = PatchEmbedNode(
            shape=(64, 48),
            name="small_patches",
            patch_size=4,
            image_size=32,
            in_channels=3,
        )

        assert node.num_patches == 64
        assert node.patch_dim == 48  # 4×4×3

    def test_different_embed_dims(self):
        """Test various embedding dimensions."""
        for embed_dim in [32, 64, 128, 256]:
            node = PatchEmbedNode(
                shape=(16, embed_dim),
                name=f"embed_{embed_dim}",
                patch_size=8,
                image_size=32,
            )
            assert node.shape == (16, embed_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
