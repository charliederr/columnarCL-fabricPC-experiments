"""
Visual stem for CIFAR-10 columnar architecture.

Implements patch embedding as described in HiBaCaML Section 4.1:
> Each image is converted into a sequence of sixteen 7×7 non-overlapping patches
> in raster order. Each patch is linearly embedded, coordinates are appended.

For CIFAR-10 (32×32 images), we use 8×8 patches:
- 4×4 grid = 16 patches
- Each patch: 8×8×3 = 192 dimensions
- Embedding dimension: 96 (configurable)
- Output: (batch, 16, embed_dim)

Architecture::

    CIFAR-10 Image (32×32×3)
    ┌────┬────┬────┬────┐
    │ P0 │ P1 │ P2 │ P3 │   Each patch Pn is 8×8×3 = 192 dims
    ├────┼────┼────┼────┤
    │ P4 │ P5 │ P6 │ P7 │   Patches extracted in raster order:
    ├────┼────┼────┼────┤   P0, P1, P2, ... P15
    │ P8 │ P9 │P10 │P11 │
    ├────┼────┼────┼────┤
    │P12 │P13 │P14 │P15 │
    └────┴────┴────┴────┘
            │
            ▼
    ┌───────────────────┐
    │  Linear Projection │   W_proj: (192, embed_dim)
    │  192 → embed_dim   │   Applied to each patch independently
    └───────────────────┘
            │
            ▼
    ┌───────────────────┐
    │ + Position Embed  │   pos_embed: (1, 16, embed_dim)
    │   (learnable)     │   Adds spatial information
    └───────────────────┘
            │
            ▼
    Output: (batch, 16, embed_dim)

    16 tokens, each of dimension embed_dim, ready for columnar processing
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.activations import ActivationBase, IdentityActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.core.initializers import (
    InitializerBase,
    KaimingInitializer,
    NormalInitializer,
    initialize,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec


class PatchEmbedNode(NodeBase):
    """
    Patch embedding node for image inputs.

    Converts images into sequences of embedded patches suitable for
    columnar processing.

    Input: (batch, height, width, channels) e.g., (B, 32, 32, 3)
    Output: (batch, num_patches, embed_dim) e.g., (B, 16, 96)

    Architecture:
    1. Extract non-overlapping patches from input image
    2. Flatten each patch to a vector
    3. Apply linear projection to embedding dimension
    4. Add learnable position embeddings
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        patch_size: int = 8,
        image_size: int = 32,
        in_channels: int = 3,
        add_pos_embed: bool = True,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize patch embedding node.

        Args:
            shape: Output shape (num_patches, embed_dim), e.g., (16, 96)
            name: Node name
            patch_size: Size of each patch (default: 8 for 8×8 patches)
            image_size: Input image size (default: 32 for CIFAR-10)
            in_channels: Number of input channels (default: 3)
            add_pos_embed: Whether to add learnable position embeddings
            activation: Activation function (default: Identity)
            energy: Energy functional (default: Gaussian)
            use_bias: Whether to use bias in projection
            weight_init: Initializer for projection weights
            latent_init: Initializer for latent states
        """
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.add_pos_embed = add_pos_embed

        # Compute derived quantities
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Validate shape
        if len(shape) != 2:
            raise ValueError(
                f"PatchEmbedNode shape must be (num_patches, embed_dim), got {shape}"
            )
        expected_patches = self.num_patches
        if shape[0] != expected_patches:
            raise ValueError(
                f"Shape[0]={shape[0]} != expected num_patches={expected_patches}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            patch_size=patch_size,
            image_size=image_size,
            in_channels=in_channels,
            add_pos_embed=add_pos_embed,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single input slot for image data."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize patch projection and position embedding parameters.

        Parameters:
            - W_proj: (patch_dim, embed_dim) - projects flattened patches
            - b_proj: (1, 1, embed_dim) - bias for projection
            - pos_embed: (1, num_patches, embed_dim) - learnable positions
        """
        if config is None:
            config = {}

        if weight_init is None:
            weight_init = KaimingInitializer()

        num_patches, embed_dim = node_shape
        patch_size = config.get("patch_size", 8)
        in_channels = config.get("in_channels", 3)
        add_pos_embed = config.get("add_pos_embed", True)
        use_bias = config.get("use_bias", True)

        patch_dim = patch_size * patch_size * in_channels

        # Split keys
        key_proj, key_pos = jax.random.split(key)

        weights = {}
        biases = {}

        # Projection weights: (patch_dim, embed_dim)
        proj_shape = (patch_dim, embed_dim)
        weights["W_proj"] = initialize(key_proj, proj_shape, weight_init)

        # Projection bias
        if use_bias:
            biases["b_proj"] = jnp.zeros((1, 1, embed_dim))

        # Position embeddings (initialized from truncated normal)
        if add_pos_embed:
            pos_init = NormalInitializer(std=0.02)
            weights["pos_embed"] = initialize(
                key_pos, (1, num_patches, embed_dim), pos_init
            )

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def extract_patches(
        images: jnp.ndarray,
        patch_size: int,
    ) -> jnp.ndarray:
        """
        Extract non-overlapping patches from images.

        Args:
            images: (batch, height, width, channels)
            patch_size: Size of each square patch

        Returns:
            patches: (batch, num_patches, patch_dim)
        """
        batch, h, w, c = images.shape
        num_h = h // patch_size
        num_w = w // patch_size

        # Reshape to extract patches
        # (batch, num_h, patch_size, num_w, patch_size, c)
        x = images.reshape(batch, num_h, patch_size, num_w, patch_size, c)

        # Transpose to (batch, num_h, num_w, patch_size, patch_size, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        # Flatten patches: (batch, num_patches, patch_dim)
        num_patches = num_h * num_w
        patch_dim = patch_size * patch_size * c
        patches = x.reshape(batch, num_patches, patch_dim)

        return patches

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: extract patches, project, add position embedding.

        Steps:
        1. Extract patches from input image(s)
        2. Project patches to embedding dimension
        3. Add position embeddings
        4. Compute prediction (z_mu) via activation
        5. Compute error and energy
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape  # (num_patches, embed_dim)
        config = node_info.node_config

        patch_size = config.get("patch_size", 8)
        add_pos_embed = config.get("add_pos_embed", True)

        # Combine inputs (typically just one image input)
        # Each input should be (batch, height, width, channels)
        images = None
        for edge_key, x in inputs.items():
            if images is None:
                images = x
            else:
                images = images + x

        # Extract patches: (batch, num_patches, patch_dim)
        patches = PatchEmbedNode.extract_patches(images, patch_size)

        # Project: (batch, num_patches, embed_dim)
        W_proj = params.weights["W_proj"]
        pre_activation = jnp.matmul(patches, W_proj)

        # Add bias if present
        if "b_proj" in params.biases:
            pre_activation = pre_activation + params.biases["b_proj"]

        # Add position embeddings
        if add_pos_embed and "pos_embed" in params.weights:
            pre_activation = pre_activation + params.weights["pos_embed"]

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


def create_cifar10_patch_embed(
    name: str = "patch_embed",
    embed_dim: int = 96,
    patch_size: int = 8,
) -> PatchEmbedNode:
    """
    Create a PatchEmbedNode configured for CIFAR-10.

    Args:
        name: Node name
        embed_dim: Embedding dimension (default: 96)
        patch_size: Patch size (default: 8 for 8×8 patches)

    Returns:
        PatchEmbedNode configured for 32×32×3 CIFAR-10 images
    """
    image_size = 32
    num_patches = (image_size // patch_size) ** 2  # 16 for 8×8 patches

    return PatchEmbedNode(
        shape=(num_patches, embed_dim),
        name=name,
        patch_size=patch_size,
        image_size=image_size,
        in_channels=3,
        add_pos_embed=True,
    )


class ConvStemNode(NodeBase):
    """
    Convolutional stem for CIFAR-10 as recommended by HiBaCaML Section 6.

    From the paper:
    > Shared visual stem. A shallow convolutional stem extracts low-level
    > visual primitives before support selection begins:
    > Conv(3, 32, 3×3) → Conv(32, 64, 3×3, stride 2) → Conv(64, 64, 3×3),
    > followed by light patch pooling to 64–96 tokens of width 64–96.

    Architecture::

        CIFAR-10 Image (32×32×3)
                │
                ▼
        ┌───────────────────┐
        │ Conv 3×3, 32 ch   │  32×32×3 → 32×32×32
        │ + GELU            │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Conv 3×3, stride 2│  32×32×32 → 16×16×64
        │ 64 ch + GELU      │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Conv 3×3, 64 ch   │  16×16×64 → 16×16×64
        │ + GELU            │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Reshape to tokens │  16×16×64 → (256, 64)
        │ + Position embed  │  or pool to (64, 64)
        └─────────┬─────────┘
                  │
                  ▼
        Output: (batch, num_tokens, embed_dim)

    Input: (batch, 32, 32, 3) NHWC format
    Output: (batch, num_tokens, embed_dim)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        num_tokens: int = 64,
        embed_dim: int = 64,
        add_pos_embed: bool = True,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize convolutional stem.

        Args:
            shape: Output shape (num_tokens, embed_dim)
            name: Node name
            num_tokens: Number of output tokens (64 or 256)
            embed_dim: Output embedding dimension (typically 64)
            add_pos_embed: Whether to add learnable position embeddings
            activation: Output activation (default: Identity)
            energy: Energy functional (default: Gaussian)
            weight_init: Weight initializer
            latent_init: Latent state initializer
        """
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.add_pos_embed = add_pos_embed

        if len(shape) != 2:
            raise ValueError(
                f"ConvStemNode shape must be (num_tokens, embed_dim), got {shape}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            add_pos_embed=add_pos_embed,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single input slot for image data."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize conv stem parameters.

        Three conv layers: 3→32, 32→64, 64→64
        Plus optional position embeddings.
        """
        if config is None:
            config = {}

        if weight_init is None:
            weight_init = KaimingInitializer()

        num_tokens, embed_dim = node_shape
        add_pos_embed = config.get("add_pos_embed", True)

        # Split keys for each layer
        keys = jax.random.split(key, 5)

        weights = {}
        biases = {}

        # Conv1: 3×3, 3 → 32 channels
        weights["conv1_W"] = initialize(keys[0], (3, 3, 3, 32), weight_init)
        biases["conv1_b"] = jnp.zeros((1, 1, 1, 32))

        # Conv2: 3×3, 32 → 64 channels, stride 2
        weights["conv2_W"] = initialize(keys[1], (3, 3, 32, 64), weight_init)
        biases["conv2_b"] = jnp.zeros((1, 1, 1, 64))

        # Conv3: 3×3, 64 → 64 channels
        weights["conv3_W"] = initialize(keys[2], (3, 3, 64, 64), weight_init)
        biases["conv3_b"] = jnp.zeros((1, 1, 1, 64))

        # Token projection if embed_dim != 64
        if embed_dim != 64:
            weights["proj_W"] = initialize(keys[3], (64, embed_dim), weight_init)
            biases["proj_b"] = jnp.zeros((embed_dim,))

        # Position embeddings
        if add_pos_embed:
            pos_init = NormalInitializer(std=0.02)
            weights["pos_embed"] = initialize(
                keys[4], (1, num_tokens, embed_dim), pos_init
            )

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass through conv stem.

        Steps:
        1. Apply three conv layers with GELU activation
        2. Reshape spatial features to token sequence
        3. Optionally pool to reduce token count
        4. Add position embeddings
        5. Compute error and energy
        """
        config = node_info.node_config
        num_tokens = config.get("num_tokens", 64)
        embed_dim = config.get("embed_dim", 64)
        add_pos_embed = config.get("add_pos_embed", True)

        # Combine inputs
        x = None
        for edge_key, inp in inputs.items():
            if x is None:
                x = inp
            else:
                x = x + inp

        # x shape: (batch, 32, 32, 3)
        batch_size = x.shape[0]

        # Conv1: (batch, 32, 32, 3) → (batch, 32, 32, 32)
        conv1_W = params.weights["conv1_W"]
        conv1_b = params.biases["conv1_b"]
        x = jax.lax.conv_general_dilated(
            x, conv1_W,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + conv1_b
        x = jax.nn.gelu(x)

        # Conv2 with stride 2: (batch, 32, 32, 32) → (batch, 16, 16, 64)
        conv2_W = params.weights["conv2_W"]
        conv2_b = params.biases["conv2_b"]
        x = jax.lax.conv_general_dilated(
            x, conv2_W,
            window_strides=(2, 2),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + conv2_b
        x = jax.nn.gelu(x)

        # Conv3: (batch, 16, 16, 64) → (batch, 16, 16, 64)
        conv3_W = params.weights["conv3_W"]
        conv3_b = params.biases["conv3_b"]
        x = jax.lax.conv_general_dilated(
            x, conv3_W,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + conv3_b
        x = jax.nn.gelu(x)

        # Reshape to tokens: (batch, 16, 16, 64) → (batch, 256, 64)
        x = x.reshape(batch_size, 256, 64)

        # Pool if needed to reduce token count (e.g., 256 → 64)
        if num_tokens < 256:
            # Simple average pooling over groups of tokens
            pool_size = 256 // num_tokens
            x = x.reshape(batch_size, num_tokens, pool_size, 64)
            x = jnp.mean(x, axis=2)  # (batch, num_tokens, 64)

        # Project to embed_dim if needed
        if "proj_W" in params.weights:
            proj_W = params.weights["proj_W"]
            proj_b = params.biases["proj_b"]
            x = jnp.matmul(x, proj_W) + proj_b

        pre_activation = x

        # Add position embeddings
        if add_pos_embed and "pos_embed" in params.weights:
            pre_activation = pre_activation + params.weights["pos_embed"]

        # Apply output activation
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


def create_cifar10_conv_stem(
    name: str = "conv_stem",
    num_tokens: int = 64,
    embed_dim: int = 64,
) -> ConvStemNode:
    """
    Create a ConvStemNode configured for CIFAR-10 per HiBaCaML recommendations.

    Args:
        name: Node name
        num_tokens: Number of output tokens (default: 64)
        embed_dim: Embedding dimension (default: 64)

    Returns:
        ConvStemNode for 32×32×3 CIFAR-10 images
    """
    return ConvStemNode(
        shape=(num_tokens, embed_dim),
        name=name,
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        add_pos_embed=True,
    )
