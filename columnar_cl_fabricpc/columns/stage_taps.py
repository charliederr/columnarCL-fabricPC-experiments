"""
Stage-tapping infrastructure for depth-spanning columnar architecture.

These nodes convert ResNet stage outputs (at different spatial resolutions) into
a common token format suitable for depth-spanning columns. Each column receives
inputs from multiple stages via skip connections.

Architecture::

    ResNet Stage 2 (32×32×C2)    ResNet Stage 3 (16×16×C3)    ResNet Stage 4 (8×8×C4)
           │                            │                            │
           ▼                            ▼                            ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │ StageTapTokenizer│        │ StageTapTokenizer│        │ StageTapTokenizer│
    │ pool to 8×8      │        │ pool to 8×8      │        │ direct (8×8)     │
    │ project → embed  │        │ project → embed  │        │ project → embed  │
    └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
             │                           │                           │
             ▼                           ▼                           ▼
    (batch, tokens, embed)      (batch, tokens, embed)      (batch, tokens, embed)
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
                              DepthSpanningColumn slots:
                              stage2, stage3, stage4

    Additionally, GlobalPoolNode provides a single context vector for B pathway:

    ResNet Stage 4 (8×8×C4)
           │
           ▼
    ┌─────────────────┐
    │  GlobalPoolNode │
    │  spatial avg    │
    │  project → embed│
    └────────┬────────┘
             │
             ▼
    (batch, 1, embed)  → B pathway's global context
"""

from __future__ import annotations

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
    ZerosInitializer,
    initialize,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.utils.helpers import layernorm


class StageTapTokenizer(NodeBase):
    """
    Convert a ResNet stage's spatial features into tokens at a target resolution.

    Handles spatial mismatch via adaptive average pooling:
    - Stage 2 (32×32) → pool to target_grid (e.g., 8×8) → 64 tokens
    - Stage 3 (16×16) → pool to target_grid (e.g., 8×8) → 64 tokens
    - Stage 4 (8×8)   → direct (no pooling needed)      → 64 tokens

    Input: (batch, H, W, C) spatial features from a ResNet stage
    Output: (batch, tokens, embed_dim) tokenized representation

    Architecture::

        Stage features (H×W×C)
               │
               ▼
        ┌─────────────────┐
        │ Adaptive Pool   │  H×W → target_h × target_w
        │ (if H > target) │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Reshape to      │  (target_h × target_w, C)
        │ token sequence  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Linear project  │  C → embed_dim
        │ W_proj + b_proj │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ + Position embed│  learnable (1, tokens, embed_dim)
        │   (optional)    │
        └────────┬────────┘
                 │
                 ▼
        Output: (batch, tokens, embed_dim)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        source_channels: int,
        target_grid: Tuple[int, int] = (8, 8),
        add_pos_embed: bool = True,
        apply_layer_norm: bool = False,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize stage tap tokenizer.

        Args:
            shape: Output shape (tokens, embed_dim), e.g., (64, 64)
            name: Node name
            source_channels: Number of channels in source stage (e.g., 64 for stage4)
            target_grid: Target spatial grid (h, w) to pool/interpolate to
            add_pos_embed: Whether to add learnable position embeddings
            apply_layer_norm: If True, layer-normalize the post-projection
                pre_activation along the embed_dim axis before the activation
                function. Pins the output magnitude regardless of upstream scale.
            activation: Output activation (default: Identity)
            energy: Energy functional (default: Gaussian)
            weight_init: Weight initializer
            latent_init: Latent state initializer
        """
        if len(shape) != 2:
            raise ValueError(
                f"StageTapTokenizer shape must be (tokens, embed_dim), got {shape}"
            )

        tokens, embed_dim = shape
        expected_tokens = target_grid[0] * target_grid[1]
        if tokens != expected_tokens:
            raise ValueError(
                f"shape[0]={tokens} != target_grid product {expected_tokens}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            source_channels=source_channels,
            target_grid=target_grid,
            add_pos_embed=add_pos_embed,
            apply_layer_norm=apply_layer_norm,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single input slot for stage features."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Fan-in is the source channel count."""
        return config.get("source_channels", source_shape[-1])

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize projection weights and position embeddings.

        Parameters:
            - W_proj: (source_channels, embed_dim) - projects stage features
            - b_proj: (embed_dim,) - bias for projection
            - pos_embed: (1, tokens, embed_dim) - learnable positions (optional)
        """
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = KaimingInitializer()

        tokens, embed_dim = node_shape
        source_channels = config.get("source_channels")
        add_pos_embed = config.get("add_pos_embed", True)
        apply_layer_norm = config.get("apply_layer_norm", False)

        if source_channels is None:
            raise ValueError("StageTapTokenizer requires source_channels in config")

        keys = jax.random.split(key, 2)

        weights = {
            "W_proj": initialize(keys[0], (source_channels, embed_dim), weight_init),
        }
        biases = {
            "b_proj": jnp.zeros((embed_dim,)),
        }

        if add_pos_embed:
            pos_init = NormalInitializer(std=0.02)
            weights["pos_embed"] = initialize(
                keys[1], (1, tokens, embed_dim), pos_init
            )

        if apply_layer_norm:
            weights["ln_gamma"] = jnp.ones((embed_dim,))
            biases["ln_beta"] = jnp.zeros((embed_dim,))

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def _adaptive_avg_pool_2d(
        x: jnp.ndarray,
        target_h: int,
        target_w: int,
    ) -> jnp.ndarray:
        """
        Adaptive average pooling to target spatial size.

        Args:
            x: Input tensor (batch, H, W, C)
            target_h: Target height
            target_w: Target width

        Returns:
            Pooled tensor (batch, target_h, target_w, C)
        """
        batch, h, w, c = x.shape

        if h == target_h and w == target_w:
            return x

        if h % target_h != 0 or w % target_w != 0:
            # Use reshape-based pooling when dimensions divide evenly
            # For non-divisible cases, use strided slicing with averaging
            pool_h = h // target_h
            pool_w = w // target_w
            # Crop to divisible size
            crop_h = pool_h * target_h
            crop_w = pool_w * target_w
            x = x[:, :crop_h, :crop_w, :]
            h, w = crop_h, crop_w

        pool_h = h // target_h
        pool_w = w // target_w

        # Reshape to (batch, target_h, pool_h, target_w, pool_w, C)
        x = x.reshape(batch, target_h, pool_h, target_w, pool_w, c)
        # Average over pooling windows
        x = jnp.mean(x, axis=(2, 4))

        return x

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: pool, reshape to tokens, project, add position embedding.

        Steps:
        1. Adaptive pool to target grid size
        2. Reshape spatial to token sequence
        3. Project to embedding dimension
        4. Add position embeddings
        5. Compute error and energy
        """
        config = node_info.node_config
        target_grid = config.get("target_grid", (8, 8))
        add_pos_embed = config.get("add_pos_embed", True)
        apply_layer_norm = config.get("apply_layer_norm", False)
        target_h, target_w = target_grid
        tokens = target_h * target_w

        # Combine inputs (typically just one stage input)
        x = None
        for edge_key, inp in inputs.items():
            if x is None:
                x = inp
            else:
                x = x + inp

        batch_size = x.shape[0]

        # Adaptive pool to target grid
        x = StageTapTokenizer._adaptive_avg_pool_2d(x, target_h, target_w)

        # Reshape to tokens: (batch, target_h, target_w, C) → (batch, tokens, C)
        x = x.reshape(batch_size, tokens, -1)

        # Project to embed_dim
        pre_activation = jnp.matmul(x, params.weights["W_proj"]) + params.biases["b_proj"]

        # Add position embeddings
        if add_pos_embed and "pos_embed" in params.weights:
            pre_activation = pre_activation + params.weights["pos_embed"]

        # Optional LayerNorm along embed_dim — pins output magnitude
        if apply_layer_norm and "ln_gamma" in params.weights:
            pre_activation = layernorm(
                pre_activation,
                params.weights["ln_gamma"],
                params.biases["ln_beta"],
            )

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


class GlobalPoolNode(NodeBase):
    """
    Global average pooling node for the B (Bridge) pathway.

    Takes spatial features and produces a single context vector per sample,
    which is then projected and broadcast to all token positions.

    Input: (batch, H, W, C) spatial features
    Output: (batch, 1, embed_dim) global context vector

    Architecture::

        Stage features (H×W×C)
               │
               ▼
        ┌─────────────────┐
        │ Global Avg Pool │  H×W×C → C
        │ (spatial mean)  │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Linear project  │  C → embed_dim
        │ W_proj + b_proj │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Expand to       │  embed_dim → (1, embed_dim)
        │ token format    │
        └────────┬────────┘
                 │
                 ▼
        Output: (batch, 1, embed_dim)

    The single token can be broadcast to all positions in the B pathway
    of a depth-spanning column.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        source_channels: int,
        apply_layer_norm: bool = False,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize global pool node.

        Args:
            shape: Output shape (1, embed_dim), e.g., (1, 64)
            name: Node name
            source_channels: Number of channels in source stage
            apply_layer_norm: If True, layer-normalize the post-projection
                pre_activation along the embed_dim axis before the activation.
            activation: Output activation (default: Identity)
            energy: Energy functional (default: Gaussian)
            weight_init: Weight initializer
            latent_init: Latent state initializer
        """
        if len(shape) != 2:
            raise ValueError(
                f"GlobalPoolNode shape must be (1, embed_dim), got {shape}"
            )
        if shape[0] != 1:
            raise ValueError(
                f"GlobalPoolNode shape[0] must be 1 (single token), got {shape[0]}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            source_channels=source_channels,
            apply_layer_norm=apply_layer_norm,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single input slot for stage features."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Fan-in is the source channel count."""
        return config.get("source_channels", source_shape[-1])

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize projection weights.

        Parameters:
            - W_proj: (source_channels, embed_dim) - projects pooled features
            - b_proj: (embed_dim,) - bias for projection
        """
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = KaimingInitializer()

        _, embed_dim = node_shape
        source_channels = config.get("source_channels")
        apply_layer_norm = config.get("apply_layer_norm", False)

        if source_channels is None:
            raise ValueError("GlobalPoolNode requires source_channels in config")

        key_w, _ = jax.random.split(key)

        weights = {
            "W_proj": initialize(key_w, (source_channels, embed_dim), weight_init),
        }
        biases = {
            "b_proj": jnp.zeros((embed_dim,)),
        }

        if apply_layer_norm:
            weights["ln_gamma"] = jnp.ones((embed_dim,))
            biases["ln_beta"] = jnp.zeros((embed_dim,))

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: global pool, project, expand to token format.

        Steps:
        1. Global average pool over spatial dimensions
        2. Project to embedding dimension
        3. Expand to (batch, 1, embed_dim) token format
        4. Compute error and energy
        """
        # Combine inputs
        x = None
        for edge_key, inp in inputs.items():
            if x is None:
                x = inp
            else:
                x = x + inp

        batch_size = x.shape[0]

        # Global average pool: (batch, H, W, C) → (batch, C)
        x = jnp.mean(x, axis=(1, 2))

        # Project to embed_dim: (batch, C) → (batch, embed_dim)
        x = jnp.matmul(x, params.weights["W_proj"]) + params.biases["b_proj"]

        # Expand to token format: (batch, embed_dim) → (batch, 1, embed_dim)
        pre_activation = x[:, None, :]

        # Optional LayerNorm along embed_dim
        config = node_info.node_config
        if config.get("apply_layer_norm", False) and "ln_gamma" in params.weights:
            pre_activation = layernorm(
                pre_activation,
                params.weights["ln_gamma"],
                params.biases["ln_beta"],
            )

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


# =============================================================================
# Factory functions for common configurations
# =============================================================================


def create_stage_tap(
    name: str,
    source_channels: int,
    embed_dim: int = 64,
    target_grid: Tuple[int, int] = (8, 8),
    add_pos_embed: bool = True,
    apply_layer_norm: bool = False,
) -> StageTapTokenizer:
    """
    Create a StageTapTokenizer for a ResNet stage.

    Args:
        name: Node name (e.g., "stage2_tap", "stage3_tap", "stage4_tap")
        source_channels: Channel count of the source stage
        embed_dim: Output embedding dimension
        target_grid: Target spatial grid (h, w)
        add_pos_embed: Whether to add position embeddings
        apply_layer_norm: If True, LayerNorm the output along embed_dim

    Returns:
        Configured StageTapTokenizer
    """
    tokens = target_grid[0] * target_grid[1]
    return StageTapTokenizer(
        shape=(tokens, embed_dim),
        name=name,
        source_channels=source_channels,
        target_grid=target_grid,
        add_pos_embed=add_pos_embed,
        apply_layer_norm=apply_layer_norm,
    )


def create_global_pool(
    name: str,
    source_channels: int,
    embed_dim: int = 64,
    apply_layer_norm: bool = False,
) -> GlobalPoolNode:
    """
    Create a GlobalPoolNode for the B pathway.

    Args:
        name: Node name (e.g., "stage4_pool")
        source_channels: Channel count of the source stage
        embed_dim: Output embedding dimension
        apply_layer_norm: If True, LayerNorm the output along embed_dim

    Returns:
        Configured GlobalPoolNode
    """
    return GlobalPoolNode(
        shape=(1, embed_dim),
        name=name,
        source_channels=source_channels,
        apply_layer_norm=apply_layer_norm,
    )


def create_cifar10_stage_taps(
    embed_dim: int = 64,
    target_grid: Tuple[int, int] = (8, 8),
) -> Dict[str, NodeBase]:
    """
    Create stage taps for a CIFAR-10 ResNet backbone.

    Assumes a ResNet with the following stage output shapes:
    - Stage 2: (32, 32, 16)
    - Stage 3: (16, 16, 32)
    - Stage 4: (8, 8, 64)

    Args:
        embed_dim: Output embedding dimension for all taps
        target_grid: Target spatial grid for tokenization

    Returns:
        Dictionary with keys "stage2_tap", "stage3_tap", "stage4_tap", "stage4_pool"
    """
    return {
        "stage2_tap": create_stage_tap(
            name="stage2_tap",
            source_channels=16,
            embed_dim=embed_dim,
            target_grid=target_grid,
            add_pos_embed=True,
        ),
        "stage3_tap": create_stage_tap(
            name="stage3_tap",
            source_channels=32,
            embed_dim=embed_dim,
            target_grid=target_grid,
            add_pos_embed=True,
        ),
        "stage4_tap": create_stage_tap(
            name="stage4_tap",
            source_channels=64,
            embed_dim=embed_dim,
            target_grid=target_grid,
            add_pos_embed=True,
        ),
        "stage4_pool": create_global_pool(
            name="stage4_pool",
            source_channels=64,
            embed_dim=embed_dim,
        ),
    }
