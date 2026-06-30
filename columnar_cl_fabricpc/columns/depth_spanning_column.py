"""
Depth-spanning columnar node for predictive coding networks.

This node implements a cortical column that spans multiple ResNet stages,
receiving skip connections from stages 2, 3, and 4. Each column contains
three microcolumn pathways (K, L, B) and a typed output shell layout. The
feature axis is partitioned into a hard kernel, inner shell, middle shell, and
outer shell.

Architecture::

                              ┌─────────────────────────────────────────────────────────┐
                              │                    COLUMN c                              │
                              │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
                              │  │     K      │  │     L      │  │     B      │          │
                              │  │  (kernel)  │  │  (lateral) │  │  (bridge)  │          │
     ┌───────────┐            │  │            │  │            │  │            │          │
     │  stage4   │───────────────▶ K_deep    ───▶ L_deep    ───▶ B_deep ◀─── stage4_pool │
     │  tokens   │            │  │   ↓        │  │   ↓        │  │   ↓        │          │
     └─────┬─────┘            │  │ K_mid ◀────│──│───────────│──│───────────│──stage3   │
           │                  │  │   ↓        │  │ L_conv    │  │           │          │
     ┌─────▼─────┐            │  │ K_out ◀────│──│───────────│──│───────────│──stage2   │
     │  stage3   │────────────│──▶           │  │   ↓        │  │           │          │
     │  tokens   │            │  │            │  │ L_out     │  │ B_bcast   │          │
     └─────┬─────┘            │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘          │
           │                  │         │              │              │                  │
     ┌─────▼─────┐            │         └──────────────┼──────────────┘                  │
     │  stage2   │────────────│─────────────────────▶  ▼                                 │
     │  tokens   │            │                  ┌───────────┐                           │
     └───────────┘            │                  │  K + L + B │  column output           │
                              │                  │  weighted  │  (tokens, embed_dim)     │
     ┌───────────┐            │                  └─────┬─────┘                           │
     │stage4_pool│────────────│─────────────────────────────────────────▶ B pathway      │
     │ (1, embed)│            │                        │                                 │
     └───────────┘            └────────────────────────┼─────────────────────────────────┘
                                                       │
                                                       ▼
                                              column output (tokens, embed)

K (Kernel) pathway:
    - Receives concatenated features from all three stages
    - Stable, multi-scale processing via depth-wise integration
    - Skip connections add stage features at each layer

L (Lateral) pathway:
    - Receives stage4 tokens, applies 3×3 conv for local refinement
    - Captures local spatial relationships on token grid

B (Bridge) pathway:
    - Receives global pooled features from stage4
    - Broadcasts context to all token positions
    - Provides global-to-local information flow
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.activations import ActivationBase, IdentityActivation, LeakyReLUActivation
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


SHELL_NAMES = ("hard_kernel", "inner_shell", "middle_shell", "outer_shell")
SHELL_EVIDENCE_CASCADE_PAIRS = tuple(zip(SHELL_NAMES[:-1], SHELL_NAMES[1:]))
DEFAULT_SHELL_PROPORTIONS = (32, 10, 20, 30)
DEFAULT_SHELL_EVIDENCE_CASCADE_SCALE = (0.05, 0.05, 0.05)
DEFAULT_SHELL_INHIBITION_STRENGTHS = (0.0, 0.35, 0.22, 0.10)
DEFAULT_SHELL_PATH_MASK = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def compute_shell_sizes(
    output_dim: int,
    shell_proportions: Tuple[int, int, int, int] = DEFAULT_SHELL_PROPORTIONS,
) -> Tuple[int, int, int, int]:
    """
    Partition the output feature axis into hard-kernel and shell slices.

    `output_dim` is the column output width. `shell_proportions` gives the
    relative widths for hard kernel, inner shell, middle shell, and outer shell.
    """
    if output_dim < len(SHELL_NAMES):
        raise ValueError(
            f"output_dim={output_dim} must be at least {len(SHELL_NAMES)}"
        )
    if len(shell_proportions) != len(SHELL_NAMES):
        raise ValueError("shell_proportions must have four entries")
    if any(value <= 0 for value in shell_proportions):
        raise ValueError("shell_proportions entries must be positive")

    proportions = np.asarray(shell_proportions, dtype=np.float64)
    raw_sizes = proportions * (float(output_dim) / float(proportions.sum()))
    sizes = np.maximum(1, np.floor(raw_sizes).astype(np.int64))

    while int(sizes.sum()) > output_dim:
        removable = np.where(sizes > 1, sizes - raw_sizes, -np.inf)
        idx = int(np.argmax(removable))
        sizes[idx] -= 1

    while int(sizes.sum()) < output_dim:
        deficits = raw_sizes - sizes
        idx = int(np.argmax(deficits))
        sizes[idx] += 1

    return tuple(int(value) for value in sizes)


def get_shell_slices(
    output_dim: int,
    shell_proportions: Tuple[int, int, int, int] = DEFAULT_SHELL_PROPORTIONS,
) -> Dict[str, Tuple[int, int]]:
    """Return feature-axis slice bounds for the column shell layout."""
    sizes = compute_shell_sizes(output_dim, shell_proportions)
    start = 0
    slices: Dict[str, Tuple[int, int]] = {}
    for name, size in zip(SHELL_NAMES, sizes):
        end = start + size
        slices[name] = (start, end)
        start = end
    return slices


def default_shell_path_scale() -> jnp.ndarray:
    """
    Initial shell-to-path mixing weights.

    Rows correspond to hard kernel, inner shell, middle shell, and outer shell.
    Columns correspond to K, L, and B pathways.
    """
    row_counts = np.maximum(DEFAULT_SHELL_PATH_MASK.sum(axis=1, keepdims=True), 1.0)
    return jnp.asarray(DEFAULT_SHELL_PATH_MASK / np.sqrt(row_counts), dtype=jnp.float32)


def default_shell_evidence_cascade_scale() -> jnp.ndarray:
    """
    Initial outward shell evidence-cascade gains.

    Entries correspond to hard-kernel to inner-shell, inner-shell to middle-shell,
    and middle-shell to outer-shell evidence flow.
    """
    return jnp.asarray(DEFAULT_SHELL_EVIDENCE_CASCADE_SCALE, dtype=jnp.float32)


def default_shell_inhibition_strengths() -> jnp.ndarray:
    """
    Initial same-tier inhibition strengths.

    Entries correspond to hard kernel, inner shell, middle shell, and outer shell.
    """
    return jnp.asarray(DEFAULT_SHELL_INHIBITION_STRENGTHS, dtype=jnp.float32)


def shell_evidence_cascade_weight_name(source_shell: str, target_shell: str) -> str:
    """Return the parameter name for one outward shell evidence matrix."""
    return f"shell_evidence_cascade_{source_shell}_to_{target_shell}"


def shell_evidence_cascade_bias_name(source_shell: str, target_shell: str) -> str:
    """Return the parameter name for one outward shell evidence bias."""
    return f"shell_evidence_cascade_b_{source_shell}_to_{target_shell}"


def _hidden_activation(x: jax.Array, activation_name: str, leaky_alpha: float) -> jax.Array:
    """Apply hidden layer activation function."""
    if activation_name == "relu":
        return jax.nn.relu(x)
    if activation_name == "gelu":
        return jax.nn.gelu(x)
    if activation_name == "leaky_relu":
        return jax.nn.leaky_relu(x, negative_slope=leaky_alpha)
    if activation_name == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unknown hidden activation: {activation_name}")


def _shellwise_layernorm(
    shell_outputs: Tuple[jax.Array, ...],
    shell_slices: Dict[str, Tuple[int, int]],
    params: NodeParams,
    fix_ln_gamma: bool,
) -> Tuple[jax.Array, ...]:
    """Normalize each shell output independently along its feature axis."""
    if fix_ln_gamma:
        gamma = jnp.float32(1.0)
        beta = jnp.float32(0.0)
    else:
        gamma = params.weights["ln_gamma"]
        beta = params.biases["ln_beta"]

    normalized = []
    for shell_name, shell_output in zip(SHELL_NAMES, shell_outputs):
        start, end = shell_slices[shell_name]
        if fix_ln_gamma:
            shell_gamma = gamma
            shell_beta = beta
        else:
            shell_gamma = gamma[start:end]
            shell_beta = beta[start:end]
        normalized.append(layernorm(shell_output, shell_gamma, shell_beta))
    return tuple(normalized)


def _same_tier_inhibition(
    shell_output: jax.Array,
    strength: jax.Array,
) -> jax.Array:
    """
    Suppress crowded same-shell activations.

    `strength` is the inhibition coefficient for one shell tier. Each feature's
    magnitude is reduced by the mean magnitude of the other features in the
    same tier, while the feature sign is preserved.
    """
    width = shell_output.shape[-1]
    if width <= 1:
        return shell_output
    magnitude = jnp.abs(shell_output)
    other_mean = (jnp.sum(magnitude, axis=-1, keepdims=True) - magnitude) / float(
        width - 1
    )
    inhibited_magnitude = jax.nn.relu(magnitude - strength * other_mean)
    return jnp.sign(shell_output) * inhibited_magnitude


def _apply_pathway_shell_inhibition(
    pathway_output: jax.Array,
    shell_slices: Dict[str, Tuple[int, int]],
    inhibition_strengths: Tuple[float, float, float, float],
) -> jax.Array:
    """Apply same-tier inhibition independently to each shell of one pathway."""
    inhibited_shells = []
    for shell_idx, shell_name in enumerate(SHELL_NAMES):
        start, end = shell_slices[shell_name]
        inhibited_shells.append(
            _same_tier_inhibition(
                pathway_output[..., start:end],
                jnp.asarray(inhibition_strengths[shell_idx], dtype=pathway_output.dtype),
            )
        )
    return jnp.concatenate(inhibited_shells, axis=-1)


class DepthSpanningColumnNode(NodeBase):
    """
    A cortical column that spans multiple ResNet stages via skip connections.

    This node receives inputs from three tokenized stage outputs plus a global
    pool, and processes them through three specialized microcolumn pathways:

    K (Kernel): Multi-scale stable processing
        - K_deep: stage4 → project to μd
        - K_mid: K_deep + stage3 → project to μd (skip connection)
        - K_out: K_mid + stage2 → project to output_dim (skip connection)

    L (Lateral): Local spatial refinement
        - L_deep: stage4 → project to μd
        - L_conv: 3×3 depthwise conv on token grid
        - L_out: project to output_dim

    B (Bridge): Global context broadcast
        - B_deep: stage4_pool → project to μd
        - B_bcast: broadcast to all tokens, project to output_dim

    Output: Shell-typed mixture of K, L, and B pathways

    Input slots:
        - stage2: (batch, tokens, embed_dim) from stage 2 tap
        - stage3: (batch, tokens, embed_dim) from stage 3 tap
        - stage4: (batch, tokens, embed_dim) from stage 4 tap
        - stage4_pool: (batch, 1, embed_dim) global pool from stage 4

    Output: (batch, tokens, output_dim)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        input_dim: int,
        microcolumn_dim: int = 32,
        grid_size: Tuple[int, int] = (8, 8),
        hidden_activation: str = "leaky_relu",
        leaky_alpha: float = 0.1,
        shell_proportions: Tuple[int, int, int, int] = DEFAULT_SHELL_PROPORTIONS,
        shell_evidence_cascade_scale: Tuple[
            float, float, float
        ] = DEFAULT_SHELL_EVIDENCE_CASCADE_SCALE,
        shell_inhibition_strengths: Tuple[
            float, float, float, float
        ] = DEFAULT_SHELL_INHIBITION_STRENGTHS,
        apply_layer_norm: bool = False,
        fix_ln_gamma: bool = False,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize depth-spanning column.

        Args:
            shape: Output shape (tokens, output_dim), e.g., (64, 64)
            name: Node name
            input_dim: Dimension of input tokens from stage taps
            microcolumn_dim: Internal dimension for microcolumn processing (μd)
            grid_size: Token grid size (h, w) for L pathway conv
            hidden_activation: Activation for hidden layers
            leaky_alpha: Alpha for leaky ReLU
            shell_proportions: Relative output widths for hard kernel, inner shell,
                middle shell, and outer shell
            shell_evidence_cascade_scale: Initial outward evidence gains for
                adjacent shell pairs
            shell_inhibition_strengths: Same-tier inhibition coefficients in
                hard kernel, inner shell, middle shell, and outer shell order
            activation: Output activation
            energy: Energy functional
            weight_init: Weight initializer
            latent_init: Latent state initializer
        """
        if len(shape) != 2:
            raise ValueError(
                f"DepthSpanningColumnNode shape must be (tokens, output_dim), got {shape}"
            )

        tokens, output_dim = shape
        expected_tokens = grid_size[0] * grid_size[1]
        if tokens != expected_tokens:
            raise ValueError(
                f"shape[0]={tokens} != grid_size product {expected_tokens}"
            )
        compute_shell_sizes(output_dim, shell_proportions)
        if len(shell_evidence_cascade_scale) != len(SHELL_EVIDENCE_CASCADE_PAIRS):
            raise ValueError(
                "shell_evidence_cascade_scale must have three entries"
            )
        if len(shell_inhibition_strengths) != len(SHELL_NAMES):
            raise ValueError("shell_inhibition_strengths must have four entries")
        if any(value < 0.0 for value in shell_evidence_cascade_scale):
            raise ValueError("shell_evidence_cascade_scale entries must be non-negative")
        if any(value < 0.0 for value in shell_inhibition_strengths):
            raise ValueError("shell_inhibition_strengths entries must be non-negative")

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            input_dim=input_dim,
            microcolumn_dim=microcolumn_dim,
            grid_size=grid_size,
            hidden_activation=hidden_activation,
            leaky_alpha=leaky_alpha,
            shell_proportions=shell_proportions,
            shell_evidence_cascade_scale=shell_evidence_cascade_scale,
            shell_inhibition_strengths=shell_inhibition_strengths,
            apply_layer_norm=apply_layer_norm,
            fix_ln_gamma=fix_ln_gamma,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Four input slots for multi-stage skip connections.

        - stage2: Tokenized features from ResNet stage 2
        - stage3: Tokenized features from ResNet stage 3
        - stage4: Tokenized features from ResNet stage 4
        - stage4_pool: Global pooled features from stage 4
        """
        return {
            "stage2": SlotSpec(name="stage2", is_multi_input=False),
            "stage3": SlotSpec(name="stage3", is_multi_input=False),
            "stage4": SlotSpec(name="stage4", is_multi_input=False),
            "stage4_pool": SlotSpec(name="stage4_pool", is_multi_input=False),
        }

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Fan-in is the input dimension."""
        return config.get("input_dim", source_shape[-1])

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize parameters for all three microcolumn pathways and the shell mixer.

        K pathway (skip connections from all stages):
            - K_W_deep: input_dim → μd (processes stage4)
            - K_W_mid: μd + input_dim → μd (skip from stage3)
            - K_W_out: μd + input_dim → output_dim (skip from stage2)

        L pathway (local refinement):
            - L_W_deep: input_dim → μd
            - L_W_conv: 3×3 depthwise conv on μd channels
            - L_W_out: μd → output_dim

        B pathway (global broadcast):
            - B_W_deep: input_dim → μd
            - B_W_out: μd → output_dim

        shell_path_scale: Learnable shell-specific weights for combining K, L, and B
        shell_evidence_cascade_*: Learned projections from each shell into the next shell
        """
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = KaimingInitializer()

        input_dim = config.get("input_dim")
        microcolumn_dim = config.get("microcolumn_dim", 32)
        grid_size = config.get("grid_size", (8, 8))
        tokens, output_dim = node_shape

        if input_dim is None:
            raise ValueError("DepthSpanningColumnNode requires input_dim in config")

        # μd is shorthand for microcolumn_dim
        μd = microcolumn_dim

        keys = jax.random.split(key, 20)
        ki = 0  # key index

        weights = {}
        biases = {}

        # =====================================================================
        # K pathway: multi-scale via skip connections
        # =====================================================================
        # K_deep: stage4 → μd
        weights["K_W_deep"] = initialize(keys[ki], (input_dim, μd), weight_init)
        biases["K_b_deep"] = jnp.zeros((μd,))
        ki += 1

        # K_mid: concat(K_deep_out, stage3) → μd
        # Input is μd + input_dim after concatenation
        weights["K_W_mid"] = initialize(keys[ki], (μd + input_dim, μd), weight_init)
        biases["K_b_mid"] = jnp.zeros((μd,))
        ki += 1

        # K_out: concat(K_mid_out, stage2) → output_dim
        weights["K_W_out"] = initialize(keys[ki], (μd + input_dim, output_dim), weight_init)
        biases["K_b_out"] = jnp.zeros((output_dim,))
        ki += 1

        # =====================================================================
        # L pathway: local spatial refinement
        # =====================================================================
        # L_deep: stage4 → μd
        weights["L_W_deep"] = initialize(keys[ki], (input_dim, μd), weight_init)
        biases["L_b_deep"] = jnp.zeros((μd,))
        ki += 1

        # L_conv: 3×3 depthwise conv on token grid
        # For JAX depthwise conv with feature_group_count=μd:
        # Kernel shape is (kh, kw, 1, μd) where 1 = in_channels per group
        weights["L_W_conv_dw"] = initialize(keys[ki], (3, 3, 1, μd), weight_init)
        biases["L_b_conv"] = jnp.zeros((1, 1, 1, μd))
        ki += 1

        # L_out: μd → output_dim
        weights["L_W_out"] = initialize(keys[ki], (μd, output_dim), weight_init)
        biases["L_b_out"] = jnp.zeros((output_dim,))
        ki += 1

        # =====================================================================
        # B pathway: global context broadcast
        # =====================================================================
        # B_deep: global pool (1, input_dim) → (1, μd)
        weights["B_W_deep"] = initialize(keys[ki], (input_dim, μd), weight_init)
        biases["B_b_deep"] = jnp.zeros((μd,))
        ki += 1

        # B_out: broadcast and project μd → output_dim
        weights["B_W_out"] = initialize(keys[ki], (μd, output_dim), weight_init)
        biases["B_b_out"] = jnp.zeros((output_dim,))
        ki += 1

        # =====================================================================
        # Shell-specific pathway combination weights
        # =====================================================================
        # Rows are hard kernel, inner shell, middle shell, and outer shell.
        # Columns are K, L, and B. The fixed mask in forward preserves the
        # intended typed connectivity even while the active weights can learn.
        weights["shell_path_scale"] = default_shell_path_scale()

        shell_sizes = compute_shell_sizes(
            output_dim,
            tuple(config.get("shell_proportions", DEFAULT_SHELL_PROPORTIONS)),
        )
        shell_size_map = dict(zip(SHELL_NAMES, shell_sizes))
        cascade_init = NormalInitializer(std=0.02)
        for source_shell, target_shell in SHELL_EVIDENCE_CASCADE_PAIRS:
            source_width = shell_size_map[source_shell]
            target_width = shell_size_map[target_shell]
            weights[
                shell_evidence_cascade_weight_name(source_shell, target_shell)
            ] = initialize(
                keys[ki],
                (source_width, target_width),
                cascade_init,
            )
            biases[
                shell_evidence_cascade_bias_name(source_shell, target_shell)
            ] = jnp.zeros((target_width,))
            ki += 1
        weights["shell_evidence_cascade_scale"] = jnp.asarray(
            config.get(
                "shell_evidence_cascade_scale",
                DEFAULT_SHELL_EVIDENCE_CASCADE_SCALE,
            ),
            dtype=jnp.float32,
        )

        # Optional LayerNorm on the combined output along the output_dim axis
        if config.get("apply_layer_norm", False) and not config.get("fix_ln_gamma", False):
            weights["ln_gamma"] = jnp.ones((output_dim,))
            biases["ln_beta"] = jnp.zeros((output_dim,))

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass through all three microcolumn pathways.

        1. K pathway: stage4 → K_deep → concat(stage3) → K_mid → concat(stage2) → K_out
        2. L pathway: stage4 → L_deep → 3×3 conv → L_out
        3. B pathway: stage4_pool → B_deep → broadcast → B_out
        4. Combine: each output shell receives its masked K/L/B mixture
        """
        config = node_info.node_config
        grid_h, grid_w = config.get("grid_size", (8, 8))
        hidden_activation = config.get("hidden_activation", "leaky_relu")
        leaky_alpha = config.get("leaky_alpha", 0.1)
        shell_proportions = tuple(
            config.get("shell_proportions", DEFAULT_SHELL_PROPORTIONS)
        )
        shell_inhibition_strengths = tuple(
            config.get(
                "shell_inhibition_strengths",
                DEFAULT_SHELL_INHIBITION_STRENGTHS,
            )
        )

        # Extract inputs by slot name
        stage2 = None
        stage3 = None
        stage4 = None
        stage4_pool = None

        for edge_key, inp in inputs.items():
            if edge_key.endswith(":stage2"):
                stage2 = inp
            elif edge_key.endswith(":stage3"):
                stage3 = inp
            elif edge_key.endswith(":stage4_pool"):
                stage4_pool = inp
            elif edge_key.endswith(":stage4"):
                stage4 = inp

        if stage2 is None or stage3 is None or stage4 is None or stage4_pool is None:
            raise ValueError(
                f"DepthSpanningColumnNode requires all four inputs. "
                f"Got edge keys: {list(inputs.keys())}"
            )

        batch_size, tokens, _ = stage4.shape

        # =====================================================================
        # K pathway: multi-scale with skip connections
        # =====================================================================
        # K_deep: stage4 → μd
        k_deep = jnp.matmul(stage4, params.weights["K_W_deep"]) + params.biases["K_b_deep"]
        k_deep = _hidden_activation(k_deep, hidden_activation, leaky_alpha)

        # K_mid: concat(k_deep, stage3) → μd (skip connection)
        k_mid_in = jnp.concatenate([k_deep, stage3], axis=-1)
        k_mid = jnp.matmul(k_mid_in, params.weights["K_W_mid"]) + params.biases["K_b_mid"]
        k_mid = _hidden_activation(k_mid, hidden_activation, leaky_alpha)

        # K_out: concat(k_mid, stage2) → output_dim (skip connection)
        k_out_in = jnp.concatenate([k_mid, stage2], axis=-1)
        k_out = jnp.matmul(k_out_in, params.weights["K_W_out"]) + params.biases["K_b_out"]

        # =====================================================================
        # L pathway: local spatial refinement via 3×3 conv
        # =====================================================================
        # L_deep: stage4 → μd
        l_deep = jnp.matmul(stage4, params.weights["L_W_deep"]) + params.biases["L_b_deep"]
        l_deep = _hidden_activation(l_deep, hidden_activation, leaky_alpha)

        # Reshape to spatial grid for conv: (batch, tokens, μd) → (batch, h, w, μd)
        μd = l_deep.shape[-1]
        l_grid = l_deep.reshape(batch_size, grid_h, grid_w, μd)

        # 3×3 depthwise conv with SAME padding
        # feature_group_count=μd makes it depthwise (each channel convolved separately)
        l_conv = jax.lax.conv_general_dilated(
            lhs=l_grid,
            rhs=params.weights["L_W_conv_dw"],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=μd,  # Depthwise convolution
        ) + params.biases["L_b_conv"]
        l_conv = _hidden_activation(l_conv, hidden_activation, leaky_alpha)

        # Reshape back to tokens: (batch, h, w, μd) → (batch, tokens, μd)
        l_conv = l_conv.reshape(batch_size, tokens, μd)

        # L_out: μd → output_dim
        l_out = jnp.matmul(l_conv, params.weights["L_W_out"]) + params.biases["L_b_out"]

        # =====================================================================
        # B pathway: global context broadcast
        # =====================================================================
        # B_deep: (batch, 1, input_dim) → (batch, 1, μd)
        b_deep = jnp.matmul(stage4_pool, params.weights["B_W_deep"]) + params.biases["B_b_deep"]
        b_deep = _hidden_activation(b_deep, hidden_activation, leaky_alpha)

        # B_out: project to output_dim
        b_out = jnp.matmul(b_deep, params.weights["B_W_out"]) + params.biases["B_b_out"]

        # Broadcast to all tokens: (batch, 1, output_dim) → (batch, tokens, output_dim)
        b_out = jnp.broadcast_to(b_out, (batch_size, tokens, b_out.shape[-1]))

        # =====================================================================
        # Combine pathways through shell-typed feature slices
        # =====================================================================
        shell_path_mask = jnp.asarray(DEFAULT_SHELL_PATH_MASK, dtype=k_out.dtype)
        shell_path_scale = params.weights["shell_path_scale"] * shell_path_mask
        shell_slices = get_shell_slices(k_out.shape[-1], shell_proportions)
        k_out = _apply_pathway_shell_inhibition(
            k_out,
            shell_slices,
            shell_inhibition_strengths,
        )
        l_out = _apply_pathway_shell_inhibition(
            l_out,
            shell_slices,
            shell_inhibition_strengths,
        )
        b_out = _apply_pathway_shell_inhibition(
            b_out,
            shell_slices,
            shell_inhibition_strengths,
        )
        shell_outputs = []
        for shell_idx, shell_name in enumerate(SHELL_NAMES):
            start, end = shell_slices[shell_name]
            shell_outputs.append(
                shell_path_scale[shell_idx, 0] * k_out[..., start:end]
                + shell_path_scale[shell_idx, 1] * l_out[..., start:end]
                + shell_path_scale[shell_idx, 2] * b_out[..., start:end]
            )

        # The evidence cascade is an outward shell-to-shell communication path.
        # It is distinct from HiBaCaML consolidation, which moves reusable
        # material inward under multi-task evidence.
        for cascade_idx, (source_shell, target_shell) in enumerate(
            SHELL_EVIDENCE_CASCADE_PAIRS
        ):
            source_idx = SHELL_NAMES.index(source_shell)
            target_idx = SHELL_NAMES.index(target_shell)
            cascaded = (
                jnp.matmul(
                    shell_outputs[source_idx],
                    params.weights[
                        shell_evidence_cascade_weight_name(source_shell, target_shell)
                    ],
                )
                + params.biases[
                    shell_evidence_cascade_bias_name(source_shell, target_shell)
                ]
            )
            shell_outputs[target_idx] = (
                shell_outputs[target_idx]
                + params.weights["shell_evidence_cascade_scale"][cascade_idx]
                * cascaded
            )

        # Optional shell-wise LayerNorm pins each typed slice independently.
        if config.get("apply_layer_norm", False):
            shell_outputs = _shellwise_layernorm(
                tuple(shell_outputs),
                shell_slices,
                params,
                fix_ln_gamma=config.get("fix_ln_gamma", False),
            )

        pre_activation = jnp.concatenate(shell_outputs, axis=-1)

        # Apply output activation
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Compute error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


# =============================================================================
# Factory functions
# =============================================================================


def create_depth_spanning_column(
    name: str,
    input_dim: int = 64,
    output_dim: int = 64,
    microcolumn_dim: int = 32,
    grid_size: Tuple[int, int] = (8, 8),
    hidden_activation: str = "leaky_relu",
    shell_proportions: Tuple[int, int, int, int] = DEFAULT_SHELL_PROPORTIONS,
    shell_evidence_cascade_scale: Tuple[
        float, float, float
    ] = DEFAULT_SHELL_EVIDENCE_CASCADE_SCALE,
    shell_inhibition_strengths: Tuple[
        float, float, float, float
    ] = DEFAULT_SHELL_INHIBITION_STRENGTHS,
    apply_layer_norm: bool = False,
    fix_ln_gamma: bool = False,
) -> DepthSpanningColumnNode:
    """
    Create a single depth-spanning column.

    Args:
        name: Node name (e.g., "col0", "col1")
        input_dim: Dimension of input tokens from stage taps
        output_dim: Dimension of output tokens
        microcolumn_dim: Internal processing dimension (μd)
        grid_size: Token grid size for L pathway conv
        hidden_activation: Activation for hidden layers
        shell_proportions: Relative output widths for hard kernel, inner shell,
            middle shell, and outer shell
        shell_evidence_cascade_scale: Initial outward evidence gains for
            adjacent shell pairs
        shell_inhibition_strengths: Same-tier inhibition coefficients in shell order
        apply_layer_norm: If True, LayerNorm the column output along output_dim
        fix_ln_gamma: If True (and apply_layer_norm), gamma/beta are non-learnable
            scalar 1.0 / 0.0

    Returns:
        Configured DepthSpanningColumnNode
    """
    tokens = grid_size[0] * grid_size[1]
    return DepthSpanningColumnNode(
        shape=(tokens, output_dim),
        name=name,
        input_dim=input_dim,
        microcolumn_dim=microcolumn_dim,
        grid_size=grid_size,
        hidden_activation=hidden_activation,
        shell_proportions=shell_proportions,
        shell_evidence_cascade_scale=shell_evidence_cascade_scale,
        shell_inhibition_strengths=shell_inhibition_strengths,
        apply_layer_norm=apply_layer_norm,
        fix_ln_gamma=fix_ln_gamma,
    )


def create_depth_spanning_column_pool(
    num_columns: int,
    input_dim: int = 64,
    output_dim: int = 64,
    microcolumn_dim: int = 32,
    grid_size: Tuple[int, int] = (8, 8),
    hidden_activation: str = "leaky_relu",
    shell_proportions: Tuple[int, int, int, int] = DEFAULT_SHELL_PROPORTIONS,
    shell_evidence_cascade_scale: Tuple[
        float, float, float
    ] = DEFAULT_SHELL_EVIDENCE_CASCADE_SCALE,
    shell_inhibition_strengths: Tuple[
        float, float, float, float
    ] = DEFAULT_SHELL_INHIBITION_STRENGTHS,
    name_prefix: str = "col",
) -> Dict[str, DepthSpanningColumnNode]:
    """
    Create a pool of depth-spanning columns.

    Args:
        num_columns: Number of columns to create
        input_dim: Dimension of input tokens from stage taps
        output_dim: Dimension of output tokens
        microcolumn_dim: Internal processing dimension (μd)
        grid_size: Token grid size for L pathway conv
        hidden_activation: Activation for hidden layers
        shell_proportions: Relative output widths for hard kernel, inner shell,
            middle shell, and outer shell
        shell_evidence_cascade_scale: Initial outward evidence gains for
            adjacent shell pairs
        shell_inhibition_strengths: Same-tier inhibition coefficients in shell order
        name_prefix: Prefix for column names

    Returns:
        Dictionary mapping column names to nodes
    """
    columns = {}
    for i in range(num_columns):
        name = f"{name_prefix}{i}"
        columns[name] = create_depth_spanning_column(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            microcolumn_dim=microcolumn_dim,
            grid_size=grid_size,
            hidden_activation=hidden_activation,
            shell_proportions=shell_proportions,
            shell_evidence_cascade_scale=shell_evidence_cascade_scale,
            shell_inhibition_strengths=shell_inhibition_strengths,
        )
    return columns
