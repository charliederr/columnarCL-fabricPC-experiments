"""
Columnar node implementation for HiBaCaML/ColBa architecture.

Each column contains R=3 typed microcolumns as described in Section 4:
- K (kernel): Stable processing with moderate context
- L (lateral): Same-region discriminative refinement
- B (bridge): Cross-region context integration

Architecture::

    Input tokens (batch, num_tokens, embed_dim)
            │
            ▼
    ┌───────────────────────────────────────────────────────────┐
    │                      ColumnarNode                          │
    │                                                            │
    │    ┌─────────┐     ┌─────────┐     ┌─────────┐            │
    │    │    K    │     │    L    │     │    B    │            │
    │    │ (kernel)│     │(lateral)│     │(bridge) │            │
    │    │         │     │         │     │         │            │
    │    │ dm=32   │     │ dm=32   │     │ dm=32   │            │
    │    └────┬────┘     └────┬────┘     └────┬────┘            │
    │         │               │               │                  │
    │         └───────────────┼───────────────┘                  │
    │                         │                                  │
    │                         ▼                                  │
    │                 ┌───────────────┐                          │
    │                 │  Combination  │  Weighted sum or concat  │
    │                 └───────┬───────┘                          │
    │                         │                                  │
    └─────────────────────────┼──────────────────────────────────┘
                              ▼
                    Output (batch, num_tokens, output_dim)

    For CIFAR-10 (single task), all parameters train jointly.
    Shell structure becomes relevant only for continual learning.

Internal Microcolumn Structure (for future continual learning)::

    ┌─────────────────────────────────┐
    │         Microcolumn             │
    │  ┌─────────────────────────┐    │
    │  │    Hard Kernel (dm=32)  │    │  Protected, non-prunable
    │  │    [always active]      │    │
    │  └─────────────────────────┘    │
    │  ┌─────────────────────────┐    │
    │  │   Shell S(1): 10 units  │    │  Inner: reusable abstraction
    │  ├─────────────────────────┤    │
    │  │   Shell S(2): 20 units  │    │  Middle: semi-general
    │  ├─────────────────────────┤    │
    │  │   Shell S(3): 30 units  │    │  Outer: task-local residue
    │  └─────────────────────────┘    │
    └─────────────────────────────────┘
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.activations import ActivationBase, IdentityActivation, GeluActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.core.initializers import (
    InitializerBase,
    KaimingInitializer,
    NormalInitializer,
    initialize,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec


class ColumnarNode(NodeBase):
    """
    A column with K/L/B microcolumns for hierarchical predictive coding.

    Each column processes input tokens through three specialized pathways:
    - K (kernel): Core feature transformation
    - L (lateral): Local discriminative refinement
    - B (bridge): Cross-position context aggregation

    For CIFAR-10 (single task), this is a unified processing unit.
    For continual learning, columns will have shell-based parameter organization.

    Input: (batch, num_tokens, input_dim)
    Output: (batch, num_tokens, output_dim)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        input_dim: int,
        microcolumn_dim: int = 32,
        num_microcolumns: int = 3,
        combination: str = "sum",
        activation: Optional[ActivationBase] = GeluActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize a columnar node.

        Args:
            shape: Output shape (num_tokens, output_dim)
            name: Node name (e.g., "col_00", "col_01", ...)
            input_dim: Input feature dimension
            microcolumn_dim: Width of each microcolumn (dm in paper, default 32)
            num_microcolumns: Number of microcolumns (default 3 for K/L/B)
            combination: How to combine microcolumn outputs ("sum", "concat", "attention")
            activation: Activation function (default: GELU)
            energy: Energy functional (default: Gaussian)
            weight_init: Initializer for weights
            latent_init: Initializer for latent states
        """
        self.input_dim = input_dim
        self.microcolumn_dim = microcolumn_dim
        self.num_microcolumns = num_microcolumns
        self.combination = combination

        # Validate shape
        if len(shape) != 2:
            raise ValueError(
                f"ColumnarNode shape must be (num_tokens, output_dim), got {shape}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            input_dim=input_dim,
            microcolumn_dim=microcolumn_dim,
            num_microcolumns=num_microcolumns,
            combination=combination,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single multi-input slot for token sequences."""
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
        Initialize parameters for K/L/B microcolumns.

        Parameters per microcolumn:
            - W_in: (input_dim, microcolumn_dim) - input projection
            - b_in: (microcolumn_dim,) - input bias
            - W_out: (microcolumn_dim, output_dim) - output projection

        For combination="concat", an additional projection combines outputs.
        For combination="attention", learned attention weights are added.
        """
        if config is None:
            config = {}

        if weight_init is None:
            weight_init = KaimingInitializer()

        num_tokens, output_dim = node_shape
        input_dim = config.get("input_dim", 96)
        microcolumn_dim = config.get("microcolumn_dim", 32)
        num_microcolumns = config.get("num_microcolumns", 3)
        combination = config.get("combination", "sum")

        # Microcolumn names
        mc_names = ["K", "L", "B"][:num_microcolumns]

        # Split keys for each microcolumn
        keys = jax.random.split(key, num_microcolumns * 2 + 1)
        key_idx = 0

        weights = {}
        biases = {}

        for mc_name in mc_names:
            # Input projection: (input_dim, microcolumn_dim)
            weights[f"{mc_name}_W_in"] = initialize(
                keys[key_idx], (input_dim, microcolumn_dim), weight_init
            )
            key_idx += 1

            # Output projection: (microcolumn_dim, output_dim)
            weights[f"{mc_name}_W_out"] = initialize(
                keys[key_idx], (microcolumn_dim, output_dim), weight_init
            )
            key_idx += 1

            # Bias for input projection
            biases[f"{mc_name}_b_in"] = jnp.zeros((microcolumn_dim,))

        # Combination-specific parameters
        if combination == "concat":
            # Project concatenated outputs back to output_dim
            concat_dim = num_microcolumns * output_dim
            weights["W_combine"] = initialize(
                keys[key_idx], (concat_dim, output_dim), weight_init
            )
        elif combination == "attention":
            # Learned attention weights over microcolumns
            weights["mc_attention"] = jnp.ones((num_microcolumns,)) / num_microcolumns

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass through K/L/B microcolumns.

        1. Project input through each microcolumn's input weights
        2. Apply activation
        3. Project through output weights
        4. Combine microcolumn outputs (sum, concat, or attention)
        5. Compute error and energy
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape  # (num_tokens, output_dim)
        config = node_info.node_config

        microcolumn_dim = config.get("microcolumn_dim", 32)
        num_microcolumns = config.get("num_microcolumns", 3)
        combination = config.get("combination", "sum")

        mc_names = ["K", "L", "B"][:num_microcolumns]

        # Combine inputs
        x = None
        for edge_key, inp in inputs.items():
            if x is None:
                x = inp
            else:
                x = x + inp

        # Process through each microcolumn
        mc_outputs = []

        activation = node_info.activation

        for mc_name in mc_names:
            W_in = params.weights[f"{mc_name}_W_in"]
            W_out = params.weights[f"{mc_name}_W_out"]
            b_in = params.biases[f"{mc_name}_b_in"]

            # Input projection: (batch, num_tokens, microcolumn_dim)
            h = jnp.matmul(x, W_in) + b_in

            # Activation
            h = type(activation).forward(h, activation.config)

            # Output projection: (batch, num_tokens, output_dim)
            out = jnp.matmul(h, W_out)

            mc_outputs.append(out)

        # Combine microcolumn outputs
        if combination == "sum":
            pre_activation = sum(mc_outputs)
        elif combination == "concat":
            # Concatenate and project
            concat = jnp.concatenate(mc_outputs, axis=-1)
            W_combine = params.weights["W_combine"]
            pre_activation = jnp.matmul(concat, W_combine)
        elif combination == "attention":
            # Weighted sum with learned attention
            attn = jax.nn.softmax(params.weights["mc_attention"])
            pre_activation = sum(
                attn[i] * mc_outputs[i] for i in range(num_microcolumns)
            )
        else:
            raise ValueError(f"Unknown combination mode: {combination}")

        # z_mu (no additional activation after combination)
        z_mu = pre_activation

        # Error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


def create_column(
    name: str,
    num_tokens: int = 16,
    input_dim: int = 96,
    output_dim: int = 96,
    microcolumn_dim: int = 32,
    combination: str = "sum",
) -> ColumnarNode:
    """
    Create a ColumnarNode with standard configuration.

    Args:
        name: Column name (e.g., "col_00")
        num_tokens: Number of input/output tokens (default: 16 for CIFAR-10)
        input_dim: Input feature dimension (default: 96)
        output_dim: Output feature dimension (default: 96)
        microcolumn_dim: Width of each K/L/B microcolumn (default: 32)
        combination: How to combine outputs ("sum", "concat", "attention")

    Returns:
        Configured ColumnarNode
    """
    return ColumnarNode(
        shape=(num_tokens, output_dim),
        name=name,
        input_dim=input_dim,
        microcolumn_dim=microcolumn_dim,
        combination=combination,
    )


def create_column_pool(
    num_columns: int = 40,
    num_tokens: int = 16,
    input_dim: int = 96,
    output_dim: int = 96,
    microcolumn_dim: int = 32,
    combination: str = "sum",
    prefix: str = "col",
) -> list:
    """
    Create a pool of columns for the ColBa architecture.

    From Section 6 (CIFAR-10 configuration):
    - Ncol = 40 total columns
    - Nshared = 4 always-on shared columns
    - Nadaptive = 30 columns for selection
    - Nreserve = 6 reserve columns (for continual learning)

    Args:
        num_columns: Total number of columns (default: 40)
        num_tokens: Number of tokens per column (default: 16)
        input_dim: Input dimension (default: 96)
        output_dim: Output dimension (default: 96)
        microcolumn_dim: Microcolumn width (default: 32)
        combination: Combination mode (default: "sum")
        prefix: Name prefix for columns (default: "col")

    Returns:
        List of ColumnarNode instances
    """
    columns = []
    for i in range(num_columns):
        col = create_column(
            name=f"{prefix}_{i:02d}",
            num_tokens=num_tokens,
            input_dim=input_dim,
            output_dim=output_dim,
            microcolumn_dim=microcolumn_dim,
            combination=combination,
        )
        columns.append(col)
    return columns
