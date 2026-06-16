"""
Column combiner for HiBaCaML/ColBa architecture.

From Section 4.3:
> A small attention-style composer combines the outputs of the five active columns
> into the per-token representation that is then read out by the task-local head.

Architecture::

    Active Columns (e.g., 5 selected from pool of 40)
    ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
    │Col 0 │  │Col 3 │  │Col 7 │  │Col 12│  │Col 25│
    │(B,T,D)│  │(B,T,D)│  │(B,T,D)│  │(B,T,D)│  │(B,T,D)│
    └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
       │         │         │         │         │
       └─────────┴────┬────┴─────────┴─────────┘
                      │
                      ▼
              ┌───────────────┐
              │   Combiner    │
              │               │
              │  Options:     │
              │  - sum        │  Simple sum of column outputs
              │  - attention  │  Learned attention over columns
              │  - concat     │  Concatenate + project
              └───────┬───────┘
                      │
                      ▼
              Output (B, T, D)

    Where:
    - B = batch size
    - T = number of tokens (e.g., 16 for CIFAR-10)
    - D = embedding dimension (e.g., 96)

Classification Head::

              Combined Output (B, T, D)
                      │
                      ▼
              ┌───────────────┐
              │  Pool Tokens  │  Mean over token dimension
              └───────┬───────┘
                      │
                      ▼
              (B, D)
                      │
                      ▼
              ┌───────────────┐
              │ Classification │  Linear projection
              │     Head      │  D → num_classes
              └───────┬───────┘
                      │
                      ▼
              Logits (B, num_classes)
"""

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.activations import ActivationBase, IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.initializers import (
    InitializerBase,
    KaimingInitializer,
    NormalInitializer,
    initialize,
)
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec


class ColumnCombinerNode(NodeBase):
    """
    Combines outputs from multiple columns into a unified representation.

    Supports multiple combination strategies:
    - "sum": Simple element-wise sum of column outputs
    - "attention": Learned attention weights over columns
    - "concat": Concatenate all columns and project back

    Input: Multiple column outputs, each (batch, num_tokens, embed_dim)
    Output: Combined (batch, num_tokens, output_dim)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        num_columns: int,
        input_dim: int,
        combination: str = "attention",
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize column combiner.

        Args:
            shape: Output shape (num_tokens, output_dim)
            name: Node name
            num_columns: Number of columns to combine
            input_dim: Dimension of each column's output
            combination: Combination strategy ("sum", "attention", "concat")
            activation: Activation function
            energy: Energy functional
            weight_init: Weight initializer
            latent_init: Latent state initializer
        """
        self.num_columns = num_columns
        self.input_dim = input_dim
        self.combination = combination

        if len(shape) != 2:
            raise ValueError(
                f"ColumnCombinerNode shape must be (num_tokens, output_dim), got {shape}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_columns=num_columns,
            input_dim=input_dim,
            combination=combination,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Multi-input slot accepting all column outputs."""
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
        Initialize combiner parameters.

        For attention: learned column attention weights
        For concat: projection matrix from concat_dim to output_dim
        """
        if config is None:
            config = {}

        if weight_init is None:
            weight_init = KaimingInitializer()

        num_tokens, output_dim = node_shape
        num_columns = config.get("num_columns", len(input_shapes))
        input_dim = config.get("input_dim", output_dim)
        combination = config.get("combination", "attention")

        weights = {}
        biases = {}

        key1, key2 = jax.random.split(key)

        if combination == "attention":
            # Learned attention weights over columns
            # Initialize uniformly
            weights["col_attention"] = jnp.ones((num_columns,)) / num_columns

        elif combination == "concat":
            # Projection from concatenated columns to output_dim
            concat_dim = num_columns * input_dim
            weights["W_combine"] = initialize(
                key1, (concat_dim, output_dim), weight_init
            )
            biases["b_combine"] = jnp.zeros((output_dim,))

        # For sum, no additional parameters needed

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Combine column outputs.

        Steps:
        1. Gather all column outputs
        2. Apply combination strategy
        3. Compute error and energy
        """
        batch_size = state.z_latent.shape[0]
        config = node_info.node_config
        combination = config.get("combination", "attention")

        # Collect all inputs into a list
        # Sort by edge key for consistent ordering
        sorted_inputs = [inputs[k] for k in sorted(inputs.keys())]

        if combination == "sum":
            pre_activation = sum(sorted_inputs)

        elif combination == "attention":
            # Softmax attention over columns
            attn = jax.nn.softmax(params.weights["col_attention"])
            pre_activation = sum(
                attn[i] * sorted_inputs[i] for i in range(len(sorted_inputs))
            )

        elif combination == "concat":
            # Concatenate along feature dimension
            concat = jnp.concatenate(sorted_inputs, axis=-1)
            W = params.weights["W_combine"]
            b = params.biases["b_combine"]
            pre_activation = jnp.matmul(concat, W) + b

        else:
            raise ValueError(f"Unknown combination mode: {combination}")

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


class ClassificationHeadNode(NodeBase):
    """
    Classification head that pools tokens and projects to class logits.

    Architecture:
    1. Pool over token dimension (mean pooling)
    2. Linear projection to num_classes
    3. Optional softmax activation

    Input: (batch, num_tokens, embed_dim)
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        input_dim: int,
        num_tokens: int,
        pooling: str = "mean",
        activation: Optional[ActivationBase] = SoftmaxActivation(),
        energy: Optional[EnergyFunctional] = CrossEntropyEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        """
        Initialize classification head.

        Args:
            shape: Output shape (num_classes,)
            name: Node name
            input_dim: Embedding dimension from combiner
            num_tokens: Number of tokens to pool over
            pooling: Pooling strategy ("mean", "max", "cls")
            activation: Activation (default: Softmax for classification)
            energy: Energy functional (default: CrossEntropy)
            weight_init: Weight initializer
            latent_init: Latent state initializer
        """
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.pooling = pooling

        if len(shape) != 1:
            raise ValueError(
                f"ClassificationHeadNode shape must be (num_classes,), got {shape}"
            )

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            input_dim=input_dim,
            num_tokens=num_tokens,
            pooling=pooling,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialize classification projection weights."""
        if config is None:
            config = {}

        if weight_init is None:
            weight_init = KaimingInitializer()

        (num_classes,) = node_shape
        input_dim = config.get("input_dim", 96)

        key1, key2 = jax.random.split(key)

        weights = {
            "W_cls": initialize(key1, (input_dim, num_classes), weight_init),
        }
        biases = {
            "b_cls": jnp.zeros((num_classes,)),
        }

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: pool tokens, project to classes.

        Steps:
        1. Combine inputs (if multiple)
        2. Pool over token dimension
        3. Project to class logits
        4. Apply activation (softmax)
        5. Compute error and energy
        """
        batch_size = state.z_latent.shape[0]
        config = node_info.node_config
        pooling = config.get("pooling", "mean")

        # Combine inputs
        x = None
        for edge_key, inp in inputs.items():
            if x is None:
                x = inp
            else:
                x = x + inp

        # x shape: (batch, num_tokens, embed_dim)
        # Pool over token dimension
        if pooling == "mean":
            pooled = jnp.mean(x, axis=1)  # (batch, embed_dim)
        elif pooling == "max":
            pooled = jnp.max(x, axis=1)  # (batch, embed_dim)
        elif pooling == "cls":
            pooled = x[:, 0, :]  # Use first token as CLS
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        # Project to classes
        W = params.weights["W_cls"]
        b = params.biases["b_cls"]
        logits = jnp.matmul(pooled, W) + b

        # Apply activation (softmax)
        activation = node_info.activation
        z_mu = type(activation).forward(logits, activation.config)

        # Error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=logits, z_mu=z_mu, error=error)

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


def create_combiner(
    name: str = "combiner",
    num_columns: int = 5,
    num_tokens: int = 16,
    embed_dim: int = 96,
    combination: str = "attention",
) -> ColumnCombinerNode:
    """
    Create a column combiner for CIFAR-10.

    Args:
        name: Node name
        num_columns: Number of active columns to combine
        num_tokens: Number of tokens per column
        embed_dim: Embedding dimension
        combination: Combination strategy

    Returns:
        Configured ColumnCombinerNode
    """
    return ColumnCombinerNode(
        shape=(num_tokens, embed_dim),
        name=name,
        num_columns=num_columns,
        input_dim=embed_dim,
        combination=combination,
    )


def create_classification_head(
    name: str = "classifier",
    num_classes: int = 10,
    embed_dim: int = 96,
    num_tokens: int = 16,
    pooling: str = "mean",
) -> ClassificationHeadNode:
    """
    Create a classification head for CIFAR-10.

    Args:
        name: Node name
        num_classes: Number of output classes (10 for CIFAR-10)
        embed_dim: Input embedding dimension
        num_tokens: Number of tokens to pool
        pooling: Pooling strategy

    Returns:
        Configured ClassificationHeadNode
    """
    return ClassificationHeadNode(
        shape=(num_classes,),
        name=name,
        input_dim=embed_dim,
        num_tokens=num_tokens,
        pooling=pooling,
    )
