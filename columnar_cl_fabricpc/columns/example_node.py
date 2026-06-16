"""
Example column node demonstrating the FabricPC extension contract.

This module provides a minimal NodeBase subclass for smoke testing. It is
not intended for real experiments — it exists to verify that external custom
nodes work correctly with FabricPC's graph machinery.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import InitializerBase, NormalInitializer


class ExampleColumnNode(NodeBase):
    """
    A minimal column node for smoke testing the FabricPC extension contract.

    This node sums all inputs and applies a learned scale and bias:
        output = scale * sum(inputs) + bias

    It follows the six-step forward contract from NodeBase.forward.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(std=0.01),
        weight_init=NormalInitializer(std=0.01),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single multi-input slot accepting arbitrary number of edges."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Initialize scale (scalar) and bias (vector).

        No per-edge weights since this node sums inputs directly.
        """
        out_numel = int(np.prod(node_shape))

        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Scalar scale initialized near 1.0
        scale = jax.random.normal(subkey1, ()) * 0.1 + 1.0

        # Bias vector initialized near zero
        bias = jax.random.normal(subkey2, (out_numel,)) * 0.01

        weights = {"_scale": scale.reshape((1, 1))}
        biases = {"_bias": bias}

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass following the six-step contract.

        1. Compute z_mu (prediction)
        2. Record pre_activation
        3. Compute error = z_latent - z_mu
        4. Update state fields
        5. Populate energy via energy_functional
        6. Return (total_energy, state)
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        out_numel = int(np.prod(out_shape))

        # Sum all inputs (flatten each to match output size)
        pre_activation_flat = jnp.zeros((batch_size, out_numel))
        for edge_key, x in inputs.items():
            x_flat = x.reshape(batch_size, -1)
            # Match output size by truncating or padding
            if x_flat.shape[1] >= out_numel:
                x_flat = x_flat[:, :out_numel]
            else:
                x_flat = jnp.pad(
                    x_flat, ((0, 0), (0, out_numel - x_flat.shape[1]))
                )
            pre_activation_flat = pre_activation_flat + x_flat

        # Apply scale and bias
        scale = params.weights["_scale"].squeeze()
        bias = params.biases["_bias"]
        pre_activation_flat = scale * pre_activation_flat + bias

        # Reshape to output shape
        pre_activation = pre_activation_flat.reshape((batch_size,) + out_shape)

        # Step 1: z_mu via activation
        activation = node_info.activation
        z_mu = activation.forward(pre_activation, activation.config)

        # Step 2: pre_activation recorded above
        # Step 3: error
        error = state.z_latent - z_mu

        # Step 4: update state
        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
        )

        # Step 5: energy
        state = ExampleColumnNode.energy_functional(state, node_info)

        # Step 6: return
        return jnp.sum(state.energy), state
