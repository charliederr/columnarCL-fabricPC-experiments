"""
Accuracy-oriented ColBa nodes for plain CIFAR-10 predictive-coding experiments.

These nodes implement the first local slice of the HiBaCaML plan without
changing FabricPC. They are intentionally focused on plain CIFAR-10: one
classifier, one task, and columnar specialization before any split-task logic.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
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


def _hidden_activation(x: jax.Array, activation_name: str, leaky_alpha: float) -> jax.Array:
    if activation_name == "relu":
        return jax.nn.relu(x)
    if activation_name == "gelu":
        return jax.nn.gelu(x)
    if activation_name == "leaky_relu":
        return jax.nn.leaky_relu(x, negative_slope=leaky_alpha)
    if activation_name == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unknown hidden activation: {activation_name}")


class FeatureTokenizerNode(NodeBase):
    """
    Convert a convolutional feature map into a token sequence.

    Input: `(batch, height, width, channels)`.
    Output: `(batch, num_tokens, embed_dim)`.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        embed_dim: int,
        num_tokens: int,
        add_pos_embed: bool = True,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        if len(shape) != 2:
            raise ValueError(
                f"FeatureTokenizerNode shape must be (num_tokens, embed_dim), got {shape}"
            )
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            add_pos_embed=add_pos_embed,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        return source_shape[-1]

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = KaimingInitializer()

        if len(input_shapes) != 1:
            raise ValueError("FeatureTokenizerNode expects exactly one input edge")

        num_tokens, embed_dim = node_shape
        in_shape = next(iter(input_shapes.values()))
        if len(in_shape) != 3:
            raise ValueError(
                "FeatureTokenizerNode expects input shape (height, width, channels)"
            )
        spatial_count = in_shape[0] * in_shape[1]
        if spatial_count % num_tokens != 0:
            raise ValueError(
                f"num_tokens={num_tokens} must divide spatial_count={spatial_count}"
            )

        keys = jax.random.split(key, 3)
        in_channels = in_shape[-1]
        weights = {
            "W_token": initialize(keys[0], (in_channels, embed_dim), weight_init),
        }
        biases = {
            "b_token": initialize(keys[1], (embed_dim,), ZerosInitializer()),
        }
        if config.get("add_pos_embed", True):
            weights["pos_embed"] = initialize(
                keys[2],
                (1, num_tokens, embed_dim),
                NormalInitializer(std=0.02),
            )
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        num_tokens = config.get("num_tokens")

        x = None
        for inp in inputs.values():
            x = inp if x is None else x + inp

        batch_size, height, width, channels = x.shape
        spatial_count = height * width
        x = x.reshape(batch_size, spatial_count, channels)
        if num_tokens < spatial_count:
            pool_size = spatial_count // num_tokens
            x = x.reshape(batch_size, num_tokens, pool_size, channels)
            x = jnp.mean(x, axis=2)

        pre_activation = jnp.matmul(x, params.weights["W_token"]) + params.biases[
            "b_token"
        ]
        if "pos_embed" in params.weights:
            pre_activation = pre_activation + params.weights["pos_embed"]

        z_mu = node_info.activation.forward(pre_activation, node_info.activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class TypedColBaColumnNode(NodeBase):
    """
    Column with operational K, L, and B microcolumn pathways.

    K is a per-token residual pathway. L mixes local token neighborhoods on the
    token grid. B pools global context and broadcasts it back to each token.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        input_dim: int,
        microcolumn_dim: int = 32,
        grid_size: Tuple[int, int] = (4, 4),
        hidden_activation: str = "leaky_relu",
        leaky_alpha: float = 0.1,
        residual: bool = True,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        if len(shape) != 2:
            raise ValueError(
                f"TypedColBaColumnNode shape must be (num_tokens, output_dim), got {shape}"
            )
        if shape[0] != grid_size[0] * grid_size[1]:
            raise ValueError(
                f"shape token count {shape[0]} must equal grid_size product {grid_size}"
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
            grid_size=grid_size,
            hidden_activation=hidden_activation,
            leaky_alpha=leaky_alpha,
            residual=residual,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        return source_shape[-1]

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = KaimingInitializer()

        input_dim = config.get("input_dim")
        microcolumn_dim = config.get("microcolumn_dim", 32)
        output_dim = node_shape[-1]
        keys = jax.random.split(key, 12)

        weights = {
            "K_W_in": initialize(keys[0], (input_dim, microcolumn_dim), weight_init),
            "K_W_out": initialize(keys[1], (microcolumn_dim, output_dim), weight_init),
            "L_W_local": initialize(
                keys[2], (3, 3, input_dim, microcolumn_dim), weight_init
            ),
            "L_W_out": initialize(keys[3], (microcolumn_dim, output_dim), weight_init),
            "B_W_in": initialize(keys[4], (input_dim, microcolumn_dim), weight_init),
            "B_W_out": initialize(keys[5], (microcolumn_dim, output_dim), weight_init),
            "path_scale": jnp.ones((3,), dtype=jnp.float32) / jnp.sqrt(3.0),
        }
        biases = {
            "K_b_in": initialize(keys[6], (microcolumn_dim,), ZerosInitializer()),
            "K_b_out": initialize(keys[7], (output_dim,), ZerosInitializer()),
            "L_b_in": initialize(keys[8], (1, 1, 1, microcolumn_dim), ZerosInitializer()),
            "L_b_out": initialize(keys[9], (output_dim,), ZerosInitializer()),
            "B_b_in": initialize(keys[10], (microcolumn_dim,), ZerosInitializer()),
            "B_b_out": initialize(keys[11], (output_dim,), ZerosInitializer()),
        }
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        grid_h, grid_w = config.get("grid_size", (4, 4))
        hidden_activation = config.get("hidden_activation", "leaky_relu")
        leaky_alpha = config.get("leaky_alpha", 0.1)
        residual = config.get("residual", True)

        x = None
        for inp in inputs.values():
            x = inp if x is None else x + inp

        # K: per-token stable transform.
        k_hidden = jnp.matmul(x, params.weights["K_W_in"]) + params.biases["K_b_in"]
        k_hidden = _hidden_activation(k_hidden, hidden_activation, leaky_alpha)
        k_out = jnp.matmul(k_hidden, params.weights["K_W_out"]) + params.biases[
            "K_b_out"
        ]

        # L: local token-neighborhood mixing over the token grid.
        batch_size, num_tokens, input_dim = x.shape
        x_grid = x.reshape(batch_size, grid_h, grid_w, input_dim)
        l_hidden = jax.lax.conv_general_dilated(
            lhs=x_grid,
            rhs=params.weights["L_W_local"],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        ) + params.biases["L_b_in"]
        l_hidden = _hidden_activation(l_hidden, hidden_activation, leaky_alpha)
        l_hidden = l_hidden.reshape(batch_size, num_tokens, -1)
        l_out = jnp.matmul(l_hidden, params.weights["L_W_out"]) + params.biases[
            "L_b_out"
        ]

        # B: global context pooled across tokens and broadcast back.
        b_context = jnp.mean(x, axis=1)
        b_hidden = jnp.matmul(b_context, params.weights["B_W_in"]) + params.biases[
            "B_b_in"
        ]
        b_hidden = _hidden_activation(b_hidden, hidden_activation, leaky_alpha)
        b_out = jnp.matmul(b_hidden, params.weights["B_W_out"]) + params.biases[
            "B_b_out"
        ]
        b_out = jnp.broadcast_to(b_out[:, None, :], k_out.shape)

        path_scale = params.weights["path_scale"]
        pre_activation = (
            path_scale[0] * k_out + path_scale[1] * l_out + path_scale[2] * b_out
        )
        if residual and x.shape[-1] == pre_activation.shape[-1]:
            pre_activation = pre_activation + x

        z_mu = node_info.activation.forward(pre_activation, node_info.activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class MaskedColumnCombinerNode(NodeBase):
    """
    Combine a static pool of columns through a fixed support mask.

    The topology keeps all column edges in the graph. The mask controls which
    columns contribute to the combiner output.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        num_columns: int,
        support_mask: Tuple[float, ...],
        combination: str = "attention",
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        if len(shape) != 2:
            raise ValueError(
                f"MaskedColumnCombinerNode shape must be (num_tokens, output_dim), got {shape}"
            )
        if len(support_mask) != num_columns:
            raise ValueError("support_mask length must equal num_columns")
        if sum(support_mask) <= 0:
            raise ValueError("support_mask must activate at least one column")
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_columns=num_columns,
            support_mask=tuple(float(x) for x in support_mask),
            combination=combination,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        active_count = max(1.0, float(sum(config.get("support_mask", (1.0,)))))
        return int(source_shape[-1] * active_count)

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}
        num_columns = config.get("num_columns", len(input_shapes))
        return NodeParams(
            weights={"col_attention": jnp.zeros((num_columns,), dtype=jnp.float32)},
            biases={},
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        combination = config.get("combination", "attention")
        sorted_inputs = [inputs[k] for k in sorted(inputs.keys())]
        mask = jnp.asarray(config.get("support_mask"), dtype=sorted_inputs[0].dtype)
        active_count = jnp.maximum(jnp.sum(mask), 1.0)

        if combination == "sum":
            pre_activation = sum(mask[i] * sorted_inputs[i] for i in range(len(sorted_inputs)))
            pre_activation = pre_activation / jnp.sqrt(active_count)
        elif combination == "attention":
            logits = params.weights["col_attention"]
            logits = jnp.where(mask > 0.0, logits, -1.0e9)
            attn = jax.nn.softmax(logits)
            pre_activation = sum(
                attn[i] * sorted_inputs[i] for i in range(len(sorted_inputs))
            )
        else:
            raise ValueError(f"Unknown combiner mode: {combination}")

        z_mu = node_info.activation.forward(pre_activation, node_info.activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class GlobalAvgPoolNormNode(NodeBase):
    """
    Globally average a token or spatial feature tensor and normalize the result.

    This node replaces a separate average-pooling node followed by a separate
    feature-normalization node. It gives the classifier a normalized readout
    while adding only one predictive-coding latent on the readout path.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        fix_ln_gamma: bool = False,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        if len(shape) != 1:
            raise ValueError(
                f"GlobalAvgPoolNormNode shape must be (feature_dim,), got {shape}"
            )
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,
            fix_ln_gamma=fix_ln_gamma,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        return source_shape[-1]

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}

        feature_dim = node_shape[-1]
        if config.get("fix_ln_gamma", False):
            return NodeParams(weights={}, biases={})

        return NodeParams(
            weights={"ln_gamma": jnp.ones((feature_dim,), dtype=jnp.float32)},
            biases={"ln_beta": jnp.zeros((feature_dim,), dtype=jnp.float32)},
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = None
        for inp in inputs.values():
            x = inp if x is None else x + inp

        spatial_axes = tuple(range(1, x.ndim - 1))
        if spatial_axes:
            x = jnp.mean(x, axis=spatial_axes)

        if node_info.node_config.get("fix_ln_gamma", False):
            gamma = jnp.float32(1.0)
            beta = jnp.float32(0.0)
        else:
            gamma = params.weights["ln_gamma"]
            beta = params.biases["ln_beta"]

        pre_activation = layernorm(x, gamma, beta)
        z_mu = node_info.activation.forward(pre_activation, node_info.activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


def create_global_avg_pool_norm(
    name: str,
    feature_dim: int,
    fix_ln_gamma: bool = False,
) -> GlobalAvgPoolNormNode:
    """Create a global-average-pool and feature-normalization node."""
    return GlobalAvgPoolNormNode(
        shape=(feature_dim,),
        name=name,
        fix_ln_gamma=fix_ln_gamma,
    )


class FeatureSliceNode(NodeBase):
    """
    Expose a contiguous feature-axis slice as its own predictive-coding node.

    Input shape is `(feature_dim,)` or another tensor with a final feature axis.
    Output shape is the same input shape with the final axis replaced by
    `end - start`. The node has no learnable parameters.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        start: int,
        end: int,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init: Optional[InitializerBase] = NormalInitializer(std=0.02),
    ):
        if start < 0 or end <= start:
            raise ValueError(f"Invalid feature slice start={start}, end={end}")
        if shape[-1] != end - start:
            raise ValueError(
                f"FeatureSliceNode shape last axis must equal end-start, got "
                f"shape={shape}, start={start}, end={end}"
            )
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,
            start=start,
            end=end,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if len(input_shapes) != 1:
            raise ValueError("FeatureSliceNode expects exactly one input edge")
        config = config or {}
        start = int(config["start"])
        end = int(config["end"])
        in_shape = next(iter(input_shapes.values()))
        if end > in_shape[-1]:
            raise ValueError(
                f"Feature slice end={end} exceeds input feature width {in_shape[-1]}"
            )
        expected_shape = tuple(in_shape[:-1]) + (end - start,)
        if tuple(node_shape) != expected_shape:
            raise ValueError(
                f"FeatureSliceNode node_shape={node_shape} does not match "
                f"expected shape {expected_shape}"
            )
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = next(iter(inputs.values()))
        start = int(node_info.node_config["start"])
        end = int(node_info.node_config["end"])
        pre_activation = x[..., start:end]
        z_mu = node_info.activation.forward(pre_activation, node_info.activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
