"""Tests for normalized column readout and classifier-edge ablations."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode

from columnar_cl_fabricpc.columns import PooledFeatureNormNode
from scripts.train_cifar10_depth_spanning import (
    build_depth_spanning_graph,
    mask_output_input_sources,
    output_input_edge_sources,
)


def test_pooled_feature_norm_fixed_params() -> None:
    """Fixed-gamma normalization registers no learnable parameters."""
    params = PooledFeatureNormNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(4,),
        input_shapes={"source->norm:in": (4,)},
        config={"fix_ln_gamma": True},
    )
    assert params.weights == {}
    assert params.biases == {}


def test_pooled_feature_norm_learnable_params() -> None:
    """Learnable normalization registers one scale and one shift vector."""
    params = PooledFeatureNormNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(4,),
        input_shapes={"source->norm:in": (4,)},
        config={"fix_ln_gamma": False},
    )
    assert params.weights["ln_gamma"].shape == (4,)
    assert params.biases["ln_beta"].shape == (4,)


def test_pooled_feature_norm_forward_normalizes_feature_axis() -> None:
    """The output has zero feature mean and unit feature variance per sample."""
    input_node = IdentityNode(shape=(4,), name="input")
    norm_node = PooledFeatureNormNode(
        shape=(4,),
        name="norm",
        fix_ln_gamma=True,
    )
    structure = graph(
        nodes=[input_node, norm_node],
        edges=[Edge(source=input_node, target=norm_node.slot("in"))],
        task_map=TaskMap(x=input_node),
        inference=InferenceSGD(),
    )

    x = jnp.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=jnp.float32,
    )
    state = NodeState(
        z_latent=jnp.zeros_like(x),
        z_mu=jnp.zeros_like(x),
        error=jnp.zeros_like(x),
        energy=jnp.zeros((x.shape[0],), dtype=jnp.float32),
        pre_activation=jnp.zeros_like(x),
        latent_grad=jnp.zeros_like(x),
    )
    params = PooledFeatureNormNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(4,),
        input_shapes={"input->norm:in": (4,)},
        config={"fix_ln_gamma": True},
    )

    _, state = PooledFeatureNormNode.forward(
        params,
        {"input->norm:in": x},
        state,
        structure.nodes["norm"].node_info,
    )

    assert jnp.allclose(jnp.mean(state.z_mu, axis=-1), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.var(state.z_mu, axis=-1), 1.0, atol=1e-4)


def _tiny_depth_spanning_args() -> SimpleNamespace:
    return SimpleNamespace(
        model="tiny",
        activation="leaky_relu",
        column_activation="leaky_relu",
        num_columns=2,
        num_shared=1,
        active_nonshared=1,
        column_mode="all_active",
        combiner="sum",
        embed_dim=16,
        microcolumn_dim=8,
        seed=7,
        bypass_columns=True,
        layer_norm_tokens=True,
        fix_ln_gamma=True,
        label_smoothing=0.0,
        infer_steps=2,
        eta_infer=0.1,
        infer_max_norm=1.0,
    )


def test_depth_spanning_graph_routes_normalized_readout_to_output() -> None:
    """The classifier receives `column_readout_norm`, not raw `column_pool`."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())
    output_sources = output_input_edge_sources(structure)

    assert "column_readout_norm" in output_sources
    assert "bypass_pool" in output_sources
    assert "column_pool" not in output_sources


def test_mask_output_input_sources_zeroes_only_dropped_edges() -> None:
    """Ablation masking zeroes classifier weights for dropped readout sources."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())
    params = initialize_params(structure, jax.random.PRNGKey(0))
    output_sources = output_input_edge_sources(structure)

    masked = mask_output_input_sources(
        params,
        structure,
        kept_sources=("column_readout_norm",),
    )

    column_edge = output_sources["column_readout_norm"]
    bypass_edge = output_sources["bypass_pool"]
    assert jnp.allclose(
        masked.nodes["output"].weights[column_edge],
        params.nodes["output"].weights[column_edge],
    )
    assert jnp.allclose(
        masked.nodes["output"].weights[bypass_edge],
        jnp.zeros_like(params.nodes["output"].weights[bypass_edge]),
    )

