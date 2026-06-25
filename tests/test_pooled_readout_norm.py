"""Tests for column readout nodes and classifier-edge ablations."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode

from columnar_cl_fabricpc.columns import (
    GlobalAvgPoolNormNode,
    WeightedLabelSmoothedCrossEntropyEnergy,
)
from scripts.train_cifar10_depth_spanning import (
    COLUMN_TEACHER_NODE,
    COLUMN_TEACHER_TARGET,
    ColumnTeacherTargetLoader,
    build_depth_spanning_graph,
    mask_output_input_sources,
    mask_output_source_feature_slice,
    output_input_edge_sources,
)


def test_global_avg_pool_norm_fixed_params() -> None:
    """Fixed-gamma normalization registers no learnable parameters."""
    params = GlobalAvgPoolNormNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(4,),
        input_shapes={"source->norm:in": (2, 4)},
        config={"fix_ln_gamma": True},
    )
    assert params.weights == {}
    assert params.biases == {}


def test_global_avg_pool_norm_learnable_params() -> None:
    """Learnable normalization registers one scale and one shift vector."""
    params = GlobalAvgPoolNormNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(4,),
        input_shapes={"source->norm:in": (2, 4)},
        config={"fix_ln_gamma": False},
    )
    assert params.weights["ln_gamma"].shape == (4,)
    assert params.biases["ln_beta"].shape == (4,)


def test_global_avg_pool_norm_forward_pools_and_normalizes() -> None:
    """The output averages tokens, then normalizes the feature axis."""
    input_node = IdentityNode(shape=(2, 4), name="input")
    norm_node = GlobalAvgPoolNormNode(
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
            [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
            [[2.0, 4.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    output_shape = (x.shape[0], 4)
    state = NodeState(
        z_latent=jnp.zeros(output_shape, dtype=jnp.float32),
        z_mu=jnp.zeros(output_shape, dtype=jnp.float32),
        error=jnp.zeros(output_shape, dtype=jnp.float32),
        energy=jnp.zeros((x.shape[0],), dtype=jnp.float32),
        pre_activation=jnp.zeros(output_shape, dtype=jnp.float32),
        latent_grad=jnp.zeros(output_shape, dtype=jnp.float32),
    )
    params = GlobalAvgPoolNormNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(4,),
        input_shapes={"input->norm:in": (2, 4)},
        config={"fix_ln_gamma": True},
    )

    _, state = GlobalAvgPoolNormNode.forward(
        params,
        {"input->norm:in": x},
        state,
        structure.nodes["norm"].node_info,
    )

    pooled = jnp.mean(x, axis=1)
    assert state.z_mu.shape == output_shape
    assert jnp.allclose(jnp.mean(state.z_mu, axis=-1), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.var(state.z_mu, axis=-1), 1.0, atol=1e-4)
    assert not jnp.allclose(state.z_mu, pooled)


def test_weighted_cross_entropy_scales_energy_and_latent_gradient() -> None:
    """The auxiliary teacher weight scales its class energy and gradient."""
    target = jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    prediction = jnp.asarray([[0.8, 0.1, 0.1]], dtype=jnp.float32)
    base = WeightedLabelSmoothedCrossEntropyEnergy(weight=1.0, smoothing=0.0)
    weighted = WeightedLabelSmoothedCrossEntropyEnergy(weight=0.25, smoothing=0.0)

    base_energy = type(base).energy(target, prediction, base.config)
    weighted_energy = type(weighted).energy(target, prediction, weighted.config)
    base_grad = type(base).grad_latent(target, prediction, base.config)
    weighted_grad = type(weighted).grad_latent(target, prediction, weighted.config)

    assert jnp.allclose(weighted_energy, 0.25 * base_energy)
    assert jnp.allclose(weighted_grad, 0.25 * base_grad)


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
        column_teacher_weight=0.1,
        infer_steps=2,
        eta_infer=0.1,
        infer_max_norm=1.0,
    )


def test_depth_spanning_graph_routes_raw_column_pool_to_output() -> None:
    """The classifier receives the raw pooled column readout."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())
    output_sources = output_input_edge_sources(structure)

    assert "column_pool" in output_sources
    assert "bypass_pool" in output_sources
    assert "column_readout_norm" not in output_sources
    assert "column_readout_norm" not in structure.nodes


def test_depth_spanning_graph_adds_column_teacher_head() -> None:
    """The column teacher head receives class error from the pooled columns only."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())

    assert structure.task_map[COLUMN_TEACHER_TARGET] == COLUMN_TEACHER_NODE
    assert structure.nodes[COLUMN_TEACHER_NODE].node_info.energy.config["weight"] == 0.1
    teacher_sources = {
        edge.source
        for edge in structure.edges.values()
        if edge.target == COLUMN_TEACHER_NODE and edge.slot == "in"
    }

    assert teacher_sources == {"column_pool"}


def test_column_teacher_target_loader_duplicates_labels() -> None:
    """The training wrapper adds column_y without changing x or y."""
    x = jnp.ones((2, 4, 4, 3), dtype=jnp.float32)
    y = jnp.eye(10, dtype=jnp.float32)[:2]
    wrapped = ColumnTeacherTargetLoader([(x, y)])

    batch = next(iter(wrapped))

    assert set(batch) == {"x", "y", COLUMN_TEACHER_TARGET}
    assert jnp.allclose(batch["x"], x)
    assert jnp.allclose(batch["y"], y)
    assert jnp.allclose(batch[COLUMN_TEACHER_TARGET], y)


def test_mask_output_input_sources_zeroes_only_dropped_edges() -> None:
    """Ablation masking zeroes classifier weights for dropped readout sources."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())
    params = initialize_params(structure, jax.random.PRNGKey(0))
    output_sources = output_input_edge_sources(structure)

    masked = mask_output_input_sources(
        params,
        structure,
        kept_sources=("column_pool",),
    )

    column_edge = output_sources["column_pool"]
    bypass_edge = output_sources["bypass_pool"]
    assert jnp.allclose(
        masked.nodes["output"].weights[column_edge],
        params.nodes["output"].weights[column_edge],
    )
    assert jnp.allclose(
        masked.nodes["output"].weights[bypass_edge],
        jnp.zeros_like(params.nodes["output"].weights[bypass_edge]),
    )


def test_mask_output_source_feature_slice_zeroes_selected_features() -> None:
    """Shell lesions zero selected feature rows on the column readout edge."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())
    params = initialize_params(structure, jax.random.PRNGKey(0))
    output_sources = output_input_edge_sources(structure)
    column_edge = output_sources["column_pool"]

    masked = mask_output_source_feature_slice(
        params,
        structure,
        source="column_pool",
        feature_slice=(2, 5),
        keep_slice=False,
    )

    before = params.nodes["output"].weights[column_edge]
    after = masked.nodes["output"].weights[column_edge]
    assert jnp.allclose(after[:2], before[:2])
    assert jnp.allclose(after[2:5], jnp.zeros_like(before[2:5]))
    assert jnp.allclose(after[5:], before[5:])
