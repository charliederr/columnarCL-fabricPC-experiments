"""Tests for column readout nodes and classifier-edge ablations."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode

from columnar_cl_fabricpc.columns import (
    FeatureSliceNode,
    GlobalAvgPoolNormNode,
    SHELL_NAMES,
    WeightedLabelSmoothedCrossEntropyEnergy,
)
from scripts.train_cifar10_depth_spanning import (
    COLUMN_TEACHER_NODE,
    COLUMN_TEACHER_TARGET,
    ColumnTeacherTargetLoader,
    build_depth_spanning_graph,
    column_shell_pool_node_name,
    column_shell_slice_node_name,
    column_shell_teacher_node_name,
    column_shell_teacher_target_name,
    mask_output_input_sources,
    mask_output_source_feature_slice,
    output_input_edge_sources,
    parse_shell_teacher_weights,
    shell_slice_node_name,
    shell_teacher_node_name,
    shell_teacher_target_name,
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


def test_feature_slice_node_exposes_contiguous_feature_axis() -> None:
    """FeatureSliceNode predicts a configured final-axis feature slice."""
    input_node = IdentityNode(shape=(8,), name="input")
    slice_node = FeatureSliceNode(
        shape=(3,),
        name="slice",
        start=2,
        end=5,
    )
    structure = graph(
        nodes=[input_node, slice_node],
        edges=[Edge(source=input_node, target=slice_node.slot("in"))],
        task_map=TaskMap(x=input_node),
        inference=InferenceSGD(),
    )

    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    state = NodeState(
        z_latent=jnp.zeros((2, 3), dtype=jnp.float32),
        z_mu=jnp.zeros((2, 3), dtype=jnp.float32),
        error=jnp.zeros((2, 3), dtype=jnp.float32),
        energy=jnp.zeros((2,), dtype=jnp.float32),
        pre_activation=jnp.zeros((2, 3), dtype=jnp.float32),
        latent_grad=jnp.zeros((2, 3), dtype=jnp.float32),
    )
    params = FeatureSliceNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(3,),
        input_shapes={"input->slice:in": (8,)},
        config={"start": 2, "end": 5},
    )

    _, state = FeatureSliceNode.forward(
        params,
        {"input->slice:in": x},
        state,
        structure.nodes["slice"].node_info,
    )

    assert jnp.allclose(state.z_mu, x[:, 2:5])


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


def _tiny_depth_spanning_args(**overrides) -> SimpleNamespace:
    defaults = dict(
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
        shell_teacher_weights="0,0,0,0",
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        infer_steps=2,
        eta_infer=0.1,
        infer_max_norm=1.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_parse_shell_teacher_weights_maps_values_in_shell_order() -> None:
    """Shell teacher weights are parsed in the architecture's shell order."""
    weights = parse_shell_teacher_weights("0.005,0.005,0.01,0.02")

    assert [weights[shell_name] for shell_name in SHELL_NAMES] == [
        0.005,
        0.005,
        0.01,
        0.02,
    ]
    with pytest.raises(ValueError):
        parse_shell_teacher_weights("0.1,0.2")
    with pytest.raises(ValueError):
        parse_shell_teacher_weights("0.1,-0.2,0.3,0.4")


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


def test_depth_spanning_graph_adds_shell_local_teacher_heads() -> None:
    """Shell-local teacher heads receive only their pooled feature slice."""
    shell_weight_values = [0.005, 0.005, 0.01, 0.01]
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        shell_teacher_weights="0.005,0.005,0.01,0.01",
    )
    structure, _ = build_depth_spanning_graph(args)

    for shell_name, expected_weight in zip(SHELL_NAMES, shell_weight_values):
        slice_name = shell_slice_node_name(shell_name)
        teacher_name = shell_teacher_node_name(shell_name)
        target_name = shell_teacher_target_name(shell_name)

        assert structure.task_map[target_name] == teacher_name
        assert (
            structure.nodes[teacher_name].node_info.energy.config["weight"]
            == expected_weight
        )

        slice_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == slice_name and edge.slot == "in"
        }
        teacher_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == teacher_name and edge.slot == "in"
        }

        assert slice_sources == {"column_pool"}
        assert teacher_sources == {slice_name}


def test_depth_spanning_graph_adds_per_column_shell_teacher_heads() -> None:
    """Per-column shell heads attach to active column shells before combining."""
    shell_weight_values = [0.001, 0.001, 0.002, 0.002]
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0.001,0.001,0.002,0.002",
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        for shell_name, expected_weight in zip(SHELL_NAMES, shell_weight_values):
            slice_name = column_shell_slice_node_name(column_idx, shell_name)
            pool_name = column_shell_pool_node_name(column_idx, shell_name)
            teacher_name = column_shell_teacher_node_name(column_idx, shell_name)
            target_name = column_shell_teacher_target_name(column_idx, shell_name)

            assert structure.task_map[target_name] == teacher_name
            assert (
                structure.nodes[teacher_name].node_info.energy.config["weight"]
                == expected_weight
            )

            slice_sources = {
                edge.source
                for edge in structure.edges.values()
                if edge.target == slice_name and edge.slot == "in"
            }
            pool_sources = {
                edge.source
                for edge in structure.edges.values()
                if edge.target == pool_name and edge.slot == "in"
            }
            teacher_sources = {
                edge.source
                for edge in structure.edges.values()
                if edge.target == teacher_name and edge.slot == "in"
            }

            assert slice_sources == {f"col_{column_idx:02d}"}
            assert pool_sources == {slice_name}
            assert teacher_sources == {pool_name}


def test_per_column_shell_teacher_heads_skip_inactive_columns() -> None:
    """Per-column shell supervision follows the column support mask."""
    args = _tiny_depth_spanning_args(
        num_columns=3,
        num_shared=1,
        active_nonshared=1,
        column_mode="first_sparse",
        column_shell_teacher_weights="0.001,0,0,0",
    )
    structure, support_mask = build_depth_spanning_graph(args)

    assert support_mask == (1.0, 1.0, 0.0)
    assert column_shell_teacher_node_name(0, "hard_kernel") in structure.nodes
    assert column_shell_teacher_node_name(1, "hard_kernel") in structure.nodes
    assert column_shell_teacher_node_name(2, "hard_kernel") not in structure.nodes


def test_depth_spanning_graph_adds_per_column_shell_readout_edges() -> None:
    """Per-column shell readout reaches output without requiring teacher heads."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        for shell_name in SHELL_NAMES:
            slice_name = column_shell_slice_node_name(column_idx, shell_name)
            pool_name = column_shell_pool_node_name(column_idx, shell_name)
            teacher_name = column_shell_teacher_node_name(column_idx, shell_name)
            target_name = column_shell_teacher_target_name(column_idx, shell_name)

            assert pool_name in output_sources
            assert teacher_name not in structure.nodes
            assert target_name not in structure.task_map

            slice_sources = {
                edge.source
                for edge in structure.edges.values()
                if edge.target == slice_name and edge.slot == "in"
            }
            pool_sources = {
                edge.source
                for edge in structure.edges.values()
                if edge.target == pool_name and edge.slot == "in"
            }

            assert slice_sources == {f"col_{column_idx:02d}"}
            assert pool_sources == {slice_name}


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


def test_column_teacher_target_loader_duplicates_shell_labels() -> None:
    """The training wrapper adds labels for configured shell-local targets."""
    x = jnp.ones((2, 4, 4, 3), dtype=jnp.float32)
    y = jnp.eye(10, dtype=jnp.float32)[:2]
    shell_targets = ("hard_kernel_y", "outer_shell_y")
    wrapped = ColumnTeacherTargetLoader(
        [(x, y)],
        shell_target_keys=shell_targets,
    )

    batch = next(iter(wrapped))

    assert set(batch) == {"x", "y", COLUMN_TEACHER_TARGET, *shell_targets}
    assert jnp.allclose(batch["x"], x)
    assert jnp.allclose(batch["y"], y)
    for target_name in shell_targets:
        assert jnp.allclose(batch[target_name], y)


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
