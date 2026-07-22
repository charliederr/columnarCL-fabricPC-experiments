"""Tests for column readout nodes and classifier-edge ablations."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeParams, NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode

from columnar_cl_fabricpc.columns import (
    ColumnShellComposerNode,
    FeatureSliceNode,
    GlobalAvgPoolNormNode,
    MeanSquaredGaussianEnergy,
    SHELL_NAMES,
    ShellContextPredictionNode,
    SpatialReferenceGaussianEnergy,
    StageTapTokenizer,
    WeightedLabelSmoothedCrossEntropyEnergy,
    get_shell_slices,
)
from scripts.train_cifar10_depth_spanning import (
    COLUMN_TEACHER_NODE,
    COLUMN_TEACHER_TARGET,
    INWARD_SHELL_PROMOTION_PAIRS,
    OUTER_SHELL_CONTEXT_EVIDENCE_NODE,
    OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE,
    OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_TARGET,
    OUTER_SHELL_CONTEXT_TEACHER_NODE,
    OUTER_SHELL_CONTEXT_TEACHER_TARGET,
    ColumnTeacherTargetLoader,
    active_composer_components,
    apply_shell_lr_multipliers,
    build_shell_lr_multiplier_tree,
    build_depth_spanning_graph,
    build_readout_ablation_cases,
    column_shell_bridge_node_name,
    column_shell_pool_node_name,
    column_shell_slice_node_name,
    column_shell_teacher_node_name,
    column_shell_teacher_target_name,
    diagnose_composer_attention,
    diagnose_composer_projection_norms,
    diagnose_output_edge_weight_norms,
    has_shell_composer,
    inward_shell_promotion_node_name,
    inward_shell_promotion_schedule_is_active,
    inward_shell_promotion_weight_for_epoch,
    mask_column_shell_bridge_inputs,
    mask_column_shell_path_inputs,
    mask_composer_components,
    mask_outer_shell_context_inputs,
    mask_output_input_sources,
    mask_node_input_sources,
    mask_output_source_feature_slice,
    node_input_edge_sources,
    outer_shell_context_node_name,
    outer_shell_context_bridge_scale_node_name,
    output_input_edge_sources,
    parse_inward_shell_promotion_pairs,
    parse_shell_lr_multipliers,
    parse_shell_teacher_weights,
    parse_shell_evidence_cascade_scale,
    parse_shell_inhibition_strengths,
    parse_support_mask,
    set_inward_shell_promotion_objective_weight,
    shell_context_prediction_node_name,
    shell_slice_node_name,
    shell_teacher_node_name,
    shell_teacher_target_name,
    validate_inward_shell_promotion_schedule,
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


def test_stage_tap_tokenizer_resizes_smaller_grid_to_target() -> None:
    """Stage taps can map a deeper smaller stage onto an earlier token grid."""
    x = 7.0 * jnp.ones((2, 4, 4, 3), dtype=jnp.float32)

    resized = StageTapTokenizer._adaptive_avg_pool_2d(x, 8, 8)

    assert resized.shape == (2, 8, 8, 3)
    assert jnp.allclose(resized, 7.0)


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


def test_column_shell_composer_masks_inactive_column_components() -> None:
    """The shell composer ignores shell components from inactive columns."""
    col_0 = IdentityNode(shape=(2, 8), name="col_00")
    col_1 = IdentityNode(shape=(2, 8), name="col_01")
    composer = ColumnShellComposerNode(
        shape=(2, 8),
        name="composer",
        num_columns=2,
        support_mask=(1.0, 0.0),
    )
    structure = graph(
        nodes=[col_0, col_1, composer],
        edges=[
            Edge(source=col_0, target=composer.slot("in")),
            Edge(source=col_1, target=composer.slot("in")),
        ],
        task_map=TaskMap(x=col_0),
        inference=InferenceSGD(),
    )
    params = ColumnShellComposerNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(2, 8),
        input_shapes={
            "col_00->composer:in": (2, 8),
            "col_01->composer:in": (2, 8),
        },
        config={
            "num_columns": 2,
            "support_mask": (1.0, 0.0),
        },
    )
    state = NodeState(
        z_latent=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        z_mu=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        error=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        energy=jnp.zeros((1,), dtype=jnp.float32),
        pre_activation=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        latent_grad=jnp.zeros((1, 2, 8), dtype=jnp.float32),
    )

    _, state = ColumnShellComposerNode.forward(
        params,
        {
            "col_00->composer:in": jnp.zeros((1, 2, 8), dtype=jnp.float32),
            "col_01->composer:in": 100.0
            * jnp.ones((1, 2, 8), dtype=jnp.float32),
        },
        state,
        structure.nodes["composer"].node_info,
    )

    assert params.weights["component_attention"].shape == (2, len(SHELL_NAMES))
    assert jnp.allclose(state.z_mu, jnp.zeros((1, 2, 8), dtype=jnp.float32))


def test_column_shell_composer_projects_only_within_matching_shell() -> None:
    """A composer input shell cannot write into another output shell."""
    col_0 = IdentityNode(shape=(2, 8), name="col_00")
    composer = ColumnShellComposerNode(
        shape=(2, 8),
        name="composer",
        num_columns=1,
        support_mask=(1.0,),
    )
    structure = graph(
        nodes=[col_0, composer],
        edges=[Edge(source=col_0, target=composer.slot("in"))],
        task_map=TaskMap(x=col_0),
        inference=InferenceSGD(),
    )
    params = ColumnShellComposerNode.initialize_params(
        jax.random.PRNGKey(0),
        node_shape=(2, 8),
        input_shapes={"col_00->composer:in": (2, 8)},
        config={
            "num_columns": 1,
            "support_mask": (1.0,),
        },
    )
    weights = {name: jnp.zeros_like(value) for name, value in params.weights.items()}
    biases = {name: jnp.zeros_like(value) for name, value in params.biases.items()}
    shell_slices = get_shell_slices(8)
    hard_start, hard_end = shell_slices["hard_kernel"]
    hard_width = hard_end - hard_start
    weights[
        ColumnShellComposerNode._projection_weight_name(0, "hard_kernel")
    ] = jnp.ones((hard_width, hard_width), dtype=jnp.float32)
    params = params._replace(weights=weights, biases=biases)
    state = NodeState(
        z_latent=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        z_mu=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        error=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        energy=jnp.zeros((1,), dtype=jnp.float32),
        pre_activation=jnp.zeros((1, 2, 8), dtype=jnp.float32),
        latent_grad=jnp.zeros((1, 2, 8), dtype=jnp.float32),
    )
    x = jnp.zeros((1, 2, 8), dtype=jnp.float32)
    x = x.at[..., hard_start:hard_end].set(1.0)

    _, state = ColumnShellComposerNode.forward(
        params,
        {"col_00->composer:in": x},
        state,
        structure.nodes["composer"].node_info,
    )

    assert jnp.all(state.z_mu[..., hard_start:hard_end] > 0.0)
    assert jnp.allclose(state.z_mu[..., :hard_start], 0.0)
    assert jnp.allclose(state.z_mu[..., hard_end:], 0.0)


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


def test_mean_squared_gaussian_energy_normalizes_by_latent_size() -> None:
    """Normalized Gaussian energy preserves PC errors while scaling by shape."""
    z_latent = jnp.ones((2, 3, 4), dtype=jnp.float32)
    z_mu = jnp.zeros((2, 3, 4), dtype=jnp.float32)
    energy = MeanSquaredGaussianEnergy(precision=2.0)

    per_sample_energy = MeanSquaredGaussianEnergy.energy(
        z_latent,
        z_mu,
        energy.config,
    )
    latent_grad = MeanSquaredGaussianEnergy.grad_latent(
        z_latent,
        z_mu,
        energy.config,
    )

    assert jnp.allclose(per_sample_energy, jnp.ones((2,), dtype=jnp.float32))
    assert jnp.allclose(latent_grad, jnp.ones_like(z_latent) / 6.0)


def test_spatial_reference_gaussian_energy_scales_by_token_sites() -> None:
    """Spatial-reference Gaussian energy preserves feature precision per site."""
    token_latent = jnp.ones((2, 64, 3), dtype=jnp.float32)
    token_mu = jnp.zeros((2, 64, 3), dtype=jnp.float32)
    pooled_latent = jnp.ones((2, 3), dtype=jnp.float32)
    pooled_mu = jnp.zeros((2, 3), dtype=jnp.float32)
    energy = SpatialReferenceGaussianEnergy(precision=2.0, reference_sites=16.0)

    token_energy = SpatialReferenceGaussianEnergy.energy(
        token_latent,
        token_mu,
        energy.config,
    )
    token_grad = SpatialReferenceGaussianEnergy.grad_latent(
        token_latent,
        token_mu,
        energy.config,
    )
    pooled_energy = SpatialReferenceGaussianEnergy.energy(
        pooled_latent,
        pooled_mu,
        energy.config,
    )
    pooled_grad = SpatialReferenceGaussianEnergy.grad_latent(
        pooled_latent,
        pooled_mu,
        energy.config,
    )

    assert jnp.allclose(token_energy, 48.0 * jnp.ones((2,), dtype=jnp.float32))
    assert jnp.allclose(token_grad, 0.5 * jnp.ones_like(token_latent))
    assert jnp.allclose(pooled_energy, 3.0 * jnp.ones((2,), dtype=jnp.float32))
    assert jnp.allclose(pooled_grad, 2.0 * jnp.ones_like(pooled_latent))


def test_shell_context_prediction_node_keeps_terminal_local_energy() -> None:
    """The shell-context objective sends gradients to target and context inputs."""
    target = IdentityNode(shape=(2,), name="target")
    context = IdentityNode(shape=(2,), name="context")
    prediction = ShellContextPredictionNode(
        shape=(2,),
        name="prediction",
        objective_weight=0.5,
        target_gradient_scale=1.0,
    )
    structure = graph(
        nodes=[target, context, prediction],
        edges=[
            Edge(source=target, target=prediction.slot("target")),
            Edge(source=context, target=prediction.slot("context")),
        ],
        task_map=TaskMap(x=target),
        inference=InferenceSGD(),
    )
    target_edge = next(
        edge_key
        for edge_key, edge in structure.edges.items()
        if edge.target == "prediction" and edge.slot == "target"
    )
    context_edge = next(
        edge_key
        for edge_key, edge in structure.edges.items()
        if edge.target == "prediction" and edge.slot == "context"
    )
    node_info = structure.nodes["prediction"].node_info
    params = NodeParams(
        weights={context_edge: jnp.eye(2, dtype=jnp.float32)},
        biases={"b": jnp.zeros((2,), dtype=jnp.float32)},
    )
    state = NodeState(
        z_latent=jnp.zeros((1, 2), dtype=jnp.float32),
        z_mu=jnp.zeros((1, 2), dtype=jnp.float32),
        error=jnp.zeros((1, 2), dtype=jnp.float32),
        energy=jnp.zeros((1,), dtype=jnp.float32),
        pre_activation=jnp.zeros((1, 2), dtype=jnp.float32),
        latent_grad=jnp.zeros((1, 2), dtype=jnp.float32),
    )
    inputs = {
        target_edge: jnp.asarray([[1.0, -1.0]], dtype=jnp.float32),
        context_edge: jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
    }

    new_state, input_grads, self_grad = (
        ShellContextPredictionNode.forward_and_latent_grads(
            params,
            inputs,
            state,
            node_info,
            is_clamped=False,
        )
    )

    assert float(new_state.energy[0]) > 0.0
    assert not jnp.allclose(input_grads[target_edge], 0.0)
    assert not jnp.allclose(input_grads[context_edge], 0.0)
    assert jnp.allclose(self_grad, 0.0)


def test_shell_context_prediction_node_can_anchor_target_gradient() -> None:
    """Target anchoring removes target-slot gradients and keeps context gradients."""
    target = IdentityNode(shape=(2,), name="target")
    context = IdentityNode(shape=(2,), name="context")
    prediction = ShellContextPredictionNode(
        shape=(2,),
        name="prediction",
        objective_weight=0.5,
        target_gradient_scale=0.0,
    )
    structure = graph(
        nodes=[target, context, prediction],
        edges=[
            Edge(source=target, target=prediction.slot("target")),
            Edge(source=context, target=prediction.slot("context")),
        ],
        task_map=TaskMap(x=target),
        inference=InferenceSGD(),
    )
    target_edge = next(
        edge_key
        for edge_key, edge in structure.edges.items()
        if edge.target == "prediction" and edge.slot == "target"
    )
    context_edge = next(
        edge_key
        for edge_key, edge in structure.edges.items()
        if edge.target == "prediction" and edge.slot == "context"
    )
    node_info = structure.nodes["prediction"].node_info
    params = NodeParams(
        weights={context_edge: jnp.eye(2, dtype=jnp.float32)},
        biases={"b": jnp.zeros((2,), dtype=jnp.float32)},
    )
    state = NodeState(
        z_latent=jnp.zeros((1, 2), dtype=jnp.float32),
        z_mu=jnp.zeros((1, 2), dtype=jnp.float32),
        error=jnp.zeros((1, 2), dtype=jnp.float32),
        energy=jnp.zeros((1,), dtype=jnp.float32),
        pre_activation=jnp.zeros((1, 2), dtype=jnp.float32),
        latent_grad=jnp.zeros((1, 2), dtype=jnp.float32),
    )
    inputs = {
        target_edge: jnp.asarray([[1.0, -1.0]], dtype=jnp.float32),
        context_edge: jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
    }

    new_state, input_grads, self_grad = (
        ShellContextPredictionNode.forward_and_latent_grads(
            params,
            inputs,
            state,
            node_info,
            is_clamped=False,
        )
    )

    assert float(new_state.energy[0]) > 0.0
    assert jnp.allclose(input_grads[target_edge], 0.0)
    assert not jnp.allclose(input_grads[context_edge], 0.0)
    assert jnp.allclose(self_grad, 0.0)


def _tiny_depth_spanning_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        model="tiny",
        activation="leaky_relu",
        column_activation="leaky_relu",
        num_columns=2,
        num_shared=1,
        active_nonshared=1,
        column_mode="all_active",
        support_mask=None,
        combiner="sum",
        column_grid="stage4",
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
        column_shell_bridge=False,
        outer_shell_context=False,
        outer_shell_context_to_bridge=False,
        outer_shell_context_bridge_scale=0.0,
        outer_shell_context_shell_prediction_weight=0.0,
        inward_shell_promotion_weight=0.0,
        inward_shell_promotion_pairs="all",
        inward_shell_promotion_target_gradient_scale=1.0,
        inward_shell_promotion_warmup_epochs=0.0,
        inward_shell_promotion_ramp_epochs=0.0,
        outer_shell_context_evidence=False,
        outer_shell_context_evidence_teacher_weight=0.0,
        outer_shell_context_teacher_weight=0.0,
        column_gaussian_energy_mode="sum",
        column_gaussian_precision=1.0,
        column_gaussian_reference_sites=16.0,
        shell_lr_multipliers="1,1,1,1",
        shell_evidence_cascade_scale="0.05,0.05,0.05",
        shell_inhibition_strengths="0,0.35,0.22,0.10",
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


def test_parse_shell_lr_multipliers_maps_values_in_shell_order() -> None:
    """Shell learning-rate multipliers are parsed in shell order."""
    multipliers = parse_shell_lr_multipliers("1,1.5,2,3")

    assert [multipliers[shell_name] for shell_name in SHELL_NAMES] == [
        1.0,
        1.5,
        2.0,
        3.0,
    ]
    with pytest.raises(ValueError):
        parse_shell_lr_multipliers("1,2")
    with pytest.raises(ValueError):
        parse_shell_lr_multipliers("1,-2,3,4")


def test_parse_support_mask_accepts_binary_column_mask() -> None:
    """Explicit support masks are parsed as one binary value per column."""
    assert parse_support_mask("1,0,1,0", 4) == (1.0, 0.0, 1.0, 0.0)
    assert parse_support_mask(None, 4) is None
    assert parse_support_mask("", 4) is None
    assert parse_support_mask("none", 4) is None

    with pytest.raises(ValueError, match="must contain 4"):
        parse_support_mask("1,0,1", 4)
    with pytest.raises(ValueError, match="binary"):
        parse_support_mask("1,0.5,1,0", 4)
    with pytest.raises(ValueError, match="at least one"):
        parse_support_mask("0,0,0,0", 4)


def test_parse_inward_shell_promotion_pairs_selects_adjacent_pairs() -> None:
    """Inward promotion pair labels select named adjacent shell transitions."""
    assert parse_inward_shell_promotion_pairs("all") == INWARD_SHELL_PROMOTION_PAIRS
    assert parse_inward_shell_promotion_pairs("none") == ()
    assert parse_inward_shell_promotion_pairs(
        "outer_to_middle,middle_to_inner"
    ) == (
        ("outer_shell", "middle_shell"),
        ("middle_shell", "inner_shell"),
    )

    with pytest.raises(ValueError, match="duplicate pair"):
        parse_inward_shell_promotion_pairs("outer_to_middle,outer_to_middle")
    with pytest.raises(ValueError, match="must be 'all', 'none'"):
        parse_inward_shell_promotion_pairs("outer_to_hard")


def test_parse_shell_inhibition_strengths_maps_values_in_shell_order() -> None:
    """Shell inhibition strengths are parsed in the architecture's shell order."""
    strengths = parse_shell_inhibition_strengths("0,0.35,0.22,0.10")

    assert [strengths[shell_name] for shell_name in SHELL_NAMES] == [
        0.0,
        0.35,
        0.22,
        0.10,
    ]
    with pytest.raises(ValueError):
        parse_shell_inhibition_strengths("0.1,0.2")
    with pytest.raises(ValueError):
        parse_shell_inhibition_strengths("0.1,-0.2,0.3,0.4")


def test_parse_shell_evidence_cascade_scale_maps_adjacent_pairs() -> None:
    """Evidence-cascade scale has one value per adjacent shell pair."""
    assert parse_shell_evidence_cascade_scale("0.05,0.025,0.01") == (
        0.05,
        0.025,
        0.01,
    )
    with pytest.raises(ValueError):
        parse_shell_evidence_cascade_scale("0.1,0.2")
    with pytest.raises(ValueError):
        parse_shell_evidence_cascade_scale("0.1,-0.2,0.3")


def test_depth_spanning_graph_routes_raw_column_pool_to_output() -> None:
    """The classifier receives the raw pooled column readout."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())
    output_sources = output_input_edge_sources(structure)

    assert "column_pool" in output_sources
    assert "bypass_pool" in output_sources
    assert "column_readout_norm" not in output_sources
    assert "column_readout_norm" not in structure.nodes


def test_depth_spanning_graph_defaults_to_summed_gaussian_energy() -> None:
    """Historical graph construction keeps FabricPC's summed Gaussian energy."""
    structure, _ = build_depth_spanning_graph(_tiny_depth_spanning_args())

    assert type(structure.nodes["stage2_tap"].node_info.energy) is GaussianEnergy
    assert type(structure.nodes["col_00"].node_info.energy) is GaussianEnergy
    assert type(structure.nodes["combiner"].node_info.energy) is GaussianEnergy
    assert type(structure.nodes["column_pool"].node_info.energy) is GaussianEnergy


def test_depth_spanning_graph_can_use_mean_column_gaussian_energy() -> None:
    """Columnar local Gaussian nodes can use latent-size-normalized precision."""
    args = _tiny_depth_spanning_args(
        combiner="shell_attention",
        bypass_columns=False,
        column_shell_bridge=True,
        outer_shell_context=True,
        column_gaussian_energy_mode="mean",
        column_gaussian_precision=0.5,
    )
    structure, _ = build_depth_spanning_graph(args)
    normalized_nodes = [
        "stage2_tap",
        "stage3_tap",
        "stage4_tap",
        "stage4_pool",
        "col_00",
        "combiner",
        "column_pool",
        column_shell_slice_node_name(0, "hard_kernel"),
        column_shell_pool_node_name(0, "hard_kernel"),
        column_shell_bridge_node_name(0),
        outer_shell_context_node_name(0),
    ]

    for node_name in normalized_nodes:
        energy = structure.nodes[node_name].node_info.energy
        assert type(energy) is MeanSquaredGaussianEnergy
        assert energy.config["precision"] == 0.5


def test_depth_spanning_graph_can_use_spatial_reference_gaussian_energy() -> None:
    """Columnar local Gaussian nodes can scale precision by token-site count."""
    args = _tiny_depth_spanning_args(
        combiner="shell_attention",
        bypass_columns=False,
        column_shell_bridge=True,
        outer_shell_context=True,
        column_gaussian_energy_mode="spatial_reference",
        column_gaussian_precision=0.75,
        column_gaussian_reference_sites=16.0,
    )
    structure, _ = build_depth_spanning_graph(args)
    spatial_nodes = [
        "stage2_tap",
        "stage3_tap",
        "stage4_tap",
        "stage4_pool",
        "col_00",
        "combiner",
        "column_pool",
        column_shell_slice_node_name(0, "hard_kernel"),
        column_shell_pool_node_name(0, "hard_kernel"),
        column_shell_bridge_node_name(0),
        outer_shell_context_node_name(0),
    ]

    for node_name in spatial_nodes:
        energy = structure.nodes[node_name].node_info.energy
        assert type(energy) is SpatialReferenceGaussianEnergy
        assert energy.config["precision"] == 0.75
        assert energy.config["reference_sites"] == 16.0


def test_depth_spanning_graph_uses_shell_composer_combiner_mode() -> None:
    """The shell_attention combiner preserves column-shell identity in combiner."""
    args = _tiny_depth_spanning_args(
        combiner="shell_attention",
        bypass_columns=False,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    combiner_sources = {
        edge.source
        for edge in structure.edges.values()
        if edge.target == "combiner" and edge.slot == "in"
    }
    output_sources = output_input_edge_sources(structure)

    assert support_mask == (1.0, 1.0)
    assert structure.nodes["combiner"].node_info.node_class is ColumnShellComposerNode
    assert combiner_sources == {"col_00", "col_01"}
    assert set(output_sources) == {"column_pool"}
    assert "component_attention" in params.nodes["combiner"].weights
    for shell_name, (start, end) in get_shell_slices(args.embed_dim).items():
        weight_name = ColumnShellComposerNode._projection_weight_name(0, shell_name)
        bias_name = ColumnShellComposerNode._projection_bias_name(0, shell_name)
        shell_width = end - start
        assert params.nodes["combiner"].weights[weight_name].shape == (
            shell_width,
            shell_width,
        )
        assert params.nodes["combiner"].biases[bias_name].shape == (shell_width,)


def test_depth_spanning_graph_can_use_resnet18_stage3_column_grid() -> None:
    """ResNet-18 columns can run on the 8x8 stage3 grid with 96-wide features."""
    args = _tiny_depth_spanning_args(
        model="resnet18",
        combiner="shell_attention",
        bypass_columns=False,
        column_grid="stage3",
        embed_dim=96,
        microcolumn_dim=32,
        column_teacher_weight=0.0,
        column_shell_bridge=True,
        outer_shell_context=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)

    assert support_mask == (1.0, 1.0)
    assert structure.nodes["stage2_tap"].node_info.shape == (64, 96)
    assert structure.nodes["stage3_tap"].node_info.shape == (64, 96)
    assert structure.nodes["stage4_tap"].node_info.shape == (64, 96)
    assert structure.nodes["col_00"].node_info.shape == (64, 96)
    assert structure.nodes["combiner"].node_info.shape == (64, 96)
    assert structure.nodes["stage2_tap"].node_info.node_config["target_grid"] == (8, 8)
    assert structure.nodes["stage4_tap"].node_info.node_config["target_grid"] == (8, 8)

    shell_widths = [
        end - start for start, end in get_shell_slices(args.embed_dim).values()
    ]
    assert shell_widths == [33, 11, 21, 31]


def test_shell_composer_diagnostics_report_active_components() -> None:
    """Composer diagnostics expose attention and projection norms by component."""
    args = _tiny_depth_spanning_args(
        combiner="shell_attention",
        bypass_columns=False,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))

    components = active_composer_components(structure)
    attention = diagnose_composer_attention(params, structure)
    projection_norms = diagnose_composer_projection_norms(params, structure)

    assert has_shell_composer(structure)
    assert components == tuple(
        (column_idx, shell_name)
        for column_idx in (0, 1)
        for shell_name in SHELL_NAMES
    )
    assert set(attention) == {"col_00", "col_01"}
    for shell_name in SHELL_NAMES:
        assert sum(
            attention[column_name][shell_name]
            for column_name in attention
        ) == pytest.approx(1.0)
    assert set(projection_norms) == {
        f"col_{column_idx:02d}.{shell_name}"
        for column_idx in (0, 1)
        for shell_name in SHELL_NAMES
    }


def test_mask_composer_components_zeroes_selected_projection() -> None:
    """Composer lesions zero selected projection weights without changing graph."""
    args = _tiny_depth_spanning_args(
        combiner="shell_attention",
        bypass_columns=False,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    hard_weight = ColumnShellComposerNode._projection_weight_name(0, "hard_kernel")
    hard_bias = ColumnShellComposerNode._projection_bias_name(0, "hard_kernel")
    inner_weight = ColumnShellComposerNode._projection_weight_name(0, "inner_shell")

    masked = mask_composer_components(
        params,
        structure,
        dropped_components=((0, "hard_kernel"),),
    )

    assert jnp.allclose(
        masked.nodes["combiner"].weights[hard_weight],
        jnp.zeros_like(params.nodes["combiner"].weights[hard_weight]),
    )
    assert jnp.allclose(
        masked.nodes["combiner"].biases[hard_bias],
        jnp.zeros_like(params.nodes["combiner"].biases[hard_bias]),
    )
    assert jnp.allclose(
        masked.nodes["combiner"].weights[inner_weight],
        params.nodes["combiner"].weights[inner_weight],
    )


def test_shell_lr_multiplier_tree_scales_depth_column_shell_outputs() -> None:
    """Depth-column output and shell-dynamics parameters get shell update rates."""
    args = _tiny_depth_spanning_args(
        shell_lr_multipliers="1,1.5,2,3",
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    shell_lr_multipliers = parse_shell_lr_multipliers(args.shell_lr_multipliers)

    multiplier_tree = build_shell_lr_multiplier_tree(
        params,
        structure,
        shell_lr_multipliers,
    )
    column_multipliers = multiplier_tree.nodes["col_00"]
    shell_slices = get_shell_slices(args.embed_dim)

    for shell_name in SHELL_NAMES:
        start, end = shell_slices[shell_name]
        expected = shell_lr_multipliers[shell_name]
        assert jnp.allclose(
            column_multipliers.weights["K_W_out"][:, start:end],
            expected,
        )
        assert jnp.allclose(
            column_multipliers.biases["K_b_out"][start:end],
            expected,
        )
        assert jnp.allclose(
            column_multipliers.weights["shell_path_scale"][
                SHELL_NAMES.index(shell_name)
            ],
            expected,
        )

    assert jnp.allclose(
        column_multipliers.weights["shell_evidence_cascade_scale"],
        jnp.asarray([1.5, 2.0, 3.0], dtype=jnp.float32),
    )
    assert jnp.allclose(
        column_multipliers.weights[
            "shell_evidence_cascade_middle_shell_to_outer_shell"
        ],
        3.0,
    )
    assert jnp.allclose(
        column_multipliers.biases[
            "shell_evidence_cascade_b_inner_shell_to_middle_shell"
        ],
        2.0,
    )
    assert jnp.allclose(column_multipliers.weights["K_W_deep"], 1.0)


def test_shell_lr_multiplier_tree_scales_composer_and_context_paths() -> None:
    """Composer, context, and shell predictors receive shell update rates."""
    args = _tiny_depth_spanning_args(
        combiner="shell_attention",
        bypass_columns=False,
        outer_shell_context=True,
        outer_shell_context_shell_prediction_weight=0.001,
        shell_lr_multipliers="1,1.5,2,3",
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    shell_lr_multipliers = parse_shell_lr_multipliers(args.shell_lr_multipliers)

    multiplier_tree = build_shell_lr_multiplier_tree(
        params,
        structure,
        shell_lr_multipliers,
    )
    composer_multipliers = multiplier_tree.nodes["combiner"]
    expected_attention = jnp.asarray([1.0, 1.5, 2.0, 3.0], dtype=jnp.float32)
    assert jnp.allclose(
        composer_multipliers.weights["component_attention"],
        jnp.ones((args.num_columns, len(SHELL_NAMES))) * expected_attention,
    )

    outer_weight = ColumnShellComposerNode._projection_weight_name(0, "outer_shell")
    inner_bias = ColumnShellComposerNode._projection_bias_name(0, "inner_shell")
    assert jnp.allclose(composer_multipliers.weights[outer_weight], 3.0)
    assert jnp.allclose(composer_multipliers.biases[inner_bias], 1.5)

    context_name = outer_shell_context_node_name(0)
    context_multipliers = multiplier_tree.nodes[context_name]
    for value in context_multipliers.weights.values():
        assert jnp.allclose(value, 3.0)
    for value in context_multipliers.biases.values():
        assert jnp.allclose(value, 3.0)

    hard_prediction_name = shell_context_prediction_node_name(0, "hard_kernel")
    outer_prediction_name = shell_context_prediction_node_name(0, "outer_shell")
    hard_prediction_multipliers = multiplier_tree.nodes[hard_prediction_name]
    outer_prediction_multipliers = multiplier_tree.nodes[outer_prediction_name]
    for value in hard_prediction_multipliers.weights.values():
        assert jnp.allclose(value, 1.0)
    for value in hard_prediction_multipliers.biases.values():
        assert jnp.allclose(value, 1.0)
    for value in outer_prediction_multipliers.weights.values():
        assert jnp.allclose(value, 3.0)
    for value in outer_prediction_multipliers.biases.values():
        assert jnp.allclose(value, 3.0)


def test_shell_lr_multiplier_tree_scales_inward_promotion_by_target_shell() -> None:
    """Inward promotion predictors update with their target shell rate."""
    args = _tiny_depth_spanning_args(
        bypass_columns=False,
        inward_shell_promotion_weight=0.001,
        shell_lr_multipliers="1,1.5,2,3",
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))

    multiplier_tree = build_shell_lr_multiplier_tree(
        params,
        structure,
        parse_shell_lr_multipliers(args.shell_lr_multipliers),
    )

    for source_shell, target_shell in INWARD_SHELL_PROMOTION_PAIRS:
        node_name = inward_shell_promotion_node_name(0, source_shell, target_shell)
        expected = parse_shell_lr_multipliers(args.shell_lr_multipliers)[target_shell]
        promotion_multipliers = multiplier_tree.nodes[node_name]
        for value in promotion_multipliers.weights.values():
            assert jnp.allclose(value, expected)
        for value in promotion_multipliers.biases.values():
            assert jnp.allclose(value, expected)


def test_apply_shell_lr_multipliers_scales_update_tree() -> None:
    """The Optax transform helper multiplies updates with the prepared tree."""
    args = _tiny_depth_spanning_args(
        shell_lr_multipliers="1,1.5,2,3",
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    updates = jax.tree_util.tree_map(jnp.ones_like, params)
    multiplier_tree = build_shell_lr_multiplier_tree(
        params,
        structure,
        parse_shell_lr_multipliers(args.shell_lr_multipliers),
    )

    scaled_updates = apply_shell_lr_multipliers(updates, multiplier_tree)
    outer_start, outer_end = get_shell_slices(args.embed_dim)["outer_shell"]

    assert jnp.allclose(
        scaled_updates.nodes["col_00"].weights["K_W_out"][:, outer_start:outer_end],
        3.0,
    )
    assert jnp.allclose(
        scaled_updates.nodes["col_00"].weights["K_W_deep"],
        1.0,
    )


def test_output_edge_weight_norms_are_named_by_source() -> None:
    """Output edge norms use architectural source names rather than edge keys."""
    args = _tiny_depth_spanning_args(
        column_shell_readout=True,
        column_shell_bridge=True,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    norms = diagnose_output_edge_weight_norms(params, structure)

    assert "column_pool" in norms
    assert "bypass_pool" in norms
    assert column_shell_pool_node_name(0, "hard_kernel") in norms
    assert column_shell_bridge_node_name(0) in norms


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


def test_depth_spanning_graph_explicit_support_mask_overrides_column_mode() -> None:
    """A named support mask controls active per-column graph paths."""
    args = _tiny_depth_spanning_args(
        num_columns=3,
        num_shared=1,
        active_nonshared=1,
        column_mode="first_sparse",
        support_mask="1,0,1",
        column_teacher_weight=0.0,
        column_shell_bridge=True,
        outer_shell_context=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)

    assert support_mask == (1.0, 0.0, 1.0)
    assert column_shell_bridge_node_name(0) in structure.nodes
    assert column_shell_bridge_node_name(1) not in structure.nodes
    assert column_shell_bridge_node_name(2) in structure.nodes
    assert outer_shell_context_node_name(0) in structure.nodes
    assert outer_shell_context_node_name(1) not in structure.nodes
    assert outer_shell_context_node_name(2) in structure.nodes


def test_depth_spanning_graph_adds_inward_shell_promotion_objectives() -> None:
    """Inward promotion adds local shell-to-shell prediction nodes per active column."""
    args = _tiny_depth_spanning_args(
        bypass_columns=False,
        inward_shell_promotion_weight=0.001,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    output_sources = output_input_edge_sources(structure)

    assert active_columns == [0, 1]
    assert set(output_sources) == {"column_pool"}

    for column_idx in active_columns:
        for source_shell, target_shell in INWARD_SHELL_PROMOTION_PAIRS:
            node_name = inward_shell_promotion_node_name(
                column_idx,
                source_shell,
                target_shell,
            )
            source_pool = column_shell_pool_node_name(column_idx, source_shell)
            target_pool = column_shell_pool_node_name(column_idx, target_shell)
            edge_sources_by_slot = {
                edge.slot: edge.source
                for edge in structure.edges.values()
                if edge.target == node_name
            }

            assert node_name in structure.nodes
            assert structure.nodes[node_name].node_info.node_class is (
                ShellContextPredictionNode
            )
            assert (
                structure.nodes[node_name].node_info.node_config["objective_weight"]
                == 0.001
            )
            assert edge_sources_by_slot["context"] == source_pool
            assert edge_sources_by_slot["target"] == target_pool


def test_depth_spanning_graph_adds_selected_inward_shell_promotion_pairs() -> None:
    """Pair selection can leave hard-kernel promotion out of the graph."""
    selected_pairs = (
        ("outer_shell", "middle_shell"),
        ("middle_shell", "inner_shell"),
    )
    omitted_pair = ("inner_shell", "hard_kernel")
    args = _tiny_depth_spanning_args(
        bypass_columns=False,
        inward_shell_promotion_weight=0.001,
        inward_shell_promotion_pairs="outer_to_middle,middle_to_inner",
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        for source_shell, target_shell in selected_pairs:
            node_name = inward_shell_promotion_node_name(
                column_idx,
                source_shell,
                target_shell,
            )
            assert node_name in structure.nodes

        omitted_node = inward_shell_promotion_node_name(
            column_idx,
            *omitted_pair,
        )
        assert omitted_node not in structure.nodes


def test_depth_spanning_graph_sets_inward_promotion_target_gradient_scale() -> None:
    """Inward-promotion nodes carry the configured target-gradient scale."""
    args = _tiny_depth_spanning_args(
        bypass_columns=False,
        inward_shell_promotion_weight=0.001,
        inward_shell_promotion_pairs="outer_to_middle,middle_to_inner",
        inward_shell_promotion_target_gradient_scale=0.0,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    for column_idx in active_columns:
        for source_shell, target_shell in (
            ("outer_shell", "middle_shell"),
            ("middle_shell", "inner_shell"),
        ):
            node_name = inward_shell_promotion_node_name(
                column_idx,
                source_shell,
                target_shell,
            )
            assert (
                structure.nodes[node_name]
                .node_info
                .node_config["target_gradient_scale"]
                == 0.0
            )


def test_depth_spanning_graph_rejects_invalid_promotion_target_gradient_scale() -> None:
    """The inward-promotion target-gradient scale is an attenuation factor."""
    args = _tiny_depth_spanning_args(
        inward_shell_promotion_weight=0.001,
        inward_shell_promotion_target_gradient_scale=1.5,
    )

    with pytest.raises(ValueError, match="target_gradient_scale"):
        build_depth_spanning_graph(args)


def test_depth_spanning_graph_rejects_positive_promotion_with_no_pairs() -> None:
    """A positive promotion weight must select at least one pair."""
    args = _tiny_depth_spanning_args(
        inward_shell_promotion_weight=0.001,
        inward_shell_promotion_pairs="none",
    )

    with pytest.raises(ValueError, match="select at least one pair"):
        build_depth_spanning_graph(args)


def test_inward_shell_promotion_schedule_ramps_after_warmup() -> None:
    """Scheduled promotion is off during warmup and linear during ramp."""
    args = _tiny_depth_spanning_args(
        num_epochs=20,
        inward_shell_promotion_weight=0.000375,
        inward_shell_promotion_warmup_epochs=4,
        inward_shell_promotion_ramp_epochs=8,
    )

    validate_inward_shell_promotion_schedule(args)

    assert inward_shell_promotion_schedule_is_active(args)
    assert inward_shell_promotion_weight_for_epoch(args, 0) == 0.0
    assert inward_shell_promotion_weight_for_epoch(args, 3) == 0.0
    assert inward_shell_promotion_weight_for_epoch(args, 4) == pytest.approx(
        0.000375 / 8.0
    )
    assert inward_shell_promotion_weight_for_epoch(args, 11) == pytest.approx(
        0.000375
    )
    assert inward_shell_promotion_weight_for_epoch(args, 19) == pytest.approx(
        0.000375
    )


def test_inward_shell_promotion_schedule_rejects_zero_final_weight() -> None:
    """A scheduled promotion objective needs a positive final weight."""
    args = _tiny_depth_spanning_args(
        num_epochs=20,
        inward_shell_promotion_weight=0.0,
        inward_shell_promotion_warmup_epochs=4,
        inward_shell_promotion_ramp_epochs=8,
    )

    with pytest.raises(ValueError, match="positive"):
        validate_inward_shell_promotion_schedule(args)


def test_set_inward_shell_promotion_objective_weight_updates_only_config() -> None:
    """Scheduled promotion changes node weights without changing graph topology."""
    args = _tiny_depth_spanning_args(
        bypass_columns=False,
        inward_shell_promotion_weight=0.001,
        inward_shell_promotion_pairs="outer_to_middle,middle_to_inner",
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    updated = set_inward_shell_promotion_objective_weight(structure, 0.000375)

    assert updated.node_order == structure.node_order
    assert set(updated.nodes) == set(structure.nodes)
    assert set(updated.edges) == set(structure.edges)
    for column_idx in active_columns:
        for source_shell, target_shell in (
            ("outer_shell", "middle_shell"),
            ("middle_shell", "inner_shell"),
        ):
            node_name = inward_shell_promotion_node_name(
                column_idx,
                source_shell,
                target_shell,
            )
            assert (
                updated.nodes[node_name].node_info.node_config["objective_weight"]
                == 0.000375
            )


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


def test_depth_spanning_graph_adds_per_column_shell_bridge_edges() -> None:
    """Per-column shell bridge receives pooled shells and reaches output."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        bridge_name = column_shell_bridge_node_name(column_idx)
        expected_pool_names = {
            column_shell_pool_node_name(column_idx, shell_name)
            for shell_name in SHELL_NAMES
        }
        bridge_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == bridge_name and edge.slot == "in"
        }

        assert bridge_name in output_sources
        assert bridge_sources == expected_pool_names
        for pool_name in expected_pool_names:
            assert pool_name not in output_sources


def test_depth_spanning_graph_adds_outer_context_to_shell_bridge_edges() -> None:
    """Optional context-to-bridge edges condition each column's bridge latent."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=True,
        outer_shell_context=True,
        outer_shell_context_to_bridge=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        bridge_name = column_shell_bridge_node_name(column_idx)
        context_name = outer_shell_context_node_name(column_idx)
        expected_pool_names = {
            column_shell_pool_node_name(column_idx, shell_name)
            for shell_name in SHELL_NAMES
        }
        bridge_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == bridge_name and edge.slot == "in"
        }

        assert bridge_name in output_sources
        assert context_name in output_sources
        assert bridge_sources == expected_pool_names | {context_name}


def test_depth_spanning_graph_adds_scaled_outer_context_to_bridge_latents() -> None:
    """Scaled bridge conditioning inserts a fixed-scale context latent."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=True,
        outer_shell_context=True,
        outer_shell_context_bridge_scale=0.05,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        bridge_name = column_shell_bridge_node_name(column_idx)
        context_name = outer_shell_context_node_name(column_idx)
        scaled_name = outer_shell_context_bridge_scale_node_name(column_idx)
        expected_pool_names = {
            column_shell_pool_node_name(column_idx, shell_name)
            for shell_name in SHELL_NAMES
        }
        bridge_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == bridge_name and edge.slot == "in"
        }
        scaled_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == scaled_name and edge.slot == "in"
        }

        assert bridge_name in output_sources
        assert context_name in output_sources
        assert scaled_name not in output_sources
        assert bridge_sources == expected_pool_names | {scaled_name}
        assert scaled_sources == {context_name}
        assert structure.nodes[scaled_name].node_info.node_config["scale"] == 0.05


def test_depth_spanning_graph_adds_outer_shell_context_edges() -> None:
    """Outer-shell context receives pooled shells and reaches output by default."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=False,
        outer_shell_context=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    outer_start, outer_end = get_shell_slices(args.embed_dim)["outer_shell"]
    outer_width = outer_end - outer_start

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        context_name = outer_shell_context_node_name(column_idx)
        expected_pool_names = {
            column_shell_pool_node_name(column_idx, shell_name)
            for shell_name in SHELL_NAMES
        }
        context_sources = {
            edge.source
            for edge in structure.edges.values()
            if edge.target == context_name and edge.slot == "in"
        }

        assert context_name in output_sources
        assert structure.nodes[context_name].node_info.shape == (outer_width,)
        assert context_sources == expected_pool_names
        for pool_name in expected_pool_names:
            assert pool_name not in output_sources


def test_depth_spanning_graph_adds_shell_context_prediction_objectives() -> None:
    """Positive shell-prediction weight adds local objectives and keeps readout."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=False,
        outer_shell_context=True,
        outer_shell_context_shell_prediction_weight=0.001,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]

    assert active_columns == [0, 1]

    for column_idx in active_columns:
        context_name = outer_shell_context_node_name(column_idx)
        assert context_name in output_sources
        for shell_name in SHELL_NAMES:
            prediction_name = shell_context_prediction_node_name(
                column_idx,
                shell_name,
            )
            prediction_sources = {
                (edge.source, edge.slot)
                for edge in structure.edges.values()
                if edge.target == prediction_name
            }

            assert (
                structure.nodes[prediction_name].node_info.node_class
                is ShellContextPredictionNode
            )
            assert (
                structure.nodes[prediction_name].node_info.node_config[
                    "objective_weight"
                ]
                == 0.001
            )
            assert prediction_sources == {
                (context_name, "context"),
                (column_shell_pool_node_name(column_idx, shell_name), "target"),
            }


def test_depth_spanning_graph_adds_outer_shell_context_teacher_head() -> None:
    """The context teacher receives all context latents through one CE node."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=False,
        outer_shell_context=True,
        outer_shell_context_teacher_weight=0.001,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    expected_context_names = {
        outer_shell_context_node_name(column_idx) for column_idx in active_columns
    }
    teacher_sources = {
        edge.source
        for edge in structure.edges.values()
        if edge.target == OUTER_SHELL_CONTEXT_TEACHER_NODE and edge.slot == "in"
    }

    assert OUTER_SHELL_CONTEXT_TEACHER_TARGET in structure.task_map
    assert (
        structure.task_map[OUTER_SHELL_CONTEXT_TEACHER_TARGET]
        == OUTER_SHELL_CONTEXT_TEACHER_NODE
    )
    assert (
        structure.nodes[
            OUTER_SHELL_CONTEXT_TEACHER_NODE
        ].node_info.energy.config["weight"]
        == 0.001
    )
    assert teacher_sources == expected_context_names


def test_depth_spanning_graph_adds_outer_shell_context_evidence_path() -> None:
    """Context evidence receives context latents without reaching output."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=False,
        outer_shell_context=True,
        outer_shell_context_evidence=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    output_sources = output_input_edge_sources(structure)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    expected_context_names = {
        outer_shell_context_node_name(column_idx) for column_idx in active_columns
    }
    evidence_sources = {
        edge.source
        for edge in structure.edges.values()
        if edge.target == OUTER_SHELL_CONTEXT_EVIDENCE_NODE and edge.slot == "in"
    }

    assert OUTER_SHELL_CONTEXT_EVIDENCE_NODE not in output_sources
    assert structure.nodes[OUTER_SHELL_CONTEXT_EVIDENCE_NODE].node_info.shape == (10,)
    assert evidence_sources == expected_context_names
    assert OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_TARGET not in structure.task_map


def test_depth_spanning_graph_adds_outer_shell_context_evidence_teacher_head() -> None:
    """The evidence teacher receives only the class-shaped evidence latent."""
    args = _tiny_depth_spanning_args(
        column_teacher_weight=0.0,
        column_shell_teacher_weights="0,0,0,0",
        column_shell_readout=False,
        column_shell_bridge=False,
        outer_shell_context=True,
        outer_shell_context_evidence=True,
        outer_shell_context_evidence_teacher_weight=0.0005,
    )
    structure, _ = build_depth_spanning_graph(args)
    teacher_sources = {
        edge.source
        for edge in structure.edges.values()
        if (
            edge.target == OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE
            and edge.slot == "in"
        )
    }

    assert OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_TARGET in structure.task_map
    assert (
        structure.task_map[OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_TARGET]
        == OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE
    )
    assert (
        structure.nodes[
            OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE
        ].node_info.energy.config["weight"]
        == 0.0005
    )
    assert teacher_sources == {OUTER_SHELL_CONTEXT_EVIDENCE_NODE}


def test_outer_shell_context_teacher_requires_context_path() -> None:
    """A context-teacher weight without context nodes is invalid."""
    args = _tiny_depth_spanning_args(
        outer_shell_context=False,
        outer_shell_context_teacher_weight=0.001,
    )

    with pytest.raises(ValueError, match="requires --outer_shell_context"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_evidence_requires_context_path() -> None:
    """Context evidence is invalid without context latents."""
    args = _tiny_depth_spanning_args(
        outer_shell_context=False,
        outer_shell_context_evidence=True,
    )

    with pytest.raises(ValueError, match="requires --outer_shell_context"):
        build_depth_spanning_graph(args)


def test_shell_context_prediction_requires_context_path() -> None:
    """A local shell prediction objective is invalid without context latents."""
    args = _tiny_depth_spanning_args(
        outer_shell_context=False,
        outer_shell_context_shell_prediction_weight=0.001,
    )

    with pytest.raises(ValueError, match="requires --outer_shell_context"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_to_bridge_requires_context_path() -> None:
    """Bridge conditioning is invalid without an outer-context latent."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=True,
        outer_shell_context=False,
        outer_shell_context_to_bridge=True,
    )

    with pytest.raises(ValueError, match="requires --outer_shell_context"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_to_bridge_requires_shell_bridge() -> None:
    """Bridge conditioning is invalid without the bridge latent."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=False,
        outer_shell_context=True,
        outer_shell_context_to_bridge=True,
    )

    with pytest.raises(ValueError, match="requires --column_shell_bridge"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_bridge_scale_requires_context_path() -> None:
    """Scaled bridge conditioning is invalid without an outer-context latent."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=True,
        outer_shell_context=False,
        outer_shell_context_bridge_scale=0.05,
    )

    with pytest.raises(ValueError, match="requires --outer_shell_context"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_bridge_scale_requires_shell_bridge() -> None:
    """Scaled bridge conditioning is invalid without the bridge latent."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=False,
        outer_shell_context=True,
        outer_shell_context_bridge_scale=0.05,
    )

    with pytest.raises(ValueError, match="requires --column_shell_bridge"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_bridge_scale_rejects_full_strength_bridge() -> None:
    """Scaled and full-strength bridge conditioning are separate experiments."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=True,
        outer_shell_context=True,
        outer_shell_context_to_bridge=True,
        outer_shell_context_bridge_scale=0.05,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        build_depth_spanning_graph(args)


def test_outer_shell_context_evidence_teacher_requires_evidence_path() -> None:
    """An evidence-teacher weight is invalid without evidence latents."""
    args = _tiny_depth_spanning_args(
        outer_shell_context=True,
        outer_shell_context_evidence=False,
        outer_shell_context_evidence_teacher_weight=0.0005,
    )

    with pytest.raises(ValueError, match="requires --outer_shell_context_evidence"):
        build_depth_spanning_graph(args)


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
    shell_targets = (
        "hard_kernel_y",
        "outer_shell_y",
        OUTER_SHELL_CONTEXT_TEACHER_TARGET,
    )
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


def test_build_readout_ablation_cases_includes_family_lesions() -> None:
    """Readout ablation cases include bridge/context family interactions."""
    args = _tiny_depth_spanning_args(
        bypass_columns=False,
        column_shell_bridge=True,
        outer_shell_context=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    output_sources = output_input_edge_sources(structure)
    all_sources = set(output_sources)
    bridge_sources = {
        column_shell_bridge_node_name(column_idx) for column_idx in active_columns
    }
    context_sources = {
        outer_shell_context_node_name(column_idx) for column_idx in active_columns
    }
    cases = dict(build_readout_ablation_cases(structure))

    assert set(cases["combined_without_column_pool"]) == all_sources - {"column_pool"}
    assert set(cases["combined_without_column_shell_bridge"]) == (
        all_sources - bridge_sources
    )
    assert set(cases["combined_without_outer_shell_context"]) == (
        all_sources - context_sources
    )
    assert set(cases["outer_shell_context_plus_column_shell_bridge"]) == (
        context_sources | bridge_sources
    )
    assert set(cases["column_pool_plus_shell_bridge"]) == (
        {"column_pool"} | bridge_sources
    )
    assert set(cases["column_pool_plus_outer_shell_context"]) == (
        {"column_pool"} | context_sources
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


def test_mask_node_input_sources_zeroes_only_dropped_bridge_inputs() -> None:
    """Generic input masking works on the shell bridge, not only output."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=True,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    bridge_name = column_shell_bridge_node_name(0)
    bridge_sources = node_input_edge_sources(structure, bridge_name)
    kept_pool = column_shell_pool_node_name(0, "hard_kernel")

    masked = mask_node_input_sources(
        params,
        structure,
        bridge_name,
        kept_sources=(kept_pool,),
    )

    for source_name, edge_key in bridge_sources.items():
        before = params.nodes[bridge_name].weights[edge_key]
        after = masked.nodes[bridge_name].weights[edge_key]
        if source_name == kept_pool:
            assert jnp.allclose(after, before)
        else:
            assert jnp.allclose(after, jnp.zeros_like(before))


def test_mask_outer_shell_context_inputs_masks_selected_shell() -> None:
    """Outer-shell context input masking keeps only the requested shell inputs."""
    args = _tiny_depth_spanning_args(
        outer_shell_context=True,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    context_name = outer_shell_context_node_name(0)
    context_sources = node_input_edge_sources(structure, context_name)

    masked = mask_outer_shell_context_inputs(
        params,
        structure,
        (context_name,),
        "outer_shell",
        keep_shell=True,
    )

    for source_name, edge_key in context_sources.items():
        before = params.nodes[context_name].weights[edge_key]
        after = masked.nodes[context_name].weights[edge_key]
        if source_name == column_shell_pool_node_name(0, "outer_shell"):
            assert jnp.allclose(after, before)
        else:
                assert jnp.allclose(after, jnp.zeros_like(before))


def test_mask_column_shell_bridge_inputs_masks_context_conditioning() -> None:
    """Bridge shell masks also mask the context route feeding the bridge."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=True,
        outer_shell_context=True,
        outer_shell_context_to_bridge=True,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    bridge_name = column_shell_bridge_node_name(0)
    context_name = outer_shell_context_node_name(0)
    bridge_sources = node_input_edge_sources(structure, bridge_name)
    context_sources = node_input_edge_sources(structure, context_name)

    masked = mask_column_shell_bridge_inputs(
        params,
        structure,
        (bridge_name,),
        "outer_shell",
        keep_shell=False,
    )

    assert jnp.allclose(
        masked.nodes[bridge_name].weights[bridge_sources[context_name]],
        params.nodes[bridge_name].weights[bridge_sources[context_name]],
    )
    for shell_name in SHELL_NAMES:
        pool_name = column_shell_pool_node_name(0, shell_name)
        direct_bridge_edge = bridge_sources[pool_name]
        context_edge = context_sources[pool_name]
        if shell_name == "outer_shell":
            assert jnp.allclose(
                masked.nodes[bridge_name].weights[direct_bridge_edge],
                jnp.zeros_like(params.nodes[bridge_name].weights[direct_bridge_edge]),
            )
            assert jnp.allclose(
                masked.nodes[context_name].weights[context_edge],
                jnp.zeros_like(params.nodes[context_name].weights[context_edge]),
            )
        else:
            assert jnp.allclose(
                masked.nodes[bridge_name].weights[direct_bridge_edge],
                params.nodes[bridge_name].weights[direct_bridge_edge],
            )
            assert jnp.allclose(
                masked.nodes[context_name].weights[context_edge],
                params.nodes[context_name].weights[context_edge],
            )


def test_mask_column_shell_bridge_inputs_masks_scaled_context_conditioning() -> None:
    """Bridge shell masks trace through scaled context conditioning latents."""
    args = _tiny_depth_spanning_args(
        column_shell_bridge=True,
        outer_shell_context=True,
        outer_shell_context_bridge_scale=0.05,
    )
    structure, _ = build_depth_spanning_graph(args)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    bridge_name = column_shell_bridge_node_name(0)
    context_name = outer_shell_context_node_name(0)
    scaled_name = outer_shell_context_bridge_scale_node_name(0)
    bridge_sources = node_input_edge_sources(structure, bridge_name)
    context_sources = node_input_edge_sources(structure, context_name)

    masked = mask_column_shell_bridge_inputs(
        params,
        structure,
        (bridge_name,),
        "outer_shell",
        keep_shell=False,
    )

    assert scaled_name in bridge_sources
    assert context_name not in bridge_sources
    assert jnp.allclose(
        masked.nodes[bridge_name].weights[bridge_sources[scaled_name]],
        params.nodes[bridge_name].weights[bridge_sources[scaled_name]],
    )
    for shell_name in SHELL_NAMES:
        pool_name = column_shell_pool_node_name(0, shell_name)
        direct_bridge_edge = bridge_sources[pool_name]
        context_edge = context_sources[pool_name]
        if shell_name == "outer_shell":
            assert jnp.allclose(
                masked.nodes[bridge_name].weights[direct_bridge_edge],
                jnp.zeros_like(params.nodes[bridge_name].weights[direct_bridge_edge]),
            )
            assert jnp.allclose(
                masked.nodes[context_name].weights[context_edge],
                jnp.zeros_like(params.nodes[context_name].weights[context_edge]),
            )
        else:
            assert jnp.allclose(
                masked.nodes[bridge_name].weights[direct_bridge_edge],
                params.nodes[bridge_name].weights[direct_bridge_edge],
            )
            assert jnp.allclose(
                masked.nodes[context_name].weights[context_edge],
                params.nodes[context_name].weights[context_edge],
            )


def test_mask_column_shell_path_inputs_masks_direct_and_bridge_routes() -> None:
    """Combined shell-path masking applies the same shell lesion to both routes."""
    args = _tiny_depth_spanning_args(
        column_shell_readout=True,
        column_shell_bridge=True,
    )
    structure, support_mask = build_depth_spanning_graph(args)
    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    params = initialize_params(structure, jax.random.PRNGKey(0))
    output_sources = output_input_edge_sources(structure)

    masked = mask_column_shell_path_inputs(
        params,
        structure,
        "outer_shell",
        keep_shell=False,
    )

    assert jnp.allclose(
        masked.nodes["output"].weights[output_sources["column_pool"]],
        jnp.zeros_like(params.nodes["output"].weights[output_sources["column_pool"]]),
    )
    assert jnp.allclose(
        masked.nodes["output"].weights[output_sources["bypass_pool"]],
        jnp.zeros_like(params.nodes["output"].weights[output_sources["bypass_pool"]]),
    )

    for column_idx in active_columns:
        bridge_name = column_shell_bridge_node_name(column_idx)
        bridge_input_sources = node_input_edge_sources(structure, bridge_name)
        bridge_output_edge = output_sources[bridge_name]

        assert jnp.allclose(
            masked.nodes["output"].weights[bridge_output_edge],
            params.nodes["output"].weights[bridge_output_edge],
        )

        for shell_name in SHELL_NAMES:
            pool_name = column_shell_pool_node_name(column_idx, shell_name)
            pool_output_edge = output_sources[pool_name]
            bridge_input_edge = bridge_input_sources[pool_name]

            if shell_name == "outer_shell":
                assert jnp.allclose(
                    masked.nodes["output"].weights[pool_output_edge],
                    jnp.zeros_like(
                        params.nodes["output"].weights[pool_output_edge]
                    ),
                )
                assert jnp.allclose(
                    masked.nodes[bridge_name].weights[bridge_input_edge],
                    jnp.zeros_like(
                        params.nodes[bridge_name].weights[bridge_input_edge]
                    ),
                )
            else:
                assert jnp.allclose(
                    masked.nodes["output"].weights[pool_output_edge],
                    params.nodes["output"].weights[pool_output_edge],
                )
                assert jnp.allclose(
                    masked.nodes[bridge_name].weights[bridge_input_edge],
                    params.nodes[bridge_name].weights[bridge_input_edge],
                )
