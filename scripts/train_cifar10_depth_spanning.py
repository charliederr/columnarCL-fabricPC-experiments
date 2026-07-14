#!/usr/bin/env python3
"""
CIFAR-10 with depth-spanning columnar architecture.

This experiment extends the PC ResNet baseline with depth-spanning columns
that receive skip connections from multiple ResNet stages. Each column
contains K/L/B microcolumn pathways that process multi-scale features.

Architecture::

    ResNet stage2 ───────► stage2_tap ───────┐
    ResNet stage3 ───────► stage3_tap ───────┼──────────────┐
    ResNet stage4 ───────► stage4_tap ───────┤              │
           │                                  │              ▼
           └────────────► stage4_pool ────────┘     Depth-spanning columns
                                                         │
                                                         │ per-column shell slices
                                                         ▼
      optional per-column shell readout ─────────► output classifier
      optional outer-shell context ──────────────► optional output classifier readout
                         │
                         ├──── optional shell-bridge conditioning
                         │              │
                         │              ▼
      optional per-column shell bridge ──────────► output classifier
                         │
                         ├──── optional context teacher
                         │
                         ├──── optional class-shaped context evidence
                         │                 │
                         │                 └──── optional evidence teacher
                         │
                         └──── optional shell-local predictors
                                           │
                                           └──── predict pooled shell states
                                                         │
                                                         ▼
                                                shell-aware combiner
                                                         │
                                                         ▼
                                                   column_pool
                                                         │
                                                         ├────► output classifier
                                                         └────► optional column teacher

Usage:
    python scripts/train_cifar10_depth_spanning.py --quick
    python scripts/train_cifar10_depth_spanning.py --model resnet18 --num_epochs 2

`--column_grid stage3` makes all stage taps emit tokens on the intermediate
stage grid. For ResNet-18 on CIFAR-10, this gives 8 by 8 = 64 column tokens
instead of the stage4 default of 4 by 4 = 16 tokens.
"""

import argparse
import math
import os
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import ConvNode, Linear, IdentityNode, SkipConnection, AvgPool
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphParams, GraphStructure, NodeParams
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.energy import GaussianEnergy
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    GeluActivation,
    LeakyReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train import get_graph_param_gradient
from fabricpc.utils.data.dataloader import Cifar10Loader
from fabricpc.utils.dashboarding.extractors import (
    extract_node_energies,
    extract_latent_statistics,
)

from columnar_cl_fabricpc.columns import (
    StageTapTokenizer,
    GlobalPoolNode,
    DepthSpanningColumnNode,
    FeatureSliceNode,
    MeanSquaredGaussianEnergy,
    SpatialReferenceGaussianEnergy,
    WeightedLabelSmoothedCrossEntropyEnergy,
    SHELL_NAMES,
    create_stage_tap,
    create_global_pool,
    create_depth_spanning_column,
    get_shell_slices,
)
from columnar_cl_fabricpc.columns.accuracy_nodes import MaskedColumnCombinerNode
from columnar_cl_fabricpc.columns.accuracy_nodes import ColumnShellComposerNode
from columnar_cl_fabricpc.columns.accuracy_nodes import ShellContextPredictionNode

jax.config.update("jax_default_prng_impl", "threefry2x32")

COLUMN_TEACHER_TARGET = "column_y"
COLUMN_TEACHER_NODE = "column_teacher_output"
OUTER_SHELL_CONTEXT_TEACHER_TARGET = "outer_shell_context_y"
OUTER_SHELL_CONTEXT_TEACHER_NODE = "outer_shell_context_teacher_output"
OUTER_SHELL_CONTEXT_EVIDENCE_NODE = "outer_shell_context_evidence"
OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_TARGET = "outer_shell_context_evidence_teacher_y"
OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE = (
    "outer_shell_context_evidence_teacher_output"
)
SHELL_CONTEXT_PREDICTION_DEFAULT_WEIGHT = 0.0
SHELL_TEACHER_DEFAULT_WEIGHTS = "0,0,0,0"
COLUMN_SHELL_TEACHER_DEFAULT_WEIGHTS = "0,0,0,0"
SHELL_LR_DEFAULT_MULTIPLIERS = "1,1,1,1"
SHELL_EVIDENCE_CASCADE_DEFAULT_SCALE = "0.05,0.05,0.05"
SHELL_INHIBITION_DEFAULT_STRENGTHS = "0,0.35,0.22,0.10"
COLUMN_GRID_CHOICES = ("stage4", "stage3", "stage2")


def shell_slice_node_name(shell_name: str) -> str:
    """Return the graph node name for a pooled shell feature slice."""
    return f"{shell_name}_slice"


def shell_teacher_node_name(shell_name: str) -> str:
    """Return the graph node name for a shell-local classifier head."""
    return f"{shell_name}_teacher_output"


def shell_teacher_target_name(shell_name: str) -> str:
    """Return the training batch target key for a shell-local classifier head."""
    return f"{shell_name}_y"


def column_shell_slice_node_name(column_idx: int, shell_name: str) -> str:
    """Return the node name for a shell slice taken directly from one column."""
    return f"column{column_idx:02d}_{shell_name}_slice"


def column_shell_pool_node_name(column_idx: int, shell_name: str) -> str:
    """Return the node name for the token-pooled shell slice from one column."""
    return f"column{column_idx:02d}_{shell_name}_pool"


def column_shell_bridge_node_name(column_idx: int) -> str:
    """Return the node name for one column's shell-to-shell bridge latent."""
    return f"column{column_idx:02d}_shell_bridge"


def outer_shell_context_node_name(column_idx: int) -> str:
    """Return the node name for one column's outer-shell context latent."""
    return f"column{column_idx:02d}_outer_shell_context"


def shell_context_prediction_node_name(column_idx: int, shell_name: str) -> str:
    """Return the node name for a context-to-shell prediction objective."""
    return f"column{column_idx:02d}_{shell_name}_context_prediction"


def column_shell_teacher_node_name(column_idx: int, shell_name: str) -> str:
    """Return the classifier node name for one column shell teacher head."""
    return f"column{column_idx:02d}_{shell_name}_teacher_output"


def column_shell_teacher_target_name(column_idx: int, shell_name: str) -> str:
    """Return the training target key for one column shell teacher head."""
    return f"column{column_idx:02d}_{shell_name}_y"


def is_column_shell_auxiliary_node(node_name: str) -> bool:
    """Return true for per-column shell slice, pool, or teacher nodes."""
    return node_name.startswith("column") and any(
        f"_{shell_name}_" in node_name for shell_name in SHELL_NAMES
    )


def is_column_shell_teacher_node(node_name: str) -> bool:
    """Return true for per-column shell-local classifier nodes."""
    return (
        is_column_shell_auxiliary_node(node_name)
        and node_name.endswith("_teacher_output")
    )


def is_column_shell_pool_node(node_name: str) -> bool:
    """Return true for a token-pooled per-column shell readout node."""
    return (
        is_column_shell_auxiliary_node(node_name)
        and node_name.endswith("_pool")
    )


def is_column_shell_pool_for_shell(node_name: str, shell_name: str) -> bool:
    """Return true when a pooled per-column shell node belongs to `shell_name`."""
    return is_column_shell_pool_node(node_name) and node_name.endswith(
        f"_{shell_name}_pool"
    )


def is_column_shell_bridge_node(node_name: str) -> bool:
    """Return true for a per-column shell bridge latent."""
    return node_name.startswith("column") and node_name.endswith("_shell_bridge")


def is_outer_shell_context_node(node_name: str) -> bool:
    """Return true for a per-column outer-shell context latent."""
    return node_name.startswith("column") and node_name.endswith(
        "_outer_shell_context"
    )


def is_outer_shell_context_teacher_node(node_name: str) -> bool:
    """Return true for the context-only classifier head."""
    return node_name == OUTER_SHELL_CONTEXT_TEACHER_NODE


def is_outer_shell_context_evidence_node(node_name: str) -> bool:
    """Return true for the class-shaped outer-context evidence latent."""
    return node_name == OUTER_SHELL_CONTEXT_EVIDENCE_NODE


def is_outer_shell_context_evidence_teacher_node(node_name: str) -> bool:
    """Return true for the classifier head fed by context evidence."""
    return node_name == OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE


def is_shell_context_prediction_node(node_name: str) -> bool:
    """Return true for a context-to-shell local prediction objective."""
    return node_name.startswith("column") and node_name.endswith("_context_prediction")


def shell_context_prediction_shell(node_name: str) -> str | None:
    """Return the target shell encoded in a context-prediction node name."""
    if not is_shell_context_prediction_node(node_name):
        return None
    for shell_name in SHELL_NAMES:
        if f"_{shell_name}_context_prediction" in node_name:
            return shell_name
    return None


def parse_shell_ordered_values(
    value: str,
    flag_name: str,
    value_name: str,
) -> Dict[str, float]:
    """
    Parse non-negative values in `SHELL_NAMES` order.

    The expected string is four comma-separated floats: hard-kernel,
    inner-shell, middle-shell, and outer-shell.
    """
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(pieces) != len(SHELL_NAMES):
        raise ValueError(
            f"{flag_name} must contain {len(SHELL_NAMES)} comma-separated "
            f"values in {SHELL_NAMES} order, got {value!r}"
        )
    weights = {}
    for shell_name, piece in zip(SHELL_NAMES, pieces):
        weight = float(piece)
        if weight < 0.0:
            raise ValueError(f"{value_name} for {shell_name} must be >= 0")
        weights[shell_name] = weight
    return weights


def parse_shell_teacher_weights(
    value: str,
    flag_name: str = "--shell_teacher_weights",
) -> Dict[str, float]:
    """Parse shell-local teacher weights in `SHELL_NAMES` order."""
    return parse_shell_ordered_values(
        value,
        flag_name=flag_name,
        value_name="Shell teacher weight",
    )


def parse_shell_lr_multipliers(value: str) -> Dict[str, float]:
    """Parse optimizer update multipliers in `SHELL_NAMES` order."""
    return parse_shell_ordered_values(
        value,
        flag_name="--shell_lr_multipliers",
        value_name="Shell learning-rate multiplier",
    )


def parse_shell_inhibition_strengths(value: str) -> Dict[str, float]:
    """Parse same-tier inhibition strengths in `SHELL_NAMES` order."""
    return parse_shell_ordered_values(
        value,
        flag_name="--shell_inhibition_strengths",
        value_name="Shell inhibition strength",
    )


def parse_shell_evidence_cascade_scale(value: str) -> Tuple[float, float, float]:
    """
    Parse outward shell evidence-cascade gains.

    The expected string is three comma-separated floats for hard-kernel to
    inner-shell, inner-shell to middle-shell, and middle-shell to outer-shell.
    """
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(
            "--shell_evidence_cascade_scale must contain three comma-separated "
            f"values, got {value!r}"
        )
    scale = tuple(float(piece) for piece in pieces)
    if any(value < 0.0 for value in scale):
        raise ValueError("--shell_evidence_cascade_scale entries must be >= 0")
    return scale


def _constant_multiplier_like(value: jax.Array, multiplier: float) -> jax.Array:
    """Return an array-shaped update multiplier with one scalar value."""
    return jnp.ones_like(value) * jnp.asarray(multiplier, dtype=value.dtype)


def _scale_final_axis_by_shell(
    value: jax.Array,
    shell_slices: Dict[str, Tuple[int, int]],
    shell_lr_multipliers: Dict[str, float],
) -> jax.Array:
    """Return update multipliers for shell slices on the final tensor axis."""
    if value.ndim == 0:
        return jnp.ones_like(value)

    multipliers = jnp.ones_like(value)
    for shell_name in SHELL_NAMES:
        start, end = shell_slices[shell_name]
        multipliers = multipliers.at[..., start:end].set(
            shell_lr_multipliers[shell_name]
        )
    return multipliers


def _scale_shell_axis_by_shell(
    value: jax.Array,
    shell_lr_multipliers: Dict[str, float],
    shell_axis: int,
) -> jax.Array:
    """Return update multipliers for an explicit shell axis."""
    if value.ndim == 0:
        return jnp.ones_like(value)

    axis = shell_axis if shell_axis >= 0 else value.ndim + shell_axis
    if axis < 0 or axis >= value.ndim or value.shape[axis] != len(SHELL_NAMES):
        return jnp.ones_like(value)

    shell_values = jnp.asarray(
        [shell_lr_multipliers[shell_name] for shell_name in SHELL_NAMES],
        dtype=value.dtype,
    )
    broadcast_shape = [1] * value.ndim
    broadcast_shape[axis] = len(SHELL_NAMES)
    return jnp.ones_like(value) * shell_values.reshape(tuple(broadcast_shape))


def _scale_evidence_cascade_gains(
    value: jax.Array,
    shell_lr_multipliers: Dict[str, float],
) -> jax.Array:
    """Return update multipliers for adjacent outward shell-cascade gains."""
    if value.shape != (len(SHELL_NAMES) - 1,):
        return jnp.ones_like(value)
    target_values = jnp.asarray(
        [shell_lr_multipliers[shell_name] for shell_name in SHELL_NAMES[1:]],
        dtype=value.dtype,
    )
    return jnp.ones_like(value) * target_values


def _cascade_target_shell(param_name: str) -> str | None:
    """Return the target shell encoded in a shell-cascade parameter name."""
    if not param_name.startswith("shell_evidence_cascade"):
        return None
    for shell_name in SHELL_NAMES[1:]:
        if param_name.endswith(f"_to_{shell_name}"):
            return shell_name
    return None


def _composer_projection_shell(param_name: str) -> str | None:
    """Return the shell encoded in a composer projection parameter name."""
    if not (param_name.startswith("W_col") or param_name.startswith("b_col")):
        return None
    for shell_name in SHELL_NAMES:
        if param_name.endswith(f"_{shell_name}"):
            return shell_name
    return None


def parameter_shell_lr_multiplier(
    node_name: str,
    param_name: str,
    value: jax.Array,
    structure: GraphStructure,
    shell_lr_multipliers: Dict[str, float],
) -> jax.Array:
    """
    Build the update multiplier for one learnable parameter tensor.

    `shell_lr_multipliers` assigns one optimizer-update multiplier to each shell.
    Sliced output matrices use the shell on their final output axis. Shell-specific
    matrices use the shell encoded in their parameter name.
    """
    node = structure.nodes.get(node_name)
    if node is None:
        return jnp.ones_like(value)

    node_info = node.node_info
    node_class = node_info.node_class

    if node_class is DepthSpanningColumnNode:
        shell_slices = get_shell_slices(node_info.shape[-1])
        if param_name in {
            "K_W_out",
            "K_b_out",
            "L_W_out",
            "L_b_out",
            "B_W_out",
            "B_b_out",
            "ln_gamma",
            "ln_beta",
        }:
            return _scale_final_axis_by_shell(
                value,
                shell_slices,
                shell_lr_multipliers,
            )
        if param_name == "shell_path_scale":
            return _scale_shell_axis_by_shell(
                value,
                shell_lr_multipliers,
                shell_axis=0,
            )
        if param_name == "shell_evidence_cascade_scale":
            return _scale_evidence_cascade_gains(value, shell_lr_multipliers)

        target_shell = _cascade_target_shell(param_name)
        if target_shell is not None:
            return _constant_multiplier_like(
                value,
                shell_lr_multipliers[target_shell],
            )

    if node_class is ColumnShellComposerNode:
        if param_name == "component_attention":
            return _scale_shell_axis_by_shell(
                value,
                shell_lr_multipliers,
                shell_axis=1,
            )

        projection_shell = _composer_projection_shell(param_name)
        if projection_shell is not None:
            return _constant_multiplier_like(
                value,
                shell_lr_multipliers[projection_shell],
            )

    if is_outer_shell_context_node(node_name):
        return _constant_multiplier_like(value, shell_lr_multipliers["outer_shell"])

    prediction_shell = shell_context_prediction_shell(node_name)
    if prediction_shell is not None:
        return _constant_multiplier_like(
            value,
            shell_lr_multipliers[prediction_shell],
        )

    if is_column_shell_bridge_node(node_name):
        shell_slices = get_shell_slices(node_info.shape[-1])
        return _scale_final_axis_by_shell(
            value,
            shell_slices,
            shell_lr_multipliers,
        )

    return jnp.ones_like(value)


def build_shell_lr_multiplier_tree(
    params: GraphParams,
    structure: GraphStructure,
    shell_lr_multipliers: Dict[str, float],
) -> GraphParams:
    """Build a parameter-shaped tree of shell-specific update multipliers."""
    multiplier_nodes = {}
    for node_name, node_params in params.nodes.items():
        multiplier_nodes[node_name] = NodeParams(
            weights={
                param_name: parameter_shell_lr_multiplier(
                    node_name,
                    param_name,
                    value,
                    structure,
                    shell_lr_multipliers,
                )
                for param_name, value in node_params.weights.items()
            },
            biases={
                param_name: parameter_shell_lr_multiplier(
                    node_name,
                    param_name,
                    value,
                    structure,
                    shell_lr_multipliers,
                )
                for param_name, value in node_params.biases.items()
            },
        )
    return GraphParams(nodes=multiplier_nodes)


def apply_shell_lr_multipliers(
    updates: GraphParams,
    shell_lr_multiplier_tree: GraphParams,
) -> GraphParams:
    """Scale optimizer updates by a parameter-shaped multiplier tree."""
    return jax.tree_util.tree_map(
        lambda update, multiplier: update * multiplier,
        updates,
        shell_lr_multiplier_tree,
    )


def scale_updates_by_shell_lr(
    shell_lr_multiplier_tree: GraphParams,
) -> optax.GradientTransformation:
    """Create an Optax transform that applies shell-specific update multipliers."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        return apply_shell_lr_multipliers(updates, shell_lr_multiplier_tree), state

    return optax.GradientTransformation(init_fn, update_fn)


def diagnose_energy_breakdown(
    params,
    structure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Compute per-node-type energy breakdown using FabricPC's extract_node_energies.

    Categorizes nodes into:
    - backbone: ResNet conv/skip nodes (s*b*, stem)
    - stage_taps: StageTapTokenizer nodes (stage*_tap, stage*_pool)
    - columns: DepthSpanningColumnNode (col_*)
    - combiner: ColumnCombinerNode and raw pooled readout
    - classifier: CrossEntropy output node (output)

    Returns dict with per-category energy sums and E_gauss/E_ce ratio.
    """
    # Run forward pass to get final_state
    _, _, final_state = get_graph_param_gradient(params, batch, structure, rng_key)

    # Extract per-node energies using FabricPC's extractor
    node_energies = extract_node_energies(final_state)

    # Categorize nodes
    categories = {
        "backbone": 0.0,
        "stage_taps": 0.0,
        "columns": 0.0,
        "combiner": 0.0,
        "bypass": 0.0,
        "classifier": 0.0,
    }
    shell_teacher_nodes = {
        shell_teacher_node_name(shell_name) for shell_name in SHELL_NAMES
    }

    for node_name, energy_arr in node_energies.items():
        energy_sum = float(energy_arr.sum())

        if (
            node_name in ("output", COLUMN_TEACHER_NODE)
            or node_name in shell_teacher_nodes
            or is_column_shell_teacher_node(node_name)
            or is_outer_shell_context_teacher_node(node_name)
            or is_outer_shell_context_evidence_teacher_node(node_name)
        ):
            categories["classifier"] += energy_sum
        elif (
            node_name.startswith("col_")
            or is_column_shell_auxiliary_node(node_name)
            or is_column_shell_bridge_node(node_name)
            or is_outer_shell_context_node(node_name)
            or is_outer_shell_context_evidence_node(node_name)
            or is_shell_context_prediction_node(node_name)
        ):
            categories["columns"] += energy_sum
        elif node_name.startswith("stage") and ("tap" in node_name or "pool" in node_name):
            categories["stage_taps"] += energy_sum
        elif node_name in ("combiner", "column_pool"):
            categories["combiner"] += energy_sum
        elif node_name == "bypass_pool":
            categories["bypass"] += energy_sum
        else:
            # ResNet backbone nodes: stem, s*b*_conv_*, s*b*_skip_*
            categories["backbone"] += energy_sum

    # Compute Gaussian vs CrossEntropy
    e_gauss = (
        categories["backbone"]
        + categories["stage_taps"]
        + categories["columns"]
        + categories["combiner"]
        + categories["bypass"]
    )
    e_ce = categories["classifier"]

    ratio = e_gauss / e_ce if e_ce > 1e-8 else float("inf")

    return {
        **categories,
        "E_gauss": e_gauss,
        "E_ce": e_ce,
        "E_gauss/E_ce": ratio,
    }


def diagnose_column_outputs(
    params,
    structure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-column z_latent statistics after inference.

    The "columns collapse to trivial outputs" claim is a statement about the
    magnitude of column outputs. The relevant tensor is each column node's
    z_latent after the inference loop has converged: that is the value passed
    forward to the combiner. This function reports mean(|z_latent|), std, min,
    max for every col_* node, plus the combiner and stage taps for context.
    """
    _, _, final_state = get_graph_param_gradient(params, batch, structure, rng_key)
    shell_slice_nodes = {
        shell_slice_node_name(shell_name) for shell_name in SHELL_NAMES
    }
    shell_teacher_nodes = {
        shell_teacher_node_name(shell_name) for shell_name in SHELL_NAMES
    }
    interesting = [
        name
        for name in final_state.nodes
        if name.startswith("col_")
        or name.startswith("stage")
        or name in shell_slice_nodes
        or name in shell_teacher_nodes
        or is_column_shell_auxiliary_node(name)
        or is_column_shell_bridge_node(name)
        or is_outer_shell_context_node(name)
        or is_outer_shell_context_teacher_node(name)
        or is_outer_shell_context_evidence_node(name)
        or is_outer_shell_context_evidence_teacher_node(name)
        or is_shell_context_prediction_node(name)
        or name in (
            "combiner",
            "column_pool",
            "output",
            COLUMN_TEACHER_NODE,
            "bypass_pool",
        )
    ]
    return extract_latent_statistics(final_state, nodes=interesting)


def diagnose_column_correlation(
    params,
    structure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Compute the maximum off-diagonal pairwise Pearson correlation between
    column z_latent vectors after inference, on a fixed diagnostic batch.

    High correlation across columns means the four (or N) columns are producing
    redundant features. A sum combiner with redundant column outputs collapses
    to a single feature; that is the load-bearing claim of Path D in
    docs/dev-plans/2026-06-21-claude-columnar-cifar10-path-forward.md.

    Returns summary: max off-diagonal |corr|, mean off-diagonal |corr|.
    """
    _, _, final_state = get_graph_param_gradient(params, batch, structure, rng_key)
    col_names = sorted(n for n in final_state.nodes if n.startswith("col_"))
    if len(col_names) < 2:
        return {"max_abs_offdiag": 0.0, "mean_abs_offdiag": 0.0}

    flat_cols = []
    for name in col_names:
        z = final_state.nodes[name].z_latent
        # collapse over batch+spatial axes; one row per column
        flat = jnp.reshape(z, (-1,))
        flat = flat - jnp.mean(flat)
        norm = jnp.linalg.norm(flat) + 1e-8
        flat_cols.append(flat / norm)

    mat = jnp.stack(flat_cols, axis=0)  # (n_cols, n_features)
    corr = jnp.matmul(mat, mat.T)
    n = corr.shape[0]
    mask = 1.0 - jnp.eye(n)
    off = jnp.abs(corr) * mask
    max_off = float(jnp.max(off))
    mean_off = float(jnp.sum(off) / jnp.sum(mask))
    return {"max_abs_offdiag": max_off, "mean_abs_offdiag": mean_off}


def diagnose_shell_norms(
    params,
    structure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Compute shell-resolved L2 norms from inferred column latents.

    The L2 norm for a shell is the square root of the sum of squared feature
    activations over that shell slice. The reported value averages over batch
    items, token positions, and columns.
    """
    _, _, final_state = get_graph_param_gradient(params, batch, structure, rng_key)
    col_names = sorted(name for name in final_state.nodes if name.startswith("col_"))
    if not col_names:
        return {}

    output_dim = final_state.nodes[col_names[0]].z_latent.shape[-1]
    shell_slices = get_shell_slices(output_dim)
    stats: Dict[str, Dict[str, float]] = {}
    for shell_name in SHELL_NAMES:
        start, end = shell_slices[shell_name]
        per_column = []
        for col_name in col_names:
            z = final_state.nodes[col_name].z_latent[..., start:end]
            l2 = jnp.sqrt(jnp.sum(jnp.square(z), axis=-1) + 1e-8)
            per_column.append(jnp.mean(l2))
        values = jnp.stack(per_column)
        stats[shell_name] = {
            "mean_l2": float(jnp.mean(values)),
            "std_l2": float(jnp.std(values)),
            "width": float(end - start),
        }
    return stats


def diagnose_shell_context_prediction_energies(
    params,
    structure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Average local context-prediction energy by target shell.

    The reported value for a shell is the mean per-node energy across all active
    columns whose outer-context latent predicts that shell's pooled state.
    """
    _, _, final_state = get_graph_param_gradient(params, batch, structure, rng_key)
    node_energies = extract_node_energies(final_state)
    shell_values: Dict[str, List[float]] = {shell_name: [] for shell_name in SHELL_NAMES}
    for node_name, energy_arr in node_energies.items():
        shell_name = shell_context_prediction_shell(node_name)
        if shell_name is None:
            continue
        shell_values[shell_name].append(float(jnp.mean(energy_arr)))
    return {
        shell_name: float(sum(values) / len(values))
        for shell_name, values in shell_values.items()
        if values
    }


def diagnose_output_edge_weight_norms(
    params: GraphParams,
    structure: GraphStructure,
) -> Dict[str, float]:
    """
    Per-source Frobenius norm of the classifier's incoming weight matrices.

    The source name is the architectural route feeding the `output` classifier,
    such as `column_pool`, `bypass_pool`, a per-column shell pool, or a shell
    bridge node.
    """
    if "output" not in params.nodes:
        return {}
    edge_sources = output_input_edge_sources(structure)
    out_weights = params.nodes["output"].weights
    return {
        source_name: float(jnp.linalg.norm(out_weights[edge_key]))
        for source_name, edge_key in sorted(edge_sources.items())
        if edge_key in out_weights
    }


def has_shell_composer(structure: GraphStructure) -> bool:
    """Return true when the graph uses `ColumnShellComposerNode` at `combiner`."""
    if "combiner" not in structure.nodes:
        return False
    return (
        structure.nodes["combiner"].node_info.node_class
        is ColumnShellComposerNode
    )


def active_composer_components(
    structure: GraphStructure,
) -> Tuple[Tuple[int, str], ...]:
    """
    Return active `(column index, shell name)` components in the composer.

    The column index names the `col_XX` input to `combiner`. The shell name is
    one of `hard_kernel`, `inner_shell`, `middle_shell`, or `outer_shell`.
    """
    if not has_shell_composer(structure):
        return ()
    config = structure.nodes["combiner"].node_info.node_config
    num_columns = int(config.get("num_columns", 0))
    support_mask = tuple(float(value) for value in config.get("support_mask", ()))
    return tuple(
        (column_idx, shell_name)
        for column_idx in range(num_columns)
        if column_idx < len(support_mask) and support_mask[column_idx] > 0.0
        for shell_name in SHELL_NAMES
    )


def diagnose_composer_attention(
    params: GraphParams,
    structure: GraphStructure,
) -> Dict[str, Dict[str, float]]:
    """
    Report the learned attention over active `(column, shell)` components.

    `attention[c][s]` is the softmax weight on column `c` for shell `s` inside
    `ColumnShellComposerNode`. Inactive columns are masked before a separate
    softmax is applied over columns for each shell.
    """
    if not has_shell_composer(structure):
        return {}
    config = structure.nodes["combiner"].node_info.node_config
    logits = params.nodes["combiner"].weights["component_attention"]
    support_mask = jnp.asarray(config.get("support_mask"), dtype=logits.dtype)[:, None]
    masked_logits = jnp.where(support_mask > 0.0, logits, -1.0e9)
    attention = jax.nn.softmax(masked_logits, axis=0)

    rows: Dict[str, Dict[str, float]] = {}
    for column_idx, shell_name in active_composer_components(structure):
        column_key = f"col_{column_idx:02d}"
        if column_key not in rows:
            rows[column_key] = {}
        shell_idx = SHELL_NAMES.index(shell_name)
        rows[column_key][shell_name] = float(attention[column_idx, shell_idx])
    return rows


def diagnose_composer_projection_norms(
    params: GraphParams,
    structure: GraphStructure,
) -> Dict[str, float]:
    """
    Report Frobenius norms of composer projection matrices.

    Each entry is one learned projection from a `(column, shell)` feature slice
    into the same shell's output feature slice.
    """
    if not has_shell_composer(structure):
        return {}
    combiner_params = params.nodes["combiner"]
    norms = {}
    for column_idx, shell_name in active_composer_components(structure):
        weight_name = ColumnShellComposerNode._projection_weight_name(
            column_idx,
            shell_name,
        )
        key = f"col_{column_idx:02d}.{shell_name}"
        norms[key] = float(jnp.linalg.norm(combiner_params.weights[weight_name]))
    return norms


def mask_composer_components(
    params: GraphParams,
    structure: GraphStructure,
    kept_components: Tuple[Tuple[int, str], ...] | None = None,
    dropped_components: Tuple[Tuple[int, str], ...] | None = None,
) -> GraphParams:
    """
    Zero selected composer projections in a copied parameter tree.

    Components are identified as `(column index, shell name)` pairs. Keeping
    components means all other active components are zeroed. Dropping components
    means only those active components are zeroed. The graph structure and
    learned attention logits are unchanged.
    """
    if kept_components is not None and dropped_components is not None:
        raise ValueError("Use either kept_components or dropped_components, not both")
    if not has_shell_composer(structure):
        raise ValueError("Composer component masking requires shell_attention combiner")

    active_components = active_composer_components(structure)
    keep_set = set(kept_components) if kept_components is not None else None
    drop_set = set(dropped_components) if dropped_components is not None else set()
    combiner_params = params.nodes["combiner"]
    masked_weights = dict(combiner_params.weights)
    masked_biases = dict(combiner_params.biases)

    for component in active_components:
        if keep_set is not None:
            should_zero = component not in keep_set
        else:
            should_zero = component in drop_set
        if not should_zero:
            continue
        column_idx, shell_name = component
        weight_name = ColumnShellComposerNode._projection_weight_name(
            column_idx,
            shell_name,
        )
        bias_name = ColumnShellComposerNode._projection_bias_name(
            column_idx,
            shell_name,
        )
        masked_weights[weight_name] = jnp.zeros_like(masked_weights[weight_name])
        masked_biases[bias_name] = jnp.zeros_like(masked_biases[bias_name])

    masked_combiner = NodeParams(
        weights=masked_weights,
        biases=masked_biases,
    )
    return params._replace(nodes={**params.nodes, "combiner": masked_combiner})


def evaluate_composer_component_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate composer-internal lesions through the `column_pool` route only.

    The baseline keeps only `column_pool -> output`. Each lesion then zeros one
    active `(column, shell)` projection inside `ColumnShellComposerNode` before
    evaluating the same column-pool readout.
    """
    if not has_shell_composer(structure):
        return {}
    output_sources = output_input_edge_sources(structure)
    if "column_pool" not in output_sources:
        return {}

    column_only_params = mask_output_input_sources(
        params,
        structure,
        kept_sources=("column_pool",),
    )
    results = {
        "composer_column_only": evaluate_pcn(
            column_only_params,
            structure,
            loader,
            config,
            rng_key,
        )
    }
    for column_idx, shell_name in active_composer_components(structure):
        component = ((column_idx, shell_name),)
        masked = mask_composer_components(
            column_only_params,
            structure,
            dropped_components=component,
        )
        case_name = f"composer_without_col{column_idx:02d}_{shell_name}"
        results[case_name] = evaluate_pcn(
            masked,
            structure,
            loader,
            config,
            rng_key,
        )
    return results


def node_input_edge_sources(
    structure: GraphStructure,
    target_node: str,
) -> Dict[str, str]:
    """Return input-edge keys for a target node, indexed by source node name."""
    return {
        edge.source: edge_key
        for edge_key, edge in structure.edges.items()
        if edge.target == target_node and edge.slot == "in"
    }


def output_input_edge_sources(structure: GraphStructure) -> Dict[str, str]:
    """Return classifier input-edge keys by source node name."""
    return node_input_edge_sources(structure, "output")


def mask_node_input_sources(
    params: GraphParams,
    structure: GraphStructure,
    target_node: str,
    kept_sources: Tuple[str, ...],
) -> GraphParams:
    """
    Zero target-node input weights outside `kept_sources`.

    The predictive-coding graph is unchanged. Only the selected node's per-edge
    input matrices are masked in a copied parameter tree for evaluation.
    """
    kept = set(kept_sources)
    target_params = params.nodes[target_node]
    masked_weights = {}
    for edge_key, weight in target_params.weights.items():
        edge = structure.edges.get(edge_key)
        if edge is not None and edge.target == target_node and edge.source not in kept:
            masked_weights[edge_key] = jnp.zeros_like(weight)
        else:
            masked_weights[edge_key] = weight

    masked_node = NodeParams(
        weights=masked_weights,
        biases=target_params.biases,
    )
    return params._replace(nodes={**params.nodes, target_node: masked_node})


def mask_output_input_sources(
    params: GraphParams,
    structure: GraphStructure,
    kept_sources: Tuple[str, ...],
) -> GraphParams:
    """
    Zero classifier weights for output input edges outside `kept_sources`.

    The predictive-coding graph is unchanged. Only the output node's per-edge
    classifier matrices are masked in a copied parameter tree for evaluation.
    """
    return mask_node_input_sources(params, structure, "output", kept_sources)


def mask_output_source_feature_slice(
    params: GraphParams,
    structure: GraphStructure,
    source: str,
    feature_slice: Tuple[int, int],
    keep_slice: bool,
) -> GraphParams:
    """
    Mask one feature slice of one classifier input edge.

    If `keep_slice` is false, the selected slice is zeroed. If `keep_slice` is
    true, all features outside the selected slice are zeroed.
    """
    edge_sources = output_input_edge_sources(structure)
    if source not in edge_sources:
        raise ValueError(f"Output node has no input edge from {source!r}")

    edge_key = edge_sources[source]
    output_params = params.nodes["output"]
    masked_weights = dict(output_params.weights)
    weight = output_params.weights[edge_key]
    if weight.ndim < 2:
        raise ValueError(
            f"Expected classifier weight for {source!r} to have rank >= 2, got {weight.shape}"
        )

    start, end = feature_slice
    if start < 0 or end > weight.shape[0] or start >= end:
        raise ValueError(f"Invalid feature slice {feature_slice} for weight {weight.shape}")

    mask_shape = (weight.shape[0],) + (1,) * (weight.ndim - 1)
    if keep_slice:
        mask = jnp.zeros(mask_shape, dtype=weight.dtype)
        mask = mask.at[start:end].set(1.0)
    else:
        mask = jnp.ones(mask_shape, dtype=weight.dtype)
        mask = mask.at[start:end].set(0.0)
    masked_weights[edge_key] = weight * mask

    masked_output = NodeParams(
        weights=masked_weights,
        biases=output_params.biases,
    )
    return params._replace(nodes={**params.nodes, "output": masked_output})


def build_readout_ablation_cases(
    structure: GraphStructure,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Build classifier readout source sets for output-edge ablations.

    Each case is a tuple of `case_name` and source-node names to keep at the
    `output` classifier. Evaluation masks all other output input edges.
    """
    edge_sources = output_input_edge_sources(structure)
    column_source = "column_pool"
    if column_source not in edge_sources:
        raise ValueError("Readout ablations require the column_pool edge")

    all_sources = tuple(edge_sources.keys())
    cases: List[Tuple[str, Tuple[str, ...]]] = [
        ("combined", all_sources),
    ]
    column_shell_sources = tuple(
        sorted(source for source in edge_sources if is_column_shell_pool_node(source))
    )
    column_shell_bridge_sources = tuple(
        sorted(source for source in edge_sources if is_column_shell_bridge_node(source))
    )
    outer_shell_context_sources = tuple(
        sorted(source for source in edge_sources if is_outer_shell_context_node(source))
    )
    outer_shell_context_evidence_sources = tuple(
        sorted(
            source
            for source in edge_sources
            if is_outer_shell_context_evidence_node(source)
        )
    )

    def without_sources(dropped_sources: Tuple[str, ...]) -> Tuple[str, ...]:
        dropped = set(dropped_sources)
        return tuple(source for source in all_sources if source not in dropped)

    cases.append(("column_only", (column_source,)))
    if len(all_sources) > 1:
        cases.append(
            (
                "combined_without_column_pool",
                without_sources((column_source,)),
            )
        )
    if column_shell_sources:
        cases.append(("column_shell_readout_only", column_shell_sources))
        cases.append(
            ("column_pool_plus_shell_readout", (column_source, *column_shell_sources))
        )
    if column_shell_bridge_sources:
        cases.append(("column_shell_bridge_only", column_shell_bridge_sources))
        cases.append(
            ("column_pool_plus_shell_bridge", (column_source, *column_shell_bridge_sources))
        )
        cases.append(
            (
                "combined_without_column_shell_bridge",
                without_sources(column_shell_bridge_sources),
            )
        )
    if column_shell_sources and column_shell_bridge_sources:
        cases.append(
            (
                "column_shell_readout_plus_bridge",
                (*column_shell_sources, *column_shell_bridge_sources),
            )
        )
    if outer_shell_context_sources:
        cases.append(("outer_shell_context_only", outer_shell_context_sources))
        cases.append(
            (
                "column_pool_plus_outer_shell_context",
                (column_source, *outer_shell_context_sources),
            )
        )
        cases.append(
            (
                "combined_without_outer_shell_context",
                without_sources(outer_shell_context_sources),
            )
        )
    if column_shell_bridge_sources and outer_shell_context_sources:
        cases.append(
            (
                "outer_shell_context_plus_column_shell_bridge",
                (*outer_shell_context_sources, *column_shell_bridge_sources),
            )
        )
    if outer_shell_context_evidence_sources:
        cases.append(
            (
                "outer_shell_context_evidence_only",
                outer_shell_context_evidence_sources,
            )
        )
        cases.append(
            (
                "column_pool_plus_outer_shell_context_evidence",
                (column_source, *outer_shell_context_evidence_sources),
            )
        )
    if outer_shell_context_sources and outer_shell_context_evidence_sources:
        cases.append(
            (
                "outer_shell_context_plus_evidence",
                (*outer_shell_context_sources, *outer_shell_context_evidence_sources),
            )
        )
    if "bypass_pool" in edge_sources:
        cases.append(("bypass_only", ("bypass_pool",)))

    return cases


def evaluate_readout_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate classifier readout paths by masking output input edges.

    The `combined` case evaluates the trained parameters. `column_only` keeps
    the pooled column readout edge. `bypass_only` keeps the direct backbone
    bypass edge when that edge exists.
    """
    results = {}
    for case_name, kept_sources in build_readout_ablation_cases(structure):
        case_params = (
            params
            if case_name == "combined"
            else mask_output_input_sources(params, structure, kept_sources)
        )
        results[case_name] = evaluate_pcn(
            case_params, structure, loader, config, rng_key
        )
    return results


def evaluate_shell_readout_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate shell-slice readout dependence by masking `column_pool` features.

    The lesion cases keep the graph fixed and only mask the classifier matrix
    attached to the `column_pool` source.
    """
    column_source = "column_pool"
    edge_sources = output_input_edge_sources(structure)
    if column_source not in edge_sources:
        raise ValueError("Shell ablations require the column_pool edge")

    output_dim = structure.nodes[column_source].shape[-1]
    shell_slices = get_shell_slices(output_dim)
    column_only_params = mask_output_input_sources(
        params, structure, kept_sources=(column_source,)
    )

    results = {}
    for shell_name in SHELL_NAMES:
        feature_slice = shell_slices[shell_name]
        results[f"combined_without_{shell_name}"] = evaluate_pcn(
            mask_output_source_feature_slice(
                params,
                structure,
                source=column_source,
                feature_slice=feature_slice,
                keep_slice=False,
            ),
            structure,
            loader,
            config,
            rng_key,
        )
        results[f"column_without_{shell_name}"] = evaluate_pcn(
            mask_output_source_feature_slice(
                column_only_params,
                structure,
                source=column_source,
                feature_slice=feature_slice,
                keep_slice=False,
            ),
            structure,
            loader,
            config,
            rng_key,
        )
        results[f"column_{shell_name}_only"] = evaluate_pcn(
            mask_output_source_feature_slice(
                column_only_params,
                structure,
                source=column_source,
                feature_slice=feature_slice,
                keep_slice=True,
            ),
            structure,
            loader,
            config,
            rng_key,
        )
    return results


def evaluate_column_shell_readout_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate direct per-column shell readout edges by masking output inputs.

    These cases isolate the classifier evidence that reaches `output` through
    pooled `(column, shell)` vectors before the column combiner can merge columns.
    """
    edge_sources = output_input_edge_sources(structure)
    shell_sources = tuple(
        sorted(source for source in edge_sources if is_column_shell_pool_node(source))
    )
    if not shell_sources:
        return {}

    results = {
        "column_shell_readout_only": evaluate_pcn(
            mask_output_input_sources(params, structure, shell_sources),
            structure,
            loader,
            config,
            rng_key,
        )
    }
    for shell_name in SHELL_NAMES:
        shell_only_sources = tuple(
            source
            for source in shell_sources
            if is_column_shell_pool_for_shell(source, shell_name)
        )
        shell_without_sources = tuple(
            source
            for source in shell_sources
            if not is_column_shell_pool_for_shell(source, shell_name)
        )
        if shell_without_sources:
            results[f"column_shell_readout_without_{shell_name}"] = evaluate_pcn(
                mask_output_input_sources(params, structure, shell_without_sources),
                structure,
                loader,
                config,
                rng_key,
            )
        if shell_only_sources:
            results[f"column_shell_readout_{shell_name}_only"] = evaluate_pcn(
                mask_output_input_sources(params, structure, shell_only_sources),
                structure,
                loader,
                config,
                rng_key,
            )
    return results


def mask_column_shell_bridge_inputs(
    params: GraphParams,
    structure: GraphStructure,
    bridge_sources: Tuple[str, ...],
    shell_name: str,
    keep_shell: bool,
) -> GraphParams:
    """
    Mask bridge input edges by shell identity.

    `bridge_sources` are the bridge nodes connected to `output`. Each bridge
    receives pooled shell vectors from one column. The masked parameter copy
    keeps either one shell's bridge inputs or all other shell inputs.
    """
    masked = params
    for bridge_source in bridge_sources:
        bridge_input_sources = node_input_edge_sources(structure, bridge_source)
        context_sources = tuple(
            source
            for source in bridge_input_sources
            if is_outer_shell_context_node(source)
        )
        if context_sources:
            masked = mask_outer_shell_context_inputs(
                masked,
                structure,
                context_sources,
                shell_name,
                keep_shell=keep_shell,
            )
        kept_sources = tuple(
            source
            for source in bridge_input_sources
            if (
                is_column_shell_pool_for_shell(source, shell_name) == keep_shell
                or is_outer_shell_context_node(source)
            )
        )
        masked = mask_node_input_sources(
            masked,
            structure,
            bridge_source,
            kept_sources,
        )
    return masked


def evaluate_column_shell_bridge_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate shell dependence inside the per-column shell bridge path.

    The bridge path is a Gaussian predictive-coding latent that receives all
    pooled shells from one column. These ablations keep only bridge outputs at
    the classifier, then mask bridge inputs shell by shell.
    """
    output_sources = output_input_edge_sources(structure)
    bridge_sources = tuple(
        sorted(source for source in output_sources if is_column_shell_bridge_node(source))
    )
    if not bridge_sources:
        return {}

    bridge_only_params = mask_output_input_sources(params, structure, bridge_sources)
    results = {
        "column_shell_bridge_only": evaluate_pcn(
            bridge_only_params,
            structure,
            loader,
            config,
            rng_key,
        )
    }
    for shell_name in SHELL_NAMES:
        without_shell_params = mask_output_input_sources(
            mask_column_shell_bridge_inputs(
                params,
                structure,
                bridge_sources,
                shell_name,
                keep_shell=False,
            ),
            structure,
            bridge_sources,
        )
        shell_only_params = mask_output_input_sources(
            mask_column_shell_bridge_inputs(
                params,
                structure,
                bridge_sources,
                shell_name,
                keep_shell=True,
            ),
            structure,
            bridge_sources,
        )
        results[f"column_shell_bridge_without_{shell_name}"] = evaluate_pcn(
            without_shell_params,
            structure,
            loader,
            config,
            rng_key,
        )
        results[f"column_shell_bridge_{shell_name}_only"] = evaluate_pcn(
            shell_only_params,
            structure,
            loader,
            config,
            rng_key,
        )
    return results


def mask_outer_shell_context_inputs(
    params: GraphParams,
    structure: GraphStructure,
    context_sources: Tuple[str, ...],
    shell_name: str,
    keep_shell: bool,
) -> GraphParams:
    """
    Mask inputs to outer-shell context latents by shell identity.

    `context_sources` are outer-shell context nodes connected to `output`.
    Each context node receives pooled shell vectors from one column and emits an
    outer-shell-width latent. The masked parameter copy keeps either one shell's
    context inputs or all other shell inputs.
    """
    masked = params
    for context_source in context_sources:
        context_input_sources = node_input_edge_sources(structure, context_source)
        kept_sources = tuple(
            source
            for source in context_input_sources
            if is_column_shell_pool_for_shell(source, shell_name) == keep_shell
        )
        masked = mask_node_input_sources(
            masked,
            structure,
            context_source,
            kept_sources,
        )
    return masked


def evaluate_outer_shell_context_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate shell dependence inside the outer-shell context path.

    The context path is a Gaussian predictive-coding latent that receives all
    pooled shells from one column and emits only an outer-shell-width vector.
    These ablations keep only context outputs at the classifier, then mask
    context inputs shell by shell.
    """
    output_sources = output_input_edge_sources(structure)
    context_sources = tuple(
        sorted(source for source in output_sources if is_outer_shell_context_node(source))
    )
    if not context_sources:
        return {}

    context_only_params = mask_output_input_sources(params, structure, context_sources)
    results = {
        "outer_shell_context_only": evaluate_pcn(
            context_only_params,
            structure,
            loader,
            config,
            rng_key,
        )
    }
    for shell_name in SHELL_NAMES:
        without_shell_params = mask_output_input_sources(
            mask_outer_shell_context_inputs(
                params,
                structure,
                context_sources,
                shell_name,
                keep_shell=False,
            ),
            structure,
            context_sources,
        )
        shell_only_params = mask_output_input_sources(
            mask_outer_shell_context_inputs(
                params,
                structure,
                context_sources,
                shell_name,
                keep_shell=True,
            ),
            structure,
            context_sources,
        )
        results[f"outer_shell_context_without_{shell_name}"] = evaluate_pcn(
            without_shell_params,
            structure,
            loader,
            config,
            rng_key,
        )
        results[f"outer_shell_context_{shell_name}_only"] = evaluate_pcn(
            shell_only_params,
            structure,
            loader,
            config,
            rng_key,
        )
    return results


def evaluate_outer_shell_context_evidence_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate shell dependence inside the class-shaped context-evidence path.

    The evidence path receives all active outer-shell context latents and feeds
    one class-width latent into `output`. These ablations keep only that
    evidence latent at `output`, then mask context inputs by source shell.
    """
    output_sources = output_input_edge_sources(structure)
    evidence_source = OUTER_SHELL_CONTEXT_EVIDENCE_NODE
    if evidence_source not in output_sources:
        return {}

    evidence_input_sources = tuple(
        sorted(
            source
            for source in node_input_edge_sources(structure, evidence_source)
            if is_outer_shell_context_node(source)
        )
    )
    if not evidence_input_sources:
        return {}

    results = {
        "outer_shell_context_evidence_only": evaluate_pcn(
            mask_output_input_sources(params, structure, (evidence_source,)),
            structure,
            loader,
            config,
            rng_key,
        )
    }
    for shell_name in SHELL_NAMES:
        without_shell_params = mask_output_input_sources(
            mask_outer_shell_context_inputs(
                params,
                structure,
                evidence_input_sources,
                shell_name,
                keep_shell=False,
            ),
            structure,
            (evidence_source,),
        )
        shell_only_params = mask_output_input_sources(
            mask_outer_shell_context_inputs(
                params,
                structure,
                evidence_input_sources,
                shell_name,
                keep_shell=True,
            ),
            structure,
            (evidence_source,),
        )
        results[f"outer_shell_context_evidence_without_{shell_name}"] = evaluate_pcn(
            without_shell_params,
            structure,
            loader,
            config,
            rng_key,
        )
        results[f"outer_shell_context_evidence_{shell_name}_only"] = evaluate_pcn(
            shell_only_params,
            structure,
            loader,
            config,
            rng_key,
        )
    return results


def mask_column_shell_path_inputs(
    params: GraphParams,
    structure: GraphStructure,
    shell_name: str,
    keep_shell: bool,
) -> GraphParams:
    """
    Mask direct and bridged per-column shell paths with one shell criterion.

    Direct paths are output edges from pooled `(column, shell)` vectors. Bridged
    paths are output edges from per-column shell bridge nodes, with bridge input
    edges masked by shell identity before the classifier mask is applied.
    """
    output_sources = output_input_edge_sources(structure)
    direct_sources = tuple(
        sorted(source for source in output_sources if is_column_shell_pool_node(source))
    )
    bridge_sources = tuple(
        sorted(source for source in output_sources if is_column_shell_bridge_node(source))
    )
    kept_direct_sources = tuple(
        source
        for source in direct_sources
        if is_column_shell_pool_for_shell(source, shell_name) == keep_shell
    )

    masked = params
    if bridge_sources:
        masked = mask_column_shell_bridge_inputs(
            masked,
            structure,
            bridge_sources,
            shell_name,
            keep_shell=keep_shell,
        )
    return mask_output_input_sources(
        masked,
        structure,
        (*kept_direct_sources, *bridge_sources),
    )


def evaluate_column_shell_path_ablations(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate full per-column shell evidence by masking direct and bridged paths.

    The baseline keeps only per-column shell evidence at `output`: direct pooled
    shell edges and shell bridge edges. Each lesion then keeps or drops one shell
    from both routes at the same time.
    """
    output_sources = output_input_edge_sources(structure)
    direct_sources = tuple(
        sorted(source for source in output_sources if is_column_shell_pool_node(source))
    )
    bridge_sources = tuple(
        sorted(source for source in output_sources if is_column_shell_bridge_node(source))
    )
    shell_path_sources = (*direct_sources, *bridge_sources)
    if not shell_path_sources:
        return {}

    results = {
        "column_shell_paths_only": evaluate_pcn(
            mask_output_input_sources(params, structure, shell_path_sources),
            structure,
            loader,
            config,
            rng_key,
        )
    }
    for shell_name in SHELL_NAMES:
        results[f"column_shell_paths_without_{shell_name}"] = evaluate_pcn(
            mask_column_shell_path_inputs(
                params,
                structure,
                shell_name,
                keep_shell=False,
            ),
            structure,
            loader,
            config,
            rng_key,
        )
        results[f"column_shell_paths_{shell_name}_only"] = evaluate_pcn(
            mask_column_shell_path_inputs(
                params,
                structure,
                shell_name,
                keep_shell=True,
            ),
            structure,
            loader,
            config,
            rng_key,
        )
    return results


def print_ablation_results(title: str, results: Dict[str, Dict[str, float]]) -> None:
    """Print readout ablation metrics in a compact table."""
    print("\n" + title)
    print("-" * len(title))
    for case_name, metrics in results.items():
        accuracy = float(metrics.get("accuracy", 0.0))
        energy = float(metrics.get("energy", 0.0))
        print(f"  {case_name:32s} acc={accuracy:.4f} energy={energy:.4f}")


def print_shell_norms(title: str, stats: Dict[str, Dict[str, float]]) -> None:
    """Print shell L2 norm diagnostics."""
    if not stats:
        return
    print("\n" + title)
    print("-" * len(title))
    for shell_name in SHELL_NAMES:
        shell_stats = stats[shell_name]
        print(
            f"  {shell_name:12s} width={int(shell_stats['width']):2d} "
            f"mean_l2={shell_stats['mean_l2']:.4f} "
            f"std_l2={shell_stats['std_l2']:.4f}"
        )


def print_composer_attention(title: str, rows: Dict[str, Dict[str, float]]) -> None:
    """Print the shell composer's learned component attention table."""
    if not rows:
        return
    print("\n" + title)
    print("-" * len(title))
    header = "  column       " + " ".join(f"{shell_name:>13s}" for shell_name in SHELL_NAMES)
    print(header)
    for column_name in sorted(rows):
        values = rows[column_name]
        pieces = " ".join(
            f"{values.get(shell_name, 0.0):13.6f}" for shell_name in SHELL_NAMES
        )
        print(f"  {column_name:10s} {pieces}")


def print_scalar_diagnostics(title: str, values: Dict[str, float]) -> None:
    """Print a sorted scalar diagnostic table."""
    if not values:
        return
    print("\n" + title)
    print("-" * len(title))
    for key, value in sorted(values.items()):
        print(f"  {key:40s} {value:.6f}")


def make_classifier_energy(label_smoothing: float, weight: float = 1.0):
    """
    Create the weighted cross-entropy energy used by CIFAR-10 classifier heads.

    `weight` multiplies the per-sample class energy. The main classifier uses
    weight 1.0. The column teacher head uses a smaller weight because it is an
    auxiliary target attached to the shared column pathway.
    """
    return WeightedLabelSmoothedCrossEntropyEnergy(
        weight=weight,
        smoothing=label_smoothing,
        num_classes=10,
    )


def make_column_gaussian_energy(args):
    """
    Create the local Gaussian energy used by columnar predictive-coding nodes.

    `column_gaussian_energy_mode` selects how local Gaussian prediction errors
    are scaled. The summed mode keeps FabricPC's historical behavior. The mean
    mode divides by every non-batch latent element. The spatial-reference mode
    divides only by replicated spatial or token sites beyond the historical
    reference grid.
    """
    if args.column_gaussian_energy_mode == "mean":
        return MeanSquaredGaussianEnergy(precision=args.column_gaussian_precision)
    if args.column_gaussian_energy_mode == "spatial_reference":
        return SpatialReferenceGaussianEnergy(
            precision=args.column_gaussian_precision,
            reference_sites=args.column_gaussian_reference_sites,
        )
    return GaussianEnergy(precision=args.column_gaussian_precision)


def batch_to_task_dict(batch_data: Any) -> Dict[str, jnp.ndarray]:
    """Convert CIFAR loader batches into FabricPC task-key arrays."""
    if isinstance(batch_data, (list, tuple)):
        return {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
    if isinstance(batch_data, dict):
        return {key: jnp.array(value) for key, value in batch_data.items()}
    raise ValueError(f"Unsupported batch format: {type(batch_data)}")


def add_column_teacher_targets(
    batch: Dict[str, Any],
    shell_target_keys: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    """
    Add auxiliary class targets clamped to column and shell teacher heads.

    Each auxiliary target uses the same one-hot CIFAR-10 label tensor as `y`.
    During training, FabricPC clamps every batch key present in the graph task
    map, so these targets make the column branch a supervised predictive-coding
    path.
    """
    if "y" not in batch:
        raise ValueError("Column teacher batches require a 'y' label")
    updated = dict(batch)
    updated.setdefault(COLUMN_TEACHER_TARGET, batch["y"])
    for target_key in shell_target_keys:
        updated.setdefault(target_key, batch["y"])
    return updated


class ColumnTeacherTargetLoader:
    """
    Loader wrapper that duplicates CIFAR-10 labels for the column teacher head.

    The underlying CIFAR loader is unchanged. This wrapper only changes the
    batch task keys passed into FabricPC training.
    """

    def __init__(
        self,
        base_loader: Any,
        shell_target_keys: Tuple[str, ...] = (),
    ):
        self.base_loader = base_loader
        self.shell_target_keys = shell_target_keys

    def __len__(self) -> int:
        return len(self.base_loader)

    def __iter__(self):
        for batch_data in self.base_loader:
            yield add_column_teacher_targets(
                batch_to_task_dict(batch_data),
                shell_target_keys=self.shell_target_keys,
            )


def evaluate_output_node(
    params: GraphParams,
    structure: GraphStructure,
    output_node: str,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Evaluate a named classifier node against CIFAR-10 labels.

    The graph structure is unchanged except that the task-map key `y` points at
    `output_node` for evaluation, so FabricPC reads predictions from that node.
    """
    if output_node not in structure.nodes:
        raise ValueError(f"Unknown output node: {output_node}")
    eval_structure = structure._replace(
        task_map={**structure.task_map, "y": output_node}
    )
    return evaluate_pcn(params, eval_structure, loader, config, rng_key)


def evaluate_shell_teacher_heads(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """Evaluate each configured shell-local classifier against CIFAR-10 labels."""
    results = {}
    for shell_name in SHELL_NAMES:
        node_name = shell_teacher_node_name(shell_name)
        if node_name in structure.nodes:
            results[node_name] = evaluate_output_node(
                params,
                structure,
                node_name,
                loader,
                config,
                rng_key,
            )
    return results


def evaluate_column_shell_teacher_heads(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """Evaluate configured per-column shell classifiers against CIFAR-10 labels."""
    results = {}
    for node_name in sorted(structure.nodes):
        if is_column_shell_teacher_node(node_name):
            results[node_name] = evaluate_output_node(
                params,
                structure,
                node_name,
                loader,
                config,
                rng_key,
            )
    return results


def evaluate_outer_shell_context_teacher_head(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the context-only classifier against CIFAR-10 labels."""
    if OUTER_SHELL_CONTEXT_TEACHER_NODE not in structure.nodes:
        return {}
    return {
        OUTER_SHELL_CONTEXT_TEACHER_NODE: evaluate_output_node(
            params,
            structure,
            OUTER_SHELL_CONTEXT_TEACHER_NODE,
            loader,
            config,
            rng_key,
        )
    }


def evaluate_outer_shell_context_evidence_node(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the class-width context-evidence latent against CIFAR-10 labels."""
    if OUTER_SHELL_CONTEXT_EVIDENCE_NODE not in structure.nodes:
        return {}
    return {
        OUTER_SHELL_CONTEXT_EVIDENCE_NODE: evaluate_output_node(
            params,
            structure,
            OUTER_SHELL_CONTEXT_EVIDENCE_NODE,
            loader,
            config,
            rng_key,
        )
    }


def evaluate_outer_shell_context_evidence_teacher_head(
    params: GraphParams,
    structure: GraphStructure,
    loader,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the classifier head fed by the context-evidence latent."""
    if OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE not in structure.nodes:
        return {}
    return {
        OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE: evaluate_output_node(
            params,
            structure,
            OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE,
            loader,
            config,
            rng_key,
        )
    }


# Model configurations with explicit stage channel counts for stage taps
MODEL_CONFIGS = {
    "tiny": {
        "stem_channels": 16,
        # (channels, first_stride, num_blocks)
        "stages": [(16, 1, 1), (32, 2, 1), (64, 2, 1)],
        # Stage output channels for taps (after each stage completes)
        "stage_channels": [16, 32, 64],
    },
    "resnet18": {
        "stem_channels": 32,
        "stages": [(32, 1, 2), (64, 2, 2), (128, 2, 2), (256, 2, 2)],
        "stage_channels": [32, 64, 128, 256],
    },
}


def get_activation(name):
    factories = {
        "relu": ReLUActivation,
        "tanh": TanhActivation,
        "gelu": GeluActivation,
        "leaky_relu": lambda: LeakyReLUActivation(alpha=0.1),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation: {name}")
    return factories[name]()


def strided_same_dim(size, stride):
    return int(math.ceil(size / stride))


def make_residual_block(prev_node, channels, stride, block_name, weight_init, activation):
    """Create one predictive-coding residual block."""
    in_h, in_w, in_channels = prev_node.shape
    out_h = strided_same_dim(in_h, stride)
    out_w = strided_same_dim(in_w, stride)

    conv_a = ConvNode(
        shape=(out_h, out_w, channels),
        name=f"{block_name}_conv_a",
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
    )
    conv_b = ConvNode(
        shape=(out_h, out_w, channels),
        name=f"{block_name}_conv_b",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
    )
    skip = SkipConnection(shape=(out_h, out_w, channels), name=f"{block_name}_skip_sum")

    nodes = [conv_a, conv_b, skip]
    edges = [
        Edge(source=prev_node, target=conv_a.slot("in")),
        Edge(source=conv_a, target=conv_b.slot("in")),
        Edge(source=conv_b, target=skip.slot("in")),
    ]

    if stride != 1 or in_channels != channels:
        projection = ConvNode(
            shape=(out_h, out_w, channels),
            name=f"{block_name}_skip_proj",
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
        )
        nodes.append(projection)
        edges.append(Edge(source=prev_node, target=projection.slot("in")))
        edges.append(Edge(source=projection, target=skip.slot("in")))
    else:
        edges.append(Edge(source=prev_node, target=skip.slot("in")))

    return nodes, edges, skip


def build_support_mask(
    mode: str,
    num_columns: int,
    num_shared: int,
    active_nonshared: int,
    seed: int,
) -> Tuple[float, ...]:
    """Build column activation mask."""
    if num_shared > num_columns:
        raise ValueError("num_shared cannot exceed num_columns")

    mask = [0.0] * num_columns
    if mode == "all_active":
        return tuple([1.0] * num_columns)

    for idx in range(num_shared):
        mask[idx] = 1.0

    selectable = list(range(num_shared, num_columns))
    if active_nonshared > len(selectable):
        raise ValueError("active_nonshared exceeds available non-shared columns")

    if mode == "first_sparse":
        selected = selectable[:active_nonshared]
    elif mode == "random_sparse":
        key = jax.random.PRNGKey(seed)
        perm = list(map(int, jax.random.permutation(key, jnp.asarray(selectable))))
        selected = perm[:active_nonshared]
    else:
        raise ValueError(f"Unknown column mode: {mode}")

    for idx in selected:
        mask[idx] = 1.0
    return tuple(mask)


def resolve_column_target_grid(
    column_grid: str,
    stage2_shape: Tuple[int, int, int],
    stage3_shape: Tuple[int, int, int],
    stage4_shape: Tuple[int, int, int],
) -> Tuple[int, int]:
    """Return the token grid selected for depth-spanning column inputs."""
    stage_grids = {
        "stage2": stage2_shape[:2],
        "stage3": stage3_shape[:2],
        "stage4": stage4_shape[:2],
    }
    if column_grid not in stage_grids:
        raise ValueError(
            f"Unknown column_grid {column_grid!r}; expected one of "
            f"{', '.join(COLUMN_GRID_CHOICES)}"
        )
    return tuple(int(value) for value in stage_grids[column_grid])


def build_depth_spanning_graph(args):
    """
    Build a PC graph with depth-spanning columns.

    The ResNet backbone is built with stage outputs tracked. Stage taps
    tokenize each stage's output, and depth-spanning columns receive
    skip connections from all stages.
    """
    if args.outer_shell_context_shell_prediction_weight < 0.0:
        raise ValueError("--outer_shell_context_shell_prediction_weight must be >= 0")
    if (
        args.outer_shell_context_shell_prediction_weight > 0.0
        and not args.outer_shell_context
    ):
        raise ValueError(
            "--outer_shell_context_shell_prediction_weight requires "
            "--outer_shell_context"
        )
    if args.outer_shell_context_to_bridge and not args.outer_shell_context:
        raise ValueError("--outer_shell_context_to_bridge requires --outer_shell_context")
    if args.outer_shell_context_to_bridge and not args.column_shell_bridge:
        raise ValueError("--outer_shell_context_to_bridge requires --column_shell_bridge")
    if args.column_gaussian_precision < 0.0:
        raise ValueError("--column_gaussian_precision must be >= 0")
    if args.column_gaussian_reference_sites <= 0.0:
        raise ValueError("--column_gaussian_reference_sites must be > 0")

    model_config = MODEL_CONFIGS[args.model]
    weight_init = MuPCInitializer()
    activation = get_activation(args.activation)
    shell_teacher_weights = parse_shell_teacher_weights(args.shell_teacher_weights)
    column_shell_teacher_weights = parse_shell_teacher_weights(
        args.column_shell_teacher_weights,
        flag_name="--column_shell_teacher_weights",
    )
    shell_inhibition_strengths = parse_shell_inhibition_strengths(
        args.shell_inhibition_strengths
    )
    shell_inhibition_tuple = tuple(
        shell_inhibition_strengths[shell_name] for shell_name in SHELL_NAMES
    )
    shell_evidence_cascade_scale = parse_shell_evidence_cascade_scale(
        args.shell_evidence_cascade_scale
    )
    column_gaussian_energy = make_column_gaussian_energy(args)

    # Input
    image = IdentityNode(shape=(32, 32, 3), name="input")

    # Stem
    stem_channels = model_config["stem_channels"]
    stem = ConvNode(
        shape=(32, 32, stem_channels),
        name="stem",
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
    )

    nodes = [image, stem]
    edges = [Edge(source=image, target=stem.slot("in"))]
    prev = stem

    # Build ResNet stages, tracking the output of each stage
    stage_outputs = []  # Will hold the final node of each stage
    num_stages = len(model_config["stages"])

    for stage_idx, (channels, first_stride, num_blocks) in enumerate(
        model_config["stages"], start=1
    ):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            block_nodes, block_edges, prev = make_residual_block(
                prev_node=prev,
                channels=channels,
                stride=stride,
                block_name=f"s{stage_idx}b{block_idx + 1}",
                weight_init=weight_init,
                activation=activation,
            )
            nodes.extend(block_nodes)
            edges.extend(block_edges)

        # Track the output of this stage
        stage_outputs.append(prev)

    # For depth-spanning columns, we need at least 3 stages
    # Use the last 3 stages if there are more than 3
    if len(stage_outputs) < 3:
        raise ValueError(
            f"Depth-spanning columns require at least 3 stages, got {len(stage_outputs)}"
        )

    # Get the last 3 stages (or all if exactly 3)
    stage2_out = stage_outputs[-3]  # Third from last
    stage3_out = stage_outputs[-2]  # Second from last
    stage4_out = stage_outputs[-1]  # Last stage

    # Get channel counts for the stages we're using
    stage_channels = model_config["stage_channels"]
    stage2_channels = stage_channels[-3]
    stage3_channels = stage_channels[-2]
    stage4_channels = stage_channels[-1]

    # Determine the shared token grid used by every stage tap and column.
    target_grid = resolve_column_target_grid(
        args.column_grid,
        stage2_out.shape,
        stage3_out.shape,
        stage4_out.shape,
    )
    num_tokens = target_grid[0] * target_grid[1]

    print(f"Stage outputs: stage2={stage2_out.shape}, stage3={stage3_out.shape}, stage4={stage4_out.shape}")
    print(f"Column grid source: {args.column_grid}")
    print(f"Target grid: {target_grid}, tokens: {num_tokens}")

    # Create stage taps (tokenizers)
    stage2_tap = create_stage_tap(
        name="stage2_tap",
        source_channels=stage2_channels,
        embed_dim=args.embed_dim,
        target_grid=target_grid,
        add_pos_embed=True,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
        energy=column_gaussian_energy,
    )
    stage3_tap = create_stage_tap(
        name="stage3_tap",
        source_channels=stage3_channels,
        embed_dim=args.embed_dim,
        target_grid=target_grid,
        add_pos_embed=True,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
        energy=column_gaussian_energy,
    )
    stage4_tap = create_stage_tap(
        name="stage4_tap",
        source_channels=stage4_channels,
        embed_dim=args.embed_dim,
        target_grid=target_grid,
        add_pos_embed=True,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
        energy=column_gaussian_energy,
    )
    stage4_pool = create_global_pool(
        name="stage4_pool",
        source_channels=stage4_channels,
        embed_dim=args.embed_dim,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
        energy=column_gaussian_energy,
    )

    nodes.extend([stage2_tap, stage3_tap, stage4_tap, stage4_pool])
    edges.extend([
        Edge(source=stage2_out, target=stage2_tap.slot("in")),
        Edge(source=stage3_out, target=stage3_tap.slot("in")),
        Edge(source=stage4_out, target=stage4_tap.slot("in")),
        Edge(source=stage4_out, target=stage4_pool.slot("in")),
    ])

    # Create depth-spanning columns
    columns = []
    for idx in range(args.num_columns):
        col = create_depth_spanning_column(
            name=f"col_{idx:02d}",
            input_dim=args.embed_dim,
            output_dim=args.embed_dim,
            microcolumn_dim=args.microcolumn_dim,
            grid_size=target_grid,
            hidden_activation=args.column_activation,
            shell_evidence_cascade_scale=shell_evidence_cascade_scale,
            shell_inhibition_strengths=shell_inhibition_tuple,
            apply_layer_norm=args.layer_norm_tokens,
            fix_ln_gamma=args.fix_ln_gamma,
            energy=column_gaussian_energy,
        )
        columns.append(col)

        # Connect stage taps to column slots
        edges.extend([
            Edge(source=stage2_tap, target=col.slot("stage2")),
            Edge(source=stage3_tap, target=col.slot("stage3")),
            Edge(source=stage4_tap, target=col.slot("stage4")),
            Edge(source=stage4_pool, target=col.slot("stage4_pool")),
        ])

    nodes.extend(columns)

    # Build support mask for column selection
    support_mask = build_support_mask(
        args.column_mode,
        args.num_columns,
        args.num_shared,
        args.active_nonshared,
        args.seed,
    )
    active_column_indices = tuple(
        idx for idx, value in enumerate(support_mask) if value > 0.0
    )

    # Combiner. The shell_attention mode preserves column and shell identity
    # inside one predictive-coding composer node before column_pool.
    if args.combiner == "shell_attention":
        combiner = ColumnShellComposerNode(
            shape=(num_tokens, args.embed_dim),
            name="combiner",
            num_columns=args.num_columns,
            support_mask=support_mask,
            energy=column_gaussian_energy,
        )
    else:
        combiner = MaskedColumnCombinerNode(
            shape=(num_tokens, args.embed_dim),
            name="combiner",
            num_columns=args.num_columns,
            support_mask=support_mask,
            combination=args.combiner,
            energy=column_gaussian_energy,
        )
    nodes.append(combiner)

    for col in columns:
        edges.append(Edge(source=col, target=combiner.slot("in")))

    # Raw pooled readout, combined classifier, and column-only teacher head.
    column_pool = AvgPool(
        shape=(args.embed_dim,),
        name="column_pool",
        global_pool=True,
        energy=column_gaussian_energy,
    )
    output = Linear(
        shape=(10,),
        name="output",
        activation=SoftmaxActivation(),
        energy=make_classifier_energy(args.label_smoothing),
        flatten_input=True,
        weight_init=XavierInitializer(),
    )
    column_teacher_output = Linear(
        shape=(10,),
        name=COLUMN_TEACHER_NODE,
        activation=SoftmaxActivation(),
        energy=make_classifier_energy(args.label_smoothing, args.column_teacher_weight),
        flatten_input=True,
        weight_init=XavierInitializer(),
    )

    nodes.extend([column_pool, output, column_teacher_output])
    edges.extend([
        Edge(source=combiner, target=column_pool.slot("in")),
        Edge(source=column_pool, target=output.slot("in")),
        Edge(source=column_pool, target=column_teacher_output.slot("in")),
    ])

    shell_task_map = {}
    shell_slices = get_shell_slices(args.embed_dim)
    for shell_name in SHELL_NAMES:
        shell_weight = shell_teacher_weights[shell_name]
        if shell_weight <= 0.0:
            continue

        start, end = shell_slices[shell_name]
        shell_slice = FeatureSliceNode(
            shape=(end - start,),
            name=shell_slice_node_name(shell_name),
            start=start,
            end=end,
            energy=column_gaussian_energy,
        )
        shell_teacher_output = Linear(
            shape=(10,),
            name=shell_teacher_node_name(shell_name),
            activation=SoftmaxActivation(),
            energy=make_classifier_energy(args.label_smoothing, shell_weight),
            flatten_input=True,
            weight_init=XavierInitializer(),
        )
        nodes.extend([shell_slice, shell_teacher_output])
        edges.extend([
            Edge(source=column_pool, target=shell_slice.slot("in")),
            Edge(source=shell_slice, target=shell_teacher_output.slot("in")),
        ])
        shell_task_map[shell_teacher_target_name(shell_name)] = shell_teacher_output

    column_shell_task_map = {}
    outer_shell_context_nodes = []
    for column_idx in active_column_indices:
        column = columns[column_idx]
        column_shell_pools = []
        column_shell_pools_by_shell = {}
        outer_context = None
        for shell_name in SHELL_NAMES:
            shell_weight = column_shell_teacher_weights[shell_name]
            needs_shell_pool = (
                shell_weight > 0.0
                or args.column_shell_readout
                or args.column_shell_bridge
                or args.outer_shell_context
            )
            if not needs_shell_pool:
                continue

            start, end = shell_slices[shell_name]
            shell_slice = FeatureSliceNode(
                shape=(num_tokens, end - start),
                name=column_shell_slice_node_name(column_idx, shell_name),
                start=start,
                end=end,
                energy=column_gaussian_energy,
            )
            shell_pool = AvgPool(
                shape=(end - start,),
                name=column_shell_pool_node_name(column_idx, shell_name),
                global_pool=True,
                energy=column_gaussian_energy,
            )
            shell_teacher_output = Linear(
                shape=(10,),
                name=column_shell_teacher_node_name(column_idx, shell_name),
                activation=SoftmaxActivation(),
                energy=make_classifier_energy(args.label_smoothing, shell_weight),
                flatten_input=True,
                weight_init=XavierInitializer(),
            )
            nodes.extend([shell_slice, shell_pool])
            edges.extend([
                Edge(source=column, target=shell_slice.slot("in")),
                Edge(source=shell_slice, target=shell_pool.slot("in")),
            ])
            column_shell_pools.append(shell_pool)
            column_shell_pools_by_shell[shell_name] = shell_pool
            if args.column_shell_readout:
                edges.append(Edge(source=shell_pool, target=output.slot("in")))
            if shell_weight > 0.0:
                nodes.append(shell_teacher_output)
                edges.append(
                    Edge(source=shell_pool, target=shell_teacher_output.slot("in"))
                )
                column_shell_task_map[
                    column_shell_teacher_target_name(column_idx, shell_name)
                ] = shell_teacher_output

        if args.outer_shell_context:
            outer_start, outer_end = shell_slices["outer_shell"]
            outer_context = Linear(
                shape=(outer_end - outer_start,),
                name=outer_shell_context_node_name(column_idx),
                activation=IdentityActivation(),
                flatten_input=False,
                weight_init=XavierInitializer(),
                energy=column_gaussian_energy,
            )
            nodes.append(outer_context)
            outer_shell_context_nodes.append(outer_context)
            for shell_pool in column_shell_pools:
                edges.append(Edge(source=shell_pool, target=outer_context.slot("in")))
            edges.append(Edge(source=outer_context, target=output.slot("in")))
            if args.outer_shell_context_shell_prediction_weight > 0.0:
                for shell_name, shell_pool in column_shell_pools_by_shell.items():
                    context_prediction = ShellContextPredictionNode(
                        shape=shell_pool.shape,
                        name=shell_context_prediction_node_name(
                            column_idx,
                            shell_name,
                        ),
                        objective_weight=(
                            args.outer_shell_context_shell_prediction_weight
                        ),
                        weight_init=XavierInitializer(),
                    )
                    nodes.append(context_prediction)
                    edges.extend([
                        Edge(
                            source=shell_pool,
                            target=context_prediction.slot("target"),
                        ),
                        Edge(
                            source=outer_context,
                            target=context_prediction.slot("context"),
                        ),
                    ])

        if args.column_shell_bridge:
            shell_bridge = Linear(
                shape=(args.embed_dim,),
                name=column_shell_bridge_node_name(column_idx),
                activation=IdentityActivation(),
                flatten_input=False,
                weight_init=XavierInitializer(),
                energy=column_gaussian_energy,
            )
            nodes.append(shell_bridge)
            for shell_pool in column_shell_pools:
                edges.append(Edge(source=shell_pool, target=shell_bridge.slot("in")))
            if args.outer_shell_context_to_bridge:
                if outer_context is None:
                    raise ValueError(
                        "--outer_shell_context_to_bridge requires an outer context "
                        f"node for column {column_idx}"
                    )
                edges.append(Edge(source=outer_context, target=shell_bridge.slot("in")))
            edges.append(Edge(source=shell_bridge, target=output.slot("in")))

    outer_shell_context_task_map = {}
    outer_shell_context_evidence = None
    if args.outer_shell_context_evidence:
        if not args.outer_shell_context:
            raise ValueError(
                "--outer_shell_context_evidence requires --outer_shell_context"
            )
        if not outer_shell_context_nodes:
            raise ValueError(
                "Outer-shell context evidence requires at least one context node"
            )
        outer_shell_context_evidence = Linear(
            shape=(10,),
            name=OUTER_SHELL_CONTEXT_EVIDENCE_NODE,
            activation=IdentityActivation(),
            flatten_input=True,
            weight_init=XavierInitializer(),
            energy=column_gaussian_energy,
        )
        nodes.append(outer_shell_context_evidence)
        for outer_context in outer_shell_context_nodes:
            edges.append(
                Edge(
                    source=outer_context,
                    target=outer_shell_context_evidence.slot("in"),
                )
            )

    if args.outer_shell_context_evidence_teacher_weight > 0.0:
        if outer_shell_context_evidence is None:
            raise ValueError(
                "--outer_shell_context_evidence_teacher_weight requires "
                "--outer_shell_context_evidence"
            )
        outer_context_evidence_teacher = Linear(
            shape=(10,),
            name=OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_NODE,
            activation=SoftmaxActivation(),
            energy=make_classifier_energy(
                args.label_smoothing,
                args.outer_shell_context_evidence_teacher_weight,
            ),
            flatten_input=True,
            weight_init=XavierInitializer(),
        )
        nodes.append(outer_context_evidence_teacher)
        edges.append(
            Edge(
                source=outer_shell_context_evidence,
                target=outer_context_evidence_teacher.slot("in"),
            )
        )
        outer_shell_context_task_map[
            OUTER_SHELL_CONTEXT_EVIDENCE_TEACHER_TARGET
        ] = outer_context_evidence_teacher

    if args.outer_shell_context_teacher_weight > 0.0:
        if not args.outer_shell_context:
            raise ValueError(
                "--outer_shell_context_teacher_weight requires --outer_shell_context"
            )
        if not outer_shell_context_nodes:
            raise ValueError(
                "Outer-shell context teacher requires at least one context node"
            )
        outer_context_teacher = Linear(
            shape=(10,),
            name=OUTER_SHELL_CONTEXT_TEACHER_NODE,
            activation=SoftmaxActivation(),
            energy=make_classifier_energy(
                args.label_smoothing,
                args.outer_shell_context_teacher_weight,
            ),
            flatten_input=True,
            weight_init=XavierInitializer(),
        )
        nodes.append(outer_context_teacher)
        for outer_context in outer_shell_context_nodes:
            edges.append(
                Edge(
                    source=outer_context,
                    target=outer_context_teacher.slot("in"),
                )
            )
        outer_shell_context_task_map[
            OUTER_SHELL_CONTEXT_TEACHER_TARGET
        ] = outer_context_teacher

    # Optional bypass: stage4 backbone features go directly to the classifier in
    # parallel with the columnar pathway, so readout ablations can separate
    # backbone-only evidence from column-mediated evidence.
    if args.bypass_columns:
        bypass_pool = AvgPool(
            shape=(stage4_channels,),
            name="bypass_pool",
            global_pool=True,
        )
        nodes.append(bypass_pool)
        edges.append(Edge(source=stage4_out, target=bypass_pool.slot("in")))
        edges.append(Edge(source=bypass_pool, target=output.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(
            x=image,
            y=output,
            column_y=column_teacher_output,
            **shell_task_map,
            **column_shell_task_map,
            **outer_shell_context_task_map,
        ),
        inference=InferenceSGDNormClip(
            eta_infer=args.eta_infer,
            infer_steps=args.infer_steps,
            max_norm=args.infer_max_norm,
        ),
        scaling=MuPCConfig(include_output=False),
    )

    return structure, support_mask


def train_cifar10_depth_spanning(args):
    if args.outer_shell_context_teacher_weight < 0.0:
        raise ValueError("--outer_shell_context_teacher_weight must be >= 0")
    if args.outer_shell_context_evidence_teacher_weight < 0.0:
        raise ValueError("--outer_shell_context_evidence_teacher_weight must be >= 0")
    if args.outer_shell_context_shell_prediction_weight < 0.0:
        raise ValueError("--outer_shell_context_shell_prediction_weight must be >= 0")
    if (
        args.outer_shell_context_shell_prediction_weight > 0.0
        and not args.outer_shell_context
    ):
        raise ValueError(
            "--outer_shell_context_shell_prediction_weight requires "
            "--outer_shell_context"
        )
    if args.outer_shell_context_to_bridge and not args.outer_shell_context:
        raise ValueError("--outer_shell_context_to_bridge requires --outer_shell_context")
    if args.outer_shell_context_to_bridge and not args.column_shell_bridge:
        raise ValueError("--outer_shell_context_to_bridge requires --column_shell_bridge")
    if args.column_gaussian_precision < 0.0:
        raise ValueError("--column_gaussian_precision must be >= 0")
    if args.column_gaussian_reference_sites <= 0.0:
        raise ValueError("--column_gaussian_reference_sites must be > 0")

    print("=" * 70)
    print("CIFAR-10 Depth-Spanning Columnar Architecture")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Activation: {args.activation}")
    print(f"Column activation: {args.column_activation}")
    print(f"Columns: {args.num_columns}")
    print(f"Column mode: {args.column_mode}")
    print(f"Shared columns: {args.num_shared}")
    print(f"Active non-shared: {args.active_nonshared}")
    print(f"Combiner: {args.combiner}")
    print(f"Column grid: {args.column_grid}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Microcolumn dim: {args.microcolumn_dim}")
    shell_slices = get_shell_slices(args.embed_dim)
    shell_widths = ", ".join(
        f"{name}={end - start}" for name, (start, end) in shell_slices.items()
    )
    print(f"Shell widths: {shell_widths}")
    shell_inhibition_strengths = parse_shell_inhibition_strengths(
        args.shell_inhibition_strengths
    )
    shell_inhibition_summary = ", ".join(
        f"{name}={shell_inhibition_strengths[name]:.6g}" for name in SHELL_NAMES
    )
    shell_evidence_cascade_scale = parse_shell_evidence_cascade_scale(
        args.shell_evidence_cascade_scale
    )
    shell_evidence_cascade_summary = ", ".join(
        f"{source}->{target}={scale:.6g}"
        for (source, target), scale in zip(
            zip(SHELL_NAMES[:-1], SHELL_NAMES[1:]),
            shell_evidence_cascade_scale,
        )
    )
    print("Shell evidence cascade: enabled")
    print(f"Shell evidence cascade scale: {shell_evidence_cascade_summary}")
    print(f"Shell inhibition strengths: {shell_inhibition_summary}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    shell_lr_multipliers = parse_shell_lr_multipliers(args.shell_lr_multipliers)
    shell_lr_summary = ", ".join(
        f"{name}={shell_lr_multipliers[name]:.6g}" for name in SHELL_NAMES
    )
    print(f"Shell LR multipliers: {shell_lr_summary}")
    print(f"Inference steps: {args.infer_steps}")
    print(f"Inference eta: {args.eta_infer}")
    print(f"Column Gaussian energy mode: {args.column_gaussian_energy_mode}")
    print(f"Column Gaussian precision: {args.column_gaussian_precision}")
    print(f"Column Gaussian reference sites: {args.column_gaussian_reference_sites}")
    print("Column teacher head: enabled")
    print(f"Column teacher weight: {args.column_teacher_weight}")
    shell_teacher_weights = parse_shell_teacher_weights(args.shell_teacher_weights)
    shell_teacher_summary = ", ".join(
        f"{name}={shell_teacher_weights[name]:.6g}" for name in SHELL_NAMES
    )
    print(f"Shell teacher weights: {shell_teacher_summary}")
    column_shell_teacher_weights = parse_shell_teacher_weights(
        args.column_shell_teacher_weights,
        flag_name="--column_shell_teacher_weights",
    )
    column_shell_teacher_summary = ", ".join(
        f"{name}={column_shell_teacher_weights[name]:.6g}"
        for name in SHELL_NAMES
    )
    print(f"Column shell teacher weights: {column_shell_teacher_summary}")
    print(f"Column shell readout: {'enabled' if args.column_shell_readout else 'disabled'}")
    print(f"Column shell bridge: {'enabled' if args.column_shell_bridge else 'disabled'}")
    print(f"Outer shell context: {'enabled' if args.outer_shell_context else 'disabled'}")
    if args.outer_shell_context:
        print("Outer shell context direct readout: enabled")
    print(
        "Outer shell context to shell bridge: "
        f"{'enabled' if args.outer_shell_context_to_bridge else 'disabled'}"
    )
    print(f"Outer shell context teacher weight: {args.outer_shell_context_teacher_weight}")
    print(
        "Outer shell context shell prediction weight: "
        f"{args.outer_shell_context_shell_prediction_weight}"
    )
    print(
        "Outer shell context evidence: "
        f"{'enabled' if args.outer_shell_context_evidence else 'disabled'}"
    )
    print(
        "Outer shell context evidence teacher weight: "
        f"{args.outer_shell_context_evidence_teacher_weight}"
    )
    print()

    master_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)

    structure, support_mask = build_depth_spanning_graph(args)
    params = initialize_params(structure, graph_key)
    auxiliary_target_keys = tuple(
        key
        for key in structure.task_map
        if key not in ("x", "y", COLUMN_TEACHER_TARGET)
    )

    active_columns = [idx for idx, value in enumerate(support_mask) if value > 0.0]
    print(f"\nActive columns: {active_columns}")
    print(f"Graph: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {total_params:,}")

    print("\nLoading CIFAR-10...")
    train_loader = Cifar10Loader(
        "train[:90%]",
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        tensor_format="NHWC",
    )
    val_loader = Cifar10Loader(
        "train[90%:]",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        tensor_format="NHWC",
    )
    test_loader = Cifar10Loader(
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        tensor_format="NHWC",
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    train_loader_with_teacher = ColumnTeacherTargetLoader(
        train_loader,
        shell_target_keys=auxiliary_target_keys,
    )

    # Energy, column-output, and shell diagnosis: sample batch for analysis
    diag_batch = None
    diag_key = None
    if args.diagnose_energy or args.diagnose_shells:
        diag_batch_raw = next(iter(train_loader_with_teacher))
        diag_batch = batch_to_task_dict(diag_batch_raw)
        diag_key = jax.random.PRNGKey(args.seed + 999)

    if args.diagnose_energy:
        print("\n" + "-" * 70)
        print("Energy Diagnosis (before training)")
        print("-" * 70)
        energy_breakdown = diagnose_energy_breakdown(params, structure, diag_batch, diag_key)
        for key, val in energy_breakdown.items():
            print(f"  {key}: {val:.4f}")

        col_stats = diagnose_column_outputs(params, structure, diag_batch, diag_key)
        print("Column outputs (z_latent statistics):")
        for name in sorted(col_stats):
            s = col_stats[name]
            print(
                f"  {name:18s} mean={s['mean']:+.4f} std={s['std']:.4f} "
                f"min={s['min']:+.4f} max={s['max']:+.4f}"
            )
        print("-" * 70)

    if args.diagnose_shells and diag_batch is not None:
        print_shell_norms(
            "Shell Norms (before training)",
            diagnose_shell_norms(params, structure, diag_batch, diag_key),
        )
        print_scalar_diagnostics(
            "Shell Context Prediction Energy (before training)",
            diagnose_shell_context_prediction_energies(
                params,
                structure,
                diag_batch,
                diag_key,
            ),
        )

    steps_per_epoch = len(train_loader_with_teacher)
    total_steps = max(1, round(args.num_epochs * steps_per_epoch))
    warmup_steps = min(total_steps - 1, int(0.05 * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.adamw(schedule, weight_decay=args.weight_decay),
        scale_updates_by_shell_lr(
            build_shell_lr_multiplier_tree(
                params,
                structure,
                shell_lr_multipliers,
            )
        ),
    )
    train_config = {"num_epochs": args.num_epochs}

    best_val_acc = -1.0
    best_val_epoch = None
    best_params = None
    final_epoch = math.ceil(args.num_epochs)

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        nonlocal best_val_acc, best_val_epoch, best_params
        epoch_num = epoch_idx + 1
        if args.eval_every <= 0:
            return None
        if epoch_num % args.eval_every != 0 and epoch_num != final_epoch:
            return None

        metrics = evaluate_pcn(params, structure, val_loader, config, eval_key)
        val_acc = float(metrics.get("accuracy", 0.0))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch_num
            best_params = params
        print(f"  Epoch {epoch_num}: val_acc={val_acc:.4f}")

        if args.diagnose_energy and diag_batch is not None:
            energy_breakdown = diagnose_energy_breakdown(
                params, structure, diag_batch, diag_key
            )
            ratio = energy_breakdown["E_gauss/E_ce"]
            print(
                f"    E_gauss={energy_breakdown['E_gauss']:.4f} "
                f"E_ce={energy_breakdown['E_ce']:.4f} "
                f"ratio={ratio:.4f} "
                f"combiner={energy_breakdown['combiner']:.4f} "
                f"columns={energy_breakdown['columns']:.4f} "
                f"stage_taps={energy_breakdown['stage_taps']:.4f}"
            )
            col_stats = diagnose_column_outputs(
                params, structure, diag_batch, diag_key
            )
            col_names = sorted(n for n in col_stats if n.startswith("col_"))
            if col_names:
                mean_abs_per_col = [
                    (abs(col_stats[n]["mean"]) + col_stats[n]["std"]) for n in col_names
                ]
                avg_col_magnitude = sum(mean_abs_per_col) / len(mean_abs_per_col)
                print(
                    f"    column |z|≈{avg_col_magnitude:.4f} "
                    f"(per-col std range "
                    f"{min(col_stats[n]['std'] for n in col_names):.4f}–"
                    f"{max(col_stats[n]['std'] for n in col_names):.4f})"
                )
            if "combiner" in col_stats:
                cs = col_stats["combiner"]
                print(
                    f"    combiner z_latent mean={cs['mean']:+.4f} "
                    f"std={cs['std']:.4f}"
                )

            corr = diagnose_column_correlation(
                params, structure, diag_batch, diag_key
            )
            print(
                f"    col-pair |corr|: max={corr['max_abs_offdiag']:.4f} "
                f"mean={corr['mean_abs_offdiag']:.4f}"
            )

            edge_norms = diagnose_output_edge_weight_norms(params, structure)
            if edge_norms:
                pieces = [f"{k}={v:.4f}" for k, v in sorted(edge_norms.items())]
                print(f"    output ||W|| per edge: {' '.join(pieces)}")

        return metrics

    iter_cb = None
    if args.per_batch_energy_log:
        log_every = max(1, int(args.per_batch_energy_log))

        def iter_cb(epoch_idx, batch_idx, energy):
            if batch_idx % log_every == 0:
                print(
                    f"    [iter] ep={epoch_idx + 1} batch={batch_idx} "
                    f"energy={float(energy):.6f}"
                )
            return float(energy)

    print(f"\nTraining for {args.num_epochs} epochs...")
    start_time = time.time()
    final_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader_with_teacher,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=False,
        epoch_callback=epoch_callback,
        iter_callback=iter_cb,
    )
    elapsed = time.time() - start_time

    # Energy + column-output diagnosis: after training
    if args.diagnose_energy:
        print("\n" + "-" * 70)
        print("Energy Diagnosis (after training)")
        print("-" * 70)
        energy_breakdown = diagnose_energy_breakdown(final_params, structure, diag_batch, diag_key)
        for key, val in energy_breakdown.items():
            print(f"  {key}: {val:.4f}")

        col_stats = diagnose_column_outputs(final_params, structure, diag_batch, diag_key)
        print("Column outputs (z_latent statistics):")
        for name in sorted(col_stats):
            s = col_stats[name]
            print(
                f"  {name:18s} mean={s['mean']:+.4f} std={s['std']:.4f} "
                f"min={s['min']:+.4f} max={s['max']:+.4f}"
            )
        print("-" * 70)

    if args.diagnose_shells and diag_batch is not None:
        print_shell_norms(
            "Shell Norms (after training)",
            diagnose_shell_norms(final_params, structure, diag_batch, diag_key),
        )
        print_scalar_diagnostics(
            "Shell Context Prediction Energy (after training)",
            diagnose_shell_context_prediction_energies(
                final_params,
                structure,
                diag_batch,
                diag_key,
            ),
        )

    eval_params = best_params if best_params is not None else final_params
    if best_params is not None:
        print(
            f"\nTraining time: {elapsed:.1f}s"
            f"\nEvaluating best validation params from epoch {best_val_epoch} on test set..."
        )
    else:
        print(f"\nTraining time: {elapsed:.1f}s")
        print("Evaluating final params on test set...")

    ablation_key = jax.random.PRNGKey(args.seed + 2026)
    if args.diagnose_composer:
        print_composer_attention(
            "Composer Attention (selected params)",
            diagnose_composer_attention(eval_params, structure),
        )
        print_scalar_diagnostics(
            "Composer Projection Norms (selected params)",
            diagnose_composer_projection_norms(eval_params, structure),
        )
        print_scalar_diagnostics(
            "Output Edge Weight Norms (selected params)",
            diagnose_output_edge_weight_norms(eval_params, structure),
        )
        val_composer_metrics = evaluate_composer_component_ablations(
            eval_params,
            structure,
            val_loader,
            train_config,
            ablation_key,
        )
        if val_composer_metrics:
            print_ablation_results(
                "Validation Composer Component Lesions",
                val_composer_metrics,
            )

    if args.diagnose_shells and diag_batch is not None:
        print_scalar_diagnostics(
            "Shell Context Prediction Energy (selected params)",
            diagnose_shell_context_prediction_energies(
                eval_params,
                structure,
                diag_batch,
                diag_key,
            ),
        )

    val_ablation_metrics = evaluate_readout_ablations(
        eval_params, structure, val_loader, train_config, ablation_key
    )
    print_ablation_results("Validation Readout Ablations", val_ablation_metrics)
    val_teacher_metrics = evaluate_output_node(
        eval_params,
        structure,
        COLUMN_TEACHER_NODE,
        val_loader,
        train_config,
        ablation_key,
    )
    print_ablation_results(
        "Validation Column Teacher Head",
        {COLUMN_TEACHER_NODE: val_teacher_metrics},
    )
    val_shell_teacher_metrics = evaluate_shell_teacher_heads(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_shell_teacher_metrics:
        print_ablation_results(
            "Validation Shell Teacher Heads",
            val_shell_teacher_metrics,
        )
    val_column_shell_teacher_metrics = evaluate_column_shell_teacher_heads(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_column_shell_teacher_metrics:
        print_ablation_results(
            "Validation Per-Column Shell Teacher Heads",
            val_column_shell_teacher_metrics,
        )
    val_outer_shell_context_teacher_metrics = evaluate_outer_shell_context_teacher_head(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_outer_shell_context_teacher_metrics:
        print_ablation_results(
            "Validation Outer-Shell Context Teacher Head",
            val_outer_shell_context_teacher_metrics,
        )
    val_outer_shell_context_evidence_metrics = (
        evaluate_outer_shell_context_evidence_node(
            eval_params,
            structure,
            val_loader,
            train_config,
            ablation_key,
        )
    )
    if val_outer_shell_context_evidence_metrics:
        print_ablation_results(
            "Validation Outer-Shell Context Evidence",
            val_outer_shell_context_evidence_metrics,
        )
    val_outer_shell_context_evidence_teacher_metrics = (
        evaluate_outer_shell_context_evidence_teacher_head(
            eval_params,
            structure,
            val_loader,
            train_config,
            ablation_key,
        )
    )
    if val_outer_shell_context_evidence_teacher_metrics:
        print_ablation_results(
            "Validation Outer-Shell Context Evidence Teacher Head",
            val_outer_shell_context_evidence_teacher_metrics,
        )
    if args.diagnose_shells:
        val_shell_metrics = evaluate_shell_readout_ablations(
            eval_params, structure, val_loader, train_config, ablation_key
        )
        print_ablation_results("Validation Shell Readout Ablations", val_shell_metrics)
    val_column_shell_readout_metrics = evaluate_column_shell_readout_ablations(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_column_shell_readout_metrics:
        print_ablation_results(
            "Validation Per-Column Shell Readout Ablations",
            val_column_shell_readout_metrics,
        )
    val_column_shell_bridge_metrics = evaluate_column_shell_bridge_ablations(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_column_shell_bridge_metrics:
        print_ablation_results(
            "Validation Per-Column Shell Bridge Ablations",
            val_column_shell_bridge_metrics,
        )
    val_outer_shell_context_metrics = evaluate_outer_shell_context_ablations(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_outer_shell_context_metrics:
        print_ablation_results(
            "Validation Outer-Shell Context Ablations",
            val_outer_shell_context_metrics,
        )
    val_outer_shell_context_evidence_ablation_metrics = (
        evaluate_outer_shell_context_evidence_ablations(
            eval_params,
            structure,
            val_loader,
            train_config,
            ablation_key,
        )
    )
    if val_outer_shell_context_evidence_ablation_metrics:
        print_ablation_results(
            "Validation Outer-Shell Context Evidence Ablations",
            val_outer_shell_context_evidence_ablation_metrics,
        )
    val_column_shell_path_metrics = evaluate_column_shell_path_ablations(
        eval_params,
        structure,
        val_loader,
        train_config,
        ablation_key,
    )
    if val_column_shell_path_metrics:
        print_ablation_results(
            "Validation Combined Column Shell Path Ablations",
            val_column_shell_path_metrics,
        )

    test_ablation_metrics = evaluate_readout_ablations(
        eval_params, structure, test_loader, train_config, ablation_key
    )
    test_teacher_metrics = evaluate_output_node(
        eval_params,
        structure,
        COLUMN_TEACHER_NODE,
        test_loader,
        train_config,
        ablation_key,
    )
    test_shell_teacher_metrics = evaluate_shell_teacher_heads(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_column_shell_teacher_metrics = evaluate_column_shell_teacher_heads(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_outer_shell_context_teacher_metrics = evaluate_outer_shell_context_teacher_head(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_outer_shell_context_evidence_metrics = (
        evaluate_outer_shell_context_evidence_node(
            eval_params,
            structure,
            test_loader,
            train_config,
            ablation_key,
        )
    )
    test_outer_shell_context_evidence_teacher_metrics = (
        evaluate_outer_shell_context_evidence_teacher_head(
            eval_params,
            structure,
            test_loader,
            train_config,
            ablation_key,
        )
    )
    test_column_shell_readout_metrics = evaluate_column_shell_readout_ablations(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_column_shell_bridge_metrics = evaluate_column_shell_bridge_ablations(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_outer_shell_context_metrics = evaluate_outer_shell_context_ablations(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_outer_shell_context_evidence_ablation_metrics = (
        evaluate_outer_shell_context_evidence_ablations(
            eval_params,
            structure,
            test_loader,
            train_config,
            ablation_key,
        )
    )
    test_column_shell_path_metrics = evaluate_column_shell_path_ablations(
        eval_params,
        structure,
        test_loader,
        train_config,
        ablation_key,
    )
    test_metrics = test_ablation_metrics["combined"]
    test_acc = float(test_metrics.get("accuracy", 0.0))

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    if best_params is not None:
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Best Val Epoch: {best_val_epoch}")
    print_ablation_results("Test Readout Ablations", test_ablation_metrics)
    print_ablation_results(
        "Test Column Teacher Head",
        {COLUMN_TEACHER_NODE: test_teacher_metrics},
    )
    if test_shell_teacher_metrics:
        print_ablation_results(
            "Test Shell Teacher Heads",
            test_shell_teacher_metrics,
        )
    if test_column_shell_teacher_metrics:
        print_ablation_results(
            "Test Per-Column Shell Teacher Heads",
            test_column_shell_teacher_metrics,
        )
    if test_outer_shell_context_teacher_metrics:
        print_ablation_results(
            "Test Outer-Shell Context Teacher Head",
            test_outer_shell_context_teacher_metrics,
        )
    if test_outer_shell_context_evidence_metrics:
        print_ablation_results(
            "Test Outer-Shell Context Evidence",
            test_outer_shell_context_evidence_metrics,
        )
    if test_outer_shell_context_evidence_teacher_metrics:
        print_ablation_results(
            "Test Outer-Shell Context Evidence Teacher Head",
            test_outer_shell_context_evidence_teacher_metrics,
        )
    if test_column_shell_readout_metrics:
        print_ablation_results(
            "Test Per-Column Shell Readout Ablations",
            test_column_shell_readout_metrics,
        )
    if test_column_shell_bridge_metrics:
        print_ablation_results(
            "Test Per-Column Shell Bridge Ablations",
            test_column_shell_bridge_metrics,
        )
    if test_outer_shell_context_metrics:
        print_ablation_results(
            "Test Outer-Shell Context Ablations",
            test_outer_shell_context_metrics,
        )
    if test_outer_shell_context_evidence_ablation_metrics:
        print_ablation_results(
            "Test Outer-Shell Context Evidence Ablations",
            test_outer_shell_context_evidence_ablation_metrics,
        )
    if test_column_shell_path_metrics:
        print_ablation_results(
            "Test Combined Column Shell Path Ablations",
            test_column_shell_path_metrics,
        )
    if args.diagnose_shells:
        test_shell_metrics = evaluate_shell_readout_ablations(
            eval_params, structure, test_loader, train_config, ablation_key
        )
        print_ablation_results("Test Shell Readout Ablations", test_shell_metrics)
    return test_acc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="tiny")
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        default="leaky_relu",
    )
    parser.add_argument(
        "--column_activation",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        default="leaky_relu",
    )
    parser.add_argument("--num_columns", type=int, default=4)
    parser.add_argument("--num_shared", type=int, default=2)
    parser.add_argument("--active_nonshared", type=int, default=2)
    parser.add_argument(
        "--column_mode",
        choices=["all_active", "first_sparse", "random_sparse"],
        default="all_active",
    )
    parser.add_argument(
        "--combiner",
        choices=["attention", "sum", "shell_attention"],
        default="sum",
    )
    parser.add_argument(
        "--column_grid",
        choices=COLUMN_GRID_CHOICES,
        default="stage4",
        help=(
            "Backbone stage whose spatial grid defines the token grid used by "
            "all stage taps and depth-spanning columns. The default stage4 "
            "keeps the historical 4x4 ResNet-18 setting; stage3 gives 8x8 "
            "tokens on CIFAR-10 ResNet-18."
        ),
    )
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--microcolumn_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--shell_lr_multipliers",
        type=str,
        default=SHELL_LR_DEFAULT_MULTIPLIERS,
        help=(
            "Comma-separated optimizer update multipliers for hard_kernel, "
            "inner_shell, middle_shell, outer_shell order. These multipliers "
            "are applied after AdamW computes its scheduled update, so they "
            "change learned shell plasticity without changing inference eta."
        ),
    )
    parser.add_argument("--infer_steps", type=int, default=40)
    parser.add_argument("--eta_infer", type=float, default=0.1)
    parser.add_argument("--infer_max_norm", type=float, default=1.0)
    parser.add_argument(
        "--column_gaussian_energy_mode",
        choices=["sum", "mean", "spatial_reference"],
        default="sum",
        help=(
            "Scaling mode for local columnar Gaussian prediction errors. "
            "'sum' keeps FabricPC's summed residual energy, 'mean' divides by "
            "all non-batch latent elements, and 'spatial_reference' divides "
            "only by spatial or token sites beyond a reference grid."
        ),
    )
    parser.add_argument(
        "--column_gaussian_precision",
        type=float,
        default=1.0,
        help=(
            "Scalar precision applied to local Gaussian energies in the "
            "columnar path after the selected Gaussian energy scaling."
        ),
    )
    parser.add_argument(
        "--column_gaussian_reference_sites",
        type=float,
        default=16.0,
        help=(
            "Reference spatial or token site count for "
            "--column_gaussian_energy_mode spatial_reference. The default 16 "
            "preserves the historical ResNet18 CIFAR-10 stage4 4x4 grid."
        ),
    )
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--diagnose_energy",
        action="store_true",
        help="Log per-node-type energy breakdown (E_gauss/E_ce ratio)",
    )
    parser.add_argument(
        "--diagnose_shells",
        action="store_true",
        help=(
            "Log shell-resolved column norms and evaluate shell-slice readout "
            "ablations after training."
        ),
    )
    parser.add_argument(
        "--diagnose_composer",
        action="store_true",
        help=(
            "For shell_attention combiner runs, log composer attention, "
            "composer projection norms, output edge norms, and validation "
            "lesions of one composer component at a time."
        ),
    )
    parser.add_argument(
        "--bypass_columns",
        action="store_true",
        help=(
            "Add a parallel global-pool of stage4 directly into the classifier, "
            "alongside the column pathway. Readout ablations then distinguish "
            "backbone-only evidence from column-mediated evidence."
        ),
    )
    parser.add_argument(
        "--layer_norm_tokens",
        action="store_true",
        help=(
            "Apply LayerNorm along the embed_dim axis at the output of every "
            "stage_tap, stage4_pool, and depth-spanning column. Pins activation "
            "magnitude through the pipeline."
        ),
    )
    parser.add_argument(
        "--fix_ln_gamma",
        action="store_true",
        help=(
            "Make LayerNorm's gamma=1.0 and beta=0.0 non-learnable scalar "
            "constants (only effective with --layer_norm_tokens). Removes the "
            "rescale path that lets training drift between magnitude basins."
        ),
    )
    parser.add_argument(
        "--per_batch_energy_log",
        type=int,
        default=0,
        help=(
            "Log the per-batch total training energy every N batches "
            "(0 = disabled). Useful for localizing collapse moments between "
            "epochs. With 352 batches/epoch, log_every=5 gives ~70 lines/epoch."
        ),
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help=(
            "Label-smoothing factor for the classifier's cross-entropy energy "
            "(0.0 = plain CE; 0.1 is the CIFAR-10 standard). Prevents the loss "
            "from saturating, keeping the gradient signal alive against weight "
            "decay erosion. Implemented in columnar_cl_fabricpc, not FabricPC."
        ),
    )
    parser.add_argument(
        "--column_teacher_weight",
        type=float,
        default=0.1,
        help=(
            "Weight on the auxiliary cross-entropy energy at column_teacher_output. "
            "The main output uses weight 1.0. The default keeps the column-only "
            "teacher in the predictive-coding graph without letting it dominate "
            "the shared column latents."
        ),
    )
    parser.add_argument(
        "--shell_teacher_weights",
        type=str,
        default=SHELL_TEACHER_DEFAULT_WEIGHTS,
        help=(
            "Comma-separated auxiliary cross-entropy weights for shell-local "
            "classifier heads in hard_kernel, inner_shell, middle_shell, "
            "outer_shell order. A zero weight omits that shell head."
        ),
    )
    parser.add_argument(
        "--column_shell_teacher_weights",
        type=str,
        default=COLUMN_SHELL_TEACHER_DEFAULT_WEIGHTS,
        help=(
            "Comma-separated auxiliary cross-entropy weights for per-column "
            "shell-local classifier heads in hard_kernel, inner_shell, "
            "middle_shell, outer_shell order. A zero weight omits those heads. "
            "These heads attach before the column combiner."
        ),
    )
    parser.add_argument(
        "--column_shell_readout",
        action="store_true",
        help=(
            "Connect each active column's pooled shell vectors directly to the "
            "main output classifier. This preserves column and shell identity "
            "at readout while keeping the original column_pool edge for ablation."
        ),
    )
    parser.add_argument(
        "--column_shell_bridge",
        action="store_true",
        help=(
            "Connect each active column's pooled shell vectors through one "
            "Gaussian shell bridge latent before the main output classifier."
        ),
    )
    parser.add_argument(
        "--outer_shell_context",
        action="store_true",
        help=(
            "Connect each active column's pooled shells through a Gaussian "
            "latent whose output width matches the outer_shell slice. With "
            "zero shell-prediction weight, this context vector also feeds the "
            "main classifier. With positive shell-prediction weight, it predicts "
            "local shell states instead."
        ),
    )
    parser.add_argument(
        "--outer_shell_context_to_bridge",
        action="store_true",
        help=(
            "Feed each active column's outer-shell context latent into that "
            "column's shell bridge latent. Requires --outer_shell_context and "
            "--column_shell_bridge."
        ),
    )
    parser.add_argument(
        "--outer_shell_context_teacher_weight",
        type=float,
        default=0.0,
        help=(
            "Weight on an auxiliary context-only classifier fed by all "
            "outer-shell context latents. A zero weight omits the head. "
            "This keeps the auxiliary target on the route that already "
            "feeds the main classifier instead of supervising each raw "
            "per-column shell independently."
        ),
    )
    parser.add_argument(
        "--outer_shell_context_shell_prediction_weight",
        type=float,
        default=SHELL_CONTEXT_PREDICTION_DEFAULT_WEIGHT,
        help=(
            "Weight on local Gaussian objectives where each active column's "
            "outer-shell context predicts that column's pooled hard-kernel, "
            "inner-shell, middle-shell, and outer-shell states. This local "
            "objective is added alongside the direct outer-context-to-output "
            "readout."
        ),
    )
    parser.add_argument(
        "--outer_shell_context_evidence",
        action="store_true",
        help=(
            "Feed all active outer-shell context latents through one "
            "class-width Gaussian evidence latent. This diagnostic path no "
            "longer feeds the main output classifier. Requires "
            "--outer_shell_context."
        ),
    )
    parser.add_argument(
        "--outer_shell_context_evidence_teacher_weight",
        type=float,
        default=0.0,
        help=(
            "Weight on an auxiliary classifier fed by the class-width "
            "outer_shell_context_evidence latent. A zero weight omits the "
            "teacher head while keeping the evidence path active when "
            "--outer_shell_context_evidence is set."
        ),
    )
    parser.add_argument(
        "--shell_evidence_cascade_scale",
        type=str,
        default=SHELL_EVIDENCE_CASCADE_DEFAULT_SCALE,
        help=(
            "Comma-separated gains for outward shell evidence flow in "
            "hard_kernel->inner_shell, inner_shell->middle_shell, and "
            "middle_shell->outer_shell order. This is not HiBaCaML "
            "consolidation; consolidation moves reusable material inward."
        ),
    )
    parser.add_argument(
        "--shell_inhibition_strengths",
        type=str,
        default=SHELL_INHIBITION_DEFAULT_STRENGTHS,
        help=(
            "Comma-separated same-tier inhibition strengths in hard_kernel, "
            "inner_shell, middle_shell, outer_shell order. The default uses "
            "the CIFAR shell-dynamics values from the HiBaCaML paper with no "
            "inhibition in the protected hard kernel."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.model = "tiny"
        args.num_columns = 2
        args.num_shared = 1
        args.active_nonshared = 1
        args.embed_dim = 32
        args.microcolumn_dim = 16
        args.num_epochs = 0.01
        args.infer_steps = 2
        args.eval_every = 0
    train_cifar10_depth_spanning(args)


if __name__ == "__main__":
    main()
