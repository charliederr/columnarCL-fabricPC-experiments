#!/usr/bin/env python3
"""
CIFAR-10 with depth-spanning columnar architecture.

This experiment extends the PC ResNet baseline with depth-spanning columns
that receive skip connections from multiple ResNet stages. Each column
contains K/L/B microcolumn pathways that process multi-scale features.

Architecture::

    ResNet Stage 2 (32×32) ────┬─────────────────────────────────────┐
           │                   │                                     │
           ▼                   ▼                                     │
    ResNet Stage 3 (16×16) ────┼───────────────────────┐             │
           │                   │                       │             │
           ▼                   ▼                       ▼             ▼
    ResNet Stage 4 (8×8) ──────┼─────────────┐    stage3_tap    stage2_tap
           │                   │             │         │             │
           ▼                   ▼             ▼         │             │
      stage4_tap         stage4_pool    Columns ◀──────┴─────────────┘
           │                   │             │
           └───────────────────┴─────────────┘
                               │
                               ▼
                           Combiner
                               │
                               ▼
                          Classifier

Usage:
    python scripts/train_cifar10_depth_spanning.py --quick
    python scripts/train_cifar10_depth_spanning.py --model resnet18 --num_epochs 2
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
    WeightedLabelSmoothedCrossEntropyEnergy,
    SHELL_NAMES,
    create_stage_tap,
    create_global_pool,
    create_depth_spanning_column,
    get_shell_slices,
)
from columnar_cl_fabricpc.columns.accuracy_nodes import MaskedColumnCombinerNode

jax.config.update("jax_default_prng_impl", "threefry2x32")

COLUMN_TEACHER_TARGET = "column_y"
COLUMN_TEACHER_NODE = "column_teacher_output"
SHELL_TEACHER_DEFAULT_WEIGHTS = "0,0,0,0"
COLUMN_SHELL_TEACHER_DEFAULT_WEIGHTS = "0,0,0,0"


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


def parse_shell_teacher_weights(
    value: str,
    flag_name: str = "--shell_teacher_weights",
) -> Dict[str, float]:
    """
    Parse shell-local teacher weights in `SHELL_NAMES` order.

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
            raise ValueError(f"Shell teacher weight for {shell_name} must be >= 0")
        weights[shell_name] = weight
    return weights


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
        ):
            categories["classifier"] += energy_sum
        elif (
            node_name.startswith("col_")
            or is_column_shell_auxiliary_node(node_name)
            or is_column_shell_bridge_node(node_name)
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


def diagnose_classifier_edge_weights(params) -> Dict[str, float]:
    """
    Per-edge Frobenius norm of the classifier (Linear "output" node)'s weight
    matrices, one per incoming edge. With --bypass_columns, this surfaces whether
    the classifier is using the bypass edge or the pooled column-readout edge.
    """
    if "output" not in params.nodes:
        return {}
    out_weights = params.nodes["output"].weights
    return {
        edge_key: float(jnp.linalg.norm(W))
        for edge_key, W in out_weights.items()
    }


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
    edge_sources = output_input_edge_sources(structure)
    column_source = "column_pool"
    if column_source not in edge_sources:
        raise ValueError("Readout ablations require the column_pool edge")

    cases: List[Tuple[str, Tuple[str, ...]]] = [
        ("combined", tuple(edge_sources.keys())),
    ]
    column_shell_sources = tuple(
        sorted(source for source in edge_sources if is_column_shell_pool_node(source))
    )
    column_shell_bridge_sources = tuple(
        sorted(source for source in edge_sources if is_column_shell_bridge_node(source))
    )
    if column_source in edge_sources:
        cases.append(("column_only", (column_source,)))
    if column_shell_sources:
        cases.append(("column_shell_readout_only", column_shell_sources))
    if column_source in edge_sources and column_shell_sources:
        cases.append(
            ("column_pool_plus_shell_readout", (column_source, *column_shell_sources))
        )
    if column_shell_bridge_sources:
        cases.append(("column_shell_bridge_only", column_shell_bridge_sources))
    if column_source in edge_sources and column_shell_bridge_sources:
        cases.append(
            ("column_pool_plus_shell_bridge", (column_source, *column_shell_bridge_sources))
        )
    if column_shell_sources and column_shell_bridge_sources:
        cases.append(
            (
                "column_shell_readout_plus_bridge",
                (*column_shell_sources, *column_shell_bridge_sources),
            )
        )
    if "bypass_pool" in edge_sources:
        cases.append(("bypass_only", ("bypass_pool",)))

    results = {}
    for case_name, kept_sources in cases:
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
        kept_sources = tuple(
            source
            for source in bridge_input_sources
            if is_column_shell_pool_for_shell(source, shell_name) == keep_shell
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


def build_depth_spanning_graph(args):
    """
    Build a PC graph with depth-spanning columns.

    The ResNet backbone is built with stage outputs tracked. Stage taps
    tokenize each stage's output, and depth-spanning columns receive
    skip connections from all stages.
    """
    model_config = MODEL_CONFIGS[args.model]
    weight_init = MuPCInitializer()
    activation = get_activation(args.activation)
    shell_teacher_weights = parse_shell_teacher_weights(args.shell_teacher_weights)
    column_shell_teacher_weights = parse_shell_teacher_weights(
        args.column_shell_teacher_weights,
        flag_name="--column_shell_teacher_weights",
    )

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

    # Determine target grid from final stage
    final_h, final_w, _ = stage4_out.shape
    target_grid = (final_h, final_w)
    num_tokens = final_h * final_w

    print(f"Stage outputs: stage2={stage2_out.shape}, stage3={stage3_out.shape}, stage4={stage4_out.shape}")
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
    )
    stage3_tap = create_stage_tap(
        name="stage3_tap",
        source_channels=stage3_channels,
        embed_dim=args.embed_dim,
        target_grid=target_grid,
        add_pos_embed=True,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
    )
    stage4_tap = create_stage_tap(
        name="stage4_tap",
        source_channels=stage4_channels,
        embed_dim=args.embed_dim,
        target_grid=target_grid,
        add_pos_embed=True,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
    )
    stage4_pool = create_global_pool(
        name="stage4_pool",
        source_channels=stage4_channels,
        embed_dim=args.embed_dim,
        apply_layer_norm=args.layer_norm_tokens,
        fix_ln_gamma=args.fix_ln_gamma,
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
            apply_layer_norm=args.layer_norm_tokens,
            fix_ln_gamma=args.fix_ln_gamma,
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

    # Combiner
    combiner = MaskedColumnCombinerNode(
        shape=(num_tokens, args.embed_dim),
        name="combiner",
        num_columns=args.num_columns,
        support_mask=support_mask,
        combination=args.combiner,
    )
    nodes.append(combiner)

    for col in columns:
        edges.append(Edge(source=col, target=combiner.slot("in")))

    # Raw pooled readout, combined classifier, and column-only teacher head.
    column_pool = AvgPool(
        shape=(args.embed_dim,),
        name="column_pool",
        global_pool=True,
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
    for column_idx in active_column_indices:
        column = columns[column_idx]
        column_shell_pools = []
        for shell_name in SHELL_NAMES:
            shell_weight = column_shell_teacher_weights[shell_name]
            needs_shell_pool = (
                shell_weight > 0.0
                or args.column_shell_readout
                or args.column_shell_bridge
            )
            if not needs_shell_pool:
                continue

            start, end = shell_slices[shell_name]
            shell_slice = FeatureSliceNode(
                shape=(num_tokens, end - start),
                name=column_shell_slice_node_name(column_idx, shell_name),
                start=start,
                end=end,
            )
            shell_pool = AvgPool(
                shape=(end - start,),
                name=column_shell_pool_node_name(column_idx, shell_name),
                global_pool=True,
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

        if args.column_shell_bridge:
            shell_bridge = Linear(
                shape=(args.embed_dim,),
                name=column_shell_bridge_node_name(column_idx),
                activation=IdentityActivation(),
                flatten_input=False,
                weight_init=XavierInitializer(),
            )
            nodes.append(shell_bridge)
            for shell_pool in column_shell_pools:
                edges.append(Edge(source=shell_pool, target=shell_bridge.slot("in")))
            edges.append(Edge(source=shell_bridge, target=output.slot("in")))

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
    print(f"Embed dim: {args.embed_dim}")
    print(f"Microcolumn dim: {args.microcolumn_dim}")
    shell_slices = get_shell_slices(args.embed_dim)
    shell_widths = ", ".join(
        f"{name}={end - start}" for name, (start, end) in shell_slices.items()
    )
    print(f"Shell widths: {shell_widths}")
    print("Shell promotion: enabled")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Inference steps: {args.infer_steps}")
    print(f"Inference eta: {args.eta_infer}")
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
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
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

            edge_norms = diagnose_classifier_edge_weights(params)
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
    parser.add_argument("--combiner", choices=["attention", "sum"], default="sum")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--microcolumn_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--infer_steps", type=int, default=40)
    parser.add_argument("--eta_infer", type=float, default=0.1)
    parser.add_argument("--infer_max_norm", type=float, default=1.0)
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
