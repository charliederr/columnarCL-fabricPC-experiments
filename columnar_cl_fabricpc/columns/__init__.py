"""
Columnar node implementations for predictive coding networks.

This module provides custom NodeBase subclasses that implement columnar
organization for continual learning.
"""

from columnar_cl_fabricpc.columns.example_node import ExampleColumnNode
from columnar_cl_fabricpc.columns.visual_stem import (
    ConvStemNode,
    PatchEmbedNode,
    create_cifar10_conv_stem,
    create_cifar10_patch_embed,
)
from columnar_cl_fabricpc.columns.column import (
    ColumnarNode,
    create_column,
    create_column_pool,
)
from columnar_cl_fabricpc.columns.combiner import (
    ColumnCombinerNode,
    ClassificationHeadNode,
    create_combiner,
    create_classification_head,
)
from columnar_cl_fabricpc.columns.stage_taps import (
    StageTapTokenizer,
    GlobalPoolNode,
    create_stage_tap,
    create_global_pool,
    create_cifar10_stage_taps,
)
from columnar_cl_fabricpc.columns.depth_spanning_column import (
    DepthSpanningColumnNode,
    create_depth_spanning_column,
    create_depth_spanning_column_pool,
)
from columnar_cl_fabricpc.columns.label_smoothed_ce import (
    LabelSmoothedCrossEntropyEnergy,
)
from columnar_cl_fabricpc.columns.accuracy_nodes import (
    GlobalAvgPoolNormNode,
    create_global_avg_pool_norm,
)

__all__ = [
    "ClassificationHeadNode",
    "ColumnCombinerNode",
    "ColumnarNode",
    "LabelSmoothedCrossEntropyEnergy",
    "ConvStemNode",
    "DepthSpanningColumnNode",
    "ExampleColumnNode",
    "GlobalAvgPoolNormNode",
    "GlobalPoolNode",
    "PatchEmbedNode",
    "StageTapTokenizer",
    "create_cifar10_conv_stem",
    "create_cifar10_patch_embed",
    "create_cifar10_stage_taps",
    "create_classification_head",
    "create_column",
    "create_column_pool",
    "create_combiner",
    "create_depth_spanning_column",
    "create_depth_spanning_column_pool",
    "create_global_avg_pool_norm",
    "create_global_pool",
    "create_stage_tap",
]
