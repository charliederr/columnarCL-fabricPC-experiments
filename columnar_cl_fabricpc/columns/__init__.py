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

__all__ = [
    "ClassificationHeadNode",
    "ColumnCombinerNode",
    "ColumnarNode",
    "ConvStemNode",
    "ExampleColumnNode",
    "PatchEmbedNode",
    "create_cifar10_conv_stem",
    "create_cifar10_patch_embed",
    "create_classification_head",
    "create_column",
    "create_column_pool",
    "create_combiner",
]
