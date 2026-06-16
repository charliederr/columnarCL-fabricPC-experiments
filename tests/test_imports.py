"""
Smoke tests for columnar_cl_fabricpc package.

Verifies that:
1. Package imports succeed
2. FabricPC resolves to upstream (not cFabricPC)
3. ExampleColumnNode works with FabricPC graph machinery
"""

import pytest
import jax
import jax.numpy as jnp


class TestImports:
    """Test that all package imports succeed."""

    def test_import_columnar_cl_fabricpc(self):
        """Test that the main package imports."""
        import columnar_cl_fabricpc

        assert hasattr(columnar_cl_fabricpc, "__version__")

    def test_import_columns(self):
        """Test that columns subpackage imports."""
        from columnar_cl_fabricpc.columns import example_node

        assert hasattr(example_node, "ExampleColumnNode")

    def test_import_utils(self):
        """Test that utils subpackage imports."""
        from columnar_cl_fabricpc.utils import metadata

        assert hasattr(metadata, "NodeMetadataRegistry")

    def test_import_data(self):
        """Test that data subpackage imports."""
        from columnar_cl_fabricpc import data

        assert data is not None

    def test_import_experiments(self):
        """Test that experiments subpackage imports."""
        from columnar_cl_fabricpc import experiments

        assert experiments is not None


class TestFabricPCResolution:
    """Test that FabricPC resolves to upstream."""

    def test_fabricpc_resolves_to_upstream(self):
        """Verify fabricpc imports from FabricPC, not cFabricPC."""
        import fabricpc

        # The path should contain "FabricPC", not "cFabricPC"
        assert "FabricPC" in fabricpc.__file__
        assert "cFabricPC" not in fabricpc.__file__

    def test_fabricpc_has_expected_modules(self):
        """Verify fabricpc has the expected structure."""
        from fabricpc.nodes.base import NodeBase, SlotSpec
        from fabricpc.core.types import NodeParams, NodeState
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge

        assert NodeBase is not None
        assert SlotSpec is not None


class TestExampleColumnNode:
    """Test that ExampleColumnNode integrates with FabricPC."""

    def test_example_node_instantiation(self):
        """Test that ExampleColumnNode can be instantiated."""
        from columnar_cl_fabricpc.columns.example_node import ExampleColumnNode

        node = ExampleColumnNode(shape=(8,), name="test")
        assert node.name == "test"
        assert node.shape == (8,)

    def test_example_node_slots(self):
        """Test that ExampleColumnNode has expected slots."""
        from columnar_cl_fabricpc.columns.example_node import ExampleColumnNode

        slots = ExampleColumnNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True

    def test_example_node_in_graph(self, rng_key):
        """Test that ExampleColumnNode can be placed in a FabricPC graph."""
        from columnar_cl_fabricpc.columns.example_node import ExampleColumnNode
        from fabricpc.nodes import Linear
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params

        input_node = Linear(shape=(8,), name="input")
        column_node = ExampleColumnNode(shape=(8,), name="column")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, column_node, output_node],
            edges=[
                Edge(source=input_node, target=column_node.slot("in")),
                Edge(source=column_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        assert len(structure.nodes) == 3
        assert "column" in structure.nodes

        params = initialize_params(structure, rng_key)
        assert "column" in params.nodes

    def test_example_node_forward(self, rng_key):
        """Test that ExampleColumnNode forward pass works."""
        from columnar_cl_fabricpc.columns.example_node import ExampleColumnNode
        from fabricpc.nodes import Linear
        from fabricpc.graph_assembly import graph, TaskMap
        from fabricpc.core.topology import Edge
        from fabricpc.core.inference import InferenceSGD
        from fabricpc.graph_initialization import initialize_params
        from fabricpc.graph_initialization.state_initializer import (
            initialize_graph_state,
        )

        input_node = Linear(shape=(8,), name="input")
        column_node = ExampleColumnNode(shape=(8,), name="column")

        structure = graph(
            nodes=[input_node, column_node],
            edges=[
                Edge(source=input_node, target=column_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)
        x_data = jax.random.normal(key1, (batch_size, 8))
        clamps = {"input": x_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Verify state was created for column node
        assert "column" in state.nodes
        column_state = state.nodes["column"]
        assert column_state.z_latent.shape == (batch_size, 8)


class TestMetadataRegistry:
    """Test the NodeMetadataRegistry."""

    def test_registry_basic_operations(self):
        """Test register and get operations."""
        from columnar_cl_fabricpc.utils.metadata import NodeMetadataRegistry

        registry = NodeMetadataRegistry()
        registry.register("col_00", role="shared", shell=0)

        assert registry.get("col_00", "role") == "shared"
        assert registry.get("col_00", "shell") == 0
        assert registry.get("col_00", "missing", "default") == "default"

    def test_registry_get_all(self):
        """Test get_all returns all metadata."""
        from columnar_cl_fabricpc.utils.metadata import NodeMetadataRegistry

        registry = NodeMetadataRegistry()
        registry.register("node1", a=1, b=2)

        all_meta = registry.get_all("node1")
        assert all_meta == {"a": 1, "b": 2}

    def test_registry_nodes_with(self):
        """Test finding nodes by metadata."""
        from columnar_cl_fabricpc.utils.metadata import NodeMetadataRegistry

        registry = NodeMetadataRegistry()
        registry.register("col_00", role="shared", shell=0)
        registry.register("col_01", role="shared", shell=0)
        registry.register("col_02", role="local", shell=1)

        shared = registry.nodes_with("role", "shared")
        assert set(shared) == {"col_00", "col_01"}

        shell_0 = registry.nodes_with("shell", 0)
        assert set(shell_0) == {"col_00", "col_01"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
