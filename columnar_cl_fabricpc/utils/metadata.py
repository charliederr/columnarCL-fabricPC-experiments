"""
Node metadata registry for columnar experiments.

This module provides a registry for associating metadata (roles, shell IDs,
support groups, etc.) with FabricPC nodes by name.

IMPORTANT: This registry is for graph-construction-time queries only. It
cannot be consulted inside a jax.jit-compiled forward pass because Python
dict lookups against traced values are not traceable. If metadata is needed
inside the forward path, it must be passed through node_info.node_config or
baked into the node's parameters at construction time.
"""

from typing import Any, Dict, Optional


class NodeMetadataRegistry:
    """
    A registry mapping node names to metadata dictionaries.

    This is intentionally a thin wrapper around a dict-of-dicts. It provides
    a clear API for attaching and querying metadata without modifying FabricPC.

    Usage:
        registry = NodeMetadataRegistry()
        registry.register("col_00", role="shared_column", shell=0)
        registry.register("col_07", role="local_column", task=1, shell=1)

        # Query
        registry.get("col_00", "role")  # "shared_column"
        registry.get_all("col_00")      # {"role": "shared_column", "shell": 0}

        # Bulk registration
        registry.register_many({
            "stem": {"role": "visual_stem"},
            "head": {"role": "readout", "task": None},
        })
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, node_name: str, **metadata: Any) -> None:
        """
        Register metadata for a node.

        Args:
            node_name: The name of the node (must match the name used in graph()).
            **metadata: Key-value pairs to associate with this node.

        If the node already has metadata, the new values are merged (updating
        existing keys and adding new ones).
        """
        if node_name not in self._registry:
            self._registry[node_name] = {}
        self._registry[node_name].update(metadata)

    def register_many(self, entries: Dict[str, Dict[str, Any]]) -> None:
        """
        Register metadata for multiple nodes at once.

        Args:
            entries: Dict mapping node names to metadata dicts.
        """
        for node_name, metadata in entries.items():
            self.register(node_name, **metadata)

    def get(self, node_name: str, key: str, default: Any = None) -> Any:
        """
        Get a specific metadata value for a node.

        Args:
            node_name: The node name.
            key: The metadata key to retrieve.
            default: Value to return if node or key not found.

        Returns:
            The metadata value, or default if not found.
        """
        return self._registry.get(node_name, {}).get(key, default)

    def get_all(self, node_name: str) -> Dict[str, Any]:
        """
        Get all metadata for a node.

        Args:
            node_name: The node name.

        Returns:
            A copy of the metadata dict, or empty dict if node not registered.
        """
        return dict(self._registry.get(node_name, {}))

    def has(self, node_name: str) -> bool:
        """Check if a node has any registered metadata."""
        return node_name in self._registry

    def nodes_with(self, key: str, value: Any = None) -> list:
        """
        Find all nodes that have a specific metadata key (and optionally value).

        Args:
            key: The metadata key to search for.
            value: If provided, only return nodes where metadata[key] == value.

        Returns:
            List of node names matching the criteria.
        """
        results = []
        for node_name, metadata in self._registry.items():
            if key in metadata:
                if value is None or metadata[key] == value:
                    results.append(node_name)
        return results

    def clear(self) -> None:
        """Remove all registered metadata."""
        self._registry.clear()

    def __repr__(self) -> str:
        return f"NodeMetadataRegistry({len(self._registry)} nodes)"
