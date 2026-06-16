"""Shared test fixtures and configuration for columnar_cl_fabricpc tests."""

import os

# JAX environment configuration - must be set before importing JAX
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
os.environ.setdefault("XLA_GPU_DETERMINISTIC_OPS", "true")

import pytest
import jax


@pytest.fixture
def rng_key():
    """Provide a JAX random key for tests."""
    return jax.random.PRNGKey(42)
