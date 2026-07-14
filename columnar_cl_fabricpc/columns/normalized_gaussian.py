"""
Shape-normalized Gaussian energy for columnar predictive-coding nodes.

FabricPC's default Gaussian energy sums squared prediction error over every
non-batch latent dimension. That is appropriate when latent sizes are fixed, but
it changes local precision when an experiment changes token count or feature
width. This local energy keeps predictive-coding Gaussian errors while measuring
mean squared error per latent element.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from fabricpc.core.energy import EnergyFunctional


def _latent_size(x: jnp.ndarray) -> float:
    """Return the number of non-batch latent elements as a Python float."""
    size = 1
    for dim in x.shape[1:]:
        size *= int(dim)
    return float(max(size, 1))


class MeanSquaredGaussianEnergy(EnergyFunctional):
    """
    Gaussian prediction-error energy normalized by latent size.

    `precision` is the scalar Gaussian precision after latent-size
    normalization. For an error tensor `e = z_latent - z_mu`, the per-sample
    energy is `0.5 * precision * sum(e**2) / D`, where `D` is the number of
    non-batch latent elements in the node.
    """

    def __init__(self, precision: float = 1.0):
        if precision < 0.0:
            raise ValueError(f"precision must be >= 0, got {precision}")
        super().__init__(precision=precision)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        precision = config.get("precision", 1.0) if config else 1.0
        diff = z_latent - z_mu
        axes_to_sum = tuple(range(1, len(diff.shape)))
        return 0.5 * precision * jnp.sum(diff**2, axis=axes_to_sum) / _latent_size(
            diff
        )

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        precision = config.get("precision", 1.0) if config else 1.0
        return precision * (z_latent - z_mu) / _latent_size(z_latent)
