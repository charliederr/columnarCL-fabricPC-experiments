"""
Shape-aware Gaussian energies for columnar predictive-coding nodes.

FabricPC's default Gaussian energy sums squared prediction error over every
non-batch latent dimension. That is appropriate when latent sizes are fixed, but
it changes local precision when an experiment changes token count or feature
width. These local energies keep Gaussian prediction errors while making the
precision scaling explicit.
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


def _spatial_site_count(x: jnp.ndarray) -> float:
    """
    Return the number of spatial or token sites in a latent tensor.

    Columnar token tensors use shape `(batch, sites, features)` or
    `(batch, height, width, features)`. Pooled shell and context tensors use
    shape `(batch, features)` and therefore have one spatial site.
    """
    if len(x.shape) <= 2:
        return 1.0
    sites = 1
    for dim in x.shape[1:-1]:
        sites *= int(dim)
    return float(max(sites, 1))


def _spatial_reference_scale(x: jnp.ndarray, reference_sites: float) -> float:
    """
    Return the predictive-coding precision divisor for spatial replication.

    `reference_sites` is the historical token-site count whose summed Gaussian
    precision should be preserved. For ResNet18 CIFAR-10 stage4 columns this is
    16, corresponding to a 4 by 4 token grid.
    """
    return max(1.0, _spatial_site_count(x) / reference_sites)


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


class SpatialReferenceGaussianEnergy(EnergyFunctional):
    """
    Gaussian prediction-error energy scaled by spatial-site count.

    `precision` is the scalar Gaussian precision after spatial scaling.
    `reference_sites` is the token-site count whose summed residual precision is
    preserved. For an error tensor `e = z_latent - z_mu`, the per-sample energy
    is `0.5 * precision * sum(e**2) / A`, where `A = max(1, S / S_ref)`, `S`
    is the number of spatial or token sites in the latent tensor, and `S_ref`
    is the reference site count.
    """

    def __init__(self, precision: float = 1.0, reference_sites: float = 16.0):
        if precision < 0.0:
            raise ValueError(f"precision must be >= 0, got {precision}")
        if reference_sites <= 0.0:
            raise ValueError(
                f"reference_sites must be > 0, got {reference_sites}"
            )
        super().__init__(precision=precision, reference_sites=reference_sites)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        precision = config.get("precision", 1.0) if config else 1.0
        reference_sites = config.get("reference_sites", 16.0) if config else 16.0
        diff = z_latent - z_mu
        axes_to_sum = tuple(range(1, len(diff.shape)))
        scale = _spatial_reference_scale(diff, reference_sites)
        return 0.5 * precision * jnp.sum(diff**2, axis=axes_to_sum) / scale

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        precision = config.get("precision", 1.0) if config else 1.0
        reference_sites = config.get("reference_sites", 16.0) if config else 16.0
        scale = _spatial_reference_scale(z_latent, reference_sites)
        return precision * (z_latent - z_mu) / scale
