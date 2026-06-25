"""
Label-smoothed cross-entropy energy for the classifier output node.

The standard `CrossEntropyEnergy` in FabricPC computes `E = -sum z_i log mu_i`
with one-hot `z_latent`. When the model becomes confident on the training set,
`mu_y` approaches 1 for the correct class, `-log mu_y` approaches 0, and the
gradient signal vanishes. With AdamW + weight decay, the optimizer continues
to shrink the weights that produced those confident predictions, slowly
eroding the discriminative directions. Eventually the model escapes its
saturated basin into a much-worse regime.

Label smoothing replaces the one-hot target with
    target = (1 - smoothing) * one_hot + smoothing / K
which lower-bounds the achievable cross-entropy at
    -(1-eps) log(1-eps) - eps log(eps/(K-1))
and keeps the gradient signal alive throughout training.

The implementation lives in the experiments repo, not in FabricPC.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from fabricpc.core.energy import EnergyFunctional


class LabelSmoothedCrossEntropyEnergy(EnergyFunctional):
    """
    Label-smoothed categorical cross-entropy.

    Expects `z_latent` to be one-hot encoded (the framework clamps it during
    training). The smoothed target is computed inside `energy()` as
        target = (1 - smoothing) * z_latent + smoothing / K
    and the energy is
        E = -sum target_i * log(mu_i_safe)
          = (1 - smoothing) * E_one_hot + (smoothing / K) * -sum log(mu_i_safe)

    Args:
        smoothing: Label-smoothing factor in [0, 1). 0.0 reduces to plain CE.
            Typical values: 0.1 (CIFAR-10 baseline), 0.05 (mild).
        num_classes: K. Required because the smoothed-uniform term needs to
            know how many classes the softmax covers.
        eps: Numerical floor for log() (default 1e-7).
        axis: Class axis (default -1).
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 10,
        eps: float = 1e-7,
        axis: int = -1,
    ):
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        super().__init__(
            smoothing=smoothing,
            num_classes=num_classes,
            eps=eps,
            axis=axis,
        )

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        config = config or {}
        smoothing = float(config.get("smoothing", 0.1))
        num_classes = int(config.get("num_classes", z_mu.shape[-1]))
        eps = float(config.get("eps", 1e-7))

        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        log_mu = jnp.log(z_mu_safe)

        target = (1.0 - smoothing) * z_latent + (smoothing / num_classes)
        ce = -target * log_mu

        axes_to_sum = tuple(range(1, len(ce.shape)))
        return jnp.sum(ce, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        """
        dE/dz_latent_i = -(1 - smoothing) * log(mu_i_safe)

        The smoothing/K term in `target` does not depend on z_latent, so its
        gradient is zero. The smoothing factor scales the per-class gradient
        relative to plain CE.
        """
        config = config or {}
        smoothing = float(config.get("smoothing", 0.1))
        eps = float(config.get("eps", 1e-7))

        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        return -(1.0 - smoothing) * jnp.log(z_mu_safe)


class WeightedLabelSmoothedCrossEntropyEnergy(EnergyFunctional):
    """
    Weighted categorical cross-entropy with optional label smoothing.

    `z_latent` is the one-hot class target clamped by FabricPC. `z_mu` is the
    softmax probability vector predicted by a classifier node. `weight` is a
    scalar multiplier on the per-sample class energy. This lets auxiliary
    predictive-coding targets contribute a controlled error signal without
    changing the main classifier energy.
    """

    def __init__(
        self,
        weight: float = 1.0,
        smoothing: float = 0.0,
        num_classes: int = 10,
        eps: float = 1e-7,
        axis: int = -1,
    ):
        if weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        super().__init__(
            weight=weight,
            smoothing=smoothing,
            num_classes=num_classes,
            eps=eps,
            axis=axis,
        )

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        config = config or {}
        weight = float(config.get("weight", 1.0))
        smoothing = float(config.get("smoothing", 0.0))
        num_classes = int(config.get("num_classes", z_mu.shape[-1]))
        eps = float(config.get("eps", 1e-7))

        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        log_mu = jnp.log(z_mu_safe)
        target = (1.0 - smoothing) * z_latent + (smoothing / num_classes)
        ce = -target * log_mu

        axes_to_sum = tuple(range(1, len(ce.shape)))
        return weight * jnp.sum(ce, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        config = config or {}
        weight = float(config.get("weight", 1.0))
        smoothing = float(config.get("smoothing", 0.0))
        eps = float(config.get("eps", 1e-7))

        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        return -weight * (1.0 - smoothing) * jnp.log(z_mu_safe)
