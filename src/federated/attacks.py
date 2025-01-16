"""Byzantine and poisoning behaviour of attacking clients (robustness suite).

Update transforms act on the client's update ``Δ = θ_client − θ_global`` after honest local
training; the attacker then sends ``θ_global + Δ'``:

* ``sign_flip`` — ``Δ' = −Δ``;
* ``scale`` — ``Δ' = scale_factor · Δ`` (model-replacement style boosting);
* ``gaussian_noise`` — ``Δ' = N(0, noise_std²)`` per coordinate (a random, useless update),
  seeded with ``derive_seed(seed, "attack", client, round)``.

``label_flip`` is data poisoning: before training the attacker relabels every row
``y → (C − 1) − y`` (for a binary task: swaps the two labels) and then trains honestly.

All attack parameters are in YAML (``robustness.attack``) and are
``[Assumption — not in paper]``.
"""

from __future__ import annotations

import numpy as np

from src.common.seeding import rng_from_seed

UPDATE_ATTACKS = ("sign_flip", "scale", "gaussian_noise")
DATA_ATTACKS = ("label_flip",)
ATTACK_KINDS = ("none", *UPDATE_ATTACKS, *DATA_ATTACKS)


def sign_flip(weights: list[np.ndarray], global_weights: list[np.ndarray]) -> list[np.ndarray]:
    """Send the negated update: ``θ_g − (θ − θ_g)``."""
    return [(2.0 * g - w).astype(w.dtype) for w, g in zip(weights, global_weights, strict=True)]


def scale(
    weights: list[np.ndarray], global_weights: list[np.ndarray], factor: float
) -> list[np.ndarray]:
    """Send a boosted update: ``θ_g + factor · (θ − θ_g)``."""
    return [
        (g + factor * (w - g)).astype(w.dtype) for w, g in zip(weights, global_weights, strict=True)
    ]


def gaussian_noise(
    weights: list[np.ndarray],
    global_weights: list[np.ndarray],
    std: float,
    seed: int,
    client: int,
    server_round: int,
) -> list[np.ndarray]:
    """Replace the update by Gaussian noise: ``θ_g + N(0, std²)``."""
    rng = rng_from_seed(seed, "attack", client, server_round)
    return [(g + rng.normal(0.0, std, size=g.shape)).astype(w.dtype) for w, g in zip(weights, global_weights, strict=True)]  # fmt: skip


def label_flip(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Data poisoning: ``y → (num_classes − 1) − y``."""
    return (num_classes - 1) - np.asarray(labels, dtype=np.int64)


def apply_update_attack(
    kind: str,
    weights: list[np.ndarray],
    global_weights: list[np.ndarray],
    scale_factor: float,
    noise_std: float,
    seed: int,
    client: int,
    server_round: int,
) -> list[np.ndarray]:
    """Apply the configured update transform (identity for ``none`` and data attacks)."""
    if kind == "sign_flip":
        return sign_flip(weights, global_weights)
    if kind == "scale":
        return scale(weights, global_weights, scale_factor)
    if kind == "gaussian_noise":
        return gaussian_noise(weights, global_weights, noise_std, seed, client, server_round)
    if kind in ("none", *DATA_ATTACKS):
        return weights
    raise ValueError(f"Unknown attack {kind!r}; choose from {ATTACK_KINDS}")


def byzantine_clients(num_clients: int, num_byzantine: int) -> tuple[int, ...]:
    """The attacking client ids: the last ``num_byzantine`` clients of an IID split.

    With IID clients every client is statistically identical, so which ids attack does not
    matter; a fixed choice keeps runs comparable across aggregation rules.
    """
    if not 0 <= num_byzantine < num_clients:
        raise ValueError("need 0 <= num_byzantine < num_clients")
    return tuple(range(num_clients - num_byzantine, num_clients))
