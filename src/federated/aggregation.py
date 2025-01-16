"""Server-side aggregation rules as pure NumPy functions.

Every function takes ``updates: list[list[np.ndarray]]`` (one list of parameter arrays per
client, same shapes and order for every client), ``num_examples: list[int]`` (the client's
training-row count ``n_k``) and optionally ``client_ids``. Inputs are sorted by client id
before anything is summed, so the result does not depend on the order in which replies
arrived. Sums are computed in float64 and cast back to each array's dtype.

Aggregation rules
-----------------
:func:`weighted_average` is sample-weighted FedAvg (McMahan et al. 2017) and is the default:
each client's update is weighted by its number of rows, ``n_k / Σ n``, so that a client's
influence on the global model matches its share of the data. :func:`unweighted_average` is
the unweighted mean ``θ = (1/N) Σ θ_i``, kept as an ablation
(``federation.aggregation_rule: unweighted``); the two agree only when every client holds
the same number of rows.

Robust rules (:func:`coordinate_median`, :func:`trimmed_mean`) ignore ``num_examples``, as is
usual for Byzantine-robust aggregation: a sample-count weight would let an attacker claim a
huge ``n_k``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Update = list[np.ndarray]


def _sorted(
    updates: Sequence[Update], num_examples: Sequence[int], client_ids: Sequence[int] | None
) -> tuple[list[Update], list[int]]:
    if len(updates) == 0:
        raise ValueError("aggregation needs at least one client update")
    if len(updates) != len(num_examples):
        raise ValueError("updates and num_examples have different lengths")
    if client_ids is None:
        client_ids = list(range(len(updates)))
    if len(client_ids) != len(updates):
        raise ValueError("client_ids and updates have different lengths")
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("client_ids must be unique")
    order = sorted(range(len(updates)), key=lambda i: client_ids[i])
    first = updates[order[0]]
    for i in order:
        if len(updates[i]) != len(first) or any(
            a.shape != b.shape for a, b in zip(updates[i], first, strict=True)
        ):
            raise ValueError("client updates have inconsistent shapes")
    return [list(updates[i]) for i in order], [int(num_examples[i]) for i in order]


def _stack(updates: Sequence[Update], layer: int) -> np.ndarray:
    return np.stack([np.asarray(update[layer], dtype=np.float64) for update in updates])


def weighted_average(
    updates: Sequence[Update],
    num_examples: Sequence[int],
    client_ids: Sequence[int] | None = None,
) -> Update:
    """Sample-count-weighted mean ``Σ_k (n_k / Σ_j n_j) θ_k`` (FedAvg; the default rule)."""
    ordered, counts = _sorted(updates, num_examples, client_ids)
    if any(count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("num_examples must be non-negative with a positive total")
    weights = np.asarray(counts, dtype=np.float64) / float(sum(counts))
    result = []
    for layer, reference in enumerate(ordered[0]):
        stacked = _stack(ordered, layer)
        total = np.tensordot(weights, stacked, axes=1)
        result.append(total.astype(reference.dtype, copy=False))
    return result


def unweighted_average(
    updates: Sequence[Update],
    num_examples: Sequence[int],
    client_ids: Sequence[int] | None = None,
) -> Update:
    """Plain mean ``(1/N) Σ_k θ_k`` — the unweighted mean, kept as an ablation."""
    ordered, counts = _sorted(updates, num_examples, client_ids)
    return weighted_average(ordered, [1] * len(counts))


def coordinate_median(
    updates: Sequence[Update],
    num_examples: Sequence[int],
    client_ids: Sequence[int] | None = None,
) -> Update:
    """Coordinate-wise median over clients (Yin et al. 2018); ignores ``num_examples``."""
    ordered, _ = _sorted(updates, num_examples, client_ids)
    return [
        np.median(_stack(ordered, layer), axis=0).astype(reference.dtype, copy=False)
        for layer, reference in enumerate(ordered[0])
    ]


def trimmed_mean(
    updates: Sequence[Update],
    num_examples: Sequence[int],
    client_ids: Sequence[int] | None = None,
    *,
    beta: float,
) -> Update:
    """Coordinate-wise β-trimmed mean: drop the ``floor(β·N)`` largest and smallest values.

    ``beta`` comes from ``federation.trimmed_mean_beta`` in YAML; ``beta = 0`` is the plain
    unweighted mean. ``num_examples`` is ignored.
    """
    if not 0.0 <= beta < 0.5:
        raise ValueError("beta must be in [0, 0.5)")
    ordered, _ = _sorted(updates, num_examples, client_ids)
    count = len(ordered)
    trim = int(np.floor(beta * count))
    if count - 2 * trim <= 0:
        raise ValueError("trimmed_mean would discard every client")
    result = []
    for layer, reference in enumerate(ordered[0]):
        stacked = np.sort(_stack(ordered, layer), axis=0)
        kept = stacked[trim : count - trim]
        result.append(kept.mean(axis=0).astype(reference.dtype, copy=False))
    return result


AGGREGATORS: dict[str, Callable[..., Update]] = {
    "weighted": weighted_average,
    "unweighted": unweighted_average,
    "coordinate_median": coordinate_median,
    "trimmed_mean": trimmed_mean,
}


def aggregate(
    rule: str,
    updates: Sequence[Update],
    num_examples: Sequence[int],
    client_ids: Sequence[int] | None = None,
    *,
    beta: float,
) -> Update:
    """Dispatch to the rule named in ``federation.aggregation_rule``."""
    if rule not in AGGREGATORS:
        raise ValueError(f"Unknown aggregation rule {rule!r}; choose from {sorted(AGGREGATORS)}")
    if rule == "trimmed_mean":
        return trimmed_mean(updates, num_examples, client_ids, beta=beta)
    return AGGREGATORS[rule](updates, num_examples, client_ids)
