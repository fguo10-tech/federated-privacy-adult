"""Statistics and classification metrics used for every reported number.

Reporting convention used everywhere in this repository
-------------------------------------------------------
``mean ± std`` means: first average a metric *within* each seed (over evaluation
episodes, clients or rounds as appropriate), then take the mean and the **sample**
standard deviation (``ddof = 1``) of those per-seed values *across* seeds. With
five seeds the standard deviation therefore has four degrees of freedom. It
describes run-to-run variation; it is not a confidence interval.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (``ddof=1``) of ``values``.

    The standard deviation is ``nan`` when fewer than two values are given,
    because one run carries no information about run-to-run variation.
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan, math.nan
    if array.size == 1:
        return float(array[0]), math.nan
    if not np.all(np.isfinite(array)):
        # For example ε = ∞ for a run without differential privacy: the mean is infinite and a
        # spread is not defined.
        return float(np.sum(array)) if not np.any(np.isnan(array)) else math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=1))


def format_mean_std(mean: float, std: float, digits: int = 3, n: int | None = None) -> str:
    """Render ``mean ± std`` with a fixed number of decimals.

    A missing mean renders as ``TBD``; a missing standard deviation (single run)
    renders as ``mean (n=1)`` rather than inventing a spread.
    """
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "TBD"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        suffix = f" (n={n})" if n is not None else " (n=1)"
        return f"{mean:.{digits}f}{suffix}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarise_by_seed(
    rows: Iterable[Mapping[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
    seed_key: str = "seed",
) -> list[dict[str, Any]]:
    """Two-stage aggregation: mean within each seed, then mean ± std across seeds.

    Returns one dict per group with the group keys, ``n_seeds`` and, for each value
    key ``v``, the columns ``v_mean`` and ``v_std``. Rows whose value is empty or
    non-numeric are skipped for that value.
    """
    per_seed: dict[tuple, dict[Any, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    order: list[tuple] = []
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        if group not in per_seed:
            order.append(group)
        seed = row[seed_key]
        for key in value_keys:
            number = _to_float(row.get(key))
            if number is not None:
                per_seed[group][seed][key].append(number)
    summary: list[dict[str, Any]] = []
    for group in order:
        seeds = per_seed[group]
        record: dict[str, Any] = dict(zip(group_keys, group, strict=True))
        record["n_seeds"] = len(seeds)
        for key in value_keys:
            seed_means = [float(np.mean(v[key])) for v in seeds.values() if v.get(key)]
            mean, std = mean_std(seed_means)
            record[f"{key}_mean"] = mean
            record[f"{key}_std"] = std
        summary.append(record)
    return summary


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Confusion matrix ``C`` with ``C[i, j]`` = number of samples of class ``i`` predicted ``j``."""
    true = np.asarray(y_true, dtype=np.int64).ravel()
    pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    positive_class: int | None = None,
) -> dict[str, float]:
    """Accuracy plus macro- and support-weighted precision, recall and F1.

    A class that is never predicted has precision 0 (not undefined), and a class
    with no true samples has recall 0; this matches ``zero_division=0`` in
    scikit-learn. If ``positive_class`` is given, the result also contains the
    one-vs-rest ``precision``, ``recall`` and ``f1`` of that class, which is the
    conventional report for a binary task such as UCI Adult (positive = ``>50K``).
    """
    matrix = confusion_matrix(y_true, y_pred, num_classes).astype(float)
    total = matrix.sum()
    if total == 0:
        raise ValueError("classification_metrics needs at least one sample")
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros(num_classes), where=denominator > 0
    )
    weights = support / total
    result = {
        "accuracy": float(true_positive.sum() / total),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
    }
    if positive_class is not None:
        result["precision"] = float(precision[positive_class])
        result["recall"] = float(recall[positive_class])
        result["f1"] = float(f1[positive_class])
    return result


def fairness_gap(values: Iterable[float]) -> float:
    """Largest minus smallest value across clients or groups (0 means perfectly even)."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan
    return float(array.max() - array.min())


def constant_predictor_weighted_scores(majority_fraction: float) -> dict[str, float]:
    """Support-weighted scores of a classifier that always predicts one class.

    If the predicted class makes up a fraction ``p`` of the test set, then under
    support-weighted averaging: accuracy = ``p``, recall = ``p``, precision = ``p**2``
    and F1 = ``2 p**2 / (1 + p)``.
    """
    p = float(majority_fraction)
    if not 0.0 < p <= 1.0:
        raise ValueError("majority_fraction must be in (0, 1]")
    return {
        "accuracy": p,
        "recall_weighted": p,
        "precision_weighted": p * p,
        "f1_weighted": 2.0 * p * p / (1.0 + p),
    }
