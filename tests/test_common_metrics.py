"""Statistics, classification metrics and the constant-predictor identity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.metrics import (
    classification_metrics,
    confusion_matrix,
    constant_predictor_weighted_scores,
    fairness_gap,
    format_mean_std,
    mean_std,
    summarise_by_seed,
)
from src.common.seeding import derive_seed, rng_from_seed


def test_mean_std_uses_sample_standard_deviation() -> None:
    mean, std = mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(math.sqrt(2.5))  # ddof = 1
    single_mean, single_std = mean_std([7.0])
    assert single_mean == 7.0 and math.isnan(single_std)
    assert format_mean_std(3.0, math.sqrt(2.5), digits=2) == "3.00 ± 1.58"
    assert format_mean_std(7.0, math.nan, digits=1) == "7.0 (n=1)"
    assert format_mean_std(math.nan, math.nan) == "TBD"


def test_summarise_by_seed_averages_within_seed_first() -> None:
    rows = [
        {"method": "a", "seed": 0, "value": 1.0},
        {"method": "a", "seed": 0, "value": 3.0},  # seed 0 mean = 2
        {"method": "a", "seed": 1, "value": 4.0},  # seed 1 mean = 4
        {"method": "b", "seed": 0, "value": 10.0},
        {"method": "b", "seed": 1, "value": ""},  # missing value is skipped
    ]
    summary = {row["method"]: row for row in summarise_by_seed(rows, ["method"], ["value"])}
    assert summary["a"]["n_seeds"] == 2
    assert summary["a"]["value_mean"] == pytest.approx(3.0)
    assert summary["a"]["value_std"] == pytest.approx(math.sqrt(2.0))
    assert summary["b"]["value_mean"] == pytest.approx(10.0)
    assert math.isnan(summary["b"]["value_std"])


def test_classification_metrics_on_hand_computed_example() -> None:
    #            pred: 0  1  2
    # true 0 (4):     3  1  0
    # true 1 (3):     1  2  0
    # true 2 (3):     0  1  2
    y_true = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 0, 1, 1, 1, 2, 2]
    matrix = confusion_matrix(y_true, y_pred, 3)
    assert matrix.tolist() == [[3, 1, 0], [1, 2, 0], [0, 1, 2]]
    metrics = classification_metrics(y_true, y_pred, 3, positive_class=1)
    precision = [3 / 4, 2 / 4, 2 / 2]
    recall = [3 / 4, 2 / 3, 2 / 3]
    f1 = [2 * p * r / (p + r) for p, r in zip(precision, recall, strict=True)]
    assert metrics["accuracy"] == pytest.approx(0.7)
    assert metrics["precision_macro"] == pytest.approx(np.mean(precision))
    assert metrics["recall_macro"] == pytest.approx(np.mean(recall))
    assert metrics["f1_macro"] == pytest.approx(np.mean(f1))
    assert metrics["f1_weighted"] == pytest.approx(0.4 * f1[0] + 0.3 * f1[1] + 0.3 * f1[2])
    assert metrics["recall_weighted"] == pytest.approx(metrics["accuracy"])
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(f1[1])


def test_classification_metrics_match_scikit_learn() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = rng_from_seed(11, "metrics")
    y_true = rng.integers(0, 4, size=500)
    y_pred = rng.integers(0, 4, size=500)
    ours = classification_metrics(y_true, y_pred, 4)
    for average in ("macro", "weighted"):
        p, r, f, _ = sklearn_metrics.precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        assert ours[f"precision_{average}"] == pytest.approx(p)
        assert ours[f"recall_{average}"] == pytest.approx(r)
        assert ours[f"f1_{average}"] == pytest.approx(f)
    assert ours["accuracy"] == pytest.approx(sklearn_metrics.accuracy_score(y_true, y_pred))


def test_never_predicted_class_scores_zero_not_nan() -> None:
    metrics = classification_metrics([0, 1, 1], [1, 1, 1], 2, positive_class=0)
    assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0 and metrics["f1"] == 0.0
    assert not any(math.isnan(value) for value in metrics.values())


def test_constant_predictor_weighted_scores_match_direct_computation() -> None:
    """The closed-form scores of a constant predictor equal the computed metrics.

    A model that always predicts one class covering a fraction ``p`` of the test set has,
    under support-weighted averaging, accuracy ``p``, recall ``p``, precision ``p**2`` and
    F1 ``2 p**2 / (1 + p)``. Here ``p = 0.35`` on a three-class test set, and
    :func:`constant_predictor_weighted_scores` must match :func:`classification_metrics`
    applied to the constant predictions.
    """
    y_true = np.array([0] * 35 + [1] * 33 + [2] * 32)
    y_pred = np.zeros_like(y_true)
    measured = classification_metrics(y_true, y_pred, 3)
    analytic = constant_predictor_weighted_scores(0.35)
    for key in ("accuracy", "recall_weighted", "precision_weighted", "f1_weighted"):
        assert measured[key] == pytest.approx(analytic[key])


def test_fairness_gap() -> None:
    assert fairness_gap([0.8, 0.9, 0.85]) == pytest.approx(0.1)
    assert math.isnan(fairness_gap([]))


def test_seed_derivation_is_stable_and_label_sensitive() -> None:
    assert derive_seed(3, "client", 1) == derive_seed(3, "client", 1)
    assert derive_seed(3, "client", 1) != derive_seed(3, "client", 2)
    assert derive_seed(3, "client", 1) != derive_seed(4, "client", 1)
    first = rng_from_seed(5, "x").normal(size=4)
    second = rng_from_seed(5, "x").normal(size=4)
    assert np.array_equal(first, second)
