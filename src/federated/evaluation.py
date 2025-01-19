"""Test-set metrics, per-client (per-domain) breakdowns and per-group fairness.

The same function evaluates every method — federated, local-only and centralised baselines —
on the same shared evaluation set, so all numbers are comparable:

* **overall**: accuracy, and precision / recall / F1 of the positive class (``>50K`` for
  Adult; about 24% of rows, so accuracy alone is misleading), plus macro F1. For a
  multi-class dataset without a positive class (FEMNIST), precision / recall / F1 are macro.
* **per client**: the metrics on the evaluation rows of each natural domain (for Adult: the
  Private / Government / Self-employed test rows). The **per-client fairness gap** is
  max − min of that per-domain accuracy (``fairness_gap_accuracy``), likewise for F1. It is
  computed identically for the workclass split, the IID control and the baselines.
* **per group** on the evaluation-only attributes (sex, race, age band): accuracy,
  positive-prediction rate and true-positive rate, with max − min gaps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.common.metrics import classification_metrics, fairness_gap

from .datasets import FederatedDataset, TabularSplit

OVERALL_METRICS = ("accuracy", "precision", "recall", "f1", "f1_macro")
GROUP_FIELDS = (
    "attribute",
    "group",
    "n",
    "accuracy",
    "positive_prediction_rate",
    "true_positive_rate",
)


@dataclass(frozen=True)
class Evaluation:
    """All evaluation numbers of one model on the shared evaluation set."""

    overall: dict[str, float]
    per_domain: dict[str, dict[str, float]]
    fairness_gap_accuracy: float
    fairness_gap_f1: float
    per_group: list[dict[str, object]]
    group_gaps: dict[str, dict[str, float]]


def metrics_for(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, positive_class: int | None
) -> dict[str, float]:
    """:data:`OVERALL_METRICS` (NaN for an empty set)."""
    if len(y_true) == 0:
        return {name: math.nan for name in OVERALL_METRICS} | {"n": 0}
    raw = classification_metrics(y_true, y_pred, num_classes, positive_class=positive_class)
    if positive_class is None:
        precision, recall, f1 = raw["precision_macro"], raw["recall_macro"], raw["f1_macro"]
    else:
        precision, recall, f1 = raw["precision"], raw["recall"], raw["f1"]
    return {
        "accuracy": raw["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": raw["f1_macro"],
        "n": int(len(y_true)),
    }


def per_domain_metrics(
    split: TabularSplit,
    y_pred: np.ndarray,
    domain_names: Sequence[str],
    num_classes: int,
    positive_class: int | None,
) -> dict[str, dict[str, float]]:
    """Metrics on the evaluation rows of each natural domain."""
    result = {}
    for code, name in enumerate(domain_names):
        mask = split.domain == code
        result[name] = metrics_for(split.y[mask], y_pred[mask], num_classes, positive_class)
    return result


def group_breakdown(
    split: TabularSplit,
    y_pred: np.ndarray,
    attributes: Sequence[str],
    positive_class: int | None,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    """Per-group accuracy, positive-prediction rate and TPR, plus max − min gaps.

    Groups are reported in sorted order; attributes absent from the dataset are skipped.
    For a dataset without a positive class only accuracy is filled in.
    """
    rows: list[dict[str, object]] = []
    gaps: dict[str, dict[str, float]] = {}
    for attribute in attributes:
        if attribute not in split.groups:
            continue
        values = split.groups[attribute]
        accuracies, ppr, tpr = [], [], []
        for group in sorted({str(v) for v in values}):
            mask = np.asarray([str(v) == group for v in values])
            truth, pred = split.y[mask], y_pred[mask]
            accuracy = float(np.mean(truth == pred)) if mask.any() else math.nan
            if positive_class is not None:
                positive_rate = float(np.mean(pred == positive_class))
                actual = truth == positive_class
                true_positive = (
                    float(np.mean(pred[actual] == positive_class)) if actual.any() else math.nan
                )
            else:
                positive_rate, true_positive = math.nan, math.nan
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "n": int(mask.sum()),
                    "accuracy": accuracy,
                    "positive_prediction_rate": positive_rate,
                    "true_positive_rate": true_positive,
                }
            )
            accuracies.append(accuracy)
            ppr.append(positive_rate)
            tpr.append(true_positive)
        gaps[attribute] = {
            "accuracy": fairness_gap([v for v in accuracies if not math.isnan(v)]),
            "positive_prediction_rate": fairness_gap([v for v in ppr if not math.isnan(v)]),
            "true_positive_rate": fairness_gap([v for v in tpr if not math.isnan(v)]),
        }
    return rows, gaps


def evaluate_predictions(
    dataset: FederatedDataset, y_pred: np.ndarray, group_attributes: Sequence[str]
) -> Evaluation:
    """Evaluate predictions for every row of ``dataset.test`` (the shared evaluation set)."""
    split = dataset.test
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_pred.shape != split.y.shape:
        raise ValueError("one prediction per evaluation row is required")
    overall = metrics_for(split.y, y_pred, dataset.num_classes, dataset.positive_class)
    per_domain = per_domain_metrics(
        split, y_pred, dataset.domain_names, dataset.num_classes, dataset.positive_class
    )
    accuracies = [m["accuracy"] for m in per_domain.values() if not math.isnan(m["accuracy"])]
    f1s = [m["f1"] for m in per_domain.values() if not math.isnan(m["f1"])]
    rows, gaps = group_breakdown(split, y_pred, group_attributes, dataset.positive_class)
    return Evaluation(
        overall=overall,
        per_domain=per_domain,
        fairness_gap_accuracy=fairness_gap(accuracies),
        fairness_gap_f1=fairness_gap(f1s),
        per_group=rows,
        group_gaps=gaps,
    )
