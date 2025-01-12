"""Before / After comparison tables (one rule set for every table).

A Before / After table answers "what was it like before using this method, and
what is it like after?". It is *not* the reference table (Paper-reported versus
Reproduced columns).

Rules implemented here (and unit-tested in ``tests/test_before_after.py``):

* ``Delta = After − Before`` in the metric's own unit.
* Higher is better: ``Improvement % = (After − Before) / |Before| × 100``.
* Lower is better:  ``Improvement % = (Before − After) / |Before| × 100``.
  A positive Improvement % therefore always means "better after".
* If Before is zero or near zero, or the two sides are not measured under the
  same config, dataset, workload and seeds, no percentage is computed; the cell
  reads ``N/A — baseline zero / not comparable`` and a note explains why.
* If either side has not been run, the value, Delta and Improvement % are ``TBD`` and the
  row's source label becomes ``[Target / TBD]``.
* A negative Improvement % is reported as it is.

Delta and Improvement % are always computed from unrounded means.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_csv
from .labels import NOT_COMPARABLE, REPRODUCED, SOURCE_LABELS, TARGET_TBD, TBD

HIGHER = "Higher is better"
LOWER = "Lower is better"

COLUMNS = (
    "Metric",
    "Unit",
    "Direction",
    "Before Method",
    "Before Value",
    "After Method",
    "After Value",
    "Delta (absolute)",
    "Improvement %",
    "Source Label",
)

COMPACT_COLUMNS = (
    "Metric",
    "Before Method",
    "Before Value",
    "After Method",
    "After Value",
    "Delta (absolute)",
    "Improvement %",
    "Source Label",
)

CONVENTION_NOTE = (
    "Improvement % convention: a positive value always means better after. "
    "For higher-is-better metrics, Improvement % = (After − Before) / |Before| × 100; "
    "for lower-is-better metrics, Improvement % = (Before − After) / |Before| × 100. "
    "Delta = After − Before in the metric's own unit. Both are computed from unrounded "
    "means measured under the same config, dataset, workload and seeds; a regression "
    "shows as a negative Improvement %."
)

CSV_FIELDS = (
    "track",
    "metric_key",
    "metric",
    "unit",
    "direction",
    "before_method",
    "before_mean",
    "before_std",
    "before_n",
    "after_method",
    "after_mean",
    "after_std",
    "after_n",
    "delta",
    "improvement_pct",
    "status",
    "source_label",
    "note",
)


@dataclass(frozen=True)
class MetricSpec:
    """How one metric is named, oriented and displayed."""

    key: str
    label: str
    unit: str
    higher_is_better: bool
    near_zero: float
    digits: int

    @property
    def direction(self) -> str:
        """The Direction cell: "Higher is better" or "Lower is better"."""
        return HIGHER if self.higher_is_better else LOWER


@dataclass(frozen=True)
class Measurement:
    """A mean over seeds with its sample standard deviation; ``mean=None`` means not run."""

    mean: float | None = None
    std: float | None = None
    n: int | None = None

    @property
    def available(self) -> bool:
        """True when the value was measured (not ``None`` and not NaN)."""
        return self.mean is not None and not (
            isinstance(self.mean, float) and math.isnan(self.mean)
        )

    @property
    def finite(self) -> bool:
        """True when the value was measured and is a finite number."""
        return self.available and math.isfinite(float(self.mean))  # type: ignore[arg-type]


@dataclass(frozen=True)
class BeforeAfterRow:
    """One line of a Before / After table, with its derived Delta and Improvement %."""

    metric: MetricSpec
    before_method: str
    before: Measurement
    after_method: str
    after: Measurement
    source_label: str
    delta: float | None
    improvement_pct: float | None
    status: str  # "ok" | "tbd" | "not_comparable"
    note: str = ""


def compute_delta(before: float, after: float) -> float:
    """``After − Before`` in the metric's own unit."""
    return after - before


def compute_improvement_pct(before: float, after: float, higher_is_better: bool) -> float:
    """Signed improvement in percent; positive always means "better after".

    Raises ``ZeroDivisionError`` for a zero baseline; callers must apply the
    near-zero rule first (see :func:`make_row`).
    """
    if before == 0:
        raise ZeroDivisionError("Improvement % is undefined for a zero baseline")
    change = (after - before) if higher_is_better else (before - after)
    return change / abs(before) * 100.0


def make_row(
    metric: MetricSpec,
    before_method: str,
    before: Measurement,
    after_method: str,
    after: Measurement,
    source_label: str = REPRODUCED,
    comparable: bool = True,
    note: str = "",
) -> BeforeAfterRow:
    """Build one table row, applying the TBD, not-comparable and near-zero rules.

    ``comparable=False`` must be passed whenever the two sides were not measured
    under the same config, dataset, workload and seeds; neither Delta nor
    Improvement % is then computed.
    """
    if source_label not in SOURCE_LABELS:
        raise ValueError(f"source_label must be one of {SOURCE_LABELS}, got {source_label!r}")
    if not before.available or not after.available:
        # A row with an unmeasured side is a target, whatever label the caller intended.
        return BeforeAfterRow(
            metric, before_method, before, after_method, after, TARGET_TBD, None, None, "tbd",
            note or "One side has not been run yet.",
        )  # fmt: skip
    if not comparable:
        return BeforeAfterRow(
            metric, before_method, before, after_method, after, source_label, None, None,
            "not_comparable",
            note or "The two sides were not measured under the same config, data and seeds.",
        )  # fmt: skip
    if not before.finite or not after.finite:
        return BeforeAfterRow(
            metric, before_method, before, after_method, after, source_label, None, None,
            "not_comparable",
            note or "One side is infinite, so a difference or ratio is not meaningful.",
        )  # fmt: skip
    before_mean, after_mean = float(before.mean), float(after.mean)  # type: ignore[arg-type]
    delta = compute_delta(before_mean, after_mean)
    if abs(before_mean) <= metric.near_zero:
        return BeforeAfterRow(
            metric, before_method, before, after_method, after, source_label, delta, None,
            "not_comparable",
            note
            or f"Before is zero or near zero (|Before| <= {metric.near_zero:g} {metric.unit}), "
            "so a percentage would be meaningless.",
        )  # fmt: skip
    pct = compute_improvement_pct(before_mean, after_mean, metric.higher_is_better)
    return BeforeAfterRow(
        metric, before_method, before, after_method, after, source_label, delta, pct, "ok", note
    )


def format_value(measurement: Measurement, digits: int) -> str:
    """``mean ± std`` for a table cell, ``TBD`` if not run, ``∞`` if infinite."""
    if not measurement.available:
        return TBD
    mean = float(measurement.mean)  # type: ignore[arg-type]
    if math.isinf(mean):
        return "∞"
    std = measurement.std
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f} (n={measurement.n if measurement.n is not None else 1})"
    return f"{mean:.{digits}f} ± {float(std):.{digits}f}"


def format_delta(row: BeforeAfterRow) -> str:
    """Signed Delta cell."""
    if row.status == "tbd":
        return TBD
    if row.delta is None:
        return NOT_COMPARABLE
    return f"{row.delta:+.{row.metric.digits}f}"


def format_improvement(row: BeforeAfterRow) -> str:
    """Signed Improvement % cell with one decimal."""
    if row.status == "tbd":
        return TBD
    if row.improvement_pct is None:
        return NOT_COMPARABLE
    return f"{row.improvement_pct:+.1f}%"


def row_cells(row: BeforeAfterRow) -> dict[str, str]:
    """All ten display cells of a row, keyed by the column names in :data:`COLUMNS`."""
    return {
        "Metric": row.metric.label,
        "Unit": row.metric.unit,
        "Direction": row.metric.direction,
        "Before Method": row.before_method,
        "Before Value": format_value(row.before, row.metric.digits),
        "After Method": row.after_method,
        "After Value": format_value(row.after, row.metric.digits),
        "Delta (absolute)": format_delta(row),
        "Improvement %": format_improvement(row),
        "Source Label": row.source_label,
    }


def render_markdown(
    rows: Iterable[BeforeAfterRow],
    columns: Sequence[str] = COLUMNS,
    include_notes: bool = True,
    include_convention: bool = True,
) -> str:
    """Render rows as a GitHub Markdown table, followed by notes and the sign convention."""
    rows = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    notes: list[str] = []
    for row in rows:
        cells = row_cells(row)
        lines.append("| " + " | ".join(cells[column] for column in columns) + " |")
        if row.note and row.status == "not_comparable":
            text = f"{row.metric.label}, {row.before_method} → {row.after_method}: {row.note}"
            if text not in notes:
                notes.append(text)
    output = "\n".join(lines)
    if include_notes and notes:
        output += "\n\n" + "\n".join(f"- {note}" for note in notes)
    if include_convention:
        output += "\n\n" + CONVENTION_NOTE
    return output + "\n"


def rows_to_records(rows: Iterable[BeforeAfterRow], track: str) -> list[dict[str, Any]]:
    """Flatten rows into plain dicts (the CSV that sits next to every table and chart)."""
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "track": track,
                "metric_key": row.metric.key,
                "metric": row.metric.label,
                "unit": row.metric.unit,
                "direction": row.metric.direction,
                "before_method": row.before_method,
                "before_mean": _num(row.before.mean),
                "before_std": _num(row.before.std),
                "before_n": row.before.n if row.before.n is not None else "",
                "after_method": row.after_method,
                "after_mean": _num(row.after.mean),
                "after_std": _num(row.after.std),
                "after_n": row.after.n if row.after.n is not None else "",
                "delta": _num(row.delta),
                "improvement_pct": _num(row.improvement_pct),
                "status": row.status,
                "source_label": row.source_label,
                "note": row.note,
            }
        )
    return records


def _num(value: float | None) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return float(value)


def write_rows_csv(rows: Iterable[BeforeAfterRow], path: str | Path, track: str) -> Path:
    """Write the machine-readable version of a Before / After table."""
    return write_csv(path, rows_to_records(rows, track), CSV_FIELDS)
