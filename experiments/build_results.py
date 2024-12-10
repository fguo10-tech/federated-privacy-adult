"""Build every results table, Before / After CSV and chart from the reference run.

Reads ``configs/before_after.yaml`` and the CSV files under
``experiments/results/reference/`` and writes:

* ``generated/<block_id>.md`` — Markdown blocks that ``experiments/sync_docs.py`` injects into
  README, RESULTS.md, the project report, the poster and the slides, so a value can never
  differ between placements;
* ``before_after/*.csv`` and ``.png`` — the machine-readable Before / After table of the
  federated benchmark and its charts, side by side;
* ``experiments/results/templates/*.csv`` — header-only schema templates of every result file.

If a CSV of the reference run does not exist, every value that depends on it is rendered as
``TBD``. This script never invents, rounds up or copies a number: paper-reported values come
only from ``src/common/paper_values.py`` and reproduced values only from the CSV files.

Usage::

    python -m experiments.build_results [--config configs/before_after.yaml] [--no-figures]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.before_after import (
    COMPACT_COLUMNS,
    CONVENTION_NOTE,
    BeforeAfterRow,
    Measurement,
    MetricSpec,
    format_value,
    make_row,
    render_markdown,
    row_cells,
)
from src.common.config import load_config
from src.common.io import TRACKED_PACKAGES, ensure_dir, read_csv, read_json, write_csv
from src.common.labels import PAPER_REPORTED, REPRODUCED, TARGET_TBD, TBD
from src.common.logging_utils import get_logger, setup_logging
from src.common.paper_values import PAPER_TABLE1

LOGGER = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "before_after.yaml"

# --------------------------------------------------------------------------------------
# small data helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricCfg:
    """One metric entry of ``configs/before_after.yaml``."""

    key: str
    label: str
    unit: str
    higher_is_better: bool
    near_zero: float
    digits: int
    scale: float

    @property
    def spec(self) -> MetricSpec:
        return MetricSpec(
            self.key, self.label, self.unit, self.higher_is_better, self.near_zero, self.digits
        )


@dataclass(frozen=True)
class Paths:
    """Resolved output and input folders."""

    reference: Path
    generated: Path
    before_after: Path
    templates: Path


def read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file, or an empty list when the file does not exist yet."""
    if not path.is_file():
        LOGGER.warning("missing %s -> the tables that need it will show TBD", _rel(path))
        return []
    return read_csv(path)


def find_row(rows: Iterable[Mapping[str, str]], **match: str) -> Mapping[str, str] | None:
    """First row whose columns equal all ``match`` values."""
    for row in rows:
        if all(row.get(key) == value for key, value in match.items()):
            return row
    return None


def to_float(value: Any) -> float | None:
    """Parse a CSV cell; empty or non-numeric cells become ``None`` (``inf`` is kept)."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def measurement(row: Mapping[str, str] | None, metric: MetricCfg) -> Measurement:
    """Mean ± std of ``metric`` from a summary row (``Measurement()`` if absent = TBD)."""
    if row is None:
        return Measurement()
    mean = to_float(row.get(f"{metric.key}_mean"))
    if mean is None:
        return Measurement()
    std = to_float(row.get(f"{metric.key}_std"))
    count = to_float(row.get("n_seeds"))
    if math.isinf(mean):
        return Measurement(mean, None, int(count) if count is not None else None)
    return Measurement(
        mean * metric.scale,
        None if std is None else std * metric.scale,
        int(count) if count is not None else None,
    )


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _label(rows: Sequence[BeforeAfterRow]) -> str:
    """Source label of a whole table or chart: Reproduced only if something was measured."""
    return REPRODUCED if any(row.status != "tbd" for row in rows) else TARGET_TBD


def _value_cell(value: Measurement, digits: int, unit: str) -> str:
    """A reproduced-value cell with its label (or TBD with the TBD label)."""
    if not value.available:
        return f"{TBD} {TARGET_TBD}"
    return f"{format_value(value, digits)} {unit} {REPRODUCED}".replace("  ", " ")


def _paper_cell(value: float | None, unit: str, digits: int) -> str:
    if value is None:
        return "not reported"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{value:.0f}"
    return f"{text} {unit} {PAPER_REPORTED}".replace("  ", " ")


def _table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(cells) + " |" for cells in body)
    return "\n".join(lines) + "\n"


def _seed_note(rows: Sequence[BeforeAfterRow]) -> str:
    counts = sorted(
        {r.after.n for r in rows if r.after.n} | {r.before.n for r in rows if r.before.n}
    )
    if not counts:
        return "No run found: every value is TBD."
    seeds = " or ".join(str(count) for count in counts)
    return f"Values are mean ± sample standard deviation over {seeds} seeds."


# --------------------------------------------------------------------------------------
# federated learning on UCI Adult: Before / After tables, privacy sweep, ablations
# --------------------------------------------------------------------------------------


def build_federated(cfg: Mapping[str, Any], paths: Paths, figures: bool) -> dict[str, str]:
    """All generated blocks, the CSVs and the charts of the federated benchmark."""
    track = cfg["federated"]
    summary = [row for row in read_rows(paths.reference / track["summary_csv"])
               if row.get("suite", track["suite"]) == track["suite"]]  # fmt: skip
    metrics = [MetricCfg(**entry) for entry in track["metrics"]]
    by_key = {metric.key: metric for metric in metrics}
    labels: dict[str, str] = track["method_labels"]
    splits: dict[str, str] = track["splits"]
    primary = next(iter(splits))
    independent = set(track["split_independent_methods"])
    dataset_line = track["dataset_line"]

    def row_of(split: str, method: str) -> Mapping[str, str] | None:
        found = find_row(summary, split=split, method=method)
        if found is None and method in independent:
            found = find_row(summary, method=method)
        return found

    def value(split: str, method: str, metric: MetricCfg) -> Measurement:
        return measurement(row_of(split, method), metric)

    def pair_rows(
        split: str, pairs: Sequence[Sequence[str]], chosen: Sequence[MetricCfg]
    ) -> list[BeforeAfterRow]:
        rows = []
        for metric in chosen:
            for before, after in pairs:
                rows.append(
                    make_row(metric.spec, labels[before], value(split, before, metric),
                             f"{labels[after]}", value(split, after, metric))
                )  # fmt: skip
        return rows

    full_by_split = {split: pair_rows(split, track["pairs"], metrics) for split in splits}
    summary_rows = pair_rows(
        primary, track["summary_pairs"], [by_key[k] for k in track["summary_metrics"]]
    )
    all_rows = [row for rows in full_by_split.values() for row in rows]
    label = _label(all_rows)
    csv_rel = _rel(paths.before_after / "federated.csv")
    run_seeds = sorted(
        {r.after.n for r in all_rows if r.after.n} | {r.before.n for r in all_rows if r.before.n}
    )
    seeds_text = f"{run_seeds[-1]} seeds" if run_seeds else "5 seeds (not run yet)"
    protocol = (
        f"Protocol for every Before and After value: 20 communication rounds, {seeds_text}, mean ± sample "
        "standard deviation, shared Adult test set (16,281 rows)."
    )

    blocks: dict[str, str] = {}
    parts = [
        f"{dataset_line}\n",
        f"Source label: {label}. {protocol} {_seed_note(all_rows)} Underlying CSV: `{csv_rel}`.\n",
    ]
    for split, split_label in splits.items():
        parts.append(f"**Clients: {split_label}**\n")
        parts.append(render_markdown(full_by_split[split], include_convention=False))
    parts.append(CONVENTION_NOTE + "\n")
    blocks["before_after_full"] = "\n".join(parts)
    blocks["before_after_summary"] = (
        f"{dataset_line}\n\nSource label: {_label(summary_rows)}. Clients: {splits[primary]}. {protocol} "
        f"The full table (all seven metrics, both client splits) is in `experiments/results/RESULTS.md`.\n\n"
        + render_markdown(summary_rows)
    )
    blocks["before_after_compact"] = (
        f"{dataset_line}\n\nSource label: {_label(summary_rows)}. Clients: {splits[primary]}; ε in {{1, 3, 8, no noise}}.\n\n"
        + render_markdown(summary_rows, columns=COMPACT_COLUMNS, include_convention=False)
        + "\nA positive Improvement % always means better after; a negative value is a regression.\n"
    )
    blocks["epsilon_tradeoff"] = _epsilon_table(track, value, by_key, dataset_line)
    blocks["paper_vs_reproduced"] = _federated_paper_table(
        track, value, by_key, primary, dataset_line
    )
    blocks["all_methods"] = _federated_all_methods(track, summary, metrics, dataset_line)
    blocks["findings"] = _federated_findings(track, value, by_key, dataset_line)
    blocks["ablations"] = _federated_ablations(track, paths, by_key)
    blocks["per_group"] = _per_group_table(track, paths)
    blocks["dp_extended"] = _dp_extended(track, paths, by_key, figures)
    blocks["adult_partition_stats"] = _partition_stats(
        paths.reference / track["partition_stats_csv"]
    )

    records_rows: list[BeforeAfterRow] = []
    csv_records: list[dict[str, Any]] = []
    from src.common.before_after import CSV_FIELDS, rows_to_records

    for split, rows in full_by_split.items():
        csv_records.extend(rows_to_records(rows, track=f"federated:{split}"))
        records_rows.extend(rows)
    write_csv(paths.before_after / "federated.csv", csv_records, CSV_FIELDS)
    _epsilon_csv(track, value, by_key, paths.before_after / "epsilon_tradeoff.csv")
    if figures:
        for split, split_label in splits.items():
            _federated_chart(
                track, metrics, value, split, split_label, paths, _label(full_by_split[split])
            )
        _epsilon_chart(track, value, by_key, paths, label)
    return blocks


def _epsilon_table(track, value, by_key, dataset_line: str) -> str:
    header = ["Clients", "Target ε", "Achieved ε (max over clients)", "Accuracy", "F1 (class >50K)",
              "Precision", "Recall", "Accuracy change vs no noise"]  # fmt: skip
    body = []
    measured = False
    for split, split_label in track["splits"].items():
        baseline = value(split, "fedavg", by_key["accuracy"])
        for condition in track["epsilon_conditions"]:
            method = condition["method"]
            accuracy = value(split, method, by_key["accuracy"])
            measured = measured or accuracy.available
            epsilon = value(split, method, by_key["epsilon_max"])
            if accuracy.available and baseline.available and method != "fedavg":
                change = f"{accuracy.mean - baseline.mean:+.4f}"
            elif method == "fedavg" and accuracy.available:
                change = "reference"
            else:
                change = TBD
            body.append([
                split_label, condition["label"], format_value(epsilon, 3),
                format_value(accuracy, 4), format_value(value(split, method, by_key["f1"]), 4),
                format_value(value(split, method, by_key["precision"]), 4),
                format_value(value(split, method, by_key["recall"]), 4), change,
            ])  # fmt: skip
    return (
        f"{dataset_line}\n\nSource label: {REPRODUCED if measured else TARGET_TBD}. Weighted FedAvg, 20 rounds, "
        "5 seeds; ε is the record-level budget each client spends over all rounds (largest value over "
        "the three clients), at the δ set in `configs/federated_adult.yaml`. The ∞ row is the run "
        "with no noise, that is, with no differential privacy at all.\n\n" + _table(header, body)
    )


def _epsilon_csv(track, value, by_key, path: Path) -> None:
    records = []
    for split in track["splits"]:
        for condition in track["epsilon_conditions"]:
            record: dict[str, Any] = {
                "split": split,
                "target_epsilon": condition["label"],
                "method": condition["method"],
            }
            for key in ("epsilon_max", "accuracy", "f1", "precision", "recall"):
                measured = value(split, condition["method"], by_key[key])
                record[f"{key}_mean"] = "" if not measured.available else measured.mean
                record[f"{key}_std"] = "" if measured.std is None else measured.std
            record["n_seeds"] = value(split, condition["method"], by_key["accuracy"]).n or ""
            records.append(record)
    write_csv(path, records)


def _federated_paper_table(track, value, by_key, primary: str, dataset_line: str) -> str:
    rows = [("SVM", "svm"), ("RF", "random_forest"), ("DT", "decision_tree"),
            ("Ours (federated)", "fedavg"), (None, "majority_class")]  # fmt: skip
    metric_map = [("Accuracy", "accuracy", "accuracy"), ("Precision", "precision", "precision"),
                  ("Recall", "recall", "recall"), ("F1", "f1", "f1")]  # fmt: skip
    labels = track["method_labels"]
    notes = {
        "svm": "Centralised on pooled Adult training rows.",
        "random_forest": "Centralised on pooled Adult training rows.",
        "decision_tree": "Centralised on pooled Adult training rows.",
        "fedavg": "Weighted FedAvg, workclass clients, no DP, 20 rounds.",
        "majority_class": "Always predicts <=50K; shown because accuracy alone is misleading.",
    }
    body = []
    for paper_name, method in rows:
        for name, paper_key, key in metric_map:
            paper_value = PAPER_TABLE1[paper_name][paper_key] if paper_name else None
            cell = "not reported" if paper_value is None else f"{paper_value:.4f} {PAPER_REPORTED}"
            body.append(
                [
                    labels[method],
                    name,
                    cell,
                    _value_cell(value(primary, method, by_key[key]), 4, ""),
                    notes[method],
                ]
            )
    text = f"{dataset_line}\n\n" + _table(
        ["Method", "Metric", "Paper-reported", "Reproduced (mean ± std)", "Notes"], body
    )
    text += (
        "\nThe Paper-reported column is shown for reference only: it comes from a different setup, "
        "while the Reproduced column is measured on UCI Adult (binary; precision, recall and F1 are "
        "those of the >50K class) with the workclass clients. No difference between the two columns "
        "is computed anywhere in this repository.\n"
    )
    return text


def _method_label(labels: Mapping[str, str], method: str) -> str:
    """Display name of a method; ``local_only:<client>`` becomes a readable label."""
    if method in labels:
        return labels[method]
    if method.startswith("local_only:"):
        return f"Local-only model of client {method.split(':', 1)[1]}"
    return method


def _federated_all_methods(track, summary, metrics, dataset_line: str) -> str:
    extra = MetricCfg("f1_macro", "F1 (macro)", "fraction", True, 0.0, 4, 1.0)
    columns = (
        list(metrics[:4])
        + ([extra] if any("f1_macro_mean" in row for row in summary) else [])
        + list(metrics[4:])
    )
    labels = track["method_labels"]
    header = ["Clients", "Method"] + [f"{m.label} ({m.unit})" for m in columns]
    body = []
    for split, split_label in track["splits"].items():
        seen: list[str] = []
        for row in summary:
            if row.get("split") == split or row.get("method") in track["split_independent_methods"]:
                if row["method"] not in seen:
                    seen.append(row["method"])
        ordered = [m for m in labels if m in seen] + sorted(m for m in seen if m not in labels)
        for method in ordered:
            row = find_row(summary, split=split, method=method) or find_row(summary, method=method)
            body.append(
                [split_label, _method_label(labels, method)]
                + [format_value(measurement(row, m), m.digits) for m in columns]
            )
    measured = bool(body)
    if not body:
        body = [
            [split_label, TBD] + [TBD] * len(columns) for split_label in track["splits"].values()
        ]
    return (
        f"{dataset_line}\n\nSource label: {REPRODUCED if measured else TARGET_TBD}. Every cell is mean ± sample "
        "standard deviation across seeds on the shared Adult test set (16,281 rows).\n\n"
        + _table(header, body)
    )


def _federated_findings(track, value, by_key, dataset_line: str) -> str:
    primary, control = list(track["splits"])[:2]
    labels = track["method_labels"]
    accuracy, f1, gap = by_key["accuracy"], by_key["f1"], by_key["fairness_gap_accuracy"]
    fedavg = value(primary, "fedavg", accuracy)
    if not fedavg.available:
        return (
            f"{dataset_line}\n\n- {TBD} {TARGET_TBD}: the federated benchmark has not been run.\n"
        )

    def pair(before: str, after: str, metric: MetricCfg, split: str = primary) -> str:
        row = make_row(
            metric.spec,
            labels[before],
            value(split, before, metric),
            labels[after],
            value(split, after, metric),
        )
        cells = row_cells(row)
        return (f"{metric.label}: {labels[before]} {cells['Before Value']} → {labels[after]} {cells['After Value']} "
                f"(Delta {cells['Delta (absolute)']}, Improvement {cells['Improvement %']})")  # fmt: skip

    lines = [f"{dataset_line}", ""]
    lines.append(
        f"- Weighted FedAvg without DP on the {track['splits'][primary]}: accuracy "
        f"{format_value(fedavg, 4)}, F1 {format_value(value(primary, 'fedavg', f1), 4)}. {REPRODUCED}"
    )
    for before in ("centralised_mlp", "local_only", "svm", "random_forest", "decision_tree"):
        lines.append(
            f"- {pair(before, 'fedavg', accuracy)}; {pair(before, 'fedavg', f1)}. {REPRODUCED}"
        )
    for after in ("fedavg_dp_eps8", "fedavg_dp_eps3", "fedavg_dp_eps1"):
        epsilon = value(primary, after, by_key["epsilon_max"])
        lines.append(
            f"- Privacy cost — {pair('fedavg', after, accuracy)}; {pair('fedavg', after, f1)}; "
            f"achieved ε {format_value(epsilon, 2)}. {REPRODUCED}"
        )
    control_value = value(control, "fedavg", accuracy)
    if control_value.available:
        lines.append(
            f"- Non-IID effect — weighted FedAvg without DP: accuracy {format_value(fedavg, 4)} on the "
            f"{track['splits'][primary]} versus {format_value(control_value, 4)} on the {track['splits'][control]} "
            f"(difference {fedavg.mean - control_value.mean:+.4f}). {REPRODUCED}"
        )
    majority = value(primary, "majority_class", accuracy)
    if majority.available:
        lines.append(
            f"- Majority-class reference: accuracy {format_value(majority, 4)}, F1 "
            f"{format_value(value(primary, 'majority_class', f1), 4)} — the floor any accuracy figure must be read against. {REPRODUCED}"
        )
    lines.append(
        f"- Per-client fairness gap (accuracy, max − min over the three workclass domains): centralised MLP "
        f"{format_value(value(primary, 'centralised_mlp', gap), 4)}, weighted FedAvg "
        f"{format_value(value(primary, 'fedavg', gap), 4)}, FedAvg + DP (ε = 3) "
        f"{format_value(value(primary, 'fedavg_dp_eps3', gap), 4)}. {REPRODUCED}"
    )
    return "\n".join(lines) + "\n"


def _partition_stats(path: Path) -> str:
    """Per-client row counts and label rates of the workclass split and the IID control."""
    rows = read_rows(path)
    if not rows:
        return (
            f"{TBD} {TARGET_TBD}: run `make data` and the federated benchmark to compute the "
            "per-client row counts.\n"
        )
    seeds = sorted({int(row["seed"]) for row in rows if row.get("seed", "") != ""})
    first_seed = str(seeds[0]) if seeds else ""
    names = {"workclass": "Workclass split (primary, non-IID)", "iid": "IID control"}
    body = []
    excluded: Mapping[str, str] | None = None
    for row in rows:
        suite, _, split = row["split"].rpartition(":")
        # Only the main suite's two splits belong in the dataset card; the ablation suites
        # (aggregation, partitions, robustness) reuse the same split names.
        if suite not in ("", "main") or split not in names:
            continue
        if row.get("seed", first_seed) != first_seed:
            continue
        if row["client"] == "(excluded)":
            excluded = row
            continue
        label = names[split] + (f", seed {first_seed}" if split == "iid" else "")
        body.append(
            [
                label,
                row["client"],
                f"{int(float(row['rows'])):,}",
                f"{int(float(row['positives'])):,}",
                f"{float(row['positive_rate']):.4f}",
                f"{float(row['share_of_client_rows']):.4f}",
                row.get("composition", "").replace(";", ", ").replace(":", " "),
            ]
        )
    text = (
        f"Computed from the downloaded `adult.data` (training rows only); source CSV `{_rel(path)}`. "
        f"{REPRODUCED}\n\n"
        + _table(
            ["Split", "Client", "Training rows", "Positives (>50K)", "Positive rate",
             "Share of client rows", "Workclass composition"],
            body,
        )
    )  # fmt: skip
    if excluded is not None:
        text += (
            f"\nExcluded from every client split: {int(float(excluded['rows'])):,} training rows "
            f"({excluded.get('composition', '').replace(';', ', ').replace(':', ' ')}; positive rate "
            f"{float(excluded['positive_rate']):.4f}). The workclass split is the same for every seed; "
            "the IID control is reshuffled with each run seed, so its per-client positive rates vary "
            "slightly between seeds while its client sizes stay fixed.\n"
        )
    return text


def _federated_chart(
    track, metrics, value, split: str, split_label: str, paths: Paths, label: str
) -> None:
    from src.common.plotting import Bar, BeforeAfterPanel, before_after_figure

    panels = []
    counts: set[int] = set()
    for metric in metrics:
        bars = []
        for entry in track["chart_methods"]:
            measured = value(split, entry["method"], metric)
            if measured.n:
                counts.add(measured.n)
            bars.append(
                Bar(
                    entry["short"],
                    entry["role"],
                    measured.mean if measured.available else None,
                    measured.std,
                )
            )
        panels.append(
            BeforeAfterPanel(
                metric.label, metric.unit, metric.spec.direction, tuple(bars), min(metric.digits, 3)
            )
        )
    seed_text = f"{max(counts)} seeds" if counts else "no completed run"
    before_after_figure(
        panels,
        paths.before_after / f"federated_{split}.png",
        f"{track['title']} ({split_label}): Before / After",
        label,
        caption=(
            "UCI Adult / Census Income (1994 U.S. Census extraction), shared test set of 16,281 rows; 20 rounds; "
            f"bars: mean across {seed_text}; error bars: sample standard deviation. FedAvg (no DP) is the After method "
            "relative to the centralised and local-only baselines and the Before method relative to the DP runs. "
            "∞ = no privacy guarantee. Communication is zero for methods that never exchange models."
        ),
        columns=3,
    )


def _epsilon_chart(
    track,
    value,
    by_key,
    paths: Paths,
    label: str,
    conditions: Sequence[Mapping[str, str]] | None = None,
    filename: str = "epsilon_tradeoff.png",
    subtitle: str = "ε versus utility",
) -> None:
    """ε versus accuracy and F1: one line per client split, references in grey."""
    import matplotlib.pyplot as plt

    from src.common import plotting

    plotting.apply_style()
    conditions = list(conditions if conditions is not None else track["epsilon_conditions"])
    positions = list(range(len(conditions)))
    primary = next(iter(track["splits"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    seeds: set[int] = set()
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "f1", "F1 (class >50K)")):
        series = []
        for index, (split, split_label) in enumerate(track["splits"].items()):
            measured = [value(split, c["method"], by_key[key]) for c in conditions]
            if all(m.available for m in measured):
                series.append((index, split, split_label, measured))
                seeds.update(m.n for m in measured if m.n)
        for index, split, split_label, measured in series:
            means = [m.mean for m in measured]
            color = plotting.SPLIT_COLORS.get(split, plotting.REFERENCE_COLOR)
            ax.errorbar(
                positions,
                means,
                yerr=[m.std or 0.0 for m in measured],
                color=color,
                marker=plotting.MARKERS[index],
                capsize=3,
                label=f"Weighted FedAvg, {split_label}",
                markeredgecolor=plotting.SURFACE,
                markeredgewidth=1.2,
            )
            # Direct labels on the two ends only; the table next to the chart has every value.
            for position in (positions[0], positions[-1]):
                others = [s[3][position].mean for s in series if s[1] != split]
                above = not others or means[position] >= max(others)
                ax.annotate(
                    f"{means[position]:.3f}",
                    (position, means[position]),
                    xytext=(-16 if position == 0 else 16, 9 if above else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=plotting.INK,
                )
        references = [("centralised_mlp", "Centralised MLP (no privacy)", (0, (5, 3)))]
        if key == "accuracy":
            references.append(("majority_class", "Majority class (always <=50K)", (0, (1, 2))))
        for method, name, dashes in references:
            reference = value(primary, method, by_key[key])
            if reference.available:
                ax.axhline(reference.mean, color=plotting.REFERENCE_COLOR, linewidth=1.3,
                           linestyle=dashes, label=name)  # fmt: skip
        if not series:
            ax.text(0.5, 0.5, "TBD — not run", transform=ax.transAxes, ha="center", va="center",
                    color=plotting.INK_MUTED)  # fmt: skip
        ax.set_xticks(positions)
        ax.set_xticklabels([c["label"] for c in conditions])
        ax.set_xlim(-0.9, len(conditions) - 0.1)
        ax.set_xlabel("Privacy budget ε per client (smaller = more private; ∞ = no noise)")
        ax.set_ylabel(f"{title} on the Adult test set")
        ax.set_title(title, loc="left")
    handles, names = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, names, loc="upper right", ncols=2, bbox_to_anchor=(0.995, 0.965),
                   fontsize=8.5)  # fmt: skip
    seed_text = f"{max(seeds)} seeds" if seeds else "no completed run"
    fig.suptitle(f"{track['title']}: {subtitle} {label}", x=0.01, ha="left", fontsize=11.5,
                 fontweight="bold", color=plotting.INK)  # fmt: skip
    fig.text(
        0.01,
        0.005,
        f"UCI Adult / Census Income (1994 U.S. Census extraction); 20 rounds; mean ± sample standard "
        f"deviation over {seed_text}; record-level ε per client. The y axes do not start at zero: read "
        "the values, not the slopes. The majority-class line is the accuracy of predicting <=50K for "
        "everyone.",
        ha="left", va="bottom", fontsize=8, color=plotting.INK_SECONDARY, wrap=True,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))
    plotting.save_figure(fig, paths.before_after / filename)


def _federated_ablations(
    track: Mapping[str, Any], paths: Paths, by_key: Mapping[str, MetricCfg]
) -> str:
    """Aggregation-rule, partition and robustness ablations of the federated benchmark."""
    rows = read_rows(paths.reference / track["summary_csv"])
    dataset_line = track["dataset_line"]
    shown = [
        by_key[key] for key in ("accuracy", "f1", "precision", "recall", "fairness_gap_accuracy")
    ]
    header = ["Variant"] + [f"{m.label} ({m.unit})" for m in shown]

    def cells(row: Mapping[str, str] | None) -> list[str]:
        return [format_value(measurement(row, metric), metric.digits) for metric in shown]

    def section(
        title: str, entries: list[tuple[str, Mapping[str, str] | None]], note: str, command: str
    ) -> str:
        body = [[name, *cells(row)] for name, row in entries]
        measured = any(cell != TBD for line in body for cell in line[1:])
        label = REPRODUCED if measured else TARGET_TBD
        text = f"**{title}** — source label: {label}. {note}"
        if not measured:
            text += f" Not run yet; command: `{command}`."
        return text + "\n\n" + _table(header, body)

    command = "python -m experiments.run_federated_benchmark --config configs/federated_adult.yaml --suite {}"
    weighted = find_row(rows, suite="aggregation", split="workclass", method="fedavg_weighted")
    unweighted = find_row(rows, suite="aggregation", split="workclass", method="fedavg_unweighted")
    parts = [dataset_line, ""]
    aggregation_note = (
        "Sample-weighted FedAvg (weights n_k / Σ n, the default) against the unweighted mean "
        "(1/N) Σ θ_k, on the workclass clients, whose sizes differ by a factor of about six; "
        "no DP, 20 rounds."
    )
    parts.append(
        section(
            "Aggregation rule",
            [
                ("Weighted FedAvg (default)", weighted),
                ("Unweighted mean (ablation)", unweighted),
            ],
            aggregation_note,
            command.format("aggregation"),
        )
    )
    if weighted is not None and unweighted is not None:
        comparisons = []
        for key in ("accuracy", "f1"):
            compared = make_row(
                by_key[key].spec, "weighted", measurement(weighted, by_key[key]),
                "unweighted", measurement(unweighted, by_key[key]),
            )  # fmt: skip
            comparison = row_cells(compared)
            comparisons.append(
                f"{by_key[key].label}: Delta {comparison['Delta (absolute)']} "
                f"({comparison['Improvement %']})"
            )
        parts.append(
            "Unweighted relative to weighted — "
            + "; ".join(comparisons)
            + f". A negative value means the unweighted mean is lower. {REPRODUCED}\n"
        )
    partition_entries = [
        ("Workclass clients (primary)", find_row(rows, suite="main", split="workclass", method="fedavg")),
        ("IID control, equal sizes", find_row(rows, suite="main", split="iid", method="fedavg")),
        ("IID, sizes matched to the workclass clients", find_row(rows, suite="partitions", split="iid_size_matched", method="fedavg")),
        ("Label skew (Dirichlet)", find_row(rows, suite="partitions", split="label_skew", method="fedavg")),
        ("Quantity skew", find_row(rows, suite="partitions", split="quantity_skew", method="fedavg")),
    ]  # fmt: skip
    parts.append(
        section(
            "Client partition",
            partition_entries,
            "Weighted FedAvg without DP on five partitions of the same client rows; partition parameters "
            "are in `configs/federated_adult.yaml`.",
            command.format("partitions"),
        )
    )
    robustness = [row for row in rows if row.get("suite") == "robustness"]
    names = {
        "weighted_no_attack": "Weighted FedAvg, no attacker",
        "weighted": "Weighted FedAvg under attack",
        "coordinate_median": "Coordinate-wise median under attack",
        "trimmed_mean": "Trimmed mean under attack",
    }
    entries: list[tuple[str, Mapping[str, str] | None]] = []
    for row in robustness:
        method = row["method"]
        if method == "weighted_no_attack":
            entries.append((names[method], row))
            continue
        for rule in ("coordinate_median", "trimmed_mean", "weighted"):
            if method.startswith(rule + "_"):
                attack = method[len(rule) + 1 :].replace("_", " ")
                entries.append((f"{names[rule]} ({attack})", row))
                break
        else:
            entries.append((method, row))
    if not entries:
        entries = [(names[key], None) for key in names]
    config_path = REPO_ROOT / "configs" / "federated_adult.yaml"
    setting = load_config(config_path).get("robustness", {}) if config_path.is_file() else {}
    parts.append(
        section(
            "Robustness to Byzantine clients",
            entries,
            f"IID partition of the client rows into {setting.get('num_clients', '?')} clients of which "
            f"{setting.get('num_byzantine', '?')} send poisoned updates; no DP. Differential privacy is "
            "not a defence against poisoning.",
            command.format("robustness"),
        )
    )
    return "\n".join(parts)


def _per_group_table(track: Mapping[str, Any], paths: Paths) -> str:
    """Accuracy and positive-prediction rate by sex, race and age band (evaluation-only attributes)."""
    from src.common.metrics import summarise_by_seed

    path = paths.reference / Path(track["summary_csv"]).parent / "per_group.csv"
    dataset_line = track["dataset_line"]
    rows = [row for row in read_rows(path) if row.get("suite", track["suite"]) == track["suite"]]
    primary = next(iter(track["splits"]))
    methods = ["centralised_mlp", "fedavg", "fedavg_dp_eps3"]
    labels = track["method_labels"]
    if not rows:
        return f"{dataset_line}\n\n{TBD} {TARGET_TBD}: the per-group breakdown has not been computed.\n"
    tables: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for method in methods:
        selected = [r for r in rows if r["method"] == method and r["split"] == primary]
        if not selected:
            selected = [r for r in rows if r["method"] == method]
            first_split = selected[0]["split"] if selected else ""
            selected = [r for r in selected if r["split"] == first_split]
        summary = summarise_by_seed(
            selected,
            ["attribute", "group"],
            ["n", "accuracy", "positive_prediction_rate", "true_positive_rate"],
        )
        tables[method] = {(r["attribute"], r["group"]): r for r in summary}
    keys = list(next(iter(tables.values())).keys())
    header = ["Attribute", "Group", "Test rows"]
    for method in methods:
        header += [f"Accuracy — {labels[method]}", f"Predicted >50K rate — {labels[method]}"]
    body = []
    for attribute, group in keys:
        first = tables[methods[0]].get((attribute, group), {})
        count = first.get("n_mean")
        is_gap = group.startswith("(")
        line = [attribute, "max − min gap" if is_gap else group,
                "" if is_gap or count is None or count != count else f"{count:,.0f}"]  # fmt: skip
        for method in methods:
            record = tables[method].get((attribute, group), {})
            for key in ("accuracy", "positive_prediction_rate"):
                mean, std = record.get(f"{key}_mean"), record.get(f"{key}_std")
                line.append(format_value(Measurement(mean, std, record.get("n_seeds")), 4)
                            if mean is not None else TBD)  # fmt: skip
        body.append(line)
    return (
        f"{dataset_line}\n\nSource label: {REPRODUCED}. Sex, race and age are evaluation-only attributes: "
        "they are never model inputs for any method. Rows are groups of the shared test set; the "
        f"federated columns use the {track['splits'][primary]}. Mean ± sample standard deviation across "
        "seeds. Small groups (a few hundred rows) are noisy. A gap row is the largest minus the smallest "
        "group value. Differences between groups reflect, among other things, different base rates of "
        "the >50K label in the 1994 data; they are descriptive and are not a fairness certification.\n\n"
        + _table(header, body)
    )


def _dp_extended(
    track: Mapping[str, Any], paths: Paths, by_key: Mapping[str, MetricCfg], figures: bool
) -> str:
    """Extended privacy study (finer ε grid, clipping norms, fixed noise) from its own run."""
    import re

    folder = paths.reference / "federated_adult_dp"
    dataset_line = track["dataset_line"]
    command = (
        "python -m experiments.run_federated_benchmark --config configs/federated_dp.yaml "
        "--suite main --out experiments/results/runs/federated_adult_dp"
    )
    rows = [row for row in read_rows(folder / "summary.csv") if row.get("suite", "main") == "main"]
    if not rows:
        return (
            f"{dataset_line}\n\n{TBD} {TARGET_TBD}: the extended privacy study in "
            f"`configs/federated_dp.yaml` has not been run. Command: `{command}`.\n"
        )
    independent = set(track["split_independent_methods"])

    def value(split: str, method: str, metric: MetricCfg) -> Measurement:
        found = find_row(rows, split=split, method=method)
        if found is None and method in independent:
            found = find_row(rows, method=method)
        return measurement(found, metric)

    methods = {row["method"] for row in rows}
    grid = sorted(
        (float(m.group(1)), name)
        for name in methods
        if (m := re.fullmatch(r"fedavg_dp_eps([0-9.]+)", name))
    )
    clips = sorted(
        (float(m.group(2)), float(m.group(1)), name)
        for name in methods
        if (m := re.fullmatch(r"fedavg_dp_eps([0-9.]+)_clip([0-9.]+)", name))
    )
    sigmas = sorted(
        (float(m.group(1)), name)
        for name in methods
        if (m := re.fullmatch(r"fedavg_dp_sigma([0-9.]+)", name))
    )
    shown = [by_key[key] for key in ("epsilon_max", "accuracy", "f1", "precision", "recall")]
    header = [
        "Clients",
        "Setting",
        "Achieved ε (max over clients)",
        "Accuracy",
        "F1 (class >50K)",
        "Precision",
        "Recall",
    ]

    def line(split_label: str, setting: str, split: str, method: str) -> list[str]:
        cells = [format_value(value(split, method, shown[0]), 3)]
        cells += [format_value(value(split, method, metric), 4) for metric in shown[1:]]
        return [split_label, setting, *cells]

    parts = [
        dataset_line,
        "",
        f"Source label: {REPRODUCED}. Extended privacy study from `configs/federated_dp.yaml` (it inherits "
        "the locked Adult protocol and adds settings); mean ± sample standard deviation across seeds on "
        f"the shared test set; source `{_rel(folder / 'summary.csv')}`. The locked sweep ε in {{1, 3, 8, no "
        "noise}} is reported separately; this table shows where utility actually starts to fall.\n",
        "**Target ε grid**\n",
    ]
    body = []
    for split, split_label in track["splits"].items():
        for epsilon, name in grid:
            body.append(line(split_label, f"target ε = {epsilon:g}", split, name))
        body.append(line(split_label, "no noise (ε = ∞)", split, "fedavg"))
    parts.append(_table(header, body))
    if clips:
        parts.append(
            "**Clipping norm at a fixed target ε** (the default clipping norm is in the config)\n"
        )
        body = [
            line(split_label, f"target ε = {epsilon:g}, clipping norm {clip:g}", split, name)
            for split, split_label in track["splits"].items()
            for clip, epsilon, name in clips
        ]
        parts.append(_table(header, body))
    if sigmas:
        parts.append("**Fixed noise multiplier** (ε is whatever results)\n")
        body = [
            line(split_label, f"noise multiplier {sigma:g}", split, name)
            for split, split_label in track["splits"].items()
            for sigma, name in sigmas
        ]
        parts.append(_table(header, body))
    if figures and grid:
        conditions = [{"method": name, "label": f"{epsilon:g}"} for epsilon, name in grid]
        conditions.append({"method": "fedavg", "label": "∞"})
        _epsilon_chart(
            track, value, by_key, paths, REPRODUCED, conditions,
            filename="epsilon_extended.png", subtitle="ε versus utility, extended grid",
        )  # fmt: skip
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# blocks that do not depend on the Before / After definition
# --------------------------------------------------------------------------------------


def build_static(paths: Paths) -> dict[str, str]:
    """Config listings, CSV schemas, state / action tables, run information."""
    blocks: dict[str, str] = {}
    for config_path in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        block_id = "config_" + config_path.stem
        blocks[block_id] = (
            f"`{_rel(config_path)}` (verbatim; comments carry the provenance tag of every key):\n\n"
            f"```yaml\n{config_path.read_text(encoding='utf-8').rstrip()}\n```\n"
        )
    blocks["csv_schemas"] = _csv_schemas(paths)
    blocks["reference_run_info"] = _run_info(paths)
    blocks.update(_code_tables())
    blocks.update(_diagram_blocks())
    return blocks


DIAGRAM_FILES = {
    "diagram_system_overview": "01-system-overview.md",
    "diagram_federated_detail": "02-federated-detail.md",
    "diagram_trust_boundary": "03-trust-boundary.md",
    "diagram_experiment_flow": "04-experiment-flow.md",
}

REFERENCE_RUNS = ("federated_adult", "federated_adult_dp")


def _code_tables() -> dict[str, str]:
    """Tables generated from constants in the code, so the docs cannot drift from it."""
    from src.federated import privacy
    from src.federated.datasets import adult

    return {
        "adult_attribute_roles": adult.attribute_roles_markdown().rstrip("\n") + "\n",
        "privacy_released_quantities": privacy.released_quantities_markdown().rstrip("\n") + "\n",
    }


def _diagram_blocks() -> dict[str, str]:
    """The first Mermaid block of each architecture file, so every document embeds the same one."""
    blocks: dict[str, str] = {}
    for block_id, name in DIAGRAM_FILES.items():
        path = REPO_ROOT / "docs" / "architecture" / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        start = text.find("```mermaid")
        end = text.find("```", start + 10) if start >= 0 else -1
        if start < 0 or end < 0:
            blocks[block_id] = (
                f"Diagram `docs/architecture/{name}` has not been written yet. {TARGET_TBD}\n"
            )
            continue
        blocks[block_id] = text[start : end + 3] + "\n"
    return blocks


def _csv_schemas(paths: Paths) -> str:
    """Schema table and header-only templates, read from the reference run's own CSV files."""
    body = []
    ensure_dir(paths.templates)
    for csv_path in sorted(paths.reference.rglob("*.csv")):
        relative = csv_path.relative_to(paths.reference)
        if relative.parts[0] in ("generated", "before_after") and relative.parts[0] == "generated":
            continue
        if "figures" in relative.parts or "ablations" in relative.parts:
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
        if not header:
            continue
        template = paths.templates / ("__".join(relative.parts))
        template.write_text(header + "\n", encoding="utf-8")
        body.append(
            [f"`{relative.as_posix()}`", ", ".join(f"`{name}`" for name in header.split(","))]
        )
    if not body:
        return f"{TBD} {TARGET_TBD}: no reference run found, so no CSV schema can be listed yet.\n"
    return (
        "Column lists are read from the header line of each CSV in `experiments/results/reference/`; "
        "header-only copies are in `experiments/results/templates/`.\n\n"
        + _table(["File", "Columns"], body)
    )


def _run_info(paths: Paths) -> str:
    body = []
    for name in REFERENCE_RUNS:
        meta_path = paths.reference / name / "metadata.json"
        if not meta_path.is_file():
            body.append([name, TBD, TBD, TBD, TBD])
            continue
        meta = read_json(meta_path)
        environment = meta.get("environment", {})
        packages = environment.get("packages", {})
        shown = ", ".join(
            f"{k} {v}"
            for k, v in packages.items()
            if v != "not installed" and k in TRACKED_PACKAGES
        )
        seeds = ", ".join(str(seed) for seed in meta.get("seeds", []))
        extra = meta.get("extra", {}) or {}
        runtime = extra.get("wall_clock_seconds")
        runtime_text = (
            f"{float(runtime):.0f} s with {extra.get('workers', '?')} worker processes"
            if runtime
            else "not recorded"
        )
        body.append([
            name, meta.get("created_utc", "?"), seeds,
            f"Python {environment.get('python', '?')} on {environment.get('platform', '?')}; {shown}", runtime_text,
        ])  # fmt: skip
    return _table(
        ["Reference run", "Created (UTC)", "Seeds", "Environment", "Wall-clock time"], body
    )


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Before / After definitions")
    parser.add_argument("--no-figures", action="store_true", help="write tables and CSV only")
    args = parser.parse_args(argv)
    setup_logging("INFO")
    cfg = load_config(args.config)
    paths = Paths(
        reference=REPO_ROOT / cfg["reference_root"],
        generated=ensure_dir(REPO_ROOT / cfg["generated_dir"]),
        before_after=ensure_dir(REPO_ROOT / cfg["before_after_dir"]),
        templates=REPO_ROOT / cfg["templates_dir"],
    )
    figures = not args.no_figures
    if figures:
        from src.common.plotting import use_headless_backend

        use_headless_backend()
    blocks: dict[str, str] = {}
    blocks.update(build_federated(cfg, paths, figures))
    blocks.update(build_static(paths))
    for block_id, text in sorted(blocks.items()):
        (paths.generated / f"{block_id}.md").write_text(text.rstrip("\n") + "\n", encoding="utf-8")
    LOGGER.info("wrote %d generated blocks to %s", len(blocks), _rel(paths.generated))
    LOGGER.info("Before / After CSV and charts are in %s", _rel(paths.before_after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
