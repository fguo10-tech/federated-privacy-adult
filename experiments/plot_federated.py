"""Figures of a federated benchmark run, each saved next to the CSV it was drawn from.

Usage::

    python -m experiments.plot_federated --run-dir experiments/results/runs/federated_adult

Writes into ``<run-dir>/figures/``:

* ``dp_tradeoff.png`` — ε versus accuracy and ε versus F1 (x positions: the finite target ε
  values, then "no noise"; one line per split; error bars = std across seeds; grey dashed
  line = centralised MLP);
* ``accuracy_per_round.png`` — accuracy of every federated / local-only condition per round;
* ``per_client_accuracy.png`` — final accuracy on each client domain's evaluation rows;
* ``communication_cost.png`` — cumulative parameter bytes (up + down) per round;
* ``partition_distribution.png`` — rows and positive rate per client and split.

Colour carries role (``src.common.plotting``): baselines and local-only training are
``BEFORE_COLOR``, FedAvg (with or without DP) is ``AFTER_COLOR``, references and ablations are
``REFERENCE_COLOR``; the two client splits use ``SPLIT_COLORS``. Every number drawn is
``[Reproduced]`` only if the run directory is a real run of this code.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common import plotting
from src.common.io import read_csv, write_csv
from src.common.labels import REPRODUCED
from src.common.logging_utils import get_logger, setup_logging
from src.common.metrics import mean_std

LOGGER = get_logger("experiments.plot_federated")

BASELINES = (
    "centralised_mlp",
    "centralised_mlp_client_rows",
    "svm",
    "random_forest",
    "decision_tree",
    "majority_class",
)


def role_of(method: str) -> str:
    """Plot role of a method name: before (baselines), after (FedAvg), reference (rest)."""
    if method in BASELINES or method.startswith("local_only"):
        return "before"
    if method == "fedavg" or method.startswith("fedavg_dp") or method == "fedavg_weighted":
        return "after"
    return "reference"


def split_color(split: str) -> str:
    """Colour of a split; the toy and FEMNIST natural splits reuse the workclass colour."""
    if split in plotting.SPLIT_COLORS:
        return plotting.SPLIT_COLORS[split]
    if split in ("domain", "writer"):
        return plotting.SPLIT_COLORS["workclass"]
    return plotting.REFERENCE_COLOR


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def plot_dp_tradeoff(run_dir: Path, figures: Path) -> Path | None:
    """ε versus accuracy and F1, one line per split, centralised MLP as a grey reference."""
    import matplotlib.pyplot as plt

    path = run_dir / "dp_tradeoff.csv"
    if not path.is_file():
        return None
    rows = read_csv(path)
    summary = read_csv(run_dir / "summary.csv")
    targets = sorted(
        {_f(r["target_epsilon"]) for r in rows if math.isfinite(_f(r["target_epsilon"]))}
    )
    labels = [f"{t:g}" for t in targets] + ["no noise"]
    positions = {t: i for i, t in enumerate(targets)}
    plotted = []
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), squeeze=False)
    for ax, metric in zip(axes[0], ("accuracy", "f1"), strict=True):
        for split in sorted({r["split"] for r in rows}):
            points = []
            for r in rows:
                if r["split"] != split or r["mode"] not in ("target_epsilon", "no_noise"):
                    continue
                target = _f(r["target_epsilon"])
                # Clipping / fixed-noise ablations share a target with the main entry; only the
                # canonical "fedavg_dp_eps<target>" entry (and no noise) is on the curve.
                if math.isfinite(target) and r["condition"] != f"fedavg_dp_eps{target:g}":
                    continue
                x = len(targets) if math.isinf(target) else positions[target]
                points.append((x, _f(r[f"{metric}_mean"]), _f(r[f"{metric}_std"]), r["condition"]))
            points.sort()
            if not points:
                continue
            ax.errorbar(
                [p[0] for p in points],
                [p[1] for p in points],
                yerr=[0.0 if math.isnan(p[2]) else p[2] for p in points],
                color=split_color(split),
                marker="o",
                capsize=3,
                label=split,
            )
            for x, mean, std, condition in points:
                plotted.append({"metric": metric, "split": split, "condition": condition, "x_label": labels[x], "mean": mean, "std": std})  # fmt: skip
        reference = [
            r for r in summary if r["method"] == "centralised_mlp" and r["suite"] == "main"
        ]
        if reference:
            value = _f(reference[0][f"{metric}_mean"])
            ax.axhline(value, color=plotting.REFERENCE_COLOR, linestyle="--", linewidth=1.2,
                       label="centralised MLP")  # fmt: skip
            plotted.append({"metric": metric, "split": "reference", "condition": "centralised_mlp", "x_label": "", "mean": value, "std": _f(reference[0][f"{metric}_std"])})  # fmt: skip
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("per-client target ε (δ fixed)")
        ax.set_ylabel(f"{metric} (positive class)" if metric == "f1" else metric)
        ax.set_title(f"ε versus {metric}", loc="left")
        ax.legend(loc="best", fontsize=7.5)
    fig.suptitle(f"Privacy–utility trade-off {REPRODUCED}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out = plotting.save_figure(fig, figures / "dp_tradeoff.png")
    write_csv(figures / "dp_tradeoff.csv", plotted)
    return out


def plot_accuracy_per_round(run_dir: Path, figures: Path) -> Path | None:
    """Mean ± std accuracy per round of every federated / local-only condition, per split."""
    import matplotlib.pyplot as plt

    rows = [r for r in read_csv(run_dir / "per_round.csv") if ":" not in r["condition"]]
    if not rows:
        return None
    groups: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        groups[(r["suite"], r["split"], r["condition"])][int(r["round"])].append(_f(r["accuracy"]))
    panels = sorted({(suite, split) for suite, split, _ in groups})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.5), squeeze=False)
    plotted = []
    for ax, (suite, split) in zip(axes[0], panels, strict=True):
        conditions = sorted(c for s, sp, c in groups if s == suite and sp == split)
        for index, condition in enumerate(conditions):
            series = groups[(suite, split, condition)]
            rounds = sorted(series)
            stats = [mean_std(series[r]) for r in rounds]
            role = role_of(condition)
            plotting.line_with_band(
                ax,
                rounds,
                [s[0] for s in stats],
                [s[1] for s in stats],
                label=condition,
                color=plotting.ROLE_COLORS[role],
                marker=plotting.MARKERS[index % len(plotting.MARKERS)],
                linestyle="-" if condition in ("fedavg", "local_only") else "--",
            )
            for r, (mean, std) in zip(rounds, stats, strict=True):
                plotted.append({"suite": suite, "split": split, "condition": condition, "round": r, "accuracy_mean": mean, "accuracy_std": std})  # fmt: skip
        ax.set_xlabel("round")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{suite} · {split}", loc="left")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Accuracy per round {REPRODUCED}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    out = plotting.save_figure(fig, figures / "accuracy_per_round.png")
    write_csv(figures / "accuracy_per_round.csv", plotted)
    return out


def plot_per_client_accuracy(run_dir: Path, figures: Path) -> Path | None:
    """Final per-domain accuracy (main suite) by condition: one small panel per split × domain."""
    import matplotlib.pyplot as plt

    rows = read_csv(run_dir / "per_client.csv")
    final_round: dict[tuple[str, str, str], int] = {}
    for r in rows:
        key = (r["suite"], r["split"], r["condition"])
        final_round[key] = max(final_round.get(key, 0), int(r["round"]))
    values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r["suite"] != "main" or r["domain"] == "" or r["domain_accuracy"] == "":
            continue
        if int(r["round"]) != final_round[(r["suite"], r["split"], r["condition"])]:
            continue
        values[(r["split"], r["domain"], r["condition"])][r["seed"]].append(
            _f(r["domain_accuracy"])
        )
    if not values:
        return None
    panels = sorted({(split, domain) for split, domain, _ in values})
    columns = len({domain for _, domain in panels})
    nrows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(4.6 * columns, 3.6 * nrows), squeeze=False)
    plotted = []
    for index, (split, domain) in enumerate(panels):
        ax = axes[index // columns][index % columns]
        conditions = sorted(c for sp, d, c in values if sp == split and d == domain)
        means, stds = [], []
        for condition in conditions:
            per_seed = [sum(v) / len(v) for v in values[(split, domain, condition)].values()]
            mean, std = mean_std(per_seed)
            means.append(mean)
            stds.append(std)
            plotted.append({"split": split, "domain": domain, "condition": condition, "accuracy_mean": mean, "accuracy_std": std, "n_seeds": len(per_seed)})  # fmt: skip
        plotting.bar_panel(
            ax,
            conditions,
            means,
            stds,
            [plotting.ROLE_COLORS[role_of(c)] for c in conditions],
            ylabel="accuracy",
            title=f"{split} split · {domain} rows",
        )
    for index in range(len(panels), nrows * columns):
        axes[index // columns][index % columns].set_visible(False)
    fig.suptitle(f"Accuracy on each client domain's evaluation rows {REPRODUCED}", x=0.01,
                 ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    out = plotting.save_figure(fig, figures / "per_client_accuracy.png")
    write_csv(figures / "per_client_accuracy.csv", plotted)
    return out


def plot_communication(run_dir: Path, figures: Path) -> Path | None:
    """Cumulative parameter bytes (uplink + downlink) of the FedAvg conditions per round."""
    import matplotlib.pyplot as plt

    rows = [r for r in read_csv(run_dir / "per_round.csv") if r["condition"].startswith("fedavg")]
    if not rows:
        return None
    series: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["condition"] not in ("fedavg", "fedavg_weighted"):
            continue
        total = (_f(r["bytes_up_cum"]) + _f(r["bytes_down_cum"])) / 1e6
        series[(r["suite"], r["split"])][int(r["round"])].append(total)
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    plotted = []
    for index, ((suite, split), by_round) in enumerate(sorted(series.items())):
        rounds = sorted(by_round)
        means = [sum(by_round[r]) / len(by_round[r]) for r in rounds]
        ax.plot(rounds, means, color=split_color(split), marker=plotting.MARKERS[index % 7],
                label=f"{suite} · {split}")  # fmt: skip
        for r, m in zip(rounds, means, strict=True):
            plotted.append({"suite": suite, "split": split, "round": r, "megabytes_cumulative": m})
    ax.set_ylim(bottom=0)
    ax.set_xlabel("round")
    ax.set_ylabel("cumulative parameter traffic (MB)")
    ax.set_title(f"Communication cost of FedAvg {REPRODUCED}", loc="left")
    ax.legend(fontsize=8)
    out = plotting.save_figure(fig, figures / "communication_cost.png")
    write_csv(figures / "communication_cost.csv", plotted)
    return out


def plot_partitions(run_dir: Path, figures: Path) -> Path | None:
    """Rows and positive rate per client of every split (first seed of the run)."""
    import matplotlib.pyplot as plt

    rows = [r for r in read_csv(run_dir / "partition_stats.csv") if r["client"] != "(excluded)"]
    if not rows:
        return None
    first_seed = min(int(r["seed"]) for r in rows)
    rows = [r for r in rows if int(r["seed"]) == first_seed]
    splits = list(dict.fromkeys(r["split"] for r in rows))
    has_rate = any(r["positive_rate"] != "" for r in rows)
    ncols = 2 if has_rate else 1
    fig, axes = plt.subplots(
        len(splits), ncols, figsize=(4.8 * ncols, 3.0 * len(splits)), squeeze=False
    )
    for index, split in enumerate(splits):
        subset = [r for r in rows if r["split"] == split]
        names = [r["client"] for r in subset]
        color = split_color(split.split(":")[-1])
        plotting.bar_panel(axes[index][0], names, [_f(r["rows"]) for r in subset], [None] * len(subset),
                           [color] * len(subset), ylabel="training rows", title=f"{split} · rows",
                           digits=0)  # fmt: skip
        if has_rate:
            plotting.bar_panel(axes[index][1], names, [_f(r["positive_rate"]) for r in subset],
                               [None] * len(subset), [color] * len(subset),
                               ylabel="positive rate", title=f"{split} · positive rate")  # fmt: skip
    fig.suptitle(f"Client partitions (seed {first_seed}) {REPRODUCED}", x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    out = plotting.save_figure(fig, figures / "partition_distribution.png")
    write_csv(figures / "partition_distribution.csv", rows)
    return out


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Plot a federated benchmark run.")
    parser.add_argument("--run-dir", required=True, help="output folder of run_federated_benchmark")
    args = parser.parse_args(argv)
    setup_logging()
    plotting.use_headless_backend()
    plotting.apply_style()
    run_dir = Path(args.run_dir)
    if not (run_dir / "final.csv").is_file():
        LOGGER.error("%s has no final.csv; run experiments.run_federated_benchmark first", run_dir)
        return 2
    figures = run_dir / "figures"
    written = []
    for function in (
        plot_dp_tradeoff,
        plot_accuracy_per_round,
        plot_per_client_accuracy,
        plot_communication,
        plot_partitions,
    ):
        result = function(run_dir, figures)
        if result is not None:
            written.append(result)
    LOGGER.info("Wrote %d figures into %s", len(written), figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
