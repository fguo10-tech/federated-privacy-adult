"""Federated benchmark runner: raw CSV files, summaries and the DP trade-off table.

Usage (from the repository root)::

    python -m experiments.run_federated_benchmark --config configs/federated_adult.yaml \
        --suite main|partitions|aggregation|robustness|all \
        --out experiments/results/runs/federated_adult [--seeds 100 101] [--workers 8] \
        [--smoke] [--set training.rounds=5]

Writes into ``--out``: ``config_used.yaml``, ``metadata.json``, ``per_round.csv``,
``per_client.csv``, ``final.csv``, ``summary.csv``, ``per_group.csv``,
``partition_stats.csv``, ``dp_tradeoff.csv`` and ``timings.csv`` (column tuples in
:mod:`src.federated.runner` and below). Centralised baselines are trained once per seed and
copied under every split; the ``shared_across_splits`` column marks those copies
(:data:`src.federated.runner.SHARED_RUN_NOTE`). Figures come from
``python -m experiments.plot_federated --run-dir <out>``.

Never pass the reporting seeds 0–4 while developing or tuning; use seeds >= 100.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io import save_run_metadata, write_csv
from src.common.logging_utils import get_logger, setup_logging
from src.common.metrics import mean_std, summarise_by_seed
from src.federated.config import FederatedConfig, load_federated_config
from src.federated.datasets import load_dataset, resolve_path
from src.federated.partition import PARTITION_STATS_FIELDS, partition_statistics
from src.federated.runner import (
    FINAL_FIELDS,
    FINAL_METRICS,
    PER_CLIENT_FIELDS,
    PER_GROUP_FIELDS,
    PER_ROUND_FIELDS,
    TIMING_FIELDS,
    Task,
    TaskOutput,
    build_partition,
    build_tasks,
    partitions_for_suite,
    run_task,
)

LOGGER = get_logger("experiments.run_federated_benchmark")

SUITES = ("main", "partitions", "aggregation", "robustness")
SUMMARY_KEYS = ("suite", "split", "method", "shared_across_splits")
DP_TRADEOFF_FIELDS = (
    "split",
    "condition",
    "mode",
    "target_epsilon",
    "max_grad_norm",
    "delta",
    "achieved_epsilon_mean",
    "achieved_epsilon_max",
    "noise_multiplier_per_client",
    "accuracy_mean",
    "accuracy_std",
    "f1_mean",
    "f1_std",
    "n_seeds",
)
"""``dp_tradeoff.csv``: one row per split × DP condition (plus ``fedavg`` = no noise, target
``inf``). Achieved ε is the per-run maximum over clients, averaged (and maximised) over
seeds; ``noise_multiplier_per_client`` lists ``client:mean noise multiplier over seeds``."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Run the federated benchmark.")
    parser.add_argument("--config", required=True, help="YAML config (configs/federated_*.yaml)")
    parser.add_argument("--suite", default="main", choices=(*SUITES, "all"))
    parser.add_argument(
        "--out", default=None, help="output folder (default: experiment.output_dir)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds")
    parser.add_argument("--workers", type=int, default=None, help="worker processes")
    parser.add_argument("--smoke", action="store_true", help="apply smoke_overrides (CI scale)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="key.sub=value"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _run_tasks(raw: dict[str, Any], tasks: list[Task], workers: int) -> list[TaskOutput]:
    if workers <= 1 or len(tasks) <= 1:
        return [run_task(raw, task) for task in tasks]
    context = mp.get_context("spawn")
    with context.Pool(processes=min(workers, len(tasks))) as pool:
        return pool.starmap(run_task, [(raw, task) for task in tasks], chunksize=1)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def dp_tradeoff_rows(
    config: FederatedConfig, final: list[dict[str, Any]], per_client: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarise the main suite's DP sweep (and the no-noise ``fedavg`` row) per split."""
    rows = []
    entries = [("fedavg", None)] + [(entry.name, entry) for entry in config.dp.sweep]
    rounds = config.training.rounds
    for split in config.partitions.splits:
        for name, entry in entries:
            selected = [
                r
                for r in final
                if r["suite"] == "main" and r["split"] == split and r["method"] == name
            ]
            if not selected:
                continue
            accuracy = mean_std(_float(r["accuracy"]) for r in selected)
            f1 = mean_std(_float(r["f1"]) for r in selected)
            eps = [_float(r["epsilon_max"]) for r in selected]
            noise: dict[str, list[float]] = defaultdict(list)
            for r in per_client:
                if (
                    r["suite"] == "main"
                    and r["split"] == split
                    and r["condition"] == name
                    and int(r["round"]) == rounds
                    and r["client"] != ""
                ):
                    noise[str(r["client"])].append(_float(r["noise_multiplier"]))
            rows.append(
                {
                    "split": split,
                    "condition": name,
                    "mode": "no_noise" if entry is None else entry.mode,
                    "target_epsilon": (
                        math.inf
                        if entry is None
                        else (entry.target_epsilon if entry.target_epsilon is not None else "")
                    ),
                    "max_grad_norm": "" if entry is None else entry.max_grad_norm,
                    "delta": "" if entry is None else config.dp.delta,
                    "achieved_epsilon_mean": sum(eps) / len(eps),
                    "achieved_epsilon_max": max(eps),
                    "noise_multiplier_per_client": ";".join(
                        f"{client}:{sum(v) / len(v):.4f}" for client, v in noise.items()
                    ),
                    "accuracy_mean": accuracy[0],
                    "accuracy_std": accuracy[1],
                    "f1_mean": f1[0],
                    "f1_std": f1[1],
                    "n_seeds": len(selected),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    setup_logging(args.log_level)
    config, raw = load_federated_config(args.config, overrides=args.overrides, smoke=args.smoke)
    seeds = list(args.seeds) if args.seeds else list(config.experiment.seeds)
    workers = int(args.workers) if args.workers else int(config.runtime.workers)
    out = resolve_path(args.out or config.experiment.output_dir)
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    if any(seed in range(0, 5) for seed in seeds):
        LOGGER.warning(
            "Seeds %s include reporting seeds 0-4: do not use them for development or tuning.",
            seeds,
        )
    try:
        datasets = {seed: load_dataset(config.dataset, config.evaluation, seed) for seed in seeds}
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        LOGGER.error("The UCI Adult files are missing: run `make data` first.")
        return 2

    started = time.perf_counter()
    tasks = [task for suite in suites for task in build_tasks(config, suite, seeds)]
    LOGGER.info(
        "Running %d tasks (%s) on %d worker(s), seeds %s",
        len(tasks),
        ", ".join(suites),
        workers,
        seeds,
    )
    outputs = _run_tasks(raw, tasks, workers)
    elapsed = time.perf_counter() - started

    collected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for output in outputs:
        collected["per_round"].extend(output.per_round)
        collected["per_client"].extend(output.per_client)
        collected["final"].extend(output.final)
        collected["per_group"].extend(output.per_group)
        collected["timing"].extend(output.timing)

    stats = []
    for suite in suites:
        for split, clients in partitions_for_suite(config, suite):
            for seed in seeds:
                partition = build_partition(config, datasets[seed], split, seed, clients)
                for row in partition_statistics(partition, datasets[seed], seed):
                    stats.append({**row, "split": f"{suite}:{row['split']}"})

    write_csv(out / "per_round.csv", collected["per_round"], PER_ROUND_FIELDS)
    write_csv(out / "per_client.csv", collected["per_client"], PER_CLIENT_FIELDS)
    write_csv(out / "final.csv", collected["final"], FINAL_FIELDS)
    write_csv(out / "per_group.csv", collected["per_group"], PER_GROUP_FIELDS)
    write_csv(out / "timings.csv", collected["timing"], TIMING_FIELDS)
    write_csv(out / "partition_stats.csv", stats, PARTITION_STATS_FIELDS)
    for row in collected["final"]:
        row.setdefault("shared_across_splits", "")
    summary = summarise_by_seed(collected["final"], SUMMARY_KEYS, FINAL_METRICS)
    summary_fields = [*SUMMARY_KEYS, "n_seeds"] + [
        f"{metric}_{stat}" for metric in FINAL_METRICS for stat in ("mean", "std")
    ]
    write_csv(out / "summary.csv", summary, summary_fields)
    if "main" in suites:
        write_csv(
            out / "dp_tradeoff.csv",
            dp_tradeoff_rows(config, collected["final"], collected["per_client"]),
            DP_TRADEOFF_FIELDS,
        )
    save_run_metadata(
        out,
        raw,
        seeds,
        extra={
            "suites": suites,
            "workers": workers,
            "smoke": bool(args.smoke),
            "wall_time_s": round(elapsed, 2),
            "num_tasks": len(tasks),
            "eval_on": config.evaluation.eval_on,
        },
    )
    LOGGER.info("Wrote %s in %.1f s", Path(out), elapsed)
    for row in summary:
        LOGGER.info(
            "%-10s %-16s %-28s acc %.4f  f1 %.4f  eps %s",
            row["suite"],
            row["split"],
            row["method"],
            row["accuracy_mean"],
            row["f1_mean"],
            row["epsilon_max_mean"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
