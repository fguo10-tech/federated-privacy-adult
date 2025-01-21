"""Run one benchmark task (a federated condition, local-only training or the baselines).

A :class:`Task` names a suite, split, condition and seed. :func:`run_task` executes it in
the current process and returns plain row dicts for the result CSV files whose columns are
the module-level tuples below. Every task is a pure function of (config, task): the run
seed drives the partition, the model initialisation (``derive_seed(seed, "model_init")``),
every client's minibatch order and DP noise, and the baselines' ``random_state``. Wall-clock
time is reported separately (:data:`TIMING_FIELDS`) so that result rows are identical for
any number of worker processes.

Federated conditions run through the real Flower apps (:mod:`.client_app`,
:mod:`.server_app`) on the in-process grid (:mod:`.inprocess`) by default, or on Flower's
Ray engine with ``federation.engine: ray``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.common.config import from_dict, to_dict
from src.common.logging_utils import get_logger
from src.common.seeding import derive_seed

from .client import LocalTrainingSpec, predict_logits, spec_to_config
from .config import AttackConfig, DPSweepEntry, FederatedConfig
from .datasets import FederatedDataset, load_dataset
from .evaluation import Evaluation, evaluate_predictions
from .model import build_model, initial_weights, parameter_names
from .partition import Partition, make_partition
from .privacy import GUARANTEE

LOGGER = get_logger(__name__)

PER_ROUND_FIELDS = (
    "suite",
    "split",
    "condition",
    "seed",
    "round",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "train_loss",
    "bytes_up_cum",
    "bytes_down_cum",
    "epsilon_max",
    "delta",
)
"""``per_round.csv``: metrics of the evaluated model on the shared evaluation set after each
round (round 0 = initial model). ``train_loss`` is the sample-weighted mean client loss of
that round; it is EMPTY for DP conditions, because a DP client does not send its training loss
(an un-noised statistic of its private rows). ``epsilon_max`` is the largest cumulative
per-client ε (``inf`` without DP)."""

PER_CLIENT_FIELDS = (
    "suite",
    "split",
    "condition",
    "seed",
    "round",
    "client",
    "domain",
    "n_train",
    "sample_rate",
    "noise_multiplier",
    "epsilon",
    "update_norm",
    "train_loss",
    "local_eval_accuracy",
    "domain_n",
    "domain_accuracy",
    "domain_precision",
    "domain_recall",
    "domain_f1",
    "shared_across_splits",
)
"""``per_client.csv``: row ``k`` of a round pairs training client ``k`` (its size, sampling
rate, noise multiplier, cumulative ε, update norm, loss, and the accuracy of the global model
on its own training rows reported by the Flower evaluate handler) with natural domain ``k``
(the evaluated model's metrics on that domain's evaluation rows). Client ``k`` *is* domain
``k`` only on the natural split; on other splits the pairing is by index only. For
``local_only`` the evaluated model of row ``k`` is client ``k``'s own model; for baselines the
row carries only domain metrics.

``train_loss`` is empty for DP conditions (not sent; see
:data:`src.federated.privacy.RELEASED_QUANTITIES`). ``local_eval_accuracy`` is filled only when
``federation.fraction_evaluate > 0``; the default is 0 for every condition, because evaluating
on a client's own training rows releases an un-noised statistic of them, and it would never
be filled for a DP condition. All reported metrics come from the server-side evaluation on the
shared evaluation set."""

FINAL_FIELDS = (
    "suite",
    "split",
    "method",
    "seed",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "train_loss",
    "fairness_gap_accuracy",
    "fairness_gap_f1",
    "rounds",
    "bytes_total",
    "bytes_up",
    "bytes_down",
    "epsilon_max",
    "delta",
    "train_rows",
    "num_clients",
    "shared_across_splits",
)
SHARED_RUN_NOTE = (
    "``shared_across_splits``: empty for rows produced by a run of their own split. For the "
    "centralised baselines, which do not depend on the client split, it lists the splits "
    "(e.g. ``workclass;iid``) under which the SAME single training run is written again for "
    "convenience; those rows are copies, not independent runs, and must not be counted twice."
)
"""Documentation of the ``shared_across_splits`` column (all result CSV files)."""

"""``final.csv``: one row per suite × split × method × seed with final-round values.
Baselines have ``bytes_total = 0`` and ``epsilon_max = inf``; ``local_only`` appears once per
client model (``local_only:<client>``) and once as the mean of those rows (``local_only``)."""

PER_GROUP_FIELDS = (
    "suite",
    "split",
    "method",
    "seed",
    "attribute",
    "group",
    "n",
    "accuracy",
    "positive_prediction_rate",
    "true_positive_rate",
    "shared_across_splits",
)
"""``per_group.csv``: final model, evaluation-only attributes. ``group == "(max-min gap)"``
rows hold the gap across the groups of that attribute."""

TIMING_FIELDS = ("suite", "split", "condition", "seed", "wall_time_s", "shared_across_splits")
"""``timings.csv``: wall-clock seconds per task (machine-dependent; kept out of results)."""

FINAL_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "train_loss",
    "fairness_gap_accuracy",
    "fairness_gap_f1",
    "bytes_total",
    "epsilon_max",
    "train_rows",
)
"""Numeric columns of ``final.csv`` summarised as ``<metric>_mean`` / ``<metric>_std``."""


@dataclass(frozen=True)
class Task:
    """One unit of work for a worker process."""

    suite: str
    kind: str  # "federated" | "local_only" | "baselines"
    split: str
    condition: str
    seed: int
    aggregation_rule: str
    dp: DPSweepEntry | None
    num_clients: int | None
    attack: AttackConfig | None
    num_byzantine: int
    report_splits: tuple[str, ...]


@dataclass
class TaskOutput:
    """Rows produced by one task."""

    per_round: list[dict[str, Any]]
    per_client: list[dict[str, Any]]
    final: list[dict[str, Any]]
    per_group: list[dict[str, Any]]
    timing: list[dict[str, Any]]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def quiet_flower_logging() -> None:
    """Flower logs every round at INFO through its own handler; keep it to errors.

    ``flwr.supercore.logger`` sets its logger to DEBUG at import time, so it is imported
    first and adjusted afterwards.
    """
    import flwr.supercore.logger  # noqa: F401

    # ERROR, not WARNING: with fraction_evaluate = 0 Flower's FedAvg warns on every run that
    # federated evaluation is skipped, which is intended here.
    logging.getLogger("flwr").setLevel(logging.ERROR)


def training_spec(
    config: FederatedConfig,
    seed: int,
    dp: DPSweepEntry | None,
    epochs: int,
    rounds: int,
    *,
    run_label: str,
) -> LocalTrainingSpec:
    """The client training spec for a condition (``server_round`` is filled per round).

    ``run_label`` (``"<suite>|<split>|<condition>"``) makes the DP noise and Poisson sampling of
    every run independent (see :mod:`src.federated.client`).
    """
    training = config.training
    return LocalTrainingSpec(
        seed=int(seed),
        server_round=0,
        num_rounds=int(rounds),
        local_epochs=int(epochs),
        batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        momentum=training.momentum,
        weight_decay=training.weight_decay,
        hidden_sizes=tuple(config.model.hidden_sizes),
        num_classes=0,
        dp_mode="none" if dp is None else dp.mode,
        target_epsilon=(
            float(dp.target_epsilon) if dp is not None and dp.target_epsilon else math.nan
        ),
        noise_multiplier=(
            float(dp.noise_multiplier) if dp is not None and dp.noise_multiplier else 0.0
        ),
        max_grad_norm=float(dp.max_grad_norm) if dp is not None else 0.0,
        delta=float(config.dp.delta),
        accountant=config.dp.accountant,
        epsilon_tolerance=float(config.dp.epsilon_tolerance),
        run_label=str(run_label),
    )


def run_label(task: Task) -> str:
    """``"<suite>|<split>|<condition>"``: the identity that seeds a run's DP randomness."""
    return f"{task.suite}|{task.split}|{task.condition}"


def _with(spec: LocalTrainingSpec, **changes: Any) -> LocalTrainingSpec:
    values = to_dict(spec)
    values["hidden_sizes"] = tuple(values["hidden_sizes"])
    values.update(changes)
    return LocalTrainingSpec(**values)


def _predict(weights: list[np.ndarray], dataset: FederatedDataset, hidden) -> np.ndarray:
    return predict_logits(weights, dataset.test.x, tuple(hidden), dataset.num_classes).argmax(1)


def _round_metrics(evaluation: Evaluation) -> dict[str, float]:
    return {key: evaluation.overall[key] for key in ("accuracy", "precision", "recall", "f1", "f1_macro")}  # fmt: skip


def _group_rows(base: dict[str, Any], evaluation: Evaluation) -> list[dict[str, Any]]:
    rows = [dict(base, **row) for row in evaluation.per_group]
    for attribute, gaps in evaluation.group_gaps.items():
        rows.append(
            dict(
                base,
                attribute=attribute,
                group="(max-min gap)",
                n="",
                accuracy=gaps["accuracy"],
                positive_prediction_rate=gaps["positive_prediction_rate"],
                true_positive_rate=gaps["true_positive_rate"],
            )
        )
    return rows


def _domain_columns(evaluation: Evaluation | None, domain: str) -> dict[str, Any]:
    if evaluation is None or domain == "":
        return {"domain_n": "", "domain_accuracy": "", "domain_precision": "", "domain_recall": "", "domain_f1": ""}  # fmt: skip
    metrics = evaluation.per_domain[domain]
    return {
        "domain_n": metrics["n"],
        "domain_accuracy": metrics["accuracy"],
        "domain_precision": metrics["precision"],
        "domain_recall": metrics["recall"],
        "domain_f1": metrics["f1"],
    }


def _final_row(
    task: Task,
    split: str,
    method: str,
    evaluation: Evaluation,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "suite": task.suite,
        "split": split,
        "method": method,
        "seed": task.seed,
        **_round_metrics(evaluation),
        "fairness_gap_accuracy": evaluation.fairness_gap_accuracy,
        "fairness_gap_f1": evaluation.fairness_gap_f1,
    }
    row.update(extra)
    return row


# --------------------------------------------------------------------------------------
# federated conditions through Flower
# --------------------------------------------------------------------------------------


def run_federated(config: FederatedConfig, task: Task) -> TaskOutput:
    """Run one federated condition through the Flower ``ServerApp`` / ``ClientApp``."""
    from .client_app import AttackSpec, ClientDataSource, make_client_app
    from .inprocess import run_inprocess, run_with_ray
    from .server_app import ServerResult, ServerSpec, make_server_app

    dataset = load_dataset(config.dataset, config.evaluation, task.seed)
    natural = task.split == dataset.natural_split_name
    partition = make_partition(
        task.split, dataset, config.partitions, task.seed, None if natural else task.num_clients
    )
    hidden = list(config.model.hidden_sizes)
    rounds = config.training.rounds
    spec = _with(
        training_spec(
            config,
            task.seed,
            task.dp,
            config.training.local_epochs,
            rounds,
            run_label=run_label(task),
        ),
        num_classes=dataset.num_classes,
    )
    names = tuple(
        parameter_names(build_model(dataset.num_features, hidden, dataset.num_classes, 0))
    )
    weights0 = initial_weights(dataset.num_features, hidden, dataset.num_classes, task.seed)
    evaluations: dict[int, Evaluation] = {}

    def evaluate_fn(server_round: int, weights: list[np.ndarray]) -> dict[str, float]:
        evaluation = evaluate_predictions(
            dataset, _predict(weights, dataset, hidden), config.evaluation.group_attributes
        )
        evaluations[server_round] = evaluation
        return _round_metrics(evaluation)

    sink = ServerResult()
    server_spec = ServerSpec(
        num_rounds=rounds,
        num_clients=partition.num_clients,
        aggregation_rule=task.aggregation_rule,
        trimmed_mean_beta=config.federation.trimmed_mean_beta,
        fraction_evaluate=config.federation.fraction_evaluate,
        train_config=spec_to_config(spec),
        evaluate_config={
            "hidden-sizes": [int(h) for h in hidden],
            "num-classes": dataset.num_classes,
        },
        parameter_names=names,
    )
    source = ClientDataSource(
        dataset=config.dataset,
        evaluation=config.evaluation,
        partitions=config.partitions,
        split=task.split,
        num_clients=partition.num_clients,
        seed=task.seed,
    )
    attack = None
    if task.attack is not None and task.attack.kind != "none" and task.num_byzantine > 0:
        from .attacks import byzantine_clients

        attack = AttackSpec(
            kind=task.attack.kind,
            byzantine_ids=byzantine_clients(partition.num_clients, task.num_byzantine),
            scale_factor=task.attack.scale_factor,
            noise_std=task.attack.noise_std,
        )
    server_app = make_server_app(server_spec, weights0, evaluate_fn, sink)
    client_app = make_client_app(source, attack)
    if config.federation.engine == "ray":
        run_with_ray(server_app, client_app, partition.num_clients)
    else:
        run_inprocess(server_app, client_app, partition.num_clients, derive_seed(task.seed, "run", task.condition, task.split), secret_seed=task.seed)  # fmt: skip
    if sink.strategy is None or sink.final_weights is None:
        raise RuntimeError("the ServerApp did not finish")
    return _federated_rows(config, task, dataset, partition, sink, evaluations)


def _federated_rows(config, task, dataset, partition, sink, evaluations) -> TaskOutput:
    strategy = sink.strategy
    rounds = config.training.rounds
    dp_on = task.dp is not None
    delta = config.dp.delta if dp_on else ""
    per_round, per_client = [], []
    base = {
        "suite": task.suite,
        "split": task.split,
        "condition": task.condition,
        "seed": task.seed,
    }
    for server_round in range(0, rounds + 1):
        evaluation = evaluations[server_round]
        clients = strategy.client_history.get(server_round, [])
        local_eval = {int(m["partition-id"]): m for m in strategy.client_eval_history.get(server_round, [])}  # fmt: skip
        by_id = {int(m["partition-id"]): m for m in clients}
        bytes_up, bytes_down = strategy.ledger.cumulative(server_round)
        if clients:
            total = sum(m["num-examples"] for m in clients)
            train_loss = (
                sum(m["train-loss"] * m["num-examples"] for m in clients) / total
                if not dp_on
                else ""  # DP clients do not release their training loss
            )
            epsilon_max = max(m["epsilon"] for m in clients) if dp_on else math.inf
        else:
            train_loss = ""
            epsilon_max = 0.0 if dp_on else math.inf
        per_round.append(
            dict(
                base,
                round=server_round,
                **_round_metrics(evaluation),
                train_loss=train_loss,
                bytes_up_cum=bytes_up,
                bytes_down_cum=bytes_down,
                epsilon_max=epsilon_max,
                delta=delta,
            )
        )
        if server_round == 0:
            continue
        for k in range(max(partition.num_clients, len(dataset.domain_names))):
            client = partition.client_names[k] if k < partition.num_clients else ""
            domain = dataset.domain_names[k] if k < len(dataset.domain_names) else ""
            metrics = by_id.get(k, {})
            per_client.append(
                dict(
                    base,
                    round=server_round,
                    client=client,
                    domain=domain,
                    n_train=int(metrics["num-examples"]) if metrics else "",
                    sample_rate=metrics.get("sample-rate", ""),
                    noise_multiplier=metrics.get("noise-multiplier", ""),
                    epsilon=(
                        metrics.get("epsilon", "") if dp_on else (math.inf if metrics else "")
                    ),
                    update_norm=metrics.get("update-norm", ""),
                    train_loss=metrics.get("train-loss", ""),
                    local_eval_accuracy=(
                        "" if dp_on else local_eval.get(k, {}).get("eval-accuracy", "")
                    ),
                    **_domain_columns(evaluation, domain),
                )
            )
        if dp_on and server_round == rounds:
            eps = {partition.client_names[k]: m["epsilon"] for k, m in sorted(by_id.items())}
            LOGGER.info(
                "%s %s seed %d: per-client epsilon after %d rounds %s (delta=%g; %s)",
                task.condition,
                task.split,
                task.seed,
                rounds,
                {name: round(value, 4) for name, value in eps.items()},
                config.dp.delta,
                GUARANTEE,
            )
    final_eval = evaluations[rounds]
    bytes_up, bytes_down = strategy.ledger.cumulative(rounds)
    last = per_round[-1]
    final = [
        _final_row(
            task,
            task.split,
            task.condition,
            final_eval,
            train_loss=last["train_loss"],
            rounds=rounds,
            bytes_total=bytes_up + bytes_down,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
            epsilon_max=last["epsilon_max"],
            delta=delta,
            train_rows=int(sum(partition.sizes())),
            num_clients=partition.num_clients,
        )
    ]
    group_base = {"suite": task.suite, "split": task.split, "method": task.condition, "seed": task.seed}  # fmt: skip
    return TaskOutput(per_round, per_client, final, _group_rows(group_base, final_eval), [])


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------


def run_task(raw_config: dict[str, Any], task: Task) -> TaskOutput:
    """Run one task in this process (entry point of the worker pool)."""
    import torch

    config = from_dict(FederatedConfig, raw_config)
    torch.set_num_threads(int(config.runtime.torch_threads))
    quiet_flower_logging()
    started = time.perf_counter()
    from .baselines import run_baselines, run_local_only

    if task.kind == "federated":
        output = run_federated(config, task)
    elif task.kind == "local_only":
        output = run_local_only(config, task)
    elif task.kind == "baselines":
        output = run_baselines(config, task)
    else:
        raise ValueError(f"Unknown task kind {task.kind!r}")
    if task.kind != "baselines":
        output.timing.append(
            {
                "suite": task.suite,
                "split": task.split,
                "condition": task.condition,
                "seed": task.seed,
                "wall_time_s": time.perf_counter() - started,
            }
        )
    return output


def build_tasks(config: FederatedConfig, suite: str, seeds: list[int]) -> list[Task]:
    """All tasks of a suite (``main``, ``partitions``, ``aggregation``, ``robustness``)."""
    tasks: list[Task] = []
    splits = tuple(config.partitions.splits)
    rule = config.federation.aggregation_rule

    def task(kind: str, split: str, condition: str, seed: int, **kw: Any) -> Task:
        values = {
            "suite": suite,
            "kind": kind,
            "split": split,
            "condition": condition,
            "seed": seed,
            "aggregation_rule": rule,
            "dp": None,
            "num_clients": None,
            "attack": None,
            "num_byzantine": 0,
            "report_splits": (),
        }
        values.update(kw)
        return Task(**values)

    for seed in seeds:
        if suite == "main":
            if config.baselines.methods:
                tasks.append(task("baselines", "", "baselines", seed, report_splits=splits))
            for split in splits:
                tasks.append(task("local_only", split, "local_only", seed))
                tasks.append(task("federated", split, "fedavg", seed))
                for entry in config.dp.sweep:
                    tasks.append(task("federated", split, entry.name, seed, dp=entry))
        elif suite == "partitions":
            for split in config.partitions.ablations:
                tasks.append(task("federated", split, "fedavg", seed))
        elif suite == "aggregation":
            for agg in config.aggregation_suite.rules:
                tasks.append(
                    task(
                        "federated",
                        config.aggregation_suite.split,
                        f"fedavg_{agg}",
                        seed,
                        aggregation_rule=agg,
                    )
                )
        elif suite == "robustness":
            rob = config.robustness
            tasks.append(
                task(
                    "federated",
                    "iid",
                    "weighted_no_attack",
                    seed,
                    aggregation_rule="weighted",
                    num_clients=rob.num_clients,
                )
            )
            for agg in rob.aggregation_rules:
                tasks.append(
                    task(
                        "federated",
                        "iid",
                        f"{agg}_{rob.attack.kind}",
                        seed,
                        aggregation_rule=agg,
                        num_clients=rob.num_clients,
                        attack=rob.attack,
                        num_byzantine=rob.num_byzantine,
                    )
                )
        else:
            raise ValueError(f"Unknown suite {suite!r}")
    return tasks


def partitions_for_suite(config: FederatedConfig, suite: str) -> list[tuple[str, int | None]]:
    """(split, num_clients) pairs used by a suite, for ``partition_stats.csv``."""
    if suite == "main":
        return [(split, None) for split in config.partitions.splits]
    if suite == "partitions":
        return [(split, None) for split in config.partitions.ablations]
    if suite == "aggregation":
        return [(config.aggregation_suite.split, None)]
    if suite == "robustness":
        return [("iid", config.robustness.num_clients)]
    raise ValueError(f"Unknown suite {suite!r}")


def build_partition(
    config: FederatedConfig,
    dataset: FederatedDataset,
    split: str,
    seed: int,
    num_clients: int | None,
) -> Partition:
    """The partition a task would use (for statistics)."""
    natural = split == dataset.natural_split_name
    return make_partition(split, dataset, config.partitions, seed, None if natural else num_clients)
