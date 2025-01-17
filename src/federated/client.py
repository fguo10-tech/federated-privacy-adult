"""Local training and evaluation as pure functions.

:func:`train_local` is the whole of a client's work in one round. Its output is a pure
function of (global weights, :class:`LocalTrainingSpec`, client rows, client id, client-held
DP secret). The minibatch order of NON-private training comes from the server-sent seed,
``derive_seed(seed, "train", client, round)``. The DP noise and Poisson sampling come ONLY from
a **client-held secret** that never appears in any message:
``derive_seed(dp_secret, "dp", run_label, round)``, where ``run_label`` is
``"<suite>|<split>|<condition>"``. Two consequences: (1) the server cannot regenerate a
client's noise or Poisson mask from anything it sent (if the DP seed were derived from the
server's ``seed``, an honest-but-curious server could subtract the noise); (2) the DP
randomness is independent across DP conditions and splits, so two conditions' updates cannot
be combined to cancel the noise. The benchmark harness fixes each client's secret from the run
seed so runs are reproducible (see :mod:`src.federated.inprocess`); that is acceptable for
research on public data and must not be done in a deployment. (Model initialisation stays
shared across conditions of a seed on purpose; that pairing reveals nothing.) Nothing is
carried from one
round to the next — the optimizer (including SGD momentum) is rebuilt every round — which is
what makes runs reproducible and independent of the execution engine.

The local update is several epochs of minibatch SGD (FedAvg): ``local_epochs`` passes over
the client's rows in minibatches of ``batch_size``. With ``local_epochs: 1`` and
``batch_size`` equal to the client size it reduces to a single full-batch gradient step per
round (federated SGD).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.common.seeding import derive_seed

from .model import MLP, get_weights, set_weights
from .privacy import dp_schedule, epsilon_spent, resolve_noise_multiplier, train_dp_sgd

DP_MODES = ("none", "target_epsilon", "fixed_noise")


@dataclass(frozen=True)
class LocalTrainingSpec:
    """Everything a client needs besides its rows and the global weights.

    It travels from the server to the client inside the Flower ``ConfigRecord`` (see
    :func:`spec_to_config` / :func:`spec_from_config`); all values come from YAML.
    """

    seed: int
    server_round: int
    num_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    hidden_sizes: tuple[int, ...]
    num_classes: int
    dp_mode: str
    target_epsilon: float
    noise_multiplier: float
    max_grad_norm: float
    delta: float
    accountant: str
    epsilon_tolerance: float
    run_label: str


@dataclass(frozen=True)
class LocalTrainResult:
    """What one client produces in one round (only weights and aggregates leave the client)."""

    weights: list[np.ndarray]
    num_examples: int
    train_loss: float
    steps: int
    sample_rate: float
    noise_multiplier: float
    epsilon: float
    delta: float
    update_norm: float
    empty_batches: int


_FLOAT_FIELDS = (
    "learning_rate",
    "momentum",
    "weight_decay",
    "target_epsilon",
    "noise_multiplier",
    "max_grad_norm",
    "delta",
    "epsilon_tolerance",
)
_INT_FIELDS = ("seed", "num_rounds", "local_epochs", "batch_size", "num_classes")
_STR_FIELDS = ("dp_mode", "accountant", "run_label")


def spec_to_config(spec: LocalTrainingSpec) -> dict[str, Any]:
    """Flatten a spec into ConfigRecord-compatible scalars (``server-round`` is set by Flower)."""
    payload = asdict(spec)
    payload.pop("server_round")
    payload["hidden_sizes"] = [int(h) for h in spec.hidden_sizes]
    return {key.replace("_", "-"): value for key, value in payload.items()}


def spec_from_config(config: Any) -> LocalTrainingSpec:
    """Rebuild a spec from a received ConfigRecord (strict: every key must be present)."""
    get = {str(key).replace("-", "_"): value for key, value in dict(config).items()}
    values: dict[str, Any] = {"server_round": int(get["server_round"])}
    for name in _FLOAT_FIELDS:
        values[name] = float(get[name])
    for name in _INT_FIELDS:
        values[name] = int(get[name])
    for name in _STR_FIELDS:
        values[name] = str(get[name])
    values["hidden_sizes"] = tuple(int(h) for h in get["hidden_sizes"])
    if values["dp_mode"] not in DP_MODES:
        raise ValueError(f"Unknown dp-mode {values['dp_mode']!r}")
    return LocalTrainingSpec(**values)


def _model_for(weights: list[np.ndarray], spec: LocalTrainingSpec, num_features: int) -> MLP:
    model = MLP(num_features, spec.hidden_sizes, spec.num_classes)
    set_weights(model, weights)
    return model


def _sgd_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    epochs: int,
    seed: int,
) -> tuple[float, int]:
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(int(seed))
    losses: list[float] = []
    steps = 0
    model.train()
    for _epoch in range(int(epochs)):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
    return (float(np.mean(losses)) if losses else math.nan), steps


def train_local(
    global_weights: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    spec: LocalTrainingSpec,
    client_id: int,
    *,
    dp_secret: int | None = None,
) -> LocalTrainResult:
    """One round of local training from ``global_weights`` on the client's rows ``(x, y)``.

    Without DP: ``local_epochs`` passes of shuffled minibatch SGD (``dp_secret`` is unused).
    With DP (``spec.dp_mode`` is ``target_epsilon`` or ``fixed_noise``): Opacus DP-SGD with
    Poisson sampling, and ε is the cumulative spend after ``server_round`` rounds (see
    :mod:`src.federated.privacy`). ``dp_secret`` is the client-held secret seed; it is required
    under DP, and ``spec.seed`` (sent by the server) has no influence on the DP randomness.
    """
    if spec.dp_mode != "none" and dp_secret is None:
        raise ValueError("DP training needs the client-held dp_secret; never the server's seed")
    if len(y) == 0:
        raise ValueError(f"client {client_id} has no training rows")
    model = _model_for(global_weights, spec, int(x.shape[1]))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=spec.learning_rate,
        momentum=spec.momentum,
        weight_decay=spec.weight_decay,
    )
    # Copies: the client's rows are read-only arrays and torch needs writable memory.
    features = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
    labels = torch.from_numpy(np.array(y, dtype=np.int64, copy=True))
    n = int(len(labels))
    if spec.dp_mode == "none":
        loss, steps = _sgd_epochs(
            model,
            optimizer,
            features,
            labels,
            spec.batch_size,
            spec.local_epochs,
            derive_seed(spec.seed, "train", client_id, spec.server_round),
        )
        sample_rate = min(1.0, spec.batch_size / n)
        noise, epsilon, empty = 0.0, math.inf, 0
    else:
        schedule = dp_schedule(n, spec.batch_size, spec.local_epochs, spec.num_rounds)
        noise = resolve_noise_multiplier(
            spec.dp_mode,
            spec.target_epsilon,
            spec.noise_multiplier,
            spec.delta,
            schedule,
            spec.accountant,
            spec.epsilon_tolerance,
        )
        # Client-held secret only (never spec.seed, which the server sends), independent per
        # run (suite | split | condition) and per round: see the module docstring.
        dp_seed = derive_seed(int(dp_secret), "dp", spec.run_label, spec.server_round)
        loss, steps, empty = train_dp_sgd(
            model,
            optimizer,
            features,
            labels,
            spec.batch_size,
            spec.local_epochs,
            noise,
            spec.max_grad_norm,
            sample_seed=derive_seed(dp_seed, "poisson"),
            noise_seed=derive_seed(dp_seed, "noise"),
        )
        sample_rate = schedule.sample_rate
        # Every scheduled step runs (empty Poisson batches take a noise-only step), so the
        # accountant is charged for exactly the steps that were executed.
        if steps != schedule.steps_per_round:
            raise RuntimeError(
                f"DP-SGD ran {steps} steps but {schedule.steps_per_round} were planned"
            )
        epsilon = epsilon_spent(
            noise,
            sample_rate,
            schedule.steps_per_round * spec.server_round,
            spec.delta,
            spec.accountant,
        )
    weights = get_weights(model)
    update_norm = float(
        np.sqrt(
            sum(
                float(np.sum((new.astype(np.float64) - old.astype(np.float64)) ** 2))
                for new, old in zip(weights, global_weights, strict=True)
            )
        )
    )
    return LocalTrainResult(
        weights=weights,
        num_examples=n,
        train_loss=loss,
        steps=steps,
        sample_rate=float(sample_rate),
        noise_multiplier=float(noise),
        epsilon=float(epsilon),
        delta=float(spec.delta),
        update_norm=update_norm,
        empty_batches=int(empty),
    )


def predict_logits(
    weights: list[np.ndarray],
    x: np.ndarray,
    hidden_sizes: tuple[int, ...] | list[int],
    num_classes: int,
) -> np.ndarray:
    """Class logits of the MLP with ``weights`` on ``x`` (no gradient)."""
    model = MLP(int(x.shape[1]), hidden_sizes, num_classes)
    set_weights(model, weights)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.array(x, dtype=np.float32, copy=True)))
    return logits.numpy()


def evaluate_local(
    weights: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    hidden_sizes: tuple[int, ...] | list[int],
    num_classes: int,
) -> dict[str, float]:
    """Mean cross-entropy loss and accuracy of ``weights`` on ``(x, y)``."""
    if len(y) == 0:
        return {"loss": math.nan, "accuracy": math.nan, "num_examples": 0}
    logits = predict_logits(weights, x, hidden_sizes, num_classes)
    tensor = torch.from_numpy(logits)
    loss = float(nn.functional.cross_entropy(tensor, torch.from_numpy(np.asarray(y, np.int64))))
    accuracy = float(np.mean(logits.argmax(axis=1) == y))
    return {"loss": loss, "accuracy": accuracy, "num_examples": int(len(y))}


def local_weights_copy(weights: list[np.ndarray]) -> list[np.ndarray]:
    """Defensive copy of a weight list."""
    return [np.array(w, copy=True) for w in weights]


__all__ = [
    "DP_MODES",
    "LocalTrainResult",
    "LocalTrainingSpec",
    "evaluate_local",
    "get_weights",
    "local_weights_copy",
    "predict_logits",
    "spec_from_config",
    "spec_to_config",
    "train_local",
]
