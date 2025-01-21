"""Record-level differential privacy for each client via Opacus DP-SGD, with honest accounting.

Mechanism
---------
Inside each client's local training: per-sample gradient clipping to ``max_grad_norm``,
Gaussian noise with standard deviation ``noise_multiplier · max_grad_norm`` added to the sum
of clipped gradients, and Poisson sampling of every batch (each row joins a batch
independently with probability ``q``). This is DP-SGD (Abadi et al., 2016) as implemented by
Opacus (``PrivacyEngine.make_private``). Every DP setting (mechanism, ε, δ, clipping norm,
accountant) is ``[Assumption — not in paper]``.

Accounting
----------
Clients are stateless, so ε is recomputed every round as a pure function of
(noise multiplier, sample rate ``q``, number of DP-SGD steps so far over **all** rounds, δ)
with Opacus's RDP accountant: all steps of all rounds compose. Per round a client takes
``local_epochs · ceil(n / batch_size)`` steps with ``q = 1 / ceil(n / batch_size)`` (exactly
what Opacus's ``DPDataLoader`` uses). Every scheduled step is executed, including steps whose
Poisson batch happens to be empty: such a step is a **noise-only** update (the summed clipped
gradient is zero, Gaussian noise is still added), exactly as the DP-SGD analysis assumes.
Skipping it would reveal through the model trajectory that the batch was empty. So the
number of optimizer steps always equals the number of steps the accountant is charged for.
Two modes:

* ``target_epsilon`` — the noise multiplier is calibrated per client with Opacus's
  ``get_noise_multiplier`` from that client's ``q`` and its total steps over all rounds, so
  that after the last round ε ≤ target (within ``epsilon_tolerance``);
* ``fixed_noise`` — the given noise multiplier is used and the resulting ε is reported.

The reported ε of a run is the **maximum over clients**.

What the (ε, δ) guarantee covers — and what it does not
--------------------------------------------------------
:data:`GUARANTEE` is printed in every DP log line. It is **record-level** (ε, δ)-DP for each
client's training rows with respect to the sequence of model updates that client sends,
against an honest-but-curious server (and anyone who sees the updates or the aggregated
models). Because clients hold disjoint rows, the same (ε, δ) holds for any single row in the
federation. Each run's ε is for **that run alone**: the joint release of several DP runs on
the same rows (e.g. the ε = 8, 3 and 1 conditions of one seed, or both splits) is not covered,
and their budgets add up. It does **not** cover: client-level (user-level) privacy, i.e. hiding
whether a whole client participated; hyperparameter tuning (choices made on validation data
are not accounted); anything a client releases outside its model updates — in particular the
``num-examples`` count each client reports for weighted averaging is sent in the clear, and the
sample rate, step count, noise multiplier and ε all depend on that count
(:data:`RELEASED_QUANTITIES` lists every quantity that leaves a DP client); and, in
``train_fitted`` scaling mode, the scaling statistics. DP clients send no training loss, no
empty-batch count and (with ``federation.fraction_evaluate: 0``, the default) no evaluation on
their own rows, because those would be un-noised statistics of private rows.

The guarantee assumes the client's noise is **secret**. Noise and Poisson sampling come from
PyTorch's pseudo-random generator seeded from ``derive_seed(dp_secret, "dp",
"<suite>|<split>|<condition>", round)``, where ``dp_secret`` is a **client-held seed that is
never transmitted** (``context.node_config["client-secret-seed"]``; see
:func:`src.federated.client_app.client_dp_secret`). Nothing the server sends — in particular
not the run ``seed`` in the config record — influences the noise or the Poisson mask, so an
honest-but-curious server cannot regenerate and subtract it. The randomness is also independent
across DP conditions and splits (with a shared noise vector two conditions' updates would let
the noise cancel). The benchmark harness fixes each client's secret from the run seed so that
runs are reproducible; that is acceptable for research on public data and **must not be done in
a deployment** (without the key the client draws its secret from OS entropy). Opacus
``secure_mode`` remains off and PyTorch's generator is not cryptographically secure, so this is
a research simulation, not a deployment-grade implementation (floating-point attacks on the
Gaussian sampler are not mitigated).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

GUARANTEE = (
    "record-level (epsilon, delta)-DP of each client's training rows w.r.t. the sequence of "
    "model updates it sends (and post-processing of them, e.g. update-norm), against an "
    "honest-but-curious server, for this run alone (several DP runs on the same rows add up); "
    "NOT client-level, NOT covering hyperparameter tuning, the reported num-examples or the "
    "quantities derived from it (sample-rate, steps, noise-multiplier, epsilon), or "
    "train_fitted scaling stats; assumes the client's noise is secret: DP randomness comes from "
    "a client-held seed never transmitted (fixed by the benchmark harness for reproducibility, "
    "not for deployment; Opacus secure_mode off)"
)

RELEASED_QUANTITIES: tuple[tuple[str, str, str], ...] = (
    (
        "model weights",
        "covered",
        "the output of DP-SGD; the quantity the (epsilon, delta) bound is about",
    ),
    (
        "update-norm",
        "covered",
        "post-processing of the sent weights and the public global weights",
    ),
    (
        "num-examples",
        "NOT covered",
        "exact count of the client's training rows, sent in the clear for weighted FedAvg",
    ),
    (
        "partition-id",
        "not data-dependent",
        "client index assigned by the simulation; says nothing about any row",
    ),
    (
        "epsilon",
        "NOT covered",
        "computed from the sample rate and step count, so it reveals ceil(n / batch_size)",
    ),
    ("delta", "not data-dependent", "a public setting from YAML"),
    (
        "noise-multiplier",
        "NOT covered",
        "in target_epsilon mode calibrated from the client's sample rate (depends on n)",
    ),
    ("sample-rate", "NOT covered", "1 / ceil(n / batch_size): depends on the row count n"),
    ("steps", "NOT covered", "local_epochs * ceil(n / batch_size): depends on the row count n"),
)
"""Every quantity a client sends under DP: (name, covered by the guarantee?, note).

``NOT covered`` items all reveal (a coarsening of) the client's row count ``n``, never a value
of any row. Nothing else leaves a DP client: no training loss, no empty-batch count, no
client-side evaluation unless ``federation.fraction_evaluate`` is raised above 0, and never the
client-held DP secret (``client-secret-seed``) from which its noise is drawn."""


def released_quantities_markdown() -> str:
    """:data:`RELEASED_QUANTITIES` as a GitHub Markdown table."""
    lines = [
        "| Quantity leaving a DP client | Covered by the (epsilon, delta) guarantee | Note |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{name}` | {covered} | {note} |" for name, covered, note in RELEASED_QUANTITIES
    )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class DPSchedule:
    """Sampling rate and step counts of one client's DP-SGD."""

    sample_rate: float
    steps_per_round: int
    total_steps: int


def dp_schedule(num_rows: int, batch_size: int, local_epochs: int, rounds: int) -> DPSchedule:
    """``q = 1 / ceil(n / B)``; ``local_epochs · ceil(n / B)`` steps per round."""
    if num_rows <= 0:
        raise ValueError("a DP client needs at least one training row")
    batches = math.ceil(num_rows / batch_size)
    steps = int(local_epochs) * batches
    return DPSchedule(sample_rate=1.0 / batches, steps_per_round=steps, total_steps=steps * rounds)


RDP_ORDERS: tuple[float, ...] = tuple(
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(12, 64))
    + [64, 72, 80, 96, 112, 128, 160, 192, 256, 320, 384, 512, 768, 1024]
)
"""RDP orders searched when converting RDP to (ε, δ): Opacus's default orders (1.1 … 63) plus
14 geometrically spaced orders 64 … 1024 (a dense grid up to 1024 made every ε evaluation
about 20 times slower). With the defaults alone ε cannot go below about ``log(1/δ) / 62`` (≈ 0.19 at
δ = 1e-5) however large the noise, so targets such as ε = 0.1 were unreachable. A superset of
orders can only tighten the bound; for the Adult sweep (ε ∈ {1, 3, 8}) the optimal order is
interior (α ≈ 3–17) and ε is unchanged. An accounting precision setting, not a model
hyperparameter."""


@lru_cache(maxsize=4096)
def epsilon_spent(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float, accountant: str
) -> float:
    """ε after ``steps`` composed DP-SGD steps (Opacus RDP accountant). ``inf`` if no noise."""
    if steps <= 0:
        return 0.0
    if noise_multiplier <= 0:
        return math.inf
    if accountant != "rdp":
        raise ValueError(f"Unsupported accountant {accountant!r}")
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    acc.history = [(float(noise_multiplier), float(sample_rate), int(steps))]
    return float(acc.get_epsilon(delta=float(delta), alphas=list(RDP_ORDERS)))


@lru_cache(maxsize=1024)
def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    total_steps: int,
    accountant: str,
    epsilon_tolerance: float,
) -> float:
    """Smallest noise multiplier (binary search, Opacus) with ε ≤ target after ``total_steps``."""
    if accountant != "rdp":
        raise ValueError(f"Unsupported accountant {accountant!r}")
    from opacus.accountants.utils import get_noise_multiplier

    # The binary search probes very large noise multipliers, for which Opacus warns that the
    # optimal RDP order is the largest order. The returned multiplier's own epsilon
    # uses an interior order (checked during development: alpha ~3-17 for the Adult sweep).
    # Orders: :data:`RDP_ORDERS`.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Optimal order is the largest alpha")
        sigma = float(
            get_noise_multiplier(
                target_epsilon=float(target_epsilon),
                target_delta=float(delta),
                sample_rate=float(sample_rate),
                steps=int(total_steps),
                accountant=accountant,
                epsilon_tolerance=float(epsilon_tolerance),
                alphas=list(RDP_ORDERS),
            )
        )
    return sigma


def resolve_noise_multiplier(
    mode: str,
    target_epsilon: float | None,
    noise_multiplier: float | None,
    delta: float,
    schedule: DPSchedule,
    accountant: str,
    epsilon_tolerance: float,
) -> float:
    """Noise multiplier of one client for the configured mode."""
    if mode == "target_epsilon":
        if target_epsilon is None:
            raise ValueError("target_epsilon mode needs target_epsilon")
        return calibrate_noise_multiplier(
            target_epsilon,
            delta,
            schedule.sample_rate,
            schedule.total_steps,
            accountant,
            epsilon_tolerance,
        )
    if mode == "fixed_noise":
        if noise_multiplier is None:
            raise ValueError("fixed_noise mode needs noise_multiplier")
        return float(noise_multiplier)
    raise ValueError(f"Unknown DP mode {mode!r}")


def train_dp_sgd(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    epochs: int,
    noise_multiplier: float,
    max_grad_norm: float,
    sample_seed: int,
    noise_seed: int,
) -> tuple[float, int, int]:
    """Run ``epochs`` passes of Opacus DP-SGD over ``(x, y)`` with Poisson sampling.

    Returns (mean training loss over non-empty batches, optimizer steps taken, empty batches).
    Both the Poisson sampler and the noise use explicitly seeded generators. An empty Poisson
    batch (probability ``(1 - q)^n``, about ``e^{-expected batch size}``) still takes the full
    zero_grad / forward / backward / step: its per-sample gradients are empty, the summed
    gradient is zero and Opacus adds the Gaussian noise, so it is a noise-only step. Its loss
    is NaN (mean over zero rows) and is left out of the reported mean. The step count therefore
    always equals ``epochs · len(dp_loader)``, the number the accountant is charged for.
    """
    from opacus import PrivacyEngine

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(int(sample_seed)),
    )
    noise_generator = torch.Generator().manual_seed(int(noise_seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = PrivacyEngine(accountant="rdp")
        dp_model, dp_optimizer, dp_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=float(noise_multiplier),
            max_grad_norm=float(max_grad_norm),
            poisson_sampling=True,
            noise_generator=noise_generator,
        )
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    steps = 0
    empty = 0
    dp_model.train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _epoch in range(int(epochs)):
            for batch_x, batch_y in dp_loader:
                is_empty = batch_x.shape[0] == 0
                dp_optimizer.zero_grad(set_to_none=True)
                loss = criterion(dp_model(batch_x), batch_y)
                loss.backward()
                dp_optimizer.step()  # noise-only update when the batch is empty
                steps += 1
                if is_empty:
                    empty += 1
                else:
                    losses.append(float(loss.detach()))
    if hasattr(dp_model, "cleanup"):
        dp_model.cleanup()  # remove Opacus hooks and per-sample attributes from the model
    mean_loss = float(sum(losses) / len(losses)) if losses else math.nan
    return mean_loss, steps, empty
