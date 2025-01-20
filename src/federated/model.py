"""A small multilayer perceptron and conversions between weights and NumPy arrays.

The network is ``Linear → ReLU`` blocks with sizes from ``model.hidden_sizes`` (YAML) and a
final ``Linear`` layer with one logit per class. There is **no BatchNorm**: Opacus needs
per-sample gradients, and BatchNorm mixes samples within a batch. The architecture is
``[Assumption — not in paper]``.

Weights travel as ``list[np.ndarray]`` in :data:`state_dict` order, which is also the order of
the keys of the Flower ``ArrayRecord`` built by :func:`weights_to_arrayrecord`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from src.common.seeding import derive_seed

if TYPE_CHECKING:
    from flwr.app import ArrayRecord


class MLP(nn.Module):
    """``input → [Linear, ReLU] × len(hidden) → Linear(num_classes)``."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = int(input_dim)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(width, int(hidden)))
            layers.append(nn.ReLU())
            width = int(hidden)
        layers.append(nn.Linear(width, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits."""
        return self.net(x)


def build_model(
    input_dim: int, hidden_sizes: Sequence[int], num_classes: int, init_seed: int
) -> MLP:
    """Build an :class:`MLP` whose initial weights depend only on ``init_seed``.

    The global torch generator is forked, so building a model never disturbs other random
    streams.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        return MLP(input_dim, hidden_sizes, num_classes)


def initial_weights(
    input_dim: int, hidden_sizes: Sequence[int], num_classes: int, seed: int
) -> list[np.ndarray]:
    """Initial global weights of a run: ``derive_seed(seed, "model_init")``."""
    model = build_model(input_dim, hidden_sizes, num_classes, derive_seed(seed, "model_init"))
    return get_weights(model)


def parameter_names(model: nn.Module) -> list[str]:
    """Names of the state-dict entries (the ArrayRecord keys)."""
    return list(model.state_dict().keys())


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Copy the model's state dict into float32 NumPy arrays."""
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_weights(model: nn.Module, weights: Sequence[np.ndarray]) -> None:
    """Load NumPy arrays (state-dict order) into ``model``; shapes must match exactly."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"expected {len(keys)} arrays, got {len(weights)}")
    state = OrderedDict()
    for key, array, current in zip(keys, weights, model.state_dict().values(), strict=True):
        if tuple(array.shape) != tuple(current.shape):
            raise ValueError(f"shape mismatch for {key}: {array.shape} vs {tuple(current.shape)}")
        state[key] = torch.as_tensor(np.asarray(array, dtype=np.float32))
    model.load_state_dict(state, strict=True)


def count_parameters(weights: Sequence[np.ndarray]) -> int:
    """Total number of scalar parameters."""
    return int(sum(int(np.prod(w.shape)) for w in weights))


def weights_to_arrayrecord(weights: Sequence[np.ndarray], names: Sequence[str]) -> ArrayRecord:
    """Build a Flower ``ArrayRecord`` keyed by parameter name."""
    from flwr.app import Array, ArrayRecord

    if len(names) != len(weights):
        raise ValueError("names and weights have different lengths")
    return ArrayRecord(
        OrderedDict(
            (name, Array(np.ascontiguousarray(w))) for name, w in zip(names, weights, strict=True)
        )
    )


def arrayrecord_to_weights(record: ArrayRecord) -> list[np.ndarray]:
    """Inverse of :func:`weights_to_arrayrecord` (key order preserved)."""
    return [np.asarray(array) for array in record.to_numpy_ndarrays()]
