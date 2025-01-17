"""Bytes-on-the-wire accounting.

The unit counted is the serialised size of the parameter payload as Flower carries it:
:meth:`flwr.app.ArrayRecord.count_bytes` (the ``.npy``-serialised bytes of every array plus
its key). Downlink traffic is counted once per receiving client for every message that
carries the global model; uplink traffic is the ``ArrayRecord`` in each training reply. With
the default ``federation.fraction_evaluate: 0`` no federated-evaluation message is sent, so the
reported cost is **training traffic only** (downlink = uplink when every client trains every
round); if client evaluation is switched on, its model downloads are counted too. Config and metric records (a few hundred bytes)
are counted separately so the parameter payload stays comparable with the textbook
``2 · rounds · clients · |θ| · 4 bytes``. Protocol framing (gRPC, protobuf headers, TLS) is
not counted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from flwr.app import ArrayRecord, RecordDict


def arrayrecord_bytes(record: ArrayRecord) -> int:
    """Serialised parameter bytes of a Flower ``ArrayRecord``."""
    return int(record.count_bytes())


def raw_parameter_bytes(weights: list[np.ndarray]) -> int:
    """Bytes of the raw parameter buffers (no serialisation header)."""
    return int(sum(int(np.asarray(w).nbytes) for w in weights))


def recorddict_bytes(content: RecordDict) -> tuple[int, int]:
    """(parameter bytes, other record bytes) of a ``RecordDict``."""
    arrays = sum(arrayrecord_bytes(record) for record in content.array_records.values())
    other = sum(int(record.count_bytes()) for record in content.config_records.values())
    other += sum(int(record.count_bytes()) for record in content.metric_records.values())
    return int(arrays), int(other)


@dataclass
class CommunicationLedger:
    """Per-round byte counters, filled by the strategy as messages are built and received."""

    down: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    up: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    other_down: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    other_up: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record_down(self, server_round: int, content: RecordDict, receivers: int) -> None:
        """Count one payload sent to ``receivers`` clients."""
        arrays, other = recorddict_bytes(content)
        self.down[server_round] += arrays * receivers
        self.other_down[server_round] += other * receivers

    def record_up(self, server_round: int, content: RecordDict) -> None:
        """Count one client reply."""
        arrays, other = recorddict_bytes(content)
        self.up[server_round] += arrays
        self.other_up[server_round] += other

    def cumulative(self, last_round: int) -> tuple[int, int]:
        """(uplink, downlink) parameter bytes summed over rounds ``1..last_round``."""
        up = sum(value for r, value in self.up.items() if r <= last_round)
        down = sum(value for r, value in self.down.items() if r <= last_round)
        return int(up), int(down)
