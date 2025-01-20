"""Run the Flower apps in one process, without Ray.

:class:`InProcessGrid` is a real subclass of ``flwr.serverapp.Grid`` (the abstract base class
the ``ServerApp`` receives). ``push_messages`` delivers every message synchronously to the
real ``ClientApp`` object in the same process and stores the reply; ``pull_messages`` hands
replies back by ``reply_to_message_id``. Flower's own ``Strategy.start`` /
``strategy_utils.send_and_receive`` loop therefore runs unmodified, and so do the
``@app.train()`` / ``@app.evaluate()`` handlers.

To keep the client boundary honest, every message is **serialised to protobuf bytes and
parsed back** in both directions (``flwr.common.serde.message_to_proto`` /
``message_from_proto``), exactly as it would be on the wire: the client never holds a Python
reference to a server object, and vice versa. Node ids are deterministic (``101, 102, …``);
each node's ``Context.node_config`` carries ``partition-id`` and ``num-partitions``, the keys
Flower's simulation engine sets. Message ids are a deterministic counter.

Each node's ``node_config`` also carries ``client-secret-seed``,
``derive_seed(secret_seed, "client-secret", partition_id)``: the client-held secret from which
the client derives its DP noise and Poisson sampling. Here the harness stands in for each
client's own entropy and fixes the secret from the run seed so that benchmark runs are
reproducible. That is acceptable for research on public data and must not be done in a
deployment, where each client draws its secret itself (the ClientApp falls back to OS entropy
when the key is absent). ``node_config`` is local to the node and is never put into a message.

:func:`run_with_ray` hands the same two apps to Flower's Ray-based simulation engine
(``flwr.simulation.run_simulation``) when ``flwr[simulation]`` is installed; that path is
untested (see its docstring).
"""

from __future__ import annotations

import traceback
from collections.abc import Iterable

from flwr.app import Context, Error, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.constant import SUPERLINK_NODE_ID, ErrorCode
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
from flwr.serverapp import Grid, ServerApp
from flwr.supercore.run import Run
from flwr.supercore.task_identity import TaskIdentity

from src.common.logging_utils import get_logger
from src.common.seeding import derive_seed

LOGGER = get_logger(__name__)

FIRST_NODE_ID = 101


def _wire_roundtrip(message: Message) -> tuple[Message, int]:
    """Serialise ``message`` to protobuf bytes and parse it back; return it and its size."""
    payload = message_to_proto(message).SerializeToString()
    return message_from_proto(ProtoMessage.FromString(payload)), len(payload)


class InProcessGrid(Grid):
    """A ``flwr.serverapp.Grid`` that runs a ``ClientApp`` synchronously in this process."""

    def __init__(
        self, client_app: ClientApp, num_nodes: int, run_id: int, *, secret_seed: int
    ) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        self._client_app = client_app
        self._run = Run.create_empty(run_id=int(run_id))
        self._node_ids = [FIRST_NODE_ID + i for i in range(int(num_nodes))]
        self._contexts = {
            node_id: Context(
                run_id=int(run_id),
                node_id=node_id,
                node_config={
                    "partition-id": i,
                    "num-partitions": int(num_nodes),
                    # Harness stand-in for the client's own entropy (see module docstring).
                    "client-secret-seed": derive_seed(int(secret_seed), "client-secret", i),
                },
                state=RecordDict(),
                run_config={},
            )
            for i, node_id in enumerate(self._node_ids)
        }
        self._replies: dict[str, Message] = {}
        self._counter = 0
        self.wire_bytes_down = 0
        self.wire_bytes_up = 0
        self.messages_delivered = 0

    # --- Grid API -------------------------------------------------------------------------
    def set_run(self, run: Run) -> None:
        """Replace the run descriptor."""
        if not isinstance(run, Run):
            raise TypeError("run must be a flwr Run")
        self._run = run

    @property
    def run(self) -> Run:
        """The run this grid operates in."""
        return self._run

    def create_message(
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        """Create an instruction message (kept for API completeness)."""
        return Message(content, dst_node_id, message_type, ttl=ttl, group_id=group_id)

    def get_node_ids(self) -> Iterable[int]:
        """All node ids (all nodes are always available)."""
        return list(self._node_ids)

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:08d}"

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Deliver every message to its node's ``ClientApp`` now; store the replies."""
        ids: list[str] = []
        for message in messages:
            if message.metadata.dst_node_id not in self._contexts:
                ids.append("")
                continue
            message_id = self._next_id("msg")
            message.metadata.__dict__["_message_id"] = message_id
            delivered, size = _wire_roundtrip(message)
            self.wire_bytes_down += size
            context = self._contexts[message.metadata.dst_node_id]
            try:
                reply = self._client_app(delivered, context)
            except Exception as exc:  # noqa: BLE001 - mirror Flower: an error reply, not a crash
                LOGGER.error("ClientApp raised on node %s: %s", context.node_id, exc)
                reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                reply = Message(
                    Error(ErrorCode.CLIENT_APP_RAISED_EXCEPTION, reason), reply_to=delivered
                )
            reply.metadata.__dict__["_message_id"] = self._next_id("reply")
            returned, size = _wire_roundtrip(reply)
            self.wire_bytes_up += size
            self._replies[message_id] = returned
            self.messages_delivered += 1
            ids.append(message_id)
        return ids

    def pull_messages(
        self,
        reply_to_message_ids: Iterable[str] | None = None,
        *,
        message_ids: Iterable[str] | None = None,
    ) -> Iterable[Message]:
        """Return (and forget) the stored replies to the given message ids."""
        if message_ids is not None:
            if reply_to_message_ids is not None:
                raise TypeError("Specify only one of reply_to_message_ids and message_ids")
            reply_to_message_ids = message_ids
        wanted = list(self._replies) if reply_to_message_ids is None else list(reply_to_message_ids)
        return [self._replies.pop(i) for i in wanted if i in self._replies]

    def send_and_receive(
        self, messages: Iterable[Message], *, timeout: float | None = None
    ) -> Iterable[Message]:
        """Push then pull; delivery is synchronous, so ``timeout`` is never reached."""
        return self.pull_messages(list(self.push_messages(messages)))


def run_inprocess(
    server_app: ServerApp,
    client_app: ClientApp,
    num_supernodes: int,
    run_id: int,
    *,
    secret_seed: int,
) -> InProcessGrid:
    """Run ``server_app`` against ``num_supernodes`` in-process nodes; return the grid.

    ``secret_seed`` (the run seed in benchmarks) fixes every client's DP secret, see
    :class:`InProcessGrid`.

    Sets Flower's process-wide ``TaskIdentity`` (as ``run_simulation`` does) for the duration
    of the run and restores it afterwards.
    """
    previous = (TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id)
    TaskIdentity.task_id = 1
    TaskIdentity.run_id = int(run_id)
    TaskIdentity.node_id = SUPERLINK_NODE_ID
    try:
        grid = InProcessGrid(client_app, num_supernodes, run_id, secret_seed=secret_seed)
        context = Context(
            run_id=int(run_id),
            node_id=SUPERLINK_NODE_ID,
            node_config={},
            state=RecordDict(),
            run_config={},
        )
        server_app(grid, context)
        return grid
    finally:
        TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id = previous


def ray_available() -> bool:
    """True if Flower's Ray simulation engine can be imported."""
    try:
        import ray  # noqa: F401
        from flwr.simulation import run_simulation  # noqa: F401
    except ImportError:
        return False
    return True


def run_with_ray(server_app: ServerApp, client_app: ClientApp, num_supernodes: int) -> None:
    """Run the same apps on Flower's Ray simulation engine (optional ``flwr[simulation]``).

    One CPU per virtual client; partition ids are assigned by Flower (0..n-1).

    UNTESTED: on the development machine (macOS, repository path containing a space, ray
    2.56) Ray worker processes crashed during start-up ("have not registered within the
    timeout") and the run hung, so this path has never completed here. The comparison test
    ``test_ray_engine_matches_inprocess`` is opt-in (``FEDERATED_RUN_RAY_TEST=1``).
    """
    from flwr.simulation import run_simulation

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=int(num_supernodes),
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
