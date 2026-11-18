"""Asyncio client for the Hydro coordinator (newline-delimited JSON over TCP).

Only control-plane metadata travels over this connection: group ids, epochs,
attempts, weight versions, token counts. Rollout payloads stay in the Python
process that produced them.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

Decision = dict[str, Any]


class CoordinatorClient:
    """One TCP connection to the coordinator.

    ``get_batch`` awaits the ``batch`` decision for its request; every other
    decision (``recycle``, ``discard``, ``cancel``, ``starved``, ``restarted``,
    ``error``) is
    passed to ``on_decision``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_decision: Callable[[Decision], None] | None = None,
        run: int | None = None,
    ):
        self._host, self._port = host, port
        # Trainer incarnation, stamped on every message. A restarted trainer
        # (or one restored from a checkpoint) must use a higher run than any
        # before it, so the coordinator resets its weight version instead of
        # treating the new run's rollouts as stale forever. Wall-clock
        # nanoseconds at startup is enough on one machine; a real deployment
        # would persist a counter with the checkpoint.
        self.run = time.time_ns() if run is None else run
        self._on_decision = on_decision or (lambda d: logger.info("coordinator: %s", d))
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._config_waiters: list[asyncio.Future] = []
        self._request_ids = itertools.count()
        self._read_task: asyncio.Task | None = None

    async def connect(self, retries: int = 50, delay: float = 0.2) -> None:
        for i in range(retries):
            try:
                self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
                break
            except OSError:
                if i == retries - 1:
                    raise
                await asyncio.sleep(delay)
        self._read_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        if self._read_task is not None:
            self._read_task.cancel()
        if self._writer is not None:
            self._writer.close()

    def send(self, event: dict[str, Any]) -> None:
        assert self._writer is not None, "connect() first"
        event.setdefault("run", self.run)
        self._writer.write((json.dumps(event) + "\n").encode())

    async def flush(self) -> None:
        assert self._writer is not None
        await self._writer.drain()

    # -- typed helpers mirroring coordinator/src/protocol.rs --------------------

    def dispatch(self, group: int, epoch: int, attempt: int) -> None:
        self.send({"type": "dispatch", "group": group, "epoch": epoch, "attempt": attempt})

    def complete(
        self,
        group: int,
        epoch: int,
        attempt: int,
        *,
        oldest_version: int,
        newest_version: int,
        n_samples: int,
        tokens: int,
    ) -> None:
        self.send(
            {
                "type": "complete",
                "group": group,
                "epoch": epoch,
                "attempt": attempt,
                "oldest_version": oldest_version,
                "newest_version": newest_version,
                "n_samples": n_samples,
                "tokens": tokens,
            }
        )

    def abort(self, group: int, epoch: int, attempt: int) -> None:
        self.send({"type": "abort", "group": group, "epoch": epoch, "attempt": attempt})

    def drop(self, group: int, epoch: int, attempt: int) -> None:
        """Close the group without training or regenerating it."""
        self.send({"type": "drop", "group": group, "epoch": epoch, "attempt": attempt})

    async def hello(self) -> dict[str, Any]:
        """The coordinator's config (`protocol.rs::Config`)."""
        fut = asyncio.get_running_loop().create_future()
        self._config_waiters.append(fut)
        self.send({"type": "hello"})
        await self.flush()
        return (await fut)["config"]

    def version(self, version: int) -> None:
        self.send({"type": "version", "version": version})

    async def get_batch(self, n_groups: int) -> Decision:
        request = next(self._request_ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[request] = fut
        self.send({"type": "get", "request": request, "n_groups": n_groups})
        await self.flush()
        return await fut

    async def _read_loop(self) -> None:
        assert self._reader is not None
        while line := await self._reader.readline():
            decision = json.loads(line)
            if decision.get("type") == "batch":
                fut = self._pending.pop(decision["request"], None)
                if fut is not None and not fut.done():
                    fut.set_result(decision)
            elif decision.get("type") == "config" and self._config_waiters:
                self._config_waiters.pop(0).set_result(decision)
            else:
                self._on_decision(decision)
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("coordinator closed the connection"))
