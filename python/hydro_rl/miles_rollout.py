"""Retried rollouts for Miles' fully-async producer, made safe by the coordinator.

Miles' `FullyAsyncRolloutFn` has no per-rollout deadline: a generation stuck on a
slow or hung engine holds its slot until it returns. Here a group whose newest
attempt runs past a deadline gets a *retry* while the original keeps running.
Whichever finishes first is trained; the coordinator discards the other result
and sends `cancel`, which stops the losing task.

You don't select this class yourself. `--fully-async` makes Miles replace any
`--rollout-function-path` with its own `FullyAsyncRolloutFn`, so instead
`HydroDataBuffer` (loaded by `--custom-async-data-buffer-path`) upgrades the
rollout function it is attached to into this subclass (`adopt`).

How it differs from Miles' producer loop:
- Every attempt generates on its own copy of the prompt group. Generation fills
  in `Sample`s in place, so two attempts on one group must not share objects;
  the original prompt group stays untouched, ready for recycling.
- Retry deadline: `factor` x the `quantile` of recent rollout latencies (at least
  `min_seconds`), once `warmup` rollouts have finished. A group gets at most
  `max_attempts` attempts per epoch.
- A retry drops the prompt's routing key, so a router that routes by key can
  send it to a different engine than the stuck original.
- Retries run on their own budget of extra concurrent groups, outside Miles'
  submission scheduler, so they can start even when hung rollouts hold every
  regular slot (the case they exist for).
- Checkpoints save each running group once, never one whose result the buffer
  already holds, and without Hydro ids (the restored job is a new run).

Settings (environment variables): HYDRO_RETRY=0 turns retries off (the rest still
applies), HYDRO_RETRY_QUANTILE (0.9), HYDRO_RETRY_FACTOR (2.0),
HYDRO_RETRY_MIN_SECONDS (10), HYDRO_RETRY_WARMUP (20), HYDRO_RETRY_MAX_ATTEMPTS
(2), HYDRO_RETRY_BUDGET (default 10% of the in-flight group budget, at least 1).

Limits: a deadline on wall time cannot tell a stuck rollout from a legitimately
long one, so long rollouts get retried too (bounded by the budget and
`max_attempts`). Whether cancelling the losing task also stops its request
inside SGLang depends on SGLang aborting on client disconnect; not verified.

STATUS: tested with stand-ins for Miles (`tests/test_miles_retries.py`); not yet
run inside a real Miles job.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, replace

from miles.rollout.base_types import stamp_kv_cache_namespace
from miles.rollout.fully_async_data_buffer import DataBufferInput
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn, _FullyAsyncRolloutState
from miles.rollout.inference_rollout.inference_rollout_common import generate_and_rm_group
from miles.utils.output_store.dropped import record_dropped_samples
from miles.utils.simple_checkpointer import save_simple_checkpoint
from miles.utils.types import Sample

from hydro_rl.miles_buffer import HYDRO_KEYS, HydroDataBuffer, Rid

logger = logging.getLogger(__name__)


def adopt(rollout_fn: object) -> bool:
    """Turn Miles' `FullyAsyncRolloutFn` instance into a `HydroFullyAsyncRolloutFn`."""
    if isinstance(rollout_fn, HydroFullyAsyncRolloutFn) or not isinstance(rollout_fn, FullyAsyncRolloutFn):
        return False
    rollout_fn.__class__ = HydroFullyAsyncRolloutFn
    logger.info("Hydro: rollout producer upgraded (retries %s)", "on" if RetryPolicy.from_env().enabled else "off")
    return True


@dataclass
class RetryPolicy:
    enabled: bool = True
    quantile: float = 0.9
    factor: float = 2.0
    min_seconds: float = 10.0
    warmup: int = 20
    max_attempts: int = 2
    budget: int | None = None

    @classmethod
    def from_env(cls) -> RetryPolicy:
        env = os.environ.get
        budget = env("HYDRO_RETRY_BUDGET")
        return cls(
            enabled=env("HYDRO_RETRY", "1") != "0",
            quantile=float(env("HYDRO_RETRY_QUANTILE", cls.quantile)),
            factor=float(env("HYDRO_RETRY_FACTOR", cls.factor)),
            min_seconds=float(env("HYDRO_RETRY_MIN_SECONDS", cls.min_seconds)),
            warmup=int(env("HYDRO_RETRY_WARMUP", cls.warmup)),
            max_attempts=int(env("HYDRO_RETRY_MAX_ATTEMPTS", cls.max_attempts)),
            budget=int(budget) if budget else None,
        )


@dataclass
class _Attempt:
    prompt_group: list[Sample]  # the group's original, never generated on
    samples: list[Sample]  # this attempt's own copy
    rid: Rid
    started: float
    retry: bool
    superseded: bool = False  # the coordinator cancelled or recycled it


class HydroFullyAsyncRolloutFn(FullyAsyncRolloutFn):
    # Set up lazily: `adopt` swaps the class of an already constructed instance.
    def _hydro_init(self) -> None:
        if getattr(self, "_attempts", None) is not None:
            return
        assert isinstance(self._output, HydroDataBuffer), (
            "HydroFullyAsyncRolloutFn needs --custom-async-data-buffer-path hydro_rl.miles_buffer.HydroDataBuffer"
        )
        self._attempts: dict[asyncio.Task, _Attempt] = {}
        self._latencies: deque[float] = deque(maxlen=200)
        self._policy = RetryPolicy.from_env()
        self._retry_stats: Counter = Counter()
        self._output.on_cancel = self._on_cancel
        self._output.on_recycle = self._on_recycle
        self._output.extra_metrics = self._take_retry_metrics

    # ------------------------------------------------------------ producer --

    async def _worker_loop(self) -> None:
        self._hydro_init()
        await self._output._ensure_connected()
        active: set[asyncio.Task] = set()
        while True:
            await self._producer_resumed.wait()
            while (attempt := self._due_retry()) is not None:
                active.add(self._start_attempt(attempt.prompt_group, retry=True))
            while self._scheduler.has_capacity(
                pending_groups=sum(not self._attempts[t].retry for t in active),
                group_budget=self._max_in_flight_groups(),
            ):
                active.add(self._submit_one_group())
            timer = asyncio.ensure_future(asyncio.sleep(self._seconds_to_next_deadline()))
            try:
                done, active = await self._scheduler.wait_for_progress(active | {timer})
            finally:
                timer.cancel()
            active.discard(timer)
            for task in done - {timer}:
                await self._finish(task)

    def _submit_one_group(self) -> asyncio.Task:
        samples = [self._retry_buffer.popleft()] if self._retry_buffer else self.data_source.get_samples(1)
        [prompt_group] = samples
        return self._start_attempt(prompt_group, retry=False)

    def _start_attempt(self, prompt_group: list[Sample], *, retry: bool) -> asyncio.Task:
        samples = copy.deepcopy(prompt_group)
        if retry:
            for s in samples:
                s.routing_key = None  # let a key-routing router pick an engine afresh
            self._retry_stats["retries"] += 1
        stamp_kv_cache_namespace([samples], namespace=self._curr_kv_cache_namespace)
        rid = self._output.assign_ids(prompt_group)
        callback = None
        if not retry:  # retries run on their own budget, outside Miles' scheduler
            self._scheduler.on_submit([samples])
            callback = self._scheduler.sample_done_callback
        task = asyncio.create_task(
            generate_and_rm_group(
                self.state,
                samples,
                sampling_params=self.state.sampling_params.copy(),
                evaluation=False,
                sample_done_callback=callback,
            )
        )
        self._attempts[task] = _Attempt(prompt_group, samples, rid, time.monotonic(), retry)
        return task

    async def _finish(self, task: asyncio.Task) -> None:
        attempt = self._attempts.pop(task)
        if not task.cancelled():
            error = task.exception()
        if attempt.superseded:
            # Lost to another attempt, or its epoch was recycled: never trained.
            self._retry_stats["cancelled"] += 1
            record_dropped_samples([attempt.samples])
            return
        if task.cancelled():
            entry = DataBufferInput(
                prompt_group=attempt.prompt_group,
                group=[replace(s, status=Sample.Status.ABORTED) for s in attempt.samples],
            )
        else:
            if error is not None:
                raise error  # as in Miles: a failing generate function ends the worker
            entry = DataBufferInput(prompt_group=attempt.prompt_group, group=task.result())
            self._latencies.append(time.monotonic() - attempt.started)
        entry.hydro_rid = attempt.rid
        await self._output.put(entry)

    # ------------------------------------------------------------- retries --

    def _deadline(self) -> float | None:
        p = self._policy
        if not p.enabled or len(self._latencies) < p.warmup:
            return None
        ordered = sorted(self._latencies)
        q = ordered[min(len(ordered) - 1, int(p.quantile * len(ordered)))]
        return max(p.min_seconds, p.factor * q)

    def _retry_budget(self) -> int:
        return self._policy.budget or max(1, self._max_in_flight_groups() // 10)

    def _retry_candidates(self) -> list[tuple[float, _Attempt]]:
        """(deadline time, newest live attempt) for groups that may get a retry."""
        newest: dict[tuple[int, int], _Attempt] = {}
        for a in self._attempts.values():
            key = a.rid[:2]
            if not a.superseded and (key not in newest or a.started > newest[key].started):
                newest[key] = a
        deadline = self._deadline()
        if deadline is None:
            return []
        return [
            (a.started + deadline, a)
            for (g, e), a in newest.items()
            if self._output.attempts_started(g, e) < self._policy.max_attempts
        ]

    def _due_retry(self) -> _Attempt | None:
        running = sum(a.retry and not a.superseded for a in self._attempts.values())
        if running >= self._retry_budget():
            return None
        now = time.monotonic()
        due = sorted((t, a.started, a) for t, a in self._retry_candidates() if t <= now)
        return due[0][2] if due else None

    def _seconds_to_next_deadline(self) -> float:
        times = [t for t, _ in self._retry_candidates()]
        return max(0.01, min(times) - time.monotonic()) if times else 3600.0

    def _on_cancel(self, rid: Rid) -> None:
        self._supersede(lambda a: a.rid == rid)

    def _on_recycle(self, group: int, next_epoch: int) -> None:
        # Before Miles' handler resets and requeues the prompt group.
        self._supersede(lambda a: a.rid[0] == group and a.rid[1] < next_epoch)

    def _supersede(self, match) -> None:
        for task, a in self._attempts.items():
            if match(a) and not a.superseded:
                a.superseded = True
                task.cancel()

    # ---------------------------------------------------------- checkpoint --

    def save(self, directory) -> None:
        self._hydro_init()
        held = {rid[0] for rid in self._output._payloads}
        running: dict[int, list[Sample]] = {}
        for a in self._attempts.values():
            if not a.superseded and a.rid[0] not in held:
                running.setdefault(id(a.prompt_group), a.prompt_group)
        state = _FullyAsyncRolloutState(
            retry_buffer=[_clean_copy(pg) for pg in self._retry_buffer],
            running=[_clean_copy(pg) for pg in running.values()],
            output=self._output.state_dict(),
        )
        save_simple_checkpoint(directory=directory, data=state)

    def _take_retry_metrics(self) -> dict[str, float]:
        """Reported through the buffer's get_metrics, once per training step."""
        m = {f"rollout/hydro/retry_{k}": float(v) for k, v in self._retry_stats.items()}
        if (d := self._deadline()) is not None:
            m["rollout/hydro/retry_deadline_s"] = d
        self._retry_stats = Counter()
        return m


def _clean_copy(prompt_group: list[Sample]) -> list[Sample]:
    """A resubmittable copy without Hydro ids: the restored job mints new ones."""
    copied = copy.deepcopy(prompt_group)
    for s in copied:
        s.reset_for_retry()
        for key in HYDRO_KEYS:
            s.metadata.pop(key, None)
    return copied
