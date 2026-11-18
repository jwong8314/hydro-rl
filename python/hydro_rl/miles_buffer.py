"""Miles integration: a `DataBuffer` that delegates group decisions to Hydro.

Use with Miles' fully-async rollout, no Miles fork needed:

    cargo run --example serve -- 7878 none 0 false   # Miles defaults, see below
    HYDRO_COORDINATOR=127.0.0.1:7878 python train_async.py --fully-async \\
        --custom-async-data-buffer-path hydro_rl.miles_buffer.HydroDataBuffer ...

Behaves like Miles' `DefaultDataBuffer` (miles/rollout/fully_async_data_buffer.py):

- the same filters on put: aborted groups go to `--async-unused-samples-handler`
  (when the coordinator agrees no newer attempt is still running); missing-reward
  and dynamic-filter rejections are dropped and never regenerated;
- `--max-weight-staleness` (default None: no limit) on the oldest versioned token;
  groups without versioned tokens are never stale; stale groups go to the handler;
- FIFO order, `get` returns exactly `num_groups` and warns after 30 s without progress;
- `put` blocks once `--async-data-buffer-capacity-factor` x rollout batch size
  results are held;
- `state_dict` / `load_state_dict` for checkpoints, with the same
  `regenerated_on_restore` handling;
- the same `rollout/fully_async/*` and dynamic-filter metrics, dropped-sample
  records and sample-ownership logs.

The coordinator's settings (staleness limit, group-size check, ordering) live in
the `serve` process. At connect this buffer asks for them and refuses to run if
they disagree with Miles' launch arguments.

What the coordinator adds: `(group, epoch, attempt)` ids so a result is trained at
most once even when rollouts are retried or delivered twice, trainer-restart run
ids, and `starved` alerts (`rollout/hydro/*` metrics).

Retried rollouts: with `--fully-async` Miles always builds its own
`FullyAsyncRolloutFn`, so this buffer upgrades it in place to
`hydro_rl.miles_rollout.HydroFullyAsyncRolloutFn` (see that module).

Result ids are assigned when generation *starts* (`assign_ids`, called by
`HydroFullyAsyncRolloutFn`) and travel with the result as
`DataBufferInput.hydro_rid`. Reading them from the prompt's metadata at put()
time is wrong: a recycle may have advanced the epoch on those same `Sample`
objects while an older generation was still in flight.

Trainer restarts: every message carries this process's run id (see
`CoordinatorClient`), so a coordinator that outlives the trainer accepts the
resumed run's lower weight version. Assumes inference serves the restored
weights before the first `get()` announces it.

STATUS: exercised with stand-ins for the Miles functions it calls
(`tests/test_miles_buffer.py`); **not yet run inside a real Miles job**. No
multi-policy support (`--custom-async-data-buffer-path-per-model`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections import Counter
from collections.abc import Callable
from typing import Any

from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.filter_hub.common_filters import (
    apply_aborted_filter,
    apply_missing_reward_filter,
    group_staleness,
    group_weight_version_stats,
)
from miles.rollout.fully_async_data_buffer import (
    NO_PROGRESS_WARN_SECS,
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    UnusedReason,
)
from miles.utils.audit_utils.sample_ownership.recorder import SampleOwnershipRecorder
from miles.utils.function_registry import load_function
from miles.utils.output_store.dropped import record_dropped_samples, regenerated_on_restore

from hydro_rl.client import CoordinatorClient

logger = logging.getLogger(__name__)

# Keys in Sample.metadata, which Sample.reset_for_retry() preserves.
GROUP_KEY = "hydro_group"
EPOCH_KEY = "hydro_epoch"
ATTEMPT_KEY = "hydro_attempt"
HYDRO_KEYS = (GROUP_KEY, EPOCH_KEY, ATTEMPT_KEY)

# coordinator/src/protocol.rs::NO_STALENESS_LIMIT
NO_STALENESS_LIMIT = 2**64 - 1

_REASONS = {
    "stale": UnusedReason.STALE,
    "aborted": UnusedReason.ABORTED,
    "partial": UnusedReason.ABORTED,
    # generated under weights a trainer restart threw away
    "restart": UnusedReason.STALE,
}

Rid = tuple[int, int, int]


def _samples(items) -> list:
    out = []
    for item in items:
        out.extend(_samples(item) if isinstance(item, list) else [item])
    return out


class HydroDataBuffer(DataBuffer):
    def __init__(self, input: DataBufferConstructorInput):
        args = self._args = input.args
        assert args.async_data_buffer_capacity_factor > 0
        self._capacity = int(args.async_data_buffer_capacity_factor * args.rollout_batch_size)
        assert self._capacity >= args.rollout_batch_size, (
            f"buffer capacity {self._capacity} is below the rollout batch of {args.rollout_batch_size} groups that "
            f"one get asks for, so --async-data-buffer-capacity-factor must be at least 1.0"
        )
        self._unused_handler_fn = input.unused_handler_fn
        self._dynamic_filter = load_function(args.dynamic_sampling_filter_path)

        host, port = os.environ.get("HYDRO_COORDINATOR", "127.0.0.1:7878").rsplit(":", 1)
        # One run per trainer process. When the job resumes from a
        # checkpoint, the new process gets a higher run (wall-clock ns, or
        # HYDRO_RUN), so the coordinator accepts its lower weight version.
        run = os.environ.get("HYDRO_RUN")
        self._client = CoordinatorClient(
            host, int(port), on_decision=self._on_decision, run=int(run) if run else None
        )
        self._connected: asyncio.Task | None = None
        # Group ids must be unique for the coordinator's lifetime. Miles'
        # prompt indices repeat across dataset epochs, so mint our own.
        self._next_group = random.getrandbits(31) << 20

        # Results sent to the coordinator and not yet trained or released.
        self._payloads: dict[Rid, DataBufferInput] = {}
        self._space = asyncio.Event()  # set whenever a payload is released
        self._prompt_groups: dict[int, list] = {}
        self._restored: list[DataBufferInput] = []  # from a checkpoint, not yet resubmitted
        # Attempts started per (group, epoch). Kept here, not in Sample metadata:
        # retries run on copies of the prompt group.
        self._next_attempt: dict[tuple[int, int], int] = {}
        # Set by HydroFullyAsyncRolloutFn to stop superseded attempts and to
        # report retry metrics.
        self.on_cancel: Callable[[Rid], None] | None = None
        self.on_recycle: Callable[[int, int], None] | None = None
        self.extra_metrics: Callable[[], dict[str, float]] | None = None

        # With --fully-async Miles always constructs its own FullyAsyncRolloutFn
        # (it overrides --rollout-function-path), and constructs this buffer
        # from it: upgrade that instance to run retried rollouts.
        owner = getattr(input.unused_handler_fn, "__self__", None)
        if owner is not None:
            from hydro_rl.miles_rollout import adopt

            adopt(owner)
        self._last_version: int | None = None
        self._current_version: int | None = None
        self._reset_metrics()

    # -------------------------------------------------------------- connect --

    async def _ensure_connected(self) -> None:
        if self._connected is None:
            self._connected = asyncio.create_task(self._connect())
        await self._connected

    async def _connect(self) -> None:
        await self._client.connect()
        self._check_config(await self._client.hello())
        self._resubmit_restored()
        await self._client.flush()

    def _check_config(self, cfg: dict[str, Any]) -> None:
        want = self._args.max_weight_staleness
        want_wire = NO_STALENESS_LIMIT if want is None else want
        problems = []
        if cfg["max_staleness"] != want_wire:
            have = "none" if cfg["max_staleness"] == NO_STALENESS_LIMIT else cfg["max_staleness"]
            problems.append(f"coordinator max_staleness={have} but --max-weight-staleness={want}")
        n = getattr(self._args, "n_samples_per_prompt", None)
        if cfg["group_size"] not in (0, n):
            problems.append(f"coordinator group_size={cfg['group_size']} but --n-samples-per-prompt={n}")
        if problems:
            raise RuntimeError(
                "Hydro coordinator config does not match Miles: "
                + "; ".join(problems)
                + ". Restart `serve` with matching arguments (`none` for no staleness limit)."
            )
        if cfg["oldest_first"]:
            logger.warning("coordinator serves oldest-first, not FIFO like Miles' DefaultDataBuffer")

    # ------------------------------------------------------------------ put --

    def assign_ids(self, prompt_group: list) -> Rid:
        """Snapshot this generation's id and report the dispatch. Call when
        generation starts; attach the result to the output as `hydro_rid`."""
        g, e, _ = self._ids(prompt_group)
        a = self._next_attempt.get((g, e), 0)
        self._next_attempt[(g, e)] = a + 1
        for s in prompt_group:
            s.metadata[ATTEMPT_KEY] = a + 1
        if self._client._writer is not None:
            self._client.dispatch(g, e, a)
        return g, e, a

    def attempts_started(self, group: int, epoch: int) -> int:
        return self._next_attempt.get((group, epoch), 0)

    def _forget_attempts(self, group: int, before_epoch: int | None = None) -> None:
        for key in [k for k in self._next_attempt if k[0] == group and (before_epoch is None or k[1] < before_epoch)]:
            del self._next_attempt[key]

    async def put(self, input: DataBufferInput) -> None:
        await self._ensure_connected()
        rid = getattr(input, "hydro_rid", None)
        if rid is None:
            logger.warning("result without hydro_rid; use HydroFullyAsyncRolloutFn so ids are fixed at dispatch")
            rid = self._ids(input.prompt_group)
        g, e, a = rid

        if not apply_aborted_filter(self._args, input.group).keep:
            self._metric_aborted_groups += 1
            # Before the handler runs (on the coordinator's recycle): a retry
            # resets the prompt group.
            record_dropped_samples([input.prompt_group, input.group])
            self._client.abort(g, e, a)
            await self._client.flush()
            return

        output = apply_missing_reward_filter(self._args, input.group)
        if not output.keep:
            await self._drop(input, rid, output.reason, "missing_reward")
            return
        self._metric_gatherer.on_group_before_dynamic_filter(self._args, input.group)
        output = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
        if not output.keep:
            await self._drop(input, rid, output.reason, "dynamic_filter")
            return

        # Backpressure, like Miles: block while the buffer is full.
        while len(self._payloads) >= self._capacity:
            self._space.clear()
            await self._space.wait()
        self._submit(input, rid)
        await self._client.flush()

    async def _drop(self, input: DataBufferInput, rid: Rid, reason: str | None, audit_reason: str) -> None:
        self._metric_gatherer.on_dynamic_filter_drop(reason=reason)
        record_dropped_samples([input.prompt_group, input.group])
        SampleOwnershipRecorder.log_dropped_samples(args=self._args, samples=input.prompt_group, reason=audit_reason)
        self._client.drop(*rid)  # closes the group: never trained, never regenerated
        await self._client.flush()

    def _submit(self, input: DataBufferInput, rid: Rid) -> None:
        g, e, a = rid
        stats = group_weight_version_stats(input.group)
        self._payloads[rid] = input
        self._client.complete(
            g,
            e,
            a,
            oldest_version=stats.oldest_version,  # None: never stale, as in Miles
            newest_version=stats.newest_version,
            n_samples=len(input.group),
            tokens=stats.versioned_token_count,
        )

    def _release(self, rid: Rid) -> DataBufferInput | None:
        entry = self._payloads.pop(rid, None)
        if entry is not None:
            self._space.set()
        return entry

    def _ids(self, prompt_group: list) -> Rid:
        meta = prompt_group[0].metadata
        if GROUP_KEY not in meta:
            g = self._next_group
            self._next_group += 1
            for s in prompt_group:
                s.metadata.update({GROUP_KEY: g, EPOCH_KEY: 0, ATTEMPT_KEY: 0})
        g = meta[GROUP_KEY]
        self._prompt_groups[g] = prompt_group
        return g, meta.get(EPOCH_KEY, 0), meta.get(ATTEMPT_KEY, 0)

    # ------------------------------------------------------------------ get --

    async def get(self, *, num_groups: int, current_version: int | None = None, **_) -> list[DataBufferInput]:
        await self._ensure_connected()
        if current_version is not None:
            self._current_version = current_version
            if current_version != self._last_version:
                self._client.version(current_version)
                self._last_version = current_version
        request = asyncio.ensure_future(self._client.get_batch(num_groups))
        while True:
            done, _ = await asyncio.wait({request}, timeout=NO_PROGRESS_WARN_SECS)
            if done:
                break
            logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        batch = request.result()

        entries = []
        for e in batch["groups"]:
            entry = self._release((e["group"], e["epoch"], e["attempt"]))
            assert entry is not None, f"coordinator sent {e}, which this buffer does not hold"
            entries.append(entry)
            self._prompt_groups.pop(e["group"], None)
            self._forget_attempts(e["group"])
            self._metrics["trained_groups"] += 1
            stats = group_weight_version_stats(entry.group)
            if (staleness := stats.oldest_lag(current_version)) is not None:
                self._metric_consumed_staleness.append(staleness)
            self._record_selected_version_stats(stats, current_version)
        return entries

    def _record_selected_version_stats(self, stats, current_version: int | None) -> None:
        # Same accounting as DefaultDataBuffer._record_selected_version_stats.
        self._metric_selected_samples += stats.sample_count
        self._metric_selected_versioned_samples += stats.versioned_sample_count
        if stats.oldest_version is not None and stats.newest_version is not None:
            self._metric_selected_version_span.append(stats.newest_version - stats.oldest_version)
        if (newest_lag := stats.newest_lag(current_version)) is not None:
            self._metric_selected_newest_lag.append(newest_lag)
        if (token_weighted_lag := stats.token_weighted_lag(current_version)) is not None:
            self._metric_selected_token_lag_sum += token_weighted_lag * stats.versioned_token_count
            self._metric_selected_versioned_tokens += stats.versioned_token_count

    # ------------------------------------------------------------ decisions --

    def _on_decision(self, d: dict[str, Any]) -> None:
        kind = d["type"]
        self._metrics[kind] += 1
        if kind == "recycle":
            g = d["group"]
            # Stop attempts of the old epoch before Miles' handler resets and
            # requeues the prompt group.
            if self.on_recycle is not None:
                self.on_recycle(g, d["next_epoch"])
            self._forget_attempts(g, before_epoch=d["next_epoch"])
            held = [self._release(rid) for rid in [r for r in self._payloads if r[0] == g]]
            prompt_group = self._prompt_groups.get(g)
            if d["reason"] == "stale":
                self._metric_stale_groups += 1
            if prompt_group is None:
                return
            record_dropped_samples([prompt_group, *(entry.group for entry in held)])
            for s in prompt_group:
                s.metadata.update({EPOCH_KEY: d["next_epoch"], ATTEMPT_KEY: 0})
            self._unused_handler_fn(prompt_group, _REASONS[d["reason"]])
        elif kind == "cancel":
            if self.on_cancel is not None:
                self.on_cancel((d["group"], d["epoch"], d["attempt"]))
        elif kind == "discard":
            # A duplicate or superseded result; its prompt group lives on.
            entry = self._release((d["group"], d["epoch"], d["attempt"]))
            if entry is not None:
                record_dropped_samples([entry.group])
        elif kind == "restarted":
            logger.info(
                "coordinator switched to run %s at version %s (%s untrained groups from earlier runs recycled)",
                d["run"],
                d["version"],
                d["recycled"],
            )
        elif kind == "starved":
            logger.warning("group %s recycled %s times (long or slow rollout?)", d["group"], d["recycles"])
        elif kind == "error":
            logger.error("coordinator rejected a message: %s", d["message"])

    # ----------------------------------------------------------- checkpoint --

    def state_dict(self) -> list[DataBufferInput]:
        """The results this buffer holds, one per group, like DefaultDataBuffer's
        list of buffered entries. Hydro ids are not saved: the restored job is a
        new trainer run and gets fresh ids."""
        seen: set[int] = set()
        state = []
        for (g, _, _), entry in self._payloads.items():
            if g not in seen:
                seen.add(g)
                state.append(entry)
        return state + list(self._restored)

    def load_state_dict(self, state: list[DataBufferInput]) -> None:
        assert not self._payloads and not self._restored
        for entry in state:
            if regenerated_on_restore(entry.group, use_output_store=self._args.use_output_store):
                record_dropped_samples([entry.prompt_group, entry.group])
                self._unused_handler_fn(entry.prompt_group, UnusedReason.RESTORED)
                continue
            for s in _samples([entry.prompt_group, entry.group]):
                for key in HYDRO_KEYS:
                    s.metadata.pop(key, None)
            entry.hydro_rid = None
            self._restored.append(entry)
        if self._client._writer is not None:  # loaded after connecting
            self._resubmit_restored()

    def _resubmit_restored(self) -> None:
        # The checkpoint saved these with the weights they were made with, so
        # their versions are valid in the restored run. Miles does not re-filter
        # restored entries, and neither do we.
        for entry in self._restored:
            rid = self.assign_ids(entry.prompt_group)
            entry.hydro_rid = rid
            self._submit(entry, rid)
        self._restored = []

    # -------------------------------------------------------------- metrics --

    def _reset_metrics(self) -> None:
        self._metrics: Counter = Counter()
        self._metric_gatherer = MetricGatherer()
        self._metric_aborted_groups = 0
        self._metric_stale_groups = 0
        self._metric_consumed_staleness: list[int] = []
        self._metric_selected_newest_lag: list[int] = []
        self._metric_selected_version_span: list[int] = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        # DefaultDataBuffer's keys first, then the coordinator's own.
        prefix = "rollout/fully_async/"
        metrics: dict[str, float] = {
            f"{prefix}queue_size": len(self._payloads),
            f"{prefix}aborted_groups_filtered": self._metric_aborted_groups,
            f"{prefix}stale_groups_filtered": self._metric_stale_groups,
            **self._metric_gatherer.collect(),
        }
        if consumed := self._metric_consumed_staleness:
            metrics[f"{prefix}avg_staleness"] = sum(consumed) / len(consumed)
            metrics[f"{prefix}max_staleness"] = max(consumed)
        if newest_lag := self._metric_selected_newest_lag:
            metrics[f"{prefix}avg_post_generation_staleness"] = sum(newest_lag) / len(newest_lag)
            metrics[f"{prefix}max_post_generation_staleness"] = max(newest_lag)
        if version_span := self._metric_selected_version_span:
            metrics[f"{prefix}avg_generation_version_span"] = sum(version_span) / len(version_span)
            metrics[f"{prefix}max_generation_version_span"] = max(version_span)
        if self._metric_selected_versioned_tokens:
            metrics[f"{prefix}token_weighted_staleness"] = (
                self._metric_selected_token_lag_sum / self._metric_selected_versioned_tokens
            )
        if self._metric_selected_samples:
            metrics[f"{prefix}weight_version_sample_coverage"] = (
                self._metric_selected_versioned_samples / self._metric_selected_samples
            )
        buffered = [
            s
            for entry in self._payloads.values()
            if (s := group_staleness(entry.group, self._current_version)) is not None
        ]
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        if self.extra_metrics is not None:
            metrics.update(self.extra_metrics())
        hydro = "rollout/hydro/"
        metrics.update({f"{hydro}{k}": float(v) for k, v in self._metrics.items()})
        metrics[f"{hydro}payloads_held"] = float(len(self._payloads))
        self._reset_metrics()
        return metrics
