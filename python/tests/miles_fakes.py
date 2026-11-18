"""Stand-ins for the Miles modules the adapter imports. Miles needs its GPU stack
to import for real; these mirror only the APIs the adapter calls, with the
behavior those calls rely on. Import this module before anything from
`hydro_rl.miles_*`.
"""

from __future__ import annotations

import asyncio
import enum
import sys
import types
import uuid
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class Sample:
    class Status(enum.Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        ABORTED = "aborted"

    index: int
    version: int | None = 0
    tokens: int = 10
    aborted: bool = False
    reward: float | None = 1.0
    store_ref: bool = False  # an output-store ref that won't survive a restart
    routing_key: str | None = None
    kv_cache_namespace: str | None = None
    response: str = ""
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def reset_for_retry(self) -> None:  # keeps index, routing_key and metadata, like Miles
        self.response = ""
        self.reward = None
        self.version = None
        self.kv_cache_namespace = None
        self.status = Sample.Status.ABORTED


class UnusedReason(str, enum.Enum):
    ABORTED = "aborted"
    STALE = "stale"
    RESTORED = "restored"


@dataclass
class DataBufferInput:
    prompt_group: list
    group: list


def _keep(flag: bool, reason: str | None = None):
    return SimpleNamespace(keep=flag, reason=reason)


@dataclass
class Stats:
    sample_count: int
    versioned_sample_count: int
    versioned_token_count: int
    token_version_sum: int
    oldest_version: int | None
    newest_version: int | None

    def oldest_lag(self, current):
        return None if self.oldest_version is None or current is None else current - self.oldest_version

    def newest_lag(self, current):
        return None if self.newest_version is None or current is None else current - self.newest_version

    def token_weighted_lag(self, current):
        if current is None or not self.versioned_token_count:
            return None
        return current - self.token_version_sum / self.versioned_token_count


def group_weight_version_stats(group):
    v = [s for s in group if s.version is not None]
    return Stats(
        sample_count=len(group),
        versioned_sample_count=len(v),
        versioned_token_count=sum(s.tokens for s in v),
        token_version_sum=sum(s.version * s.tokens for s in v),
        oldest_version=min((s.version for s in v), default=None),
        newest_version=max((s.version for s in v), default=None),
    )


class MetricGatherer:
    def __init__(self):
        self.drops = {}

    def on_group_before_dynamic_filter(self, args, group):
        pass

    def on_dynamic_filter_drop(self, reason):
        if reason:
            self.drops[reason] = self.drops.get(reason, 0) + 1

    def collect(self):
        return {f"rollout/dynamic_filter/drop_{r}": c for r, c in self.drops.items()}


DROPPED: list = []  # what record_dropped_samples received
OWNERSHIP_LOG: list = []
SAVED: list = []  # what save_simple_checkpoint received


class SampleOwnershipRecorder:
    @staticmethod
    def log_dropped_samples(*, args, samples, reason):
        OWNERSHIP_LOG.append(reason)


# ---- the rollout side ------------------------------------------------------ #


class GroupLevelSubmission:
    """Miles' group-granularity scheduler (miles/rollout/submission_scheduler.py)."""

    sample_done_callback = None

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        return pending_groups < group_budget

    def on_submit(self, groups) -> None:
        pass

    async def wait_for_progress(self, pendings):
        return await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)


class FullyAsyncRolloutFn:
    """The parts of Miles' FullyAsyncRolloutFn.__init__ the subclass relies on.
    Like Miles, the data buffer is built here with a bound method as its
    unused-samples handler."""

    def __init__(self, args, data_source, buffer_cls):
        self.args = args
        self.data_source = data_source
        self.state = SimpleNamespace(sampling_params={})
        self._scheduler = GroupLevelSubmission()
        self._handle_unused = self._recycle
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._curr_kv_cache_namespace = "ns"
        self._output = buffer_cls(SimpleNamespace(args=args, unused_handler_fn=self._handle_unused))
        self._retry_buffer: deque = deque()
        self._running_tasks: list = []

    def _max_in_flight_groups(self) -> int:
        return self.args.rollout_batch_size

    def _recycle(self, prompt_group, reason) -> None:
        for s in prompt_group:
            s.reset_for_retry()
        self._retry_buffer.append(prompt_group)


@dataclass(frozen=True)
class _FullyAsyncRolloutState:
    retry_buffer: list
    running: list
    output: object


class FakeEngines:
    """`generate_and_rm_group` against engines where some prompts get stuck.
    A stuck prompt's *first* request never returns (an overloaded or hung
    engine); later requests for it finish normally. Generation fills in the
    samples it is given, in place, as Miles' does."""

    def __init__(self, latency: float = 0.03, stuck: set[int] = frozenset()):
        self.latency, self.stuck = latency, set(stuck)
        # index -> (seconds for the first request, seconds for later ones):
        # slow but alive, so the original can beat its retry
        self.slow: dict[int, tuple[float, float]] = {}
        self.requests: dict[int, int] = {}
        self.version = 0
        self.running = 0

    async def generate_and_rm_group(self, state, group, sampling_params, evaluation=False, sample_done_callback=None):
        index = group[0].index
        n = self.requests[index] = self.requests.get(index, 0) + 1
        self.running += 1
        try:
            for s in group:
                if s.routing_key is None:
                    s.routing_key = str(uuid.uuid4())
            if index in self.stuck and n == 1:
                await asyncio.Event().wait()
            first, later = self.slow.get(index, (self.latency, self.latency))
            await asyncio.sleep(first if n == 1 else later)
            for s in group:
                s.response, s.reward, s.version = f"answer {index}", 1.0, self.version
                s.status = Sample.Status.COMPLETED
            return group
        finally:
            self.running -= 1


ENGINES = FakeEngines()


def install() -> None:
    def generate_and_rm_group(*a, **kw):
        return ENGINES.generate_and_rm_group(*a, **kw)

    mods = {
        "miles": {},
        "miles.rollout": {},
        "miles.rollout.base_types": {
            "stamp_kv_cache_namespace": lambda groups, namespace: [
                setattr(s, "kv_cache_namespace", namespace) for g in groups for s in g
            ],
        },
        "miles.rollout.filter_hub": {},
        "miles.rollout.filter_hub.base_types": {
            "call_dynamic_filter": lambda f, a, g: _keep(True),
            "MetricGatherer": MetricGatherer,
        },
        "miles.rollout.filter_hub.common_filters": {
            "apply_aborted_filter": lambda a, g: _keep(
                not any(s.aborted or s.status == Sample.Status.ABORTED for s in g)
            ),
            "apply_missing_reward_filter": lambda a, g: _keep(
                all(s.reward is not None for s in g),
                None if all(s.reward is not None for s in g) else "missing_reward",
            ),
            "group_weight_version_stats": group_weight_version_stats,
            "group_staleness": lambda g, cur: group_weight_version_stats(g).oldest_lag(cur),
        },
        "miles.rollout.fully_async_data_buffer": {
            "DataBuffer": object,
            "DataBufferConstructorInput": SimpleNamespace,
            "DataBufferInput": DataBufferInput,
            "UnusedReason": UnusedReason,
            "NO_PROGRESS_WARN_SECS": 30.0,
        },
        "miles.rollout.fully_async_rollout": {
            "FullyAsyncRolloutFn": FullyAsyncRolloutFn,
            "_FullyAsyncRolloutState": _FullyAsyncRolloutState,
        },
        "miles.rollout.inference_rollout": {},
        "miles.rollout.inference_rollout.inference_rollout_common": {
            "generate_and_rm_group": generate_and_rm_group,
        },
        "miles.utils": {},
        "miles.utils.function_registry": {"load_function": lambda path: None},
        "miles.utils.audit_utils": {},
        "miles.utils.audit_utils.sample_ownership": {},
        "miles.utils.audit_utils.sample_ownership.recorder": {"SampleOwnershipRecorder": SampleOwnershipRecorder},
        "miles.utils.output_store": {},
        "miles.utils.output_store.dropped": {
            "record_dropped_samples": lambda items: DROPPED.append(items),
            "regenerated_on_restore": lambda group, use_output_store: any(s.store_ref for s in group),
        },
        "miles.utils.simple_checkpointer": {
            "save_simple_checkpoint": lambda directory, data: SAVED.append(data),
        },
        "miles.utils.types": {"Sample": Sample},
    }
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m


install()
