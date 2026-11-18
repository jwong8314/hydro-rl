"""Retried rollouts in Miles' fully-async producer, against a live coordinator.

Every 4th prompt's first request hangs forever (a stuck engine); a second
request for it finishes normally. With 4 groups in flight, Miles' producer
fills up with hung rollouts and stops. With retries, the hung rollouts get a
second attempt, the winners are trained exactly once, and the hung originals
are cancelled.

    python tests/test_miles_retries.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import miles_fakes  # noqa: E402  (installs Miles stand-ins)
from miles_fakes import ENGINES, SAVED, FullyAsyncRolloutFn, Sample  # noqa: E402

from hydro_rl.compare import start_coordinator, stop  # noqa: E402
from hydro_rl.miles_buffer import HYDRO_KEYS, HydroDataBuffer  # noqa: E402

GROUP_SIZE = 2
IN_FLIGHT = 4  # --rollout-batch-size: groups in flight, and groups per training step
STEPS = 6
STUCK = {3, 7, 11, 15, 19, 23}
SLOW = 9  # past the retry deadline, but finishes before its retry would


class DataSource:
    def __init__(self):
        self.next = 0

    def get_samples(self, n: int):
        groups = [[Sample(index=self.next + i) for _ in range(GROUP_SIZE)] for i in range(n)]
        self.next += n
        return groups


def make_rollout_fn(retries: bool) -> FullyAsyncRolloutFn:
    os.environ.update(
        HYDRO_RETRY="1" if retries else "0",
        HYDRO_RETRY_WARMUP="4",
        HYDRO_RETRY_MIN_SECONDS="0.15",
        HYDRO_RETRY_BUDGET="2",
    )
    args = SimpleNamespace(
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        n_samples_per_prompt=GROUP_SIZE,
        rollout_batch_size=IN_FLIGHT,
        async_data_buffer_capacity_factor=2.0,
        use_output_store=False,
    )
    # Miles constructs the rollout function, which constructs the buffer.
    fn = FullyAsyncRolloutFn(args, DataSource(), HydroDataBuffer)
    assert type(fn).__name__ == "HydroFullyAsyncRolloutFn", type(fn)
    return fn


async def train(fn, step_timeout: float) -> tuple[list, int]:
    """The trainer's side: one get of IN_FLIGHT groups per step, then a weight update."""
    entries, steps = [], 0
    for step in range(STEPS):
        try:
            batch = await asyncio.wait_for(fn._output.get(num_groups=IN_FLIGHT, current_version=step), step_timeout)
        except TimeoutError:
            break
        entries += batch
        steps += 1
        ENGINES.version = step + 1
    return entries, steps


async def shutdown(fn, worker) -> None:
    worker.cancel()
    for task in list(fn._attempts):
        task.cancel()
    await asyncio.gather(worker, *fn._attempts, return_exceptions=True)


async def with_retries() -> None:
    fn = make_rollout_fn(retries=True)
    worker = asyncio.create_task(fn._worker_loop())
    entries, steps = await train(fn, step_timeout=10)
    trained = [e.group[0].index for e in entries]
    metrics = fn._output.get_metrics()

    assert steps == STEPS, f"only {steps} steps"
    assert len(trained) == len(set(trained)), f"a prompt was trained twice: {sorted(trained)}"
    stuck_trained = STUCK & set(trained)
    assert len(stuck_trained) >= 3, (stuck_trained, sorted(trained))
    assert all(ENGINES.requests[i] >= 2 for i in stuck_trained)
    # The slow prompt got a retry, but its original won: trained once, and the
    # retry was cancelled.
    assert SLOW in trained and ENGINES.requests[SLOW] == 2, ENGINES.requests.get(SLOW)
    assert metrics["rollout/hydro/retry_retries"] >= len(stuck_trained) + 1, metrics
    assert metrics["rollout/hydro/retry_cancelled"] >= len(stuck_trained) + 1, metrics
    # Trained samples are an attempt's own copy; the prompt group was never generated on.
    for e in entries:
        assert e.group[0].response.startswith("answer") and e.prompt_group[0].response == ""
        assert e.group[0] is not e.prompt_group[0]
    # No hung original of a trained group is still running.
    live = {a.rid[0] for a in fn._attempts.values() if not a.superseded}
    assert not live & {e.hydro_rid[0] for e in entries}

    # A checkpoint lists each running group once, never one whose result the
    # buffer holds, and carries no Hydro ids into the restored job.
    fn.save("ckpt")
    state = SAVED[-1]
    running = [pg[0].index for pg in state.running]
    held = [entry.group[0].index for entry in state.output]
    assert len(running) == len(set(running)) and not set(running) & set(held), (running, held)
    for pg in [*state.running, *state.retry_buffer]:
        assert not any(k in s.metadata for s in pg for k in HYDRO_KEYS)

    print(
        f"ok: retries on: {steps}/{STEPS} steps, {len(trained)} groups trained once each, "
        f"stuck prompts trained via a retry: {sorted(stuck_trained)}, "
        f"retries={metrics['rollout/hydro/retry_retries']:.0f} "
        f"cancelled={metrics['rollout/hydro/retry_cancelled']:.0f} "
        f"deadline={metrics.get('rollout/hydro/retry_deadline_s', 0):.2f}s"
    )
    await shutdown(fn, worker)


async def without_retries() -> None:
    ENGINES.requests.clear()
    ENGINES.version = 0
    fn = make_rollout_fn(retries=False)
    worker = asyncio.create_task(fn._worker_loop())
    entries, steps = await train(fn, step_timeout=3)
    trained = [e.group[0].index for e in entries]
    assert steps < STEPS, "expected the producer to stall on hung rollouts"
    assert not STUCK & set(trained)
    hung = sum(1 for a in fn._attempts.values() if a.samples[0].index in STUCK)
    print(f"ok: retries off (Miles' behavior): stalled after {steps}/{STEPS} steps, {hung} hung rollouts hold the slots")
    await shutdown(fn, worker)


async def main_async() -> None:
    ENGINES.latency, ENGINES.stuck = 0.03, STUCK
    ENGINES.slow = {SLOW: (0.4, 5.0)}
    await with_retries()
    await without_retries()


def main() -> None:
    port = 7951
    os.environ["HYDRO_COORDINATOR"] = f"127.0.0.1:{port}"
    proc = start_coordinator(port, None, 0, oldest_first=False)
    try:
        asyncio.run(main_async())
    finally:
        stop(proc)
    assert miles_fakes.DROPPED is not None


if __name__ == "__main__":
    main()
