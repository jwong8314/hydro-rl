"""End-to-end async-RL run against mock inference workers, with fault injection.

    python -m hydro_rl.mock_run --mode hydro      # needs `cargo run --example serve`
    python -m hydro_rl.mock_run --mode baseline   # in-process Miles-like buffer

Workload: prompt groups with heavy-tailed response lengths; generation time is
proportional to length; some rollouts straggle (10x slower) and some hang
forever. A trainer consumes `batch_size` groups per step and bumps the weight
version after each step. A rollout records the version it started at
(``oldest_version``) and finished at, like Miles' per-token weight-version spans.

`baseline` models Miles' `DefaultDataBuffer` + `FullyAsyncRolloutFn`: FIFO,
staleness checked at get(), stale groups recycled, and a rollout holds its
concurrency slot until it returns (no per-rollout timeout).

`hydro` adds what the coordinator makes safe: a rollout that exceeds its
deadline is *hedged* (retried while the original keeps running); whichever
attempt finishes first is admitted, the other is discarded or cancelled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from hydro_rl.client import CoordinatorClient
from hydro_rl.invariants import check_invariants


@dataclass
class Workload:
    n_prompts: int = 400
    group_size: int = 4
    batch_size: int = 8
    max_in_flight: int = 24
    train_steps: int = 30
    train_time: float = 0.25
    sec_per_token: float = 0.0004
    straggler_p: float = 0.05
    hang_p: float = 0.03
    max_staleness: int = 1
    seed: int = 0
    lengths: list[int] = field(default_factory=list)

    def __post_init__(self):
        rng = random.Random(self.seed)
        # heavy-tailed response lengths: most short, some very long
        self.lengths = [min(4000, int(100 * rng.paretovariate(1.2))) for _ in range(self.n_prompts)]


class Trainer:
    def __init__(self):
        self.version = 0


# --------------------------------------------------------------------------- #
# Buffers. Both expose the coordinator's protocol: events in, decisions out.
# --------------------------------------------------------------------------- #


class MilesLikeBuffer:
    """In-process model of Miles' DefaultDataBuffer (FIFO, stale-on-get, recycle)."""

    def __init__(self, wl: Workload, on_decision):
        self.wl, self.on_decision = wl, on_decision
        self.buffer: list[dict] = []
        self.epoch: dict[int, int] = {}
        self.cond = asyncio.Condition()

    def dispatch(self, group, epoch, attempt):
        pass

    def abort(self, group, epoch, attempt):
        self._recycle(group, "aborted")

    def complete(self, group, epoch, attempt, *, oldest_version, newest_version, n_samples, tokens):
        self.buffer.append(
            {"group": group, "epoch": epoch, "attempt": attempt, "oldest_version": oldest_version, "tokens": tokens}
        )
        asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self):
        async with self.cond:
            self.cond.notify_all()

    def version(self, v):
        self.current = v

    def _recycle(self, group, reason):
        self.epoch[group] = self.epoch.get(group, 0) + 1
        self.on_decision({"type": "recycle", "group": group, "reason": reason, "next_epoch": self.epoch[group]})

    async def get_batch(self, n_groups, current_version):
        async with self.cond:
            while True:
                kept = []
                for e in self.buffer:
                    if current_version - e["oldest_version"] > self.wl.max_staleness:
                        self._recycle(e["group"], "stale")
                    else:
                        kept.append(e)
                self.buffer = kept
                if len(self.buffer) >= n_groups:
                    break
                await self.cond.wait()
            groups, self.buffer = self.buffer[:n_groups], self.buffer[n_groups:]
        return {"type": "batch", "version": current_version, "groups": groups}

    async def flush(self):
        pass


class HydroBuffer:
    def __init__(self, client: CoordinatorClient):
        self.c = client

    def __getattr__(self, name):
        return getattr(self.c, name)

    async def get_batch(self, n_groups, current_version):
        self.c.version(current_version)
        return await self.c.get_batch(n_groups)


# --------------------------------------------------------------------------- #
# Producer: keeps up to max_in_flight groups generating.
# --------------------------------------------------------------------------- #


class Producer:
    def __init__(self, wl: Workload, trainer: Trainer, hedge: bool, rng: random.Random):
        self.wl, self.trainer, self.hedge, self.rng = wl, trainer, hedge, rng
        self.buf = None
        # Group ids must be unique for the coordinator's lifetime (it remembers
        # every group it has seen), so each run gets its own id range.
        self.base = random.Random().getrandbits(31) << 16
        self.queue: deque[int] = deque(self.base + i for i in range(wl.n_prompts))
        self.epoch: dict[int, int] = {}
        self.next_attempt: dict[int, int] = {}
        self.tasks: dict[tuple[int, int, int], asyncio.Task] = {}
        self.slots = asyncio.Semaphore(wl.max_in_flight)
        self.decisions: list[dict] = []
        self.counts: Counter = Counter()

    # decisions from the buffer / coordinator
    def on_decision(self, d: dict[str, Any]) -> None:
        self.decisions.append(d)
        self.counts[d["type"] + ("/" + d["reason"] if "reason" in d else "")] += 1
        if d["type"] == "recycle":
            g = d["group"]
            if d["next_epoch"] > self.epoch.get(g, 0):
                self.epoch[g] = d["next_epoch"]
                self.next_attempt[g] = 0
                # generations from older epochs can no longer be used
                for key in [k for k in self.tasks if k[0] == g and k[1] < d["next_epoch"]]:
                    self.tasks.pop(key).cancel()
                self.queue.appendleft(g)  # regenerate soon
        elif d["type"] == "cancel":
            t = self.tasks.pop((d["group"], d["epoch"], d["attempt"]), None)
            if t is not None:
                t.cancel()

    async def run(self) -> None:
        while True:
            if not self.queue:
                await asyncio.sleep(0.01)
                continue
            await self.slots.acquire()
            g = self.queue.popleft()
            self._start(g)

    def _start(self, g: int, hedged_slot: bool = False) -> None:
        e = self.epoch.get(g, 0)
        a = self.next_attempt.get(g, 0)
        self.next_attempt[g] = a + 1
        self.buf.dispatch(g, e, a)
        self.counts["dispatch"] += 1
        task = asyncio.create_task(self._generate(g, e, a))
        self.tasks[(g, e, a)] = task
        task.add_done_callback(lambda _t, key=(g, e, a): self._done(key))
        if self.hedge:
            asyncio.create_task(self._watchdog(g, e, a, task))

    def _done(self, key) -> None:
        self.tasks.pop(key, None)
        self.slots.release()

    async def _generate(self, g: int, e: int, a: int) -> None:
        tokens = self.wl.lengths[g - self.base]
        start_version = self.trainer.version
        latency = tokens * self.wl.sec_per_token
        roll = self.rng.random()
        if roll < self.wl.hang_p:
            self.counts["injected/hang"] += 1
            await asyncio.Event().wait()  # never finishes
        elif roll < self.wl.hang_p + self.wl.straggler_p:
            self.counts["injected/straggler"] += 1
            latency *= 10
        await asyncio.sleep(latency)
        self.buf.complete(
            g,
            e,
            a,
            oldest_version=start_version,
            newest_version=self.trainer.version,
            n_samples=self.wl.group_size,
            tokens=tokens,
        )

    async def _watchdog(self, g, e, a, task) -> None:
        # Deadline: 3x the expected time for this prompt (plus slack). Real
        # systems would use a length-independent deadline or a percentile of
        # observed latencies; the prompt's true length is a mock convenience.
        await asyncio.sleep(3 * self.wl.lengths[g - self.base] * self.wl.sec_per_token + 0.05)
        if not task.done() and self.epoch.get(g, 0) == e:
            self.counts["hedge"] += 1
            await self.slots.acquire()
            self._start(g)


# --------------------------------------------------------------------------- #


async def main_async(mode: str, host: str, port: int, wl: Workload, run: int | None = None) -> dict:
    """`run`: trainer incarnation id for the coordinator (default: a fresh,
    increasing one). Passing the same run to two jobs reproduces the livelock."""
    rng = random.Random(wl.seed + 1)
    trainer = Trainer()
    producer = Producer(wl, trainer, hedge=(mode == "hydro"), rng=rng)
    if mode == "hydro":
        client = CoordinatorClient(host, port, on_decision=producer.on_decision, run=run)
        await client.connect()
        buf = HydroBuffer(client)
    else:
        buf = MilesLikeBuffer(wl, producer.on_decision)
    producer.buf = buf

    prod_task = asyncio.create_task(producer.run())
    trained_tokens: list[int] = []
    step_times: list[float] = []
    t0 = time.monotonic()
    for _step in range(wl.train_steps):
        ts = time.monotonic()
        try:
            batch = await asyncio.wait_for(buf.get_batch(wl.batch_size, trainer.version), timeout=20)
        except TimeoutError:
            print(f"[{mode}] step {_step}: no batch in 20s (all in-flight slots stuck?)", file=sys.stderr)
            break
        producer.decisions.append(batch)
        trained_tokens += [e["tokens"] for e in batch["groups"]]
        await asyncio.sleep(wl.train_time)  # "train"
        trainer.version += 1
        step_times.append(time.monotonic() - ts)
    elapsed = time.monotonic() - t0
    prod_task.cancel()
    for t in list(producer.tasks.values()):
        t.cancel()

    check_invariants(producer.decisions, wl.max_staleness)

    pool_mean = statistics.mean(wl.lengths)
    long_cut = sorted(wl.lengths)[int(0.8 * len(wl.lengths))]
    return {
        "mode": mode,
        "train_steps_completed": len(step_times),
        "wall_s": round(elapsed, 2),
        "mean_step_s": round(statistics.mean(step_times), 3) if step_times else None,
        "groups_trained": len(trained_tokens),
        "mean_tokens_trained": round(statistics.mean(trained_tokens), 1) if trained_tokens else None,
        "mean_tokens_prompt_pool": round(pool_mean, 1),
        "share_long_trained(top20%)": round(sum(t >= long_cut for t in trained_tokens) / max(1, len(trained_tokens)), 3),
        "invariants": "ok",
        "events": dict(sorted(producer.counts.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hydro", "baseline"], default="hydro")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7878)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    wl = Workload(train_steps=args.steps, seed=args.seed)
    print(json.dumps(asyncio.run(main_async(args.mode, args.host, args.port, wl)), indent=2))


if __name__ == "__main__":
    main()
