"""Exercise `HydroDataBuffer` against a real coordinator, with stand-ins for the
Miles modules it imports (Miles needs a GPU stack to import for real). Each
scenario checks a piece of parity with Miles' `DefaultDataBuffer`.

    python tests/test_miles_buffer.py
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from miles_fakes import DROPPED, OWNERSHIP_LOG, DataBufferInput, Sample, UnusedReason  # noqa: E402  (installs Miles stand-ins)

from hydro_rl.compare import start_coordinator, stop  # noqa: E402
from hydro_rl.miles_buffer import EPOCH_KEY, GROUP_KEY, HydroDataBuffer  # noqa: E402

GROUP_SIZE = 2
MAX_STALENESS = 1


def make_buffer(*, run: int | None = None, capacity_factor: float = 4.0, max_staleness=MAX_STALENESS, sink=None):
    if run is not None:
        os.environ["HYDRO_RUN"] = str(run)
    try:
        return HydroDataBuffer(
            SimpleNamespace(
                args=SimpleNamespace(
                    dynamic_sampling_filter_path=None,
                    max_weight_staleness=max_staleness,
                    n_samples_per_prompt=GROUP_SIZE,
                    rollout_batch_size=1,
                    async_data_buffer_capacity_factor=capacity_factor,
                    use_output_store=False,
                ),
                unused_handler_fn=lambda pg, reason: (sink if sink is not None else []).append((pg, reason)),
            )
        )
    finally:
        os.environ.pop("HYDRO_RUN", None)


def prompt(index: int, **kw) -> list:
    return [Sample(index=index, **kw) for _ in range(GROUP_SIZE)]


def generate(buf, prompt_group: list, version: int | None, **kw) -> DataBufferInput:
    """What HydroFullyAsyncRolloutFn does: fix the id at dispatch, then generate.
    Like Miles, result samples share the prompt samples' metadata dict."""
    rid = buf.assign_ids(prompt_group)
    out = DataBufferInput(
        prompt_group=prompt_group,
        group=[Sample(index=s.index, version=version, metadata=s.metadata, **kw) for s in prompt_group],
    )
    out.hydro_rid = rid
    return out


async def wait_for(cond, what: str) -> None:
    for _ in range(100):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def dedup_and_staleness() -> HydroDataBuffer:
    recycled: list = []
    buf = make_buffer(sink=recycled)
    await buf._ensure_connected()
    a = generate(buf, prompt(0), version=0)
    b = generate(buf, prompt(1), version=0)
    await buf.put(a)
    await buf.put(a)  # duplicate delivery of the same result
    await buf.put(b)

    [first] = await buf.get(num_groups=1, current_version=0)
    assert first is a, "FIFO: group a first"

    # weights jump to version 2 -> b (version 0) is now stale and is recycled
    task = asyncio.create_task(buf.get(num_groups=1, current_version=2))
    await wait_for(lambda: recycled, "stale recycle")
    pg, reason = recycled[0]
    assert reason == UnusedReason.STALE and pg is b.prompt_group and pg[0].metadata[EPOCH_KEY] == 1

    await buf.put(b)  # the stale result arrives again (late duplicate) -> discarded
    regen = generate(buf, pg, version=2)  # regenerated, as Miles' retry handler does
    await buf.put(regen)
    [second] = await asyncio.wait_for(task, 5)
    assert second is regen

    m = buf.get_metrics()
    assert m["rollout/hydro/trained_groups"] == 2 and m["rollout/hydro/recycle"] == 1, m
    assert m["rollout/hydro/discard"] == 1, m
    assert m["rollout/fully_async/stale_groups_filtered"] == 1, m
    assert m["rollout/fully_async/queue_size"] == 0, m
    assert m["rollout/fully_async/avg_staleness"] == 0, m  # a at v0 trained at v0, regen at v2 at v2
    print("ok: dedup + staleness")
    return buf


async def unversioned_and_drop(buf: HydroDataBuffer) -> None:
    # A group whose tokens carry no weight version is never stale (Miles: None lag).
    u = generate(buf, prompt(10), version=None)
    await buf.put(u)
    [got] = await buf.get(num_groups=1, current_version=50)
    assert got is u

    # Missing reward: dropped, recorded, logged, and the group is closed on the
    # coordinator (no recycle now or ever).
    before = len(DROPPED)
    d = generate(buf, prompt(11), version=50, reward=None)
    await buf.put(d)
    await asyncio.sleep(0.2)
    m = buf.get_metrics()
    assert OWNERSHIP_LOG == ["missing_reward"], OWNERSHIP_LOG
    assert len(DROPPED) == before + 1
    assert m["rollout/dynamic_filter/drop_missing_reward"] == 1, m
    assert "rollout/hydro/recycle" not in m and m["rollout/hydro/payloads_held"] == 0, m
    print("ok: unversioned groups never stale; filtered groups dropped and closed")


async def backpressure() -> None:
    buf = make_buffer(capacity_factor=2.0)  # capacity 2 groups
    await buf._ensure_connected()
    await buf.put(generate(buf, prompt(20), version=0))
    await buf.put(generate(buf, prompt(21), version=0))
    third = asyncio.create_task(buf.put(generate(buf, prompt(22), version=0)))
    await asyncio.sleep(0.3)
    assert not third.done(), "put should block while the buffer is full"
    await buf.get(num_groups=1, current_version=0)
    await asyncio.wait_for(third, 5)
    assert buf.get_metrics()["rollout/fully_async/queue_size"] == 2
    print("ok: put blocks at capacity")


async def checkpoint_restore(old: HydroDataBuffer) -> None:
    # Job 1 holds two finished groups at version 5 when it checkpoints; one of
    # them has an output-store ref that won't survive the restart.
    await old.get(num_groups=0, current_version=5)
    keep = generate(old, prompt(30), version=5)
    gone = generate(old, prompt(31), version=5, store_ref=True)
    await old.put(keep)
    await old.put(gone)
    state = copy.deepcopy(old.state_dict())  # pickled with the checkpoint
    assert len(state) == 2

    # Job 2: a new trainer process restored at version 5.
    restored: list = []
    new = make_buffer(run=old._client.run + 1, sink=restored)
    new.load_state_dict(state)
    assert [r for _, r in restored] == [UnusedReason.RESTORED]
    [got] = await asyncio.wait_for(new.get(num_groups=1, current_version=5), 5)
    assert got.group[0].index == 30 and got.prompt_group[0].metadata[GROUP_KEY] != keep.hydro_rid[0]
    print("ok: checkpoint save/restore")


async def restart_at_lower_version(old: HydroDataBuffer) -> None:
    # Restarting from an earlier checkpoint (version 0) while the coordinator
    # has seen version 5: a higher run id makes the coordinator accept it.
    new = make_buffer(run=old._client.run + 10)
    await new._ensure_connected()
    get = asyncio.create_task(new.get(num_groups=1, current_version=0))  # announces the restart
    await asyncio.sleep(0.2)
    c = generate(new, prompt(40), version=0)
    await new.put(c)
    [got] = await asyncio.wait_for(get, 5)
    assert got is c
    print("ok: restarted trainer at version 0 trains")


async def config_mismatch_is_refused() -> None:
    buf = make_buffer(max_staleness=None)  # Miles default: no limit; coordinator has 1
    try:
        await buf._ensure_connected()
    except RuntimeError as e:
        assert "--max-weight-staleness=None" in str(e), e
        print("ok: mismatched coordinator config refused")
    else:
        raise AssertionError("expected a config mismatch error")


async def main_async() -> None:
    buf = await dedup_and_staleness()
    await unversioned_and_drop(buf)
    # Each new buffer is a newer trainer run, which fences off older ones, so
    # the scenarios that reuse `buf` go first.
    await checkpoint_restore(buf)
    await restart_at_lower_version(buf)
    await backpressure()
    await config_mismatch_is_refused()


def main() -> None:
    port = 7950
    os.environ["HYDRO_COORDINATOR"] = f"127.0.0.1:{port}"
    proc = start_coordinator(port, MAX_STALENESS, GROUP_SIZE, oldest_first=False)
    try:
        asyncio.run(main_async())
    finally:
        stop(proc)


if __name__ == "__main__":
    main()
