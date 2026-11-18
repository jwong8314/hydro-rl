"""Run baseline vs hydro on the same workloads, with a fresh coordinator per run.

    python -m hydro_rl.compare --seeds 0 1 2 3

A fresh coordinator per run keeps the runs independent. Reusing one is safe now
that each trainer stamps a run id (see restart_demo.py), but it would carry the
first run's leftover groups into the second run's metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import subprocess
import time

from hydro_rl.mock_run import Workload, main_async

ROOT = pathlib.Path(__file__).resolve().parents[2]
CARGO = ["cargo", "+1.98.1-aarch64-apple-darwin"]


def start_coordinator(
    port: int, max_staleness: int | None, group_size: int, oldest_first: bool = True
) -> subprocess.Popen:
    """`max_staleness=None`: no limit. The mock experiments use oldest-first;
    Miles' own buffer is FIFO (`oldest_first=False`)."""
    log = open(f"/tmp/hydro_rl_serve_{port}.log", "w")
    staleness = "none" if max_staleness is None else str(max_staleness)
    proc = subprocess.Popen(
        [
            *CARGO, "run", "-q", "--example", "serve", "--",
            str(port), staleness, str(group_size), str(oldest_first).lower(),
        ],
        cwd=ROOT / "coordinator",
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 300
    while time.time() < deadline:
        if "listening" in pathlib.Path(log.name).read_text():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(pathlib.Path(log.name).read_text())
        time.sleep(0.3)
    raise TimeoutError("coordinator did not start")


def stop(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, 15)
    proc.wait(timeout=30)


def row(r: dict) -> str:
    e = r["events"]
    return (
        f"{r['mode']:8} steps={r['train_steps_completed']:2} wall={r['wall_s']:5.1f}s "
        f"trained_tok={r['mean_tokens_trained']} pool_tok={r['mean_tokens_prompt_pool']} "
        f"long_share={r['share_long_trained(top20%)']:.3f} stale_recycles={e.get('recycle/stale', 0)} "
        f"starved={e.get('starved', 0)} hangs={e.get('injected/hang', 0)} hedges={e.get('hedge', 0)} "
        f"cancels={e.get('cancel', 0)} invariants={r['invariants']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    for seed in args.seeds:
        wl = Workload(train_steps=args.steps, seed=seed)
        print(f"seed={seed} " + row(asyncio.run(main_async("baseline", "", 0, wl))), flush=True)
        port = 7900 + seed
        proc = start_coordinator(port, wl.max_staleness, wl.group_size)
        try:
            wl = Workload(train_steps=args.steps, seed=seed)
            print(f"seed={seed} " + row(asyncio.run(main_async("hydro", "127.0.0.1", port, wl))), flush=True)
        finally:
            stop(proc)


if __name__ == "__main__":
    main()
