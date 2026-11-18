"""Trainer restart against a long-lived coordinator.

    python -m hydro_rl.restart_demo               # run ids on: both jobs train
    python -m hydro_rl.restart_demo --no-run-ids  # reproduces the livelock

Job A trains for a few steps, reaching weight version N, and exits. Job B
then starts against the *same* coordinator with a fresh trainer at version 0,
like a crash restart or a resume from an earlier checkpoint. Without run ids
the coordinator still believes the trainer is at version N, so job B's
rollouts all look stale and B never gets a batch, even though every safety
invariant holds.
"""

from __future__ import annotations

import argparse
import asyncio

from hydro_rl.compare import start_coordinator, stop
from hydro_rl.mock_run import Workload, main_async


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run-ids", action="store_true", help="both jobs use run 0 (the old protocol)")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--port", type=int, default=7960)
    args = ap.parse_args()

    wl = Workload(train_steps=args.steps, seed=0)
    proc = start_coordinator(args.port, wl.max_staleness, wl.group_size)
    try:
        for job in ("A", "B"):
            run = 0 if args.no_run_ids else None  # None: fresh, increasing run id
            r = asyncio.run(main_async("hydro", "127.0.0.1", args.port, Workload(train_steps=args.steps, seed=0), run))
            e = r["events"]
            print(
                f"job {job}: steps={r['train_steps_completed']}/{args.steps} wall={r['wall_s']}s "
                f"stale_recycles={e.get('recycle/stale', 0)} restart_recycles={e.get('recycle/restart', 0)} "
                f"discards={sum(v for k, v in e.items() if k.startswith('discard'))} "
                f"invariants={r['invariants']}",
                flush=True,
            )
    finally:
        stop(proc)


if __name__ == "__main__":
    main()
