"""Python mirror of ``coordinator/src/state.rs::check_invariants`` for live traces."""

from __future__ import annotations

from typing import Any


def check_invariants(decisions: list[dict[str, Any]], max_staleness: int) -> None:
    trained: set[int] = set()
    trained_results: set[tuple[int, int, int]] = set()
    discarded: set[tuple[int, int, int]] = set()
    min_epoch: dict[int, int] = {}
    last = (0, 0)  # (run, version)
    for i, d in enumerate(decisions):
        t = d["type"]
        if t == "batch":
            run, v = d.get("run", 0), d["version"]
            assert (run, v) >= last, f"#{i}: batch (run, version) {(run, v)} went backwards from {last}"
            last = (run, v)
            for e in d["groups"]:
                rid = (e["group"], e["epoch"], e["attempt"])
                assert e.get("run", 0) == run, f"#{i}: {rid} from run {e.get('run', 0)} trained in run {run}"
                assert e["group"] not in trained, f"#{i}: group {e['group']} trained on twice"
                trained.add(e["group"])
                lag = max(0, v - e["oldest_version"])
                assert lag <= max_staleness, f"#{i}: group {e['group']} staleness {lag} > {max_staleness}"
                assert e["epoch"] >= min_epoch.get(e["group"], 0), f"#{i}: {rid} predates a recycle"
                assert rid not in discarded, f"#{i}: {rid} was discarded, then trained on"
                trained_results.add(rid)
        elif t == "recycle":
            assert d["group"] not in trained, f"#{i}: group {d['group']} recycled after training"
            min_epoch[d["group"]] = d["next_epoch"]
        elif t == "discard":
            rid = (d["group"], d["epoch"], d["attempt"])
            assert rid not in trained_results, f"#{i}: {rid} discarded after being trained on"
            discarded.add(rid)
