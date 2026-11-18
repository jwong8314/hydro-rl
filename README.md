# hydro-rl: a Hydro control plane for Miles' async RL (prototype)

A **layer on top of Miles**, not a fork. A Hydro (Rust) coordinator takes over the
group-level decisions Miles makes in `DefaultDataBuffer`: what gets trained, what is
too stale, what gets regenerated. Miles keeps everything else.

```
           ┌──────────────── Python / Ray (unchanged Miles) ─────────────────┐
           │  FullyAsyncRolloutFn ──HTTP──► SGLang engines (PD, KV transfer)  │
           │        │  ▲                         ▲                            │
           │   put  │  │ recycle/discard         │ NCCL / gloo weight sync    │
           │        ▼  │                         │ (Megatron → SGLang)        │
           │  HydroDataBuffer ◄── get ── trainer (Megatron)                   │
           └────────┬──▲──────────────────────────────────────────────────────┘
       JSON lines   │  │   ids, epochs, versions, token counts only
       over TCP     ▼  │
           ┌─────────────────────────┐
           │  Hydro coordinator       │  Rust; safety properties checked by
           │  (rl_coordinator)        │  exhaustive simulation
           └─────────────────────────┘
```

## Your gloo / NCCL question

**They never cross the language boundary in this design, so the question doesn't
come up.** Miles' weight sync (`update_weights` → `actor_model.update_weights`, via
the `broadcast` / `cuda_ipc` / `p2p` / `delta` protocols in
`miles/backends/training_utils/weight_update/protocols/`) and SGLang's
prefill/decode KV transfer stay Python↔Python and GPU↔GPU, exactly as they are now.
Hydro only carries control messages of tens of bytes: "group 17, epoch 0, attempt 1
finished at weight version 41". The coordinator never needs tensors; it needs ids and
version numbers.

If you later wanted Hydro to *coordinate* weight sync (deciding when to pause engines
and which version each engine serves), that is still control plane: Hydro sends
"start update v42" and Python runs the NCCL broadcast. Putting Hydro on the data plane
(moving tensors itself) would mean Rust NCCL bindings and process-group bootstrap
across runtimes. I don't think that's worth doing.

## What Miles does today (relevant parts)

- `train_async.py --fully-async`: the trainer loop. `rollout_executor.get(...)` drains
  a batch, the actor trains, then `update_weights` pushes weights and bumps
  `weight_version`.
- `miles/rollout/fully_async_rollout.py`: one asyncio producer keeps up to
  `rollout_batch_size` groups in flight and `put`s finished groups into the buffer.
- `miles/rollout/fully_async_data_buffer.py`, `DefaultDataBuffer`: FIFO with a
  capacity bound. Aborted, missing-reward and dynamic-filtered groups are rejected on
  `put`. Groups past `--max-weight-staleness` are rejected on `get`. Rejected groups
  are recycled or dropped depending on `--async-unused-samples-handler`. **Pluggable
  via `--custom-async-data-buffer-path`**, which is the seam this prototype uses.
- Staleness is per token. `Sample.all_weight_version_spans` records which weight
  version produced each span, so a rollout that straddles a weight update is
  measured by its *oldest* span.
- No per-rollout deadline: a hung generation holds its concurrency slot until the
  generate function itself aborts. Producer and buffer share one asyncio loop, so
  duplicate results can't happen today. They can as soon as you add hedged retries
  or have workers report results directly, which is what the coordinator makes safe.

## What the prototype adds

`coordinator/src/state.rs` is a pure state machine, and `coordinator/src/lib.rs`
wraps it in a Hydro dataflow:

- **Result identity `(group, epoch, attempt)` with one owner per counter.** The
  coordinator bumps `epoch` when it recycles a group. The producer bumps `attempt`
  when it retries a timed-out rollout within an epoch. A result is admitted at most
  once. Late or duplicate results are `discard`ed, and redundant in-flight attempts
  get `cancel`led, which frees a hung original's inference slot when its hedge wins.
- **Staleness against the trainer's version, evaluated whenever it can change.**
  That means when a result arrives and when a new version arrives, not only at `get()`.
  The version is monotonic, so reordered version announcements can't move it backwards.
- **Optional oldest-first admission** (by oldest weight version) instead of FIFO,
  so groups closest to going stale are trained first. Off by default, to match Miles.
- **`starved` alerts** when a group has been recycled `N` times. This makes the
  long-rollout bias visible instead of silent.
- **Trainer restarts fenced by a run id**, so a coordinator can outlive the trainer
  (see finding 4).
- The one place arrival order is nondeterministic is a single
  `assume_ordering::<TotalOrder>(nondet!(...))` in `lib.rs`. The Hydro simulator
  explores it.

Safety properties (`check_invariants`, mirrored in Python for live runs):
1. no group is trained twice;
2. every trained group is within `max_staleness` of its batch's version;
3. batch runs never go backwards, and within a run neither do versions;
4. nothing from before a group's latest recycle is trained;
5. a result the coordinator told a client to discard is never trained (and vice versa);
6. a trained group is never recycled;
7. every trained result was generated in the batch's run.

## What the testing found

1. **A real bug in my first design, found by the randomized test.** My first design
   used a single `attempt` counter, advanced by both the producer (timeout retries)
   and the coordinator (recycles). A recycle could re-issue an attempt number the
   producer had already used for a retry, so a result that had been discarded was
   later trained on. The fix is the epoch/attempt split above; the regression test is
   `regression_recycle_does_not_reuse_retry_id`. I'm mentioning it because it's
   exactly the class of bug you asked about: two components independently minting
   ids, and it only shows up under one interleaving.
2. **Exhaustive simulation.** Take the scenario "a slow rollout times out, the retry
   and the original both finish, the weights move, two trainer steps each ask for a
   group". Hydro's simulator checks all **5,040** arrival orders and the invariants
   hold in every one. The same scenario against a buffer without dedup fails right
   away: the simulator finds an order where the group is trained twice
   (`sim_naive_buffer_double_trains`).
3. **A second adapter bug, found by the Miles integration test.** Reading the ids
   from prompt metadata at `put()` time mislabels a late stale result with the new
   epoch, because Miles reuses the same `Sample` objects across a recycle. IDs now
   have to be fixed at dispatch (`HydroFullyAsyncRolloutFn`).
4. **A livelock that the safety checks cannot see, now fixed.** A coordinator that
   had seen weight version 30, talking to a trainer that restarted at version 0
   (after a crash, or a resume from an older checkpoint), recycled every rollout as
   stale, forever. All six invariants held throughout. It's the safety-vs-liveness
   distinction in practice. `python -m hydro_rl.restart_demo --no-run-ids`
   reproduces it: job B trains 2 steps, then makes no progress through 3,570 stale
   recycles. Those 2 steps were job A's leftover rollouts trained by job B's
   trainer, a second bug in the same protocol.

   **The fix is a run id that acts as a fencing token.** Each trainer process
   stamps every message with a run id higher than any before it, and weight versions
   are compared only within a run. The first message from a higher run resets the
   version. It also recycles every untrained group whose results might come from
   weights the restart threw away. Messages from older runs are ignored or rejected.
   Group events from a *newer* run are parked and replayed once the restart arrives:
   a restarted job's producer can finish rollouts before its trainer's announcement
   reaches the coordinator. With run ids, `restart_demo` trains all 15 steps in
   both jobs.

   Testing found two more bugs in my own first version of the fix:
   - **The randomized test (now with trainer restarts) found a contradictory
     decision.** A result already trained in the old run, delivered again after
     the restart, was told to be discarded. That breaks the rule that a result is
     never both trained and discarded.
   - **The Miles adapter test hung.** It turned out that discarding early
     new-run results wastes the first rollouts of every restart; parking replaced
     the discard.

   The simulator also checks the liveness side for a small scenario: in all 720
   orders of a restart racing old-run messages, nothing from the old run is
   trained, and the new trainer always gets its batch
   (`sim_restart_races_old_run_messages`).
5. **Length bias, measured.** Mock workload: heavy-tailed response lengths, 5%
   stragglers (10× slower), 3% hangs. `python -m hydro_rl.compare`, 4 seeds, 30
   steps, `max_staleness=1`:

   | seed | mode | wall (s) | mean tokens trained / pool | long (top-20%) share trained | stale recycles |
   |---|---|---|---|---|---|
   | 0 | Miles-like | 14.4 | 324 / 388 | 0.179 | 145 |
   | 0 | Hydro + hedging | **8.5** | 249 / 388 | 0.142 | 405 |
   | 1 | Miles-like | 10.7 | 268 / 358 | 0.183 | 271 |
   | 1 | Hydro + hedging | **8.3** | 233 / 358 | 0.146 | 556 |
   | 2 | Miles-like | 10.8 | 277 / 374 | 0.171 | 245 |
   | 2 | Hydro + hedging | **8.5** | 231 / 374 | 0.133 | 400 |
   | 3 | Miles-like | 11.0 | 271 / 348 | 0.188 | 247 |
   | 3 | Hydro + hedging | **8.5** | 241 / 348 | 0.163 | 390 |

   Hedged retries are safe with the coordinator, and they recover hung slots: 30
   steps finish 1.3–1.7× faster, with all invariants holding on every trace.
   **Both** setups under-train long rollouts (trained mean is well below the pool
   mean), and the faster run is *more* biased: when the trainer steps faster, more
   long rollouts cross the staleness bound. The "Miles-like" mode is my in-process
   model of `DefaultDataBuffer` semantics, not Miles itself, and it emits no
   `starved` alerts, so read its zero as "not measured".

   Next experiment this suggests: give recycled/starved groups a staleness
   allowance, or continue partial rollouts across a version change (Miles already
   tracks per-token version spans, which is what makes that possible), and compare
   bias against throughput.

## Parity with Miles' `DefaultDataBuffer`

`HydroDataBuffer` is meant as a drop-in replacement. With `serve` started on its
defaults (`serve -- PORT none 0 false`), it behaves like Miles' buffer:

| Behavior | Miles `DefaultDataBuffer` | `HydroDataBuffer` |
|---|---|---|
| Filters on put | aborted → unused handler; missing reward / dynamic filter → dropped | same; a dropped group is also closed on the coordinator, so it is never regenerated |
| Staleness | `--max-weight-staleness` on the oldest versioned token, default None (off); unversioned groups never stale | same |
| Order | FIFO | FIFO (`oldest_first` is opt-in) |
| Group-size check | none | none (opt-in `group_size`) |
| Backpressure | `put` blocks at capacity-factor × rollout batch size | same, counting results held |
| Checkpoints | `state_dict` / `load_state_dict`, `regenerated_on_restore` → `RESTORED` | same; restored results get fresh ids in the new run |
| Metrics, audit | `rollout/fully_async/*`, dynamic-filter metrics, dropped-sample records, ownership logs | same keys and calls, plus `rollout/hydro/*` |
| No-progress warning | every 30 s in `get` | same |

The coordinator's settings live in the `serve` process. At connect the buffer
asks for them (`hello`) and refuses to start if they disagree with
`--max-weight-staleness` or `--n-samples-per-prompt`; it warns if the
coordinator serves oldest-first.

Remaining differences:
- An aborted result goes to the unused handler only when the coordinator agrees
  that no newer attempt of the group is still running. Without retries that's
  every time, as in Miles.
- No multi-policy runs (`--custom-async-data-buffer-path-per-model`).
- Retried rollouts (next section) are on by default; `HYDRO_RETRY=0` restores
  Miles' producer behavior.
- The mock experiments in finding 5 run the coordinator oldest-first; the
  "Miles-like" baseline there is FIFO.

`tests/test_miles_buffer.py` checks each row against a live coordinator, using
stand-ins for the Miles functions involved: dedup and staleness, unversioned
groups, filtered drops, backpressure, a checkpoint round trip, a restart at a
lower version, and a refused config mismatch.

## Retried rollouts in Miles

Miles' producer has no per-rollout deadline, so a rollout stuck on a slow or hung
engine holds its slot until it returns. Once enough slots are stuck, training
stops. `hydro_rl/miles_rollout.py` adds retries to Miles' own producer loop:
- **When:** a group whose newest attempt runs past about 2× the p90 latency of
  recent rollouts gets a second attempt. The original keeps running, and
  whichever finishes first is trained. The coordinator discards the other
  result and sends `cancel`, which stops the losing task.
- **Separate copies:** every attempt generates on its own copy of the prompt
  group. Miles fills in `Sample`s in place, so two attempts must not share them;
  the original prompt group stays clean for recycling.
- **Routing:** a retry drops the routing key, so a router that routes by key can
  send it to a different engine than the stuck one.
- **Budget:** retries run on their own small budget of extra groups (default 10%
  of the in-flight groups). They can start even when hung rollouts hold every
  regular slot, which is the case they exist for.
- **Checkpoints:** each running group is saved once, never one whose result the
  buffer already holds, and without Hydro ids.

**How it's enabled:** with `--fully-async`, Miles replaces any
`--rollout-function-path` with its own `FullyAsyncRolloutFn`. That means the
launch command I gave earlier, which passed both flags, never loaded the Hydro
subclass. Instead, `HydroDataBuffer` (loaded via `--custom-async-data-buffer-path`)
upgrades the `FullyAsyncRolloutFn` instance that builds it, so plain
`--fully-async` works. `HYDRO_RETRY=0` turns retries off; the `HYDRO_RETRY_*`
variables tune the deadline, the budget and the maximum number of attempts.

`tests/test_miles_retries.py` runs Miles' producer loop (using stand-ins for the
engines) with 4 groups in flight, against a live coordinator. Every 4th prompt's
first request hangs forever, and one slow prompt finishes before its retry:
- **Retries off (Miles' behavior):** training stalls after 3 of 6 steps, with
  every slot held by a hung rollout.
- **Retries on:** all 6 steps run, and each prompt is trained exactly once. The
  stuck prompts are trained via their retries and the hung originals are
  cancelled. The slow prompt's original wins, and its retry is cancelled.

Limits:
- A wall-clock deadline can't tell a stuck rollout from a legitimately long one,
  so long rollouts get retried too (bounded by the budget and the attempt cap).
  Per-token progress from SGLang would be a better signal.
- I haven't verified that cancelling the losing task also stops its request
  inside SGLang (that depends on SGLang aborting on client disconnect).

## Layout

```
coordinator/                  Rust crate (Hydro pinned to third_party/hydro)
  src/protocol.rs             wire types (JSON)
  src/state.rs                state machine + invariants + randomized closed-loop test
  src/lib.rs                  Hydro dataflow + exhaustive simulation tests
  examples/serve.rs           deploys it on localhost:PORT for Python clients
python/hydro_rl/
  client.py                   asyncio JSON-lines client
  miles_buffer.py             HydroDataBuffer: Miles DataBuffer plugin
  miles_rollout.py            retried rollouts in Miles' producer (HydroFullyAsyncRolloutFn)
  mock_run.py                 mock workers/trainer with stragglers + hangs
  compare.py                  baseline vs hydro, fresh coordinator per run
  restart_demo.py             two jobs on one coordinator: the livelock, and the fix
  invariants.py               Python mirror of check_invariants
python/tests/miles_fakes.py         stand-ins for the Miles modules the adapter imports
python/tests/test_miles_buffer.py   buffer parity with Miles, against a live coordinator
python/tests/test_miles_retries.py  retried rollouts in Miles' producer, against a live coordinator
third_party/{miles,hydro}     git submodules, pinned to the commits this was built against (not modified)
```

## Run it

Clone with the submodules (the coordinator builds Hydro from `third_party/hydro`):

```bash
git clone --recurse-submodules https://github.com/jwong8314/hydro-rl.git
```

Hydro needs Rust 1.98.1 (installed via rustup; the crate has a `rust-toolchain.toml`).

```bash
cd coordinator && cargo +1.98.1-aarch64-apple-darwin test --lib -- --nocapture
```

Start a coordinator for Miles. The arguments are port, staleness limit (`none` or
N, matching `--max-weight-staleness`), group-size check (0 = off) and oldest-first:

```bash
cd coordinator && cargo +1.98.1-aarch64-apple-darwin run --example serve -- 7878 none 0 false
```

```bash
cd python && python3 -m hydro_rl.compare --seeds 0 1 2 3
```

```bash
cd python && python3 tests/test_miles_buffer.py
```

```bash
cd python && python3 tests/test_miles_retries.py
```

Use it in Miles (the coordinator must already be running):

```bash
HYDRO_COORDINATOR=127.0.0.1:7878 python train_async.py --fully-async --custom-async-data-buffer-path hydro_rl.miles_buffer.HydroDataBuffer
```

```bash
cd python && python3 -m hydro_rl.restart_demo --no-run-ids
```

```bash
cd python && python3 -m hydro_rl.restart_demo
```

## Not done / caveats

- **Not run inside a real Miles job** (no GPUs here, and Miles doesn't import
  without its CUDA stack). The adapter is written against the `DataBuffer` contract
  and tested with stand-ins for the Miles functions it calls.
- Trainer restarts and Miles checkpoints are handled; **coordinator** restarts are
  not: it has no checkpointing of its own, so a coordinator crash loses its state.
- The run id defaults to wall-clock nanoseconds at process start (or `HYDRO_RUN`).
  That's fine on one machine. Across machines with clock skew, persist a counter
  with the checkpoint instead.
- A restart assumes inference already serves the restored weights when the
  trainer's first `get()` announces the new run (Miles syncs weights at startup).
  If an engine kept serving thrown-away weights after that, its results would
  carry the new run id and pass.
- After a whole-job restart, the old job's untrained groups are recycled once, to
  its closed connection (the server drops the messages).
- The coordinator keeps a small record for every group it has seen (trained,
  dropped or orphaned), so its memory grows with the number of groups in a run.
- The coordinator is a single Hydro `Process`. Replicating or partitioning it (a
  Hydro `Cluster`, keyed by group) is the obvious next Hydro step, and the place
  where the simulator's cluster support would matter.
