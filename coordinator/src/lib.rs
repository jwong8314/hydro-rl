//! Hydro control plane for async RL rollouts (prototype).
//!
//! The coordinator owns the group-level decisions that Miles makes inside
//! `DefaultDataBuffer` / `FullyAsyncRolloutFn` (admission, staleness, recycling)
//! and adds what an in-process asyncio buffer does not need but a multi-process
//! deployment does: attempt ids, so timed-out-then-retried rollouts and
//! duplicate deliveries are admitted at most once.
//!
//! Everything nondeterministic about the distributed setting is the arrival
//! order of events from different clients. That is made explicit at exactly
//! one place, the `assume_ordering` below, and the Hydro simulator explores it.

#[cfg(stageleft_runtime)]
hydro_lang::setup!();

pub mod protocol;
pub mod state;

use hydro_lang::live_collections::stream::{NoOrder, TotalOrder};
use hydro_lang::prelude::*;

use crate::protocol::{Event, Output};

pub struct Coord;

/// The coordinator dataflow.
///
/// `events` carries `(client_id, event)` from every connected client, with no
/// ordering guarantee across clients. Returns `(recipient_client_id, decision)`.
///
/// The `Config` fields are passed individually because they are spliced into
/// the generated program as constants.
#[allow(clippy::redundant_field_names, reason = "stageleft does not see shorthand captures")]
pub fn coordinator<'a>(
    events: Stream<(u64, Event), Process<'a, Coord>, Unbounded, NoOrder>,
    max_staleness: u64,
    group_size: u32,
    starvation_threshold: u32,
    oldest_first: bool,
    dedup: bool,
) -> Stream<(u64, Output), Process<'a, Coord>, Unbounded> {
    events
        .assume_ordering::<TotalOrder>(nondet!(
            /// Events from the producer, the inference workers and the trainer
            /// race each other over the network. The coordinator processes them
            /// in whatever order they arrive; its safety properties
            /// (`state::check_invariants`) must hold for every such order, which
            /// is what the simulation tests below explore.
        ))
        .scan(
            // Field shorthand is not seen as a capture by stageleft, so the
            // config values are spelled out.
            q!(move || crate::state::Coordinator::new(crate::protocol::Config {
                max_staleness: max_staleness,
                group_size: group_size,
                starvation_threshold: starvation_threshold,
                oldest_first: oldest_first,
                dedup: dedup,
            })),
            q!(|st, (client, ev)| Some(st.step(client, ev))),
        )
        .flat_map_ordered(q!(|outs| outs))
}

/// JSON-lines adapter for external (Python) clients: parses each line into an
/// `Event`, runs the coordinator, and serializes decisions back to the client
/// they are addressed to. Malformed lines are answered with an error object.
pub fn coordinator_json<'a>(
    lines: KeyedStream<u64, String, Process<'a, Coord>, Unbounded>,
    max_staleness: u64,
    group_size: u32,
    starvation_threshold: u32,
    oldest_first: bool,
) -> KeyedStream<u64, String, Process<'a, Coord>, Unbounded, NoOrder> {
    let parsed = lines.entries().map(q!(|(client, line)| (
        client,
        serde_json::from_str::<crate::protocol::Event>(&line).map_err(|e| e.to_string())
    )));

    let errors = parsed.clone().filter_map(q!(|(client, r)| r.err().map(|e| (
        client,
        serde_json::json!({ "type": "error", "message": e }).to_string()
    ))));

    let events = parsed.filter_map(q!(|(client, r)| r.ok().map(|ev| (client, ev))));

    coordinator(
        events,
        max_staleness,
        group_size,
        starvation_threshold,
        oldest_first,
        true,
    )
    .map(q!(|(client, out)| (client, serde_json::to_string(&out).unwrap())))
    .weaken_ordering::<NoOrder>()
    .merge_unordered(errors)
    .into_keyed()
}

#[cfg(test)]
mod tests {
    use hydro_lang::live_collections::stream::{ExactlyOnce, NoOrder};
    use hydro_lang::prelude::*;

    use crate::protocol::{Config, Event, Output};
    use crate::state::check_invariants;

    fn complete(group: u64, epoch: u32, attempt: u32, v: u64) -> Event {
        complete_in(group, epoch, attempt, v, 0)
    }

    fn complete_in(group: u64, epoch: u32, attempt: u32, v: u64, run: u64) -> Event {
        Event::Complete {
            group,
            epoch,
            attempt,
            oldest_version: Some(v),
            newest_version: Some(v),
            n_samples: 1,
            tokens: 10,
            run,
        }
    }

    /// Runs `events` through the coordinator under *every* arrival order the
    /// simulator can produce, checking the safety invariants on each trace.
    /// Returns the number of executions explored and how many of them served
    /// at least one batch.
    fn explore(cfg: Config, events: Vec<(u64, Event)>) -> (usize, usize) {
        let served = std::sync::atomic::AtomicUsize::new(0);
        let mut flow = FlowBuilder::new();
        let process = flow.process::<super::Coord>();
        let (in_send, input) = process.sim_input::<(u64, Event), NoOrder, ExactlyOnce>();
        let out = super::coordinator(
            input,
            cfg.max_staleness,
            cfg.group_size,
            cfg.starvation_threshold,
            cfg.oldest_first,
            cfg.dedup,
        )
        .sim_output();

        let n = flow.sim().exhaustive(async || {
            in_send.send_many_unordered(events.clone());
            let outputs: Vec<(u64, Output)> = out.collect().await;
            let outputs: Vec<Output> = outputs.into_iter().map(|(_, o)| o).collect();
            if let Err(e) = check_invariants(&outputs, &cfg) {
                panic!("invariant violated: {e}\noutputs: {outputs:#?}");
            }
            let batches = outputs
                .iter()
                .filter(|o| matches!(o, Output::Batch { .. }))
                .count();
            assert!(batches <= 2);
            if batches > 0 {
                served.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
        (n, served.into_inner())
    }

    /// A slow rollout times out and is retried while the original is still
    /// running. Both finish. Two trainer steps ask for one group each, and the
    /// weights move forward meanwhile. Explores every arrival order.
    fn timeout_retry_scenario() -> Vec<(u64, Event)> {
        vec![
            (0, Event::Dispatch { group: 1, epoch: 0, attempt: 0, run: 0 }),
            (0, Event::Dispatch { group: 1, epoch: 0, attempt: 1, run: 0 }),
            (1, complete(1, 0, 0, 0)),
            (2, complete(1, 0, 1, 1)),
            (9, Event::Version { version: 1, run: 0 }),
            (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
            (9, Event::Get { request: 2, n_groups: 1, run: 0 }),
        ]
    }

    #[test]
    fn sim_timeout_retry_is_trained_at_most_once() {
        let (n, _) = explore(Config::default(), timeout_retry_scenario());
        println!("explored {n} executions");
    }

    /// Same scenario against a buffer that trusts every completion (no attempt
    /// ids, no dedup). The simulator finds an interleaving where the group is
    /// trained on twice.
    #[test]
    #[should_panic(expected = "trained on twice")]
    fn sim_naive_buffer_double_trains() {
        explore(
            Config {
                dedup: false,
                ..Config::default()
            },
            timeout_retry_scenario(),
        );
    }

    /// A group goes stale and is recycled; the old result is then delivered a
    /// second time (e.g. a client retrying a send), racing the regenerated one
    /// and a weight update.
    #[test]
    fn sim_stale_recycle_with_duplicate_delivery() {
        let (n, _) = explore(
            Config::default(),
            vec![
                (0, complete(2, 0, 0, 0)),
                (0, complete(2, 0, 0, 0)),
                (9, Event::Version { version: 2, run: 0 }),
                (0, complete(2, 1, 0, 2)),
                (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
            ],
        );
        println!("explored {n} executions");
    }

    /// A trainer at version 30 restarts and resumes at version 0 (a new run).
    /// Racing the restart: a result from the old run for a group the
    /// coordinator hasn't seen, a zombie version announcement from the old
    /// trainer, and the new run's first result and batch request. In every
    /// order nothing from the old run is trained, and the new trainer gets its
    /// batch: the livelock is gone whichever message wins the race.
    #[test]
    fn sim_restart_races_old_run_messages() {
        let events = vec![
            (9, Event::Version { version: 30, run: 0 }),
            (0, complete_in(1, 0, 0, 30, 0)),
            (9, Event::Version { version: 31, run: 0 }),
            (9, Event::Version { version: 0, run: 1 }),
            (0, complete_in(2, 0, 0, 0, 1)),
            (9, Event::Get { request: 1, n_groups: 1, run: 1 }),
        ];
        let (n, served) = explore(Config::default(), events);
        println!("explored {n} executions, {served} served the new run's batch");
        assert_eq!(served, n, "some order left the new run without a batch");
    }
}
