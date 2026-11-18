//! The coordinator as a pure, single-threaded state machine.
//!
//! The Hydro dataflow in `lib.rs` feeds it one event at a time, after making
//! the cross-client arrival order explicit with `assume_ordering` + `nondet!`.
//! Keeping the logic here as plain Rust means the same code runs in the Hydro
//! simulator, in deployment, and in the plain unit tests at the bottom of this
//! file.
//!
//! Result identity is `(group, epoch, attempt)`. The two halves have different
//! owners so that no id is ever issued twice:
//! - `epoch` is advanced only by the coordinator, when it recycles a group.
//! - `attempt` is advanced only by the producer, when it retries a timed-out
//!   rollout within an epoch.
//!
//! (An earlier version had a single `attempt` counter advanced by both sides;
//! the randomized test below found an interleaving where a recycle reused an
//! attempt number the producer had already spent on a retry, so a result that
//! had been discarded was later trained on.)
//!
//! Trainer restarts are fenced by a `run` id. Weight versions are compared only
//! within a run. Without that, a coordinator that had seen version 30 treated
//! every rollout from a trainer restarted at version 0 as stale, forever: a
//! livelock that broke none of the safety properties. A higher run resets the
//! version and recycles every untrained group whose results may come from
//! weights the restart threw away. Messages from older runs are ignored or
//! rejected. Group events from a *newer* run (its producer started before its
//! trainer's announcement reached us) are parked and replayed at the restart.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::protocol::{
    Attempt, BatchEntry, Config, DiscardReason, Epoch, Event, GroupId, Output, RecycleReason, Run,
    Version,
};

pub type ClientId = u64;

fn event_run(ev: &Event) -> Run {
    match ev {
        Event::Dispatch { run, .. }
        | Event::Complete { run, .. }
        | Event::Abort { run, .. }
        | Event::Drop { run, .. }
        | Event::Version { run, .. }
        | Event::Get { run, .. } => *run,
        Event::Hello => 0,
    }
}

/// Only called on group events (dispatch / complete / abort / drop).
fn event_group(ev: &Event) -> GroupId {
    match ev {
        Event::Dispatch { group, .. }
        | Event::Complete { group, .. }
        | Event::Abort { group, .. }
        | Event::Drop { group, .. } => *group,
        _ => unreachable!("not a group event"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    InFlight,
    Buffered,
    Consumed,
    /// Closed without training (`Event::Drop`). Final, like `Consumed`.
    Dropped,
}

#[derive(Clone, Debug)]
struct GroupState {
    /// The client that should hear about recycles / discards for this group.
    owner: ClientId,
    /// Results from earlier epochs are invalid.
    epoch: Epoch,
    /// Highest retry the coordinator knows was started in `epoch`.
    latest_attempt: Attempt,
    /// The result that is buffered or consumed, if any.
    admitted: Option<(Epoch, Attempt)>,
    status: Status,
    recycles: u32,
    /// Latest trainer run in which a dispatch, result, abort or drop for this
    /// group was processed. A restart only needs to recycle groups active in
    /// the run it replaces: older ones were recycled by an earlier restart.
    last_run: Run,
}

impl GroupState {
    fn new(owner: ClientId) -> Self {
        GroupState {
            owner,
            epoch: 0,
            latest_attempt: 0,
            admitted: None,
            status: Status::InFlight,
            recycles: 0,
            last_run: 0,
        }
    }
}

#[derive(Clone, Debug)]
struct Buffered {
    entry: BatchEntry,
    seq: u64,
}

#[derive(Clone, Debug)]
pub struct Coordinator {
    cfg: Config,
    run: Run,
    current_version: Version,
    groups: HashMap<GroupId, GroupState>,
    buffer: Vec<Buffered>,
    /// (client, request, n_groups, run)
    pending_gets: VecDeque<(ClientId, u64, u32, Run)>,
    /// Group events from a run newer than ours: the producer of a restarted
    /// job can start before its trainer's first `Version` reaches us. Held
    /// until that restart, then replayed, so their work isn't wasted.
    parked: Vec<(ClientId, Event)>,
    next_seq: u64,
}

impl Coordinator {
    pub fn new(cfg: Config) -> Self {
        Coordinator {
            cfg,
            run: 0,
            current_version: 0,
            groups: HashMap::new(),
            buffer: Vec::new(),
            pending_gets: VecDeque::new(),
            parked: Vec::new(),
            next_seq: 0,
        }
    }

    pub fn current_version(&self) -> Version {
        self.current_version
    }

    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// Process one event from `client`, returning `(recipient, decision)` pairs.
    pub fn step(&mut self, client: ClientId, event: Event) -> Vec<(ClientId, Output)> {
        let mut out = Vec::new();
        self.handle(client, event, &mut out);
        self.evict_stale(&mut out);
        self.serve(&mut out);
        out
    }

    fn handle(&mut self, client: ClientId, event: Event, out: &mut Vec<(ClientId, Output)>) {
        let event_run = match &event {
            Event::Dispatch { run, .. }
            | Event::Complete { run, .. }
            | Event::Abort { run, .. }
            | Event::Drop { run, .. } => Some(*run),
            _ => None,
        };
        if event_run.is_some_and(|r| r > self.run) {
            self.parked.push((client, event));
            return;
        }
        match event {
            Event::Dispatch {
                group,
                epoch,
                attempt,
                run,
            } => {
                if run < self.run && self.cfg.dedup {
                    self.group(client, group);
                    out.push((client, Output::Cancel { group, epoch, attempt }));
                } else {
                    self.on_dispatch(client, group, epoch, attempt, out);
                }
            }
            Event::Complete {
                group,
                epoch,
                attempt,
                oldest_version,
                newest_version: _,
                n_samples,
                tokens,
                run,
            } => {
                if run < self.run && self.cfg.dedup {
                    // Made under an older run's weights: its versions can't be
                    // compared with ours, and those weights may have been
                    // thrown away by the restart.
                    let st = self.group(client, group);
                    // A second delivery of a result the old run already
                    // trained on gets no reply: it must never be both trained
                    // and discarded (found by the randomized test).
                    if st.admitted != Some((epoch, attempt)) {
                        out.push((
                            st.owner,
                            Output::Discard {
                                group,
                                epoch,
                                attempt,
                                reason: DiscardReason::Superseded,
                            },
                        ));
                    }
                } else {
                    let entry = BatchEntry {
                        group,
                        epoch,
                        attempt,
                        oldest_version,
                        tokens,
                        run,
                    };
                    self.on_complete(client, entry, n_samples, out);
                }
            }
            Event::Abort {
                group,
                epoch,
                attempt,
                run,
            } => {
                if run == self.run || !self.cfg.dedup {
                    self.on_abort(client, group, epoch, attempt, out);
                }
            }
            Event::Drop {
                group,
                epoch,
                attempt,
                run,
            } => {
                if run == self.run {
                    self.on_drop(client, group, epoch, attempt, out);
                }
            }
            Event::Hello => out.push((client, Output::Config { config: self.cfg })),
            Event::Version { version, run } => {
                if run > self.run {
                    self.restart(client, run, version, out);
                } else if run == self.run {
                    // Version announcements can be reordered with each other;
                    // within a run the trainer's version only moves forward.
                    self.current_version = self.current_version.max(version);
                }
                // run < self.run: a late message from a dead trainer. Ignore it.
            }
            Event::Get {
                request,
                n_groups,
                run,
            } => {
                if run >= self.run {
                    self.pending_gets.push_back((client, request, n_groups, run));
                }
            }
        }
    }

    fn on_drop(
        &mut self,
        client: ClientId,
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        out: &mut Vec<(ClientId, Output)>,
    ) {
        let st = self.active(client, group);
        if st.status != Status::InFlight || epoch != st.epoch {
            // Already decided, or a result of an epoch that was recycled
            // meanwhile: the drop changes nothing.
            return;
        }
        st.status = Status::Dropped;
        let owner = st.owner;
        for a in (0..=st.latest_attempt).filter(|&a| a != attempt) {
            out.push((owner, Output::Cancel { group, epoch, attempt: a }));
        }
    }

    fn group(&mut self, client: ClientId, group: GroupId) -> &mut GroupState {
        self.groups
            .entry(group)
            .or_insert_with(|| GroupState::new(client))
    }

    /// `group`, marked as active in the current run.
    fn active(&mut self, client: ClientId, group: GroupId) -> &mut GroupState {
        let run = self.run;
        let st = self.group(client, group);
        st.last_run = run;
        st
    }

    fn on_dispatch(
        &mut self,
        client: ClientId,
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        out: &mut Vec<(ClientId, Output)>,
    ) {
        let dedup = self.cfg.dedup;
        let st = self.active(client, group);
        st.owner = client;
        if dedup && (st.status != Status::InFlight || epoch < st.epoch) {
            // A retry raced with a result that already arrived, or with a
            // recycle: the new generation is wasted work, tell the producer.
            out.push((
                client,
                Output::Cancel {
                    group,
                    epoch,
                    attempt,
                },
            ));
            return;
        }
        if epoch == st.epoch {
            st.latest_attempt = st.latest_attempt.max(attempt);
        }
    }

    fn on_complete(
        &mut self,
        client: ClientId,
        entry: BatchEntry,
        n_samples: u32,
        out: &mut Vec<(ClientId, Output)>,
    ) {
        let cfg = self.cfg;
        let st = self.active(client, entry.group);
        let owner = st.owner;
        let (group, epoch, attempt) = (entry.group, entry.epoch, entry.attempt);
        let discard = |reason| {
            (
                owner,
                Output::Discard {
                    group,
                    epoch,
                    attempt,
                    reason,
                },
            )
        };

        if cfg.dedup {
            if st.status != Status::InFlight {
                if st.admitted != Some((epoch, attempt)) {
                    out.push(discard(DiscardReason::Duplicate));
                }
                // Otherwise: the admitted result delivered twice, same payload.
                return;
            }
            if epoch < st.epoch {
                out.push(discard(DiscardReason::Superseded));
                return;
            }
        }

        if cfg.group_size != 0 && n_samples != cfg.group_size {
            if cfg.dedup && epoch == st.epoch && attempt < st.latest_attempt {
                // A newer retry is already running; let it finish.
                out.push(discard(DiscardReason::Superseded));
                return;
            }
            self.recycle(group, RecycleReason::Partial, out);
            return;
        }

        st.status = Status::Buffered;
        st.admitted = Some((epoch, attempt));
        if cfg.dedup {
            // Hedged retries of this epoch are now wasted work (and a hung
            // original holds an inference slot): tell the owner to stop them.
            for a in (0..=st.latest_attempt).filter(|&a| a != attempt) {
                out.push((
                    owner,
                    Output::Cancel {
                        group,
                        epoch,
                        attempt: a,
                    },
                ));
            }
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.buffer.push(Buffered { entry, seq });
    }

    fn on_abort(
        &mut self,
        client: ClientId,
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        out: &mut Vec<(ClientId, Output)>,
    ) {
        let dedup = self.cfg.dedup;
        let st = self.active(client, group);
        if dedup
            && (st.status != Status::InFlight
                || epoch < st.epoch
                || (epoch == st.epoch && attempt < st.latest_attempt))
        {
            // The group already has a result, the abort is from an old epoch,
            // or a newer retry is still running: nothing changes.
            return;
        }
        self.recycle(group, RecycleReason::Aborted, out);
    }

    fn recycle(&mut self, group: GroupId, reason: RecycleReason, out: &mut Vec<(ClientId, Output)>) {
        let threshold = self.cfg.starvation_threshold;
        let st = self.groups.get_mut(&group).unwrap();
        st.recycles += 1;
        st.epoch += 1;
        st.latest_attempt = 0;
        st.status = Status::InFlight;
        st.admitted = None;
        out.push((
            st.owner,
            Output::Recycle {
                group,
                reason,
                next_epoch: st.epoch,
            },
        ));
        if st.recycles >= threshold {
            out.push((
                st.owner,
                Output::Starved {
                    group,
                    recycles: st.recycles,
                },
            ));
        }
    }

    fn restart(&mut self, client: ClientId, run: Run, version: Version, out: &mut Vec<(ClientId, Output)>) {
        let replaced = self.run;
        self.run = run;
        self.current_version = version;
        self.buffer.clear();
        self.pending_gets.retain(|g| g.3 >= run);
        let (replay, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.parked)
            .into_iter()
            .partition(|(_, ev)| event_run(ev) <= run);
        self.parked = later;
        let new_run_groups: HashSet<GroupId> = replay
            .iter()
            .filter(|(_, ev)| event_run(ev) == run)
            .map(|(_, ev)| event_group(ev))
            .collect();
        // Recycle the untrained groups of the replaced run: a buffered result
        // is from that run, so it always goes. In-flight groups the new run
        // has already dispatched or finished continue in their epoch (results
        // of theirs from an older run are still rejected by run). In-flight
        // groups last active before the replaced run were recycled by an
        // earlier restart; doing it again would only inflate their recycle
        // count (the adapter test showed false `starved` alerts).
        let mut untrained: Vec<GroupId> = self
            .groups
            .iter()
            .filter(|(g, st)| match st.status {
                Status::Consumed | Status::Dropped => false,
                Status::Buffered => true,
                Status::InFlight => st.last_run == replaced && !new_run_groups.contains(g),
            })
            .map(|(&g, _)| g)
            .collect();
        untrained.sort_unstable(); // deterministic output order
        for &g in &untrained {
            self.recycle(g, RecycleReason::Restart, out);
        }
        for (c, ev) in replay {
            self.handle(c, ev, out);
        }
        out.push((
            client,
            Output::Restarted {
                run,
                version,
                recycled: untrained.len() as u32,
            },
        ));
    }

    fn is_stale(&self, oldest_version: Option<Version>) -> bool {
        // Like Miles: a group without versioned tokens is never stale.
        let Some(oldest_version) = oldest_version else {
            return false;
        };
        // saturating: a worker may report a version the trainer's own
        // announcement has not reached us with yet.
        self.current_version.saturating_sub(oldest_version) > self.cfg.max_staleness
    }

    fn evict_stale(&mut self, out: &mut Vec<(ClientId, Output)>) {
        let (stale, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.buffer)
            .into_iter()
            .partition(|b| self.is_stale(b.entry.oldest_version));
        self.buffer = kept;
        for b in stale {
            self.recycle(b.entry.group, RecycleReason::Stale, out);
        }
    }

    fn serve(&mut self, out: &mut Vec<(ClientId, Output)>) {
        while let Some(&(client, request, n, run)) = self.pending_gets.front() {
            // A get from a run we haven't switched to yet waits for its
            // restart announcement.
            if run != self.run || self.buffer.len() < n as usize {
                break;
            }
            self.pending_gets.pop_front();
            if self.cfg.oldest_first {
                self.buffer.sort_by_key(|b| (b.entry.oldest_version, b.seq));
            } else {
                self.buffer.sort_by_key(|b| b.seq);
            }
            let groups: Vec<BatchEntry> = self
                .buffer
                .drain(..n as usize)
                .map(|b| b.entry)
                .collect();
            for e in &groups {
                if let Some(st) = self.groups.get_mut(&e.group) {
                    st.status = Status::Consumed;
                }
            }
            out.push((
                client,
                Output::Batch {
                    request,
                    run,
                    version: self.current_version,
                    groups,
                },
            ));
        }
    }
}

/// Safety properties every output trace must satisfy, whatever the arrival
/// order of events. Returns a description of the first violation.
///
/// 1. No group is trained on twice.
/// 2. Every trained group is within `max_staleness` of the batch's version.
/// 3. Batch runs never go backwards, and within a run neither do versions.
/// 4. Nothing from an epoch before a group's latest recycle is trained on.
/// 5. A result the coordinator told a client to discard is never trained on
///    (the client has already freed it), and vice versa.
/// 6. A group is never recycled after being trained on.
/// 7. Every trained result was generated in the batch's run (versions from
///    different runs can't be compared, so property 2 would be meaningless).
pub fn check_invariants(outputs: &[Output], cfg: &Config) -> Result<(), String> {
    let mut trained: HashSet<GroupId> = HashSet::new();
    let mut trained_results: HashSet<(GroupId, Epoch, Attempt)> = HashSet::new();
    let mut discarded: HashSet<(GroupId, Epoch, Attempt)> = HashSet::new();
    let mut min_epoch: HashMap<GroupId, Epoch> = HashMap::new();
    let mut last: (Run, Version) = (0, 0);

    for (i, o) in outputs.iter().enumerate() {
        match o {
            Output::Batch {
                run,
                version,
                groups,
                ..
            } => {
                if (*run, *version) < last {
                    return Err(format!(
                        "#{i}: batch (run, version) {:?} went backwards from {last:?}",
                        (run, version)
                    ));
                }
                last = (*run, *version);
                for e in groups {
                    let id = (e.group, e.epoch, e.attempt);
                    if e.run != *run {
                        return Err(format!(
                            "#{i}: result {id:?} from run {} trained in run {run}",
                            e.run
                        ));
                    }
                    if !trained.insert(e.group) {
                        return Err(format!("#{i}: group {} trained on twice", e.group));
                    }
                    let lag = e.oldest_version.map_or(0, |o| version.saturating_sub(o));
                    if lag > cfg.max_staleness {
                        return Err(format!(
                            "#{i}: group {} has staleness {lag} > {}",
                            e.group, cfg.max_staleness
                        ));
                    }
                    if let Some(&min) = min_epoch.get(&e.group)
                        && e.epoch < min
                    {
                        return Err(format!(
                            "#{i}: group {} trained on epoch {} after it was recycled to {min}",
                            e.group, e.epoch
                        ));
                    }
                    if discarded.contains(&id) {
                        return Err(format!("#{i}: result {id:?} was discarded, then trained on"));
                    }
                    trained_results.insert(id);
                }
            }
            Output::Recycle {
                group, next_epoch, ..
            } => {
                if trained.contains(group) {
                    return Err(format!("#{i}: group {group} recycled after being trained on"));
                }
                min_epoch.insert(*group, *next_epoch);
            }
            Output::Discard {
                group,
                epoch,
                attempt,
                ..
            } => {
                let id = (*group, *epoch, *attempt);
                if trained_results.contains(&id) {
                    return Err(format!("#{i}: result {id:?} discarded after being trained on"));
                }
                discarded.insert(id);
            }
            Output::Cancel { .. }
            | Output::Starved { .. }
            | Output::Restarted { .. }
            | Output::Config { .. } => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(group: GroupId, epoch: Epoch, attempt: Attempt, v: Version) -> Event {
        complete_in(group, epoch, attempt, v, 0)
    }

    fn complete_in(group: GroupId, epoch: Epoch, attempt: Attempt, v: Version, run: Run) -> Event {
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

    fn run(cfg: Config, events: &[(ClientId, Event)]) -> Vec<Output> {
        let mut c = Coordinator::new(cfg);
        events
            .iter()
            .flat_map(|(cl, e)| c.step(*cl, e.clone()))
            .map(|(_, o)| o)
            .collect()
    }

    fn batches(out: &[Output]) -> Vec<(Run, Vec<GroupId>)> {
        out.iter()
            .filter_map(|o| match o {
                Output::Batch { run, groups, .. } => {
                    Some((*run, groups.iter().map(|e| e.group).collect()))
                }
                _ => None,
            })
            .collect()
    }

    /// A trainer at version 30 restarts and resumes at version 0. These are the
    /// events from after the restart.
    fn after_restart(new_run: Run) -> Vec<(ClientId, Event)> {
        vec![
            (9, Event::Version { version: 0, run: new_run }),
            (0, complete_in(5, 0, 0, 0, new_run)),
            (9, Event::Get { request: 2, n_groups: 1, run: new_run }),
        ]
    }

    fn before_restart() -> Vec<(ClientId, Event)> {
        vec![
            (9, Event::Version { version: 30, run: 0 }),
            (0, complete(1, 0, 0, 30)),
            (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
        ]
    }

    /// The livelock from the mock runs, reproduced: if the restarted trainer
    /// can't say it is a new run, its version 0 is ignored (versions only move
    /// forward), every new rollout is stale against version 30, and the
    /// trainer never gets a batch. Nothing in `check_invariants` notices.
    #[test]
    fn livelock_without_run_ids() {
        let cfg = Config::default();
        let mut events = before_restart();
        events.extend(after_restart(0));
        let out = run(cfg, &events);
        check_invariants(&out, &cfg).unwrap();
        assert_eq!(batches(&out), vec![(0, vec![1])], "request 2 is never served");
        assert!(out.iter().any(|o| matches!(
            o,
            Output::Recycle { group: 5, reason: RecycleReason::Stale, .. }
        )));
    }

    #[test]
    fn restart_with_new_run_makes_progress() {
        let cfg = Config::default();
        let mut events = before_restart();
        events.push((0, complete(2, 0, 0, 30))); // buffered, never trained in run 0
        events.extend(after_restart(1));
        // Messages from before the restart that arrive late:
        events.push((9, Event::Version { version: 31, run: 0 })); // zombie trainer
        events.push((0, complete(3, 0, 0, 31))); // new group, old weights
        events.push((0, complete(1, 0, 0, 30))); // redelivery of a trained result
        events.push((9, Event::Get { request: 3, n_groups: 1, run: 0 }));
        // group 2 regenerated after its Restart recycle
        events.push((0, complete_in(2, 1, 0, 0, 1)));
        events.push((9, Event::Get { request: 4, n_groups: 1, run: 1 }));
        let out = run(cfg, &events);
        check_invariants(&out, &cfg).unwrap();
        assert_eq!(batches(&out), vec![(0, vec![1]), (1, vec![5]), (1, vec![2])]);
        assert!(!out.iter().any(|o| matches!(o, Output::Discard { group: 5, .. })));
        assert!(out.contains(&Output::Recycle {
            group: 2,
            reason: RecycleReason::Restart,
            next_epoch: 1
        }));
        assert!(out.contains(&Output::Discard {
            group: 3,
            epoch: 0,
            attempt: 0,
            reason: DiscardReason::Superseded
        }));
    }

    /// The new job's producer finishes a rollout before its trainer's
    /// restart announcement arrives. Parked, then admitted: no wasted work.
    /// (Discarding it instead hung the Miles adapter test.)
    #[test]
    fn early_result_of_new_run_is_parked_not_wasted() {
        let cfg = Config::default();
        let mut events = before_restart();
        events.push((0, Event::Dispatch { group: 7, epoch: 0, attempt: 0, run: 1 }));
        events.push((0, complete_in(7, 0, 0, 0, 1)));
        events.push((9, Event::Version { version: 0, run: 1 }));
        events.push((9, Event::Get { request: 2, n_groups: 1, run: 1 }));
        let out = run(cfg, &events);
        check_invariants(&out, &cfg).unwrap();
        assert_eq!(batches(&out), vec![(0, vec![1]), (1, vec![7])]);
        assert!(!out.iter().any(|o| matches!(o, Output::Recycle { group: 7, .. })));
    }

    /// A job dies with a group in flight. Its restart recycles the group once;
    /// later restarts leave it alone (the adapter test saw it recycled on every
    /// restart, raising false `starved` alerts).
    #[test]
    fn orphan_is_recycled_by_one_restart_only() {
        let cfg = Config::default();
        let out = run(
            cfg,
            &[
                (0, Event::Dispatch { group: 9, epoch: 0, attempt: 0, run: 0 }),
                (9, Event::Version { version: 0, run: 1 }),
                (9, Event::Version { version: 0, run: 2 }),
                (9, Event::Version { version: 0, run: 3 }),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
        let recycles = out.iter().filter(|o| matches!(o, Output::Recycle { group: 9, .. })).count();
        assert_eq!(recycles, 1);
    }

    /// Miles drops a group whose reward is missing or that its dynamic filter
    /// rejects, and does not regenerate it. `Drop` closes the group: a hedged
    /// retry is cancelled, its late result discarded, and a trainer restart
    /// does not bring the group back.
    #[test]
    fn dropped_group_stays_closed() {
        let cfg = Config::default();
        let out = run(
            cfg,
            &[
                (0, Event::Dispatch { group: 1, epoch: 0, attempt: 0, run: 0 }),
                (0, Event::Dispatch { group: 1, epoch: 0, attempt: 1, run: 0 }),
                (0, Event::Drop { group: 1, epoch: 0, attempt: 0, run: 0 }),
                (0, complete(1, 0, 1, 0)),
                (9, Event::Version { version: 0, run: 1 }),
                (9, Event::Get { request: 1, n_groups: 1, run: 1 }),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
        assert!(out.contains(&Output::Cancel { group: 1, epoch: 0, attempt: 1 }));
        assert!(out.contains(&Output::Discard {
            group: 1,
            epoch: 0,
            attempt: 1,
            reason: DiscardReason::Duplicate
        }));
        assert!(!out.iter().any(|o| matches!(o, Output::Recycle { .. })));
        assert!(batches(&out).is_empty());
    }

    /// Miles' defaults: no staleness limit (`--max-weight-staleness None`), no
    /// group-size check, FIFO, and groups without versioned tokens are never
    /// stale even when a limit is set.
    #[test]
    fn miles_defaults() {
        let cfg = Config {
            max_staleness: crate::protocol::NO_STALENESS_LIMIT,
            group_size: 0,
            oldest_first: false,
            ..Config::default()
        };
        let unversioned = Event::Complete {
            group: 3,
            epoch: 0,
            attempt: 0,
            oldest_version: None,
            newest_version: None,
            n_samples: 4,
            tokens: 10,
            run: 0,
        };
        let out = run(
            cfg,
            &[
                (0, complete(2, 0, 0, 0)), // 1 sample: a size check would recycle it
                (0, unversioned.clone()),
                (0, complete(1, 0, 0, 50)),
                (9, Event::Version { version: 1000, run: 0 }),
                (9, Event::Get { request: 1, n_groups: 3, run: 0 }),
                (9, Event::Hello),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
        assert_eq!(batches(&out), vec![(0, vec![2, 3, 1])], "FIFO, nothing stale");
        assert!(out.contains(&Output::Config { config: cfg }));

        // With a limit, the versioned groups go stale but the unversioned one doesn't.
        let cfg = Config { max_staleness: 1, ..cfg };
        let out = run(
            cfg,
            &[
                (0, complete(2, 0, 0, 0)),
                (0, unversioned),
                (9, Event::Version { version: 1000, run: 0 }),
                (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
        assert_eq!(batches(&out), vec![(0, vec![3])]);
    }

    /// The hole that stamping runs only on the trainer's messages leaves
    /// open: a result made under the old run's weights, for a group the
    /// coordinator hears about only after the restart, carries an old-run
    /// version that looks fresh. Here the checker is fed such a trace
    /// directly to show property 7 catches it.
    #[test]
    fn invariant_rejects_cross_run_training() {
        let cfg = Config::default();
        let out = vec![Output::Batch {
            request: 1,
            run: 1,
            version: 0,
            groups: vec![BatchEntry { group: 3, epoch: 0, attempt: 0, oldest_version: Some(0), tokens: 1, run: 0 }],
        }];
        assert!(check_invariants(&out, &cfg).unwrap_err().contains("from run 0 trained in run 1"));
    }

    #[test]
    fn retry_race_admits_once() {
        let cfg = Config::default();
        let out = run(
            cfg,
            &[
                (0, Event::Dispatch { group: 1, epoch: 0, attempt: 0, run: 0 }),
                (0, Event::Dispatch { group: 1, epoch: 0, attempt: 1, run: 0 }),
                (1, complete(1, 0, 0, 0)),
                (2, complete(1, 0, 1, 0)),
                (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
        assert!(out.contains(&Output::Discard {
            group: 1,
            epoch: 0,
            attempt: 1,
            reason: DiscardReason::Duplicate
        }));
    }

    #[test]
    fn stale_group_is_recycled_and_old_result_superseded() {
        let cfg = Config::default();
        let out = run(
            cfg,
            &[
                (0, Event::Dispatch { group: 2, epoch: 0, attempt: 0, run: 0 }),
                (0, complete(2, 0, 0, 0)),
                (9, Event::Version { version: 2, run: 0 }),
                (0, complete(2, 0, 0, 0)), // duplicate delivery after the recycle
                (0, Event::Dispatch { group: 2, epoch: 1, attempt: 0, run: 0 }),
                (0, complete(2, 1, 0, 2)),
                (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
        assert!(matches!(out.last(), Some(Output::Batch { groups, .. }) if groups[0].epoch == 1));
    }

    /// The interleaving the randomized test found against the single-counter
    /// design: attempt 1 (a producer retry) is discarded as a duplicate, the
    /// admitted attempt 0 then goes stale and is recycled before the
    /// coordinator has heard about the retry's dispatch, and a second delivery
    /// of attempt 1 arrives. With one shared counter the recycle handed out
    /// "attempt 1" again and the discarded result was admitted.
    #[test]
    fn regression_recycle_does_not_reuse_retry_id() {
        let cfg = Config::default();
        let out = run(
            cfg,
            &[
                (1, complete(1, 0, 0, 0)),
                (2, complete(1, 0, 1, 0)), // discarded: duplicate
                (9, Event::Version { version: 2, run: 0 }), // attempt 0 goes stale: recycle
                (0, Event::Dispatch { group: 1, epoch: 0, attempt: 1, run: 0 }), // late
                (2, complete(1, 0, 1, 0)), // redelivered
                (9, Event::Get { request: 1, n_groups: 1, run: 0 }),
            ],
        );
        check_invariants(&out, &cfg).unwrap();
    }

    /// Tiny deterministic PRNG so the test needs no extra crates.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Closed-loop randomized model of the whole system. A producer dispatches
    /// groups, retries some of them on "timeout" (the original keeps running),
    /// and reacts to `Recycle` / `Cancel` only once those decisions reach it.
    /// Workers finish rollouts at whatever weight version is current, sometimes
    /// partially, sometimes abort, and sometimes deliver a result twice. The
    /// trainer bumps versions and asks for batches. Every message sits in a
    /// "network" and is delivered in random order. Cheap breadth to complement
    /// the exhaustive Hydro simulation of small scenarios in `lib.rs`.
    #[test]
    fn random_closed_loop_holds_invariants() {
        const N_GROUPS: usize = 4;
        let cfg = Config {
            max_staleness: 1,
            group_size: 2,
            starvation_threshold: 3,
            oldest_first: true,
            dedup: true,
        };
        let mut rng = Lcg(7);
        // trained groups, recycles, discards, groups trained after a restart
        let mut stats = [0usize; 4];
        for _ in 0..5_000 {
            let mut coord = Coordinator::new(cfg);
            let mut to_coord: Vec<(ClientId, Event)> = Vec::new();
            let mut to_producer: Vec<Output> = Vec::new();
            let mut outputs: Vec<Output> = Vec::new();
            let mut version: Version = 0;
            let mut cur_run: Run = 0;
            let mut next_request = 0u64;
            let mut epoch_known = [0 as Epoch; N_GROUPS];
            let mut next_attempt = [0 as Attempt; N_GROUPS];
            let mut running: Vec<(GroupId, Epoch, Attempt, Run)> = Vec::new();

            let dispatch = |g: usize,
                            run: Run,
                            epoch_known: &[Epoch; N_GROUPS],
                            next_attempt: &mut [Attempt; N_GROUPS],
                            running: &mut Vec<(GroupId, Epoch, Attempt, Run)>,
                            to_coord: &mut Vec<(ClientId, Event)>| {
                let (e, a) = (epoch_known[g], next_attempt[g]);
                next_attempt[g] += 1;
                running.push((g as GroupId, e, a, run));
                to_coord.push((0, Event::Dispatch { group: g as GroupId, epoch: e, attempt: a, run }));
            };

            for g in 0..N_GROUPS {
                dispatch(g, 0, &epoch_known, &mut next_attempt, &mut running, &mut to_coord);
            }

            for step in 0..200 {
                let draining = step >= 80;
                let choice = if draining { 0 } else { rng.next() % 10 };
                match choice {
                    0..=2 if !to_coord.is_empty() => {
                        let i = (rng.next() % to_coord.len() as u64) as usize;
                        let (client, ev) = to_coord.swap_remove(i);
                        for (_, o) in coord.step(client, ev) {
                            outputs.push(o.clone());
                            to_producer.push(o);
                        }
                    }
                    3 if !to_producer.is_empty() => {
                        let i = (rng.next() % to_producer.len() as u64) as usize;
                        match to_producer.swap_remove(i) {
                            Output::Recycle { group, next_epoch, .. } => {
                                let g = group as usize;
                                if next_epoch > epoch_known[g] {
                                    epoch_known[g] = next_epoch;
                                    next_attempt[g] = 0;
                                    dispatch(g, cur_run, &epoch_known, &mut next_attempt, &mut running, &mut to_coord);
                                }
                            }
                            Output::Cancel { group, epoch, attempt } => {
                                running.retain(|r| (r.0, r.1, r.2) != (group, epoch, attempt));
                            }
                            _ => {}
                        }
                    }
                    4 | 5 if !running.is_empty() => {
                        let i = (rng.next() % running.len() as u64) as usize;
                        let (group, epoch, attempt, run) = running.swap_remove(i);
                        if rng.next().is_multiple_of(8) {
                            to_coord.push((1, Event::Abort { group, epoch, attempt, run }));
                        } else if rng.next().is_multiple_of(16) {
                            // the buffer's own filter rejected the result
                            to_coord.push((1, Event::Drop { group, epoch, attempt, run }));
                        } else {
                            let oldest = version.saturating_sub(rng.next() % 3);
                            let ev = Event::Complete {
                                group,
                                epoch,
                                attempt,
                                oldest_version: Some(oldest),
                                newest_version: Some(version),
                                n_samples: if rng.next().is_multiple_of(8) { 1 } else { 2 },
                                tokens: 1 + rng.next() % 100,
                                run,
                            };
                            if rng.next().is_multiple_of(4) {
                                to_coord.push((1, ev.clone()));
                            }
                            to_coord.push((1, ev));
                        }
                    }
                    6 if !running.is_empty() => {
                        // timeout: retry a running generation of the current epoch
                        let i = (rng.next() % running.len() as u64) as usize;
                        let (g, e, _, _) = running[i];
                        if e == epoch_known[g as usize] {
                            dispatch(g as usize, cur_run, &epoch_known, &mut next_attempt, &mut running, &mut to_coord);
                        }
                    }
                    7 => {
                        version += 1;
                        to_coord.push((9, Event::Version { version, run: cur_run }));
                    }
                    8 => {
                        to_coord.push((9, Event::Get { request: next_request, n_groups: 1, run: cur_run }));
                        next_request += 1;
                    }
                    9 if !draining && rng.next().is_multiple_of(4) => {
                        // The trainer restarts from an earlier checkpoint.
                        // Inference reloads those weights before the
                        // announcement is sent; messages of the old run are
                        // still in the network and arrive whenever.
                        cur_run += 1;
                        version = rng.next() % (version + 1);
                        to_coord.push((9, Event::Version { version, run: cur_run }));
                    }
                    _ => {
                        if draining && to_coord.is_empty() {
                            break;
                        }
                    }
                }
            }

            if let Err(e) = check_invariants(&outputs, &cfg) {
                panic!("{e}\noutputs: {outputs:#?}");
            }
            for o in &outputs {
                match o {
                    Output::Batch { groups, run, .. } => {
                        stats[0] += groups.len();
                        if *run > 0 {
                            stats[3] += groups.len();
                        }
                    }
                    Output::Recycle { .. } => stats[1] += 1,
                    Output::Discard { .. } => stats[2] += 1,
                    _ => {}
                }
            }
        }
        // make sure the model actually exercises every path
        assert!(stats.iter().all(|&s| s > 100), "{stats:?}");
        println!("trained groups / recycles / discards / trained after restart: {stats:?}");
    }
}
