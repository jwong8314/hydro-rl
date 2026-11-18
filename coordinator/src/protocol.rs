//! Wire protocol between Python processes (Miles producer, trainer, inference
//! workers) and the Hydro coordinator. JSON lines, one object per message.
//!
//! Only *control-plane metadata* crosses this boundary. Token ids, logprobs and
//! rewards stay in the Python process that produced them; the coordinator
//! decides *which* groups are admitted, recycled or discarded, by id.

use serde::{Deserialize, Serialize};

pub type GroupId = u64;
pub type Attempt = u32;
/// Recycle generation of a group. Only the coordinator advances it.
pub type Epoch = u32;
pub type Version = u64;
/// Trainer incarnation. Each new trainer process (fresh start, crash restart,
/// checkpoint restore) uses a run id higher than any before it. Weight versions
/// are only comparable within one run. It works as a fencing token: messages
/// from an older run are ignored.
pub type Run = u64;

/// Events sent to the coordinator. Different events can come from different
/// clients (producer, workers, trainer), so their arrival order is arbitrary.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// The producer started generation of `group` in `epoch` (0, or the
    /// `next_epoch` of the latest `Recycle`). `attempt` counts the producer's
    /// own timeout retries within the epoch, starting at 0.
    Dispatch {
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        #[serde(default)]
        run: Run,
    },
    /// A finished prompt group. `oldest_version`/`newest_version` summarize the
    /// per-token weight-version spans (Miles: `Sample.all_weight_version_spans`).
    Complete {
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        /// `None` when no token carries a version (Miles then never treats the
        /// group as stale).
        oldest_version: Option<Version>,
        newest_version: Option<Version>,
        n_samples: u32,
        tokens: u64,
        #[serde(default)]
        run: Run,
    },
    /// The generate function gave up on this attempt (Miles: `Sample.Status.ABORTED`).
    Abort {
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        #[serde(default)]
        run: Run,
    },
    /// The buffer dropped this result on purpose and the group must not be
    /// regenerated (Miles: missing reward, `--dynamic-sampling-filter-path`).
    /// The group is closed like a trained one; other attempts are cancelled.
    Drop {
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        #[serde(default)]
        run: Run,
    },
    /// Ask for the coordinator's `Config`, so a client can check it matches
    /// its own launch arguments.
    Hello,
    /// The trainer finished a weight update; inference now serves `version`.
    /// The first `Version` of a higher `run` is a restart: `version` may be
    /// lower than anything seen before (e.g. restored from a checkpoint), and
    /// every group not yet trained on is recycled. The trainer must send it
    /// only once inference serves the restored weights.
    Version {
        version: Version,
        #[serde(default)]
        run: Run,
    },
    /// The trainer wants a batch of `n_groups` groups. Held until the
    /// coordinator is in `run`; dropped if `run` is older.
    Get {
        request: u64,
        n_groups: u32,
        #[serde(default)]
        run: Run,
    },
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RecycleReason {
    Stale,
    Aborted,
    Partial,
    /// The trainer restarted. Results made under the previous run's weights
    /// can't be trusted, because version numbers may repeat across runs.
    Restart,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DiscardReason {
    /// The group already has an admitted result (e.g. a timed-out attempt
    /// finished after its retry did).
    Duplicate,
    /// The result is from an epoch before a recycle of this group, or a newer
    /// retry in the same epoch is still running.
    Superseded,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BatchEntry {
    pub group: GroupId,
    pub epoch: Epoch,
    pub attempt: Attempt,
    pub oldest_version: Option<Version>,
    pub tokens: u64,
    /// Run the result was generated in; always the batch's run.
    pub run: Run,
}

/// Decisions emitted by the coordinator.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Output {
    /// Train on exactly these group results, at trainer version `version`.
    Batch {
        request: u64,
        run: Run,
        version: Version,
        groups: Vec<BatchEntry>,
    },
    /// Sent to the trainer that announced a new run: the coordinator switched
    /// to `run` at `version` and recycled `recycled` untrained groups.
    Restarted {
        run: Run,
        version: Version,
        recycled: u32,
    },
    /// Regenerate `group` in epoch `next_epoch` (attempt 0); any result from
    /// an older epoch will be discarded.
    Recycle {
        group: GroupId,
        reason: RecycleReason,
        next_epoch: Epoch,
    },
    /// Drop this result payload; it will never be trained on.
    Discard {
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
        reason: DiscardReason,
    },
    /// Stop generating this attempt; the group already has a usable result.
    Cancel {
        group: GroupId,
        epoch: Epoch,
        attempt: Attempt,
    },
    /// `group` has been recycled `recycles` times; long or slow rollouts are
    /// at risk of never being trained on.
    Starved { group: GroupId, recycles: u32 },
    /// Reply to `Hello`.
    Config { config: Config },
}

pub const NO_STALENESS_LIMIT: u64 = u64::MAX;

/// Static coordinator parameters (Miles launch args).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    /// `--max-weight-staleness`. `NO_STALENESS_LIMIT` disables the filter,
    /// which is Miles' default (`None`).
    pub max_staleness: u64,
    /// Recycle results with fewer than this many samples (`--n-samples-per-prompt`).
    /// 0 disables the check, which is what Miles' buffer does.
    pub group_size: u32,
    /// Emit `Starved` once a group has been recycled this many times.
    pub starvation_threshold: u32,
    /// Prefer groups closest to going stale (oldest weight version first)
    /// instead of FIFO completion order. `false` matches Miles.
    pub oldest_first: bool,
    /// Deduplicate by (group, epoch, attempt). `false` reproduces a buffer that
    /// trusts every completion it receives; used to show what the simulator
    /// catches.
    pub dedup: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_staleness: 1,
            group_size: 1,
            starvation_threshold: 3,
            oldest_first: true,
            dedup: true,
        }
    }
}
