//! Agent I/O Scheduler: the read → think → retrieve pipeline.
//!
//! The canonical tool pipeline (`AgentEngine::parse_tool_calls` →
//! `App::claim_tool_call` → `AgentEngine::execute_proposed`) remains the ONLY
//! authority that executes model tool calls, guarded by
//! [`crate::agent::ToolDispatchRegistry`] for exactly-once semantics.
//! This module is the ONE shared I/O queue beneath it — explicit model
//! reads and speculative predictions run on the same bounded worker pool,
//! explicit work at [`IoPriority::Explicit`] genuinely outranking
//! speculation:
//!
//! ```text
//! Explicit model read
//!        ↓
//! canonical claim (dispatch registry — scheduler never touches it)
//!        ↓
//! submit_explicit_batch → shared priority queue → worker
//!        ↓                                         ↓
//!   promotion? (validated cache serve)      canonical execute_proposed
//!        ↓                                         ↓
//! ordered outcomes bound to call_id → deterministic context commit
//!
//! Explicit read result → predictor → submit_speculative → same queue
//! → worker → read-only cache (generation-versioned) → next promotion
//! ```
//!
//! Hard rules (see module docs on each item):
//! - Speculation is READ-ONLY by construction: speculative jobs carry no
//!   action and resolve only into the cache. [`ReadOrigin`] separates
//!   explicit from speculative results.
//! - Speculative jobs never enter `ToolDispatchRegistry`; explicit jobs
//!   were claimed BEFORE submit. Failed reads (error outcomes) must have
//!   their claims released by the caller — nothing with side effects ran.
//! - Every speculative result is generation-checked against writes, so the
//!   `prefetch-then-write-then-insert` race cannot poison the cache.
//! - Cancellation is COOPERATIVE, not preemptive: a worker inside a
//!   blocking filesystem read only notices afterwards. Never document or
//!   rely on immediate I/O interruption.
//! - The cache is a best-effort LATENCY optimization, not authoritative
//!   file state: external writers are detected via mtime+length, which can
//!   miss same-size/same-mtime modifications. The canonical executor
//!   always remains the source of truth.
//! - Cross-run cache sharing is INTENDED: files are not owned by runs, so
//!   a file cached during run A may serve run B. Safety comes from global
//!   monotonic generations (any canonical write bumps + invalidates) plus
//!   mtime/length/TTL revalidation — never from run isolation.
//! - Priority is NON-PREEMPTIVE: an explicit job jumps the queue but never
//!   interrupts a worker mid-read (see cooperative cancellation above).
//!   Local reads are short enough that reserved explicit capacity is not
//!   warranted unless profiling proves otherwise.
//! - Directory listings are scheduled (ordering/fairness) but never
//!   cached: local readdir is sub-millisecond, and dir mtime does not move
//!   on content changes, so a listing cache would risk stale sizes for no
//!   measurable gain. Promotion is a file-content mechanism only.
//! - Explicit NON-GOAL: injecting retrieved data into an already-running
//!   model generation. The scheduler overlaps I/O with generation and
//!   accelerates the NEXT turn; mutating an active generation's context
//!   requires model-runtime interrupt/resume support and must never be
//!   faked by rewriting prompts mid-stream.
//! - Workers are `std::thread`s, not tokio tasks: the scheduler is invoked
//!   from synchronous canonical paths (and unit tests) with no async
//!   runtime guaranteed. Cancellation reuses
//!   [`tokio_util::sync::CancellationToken`] semantics.
//!
//! `App` owns no scheduler state; it integrates at four points: claim →
//! batch submit, failure-release on error outcomes, new-run/cancel hooks,
//! and the throttled streaming-intent watermark.
//!
//! ## Consistency contract (read this before reasoning about promotion)
//!
//! Cache promotion is a **best-effort latency optimization**. It is NOT
//! linearizable against arbitrary external filesystem writers: an external
//! process can rewrite a file between the serve-time validation and the
//! caller consuming the bytes, and same-mtime/same-length rewrites are
//! undetectable by metadata at all.
//!
//! What promotion DOES guarantee, mechanically verified:
//! - every serve revalidates generation + mtime + length + TTL against
//!   the live filesystem — no entry is served without a fresh check;
//! - every canonical write/delete bumps the generation and invalidates,
//!   so Hercules' own mutations can never be shadowed by older bytes;
//! - every population path (speculative worker, read-through, warm-up)
//!   goes through the ONE `read_stable_file` primitive: sandbox gate,
//!   stable identity, generation stamp, metadata snapshot before AND
//!   after the read, generation re-check before insert.
//!
//! The canonical filesystem remains authoritative: on any validation
//! doubt the entry is dropped and the read falls through to disk.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Bounds (resource safety is a principle, not an afterthought)
// ---------------------------------------------------------------------------

/// Maximum cached files.
pub const MAX_ENTRIES: usize = 64;
/// Maximum total cached bytes (8 MiB).
pub const MAX_TOTAL_BYTES: usize = 8 * 1024 * 1024;
/// Files larger than this are never cached (256 KiB).
pub const MAX_FILE_BYTES: u64 = 256 * 1024;
/// Cache entry lifetime.
pub const CACHE_TTL: Duration = Duration::from_secs(120);
/// Upper bound on candidates accepted per speculation event.
pub const MAX_CANDIDATES_PER_EVENT: usize = 8;
/// Background worker threads.
const WORKER_COUNT: usize = 4;
/// Streaming intent re-predicts only after this much new text.
const STREAM_PREDICT_DELTA: usize = 2048;
/// Max prose-derived candidates per stream event.
const MAX_STREAM_CANDIDATES: usize = 4;

// ---------------------------------------------------------------------------
// TODO 2 — job types
// ---------------------------------------------------------------------------

/// What kind of I/O a scheduler job performs. Deliberately read-only:
/// there is no variant that can express a write, command, spawn, or any
/// other side effect — speculation *cannot* mutate state by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoJobKind {
    ExplicitRead,
    ExplicitLs,
    SpeculativeRead,
    SpeculativeLs,
    /// Reserved: semantic-retrieval stage (not yet implemented).
    #[allow(dead_code)]
    SemanticRetrieve,
    /// Reserved: code-graph retrieval stage (not yet implemented).
    #[allow(dead_code)]
    CodeGraphRetrieve,
}

/// Explicit model/user reads always outrank speculation. Lower rank pops
/// first from the priority queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IoPriority {
    Critical = 0,
    Explicit = 1,
    High = 2,
    Normal = 3,
    Speculative = 4,
    Background = 5,
}

/// Whether bytes came from an explicit request or from speculation.
/// Served results carry this so speculative hits are observable instead
/// of masquerading as explicit reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadOrigin {
    Explicit,
    Speculative,
}

/// A unit of scheduled I/O: either a speculative prediction (resolved
/// into the read-only cache) or an explicit model request (resolved via
/// the canonical executor and delivered to the caller). Both share one
/// priority queue so explicit work genuinely outranks speculation.
#[derive(Debug, Clone)]
pub struct IoJob {
    pub id: u64,
    pub run_id: u64,
    pub priority: IoPriority,
    pub kind: IoJobKind,
    pub target: PathBuf,
    /// File generation captured at submit time; the worker re-checks it
    /// after the read so a concurrent write invalidates the result.
    pub generation: u64,
    pub reason: RetrievalReason,
    /// Scheduler epoch captured at submit. `reset_for_test` bumps the
    /// epoch so stale worker jobs from a previous test can never mutate
    /// new state: popped stale jobs drop, in-flight ones refuse insert.
    pub epoch: u64,
    /// The claimed model action (explicit jobs only). The claim itself
    /// happened BEFORE submit through `App::claim_tool_call`; the worker
    /// never touches the dispatch registry.
    pub action: Option<crate::agent::ProposedAction>,
    /// Where to deliver the outcome (explicit jobs only). The worker
    /// sends EXACTLY once on every path (success, error, cancellation,
    /// panic) so callers never block forever.
    pub reply: Option<std::sync::mpsc::Sender<ExplicitReadOutcome>>,
}

/// Outcome of an explicit scheduler job, bound to the model `call_id`
/// (mirroring [`crate::agent::ToolResult` semantics).
#[derive(Debug, Clone)]
pub struct ExplicitReadOutcome {
    pub call_id: u64,
    pub result: String,
    pub served_from_cache: bool,
}

/// Handle for one submitted explicit action: yields exactly one outcome.
/// Collect receipts in submission order for deterministic result commits.
pub struct ExplicitReceipt {
    pub call_id: u64,
    pub job_id: u64,
    pub outcome: std::sync::mpsc::Receiver<ExplicitReadOutcome>,
}

// ---------------------------------------------------------------------------
// TODO 3 — speculation identity (separate from ProposedAction::call_id)
// ---------------------------------------------------------------------------

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

fn next_job_id() -> u64 {
    NEXT_JOB_ID.fetch_add(1, Ordering::SeqCst)
}

/// Identity of one speculative prediction. Independent per run, cancellable
/// independently, and never confused with a model tool `call_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeculationId(pub u64);

/// One predicted read: *why* it was predicted and *how confident* the
/// predictor is. Read-only: resolving it only ever fills the cache.
#[derive(Debug, Clone)]
pub struct SpeculativeRead {
    pub id: SpeculationId,
    pub run_id: u64,
    pub source_path: PathBuf,
    pub target_path: PathBuf,
    pub confidence: f32,
    pub reason: RetrievalReason,
}

// ---------------------------------------------------------------------------
// TODO 9/10 — retrieval reasons, candidates, confidence
// ---------------------------------------------------------------------------

/// Why a file was predicted. Drives priority via [`RetrievalConfig`].
/// Variants marked reserved belong to later stages (code-graph neighbors,
/// symbol references, semantic similarity) and are kept so the taxonomy
/// is stable when those stages land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum RetrievalReason {
    RustModule,
    JsImport,
    PythonImport,
    IncludeDirective,
    MarkdownLink,
    QuotedPath,
    SameDirectorySibling,
    RecentTaskHistory,
    ModelIntent,
    CodeGraphEdge,
    SymbolReference,
    SemanticSimilarity,
}

impl RetrievalReason {
    pub fn label(self) -> &'static str {
        match self {
            RetrievalReason::RustModule => "rust-mod",
            RetrievalReason::JsImport => "js-import",
            RetrievalReason::PythonImport => "py-import",
            RetrievalReason::IncludeDirective => "include",
            RetrievalReason::MarkdownLink => "md-link",
            RetrievalReason::QuotedPath => "quoted-path",
            RetrievalReason::SameDirectorySibling => "sibling",
            RetrievalReason::RecentTaskHistory => "recent-task",
            RetrievalReason::ModelIntent => "model-intent",
            RetrievalReason::CodeGraphEdge => "codegraph-edge",
            RetrievalReason::SymbolReference => "symbol-ref",
            RetrievalReason::SemanticSimilarity => "semantic",
        }
    }
}

/// A file the predictor believes the model will read next.
#[derive(Debug, Clone)]
pub struct RetrievalCandidate {
    pub path: PathBuf,
    pub confidence: f32,
    pub reason: RetrievalReason,
    pub estimated_bytes: u64,
}

/// Tunables for prediction → priority. No magic numbers scattered through
/// the pipeline; construct once, share via [`AgentIoScheduler`].
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub rust_module: f32,
    pub js_import: f32,
    pub python_import: f32,
    pub include_directive: f32,
    pub markdown_link: f32,
    pub quoted_path: f32,
    pub sibling: f32,
    pub recent_task: f32,
    pub model_intent: f32,
    /// confidence >= high_threshold → High priority.
    pub high_threshold: f32,
    /// confidence >= normal_threshold → Normal priority.
    pub normal_threshold: f32,
    /// confidence >= background_threshold → Background priority, else skip.
    pub background_threshold: f32,
    pub max_candidates_per_event: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            rust_module: 0.90,
            js_import: 0.85,
            python_import: 0.85,
            include_directive: 0.80,
            markdown_link: 0.50,
            quoted_path: 0.30,
            sibling: 0.30,
            recent_task: 0.35,
            model_intent: 0.30,
            high_threshold: 0.80,
            normal_threshold: 0.50,
            background_threshold: 0.25,
            max_candidates_per_event: MAX_CANDIDATES_PER_EVENT,
        }
    }
}

impl RetrievalConfig {
    pub fn confidence_for(&self, reason: RetrievalReason) -> f32 {
        match reason {
            RetrievalReason::RustModule => self.rust_module,
            RetrievalReason::JsImport => self.js_import,
            RetrievalReason::PythonImport => self.python_import,
            RetrievalReason::IncludeDirective => self.include_directive,
            RetrievalReason::MarkdownLink => self.markdown_link,
            RetrievalReason::QuotedPath => self.quoted_path,
            RetrievalReason::SameDirectorySibling => self.sibling,
            RetrievalReason::RecentTaskHistory => self.recent_task,
            RetrievalReason::ModelIntent => self.model_intent,
            RetrievalReason::CodeGraphEdge => self.high_threshold,
            RetrievalReason::SymbolReference => self.high_threshold,
            RetrievalReason::SemanticSimilarity => self.normal_threshold,
        }
    }

    /// Map a confidence to a queue priority; below background → skip.
    pub fn priority_for(&self, confidence: f32) -> Option<IoPriority> {
        if confidence >= self.high_threshold {
            Some(IoPriority::High)
        } else if confidence >= self.normal_threshold {
            Some(IoPriority::Normal)
        } else if confidence >= self.background_threshold {
            Some(IoPriority::Background)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// TODO 8 — provenance-aware cache entries
// ---------------------------------------------------------------------------

/// Where cached bytes came from. Served hits report this so a speculative
/// hit is observable (`CACHE HIT (speculative)`) rather than pretending
/// the model read the file before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOrigin {
    ExplicitRead,
    SpeculativeRead,
}

pub struct CacheEntry {
    pub content: String,
    /// File generation at fetch time; serve requires equality with the
    /// current generation (write-race guard alongside mtime+length).
    pub generation: u64,
    pub mtime: SystemTime,
    pub len: u64,
    pub fetched_at: Instant,
    pub origin: CacheOrigin,
    /// Counted toward `useful_prefetches` at most once.
    pub usefulness_counted: bool,
}

/// A successful promotion: bytes served without a second filesystem read.
#[derive(Debug, Clone)]
pub struct CacheHit {
    pub content: String,
    pub origin: ReadOrigin,
    pub generation: u64,
}

struct CacheStore {
    entries: HashMap<PathBuf, CacheEntry>,
    total_bytes: usize,
    /// Last-serve provenance for diagnostics. Bounded ring (NOT a map):
    /// must never grow with every unique path ever served.
    last_serve: VecDeque<(PathBuf, (ReadOrigin, Instant))>,
}

/// Cap on retained serve-provenance records.
const MAX_LAST_SERVE: usize = 256;

impl CacheStore {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            last_serve: VecDeque::new(),
        }
    }

    /// Record a serve; evict oldest first past the bound.
    fn note_served(&mut self, canonical: PathBuf, origin: ReadOrigin) {
        if let Some(pos) = self.last_serve.iter().position(|(p, _)| p == &canonical) {
            self.last_serve.remove(pos);
        }
        self.last_serve
            .push_back((canonical, (origin, Instant::now())));
        while self.last_serve.len() > MAX_LAST_SERVE {
            self.last_serve.pop_front();
        }
    }

    fn last_served(&self, canonical: &Path) -> Option<(ReadOrigin, Instant)> {
        self.last_serve
            .iter()
            .rev()
            .find(|(p, _)| p == canonical)
            .map(|(_, v)| *v)
    }

    fn evict_oldest(&mut self, metrics: &mut PrefetchMetrics) {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.fetched_at)
            .map(|(k, _)| k.clone());
        if let Some(k) = oldest {
            self.remove(&k, metrics);
        }
    }

    fn remove(&mut self, canonical: &Path, metrics: &mut PrefetchMetrics) {
        if let Some(old) = self.entries.remove(canonical) {
            self.total_bytes = self.total_bytes.saturating_sub(old.content.len());
            // Evicted/expired/invalidated without ever serving → wasted.
            if old.origin == CacheOrigin::SpeculativeRead && !old.usefulness_counted {
                metrics.wasted_prefetches += 1;
            }
        }
    }

    fn put(&mut self, canonical: PathBuf, entry: CacheEntry, metrics: &mut PrefetchMetrics) {
        if entry.content.len() as u64 > MAX_FILE_BYTES {
            return;
        }
        if let Some(old) = self.entries.remove(&canonical) {
            self.total_bytes = self.total_bytes.saturating_sub(old.content.len());
            if old.origin == CacheOrigin::SpeculativeRead && !old.usefulness_counted {
                metrics.wasted_prefetches += 1;
            }
        }
        while (self.entries.len() >= MAX_ENTRIES
            || self.total_bytes + entry.content.len() > MAX_TOTAL_BYTES)
            && !self.entries.is_empty()
        {
            self.evict_oldest(metrics);
        }
        self.total_bytes += entry.content.len();
        self.entries.insert(canonical, entry);
    }
}

// ---------------------------------------------------------------------------
// TODO 18/19 — metrics + budgets
// ---------------------------------------------------------------------------

/// Scheduler health. A prefetch is *useful* when the model later requests
/// the file and it is served from cache; usefulness = useful / completed.
#[derive(Debug, Clone, Default)]
pub struct PrefetchMetrics {
    pub submitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub rejected_sandbox: u64,
    pub budget_dropped: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub speculative_hits: u64,
    pub useful_prefetches: u64,
    pub wasted_prefetches: u64,
    pub bytes_read: u64,
}

impl PrefetchMetrics {
    /// Fraction of completed prefetches that later served a real request.
    pub fn usefulness(&self) -> f32 {
        if self.completed == 0 {
            0.0
        } else {
            self.useful_prefetches as f32 / self.completed as f32
        }
    }

    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f32 / total as f32
        }
    }
}

/// Per-run speculation budget: speculation must never consume unbounded
/// resources, no matter how many candidates the predictor emits.
/// Byte accounting is admission control on PREDICTED sizes (estimated at
/// submit); actual bytes read are observed via `PrefetchMetrics::bytes_read`
/// and can differ if a file changed between prediction and fetch.
#[derive(Debug, Clone)]
pub struct SpeculationBudget {
    pub max_jobs_per_run: usize,
    pub max_estimated_bytes_per_run: u64,
    pub max_candidates_per_event: usize,
}

impl Default for SpeculationBudget {
    fn default() -> Self {
        Self {
            max_jobs_per_run: 64,
            max_estimated_bytes_per_run: 16 * 1024 * 1024,
            max_candidates_per_event: MAX_CANDIDATES_PER_EVENT,
        }
    }
}

#[derive(Debug, Default)]
struct BudgetUse {
    jobs: usize,
    bytes: u64,
}

/// Observable scheduler happenings (promotion, drops, rejects). Bounded
/// ring for diagnostics — delivery of explicit outcomes goes through
/// per-job channels, never this log.
#[derive(Debug, Clone)]
pub enum SchedulerEvent {
    Promoted { path: PathBuf, origin: ReadOrigin },
    CancelledJob { job_id: u64 },
    BudgetDropped { count: u64 },
    SandboxRejected { path: PathBuf },
}

impl SchedulerEvent {
    fn label(&self) -> String {
        match self {
            SchedulerEvent::Promoted { path, origin } => {
                format!("promoted {} ({:?})", path.display(), origin)
            }
            SchedulerEvent::CancelledJob { job_id } => format!("cancelled job #{job_id}"),
            SchedulerEvent::BudgetDropped { count } => format!("budget dropped {count}"),
            SchedulerEvent::SandboxRejected { path } => {
                format!("sandbox rejected {}", path.display())
            }
        }
    }
}

/// Outcome of `cancel_job`, distinguishing every ownership state.
/// Callers must handle each variant: only `CancelledBeforeExecution`
/// guarantees the action never ran; `CancellationRequested` means the
/// worker owns it and exactly one terminal outcome is still coming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobCancelOutcome {
    /// Was queued: removed, terminal cancellation outcome delivered.
    CancelledBeforeExecution,
    /// Was active (popped, worker-owned): flagged for cooperative cancel;
    /// the worker's terminal outcome is still exactly-once delivered.
    CancellationRequested,
    /// Already finished (success, error, cancel, or panic path all land
    /// here via the active guard's Drop).
    AlreadyTerminal,
    /// No record anywhere: never submitted, or evicted from the bounded
    /// terminal ring. Safe no-op.
    UnknownJob,
}

/// Cap on retained terminal job ids.
const MAX_COMPLETED_JOBS: usize = 1024;

/// Memory caps for scheduler bookkeeping. Sweeps below are
/// reachability-based, never blind clearing: a record survives iff live
/// scheduler state (queued/active jobs, cache entries) can observe it.
const MAX_GENERATIONS: usize = 4096;
const MAX_RUN_STATES: usize = 128;

/// Cap on retained diagnostic events.
const MAX_EVENTS: usize = 64;

/// Ownership release: pop_eligible registers Active on queue removal;
/// dropping this guard performs the terminal transition (active → ring).
/// Held across the whole execution so success, error, cancellation AND
/// panic-unwind all release ownership — a panicking job can never wedge
/// a caller or leak an active record.
struct ActiveOwnershipGuard {
    sched: &'static AgentIoScheduler,
    job_id: u64,
}

impl Drop for ActiveOwnershipGuard {
    fn drop(&mut self) {
        self.sched.lock().mark_completed(self.job_id);
    }
}

/// A job currently executing on a worker (for diagnostics).
#[derive(Debug, Clone)]
struct ActiveJob {
    job_id: u64,
    kind: IoJobKind,
    target: PathBuf,
    run_id: u64,
    started: Instant,
}

// ---------------------------------------------------------------------------
// Priority queue ordering (BinaryHeap is a max-heap; invert so the lowest
// (priority rank, sequence) pops first → explicit always beats speculative)
// ---------------------------------------------------------------------------

struct QueuedJob {
    job: IoJob,
    seq: u64,
}

impl PartialEq for QueuedJob {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}
impl Eq for QueuedJob {}

impl PartialOrd for QueuedJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse: smaller priority rank / earlier seq = "greater" = pops first.
        other
            .job
            .priority
            .cmp(&self.job.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

// ---------------------------------------------------------------------------
// AgentIoScheduler
// ---------------------------------------------------------------------------

struct SchedulerState {
    queue: BinaryHeap<QueuedJob>,
    /// Speculative (kind, target) currently queued or executing — dedupe so
    /// one prediction cannot flood the queue. Explicit jobs never dedupe:
    /// the canonical dispatcher owns them.
    inflight_speculative: HashSet<(IoJobKind, PathBuf)>,
    /// Jobs currently executing on workers (diagnostics + ownership).
    /// A job is inserted here ATOMICALLY with its queue removal inside
    /// `pop_eligible`, so there is never an unowned gap where `cancel_job`
    /// could observe neither queued nor active state.
    active: HashMap<u64, ActiveJob>,
    /// Recently terminal job ids (bounded ring). Lets `cancel_job`
    /// distinguish AlreadyTerminal from UnknownJob.
    completed_jobs: VecDeque<u64>,
    /// Bounded ring of recent scheduler events (diagnostics).
    events: VecDeque<SchedulerEvent>,
    cancelled_runs: HashSet<u64>,
    /// Individually cancelled jobs (timeout protocol). Workers check this
    /// before executing; queued jobs are removed with an error outcome.
    cancelled_jobs: HashSet<u64>,
    /// Test epoch: bumped by `reset_for_test` so jobs submitted under an
    /// old epoch can never affect new state (see `IoJob.epoch`).
    epoch: u64,
    run_tokens: HashMap<u64, tokio_util::sync::CancellationToken>,
    generations: HashMap<PathBuf, u64>,
    cache: CacheStore,
    metrics: PrefetchMetrics,
    budgets: HashMap<u64, BudgetUse>,
    /// Per-run streaming watermark for throttled intent prediction.
    stream_marks: HashMap<u64, usize>,
    /// Recently explicitly-read files (recency signal for the predictor).
    recent_reads: VecDeque<PathBuf>,
    queue_seq: u64,
}

impl SchedulerState {
    fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            inflight_speculative: HashSet::new(),
            active: HashMap::new(),
            completed_jobs: VecDeque::new(),
            events: VecDeque::new(),
            cancelled_runs: HashSet::new(),
            cancelled_jobs: HashSet::new(),
            run_tokens: HashMap::new(),
            generations: HashMap::new(),
            cache: CacheStore::new(),
            metrics: PrefetchMetrics::default(),
            budgets: HashMap::new(),
            stream_marks: HashMap::new(),
            recent_reads: VecDeque::new(),
            queue_seq: 0,
            epoch: 0,
        }
    }

    /// Remove a cache entry, counting speculative waste. Takes `&mut self`
    /// once so call sites holding the mutex guard never fight DerefMut's
    /// double-borrow.
    fn cache_remove(&mut self, canonical: &Path) {
        let (cache, metrics) = (&mut self.cache, &mut self.metrics);
        cache.remove(canonical, metrics);
    }

    fn cache_put(&mut self, canonical: PathBuf, entry: CacheEntry) {
        let (cache, metrics) = (&mut self.cache, &mut self.metrics);
        cache.put(canonical, entry, metrics);
    }

    /// Bounded diagnostic event ring.
    fn push_event(state: &mut SchedulerState, ev: SchedulerEvent) {
        state.events.push_back(ev);
        while state.events.len() > MAX_EVENTS {
            state.events.pop_front();
        }
    }

    /// Retire generations no queued/active job and no cache entry can
    /// observe. Safe: a dropped record reads back as generation 0, which
    /// can only cause an extra cache miss (re-read + reinsert), never a
    /// false hit — a stale entry's nonzero generation mismatches the
    /// default, and zero-generation entries remain mtime-validated.
    fn retire_generations(&mut self) {
        if self.generations.len() <= MAX_GENERATIONS {
            return;
        }
        let mut referenced: HashSet<&PathBuf> = HashSet::new();
        for q in self.queue.iter() {
            referenced.insert(&q.job.target);
        }
        for a in self.active.values() {
            referenced.insert(&a.target);
        }
        for k in self.cache.entries.keys() {
            referenced.insert(k);
        }
        self.generations.retain(|k, _| referenced.contains(k));
    }

    /// Retire per-run cancellation/token state for runs with no queued
    /// or active jobs. Budgets and stream marks are NOT swept here: they
    /// belong to possibly-still-live runs (a run submits across many
    /// turns), and only explicit run termination retires them — see
    /// `cancel_run` (cancelled runs) and `end_run` (finished runs).
    /// A run with a still-executing worker always has an active record
    /// (atomic pop→active), so its cancellation state is retained and no
    /// stale work can slip through a retirement gap.
    fn retire_cancelled_state(&mut self) {
        let total = self.cancelled_runs.len() + self.run_tokens.len();
        if total <= MAX_RUN_STATES {
            return;
        }
        let mut live_runs: HashSet<u64> = HashSet::new();
        for q in self.queue.iter() {
            live_runs.insert(q.job.run_id);
        }
        for a in self.active.values() {
            live_runs.insert(a.run_id);
        }
        self.cancelled_runs.retain(|r| live_runs.contains(r));
        self.run_tokens.retain(|r, _| live_runs.contains(r));
    }

    /// Terminal transition: drop the ownership record, clear any pending
    /// cancel flag, and retain the id in the bounded terminal ring so
    /// late `cancel_job` calls observe AlreadyTerminal, not UnknownJob.
    fn mark_completed(&mut self, job_id: u64) {
        self.active.remove(&job_id);
        self.cancelled_jobs.remove(&job_id);
        if !self.completed_jobs.contains(&job_id) {
            self.completed_jobs.push_back(job_id);
            while self.completed_jobs.len() > MAX_COMPLETED_JOBS {
                self.completed_jobs.pop_front();
            }
        }
    }

    fn is_terminal(&self, job_id: u64) -> bool {
        self.completed_jobs.contains(&job_id)
    }
}

/// I/O orchestration layer for the agent loop. Owns speculation, caching,
/// budgets, metrics and cancellation — never tool parsing, never execution
/// authority, never dispatch claims.
pub struct AgentIoScheduler {
    state: Mutex<SchedulerState>,
    wake: Mutex<std::sync::mpsc::Sender<()>>,
    _wake_rx: Condvar,
    receiver: Mutex<std::sync::mpsc::Receiver<()>>,
    workers_started: AtomicU64,
    pub config: RetrievalConfig,
    pub budget: SpeculationBudget,
}

impl AgentIoScheduler {
    pub fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            state: Mutex::new(SchedulerState::new()),
            wake: Mutex::new(tx),
            _wake_rx: Condvar::new(),
            receiver: Mutex::new(rx),
            workers_started: AtomicU64::new(0),
            config: RetrievalConfig::default(),
            budget: SpeculationBudget::default(),
        }
    }

    pub fn global() -> &'static AgentIoScheduler {
        static SCHEDULER: OnceLock<AgentIoScheduler> = OnceLock::new();
        SCHEDULER.get_or_init(AgentIoScheduler::new)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SchedulerState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ensure_workers(&'static self) {
        if self.workers_started.swap(1, Ordering::SeqCst) == 0 {
            for i in 0..WORKER_COUNT {
                let scheduler = self;
                let _ = std::thread::Builder::new()
                    .name(format!("hercules-io-{i}"))
                    .spawn(move || scheduler.worker_loop());
            }
        }
    }

    fn kick(&self) {
        let _ = self.wake.lock().unwrap().send(());
    }

    // -- runs ------------------------------------------------------------

    /// Link an App run token so Ctrl+C / supersede propagates to workers.
    pub fn link_run_token(&self, run_id: u64, token: tokio_util::sync::CancellationToken) {
        self.lock().run_tokens.insert(run_id, token);
    }

    /// Cancel all speculative work of a run (new prompt, Ctrl+C, supersede).
    /// Queued speculative jobs are dropped; queued EXPLICIT jobs are
    /// completed immediately with a cancellation error so their callers
    /// never block (the App then releases those claims — nothing ran).
    /// In-flight filesystem reads are cooperative: a worker inside
    /// `read_to_string` only notices afterwards (see module docs).
    pub fn cancel_run(&self, run_id: u64) {
        {
            let mut st = self.lock();
            st.cancelled_runs.insert(run_id);
            let kept: BinaryHeap<QueuedJob> = std::mem::take(&mut st.queue)
                .into_iter()
                .filter(|q| {
                    let drop_it = q.job.run_id == run_id;
                    if drop_it {
                        st.inflight_speculative
                            .remove(&(q.job.kind, q.job.target.clone()));
                        st.metrics.cancelled += 1;
                        if let Some(reply) = &q.job.reply {
                            let _ = reply.send(ExplicitReadOutcome {
                                call_id: q.job.action.as_ref().map(|a| a.call_id).unwrap_or(0),
                                result: "Error: scheduler: run cancelled before read executed"
                                    .to_string(),
                                served_from_cache: false,
                            });
                        }
                        SchedulerState::push_event(
                            &mut st,
                            SchedulerEvent::CancelledJob { job_id: q.job.id },
                        );
                        st.mark_completed(q.job.id);
                    }
                    !drop_it
                })
                .collect();
            st.queue = kept;
            // Terminal-cancelled: admission/watermark state retires now
            // (this run id will never submit again). Flags and token
            // copies stay for in-flight stragglers.
            st.budgets.remove(&run_id);
            st.stream_marks.remove(&run_id);
            if let Some(tok) = st.run_tokens.remove(&run_id) {
                tok.cancel();
            }
            // Opportunistic lifecycle sweeps. Safe here: any still-live
            // worker holds an active record, which pins its state.
            st.retire_generations();
            st.retire_cancelled_state();
        }
        self.kick();
    }

    /// A run ended normally (finished or superseded — see also
    /// `cancel_run` for terminal cancellation): drop its
    /// admission/watermark state. Cancellation flags and token copies are
    /// left to the reachability sweep — in-flight stragglers still need
    /// them. Run ids are unique, so no future submission reuses them.
    pub fn end_run(&self, run_id: u64) {
        let mut st = self.lock();
        st.budgets.remove(&run_id);
        st.stream_marks.remove(&run_id);
    }

    fn run_cancelled(st: &SchedulerState, run_id: u64) -> bool {
        if st.cancelled_runs.contains(&run_id) {
            return true;
        }
        st.run_tokens
            .get(&run_id)
            .map(|t| t.is_cancelled())
            .unwrap_or(false)
    }

    /// Cancel one explicit job (timeout protocol) with an ownership-aware
    /// outcome. Queued jobs are removed and complete immediately with a
    /// cancellation error; active (worker-owned) jobs are flagged for
    /// cooperative cancel and still deliver exactly one terminal outcome.
    ///
    /// Single-owner invariant: after `cancel_job`, the caller must NEVER
    /// execute the action itself — the worker remains the sole possible
    /// owner, and its late outcome (if any) goes to the caller's dropped
    /// receiver. A timeout therefore records a local error, never a
    /// second execution.
    pub fn cancel_job(&self, job_id: u64) -> JobCancelOutcome {
        let mut st = self.lock();
        // Queued: remove + terminal error outcome.
        if let Some(pos) = st.queue.iter().position(|q| q.job.id == job_id) {
            let mut heap: Vec<QueuedJob> = std::mem::take(&mut st.queue).into_vec();
            let q = heap.remove(pos);
            st.queue = heap.into_iter().collect();
            st.inflight_speculative
                .remove(&(q.job.kind, q.job.target.clone()));
            st.metrics.cancelled += 1;
            if let Some(reply) = &q.job.reply {
                let _ = reply.send(ExplicitReadOutcome {
                    call_id: q.job.action.as_ref().map(|a| a.call_id).unwrap_or(0),
                    result: "Error: scheduler: job cancelled".to_string(),
                    served_from_cache: false,
                });
            }
            SchedulerState::push_event(&mut st, SchedulerEvent::CancelledJob { job_id });
            st.mark_completed(job_id);
            return JobCancelOutcome::CancelledBeforeExecution;
        }
        // Active (popped, worker-owned): flag for the pre-execution check.
        // If the worker already passed it, it completes normally as the
        // single owner; the caller's grace wait decides what is observed.
        if st.active.contains_key(&job_id) {
            st.cancelled_jobs.insert(job_id);
            return JobCancelOutcome::CancellationRequested;
        }
        if st.is_terminal(job_id) {
            return JobCancelOutcome::AlreadyTerminal;
        }
        JobCancelOutcome::UnknownJob
    }

    fn job_cancelled(st: &SchedulerState, job_id: u64) -> bool {
        st.cancelled_jobs.contains(&job_id)
    }

    // -- TODO 7/26: generation versioning --------------------------------

    fn generation_locked(st: &SchedulerState, canonical: &Path) -> u64 {
        st.generations.get(canonical).copied().unwrap_or(0)
    }

    /// Write/delete hook: bump the generation and drop cached bytes.
    /// Must be called on every successful write (both arms) and on
    /// deletes. Uses the stable identity (no filesystem access), so it
    /// works for deleted or not-yet-existing targets, and any later
    /// recreation under the same path starts at a newer generation that
    /// old snapshots can never satisfy.
    pub fn notify_write(&self, path: &Path) {
        let key = stable_key(path);
        let mut st = self.lock();
        let generation = st.generations.get(&key).copied().unwrap_or(0) + 1;
        st.generations.insert(key.clone(), generation);
        st.cache_remove(&key);
        // This path grows generations without queue involvement.
        st.retire_generations();
    }

    // -- TODO 8/23: promotion ---------------------------------------------

    /// Authorization-aware cache promotion: enforces the SAME
    /// `path_allowed()` sandbox gate as explicit reads, then serves
    /// validated entries. This is the ONLY public cache-read path —
    /// the raw lookup below is private so a future refactor cannot
    /// accidentally turn the cache into an authorization bypass.
    /// Sandbox logic lives in exactly one place (`path_allowed`); this
    /// wrapper only CALLS it, never duplicates it.
    pub fn serve_authorized(&self, path: &Path) -> Option<CacheHit> {
        // ONE permission sample spans gate → open → serve: a concurrent
        // set_folder_scope() cannot change the rules mid-flight (a
        // CurrentDir→AllDirs flip between two samples could otherwise
        // downgrade the object check after the name gate passed).
        let scope = crate::agent::get_tool_permissions().folder_scope;
        if crate::agent::path_allowed_with(path, scope).is_err() {
            return None;
        }
        self.serve_cached(path, scope)
    }

    /// Raw validated lookup. Private: requires the caller to have
    /// authorized `path` under `scope` (see `serve_authorized`). The scope
    /// travels as a parameter — never re-sampled — so the sandbox root
    /// below cannot disagree with the gate above.
    ///
    /// Object-bound: freshness is validated against the metadata of the
    /// securely-opened HANDLE (`secure_open_read` + `File::metadata`),
    /// never against a pathname re-resolution. A symlink swapped between
    /// the name gate and this lookup cannot redirect validation (or the
    /// served bytes, which come from the cache entry bound to this path).
    /// No content is re-read on a hit — only open, verify, fstat.
    fn serve_cached(&self, path: &Path, scope: crate::agent::FolderScope) -> Option<CacheHit> {
        let canonical = stable_key(path);
        // Snapshot the entry under lock; ALL validation happens at the
        // single final point below (after the unlocked filesystem work),
        // so there is exactly one place where staleness defeats a serve.
        let snapshot = self.lock().cache.entries.get(&canonical).map(|entry| {
            (
                entry.content.clone(),
                entry.origin,
                entry.generation,
                entry.mtime,
                entry.len,
                entry.fetched_at,
                entry.usefulness_counted,
            )
        });
        let Some((
            content,
            cache_origin,
            entry_generation,
            mtime,
            len,
            fetched_at,
            usefulness_counted,
        )) = snapshot
        else {
            return None;
        };
        // Test-only interleaving hook (see arm_serve_interleave): pause
        // AFTER the snapshot, BEFORE any filesystem work, so a test can
        // deterministically land a racing write in this exact window.
        #[cfg(test)]
        {
            let (pause, resume) = (
                SERVE_PAUSE.lock().unwrap().clone(),
                SERVE_RESUME.lock().unwrap().clone(),
            );
            if let (Some(p), Some(r)) = (pause, resume) {
                p.wait();
                r.wait();
            }
        }
        // TTL is pure (no filesystem, no generation read): reject expired
        // snapshots before opening any handle.
        if fetched_at.elapsed() > CACHE_TTL {
            let mut st = self.lock();
            st.cache_remove(&canonical);
            st.metrics.cache_misses += 1;
            return None;
        }
        // Object-bound freshness: open + verify the actual object, then
        // fstat the HANDLE. Scope-aware root from the caller's single
        // sample: AllDirs passes None (unrestricted); CurrentDir with an
        // unresolvable cwd misses fail-closed rather than degrading to
        // unrestricted.
        let sandbox_root = match scope {
            crate::agent::FolderScope::AllDirs => None,
            crate::agent::FolderScope::CurrentDir => match Self::current_sandbox_root() {
                Some(root) => Some(root),
                None => {
                    let mut st = self.lock();
                    st.cache_remove(&canonical);
                    st.metrics.cache_misses += 1;
                    return None;
                }
            },
        };
        let object_meta =
            match crate::secure_fs::secure_open_read(&canonical, sandbox_root.as_deref()) {
                Ok(sec) => std::fs::File::metadata(&sec.file).ok(),
                Err(_) => None,
            };
        let fs_fresh = object_meta.and_then(|m| m.modified().ok().map(|mt| (mt, m.len())));
        let mut st = self.lock();
        // FINAL validation point (locked): the filesystem work above ran
        // unlocked, so a Hercules write may have bumped the generation
        // and removed this entry in the meantime. Re-read BOTH the live
        // generation and the live entry: serving the snapshot now would
        // shadow that write with older bytes. No giant filesystem lock —
        // the guarantee is only that a write *before this point* defeats
        // the serve (external writers remain best-effort, as documented).
        let current_generation = Self::generation_locked(&st, &canonical);
        let entry_still_ours = st
            .cache
            .entries
            .get(&canonical)
            .map(|e| e.generation == entry_generation)
            .unwrap_or(false);
        if fs_fresh != Some((mtime, len))
            || current_generation != entry_generation
            || !entry_still_ours
        {
            // Drop only our own stale entry: a generation mismatch with a
            // live entry present means someone else already re-warmed the
            // key — leave their fresh data alone.
            if entry_still_ours {
                st.cache_remove(&canonical);
            }
            st.metrics.cache_misses += 1;
            return None;
        }
        let origin = match cache_origin {
            CacheOrigin::ExplicitRead => ReadOrigin::Explicit,
            CacheOrigin::SpeculativeRead => ReadOrigin::Speculative,
        };
        let hit = CacheHit {
            content,
            origin,
            generation: entry_generation,
        };
        st.metrics.cache_hits += 1;
        if cache_origin == CacheOrigin::SpeculativeRead {
            st.metrics.speculative_hits += 1;
            if !usefulness_counted {
                if let Some(e) = st.cache.entries.get_mut(&canonical) {
                    e.usefulness_counted = true;
                }
                st.metrics.useful_prefetches += 1;
            }
        }
        st.cache.note_served(canonical, hit.origin);
        Some(hit)
    }

    /// Current sandbox root for object-bound verification: the
    /// canonicalized cwd under CurrentDir scope, `None` under AllDirs.
    /// Callers needing fail-closed behavior on resolution failure must
    /// check the folder scope first (see `read_stable_file`); `serve`
    /// treats `None` as AllDirs, matching its pre-existing contract.
    fn current_sandbox_root() -> Option<PathBuf> {
        std::env::current_dir()
            .ok()
            .and_then(|cwd| cwd.canonicalize().ok())
    }

    // -- submission -------------------------------------------------------

    /// Submit speculative candidates. Dedupes, confidence-filters, budgets
    /// and enqueues; returns the number actually queued.
    ///
    /// Hard invariant: every filesystem path stored in scheduler state
    /// is a stable identity. Candidates are stabilized BEFORE
    /// dedupe/cache/generation lookups, so `./x.rs` and `/abs/x.rs` can
    /// never become two jobs (or two different generations) for one file.
    pub fn submit_speculative(
        &'static self,
        run_id: u64,
        candidates: Vec<RetrievalCandidate>,
    ) -> usize {
        self.ensure_workers();
        let mut st = self.lock();
        if Self::run_cancelled(&st, run_id) {
            return 0;
        }
        let mut queued = 0;
        let mut dropped = 0u64;
        let cap = candidates.len().min(self.budget.max_candidates_per_event);
        // Truncated over-cap candidates are budget drops too: the
        // predictor emitted more than one event may consume.
        dropped += (candidates.len() - cap) as u64;
        for cand in candidates.into_iter().take(cap) {
            let Some(priority) = self.config.priority_for(cand.confidence) else {
                continue; // below background threshold → skip
            };
            // Stable identity BEFORE any scheduler state: dedupe keys,
            // cache probes, generation stamps and job targets must all
            // agree. Never skipped: a file vanishing between prediction
            // and submit simply fails fast in the worker's metadata check.
            let canonical = stable_key(&cand.path);
            let kind = IoJobKind::SpeculativeRead;
            let key = (kind, canonical.clone());
            if st.inflight_speculative.contains(&key) {
                continue;
            }
            if st.cache.entries.contains_key(&canonical) {
                continue; // already warm
            }
            let use_entry = st.budgets.entry(run_id).or_default();
            if use_entry.jobs >= self.budget.max_jobs_per_run
                || use_entry.bytes + cand.estimated_bytes > self.budget.max_estimated_bytes_per_run
            {
                dropped += 1;
                continue;
            }
            use_entry.jobs += 1;
            use_entry.bytes += cand.estimated_bytes;
            let generation = Self::generation_locked(&st, &canonical);
            st.metrics.submitted += 1;
            let seq = st.queue_seq;
            st.queue_seq += 1;
            let epoch = st.epoch;
            st.queue.push(QueuedJob {
                job: IoJob {
                    id: next_job_id(),
                    run_id,
                    priority,
                    kind,
                    target: canonical.clone(),
                    generation: generation,
                    reason: cand.reason,
                    epoch,
                    action: None,
                    reply: None,
                },
                seq,
            });
            st.inflight_speculative.insert(key);
            queued += 1;
        }
        if dropped > 0 {
            st.metrics.budget_dropped += dropped;
            SchedulerState::push_event(&mut st, SchedulerEvent::BudgetDropped { count: dropped });
        }
        // Opportunistic lifecycle sweeps (no-ops below their caps).
        st.retire_generations();
        st.retire_cancelled_state();
        drop(st);
        for _ in 0..queued {
            self.kick();
        }
        queued
    }

    /// Submit an explicit batch: already-claimed model Read/Ls actions
    /// sharing ONE queue (and worker pool) with speculation at
    /// [`IoPriority::Explicit`]. Crate-visible precisely so the ONLY
    /// caller is the canonical accept path (claim → submit → execute):
    /// the scheduler must never become a second dispatch authority, so
    /// this cannot be reached from outside the crate. The claim happened
    /// BEFORE submit through `App::claim_tool_call`; workers execute via
    /// the canonical `execute_proposed` and deliver outcomes bound to
    /// `call_id`.
    /// Returns receipts in submission order; every receipt yields EXACTLY
    /// one outcome (success, error, cancellation, or worker panic mapped
    /// to an error) so callers can collect deterministically.
    /// Only Read/Ls actions are accepted; anything else is skipped.
    pub(crate) fn submit_explicit_batch(
        &'static self,
        run_id: u64,
        actions: Vec<crate::agent::ProposedAction>,
    ) -> Vec<ExplicitReceipt> {
        use crate::agent::ProposedKind;
        self.ensure_workers();
        let mut st = self.lock();
        let cancelled = Self::run_cancelled(&st, run_id);
        let mut receipts = Vec::with_capacity(actions.len());
        for action in actions {
            let kind = match action.kind {
                ProposedKind::Read => IoJobKind::ExplicitRead,
                ProposedKind::Ls => IoJobKind::ExplicitLs,
                _ => continue,
            };
            let (tx, rx) = std::sync::mpsc::channel();
            let job_id = next_job_id();
            let receipt = ExplicitReceipt {
                call_id: action.call_id,
                job_id,
                outcome: rx,
            };
            if cancelled {
                st.metrics.cancelled += 1;
                let _ = tx.send(ExplicitReadOutcome {
                    call_id: action.call_id,
                    result: "Error: scheduler: run cancelled before read executed".to_string(),
                    served_from_cache: false,
                });
                st.mark_completed(job_id);
                receipts.push(receipt);
                continue;
            }
            let target = crate::agent::AgentEngine::expand_path(&action.target);
            let generation = st
                .generations
                .get(&stable_key(&target))
                .copied()
                .unwrap_or(0);
            let seq = st.queue_seq;
            st.queue_seq += 1;
            let epoch = st.epoch;
            st.queue.push(QueuedJob {
                job: IoJob {
                    id: job_id,
                    run_id,
                    priority: IoPriority::Explicit,
                    kind,
                    target,
                    generation,
                    reason: RetrievalReason::ModelIntent,
                    epoch,
                    action: Some(action),
                    reply: Some(tx),
                },
                seq,
            });
            receipts.push(receipt);
        }
        // Opportunistic lifecycle sweeps (no-ops below their caps).
        st.retire_generations();
        st.retire_cancelled_state();
        drop(st);
        if !receipts.is_empty() {
            self.kick();
        }
        receipts
    }

    /// Explicit-read feedback (TODO 8 in implementation order): record the
    /// read for recency, run the predictor over the result, and submit.
    /// Errors never speculate.
    pub fn notify_explicit_read(&'static self, run_id: u64, path: &Path, content: &str) {
        if content.trim_start().starts_with("Error:") {
            return;
        }
        let canonical = stable_key(path);
        {
            let mut st = self.lock();
            st.recent_reads.push_back(canonical.clone());
            while st.recent_reads.len() > 32 {
                st.recent_reads.pop_front();
            }
        }
        let candidates = crate::prefetch::suggest_candidates(path, content, &self.config);
        self.submit_speculative(run_id, candidates);
    }

    /// Throttled streaming-intent hook: called every streaming tick; only
    /// predicts when the stream grew materially since the last prediction.
    /// Prose-derived candidates are low-confidence, read-only, budgeted —
    /// thinking text is an intent *signal*, never a ToolCall.
    pub fn note_stream_progress(&'static self, run_id: u64, stream: &str) {
        let dominated = {
            let st = self.lock();
            st.stream_marks.get(&run_id).copied().unwrap_or(0)
        };
        if stream.len() < dominated + STREAM_PREDICT_DELTA {
            return;
        }
        {
            let mut st = self.lock();
            st.stream_marks.insert(run_id, stream.len());
        }
        let cands = predict_from_prose(stream);
        self.submit_speculative(run_id, cands);
    }

    // -- worker pool ------------------------------------------------------

    fn pop_eligible(&self) -> Option<IoJob> {
        let mut st = self.lock();
        // Pop until an eligible job or empty. The heap orders Explicit
        // ahead of Speculative, so model-requested reads genuinely outrank
        // predictions in the ONE shared queue.
        //
        // OWNERSHIP: queue removal and `active` registration happen under
        // this ONE lock hold — a popped job is worker-owned with no
        // intermediate unowned gap, so `cancel_job` can never observe
        // "neither queued nor active". Dropped jobs are marked terminal
        // (explicit ones complete via their reply channel first).
        loop {
            let q = st.queue.pop()?;
            if q.job.epoch != st.epoch
                || Self::run_cancelled(&st, q.job.run_id)
                || Self::job_cancelled(&st, q.job.id)
            {
                st.inflight_speculative
                    .remove(&(q.job.kind, q.job.target.clone()));
                st.metrics.cancelled += 1;
                if let Some(reply) = &q.job.reply {
                    let _ = reply.send(ExplicitReadOutcome {
                        call_id: q.job.action.as_ref().map(|a| a.call_id).unwrap_or(0),
                        result: "Error: scheduler: run cancelled before read executed".to_string(),
                        served_from_cache: false,
                    });
                }
                SchedulerState::push_event(
                    &mut st,
                    SchedulerEvent::CancelledJob { job_id: q.job.id },
                );
                st.mark_completed(q.job.id);
                continue;
            }
            st.active.insert(
                q.job.id,
                ActiveJob {
                    job_id: q.job.id,
                    kind: q.job.kind,
                    target: q.job.target.clone(),
                    run_id: q.job.run_id,
                    started: Instant::now(),
                },
            );
            return Some(q.job);
        }
    }

    fn worker_loop(&'static self) {
        loop {
            let rx = self.receiver.lock().unwrap();
            let got = rx.recv();
            drop(rx);
            if got.is_err() {
                return; // scheduler dropped (tests) — exit thread
            }
            loop {
                let Some(job) = self.pop_eligible() else {
                    break;
                };
                // A panicking job must never kill the pool thread (or wed
                // an explicit caller on its reply channel): convert to an
                // error outcome and keep the worker alive.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.execute_job(job.clone())
                }));
                if result.is_err() {
                    self.finish_explicit(
                        &job,
                        ExplicitReadOutcome {
                            call_id: job.action.as_ref().map(|a| a.call_id).unwrap_or(0),
                            result: "Error: scheduler worker panicked".to_string(),
                            served_from_cache: false,
                        },
                    );
                    self.lock().metrics.cancelled += 1;
                }
            }
        }
    }

    /// Dispatch by kind. Exactly-once note: neither path touches
    /// `ToolDispatchRegistry` — speculative jobs resolve into the cache,
    /// explicit jobs were claimed BEFORE submit by the canonical gate.
    fn execute_job(&'static self, job: IoJob) {
        // Ownership was registered atomically with the queue pop; this
        // guard only RELEASES it, on every exit path (success, error,
        // cancellation, panic unwind) via the terminal transition.
        let _active = ActiveOwnershipGuard {
            sched: self,
            job_id: job.id,
        };
        match job.kind {
            IoJobKind::SpeculativeRead => self.execute_speculative(job),
            IoJobKind::SpeculativeLs => {
                // Never submitted (the predictor emits files only); Ls is
                // scheduled for ordering but always executes canonically —
                // see execute_explicit. Defensive no-op.
            }
            IoJobKind::ExplicitRead | IoJobKind::ExplicitLs => self.execute_explicit(job),
            IoJobKind::SemanticRetrieve | IoJobKind::CodeGraphRetrieve => {
                // Reserved stages: no executor yet — complete explicitly so
                // nothing wedges, counted as cancelled (never submitted
                // today; defensive only).
                self.finish_explicit(
                    &job,
                    ExplicitReadOutcome {
                        call_id: job.action.as_ref().map(|a| a.call_id).unwrap_or(0),
                        result: "Error: scheduler: retrieval stage not implemented".to_string(),
                        served_from_cache: false,
                    },
                );
                self.lock().metrics.cancelled += 1;
            }
        }
    }

    /// Deliver an explicit outcome exactly once and release the dedupe key.
    fn finish_explicit(&self, job: &IoJob, outcome: ExplicitReadOutcome) {
        if let Some(reply) = &job.reply {
            let _ = reply.send(outcome);
        }
    }

    /// Execute one EXPLICIT job: promotion first (validated cache serve),
    /// else the canonical executor. Authority never moves: the call was
    /// claimed before submit, and `execute_proposed` performs the read
    /// with all its permission/sandbox gates. Always delivers an outcome.
    fn execute_explicit(&'static self, job: IoJob) {
        let Some(action) = job.action.clone() else {
            self.finish_explicit(
                &job,
                ExplicitReadOutcome {
                    call_id: 0,
                    result: "Error: scheduler: explicit job without action".to_string(),
                    served_from_cache: false,
                },
            );
            return;
        };
        let cancelled = {
            let st = self.lock();
            Self::run_cancelled(&st, job.run_id) || Self::job_cancelled(&st, job.id)
        };
        if cancelled {
            self.lock().metrics.cancelled += 1;
            self.finish_explicit(
                &job,
                ExplicitReadOutcome {
                    call_id: action.call_id,
                    result: "Error: scheduler: run cancelled before read executed".to_string(),
                    served_from_cache: false,
                },
            );
            return;
        }
        // Promotion: validated serve (generation + mtime + length + TTL).
        // Full-file READS only, deliberately:
        // - ranged slices must never be cached as whole files, so ranged
        //   requests go straight to the canonical executor (which slices
        //   correctly);
        // - directory listings are never cached either: Ls jobs flow
        //   through the scheduler for queue ordering and fairness, but a
        //   local readdir is sub-millisecond, so a listing cache would buy
        //   no measurable latency while risking stale entry sizes (dir
        //   mtime does not move on file content changes). Ls always
        //   executes canonically.
        // Sandbox is enforced INSIDE serve_authorized (the single
        // authorization-aware cache path) — no gate is duplicated here.
        let expanded = crate::agent::AgentEngine::expand_path(&action.target);
        if matches!(action.kind, crate::agent::ProposedKind::Read) && action.line_attr.is_none() {
            if let Some(hit) = self.serve_authorized(&expanded) {
                if hit.origin == ReadOrigin::Speculative {
                    let mut st = self.lock();
                    SchedulerState::push_event(
                        &mut st,
                        SchedulerEvent::Promoted {
                            path: expanded.clone(),
                            origin: hit.origin,
                        },
                    );
                }
                // Same canonical bookkeeping as a disk read: the agent saw
                // these bytes, so the SmartSystem snapshot and the
                // speculation kickoff apply identically.
                crate::smart_system::get_smart_system().register_read(
                    crate::smart_system::AgentId::H0,
                    &expanded,
                    &hit.content,
                );
                self.notify_explicit_read(job.run_id, &expanded, &hit.content);
                self.finish_explicit(
                    &job,
                    ExplicitReadOutcome {
                        call_id: action.call_id,
                        result: hit.content,
                        served_from_cache: true,
                    },
                );
                return;
            }
        }
        let is_full_file_read =
            matches!(action.kind, crate::agent::ProposedKind::Read) && action.line_attr.is_none();
        if is_full_file_read {
            // The read itself goes through the ONE unified stable-read
            // primitive: bytes, metadata and generation come from a single
            // validated read, so the cached entry can never mix content
            // from one revision with metadata from another (the
            // read-then-stat race of post-hoc warming). This IS the
            // canonical read implementation — same sandbox gate the
            // executor enforces — plus the canonical bookkeeping below
            // (SmartSystem snapshot, run-scoped speculation kickoff).
            match self.read_through(&expanded) {
                Ok(content) => {
                    crate::smart_system::get_smart_system().register_read(
                        crate::smart_system::AgentId::H0,
                        &expanded,
                        &content,
                    );
                    self.notify_explicit_read(job.run_id, &expanded, &content);
                    self.finish_explicit(
                        &job,
                        ExplicitReadOutcome {
                            call_id: action.call_id,
                            result: content,
                            served_from_cache: false,
                        },
                    );
                }
                Err(e) => {
                    let result = crate::agent::AgentEngine::stable_read_error(&expanded, e);
                    self.finish_explicit(
                        &job,
                        ExplicitReadOutcome {
                            call_id: action.call_id,
                            result,
                            served_from_cache: false,
                        },
                    );
                }
            }
            return;
        }
        // Ranged reads and listings go through the canonical executor
        // (its internal full-file reads warm via the same primitive).
        let result = crate::agent::AgentEngine::execute_proposed(&action);
        self.finish_explicit(
            &job,
            ExplicitReadOutcome {
                call_id: action.call_id,
                result,
                served_from_cache: false,
            },
        );
    }

    /// Execute one speculative job: sandbox gate → stat → generation-stamped
    /// read → re-validate generation + cancellation → insert. Read-only.
    fn execute_speculative(&'static self, job: IoJob) {
        struct InflightGuard {
            sched: &'static AgentIoScheduler,
            key: (IoJobKind, PathBuf),
        }
        impl Drop for InflightGuard {
            fn drop(&mut self) {
                self.sched.lock().inflight_speculative.remove(&self.key);
            }
        }
        let _guard = InflightGuard {
            sched: self,
            key: (job.kind, job.target.clone()),
        };

        {
            let st = self.lock();
            if Self::run_cancelled(&st, job.run_id) || Self::job_cancelled(&st, job.id) {
                drop(st);
                self.lock().metrics.cancelled += 1;
                return;
            }
        }
        // ONE unified stable-read primitive (sandbox → stable identity →
        // generation/metadata before → read → metadata/generation after).
        // The expected generation is the submit-time stamp: a mismatch
        // means a write landed before the worker ran.
        match self.read_stable_file(&job.target, Some(job.generation)) {
            Err(StableReadError::SandboxDenied(_)) => {
                let mut st = self.lock();
                st.metrics.rejected_sandbox += 1;
                SchedulerState::push_event(
                    &mut st,
                    SchedulerEvent::SandboxRejected {
                        path: job.target.clone(),
                    },
                );
            }
            Err(_) => {
                self.lock().metrics.cancelled += 1;
            }
            Ok(stable) => {
                let mut st = self.lock();
                if Self::run_cancelled(&st, job.run_id) {
                    st.metrics.cancelled += 1;
                    return;
                }
                // Test-epoch guard: a job popped before a test reset must
                // not mutate the new epoch's state.
                if job.epoch != st.epoch {
                    st.metrics.cancelled += 1;
                    return;
                }
                // Insert gate: refuse when a Hercules write landed after
                // the stable read validated (generation moved past the
                // stamped read). Same rule as read_through: never install
                // state known-obsolete at insert time.
                if Self::generation_locked(&st, &stable_key(&job.target)) != stable.generation {
                    st.metrics.cancelled += 1;
                    return;
                }
                st.metrics.completed += 1;
                st.metrics.bytes_read += stable.content.len() as u64;
                st.cache_put(
                    stable_key(&job.target),
                    CacheEntry {
                        content: stable.content,
                        generation: stable.generation,
                        mtime: stable.mtime,
                        len: stable.len,
                        fetched_at: Instant::now(),
                        origin: CacheOrigin::SpeculativeRead,
                        usefulness_counted: false,
                    },
                );
            }
        }
    }

    // -- observability -----------------------------------------------------

    pub fn metrics(&self) -> PrefetchMetrics {
        self.lock().metrics.clone()
    }

    /// Human-readable scheduler state for a future `/scheduler` diagnostic.
    pub fn diagnostics(&self) -> String {
        let st = self.lock();
        let m = &st.metrics;
        let mut running: Vec<String> = Vec::new();
        for (_, a) in st.active.iter() {
            running.push(format!(
                "  #{} {:?} {} (run {})",
                a.job_id,
                a.kind,
                a.target.display(),
                a.run_id
            ));
        }
        let mut queued: Vec<String> = Vec::new();
        for q in st.queue.iter() {
            queued.push(format!(
                "  #{} {:?} {} ({}, {:?})",
                q.job.id,
                q.job.kind,
                q.job.target.display(),
                q.job.reason.label(),
                q.job.priority
            ));
        }
        let recent: Vec<String> = st
            .events
            .iter()
            .rev()
            .take(12)
            .map(|e| format!("  {}", e.label()))
            .collect();
        format!(
            "Agent I/O Scheduler\n\nRunning: {}\n{}\nQueued: {}\n{}\nRecent events:\n{}\nCache: {} entries, {} bytes\n\nPerformance:\n  submitted: {}\n  completed: {}\n  cancelled: {}\n  sandbox-rejected: {}\n  budget-dropped: {}\n  cache hits: {} (speculative: {})\n  hit rate: {:.1}%\n  useful prefetches: {}\n  usefulness: {:.1}%\n  wasted: {}\n  bytes read: {}",
            running.len(),
            running.join("\n"),
            queued.len(),
            queued.join("\n"),
            if recent.is_empty() {
                "  (none)".to_string()
            } else {
                recent.join("\n")
            },
            st.cache.entries.len(),
            st.cache.total_bytes,
            m.submitted,
            m.completed,
            m.cancelled,
            m.rejected_sandbox,
            m.budget_dropped,
            m.cache_hits,
            m.speculative_hits,
            m.hit_rate() * 100.0,
            m.useful_prefetches,
            m.usefulness() * 100.0,
            m.wasted_prefetches,
            m.bytes_read,
        )
    }

    /// Last-serve provenance: which origin served a path most recently.
    pub fn last_serve_origin(&self, path: &Path) -> Option<(ReadOrigin, Instant)> {
        let key = stable_key(path);
        self.lock().cache.last_served(&key)
    }

    #[cfg(test)]
    pub fn reset_for_test(&self) {
        let mut st = self.lock();
        st.queue.clear();
        st.inflight_speculative.clear();
        st.active.clear();
        st.completed_jobs.clear();
        st.cancelled_runs.clear();
        st.cancelled_jobs.clear();
        st.epoch += 1;
        st.run_tokens.clear();
        st.generations.clear();
        st.cache.entries.clear();
        st.cache.total_bytes = 0;
        st.cache.last_serve.clear();
        st.metrics = PrefetchMetrics::default();
        st.budgets.clear();
        st.stream_marks.clear();
        st.recent_reads.clear();
    }
}

impl Default for AgentIoScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Stable scheduler/cache identity for a filesystem path.
///
/// Pipeline: expand `$CURRENT`-style aliases → anchor relative paths at
/// the workspace root (process cwd) → lexically normalize `.`/`..`.
///
/// This NEVER touches the filesystem, so invalidation works for deleted
/// or not-yet-existing targets (where `canonicalize` would fail). It is an
/// IDENTITY function, not authorization (the sandbox gate still runs
/// separately) and not symlink resolution (two links to one file yield
/// two identities, each self-consistent via its own mtime validation).
/// Generations, cache keys, dedupe keys and invalidation must ALL use
/// this — never mix stable and canonical keys for one file.
pub fn stable_key(path: &Path) -> PathBuf {
    let expanded = crate::agent::AgentEngine::expand_path(&path.to_string_lossy());
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(expanded)
    };
    lexical_clean(&absolute)
}

/// Lexical `.`/`..` normalization without filesystem access.
/// Lexical `.`/`..` normalization without filesystem access. Shared
/// with `secure_fs` so both layers agree on path identity.
pub(crate) fn lexical_clean(path: &Path) -> PathBuf {
    use std::path::Component::*;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            CurDir => {}
            ParentDir => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

// ---------------------------------------------------------------------------
// Unified stable file read: the ONE cache-correctness implementation.
// Every population path (speculative worker, explicit warm-up,
// synchronous fetch) goes through this — never a second ad-hoc
// metadata/read/insert sequence.
// ---------------------------------------------------------------------------

/// Bytes validated stable across the whole read.
#[derive(Debug)]
pub struct StableRead {
    pub content: String,
    pub mtime: SystemTime,
    pub len: u64,
    pub generation: u64,
}

/// Why a stable read failed. Callers map these to user-facing errors;
/// none of them execute anything.
#[derive(Debug)]
pub enum StableReadError {
    SandboxDenied(String),
    NotFound,
    NotAFile,
    TooLarge,
    /// The open itself failed past existence checks, the capped read
    /// failed, or bytes were not valid UTF-8.
    OpenFailed,
    InvalidUtf8,
    /// mtime or length moved between the before/after snapshots: an
    /// external writer raced the read.
    UnstableReread,
    /// A canonical generation bumped between stamp and insert (or did not
    /// match the expected generation): our own write raced the read.
    GenerationChanged,
}

impl AgentIoScheduler {
    /// Read a file with full cache-correctness protocol in one place:
    /// 1. sandbox gate (`path_allowed`, same choke point as explicit reads)
    /// 2. stable identity (no existence requirement for the KEY)
    /// 3. generation stamp before
    /// 4. metadata snapshot before (existence, file kind, size cap)
    /// 5. read bytes
    /// 6. metadata snapshot after
    /// 7. reject unless mtime+length are identical across the read
    /// 8. reject unless generation is unchanged (and equals `expected`
    ///    when given — the speculative race guard)
    /// 9. return bytes + final metadata + generation for insertion.
    pub fn read_stable_file(
        &self,
        raw: &Path,
        expected_generation: Option<u64>,
    ) -> Result<StableRead, StableReadError> {
        use crate::secure_fs::{SecureOpenError, secure_open_read};
        let key = stable_key(raw);
        // ONE permission sample spans gate → root → open: capturing the
        // scope once prevents set_folder_scope() from changing the rules
        // mid-flight (a CurrentDir→AllDirs flip between two samples could
        // otherwise downgrade object verification after the name gate).
        let scope = crate::agent::get_tool_permissions().folder_scope;
        // Name-level gate first (preserves exact user-facing messages).
        if let Err(e) = crate::agent::path_allowed_with(&key, scope) {
            return Err(StableReadError::SandboxDenied(e));
        }
        // Object-bound open: authorization attaches to the opened
        // filesystem object, not the name — a symlink swapped between the
        // gate above and the open cannot redirect the read outside the
        // sandbox (the opened object is verified, not the path).
        let sandbox_root = match scope {
            crate::agent::FolderScope::CurrentDir => {
                Some(Self::current_sandbox_root().ok_or_else(|| {
                    StableReadError::SandboxDenied(
                        "Safefolder: cannot resolve current dir".to_string(),
                    )
                })?)
            }
            crate::agent::FolderScope::AllDirs => None,
        };
        let mut sec = match secure_open_read(&key, sandbox_root.as_deref()) {
            Ok(sec) => sec,
            Err(SecureOpenError::NotFound) => return Err(StableReadError::NotFound),
            Err(SecureOpenError::SandboxDenied(msg)) => {
                return Err(StableReadError::SandboxDenied(msg));
            }
            Err(_) => return Err(StableReadError::OpenFailed),
        };
        let generation_before = self.lock().generations.get(&key).copied().unwrap_or(0);
        // fstat the HANDLE (object-anchored, never re-resolved by name).
        let meta_before =
            std::fs::File::metadata(&sec.file).map_err(|_| StableReadError::NotFound)?;
        if !meta_before.is_file() {
            return Err(StableReadError::NotAFile);
        }
        if meta_before.len() > MAX_FILE_BYTES {
            return Err(StableReadError::TooLarge);
        }
        let mtime_before = meta_before.modified().ok();
        // Capped read: a racing grow past the cap is refused, never loaded.
        let bytes = crate::secure_fs::read_capped(&mut sec.file, MAX_FILE_BYTES)
            .map_err(|_| StableReadError::OpenFailed)?;
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Err(StableReadError::TooLarge);
        }
        let content = String::from_utf8(bytes).map_err(|_| StableReadError::InvalidUtf8)?;
        let meta_after =
            std::fs::File::metadata(&sec.file).map_err(|_| StableReadError::UnstableReread)?;
        if meta_after.modified().ok() != mtime_before || meta_after.len() != meta_before.len() {
            return Err(StableReadError::UnstableReread);
        }
        let generation_after = self.lock().generations.get(&key).copied().unwrap_or(0);
        if generation_after != generation_before {
            return Err(StableReadError::GenerationChanged);
        }
        if let Some(expected) = expected_generation {
            if generation_after != expected {
                return Err(StableReadError::GenerationChanged);
            }
        }
        Ok(StableRead {
            content,
            mtime: meta_after.modified().unwrap_or_else(|_| SystemTime::now()),
            len: meta_after.len(),
            generation: generation_after,
        })
    }

    /// Synchronous read-through: stable read + insert with Explicit origin.
    /// Used by the canonical `execute_read` miss path and one-shot warm-ups.
    pub fn read_through(&self, raw: &Path) -> Result<String, StableReadError> {
        let key = stable_key(raw);
        let stable = self.read_stable_file(&key, None)?;
        let mut st = self.lock();
        // Insert gate: a Hercules write that landed after the stable read
        // validated must not install obsolete state. Refusal surfaces as
        // GenerationChanged (callers retry against fresh state).
        if Self::generation_locked(&st, &key) != stable.generation {
            st.metrics.cache_misses += 1;
            return Err(StableReadError::GenerationChanged);
        }
        st.cache_put(
            key,
            CacheEntry {
                content: stable.content.clone(),
                generation: stable.generation,
                mtime: stable.mtime,
                len: stable.len,
                fetched_at: Instant::now(),
                origin: CacheOrigin::ExplicitRead,
                usefulness_counted: true,
            },
        );
        Ok(stable.content)
    }
}

#[cfg(test)]
static TEST_SERIAL: Mutex<()> = Mutex::new(());

/// Serialize tests that share the global scheduler (cache, queue,
/// metrics, generations). Used across modules (`agent_io`, `prefetch`).
#[cfg(test)]
pub fn test_serial_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
static SERVE_PAUSE: Mutex<Option<std::sync::Arc<std::sync::Barrier>>> = Mutex::new(None);
#[cfg(test)]
static SERVE_RESUME: Mutex<Option<std::sync::Arc<std::sync::Barrier>>> = Mutex::new(None);

/// Arm a two-phase rendezvous inside `serve_cached`, immediately after
/// the entry snapshot and before any filesystem work. The serving thread
/// blocks on `pause` (letting the test perform a racing write), then on
/// `resume`. Test-only deterministic interleaving; never armed in
/// production (both statics stay `None`).
#[cfg(test)]
pub(crate) fn arm_serve_interleave(
    pause: std::sync::Arc<std::sync::Barrier>,
    resume: std::sync::Arc<std::sync::Barrier>,
) {
    *SERVE_PAUSE.lock().unwrap() = Some(pause);
    *SERVE_RESUME.lock().unwrap() = Some(resume);
}

#[cfg(test)]
pub(crate) fn disarm_serve_interleave() {
    *SERVE_PAUSE.lock().unwrap() = None;
    *SERVE_RESUME.lock().unwrap() = None;
}

// ---------------------------------------------------------------------------
// TODO 12/16 (implementation order) — deterministic intent predictor
// ---------------------------------------------------------------------------

/// Deterministic intent prediction over already-read content. Heuristics
/// only (explicit paths, imports, recency) — no ML model. Returns capped,
/// confidence-scored candidates for [`AgentIoScheduler::submit_speculative`].
pub struct IntentPredictor;

impl IntentPredictor {
    pub fn predict_from_read(
        read_path: &Path,
        content: &str,
        config: &RetrievalConfig,
    ) -> Vec<RetrievalCandidate> {
        crate::prefetch::suggest_candidates(read_path, content, config)
    }

    pub fn predict_from_prose(stream: &str) -> Vec<RetrievalCandidate> {
        predict_from_prose(stream)
    }
}

/// Prose/streaming intent signal: quoted path-like tokens that resolve to
/// existing files under the cwd. Low confidence, read-only, budgeted.
/// Thinking/streaming text NEVER becomes a ToolCall — this only warms the
/// cache for files the model may explicitly request next.
fn predict_from_prose(stream: &str) -> Vec<RetrievalCandidate> {
    let tail = stream.len().saturating_sub(16 * 1024);
    let window = &stream[tail..];
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for line in window.lines() {
        if out.len() >= MAX_STREAM_CANDIDATES {
            break;
        }
        for lit in crate::prefetch::quoted_literals(line) {
            if out.len() >= MAX_STREAM_CANDIDATES {
                break;
            }
            if lit.len() > 128 || lit.contains("://") {
                continue;
            }
            for cand in crate::prefetch::resolve_literal(&cwd, &lit) {
                if cand.is_file() && seen.insert(cand.clone()) {
                    let estimated_bytes = cand.metadata().map(|m| m.len()).unwrap_or(0);
                    out.push(RetrievalCandidate {
                        path: cand,
                        confidence: 0.30,
                        reason: RetrievalReason::ModelIntent,
                        estimated_bytes,
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);
    static RUN_SEQ: AtomicU64 = AtomicU64::new(100_000);

    fn fresh_run() -> u64 {
        RUN_SEQ.fetch_add(1, Ordering::SeqCst)
    }

    fn test_dir(tag: &str) -> PathBuf {
        let n = TEST_SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::current_dir().unwrap().join(format!(
            "target/hercules-agentio-test-{tag}-{n}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sched() -> &'static AgentIoScheduler {
        AgentIoScheduler::global()
    }

    fn wait_for<F>(mut cond: F, label: &str)
    where
        F: FnMut() -> bool,
    {
        for _ in 0..200 {
            if cond() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out waiting for {label}");
    }

    #[test]
    fn queue_discipline_explicit_before_speculative() {
        // Queue discipline (unit level): the heap backing the scheduler's
        // single shared queue pops Explicit before Background.
        let mut heap = BinaryHeap::new();
        heap.push(QueuedJob {
            job: IoJob {
                id: 1,
                run_id: 1,
                priority: IoPriority::Background,
                kind: IoJobKind::SpeculativeRead,
                target: PathBuf::from("b"),
                generation: 0,
                reason: RetrievalReason::SameDirectorySibling,
                epoch: 0,
                action: None,
                reply: None,
            },
            seq: 0,
        });
        heap.push(QueuedJob {
            job: IoJob {
                id: 2,
                run_id: 1,
                priority: IoPriority::Explicit,
                kind: IoJobKind::ExplicitRead,
                target: PathBuf::from("a"),
                generation: 0,
                reason: RetrievalReason::ModelIntent,
                epoch: 0,
                action: None,
                reply: None,
            },
            seq: 1,
        });
        assert_eq!(heap.pop().unwrap().job.priority, IoPriority::Explicit);
        assert_eq!(heap.pop().unwrap().job.priority, IoPriority::Background);
    }

    #[test]
    fn scheduler_queue_serves_explicit_first() {
        // Real scheduler queue (not an artificial heap): the scheduler's
        // own pop order puts Explicit first. Jobs are pushed WITHOUT a
        // kick so parked workers stay parked; if a stale wakeup lets a
        // worker steal the probe jobs, re-push and retry (each steal
        // consumes exactly one stale wakeup, so this converges).
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let epoch = s.lock().epoch;
        for _ in 0..100 {
            {
                let mut st = s.lock();
                st.queue.clear();
                st.queue.push(QueuedJob {
                    job: IoJob {
                        id: next_job_id(),
                        run_id: run,
                        priority: IoPriority::Background,
                        kind: IoJobKind::SpeculativeRead,
                        target: PathBuf::from("spec-file"),
                        generation: 0,
                        reason: RetrievalReason::SameDirectorySibling,
                        epoch,
                        action: None,
                        reply: None,
                    },
                    seq: 0,
                });
                st.queue.push(QueuedJob {
                    job: IoJob {
                        id: next_job_id(),
                        run_id: run,
                        priority: IoPriority::Explicit,
                        kind: IoJobKind::ExplicitRead,
                        target: PathBuf::from("explicit-file"),
                        generation: 0,
                        reason: RetrievalReason::ModelIntent,
                        epoch,
                        action: None,
                        reply: None,
                    },
                    seq: 1,
                });
            }
            let first = s.pop_eligible().map(|j| j.priority);
            let second = s.pop_eligible().map(|j| j.priority);
            match (first, second) {
                (Some(IoPriority::Explicit), Some(IoPriority::Background)) => {
                    s.reset_for_test();
                    return;
                }
                // A worker stole a probe job via a stale wakeup — retry.
                _ => continue,
            }
        }
        panic!("explicit job did not pop first after retries");
    }

    #[test]
    fn explicit_batch_ordered_outcomes_with_promotion() {
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("batch");
        let a = dir.join("a.rs");
        let b = dir.join("b.rs");
        std::fs::write(&a, "mod b;\n").unwrap();
        std::fs::write(&b, "pub fn b() {}\n").unwrap();

        // Warm B through speculation first.
        s.notify_explicit_read(run, &a, "mod b;\n");
        wait_for(
            || s.serve_authorized(&b).is_some(),
            "speculative fetch of b.rs",
        );

        // Explicit batch: already-claimed actions through the shared queue.
        let mk = |target: &Path| {
            ProposedAction::new(
                ProposedKind::Read,
                target.to_string_lossy().to_string(),
                String::new(),
                None,
                ToolCallSource::ModelCompletion,
            )
        };
        let receipts = s.submit_explicit_batch(run, vec![mk(&a), mk(&b)]);
        assert_eq!(receipts.len(), 2);
        let first = receipts[0].outcome.recv().expect("first outcome delivered");
        let second = receipts[1]
            .outcome
            .recv()
            .expect("second outcome delivered");
        assert_eq!(first.result, "mod b;\n");
        assert!(!first.served_from_cache, "A was never prefetched");
        assert_eq!(second.result, "pub fn b() {}\n");
        assert!(
            second.served_from_cache,
            "B promoted from speculative cache, no 2nd fs read"
        );

        // Missing file → Error outcome (App releases the claim: nothing ran).
        let missing = dir.join("nope.rs");
        let receipts = s.submit_explicit_batch(run, vec![mk(&missing)]);
        let out = receipts[0].outcome.recv().expect("error outcome delivered");
        assert!(
            out.result.trim_start().starts_with("Error:"),
            "failed reads surface errors, got: {}",
            out.result
        );
        assert!(!out.served_from_cache);
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn read_predict_cache_hit_promotion() {
        // Basic pipeline: read A → predicts B → B cached → serve B, no
        // second filesystem read, speculative provenance recorded.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("promo");
        let a = dir.join("a.rs");
        let b = dir.join("b.rs");
        std::fs::write(&a, "mod b;\n").unwrap();
        std::fs::write(&b, "pub fn b() {}\n").unwrap();

        // Explicit read of A through the unified primitive (reads,
        // validates and warms in one step, as the canonical executor
        // would serve it).
        let content = s.read_through(&a).expect("explicit read of A succeeds");
        s.notify_explicit_read(run, &a, &content);

        wait_for(
            || s.serve_authorized(&b).is_some(),
            "speculative fetch of b.rs to complete",
        );
        let hit = s.serve_authorized(&b).expect("promotion serves B");
        assert_eq!(hit.content, "pub fn b() {}\n");
        assert_eq!(hit.origin, ReadOrigin::Speculative);
        assert_eq!(
            s.last_serve_origin(&b).map(|(o, _)| o),
            Some(ReadOrigin::Speculative)
        );
        let m = s.metrics();
        assert!(m.cache_hits >= 1);
        assert_eq!(m.useful_prefetches, 1, "first serve counts useful once");
        // Second serve: hits again but usefulness counted once.
        assert!(s.serve_authorized(&b).is_some());
        assert_eq!(s.metrics().useful_prefetches, 1);
        // Unrelated file: miss.
        assert!(s.serve_authorized(&a.join("missing.rs")).is_none());
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn stale_write_race_never_poisons_cache() {
        // T1 submits speculation, T2 writes + invalidates, T1's worker
        // finishes after: old bytes must NOT enter the cache.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("race");
        let f = dir.join("f.txt");
        std::fs::write(&f, "v1\n").unwrap();

        // Capture generation 0, then simulate the write racing ahead.
        // Write FIRST (bumps generation to 1 + invalidates)...
        std::fs::write(&f, "v2-newer\n").unwrap();
        s.notify_write(&f);
        // ...then the stale submission (stamped with old generation 0)
        // executes: worker must refuse to insert. Epoch is current (only
        // the generation is stale) so the epoch guard does not interfere.
        let epoch = s.lock().epoch;
        {
            let mut st = s.lock();
            st.queue.push(QueuedJob {
                job: IoJob {
                    id: next_job_id(),
                    run_id: run,
                    priority: IoPriority::High,
                    kind: IoJobKind::SpeculativeRead,
                    target: f.clone(),
                    generation: 0, // stale stamp
                    reason: RetrievalReason::QuotedPath,
                    epoch,
                    action: None,
                    reply: None,
                },
                seq: 0,
            });
            st.inflight_speculative
                .insert((IoJobKind::SpeculativeRead, f.clone()));
            s.kick();
        }
        // Give the worker a chance to (incorrectly) insert.
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            s.serve_authorized(&f).is_none(),
            "stale speculative result must not be cached"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn run_cancellation_stops_speculation() {
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("cancel");
        let mut cands = Vec::new();
        for i in 0..10 {
            let p = dir.join(format!("f{i}.txt"));
            std::fs::write(&p, "x\n").unwrap();
            cands.push(RetrievalCandidate {
                path: p,
                confidence: 0.9,
                reason: RetrievalReason::QuotedPath,
                estimated_bytes: 2,
            });
        }
        s.cancel_run(run);
        assert_eq!(
            s.submit_speculative(run, cands),
            0,
            "cancelled run submits nothing"
        );
        assert_eq!(s.metrics().submitted, 0);
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn budgets_bound_submissions() {
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("budget");
        let mut cands = Vec::new();
        for i in 0..100 {
            let p = dir.join(format!("g{i}.txt"));
            std::fs::write(&p, "x\n").unwrap();
            cands.push(RetrievalCandidate {
                path: p,
                confidence: 0.9,
                reason: RetrievalReason::QuotedPath,
                estimated_bytes: 2,
            });
        }
        let queued = s.submit_speculative(run, cands);
        assert!(
            queued <= MAX_CANDIDATES_PER_EVENT,
            "per-event cap enforced, got {queued}"
        );
        assert!(s.metrics().budget_dropped > 0, "over-cap drops are counted");
        // Per-run job budget: shrink via repeated submits.
        s.reset_for_test();
        let mut total = 0;
        for batch in 0..20 {
            let mut cands = Vec::new();
            for i in 0..8 {
                let p = dir.join(format!("h{batch}-{i}.txt"));
                std::fs::write(&p, "x\n").unwrap();
                cands.push(RetrievalCandidate {
                    path: p,
                    confidence: 0.9,
                    reason: RetrievalReason::QuotedPath,
                    estimated_bytes: 2,
                });
            }
            total += s.submit_speculative(run, cands);
        }
        assert!(
            total <= s.budget.max_jobs_per_run,
            "per-run job budget enforced, got {total}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn sandbox_rejection_never_caches() {
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        // Unresolvable paths queue under their stable identity but fail
        // fast in the worker's metadata check: never read, never cached.
        let ghost = PathBuf::from("/definitely-not-hercules-agentio-xyz/file.txt");
        assert_eq!(
            s.submit_speculative(
                run,
                vec![RetrievalCandidate {
                    path: ghost.clone(),
                    confidence: 0.95,
                    reason: RetrievalReason::QuotedPath,
                    estimated_bytes: 10,
                }]
            ),
            1
        );
        wait_for(
            || s.serve_authorized(&ghost).is_none() && s.metrics().rejected_sandbox >= 1,
            "worker to refuse the outside-sandbox ghost path",
        );
        assert!(s.serve_authorized(&ghost).is_none());
        // An EXISTING file outside the current-dir safefolder (default
        // FolderScope) queues fine but the worker must refuse it: never
        // read, never cached.
        let outside_dir = std::env::temp_dir().join(format!(
            "hercules-agentio-outside-{}-{}",
            std::process::id(),
            fresh_run()
        ));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let outside = outside_dir.join("secret.txt");
        std::fs::write(&outside, "outside\n").unwrap();
        let queued = s.submit_speculative(
            run,
            vec![RetrievalCandidate {
                path: outside.clone(),
                confidence: 0.95,
                reason: RetrievalReason::QuotedPath,
                estimated_bytes: 10,
            }],
        );
        assert_eq!(queued, 1, "queued (sandbox checked at execution)");
        wait_for(
            || s.metrics().rejected_sandbox >= 1,
            "worker to reject outside-sandbox path",
        );
        assert!(s.serve_authorized(&outside).is_none());
        let _ = std::fs::remove_dir_all(&outside_dir);
        s.reset_for_test();
    }

    #[test]
    fn speculation_never_touches_dispatch_registry() {
        // Type-level + behavioral: speculative jobs carry no call_id and
        // the canonical registry is untouched by scheduler activity.
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource, ToolDispatchRegistry};
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("noreg");
        let a = dir.join("a.rs");
        let b = dir.join("b.rs");
        std::fs::write(&a, "mod b;\n").unwrap();
        std::fs::write(&b, "y\n").unwrap();

        let mut reg = ToolDispatchRegistry::new();
        let before = format!("{reg:?}");
        s.notify_explicit_read(run, &a, "mod b;\n");
        wait_for(
            || s.serve_authorized(&b).is_some(),
            "speculative fetch completes",
        );
        // Scheduler has no access to the registry: it is unchanged, and a
        // LATER explicit model read still claims exactly once.
        assert_eq!(format!("{reg:?}"), before);
        let action = ProposedAction::new(
            ProposedKind::Read,
            b.to_string_lossy().to_string(),
            String::new(),
            None,
            ToolCallSource::ModelCompletion,
        );
        assert!(reg.try_claim(&action), "explicit read claims freely");
        assert!(!reg.try_claim(&action), "exactly-once still holds");
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn explicit_ls_executes_canonically_without_cache() {
        // Ls is scheduled for ordering but never cached: the outcome must
        // equal a canonical listing, and no cache entry may exist for the
        // directory afterwards.
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("lsbatch");
        std::fs::write(dir.join("one.txt"), "1\n").unwrap();
        let action = ProposedAction::new(
            ProposedKind::Ls,
            dir.to_string_lossy().to_string(),
            String::new(),
            None,
            ToolCallSource::ModelCompletion,
        );
        let receipts = s.submit_explicit_batch(run, vec![action]);
        assert_eq!(receipts.len(), 1);
        let out = receipts[0].outcome.recv().expect("ls outcome delivered");
        assert!(
            out.result.contains("one.txt"),
            "canonical listing served, got: {}",
            out.result
        );
        assert!(
            !out.served_from_cache,
            "directory listings are never promoted"
        );
        assert!(
            s.serve_authorized(&dir).is_none(),
            "no directory cache entry may exist"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn cache_sharing_across_runs_is_intended() {
        // Files are not owned by runs: run A warms foo.rs, run B is served
        // from the same entry. Generations (bumped by every canonical
        // write) keep sharing safe, not run isolation.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run_a = fresh_run();
        let dir = test_dir("xrun");
        let main = dir.join("main.rs");
        let sib = dir.join("sib.rs");
        std::fs::write(&main, "mod sib;\n").unwrap();
        std::fs::write(&sib, "shared\n").unwrap();

        s.notify_explicit_read(run_a, &main, "mod sib;\n");
        wait_for(
            || s.serve_authorized(&sib).is_some(),
            "run A warms the cache",
        );
        // Entries are not run-owned: cancelling the source run must not
        // drop cached bytes (queued work is what gets cancelled).
        s.cancel_run(run_a);
        let hit = s
            .serve_authorized(&sib)
            .expect("later run served from earlier entry");
        assert_eq!(hit.content, "shared\n");

        // A canonical write under either run invalidates for both.
        std::fs::write(&sib, "shared v2, longer\n").unwrap();
        s.notify_write(&sib);
        assert!(
            s.serve_authorized(&sib).is_none(),
            "write invalidates across run boundaries"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn cancel_then_release_allows_reclaim() {
        // Cancellation contract end-to-end (registry level, no App
        // harness): claim → submit → cancel → error outcome → release →
        // the identical action claims again. Proves cancellation can never
        // wedge a logical call for the rest of the turn.
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource, ToolDispatchRegistry};
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("reclaim");
        let f = dir.join("r.txt");
        std::fs::write(&f, "data\n").unwrap();

        let mut reg = ToolDispatchRegistry::new();
        let action = ProposedAction::new(
            ProposedKind::Read,
            f.to_string_lossy().to_string(),
            String::new(),
            None,
            ToolCallSource::ModelCompletion,
        );
        assert!(reg.try_claim(&action), "first claim succeeds");
        // Cancel the run BEFORE submit: deterministic immediate error
        // outcome (no worker race involved).
        s.cancel_run(run);
        let receipts = s.submit_explicit_batch(run, vec![action.clone()]);
        assert_eq!(receipts.len(), 1);
        let out = receipts[0]
            .outcome
            .recv_timeout(Duration::from_secs(10))
            .expect("terminal outcome always delivered");
        assert!(
            out.result.trim_start().starts_with("Error:"),
            "cancelled batch surfaces an error, got: {}",
            out.result
        );
        // App releases failed claims (nothing with side effects ran)…
        reg.release(&action);
        // …so the identical logical action claims again.
        assert!(
            reg.try_claim(&action),
            "released claim may be claimed again"
        );
        assert!(!reg.try_claim(&action), "exactly-once still holds after");
        // The terminal outcome is observable as terminal: a late cancel
        // sees AlreadyTerminal, never UnknownJob.
        assert_eq!(
            s.cancel_job(receipts[0].job_id),
            JobCancelOutcome::AlreadyTerminal
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn cancelled_job_delivers_exactly_one_outcome() {
        // Timeout protocol: submit, race cancel_job against execution —
        // however the race resolves, exactly ONE terminal outcome arrives
        // (never zero, never two). Unknown ids are a safe no-op.
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        assert_eq!(
            s.cancel_job(u64::MAX),
            JobCancelOutcome::UnknownJob,
            "unknown job id is a no-op"
        );
        let run = fresh_run();
        let dir = test_dir("once");
        let f = dir.join("o.txt");
        std::fs::write(&f, "once\n").unwrap();
        let action = ProposedAction::new(
            ProposedKind::Read,
            f.to_string_lossy().to_string(),
            String::new(),
            None,
            ToolCallSource::ModelCompletion,
        );
        let receipts = s.submit_explicit_batch(run, vec![action]);
        let job_id = receipts[0].job_id;
        let _ = s.cancel_job(job_id);
        let first = receipts[0]
            .outcome
            .recv_timeout(Duration::from_secs(10))
            .expect("exactly one terminal outcome");
        assert!(
            !first.result.is_empty(),
            "outcome carries a result either way"
        );
        assert!(
            receipts[0].outcome.try_recv().is_err(),
            "no second outcome may ever arrive"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    /// Push one explicit probe job directly (no kick) and pop it back,
    /// retrying if a stale wakeup lets a parked worker steal it. Returns
    /// the popped job quarantined from workers (no kick was sent for it).
    /// Each attempt uses a fresh target file so a steal-execution cannot
    /// pollute the next attempt's cache assertions.
    fn push_pop_probe(
        s: &'static AgentIoScheduler,
        dir: &Path,
        run: u64,
        tag: &str,
    ) -> (
        IoJob,
        std::sync::mpsc::Receiver<ExplicitReadOutcome>,
        PathBuf,
    ) {
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
        for attempt in 0..100 {
            let target = dir.join(format!("{tag}-{attempt}.txt"));
            std::fs::write(&target, "probe\n").unwrap();
            let action = ProposedAction::new(
                ProposedKind::Read,
                target.to_string_lossy().to_string(),
                String::new(),
                None,
                ToolCallSource::ModelCompletion,
            );
            let (tx, rx) = std::sync::mpsc::channel();
            let my_id = next_job_id();
            let epoch = s.lock().epoch;
            {
                let mut st = s.lock();
                st.queue.clear();
                st.queue.push(QueuedJob {
                    job: IoJob {
                        id: my_id,
                        run_id: run,
                        priority: IoPriority::Explicit,
                        kind: IoJobKind::ExplicitRead,
                        target: target.clone(),
                        generation: 0,
                        reason: RetrievalReason::ModelIntent,
                        epoch,
                        action: Some(action),
                        reply: Some(tx),
                    },
                    seq: 0,
                });
            }
            match s.pop_eligible() {
                Some(j) if j.id == my_id => return (j, rx, target),
                // Stolen by a stale-wakeup worker (or epoch-rotated):
                // each steal consumes one stale wakeup — retry converges.
                _ => continue,
            }
        }
        panic!("probe job never survived pop after retries");
    }

    #[test]
    fn ownership_visible_between_pop_and_execute() {
        // Forced interleaving for the fixed race: after the worker-equivalent
        // pop, the job MUST have an ownership record before any execution.
        // cancel_job in that gap must observe active state (never false),
        // and the subsequent execution must not run the action.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("ownrace");
        let (job, rx, target) = push_pop_probe(s, &dir, run, "own");

        // Ownership record visible immediately after queue removal.
        assert!(
            s.lock().active.contains_key(&job.id),
            "popped job must be active-registered with no unowned gap"
        );
        // Cancellation in the gap observes active state (the old code
        // returned false here — the reported hole).
        assert_eq!(
            s.cancel_job(job.id),
            JobCancelOutcome::CancellationRequested
        );
        // Worker-equivalent execution honors the flag: error outcome, and
        // the claimed action never runs (no cache entry is warmed).
        s.execute_job(job.clone());
        let out = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("terminal outcome delivered");
        assert!(
            out.result.trim_start().starts_with("Error:"),
            "cancelled execution surfaces an error, got: {}",
            out.result
        );
        assert!(
            s.serve_authorized(&target).is_none(),
            "cancelled job must not execute (no cache warming)"
        );
        assert!(rx.try_recv().is_err(), "no second outcome may ever arrive");
        // Terminal transition released ownership: late cancel observes it.
        assert!(!s.lock().active.contains_key(&job.id));
        assert_eq!(s.cancel_job(job.id), JobCancelOutcome::AlreadyTerminal);
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn queued_cancel_is_terminal_immediately() {
        // Queued explicit job: cancel observes queued state, delivers the
        // terminal error outcome at once, and a second cancel observes
        // AlreadyTerminal (not UnknownJob).
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("qcancel");
        for attempt in 0..100 {
            let target = dir.join(format!("q-{attempt}.txt"));
            std::fs::write(&target, "probe\n").unwrap();
            use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
            let action = ProposedAction::new(
                ProposedKind::Read,
                target.to_string_lossy().to_string(),
                String::new(),
                None,
                ToolCallSource::ModelCompletion,
            );
            let (tx, rx) = std::sync::mpsc::channel();
            let my_id = next_job_id();
            let epoch = s.lock().epoch;
            {
                let mut st = s.lock();
                st.queue.clear();
                st.queue.push(QueuedJob {
                    job: IoJob {
                        id: my_id,
                        run_id: run,
                        priority: IoPriority::Explicit,
                        kind: IoJobKind::ExplicitRead,
                        target: target.clone(),
                        generation: 0,
                        reason: RetrievalReason::ModelIntent,
                        epoch,
                        action: Some(action),
                        reply: Some(tx),
                    },
                    seq: 0,
                });
            }
            match s.cancel_job(my_id) {
                JobCancelOutcome::CancelledBeforeExecution => {
                    let out = rx
                        .recv_timeout(Duration::from_secs(10))
                        .expect("terminal cancellation outcome");
                    assert!(
                        out.result.trim_start().starts_with("Error:"),
                        "got: {}",
                        out.result
                    );
                    assert!(
                        s.serve_authorized(&target).is_none(),
                        "cancelled-before-execution must not run"
                    );
                    assert_eq!(s.cancel_job(my_id), JobCancelOutcome::AlreadyTerminal);
                    let _ = std::fs::remove_dir_all(&dir);
                    s.reset_for_test();
                    return;
                }
                // Raced by a stale-wakeup worker steal — retry converges.
                _ => continue,
            }
        }
        panic!("queued cancel never observed after retries");
    }

    #[test]
    fn delete_recreate_never_serves_old_bytes() {
        // P1 invariant: cache → delete → notify_write invalidation →
        // recreate same path → old bytes must never be served. Works
        // because invalidation needs no existence (stable identity) and
        // recreation starts at a newer generation.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("delre");
        let f = dir.join("gone.rs");
        std::fs::write(&f, "version one\n").unwrap();
        assert!(s.read_through(&f).is_ok(), "initial read warms cache");
        assert_eq!(
            s.serve_authorized(&f).map(|h| h.content),
            Some("version one\n".to_string())
        );

        std::fs::remove_file(&f).unwrap();
        // Canonical delete path (execute paths call notify_write on
        // successful deletes the same way as writes).
        s.notify_write(&f);
        assert!(
            s.serve_authorized(&f).is_none(),
            "deleted file must not serve cached bytes"
        );

        std::fs::write(&f, "version two, completely different\n").unwrap();
        assert!(
            s.serve_authorized(&f).is_none(),
            "recreated path must not inherit the old snapshot"
        );
        // Fresh read works and caches the NEW bytes.
        assert!(s.read_through(&f).is_ok());
        assert_eq!(
            s.serve_authorized(&f).map(|h| h.content),
            Some("version two, completely different\n".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn non_canonical_paths_share_stable_identity() {
        // `./foo.rs`, `sub/../foo.rs` and the plain path are ONE scheduler
        // identity: a write through one form invalidates reads through
        // another, and generations agree.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("noncanon");
        let f = dir.join("foo.rs");
        std::fs::write(&f, "v1\n").unwrap();

        let plain = f.clone();
        let dotted = dir.join(".").join("foo.rs");
        let dotdot = dir.join("sub").join("..").join("foo.rs");
        assert!(s.read_through(&plain).is_ok());
        // All three forms hit the same entry.
        assert!(
            s.serve_authorized(&dotted).is_some(),
            "dotted form promotes"
        );
        assert!(
            s.serve_authorized(&dotdot).is_some(),
            "dot-dot form promotes"
        );
        // Invalidate through a non-canonical form…
        std::fs::write(&f, "v2 longer\n").unwrap();
        s.notify_write(&dotted);
        // …visible through every form.
        assert!(s.serve_authorized(&plain).is_none());
        assert!(s.serve_authorized(&dotdot).is_none());
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn external_modification_detected_at_serve() {
        // Consistency contract, detectable case: an external rewrite that
        // changes length is caught by serve-time revalidation (mtime+len),
        // even with no generation bump (external writers don't notify).
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("extmod");
        let f = dir.join("ext.txt");
        std::fs::write(&f, "aaa\n").unwrap();
        assert!(s.read_through(&f).is_ok());
        assert!(s.serve_authorized(&f).is_some());

        std::fs::write(&f, "aaaaaaa much longer external edit\n").unwrap();
        assert!(
            s.serve_authorized(&f).is_none(),
            "externally length-changed file must not serve stale bytes"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn panic_releases_ownership_and_pool_survives() {
        // Panic lifecycle: active → panic unwind → guard Drop → terminal.
        // Then prove the worker pool still processes real jobs.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let job_id = next_job_id();
        {
            let mut st = s.lock();
            st.active.insert(
                job_id,
                ActiveJob {
                    job_id,
                    kind: IoJobKind::SpeculativeRead,
                    target: PathBuf::from("panic-probe"),
                    run_id: fresh_run(),
                    started: Instant::now(),
                },
            );
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _own = ActiveOwnershipGuard { sched: s, job_id };
            panic!("simulated worker panic");
        }));
        assert!(result.is_err(), "panic propagates through the guard");
        assert!(
            !s.lock().active.contains_key(&job_id),
            "ownership released despite panic"
        );
        assert!(
            s.lock().completed_jobs.contains(&job_id),
            "panicked job recorded terminal"
        );
        assert_eq!(
            s.cancel_job(job_id),
            JobCancelOutcome::AlreadyTerminal,
            "late cancel observes terminal, not unknown"
        );
        // Pool still alive: a real explicit batch completes afterwards.
        let dir = test_dir("poolalive");
        let f = dir.join("alive.txt");
        std::fs::write(&f, "alive\n").unwrap();
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
        let action = ProposedAction::new(
            ProposedKind::Read,
            f.to_string_lossy().to_string(),
            String::new(),
            None,
            ToolCallSource::ModelCompletion,
        );
        let receipts = s.submit_explicit_batch(fresh_run(), vec![action]);
        let out = receipts[0]
            .outcome
            .recv_timeout(Duration::from_secs(15))
            .expect("pool processes jobs after panic path");
        assert_eq!(out.result, "alive\n");
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn explicit_worker_warms_coherent_entry() {
        // The P1 warm-up race, structurally closed: worker-warmed bytes,
        // metadata and generation come from ONE stable read, so a later
        // serve can never mix content from one revision with metadata
        // from another. Single file, inert content → no speculation noise.
        use crate::agent::{ProposedAction, ProposedKind, ToolCallSource};
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let run = fresh_run();
        let dir = test_dir("coherent");
        let f = dir.join("only.txt");
        std::fs::write(&f, "plain content\n").unwrap();

        let action = ProposedAction::new(
            ProposedKind::Read,
            f.to_string_lossy().to_string(),
            String::new(),
            None,
            ToolCallSource::ModelCompletion,
        );
        let receipts = s.submit_explicit_batch(run, vec![action]);
        let out = receipts[0]
            .outcome
            .recv_timeout(Duration::from_secs(15))
            .expect("explicit outcome delivered");
        assert_eq!(out.result, "plain content\n");
        assert!(!out.served_from_cache, "first read comes from disk");

        // The warmed entry is coherent: same bytes the outcome carried,
        // generation equal to the live generation.
        let hit = s.serve_authorized(&f).expect("worker warmed the entry");
        assert_eq!(hit.content, "plain content\n");
        assert_eq!(hit.origin, ReadOrigin::Explicit);
        let live_generation = s
            .lock()
            .generations
            .get(&stable_key(&f))
            .copied()
            .unwrap_or(0);
        assert_eq!(
            hit.generation, live_generation,
            "served generation must equal the live generation"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn last_serve_ring_is_bounded() {
        // last_serve must never grow with every unique path ever served.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        {
            let mut st = s.lock();
            for i in 0..300 {
                st.cache.note_served(
                    PathBuf::from(format!("/tmp/probe-{i}.txt")),
                    ReadOrigin::Speculative,
                );
            }
            assert!(
                st.cache.last_serve.len() <= MAX_LAST_SERVE,
                "ring bounded, got {}",
                st.cache.last_serve.len()
            );
            // Oldest evicted, newest retained.
            assert!(
                st.cache
                    .last_served(&PathBuf::from("/tmp/probe-0.txt"))
                    .is_none(),
                "oldest record evicted"
            );
            assert!(
                st.cache
                    .last_served(&PathBuf::from("/tmp/probe-299.txt"))
                    .is_some(),
                "newest record retained"
            );
        }
        s.reset_for_test();
    }

    #[test]
    fn generation_sweep_bounds_unreferenced() {
        // Thousands of writes to distinct paths must not grow generations
        // without bound; a referenced (cached) path survives the sweep.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("gensweep");
        let live = dir.join("live.txt");
        std::fs::write(&live, "live\n").unwrap();
        assert!(s.read_through(&live).is_ok());
        assert!(s.serve_authorized(&live).is_some());

        for i in 0..5000 {
            s.notify_write(&PathBuf::from(format!("/tmp/sweep-{i}.txt")));
        }
        let st = s.lock();
        assert!(
            st.generations.len() <= MAX_GENERATIONS,
            "generations bounded, got {}",
            st.generations.len()
        );
        drop(st);
        // Referenced path (queued/active/cache) keeps its generation, so
        // its cache entry still validates.
        assert!(
            s.serve_authorized(&live).is_some(),
            "referenced path survives the sweep"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn run_state_retirement_stays_bounded() {
        // Many cancelled runs must not accumulate run state; a run with
        // live queued work keeps its budget.
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("runsweep");
        let live_file = dir.join("live.txt");
        std::fs::write(&live_file, "x\n").unwrap();

        // A live run with a queued speculative job.
        let live_run = fresh_run();
        let submitted = s.submit_speculative(
            live_run,
            vec![RetrievalCandidate {
                path: live_file.clone(),
                confidence: 0.9,
                reason: RetrievalReason::QuotedPath,
                estimated_bytes: 2,
            }],
        );
        assert_eq!(submitted, 1);

        for _ in 0..200 {
            s.cancel_run(fresh_run());
        }
        {
            let st = s.lock();
            let total = st.cancelled_runs.len()
                + st.budgets.len()
                + st.stream_marks.len()
                + st.run_tokens.len();
            assert!(total <= MAX_RUN_STATES, "run state bounded, got {total}");
            assert!(
                st.budgets.contains_key(&live_run),
                "live run keeps its budget"
            );
        }
        // The live job still completes normally afterwards.
        wait_for(
            || s.serve_authorized(&live_file).is_some(),
            "live job completes despite sweeps",
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    /// Set the global folder scope for a test, returning the previous
    /// permissions for restoration. Holds NO locks itself — callers must
    /// hold both the scheduler serial guard and the perm guard.
    fn set_scope_for_test(scope: crate::agent::FolderScope) -> crate::agent::ToolPermissions {
        let saved = crate::agent::get_tool_permissions();
        *crate::agent::TOOL_PERMS.lock().unwrap() = crate::agent::ToolPermissions {
            mode: saved.mode,
            folder_scope: scope,
            session_allow: saved.session_allow,
        };
        saved
    }

    #[test]
    fn promotion_symlink_swap_never_serves_escaped_bytes() {
        // Test A — symlink swap during cache promotion: cache an in-tree
        // file, then race its symlink between the in-tree object and an
        // out-of-tree secret while calling serve_authorized(). Unauthorized
        // cached content must NEVER be returned (misses and in-tree hits
        // are both safe outcomes).
        #[cfg(not(unix))]
        {
            return;
        }
        #[cfg(unix)]
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            let _g = test_serial_guard();
            let _p = crate::agent::perm_test_guard();
            let s = sched();
            s.reset_for_test();
            let dir = test_dir("promo-swap");
            let real = dir.join("real.txt");
            std::fs::write(&real, "IN TREE\n").unwrap();
            let outside_dir =
                std::env::temp_dir().join(format!("hercules-promo-swapout-{}", std::process::id()));
            std::fs::create_dir_all(&outside_dir).unwrap();
            let secret = outside_dir.join("secret.txt");
            std::fs::write(&secret, "SECRET\n").unwrap();
            let link = dir.join("link.txt");
            std::os::unix::fs::symlink(&real, &link).unwrap();

            // Warm the cache through the link (genuinely in-tree bytes).
            assert!(s.read_through(&link).is_ok());

            let stop = std::sync::Arc::new(AtomicBool::new(false));
            let stopper = stop.clone();
            let (link_c, real_c, secret_c) = (link.clone(), real.clone(), secret.clone());
            let swapper = std::thread::spawn(move || {
                let mut to_secret = true;
                while !stopper.load(Ordering::Relaxed) {
                    let _ = std::fs::remove_file(&link_c);
                    let target = if to_secret { &secret_c } else { &real_c };
                    let _ = std::os::unix::fs::symlink(target, &link_c);
                    to_secret = !to_secret;
                }
            });
            for _ in 0..1500 {
                if let Some(hit) = s.serve_authorized(&link) {
                    assert_ne!(
                        hit.content, "SECRET\n",
                        "sandbox escape via raced symlink promotion"
                    );
                }
            }
            stop.store(true, Ordering::Relaxed);
            swapper.join().unwrap();
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
            s.reset_for_test();
        }
    }

    #[test]
    fn promotion_scope_transition_cannot_serve_outside_object() {
        // Test B — populate cache under AllDirs, switch to CurrentDir,
        // then serve an in-tree symlink whose target is outside: the
        // previously cached outside object must not promote.
        #[cfg(not(unix))]
        {
            return;
        }
        #[cfg(unix)]
        {
            let _g = test_serial_guard();
            let _p = crate::agent::perm_test_guard();
            let s = sched();
            s.reset_for_test();
            let saved = set_scope_for_test(crate::agent::FolderScope::AllDirs);
            let dir = test_dir("promo-scope");
            let outside_dir = std::env::temp_dir()
                .join(format!("hercules-promo-scopeout-{}", std::process::id()));
            std::fs::create_dir_all(&outside_dir).unwrap();
            let secret = outside_dir.join("secret.txt");
            std::fs::write(&secret, "SECRET\n").unwrap();
            let link = dir.join("link.txt");
            std::os::unix::fs::symlink(&secret, &link).unwrap();

            // Warmed while unrestricted (links resolve outside — allowed).
            assert!(s.read_through(&link).is_ok());
            assert!(s.serve_authorized(&link).is_some());

            // Tighten the sandbox: the same object must no longer promote.
            set_scope_for_test(crate::agent::FolderScope::CurrentDir);
            for _ in 0..50 {
                assert!(
                    s.serve_authorized(&link).is_none(),
                    "outside object must not promote under CurrentDir"
                );
            }
            *crate::agent::TOOL_PERMS.lock().unwrap() = saved;
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
            s.reset_for_test();
        }
    }

    #[test]
    fn promotion_handle_bound_across_post_open_swap() {
        // Test C — validate with an already-open handle after the entry
        // is swapped out-of-tree: open/validate in-tree, replace the
        // entry, then prove the handle still describes the original object
        // (fd-bound metadata, never a re-resolved name).
        #[cfg(not(unix))]
        {
            return;
        }
        #[cfg(unix)]
        {
            let _g = test_serial_guard();
            let _p = crate::agent::perm_test_guard();
            let dir = test_dir("promo-handle");
            let real = dir.join("real.txt");
            std::fs::write(&real, "IN TREE\n").unwrap();
            let outside_dir = std::env::temp_dir()
                .join(format!("hercules-promo-handleout-{}", std::process::id()));
            std::fs::create_dir_all(&outside_dir).unwrap();
            let secret = outside_dir.join("secret.txt");
            std::fs::write(&secret, "SECRET\n").unwrap();
            let link = dir.join("link.txt");
            std::os::unix::fs::symlink(&real, &link).unwrap();

            let root = std::env::current_dir().unwrap().canonicalize().unwrap();
            let sec = crate::secure_fs::secure_open_read(&link, Some(&root))
                .expect("in-tree open validates");
            // Swap the entry out from under the open handle.
            std::fs::remove_file(&link).unwrap();
            std::os::unix::fs::symlink(&secret, &link).unwrap();

            // The handle still describes the verified in-tree object:
            // same inode, in-tree bytes — the swapped name is irrelevant.
            let before = std::fs::File::metadata(&sec.file).unwrap();
            let mut buf = Vec::new();
            use std::io::Read;
            let mut f = sec.file;
            f.read_to_end(&mut buf).unwrap();
            assert_eq!(buf, b"IN TREE\n");
            let after = std::fs::metadata(&real).unwrap();
            use std::os::unix::fs::MetadataExt;
            assert_eq!(
                (before.dev(), before.ino()),
                (after.dev(), after.ino()),
                "handle bound to the originally verified object"
            );
            // And the swapped NAME no longer validates at all.
            assert!(
                crate::secure_fs::secure_open_read(&link, Some(&root)).is_err(),
                "post-swap name must not validate"
            );
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
        }
    }

    #[test]
    fn promotion_disappeared_object_misses() {
        // Test D — populate, delete, serve: must return None (and drop the
        // entry) rather than serving bytes for a vanished object.
        let _g = test_serial_guard();
        let _p = crate::agent::perm_test_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("promo-gone");
        let f = dir.join("gone.txt");
        std::fs::write(&f, "here\n").unwrap();
        assert!(s.read_through(&f).is_ok());
        assert!(s.serve_authorized(&f).is_some());
        std::fs::remove_file(&f).unwrap();
        for _ in 0..10 {
            assert!(
                s.serve_authorized(&f).is_none(),
                "vanished object must miss"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn final_point_defeats_post_snapshot_write() {
        // The P1 race, deterministically: an entry snapshot copied BEFORE
        // a Hercules write must not serve AFTER it — even when the file
        // bytes on disk still match the snapshot (so only the final
        // generation revalidation can defeat the serve).
        //
        // Setup reproduces the exact interleaving: warm v1 (gen 0), write
        // bumps to gen 1 and removes the entry, then a stale v1 copy with
        // gen 0 is re-inserted (what an in-flight serve would hold).
        let _g = test_serial_guard();
        let _p = crate::agent::perm_test_guard();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("finalpoint");
        let f = dir.join("race.txt");
        std::fs::write(&f, "v1\n").unwrap();
        assert!(s.read_through(&f).is_ok());
        assert!(s.serve_authorized(&f).is_some());

        // Hercules write: generation 1, entry removed.
        std::fs::write(&f, "v1\n").unwrap(); // same bytes: fs_fresh WILL match
        s.notify_write(&f);
        assert!(s.serve_authorized(&f).is_none(), "post-write serves miss");

        // Re-insert the stale snapshot copy (gen 0, v1 bytes, matching
        // mtime/len since the file body is identical).
        {
            let mut st = s.lock();
            let key = stable_key(&f);
            let meta = std::fs::metadata(&f).unwrap();
            st.cache_put(
                key,
                CacheEntry {
                    content: "v1\n".to_string(),
                    generation: 0, // stale: live generation is 1
                    mtime: meta.modified().unwrap(),
                    len: meta.len(),
                    fetched_at: Instant::now(),
                    origin: CacheOrigin::SpeculativeRead,
                    usefulness_counted: false,
                },
            );
        }
        // The final validation point must defeat it: live generation (1)
        // disagrees with the snapshot (0) even though every filesystem
        // check passes.
        assert!(
            s.serve_authorized(&f).is_none(),
            "stale snapshot copy must never serve after a write"
        );
        // And the poisoned copy is dropped, not left to shadow later reads.
        assert!(
            s.lock().cache.entries.get(&stable_key(&f)).is_none(),
            "stale copy removed"
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
    }

    #[test]
    fn serve_write_interleaving_never_serves_stale() {
        // Deterministic serve-vs-write interleaving via the barrier hook:
        // serve snapshots gen 0, pauses; a Hercules write bumps to gen 1
        // and invalidates; serve resumes and MUST miss rather than return
        // the old bytes. This exercises the real end-to-end race (here the
        // content change trips fs validation; the generation-only case —
        // identical bytes — is isolated by
        // final_point_defeats_post_snapshot_write, which is
        // mutation-verified against the final check).
        let _g = test_serial_guard();
        let _p = crate::agent::perm_test_guard();
        disarm_serve_interleave();
        let s = sched();
        s.reset_for_test();
        let dir = test_dir("interleave");
        let f = dir.join("race.txt");
        std::fs::write(&f, "v1\n").unwrap();
        assert!(s.read_through(&f).is_ok());
        assert!(s.serve_authorized(&f).is_some());

        let pause = std::sync::Arc::new(std::sync::Barrier::new(2));
        let resume = std::sync::Arc::new(std::sync::Barrier::new(2));
        arm_serve_interleave(pause.clone(), resume.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        let f_clone = f.clone();
        std::thread::spawn(move || {
            let out = s.serve_authorized(&f_clone);
            let _ = tx.send(out.map(|h| h.content));
        });
        pause.wait();
        // Racing Hercules write lands strictly inside serve's window.
        std::fs::write(&f, "v2 brand new\n").unwrap();
        s.notify_write(&f);
        resume.wait();
        let served = rx
            .recv_timeout(Duration::from_secs(15))
            .expect("serve completes");
        disarm_serve_interleave();
        assert!(
            served.is_none(),
            "serve racing a write must miss, got stale bytes"
        );
        // And the fresh state serves correctly afterwards.
        assert!(s.read_through(&f).is_ok());
        assert_eq!(
            s.serve_authorized(&f).map(|h| h.content),
            Some("v2 brand new\n".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
        s.reset_for_test();
        disarm_serve_interleave();
    }

    #[test]
    fn diagnostics_renders() {
        let _g = test_serial_guard();
        let s = sched();
        s.reset_for_test();
        let text = s.diagnostics();
        assert!(text.contains("Agent I/O Scheduler"));
        assert!(text.contains("hit rate"));
        s.reset_for_test();
    }
}
