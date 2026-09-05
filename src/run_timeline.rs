//! First-class representation of an entire agent run.
//!
//! The canonical tool pipeline produces individual calls (`call_id`,
//! fingerprints, durations, chips, results); this module unifies them into
//! one coherent run: steps, state, progress, failure, recovery. It is the
//! foundation for cancellation, checkpoints, diff-first editing and the
//! automatic verify loop.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Monotonic mint for [`AgentRun::id`] / [`AgentStep::id`].
static NEXT_RUN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn next_run_id() -> u64 {
    NEXT_RUN_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Lifecycle state of a whole run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRunState {
    Planning,
    Thinking,
    Executing,
    WaitingForUser,
    Completed,
    Failed,
    Cancelled,
}

impl AgentRunState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Lifecycle state of a single step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    WaitingApproval,
    Succeeded,
    Failed,
    Skipped,
    /// Interrupted by user cancellation (distinct from Skipped: the step
    /// started executing but never finished).
    Cancelled,
}

impl StepStatus {
    pub fn is_done(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Skipped | Self::Cancelled
        )
    }

    pub fn is_active(self) -> bool {
        matches!(self, Self::Running | Self::Pending | Self::WaitingApproval)
    }
}

/// What a step represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    Think,
    Read,
    List,
    Write,
    Run,
    Search,
    Mcp,
    Skill,
    SubAgent,
    Memory,
    Generate,
}

impl StepKind {
    pub fn icon(self) -> &'static str {
        match self {
            Self::Think => "💭",
            Self::Read => "📖",
            Self::List => "📁",
            Self::Write => "✏️",
            Self::Run => "▶",
            Self::Search => "🔍",
            Self::Mcp => "🔌",
            Self::Skill => "🧰",
            Self::SubAgent => "🤖",
            Self::Memory => "🧠",
            Self::Generate => "⚙️",
        }
    }
}

/// One step inside a run, optionally bound to a canonical tool `call_id`.
#[derive(Debug, Clone)]
pub struct AgentStep {
    pub id: u64,
    pub call_id: Option<u64>,
    pub kind: StepKind,
    pub status: StepStatus,
    pub started_at: Instant,
    pub finished_at: Option<Instant>,
    pub summary: String,
    pub result_summary: Option<String>,
}

impl AgentStep {
    pub fn new(kind: StepKind, summary: String, call_id: Option<u64>) -> Self {
        Self {
            id: next_run_id(),
            call_id,
            kind,
            status: StepStatus::Running,
            started_at: Instant::now(),
            finished_at: None,
            summary,
            result_summary: None,
        }
    }

    pub fn finish(&mut self, status: StepStatus, result_summary: Option<String>) {
        self.status = status;
        self.finished_at = Some(Instant::now());
        self.result_summary = result_summary;
    }

    pub fn elapsed(&self) -> Duration {
        self.finished_at
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(self.started_at)
    }
}

/// Serializable, wall-clock summary of a finished run for history and
/// session persistence (`Instant` is monotonic-only and never serialized).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RunSummary {
    pub id: u64,
    pub prompt: String,
    pub state: String,
    pub started_epoch_secs: u64,
    pub finished_epoch_secs: Option<u64>,
    pub duration_ms: Option<u64>,
    pub steps_total: usize,
    pub steps_done: usize,
}

/// A whole agent run: one user prompt → steps → terminal state.
#[derive(Debug)]
pub struct AgentRun {
    pub id: u64,
    pub user_prompt: String,
    pub started_at: Instant,
    pub started_wall: SystemTime,
    pub finished_at: Option<Instant>,
    pub finished_wall: Option<SystemTime>,
    pub state: AgentRunState,
    pub steps: Vec<AgentStep>,
}

impl AgentRun {
    pub fn new(user_prompt: String) -> Self {
        Self {
            id: next_run_id(),
            user_prompt,
            started_at: Instant::now(),
            started_wall: SystemTime::now(),
            finished_at: None,
            finished_wall: None,
            state: AgentRunState::Planning,
            steps: Vec::new(),
        }
    }

    /// Start a step; returns its id.
    pub fn start_step(&mut self, kind: StepKind, summary: String, call_id: Option<u64>) -> u64 {
        // A step bound to an already-tracked call_id reuses that step
        // (streaming re-dispatch of the same logical call). An approval
        // wait that gets claimed revives to Running.
        if let Some(cid) = call_id {
            if let Some(s) = self.steps.iter_mut().find(|s| s.call_id == Some(cid)) {
                if s.status == StepStatus::WaitingApproval {
                    s.status = StepStatus::Running;
                    s.started_at = Instant::now();
                }
                return s.id;
            }
        }
        let step = AgentStep::new(kind, summary, call_id);
        let id = step.id;
        self.steps.push(step);
        if self.state == AgentRunState::Planning {
            self.state = AgentRunState::Executing;
        }
        id
    }

    pub fn finish_step(
        &mut self,
        step_id: u64,
        status: StepStatus,
        result_summary: Option<String>,
    ) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.id == step_id) {
            s.finish(status, result_summary);
        }
    }

    /// Convenience: finish the step bound to a tool `call_id`.
    pub fn finish_call(
        &mut self,
        call_id: u64,
        status: StepStatus,
        result_summary: Option<String>,
    ) {
        if let Some(s) = self.steps.iter_mut().find(|s| s.call_id == Some(call_id)) {
            s.finish(status, result_summary);
        }
    }

    /// Validated transition. Terminal states are sinks: once a run is
    /// Completed/Failed/Cancelled it can never leave. Returns false and
    /// changes nothing on an illegal transition.
    pub fn transition_to(&mut self, next: AgentRunState) -> bool {
        if self.state.is_terminal() {
            return false;
        }
        // Planning is the entry state; never go back to it.
        if next == AgentRunState::Planning {
            return false;
        }
        self.state = next;
        if next.is_terminal() {
            self.finished_at = Some(Instant::now());
            self.finished_wall = Some(SystemTime::now());
        }
        true
    }

    pub fn finish_run(&mut self, state: AgentRunState) {
        // Only terminal states are valid finish targets.
        if !state.is_terminal() {
            return;
        }
        self.transition_to(state);
    }

    /// Cancel the run AND every still-active step. A cancelled run never
    /// leaves `Running` steps behind.
    pub fn cancel(&mut self) {
        for s in self.steps.iter_mut() {
            if s.status.is_active() {
                s.finish(StepStatus::Cancelled, Some("Cancelled by user".to_string()));
            }
        }
        self.transition_to(AgentRunState::Cancelled);
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn progress(&self) -> (usize, usize) {
        let done = self.steps.iter().filter(|s| s.status.is_done()).count();
        (done, self.steps.len())
    }

    /// Total wall duration if finished, else elapsed so far.
    pub fn duration(&self) -> Duration {
        self.finished_at
            .unwrap_or_else(Instant::now)
            .saturating_duration_since(self.started_at)
    }

    fn epoch_secs(t: SystemTime) -> u64 {
        t.duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Persistable snapshot of a finished (or live) run.
    pub fn summarize(&self) -> RunSummary {
        let (done, total) = self.progress();
        RunSummary {
            id: self.id,
            prompt: self.user_prompt.chars().take(200).collect(),
            state: format!("{:?}", self.state),
            started_epoch_secs: Self::epoch_secs(self.started_wall),
            finished_epoch_secs: self.finished_wall.map(Self::epoch_secs),
            duration_ms: Some(self.duration().as_millis() as u64),
            steps_total: total,
            steps_done: done,
        }
    }

    /// One-line-per-step timeline for the UI panel.
    pub fn timeline_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "Agent Run #{} — {:?} ({}/{})",
            self.id,
            self.state,
            self.progress().0,
            self.progress().1
        ));
        for s in &self.steps {
            let mark = match s.status {
                StepStatus::Succeeded => "✓",
                StepStatus::Failed => "✗",
                StepStatus::Running => "▶",
                StepStatus::Pending => "○",
                StepStatus::WaitingApproval => "⏳",
                StepStatus::Skipped => "⊘",
                StepStatus::Cancelled => "⊘",
            };
            let secs = s.elapsed().as_secs_f64();
            lines.push(format!(
                "{} {} {}  {:.2}s",
                mark,
                s.kind.icon(),
                s.summary,
                secs
            ));
            if let Some(ref r) = s.result_summary {
                let first = r
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(80)
                    .collect::<String>();
                if !first.is_empty() {
                    lines.push(format!("    └ {first}"));
                }
            }
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_step_lifecycle() {
        let mut run = AgentRun::new("read it".to_string());
        assert_eq!(run.state, AgentRunState::Planning);
        let id = run.start_step(StepKind::Read, "Read Cargo.toml".to_string(), Some(7));
        assert_eq!(run.state, AgentRunState::Executing);
        // Same call_id reuses the step (no duplicate on re-dispatch).
        assert_eq!(
            run.start_step(StepKind::Read, "Read Cargo.toml".to_string(), Some(7)),
            id
        );
        assert_eq!(run.steps.len(), 1);
        run.finish_call(7, StepStatus::Succeeded, Some("42 lines".to_string()));
        assert_eq!(run.progress(), (1, 1));
        run.finish_run(AgentRunState::Completed);
        assert!(run.is_terminal());
    }

    #[test]
    fn test_run_timeline_lines() {
        let mut run = AgentRun::new("x".to_string());
        let id = run.start_step(StepKind::Run, "cargo check".to_string(), None);
        run.finish_step(id, StepStatus::Succeeded, None);
        let lines = run.timeline_lines();
        assert!(lines[0].contains("Agent Run"));
        assert!(lines.iter().any(|l| l.contains("cargo check")));
    }

    #[test]
    fn test_run_cancel_and_fail() {
        let mut run = AgentRun::new("x".to_string());
        run.finish_run(AgentRunState::Cancelled);
        assert!(run.is_terminal());
        let mut run2 = AgentRun::new("y".to_string());
        let id = run2.start_step(StepKind::Write, "w".to_string(), Some(1));
        run2.finish_step(id, StepStatus::Failed, Some("denied".to_string()));
        assert_eq!(run2.progress(), (1, 1));
    }

    #[test]
    fn test_cancel_cascades_to_active_steps() {
        let mut run = AgentRun::new("x".to_string());
        run.start_step(StepKind::Read, "a".to_string(), Some(1));
        run.finish_call(1, StepStatus::Succeeded, None);
        run.start_step(StepKind::Write, "b".to_string(), Some(2));
        run.start_step(StepKind::Run, "c".to_string(), Some(3));
        run.cancel();
        assert_eq!(run.state, AgentRunState::Cancelled);
        // Finished steps untouched; active ones become Cancelled, not Running.
        assert_eq!(run.steps[0].status, StepStatus::Succeeded);
        assert_eq!(run.steps[1].status, StepStatus::Cancelled);
        assert_eq!(run.steps[2].status, StepStatus::Cancelled);
        assert_eq!(run.progress(), (3, 3));
    }

    #[test]
    fn test_illegal_transitions_rejected() {
        let mut run = AgentRun::new("x".to_string());
        assert!(run.transition_to(AgentRunState::Executing));
        assert!(run.transition_to(AgentRunState::Completed));
        // Terminal is a sink.
        assert!(!run.transition_to(AgentRunState::Executing));
        assert!(!run.transition_to(AgentRunState::Planning));
        assert_eq!(run.state, AgentRunState::Completed);
        // finish_run with a non-terminal state is a no-op.
        let mut run2 = AgentRun::new("y".to_string());
        run2.finish_run(AgentRunState::Executing);
        assert_eq!(run2.state, AgentRunState::Planning);
    }

    #[test]
    fn test_approval_wait_revives_on_claim() {
        let mut run = AgentRun::new("x".to_string());
        let id = run.start_step(StepKind::Write, "w".to_string(), Some(9));
        run.finish_step(id, StepStatus::WaitingApproval, Some("waiting".to_string()));
        // Claiming (re-dispatch after user approval) revives to Running.
        assert_eq!(
            run.start_step(StepKind::Write, "w".to_string(), Some(9)),
            id
        );
        assert_eq!(run.steps[0].status, StepStatus::Running);
    }

    #[test]
    fn test_run_summary_serializes() {
        let mut run = AgentRun::new("do the thing".to_string());
        let id = run.start_step(StepKind::Read, "a".to_string(), Some(1));
        run.finish_step(id, StepStatus::Succeeded, None);
        run.finish_run(AgentRunState::Completed);
        let s = run.summarize();
        assert_eq!(s.state, "Completed");
        assert_eq!(s.steps_total, 1);
        assert_eq!(s.steps_done, 1);
        assert!(s.started_epoch_secs > 0);
        let json = serde_json::to_string(&s).unwrap();
        let back: RunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, s.id);
    }
}
