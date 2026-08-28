//! Hercules Smart System — Central Execution and Concurrency Coordination Layer.
//!
//! # Architecture
//!
//! Agents decide **what they want to do**; the Smart System decides **how and when it is safely executed**.
//!
//! 1. **Agent Ownership & Identification**:
//!    - Operations are tagged with agent identity (e.g. `H0` Main, `H1` Prompt Engineer, `H2` Bug Hunter, `H3` Coder).
//!    - Commands and tool results route unambiguously back to the requesting agent.
//!
//! 2. **Tool Scheduling & Classification**:
//!    - **Read / WebSearch**: Stateless / read-only — executed concurrently and registers read revision snapshots.
//!    - **Write / Replace**: Mutating operations passed through optimistic concurrency verification.
//!    - **Command Execution**: Isolated and tagged with the requesting agent ID.
//!
//! 3. **Optimistic File Consistency & Revision Tracking**:
//!    - Every file read by an agent records the file revision at that timestamp.
//!    - When an agent requests a write/replace, the Smart System verifies if any other agent has modified the file since the read.
//!    - If a conflict occurs, the write is **rejected** with a structured conflict notification, instructing the agent to re-read and reconcile.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Agent Identifier in the Hercules Swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentId {
    H0, // Main Agent
    H1, // Prompt Engineer
    H2, // Bug Hunter
    H3, // Coder / Implementer
    Custom(u32),
}

impl AgentId {
    pub fn name(self) -> &'static str {
        match self {
            AgentId::H0 => "H0 (Main)",
            AgentId::H1 => "H1 (Prompt Engineer)",
            AgentId::H2 => "H2 (Bug Hunter)",
            AgentId::H3 => "H3 (Coder)",
            AgentId::Custom(_) => "Sub-Agent",
        }
    }

    pub fn code(self) -> String {
        match self {
            AgentId::H0 => "H0".to_string(),
            AgentId::H1 => "H1".to_string(),
            AgentId::H2 => "H2".to_string(),
            AgentId::H3 => "H3".to_string(),
            AgentId::Custom(id) => format!("H{id}"),
        }
    }

    pub fn from_u32(n: u32) -> Self {
        match n {
            0 => AgentId::H0,
            1 => AgentId::H1,
            2 => AgentId::H2,
            3 => AgentId::H3,
            other => AgentId::Custom(other),
        }
    }
}

/// Metadata describing a tracked file's version in the workspace.
#[derive(Debug, Clone)]
pub struct FileRevision {
    pub revision: u64,
    pub last_modified_by: AgentId,
    pub last_modified_at: Instant,
    pub content_hash: u64,
    pub line_count: usize,
}

/// Result of an optimistic write attempt.
#[derive(Debug, Clone)]
pub enum SmartWriteResult {
    /// Write committed successfully at new revision.
    Committed {
        path: PathBuf,
        new_revision: u64,
        lines_written: usize,
    },
    /// Conflict detected! Stale write rejected.
    Conflict {
        path: PathBuf,
        agent_read_revision: u64,
        current_revision: u64,
        modified_by: AgentId,
        message: String,
    },
    /// Permission denied or IO error.
    Error(String),
}

/// The Hercules Smart System coordinator.
pub struct SmartSystem {
    /// Map of canonical canonicalized file paths to their current revision.
    revisions: RwLock<HashMap<PathBuf, FileRevision>>,
    /// Read snapshot tracker: maps (AgentId, PathBuf) -> revision when agent last read the file.
    agent_read_snapshots: RwLock<HashMap<(AgentId, PathBuf), u64>>,
    /// Global monotonic revision counter.
    next_revision: Mutex<u64>,
}

impl Default for SmartSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SmartSystem {
    pub fn new() -> Self {
        Self {
            revisions: RwLock::new(HashMap::new()),
            agent_read_snapshots: RwLock::new(HashMap::new()),
            next_revision: Mutex::new(1),
        }
    }

    /// Normalize and canonicalize target path relative to working directory.
    fn normalize_path(path: &Path) -> PathBuf {
        if let Ok(c) = path.canonicalize() {
            c
        } else {
            path.to_path_buf()
        }
    }

    /// Simple hash for content comparison.
    fn calculate_hash(content: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut s = DefaultHasher::new();
        content.hash(&mut s);
        s.finish()
    }

    /// Record a file read event by an agent.
    ///
    /// This establishes the baseline revision that future writes from this agent will be checked against.
    pub fn register_read(&self, agent: AgentId, path: &Path, content: &str) -> u64 {
        let norm_path = Self::normalize_path(path);
        let hash = Self::calculate_hash(content);
        let line_count = content.lines().count();

        let mut revs = self.revisions.write().unwrap();
        let rev = if let Some(existing) = revs.get_mut(&norm_path) {
            if existing.content_hash != hash {
                // External edit or first untracked write detected
                let mut next_rev = self.next_revision.lock().unwrap();
                *next_rev += 1;
                existing.revision = *next_rev;
                existing.content_hash = hash;
                existing.line_count = line_count;
                existing.last_modified_at = Instant::now();
            }
            existing.revision
        } else {
            let mut next_rev = self.next_revision.lock().unwrap();
            let r = *next_rev;
            *next_rev += 1;
            revs.insert(
                norm_path.clone(),
                FileRevision {
                    revision: r,
                    last_modified_by: agent,
                    last_modified_at: Instant::now(),
                    content_hash: hash,
                    line_count,
                },
            );
            r
        };

        // Record agent read snapshot
        let mut snaps = self.agent_read_snapshots.write().unwrap();
        snaps.insert((agent, norm_path), rev);
        rev
    }

    /// Optimistic write operation.
    ///
    /// Checks if the file was modified since the requesting agent last read it.
    /// If another agent updated the file in the interim, returns `SmartWriteResult::Conflict`.
    pub fn request_write(
        &self,
        agent: AgentId,
        path: &Path,
        line_range: Option<&str>,
        new_content: &str,
    ) -> SmartWriteResult {
        let norm_path = Self::normalize_path(path);

        // Check read snapshot
        let read_rev = {
            let snaps = self.agent_read_snapshots.read().unwrap();
            snaps.get(&(agent, norm_path.clone())).copied()
        };

        let mut revs = self.revisions.write().unwrap();
        if let Some(curr_rev) = revs.get(&norm_path) {
            if let Some(agent_rev) = read_rev {
                if agent_rev < curr_rev.revision && curr_rev.last_modified_by != agent {
                    // CONFLICT: Another agent modified this file after our read!
                    return SmartWriteResult::Conflict {
                        path: norm_path.clone(),
                        agent_read_revision: agent_rev,
                        current_revision: curr_rev.revision,
                        modified_by: curr_rev.last_modified_by,
                        message: format!(
                            "[SMART SYSTEM CONFLICT] Agent {} attempted to modify '{}' based on stale revision #{} \
                             (current is #{} modified by {}). Write rejected to prevent silent overwrite. \
                             Agent {} MUST re-read the file with <read src=\"{}\"> and reconcile its changes.",
                            agent.code(),
                            norm_path.display(),
                            agent_rev,
                            curr_rev.revision,
                            curr_rev.last_modified_by.code(),
                            agent.code(),
                            path.display(),
                        ),
                    };
                }
            }
        }

        // Perform actual write
        let write_res = crate::agent::AgentEngine::execute_write(
            path.to_str().unwrap_or(""),
            line_range,
            new_content,
        );

        if write_res.starts_with("Error:") || write_res.contains("Permission denied") {
            return SmartWriteResult::Error(write_res);
        }

        // Advance file revision
        let mut next_rev = self.next_revision.lock().unwrap();
        *next_rev += 1;
        let new_r = *next_rev;
        let line_count = new_content.lines().count();
        let hash = Self::calculate_hash(new_content);

        revs.insert(
            norm_path.clone(),
            FileRevision {
                revision: new_r,
                last_modified_by: agent,
                last_modified_at: Instant::now(),
                content_hash: hash,
                line_count,
            },
        );

        // Update writing agent's snapshot to latest revision
        let mut snaps = self.agent_read_snapshots.write().unwrap();
        snaps.insert((agent, norm_path.clone()), new_r);

        SmartWriteResult::Committed {
            path: norm_path,
            new_revision: new_r,
            lines_written: line_count,
        }
    }

    /// Invalidate snapshots for a deleted or replaced file.
    pub fn invalidate(&self, path: &Path) {
        let norm_path = Self::normalize_path(path);
        let mut revs = self.revisions.write().unwrap();
        revs.remove(&norm_path);
    }
}

// Global Smart System Instance
static GLOBAL_SMART_SYSTEM: Mutex<Option<Arc<SmartSystem>>> = Mutex::new(None);

pub fn get_smart_system() -> Arc<SmartSystem> {
    let mut lock = GLOBAL_SMART_SYSTEM.lock().unwrap();
    if lock.is_none() {
        *lock = Some(Arc::new(SmartSystem::new()));
    }
    lock.as_ref().unwrap().clone()
}
