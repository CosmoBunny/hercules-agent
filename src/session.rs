//! Session persistence and management for Hercules Agent.
//!
//! Sessions are stored in local app data:
//! - Linux / Unix: `$XDG_DATA_HOME/hercules/sessions` or `~/.local/share/hercules/sessions`
//! - macOS: `~/Library/Application Support/hercules/sessions`
//! - Windows: `%LOCALAPPDATA%\hercules\sessions`

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(windows)]
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Session {
    pub session_id: String,
    pub working_dir: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub messages: Vec<String>,
    #[serde(default)]
    pub selected_model: Option<String>,
}

impl Session {
    pub fn new(session_id: String, working_dir: String) -> Self {
        let now = now_unix();
        Self {
            session_id,
            working_dir,
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            selected_model: None,
        }
    }
}

/// Returns the cross-platform local app data directory for hercules sessions.
pub fn sessions_dir() -> PathBuf {
    if let Some(data_dir) = dirs::data_local_dir() {
        data_dir.join("hercules").join("sessions")
    } else if let Some(home) = dirs::home_dir() {
        home.join(".local").join("share").join("hercules").join("sessions")
    } else {
        PathBuf::from(".hercules_sessions")
    }
}

/// Generates a deterministic session ID tied to the given directory path.
/// Format: `{clean_dirname}-{8_char_path_hash}`
pub fn session_id_for_dir(dir: &Path) -> String {
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let path_str = canonical.to_string_lossy();

    let dir_name = canonical
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "root".to_string());

    let clean_dir_name: String = dir_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let clean_dir_name = if clean_dir_name.is_empty() {
        "workspace".to_string()
    } else {
        clean_dir_name
    };

    // 64-bit FNV-1a hash over canonical path
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in path_str.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let short_hash = format!("{:08x}", ((hash >> 32) ^ hash) as u32);

    format!("{}-{}", clean_dir_name, short_hash)
}

/// Generates a new unique session ID for the given directory so past sessions are preserved.
/// Format: `{clean_dirname}-{8_char_path_hash}-{hex_suffix}`
pub fn new_session_id_for_dir(dir: &Path) -> String {
    let base = session_id_for_dir(dir);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let millis = now.as_millis();
    let rand_suffix = (millis ^ 0x517cc1b727220a95) as u32;
    format!("{}-{:08x}", base, rand_suffix)
}

/// Returns the most recently updated saved session for the given directory, if any.
pub fn latest_session_for_dir(dir: &Path) -> Option<Session> {
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let canonical_str = canonical.to_string_lossy();
    let base_prefix = session_id_for_dir(dir);

    let sessions = list_sessions();
    sessions.into_iter().find(|s| {
        s.working_dir == canonical_str
            || s.session_id == base_prefix
            || s.session_id.starts_with(&format!("{}-", base_prefix))
    })
}

/// Path to the JSON file for a given session ID.
pub fn session_file_path(session_id: &str) -> PathBuf {
    sessions_dir().join(format!("{}.json", session_id))
}

/// Path to the lock file for a given session ID.
pub fn lock_file_path(session_id: &str) -> PathBuf {
    sessions_dir().join(format!("{}.lock", session_id))
}

/// Checks if a PID is still actively running.
pub fn is_pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(windows)]
    {
        let s = System::new_all();
        s.process(sysinfo::Pid::from_u32(pid)).is_some()
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

/// Checks if a session is currently locked by an active process.
pub fn is_session_locked(session_id: &str) -> bool {
    let lock_path = lock_file_path(session_id);
    if !lock_path.exists() {
        return false;
    }

    if let Ok(content) = fs::read_to_string(&lock_path) {
        if let Ok(pid) = content.trim().parse::<u32>() {
            if pid == std::process::id() {
                return true;
            }
            if is_pid_alive(pid) {
                return true;
            }
        }
    }

    // Stale lock file — clean it up
    let _ = fs::remove_file(lock_path);
    false
}

/// RAII lock guard that releases session lock when dropped.
pub struct SessionLockGuard {
    session_id: String,
}

impl Drop for SessionLockGuard {
    fn drop(&mut self) {
        release_session_lock(&self.session_id);
    }
}

/// Acquires an exclusive lock file for the given session ID.
pub fn acquire_session_lock(session_id: &str) -> Result<SessionLockGuard, String> {
    let dir = sessions_dir();
    let _ = fs::create_dir_all(&dir);

    if is_session_locked(session_id) {
        let lock_path = lock_file_path(session_id);
        let pid_str = fs::read_to_string(&lock_path).unwrap_or_else(|_| "?".to_string());
        return Err(format!(
            "Session '{}' is currently open/locked by process (PID: {}).",
            session_id,
            pid_str.trim()
        ));
    }

    let lock_path = lock_file_path(session_id);
    let pid = std::process::id();
    fs::write(&lock_path, pid.to_string())
        .map_err(|e| format!("Failed to acquire lock for session {}: {}", session_id, e))?;

    Ok(SessionLockGuard {
        session_id: session_id.to_string(),
    })
}

/// Releases the lock file for the given session ID.
pub fn release_session_lock(session_id: &str) {
    let lock_path = lock_file_path(session_id);
    if lock_path.exists() {
        if let Ok(content) = fs::read_to_string(&lock_path) {
            if let Ok(pid) = content.trim().parse::<u32>() {
                if pid == std::process::id() {
                    let _ = fs::remove_file(&lock_path);
                }
            } else {
                let _ = fs::remove_file(&lock_path);
            }
        }
    }
}

/// Saves the given session to disk.
pub fn save_session(session: &Session) -> Result<PathBuf, String> {
    let dir = sessions_dir();
    fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to create sessions directory {}: {}", dir.display(), e))?;

    let path = session_file_path(&session.session_id);
    let json = serde_json::to_string_pretty(session)
        .map_err(|e| format!("Failed to serialize session {}: {}", session.session_id, e))?;

    fs::write(&path, json)
        .map_err(|e| format!("Failed to write session file {}: {}", path.display(), e))?;

    Ok(path)
}

/// Loads a session by exact ID or prefix match.
pub fn load_session(session_id_or_prefix: &str) -> Option<Session> {
    let exact_path = session_file_path(session_id_or_prefix);
    if exact_path.exists() {
        if let Ok(content) = fs::read_to_string(&exact_path) {
            if let Ok(session) = serde_json::from_str::<Session>(&content) {
                return Some(session);
            }
        }
    }

    // Check directory for prefix match if exact file not found
    let dir = sessions_dir();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if stem == session_id_or_prefix || stem.starts_with(session_id_or_prefix) {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(session) = serde_json::from_str::<Session>(&content) {
                                return Some(session);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Loads an existing session or creates a new one for the specified session ID and directory.
pub fn load_or_create_session(session_id: &str, dir: &Path) -> Session {
    if let Some(existing) = load_session(session_id) {
        existing
    } else {
        let working_dir = dir
            .canonicalize()
            .unwrap_or_else(|_| dir.to_path_buf())
            .to_string_lossy()
            .to_string();
        Session::new(session_id.to_string(), working_dir)
    }
}

/// Lists all saved sessions sorted by most recently updated first.
pub fn list_sessions() -> Vec<Session> {
    let dir = sessions_dir();
    let mut sessions = Vec::new();

    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(session) = serde_json::from_str::<Session>(&content) {
                        sessions.push(session);
                    }
                }
            }
        }
    }

    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    sessions
}

/// Clears all unlocked sessions belonging to the specified directory.
/// Returns (cleared_count, locked_skipped_count).
pub fn clear_session_for_dir(dir: &Path) -> (usize, usize) {
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let canonical_str = canonical.to_string_lossy();
    let current_sid = session_id_for_dir(dir);

    let mut cleared = 0;
    let mut skipped = 0;

    let dir_path = sessions_dir();
    if let Ok(entries) = fs::read_dir(&dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    let matches_dir = stem == current_sid || {
                        if let Ok(content) = fs::read_to_string(&path) {
                            if let Ok(session) = serde_json::from_str::<Session>(&content) {
                                session.working_dir == canonical_str
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    };

                    if matches_dir {
                        if is_session_locked(stem) {
                            skipped += 1;
                        } else {
                            let _ = fs::remove_file(&path);
                            let _ = fs::remove_file(lock_file_path(stem));
                            cleared += 1;
                        }
                    }
                }
            }
        }
    }

    (cleared, skipped)
}

/// Clears all unlocked sessions across all directories.
/// Returns (cleared_count, locked_skipped_count).
pub fn clear_all_sessions() -> (usize, usize) {
    let mut cleared = 0;
    let mut skipped = 0;

    let dir_path = sessions_dir();
    if let Ok(entries) = fs::read_dir(&dir_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if is_session_locked(stem) {
                        skipped += 1;
                    } else {
                        let _ = fs::remove_file(&path);
                        let _ = fs::remove_file(lock_file_path(stem));
                        cleared += 1;
                    }
                }
            }
        }
    }

    (cleared, skipped)
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_session_id_deterministic() {
        let cwd = env::current_dir().unwrap();
        let id1 = session_id_for_dir(&cwd);
        let id2 = session_id_for_dir(&cwd);
        assert_eq!(id1, id2);
        assert!(id1.contains('-'));
    }

    #[test]
    fn test_session_save_and_load() {
        let test_session_id = "test-session-save-load-1234";
        let mut session = Session::new(test_session_id.to_string(), "/test/dir".to_string());
        session.messages.push("System: Hello".to_string());
        session.messages.push("You: How are you?".to_string());

        let save_res = save_session(&session);
        assert!(save_res.is_ok());

        let loaded = load_session(test_session_id);
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.session_id, test_session_id);
        assert_eq!(loaded.messages.len(), 2);

        let path = session_file_path(test_session_id);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_session_locking_and_clearing() {
        let temp_dir = env::temp_dir().join(format!("hercules-test-dir-{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);
        let sid = session_id_for_dir(&temp_dir);
        let session = Session::new(sid.clone(), temp_dir.to_string_lossy().to_string());
        let _ = save_session(&session);

        assert!(!is_session_locked(&sid));

        let guard = acquire_session_lock(&sid);
        assert!(guard.is_ok());
        assert!(is_session_locked(&sid));

        // When locked, clear for this dir should skip it
        let (cleared, skipped) = clear_session_for_dir(&temp_dir);
        assert_eq!(cleared, 0);
        assert_eq!(skipped, 1);

        drop(guard);
        assert!(!is_session_locked(&sid));

        // When unlocked, clear for this dir should delete it
        let (cleared, skipped) = clear_session_for_dir(&temp_dir);
        assert_eq!(cleared, 1);
        assert_eq!(skipped, 0);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_multiple_sessions_preserved() {
        let temp_dir = env::temp_dir().join(format!("hercules-multi-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&temp_dir);

        let sid1 = new_session_id_for_dir(&temp_dir);
        // Small delay to ensure timestamp difference
        std::thread::sleep(std::time::Duration::from_millis(5));
        let sid2 = new_session_id_for_dir(&temp_dir);
        assert_ne!(sid1, sid2);

        let mut s1 = Session::new(sid1.clone(), temp_dir.to_string_lossy().to_string());
        s1.messages.push("Conversation 1".to_string());
        let _ = save_session(&s1);

        let mut s2 = Session::new(sid2.clone(), temp_dir.to_string_lossy().to_string());
        s2.messages.push("Conversation 2".to_string());
        let _ = save_session(&s2);

        // Verify both sessions exist simultaneously and do not overwrite each other
        let loaded1 = load_session(&sid1).expect("Session 1 should exist");
        let loaded2 = load_session(&sid2).expect("Session 2 should exist");
        assert_eq!(loaded1.messages[0], "Conversation 1");
        assert_eq!(loaded2.messages[0], "Conversation 2");

        let (cleared, _) = clear_session_for_dir(&temp_dir);
        assert_eq!(cleared, 2);
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
