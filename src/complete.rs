//! Host-side input completion engine (no LLM): `/` slash commands, `@`
//! workspace paths (recursive), and the `$CURRENT` token.
//!
//! Pure functions over explicit inputs so the behavior is unit-testable
//! without the app. The UI layer only renders candidates, tracks the
//! selected index/scroll, and splices the accepted insert into
//! `[start..end)` — the token surrounding the cursor.
//!
//! Path roots bottom out at the same place tool execution does:
//! listings are read relative to the passed `cwd`, and `@…` inserts are
//! rewritten to `$CURRENT/…` form at submit time, which
//! [`crate::agent::AgentEngine::expand_path`] (the canonical resolver
//! tools use) expands against the process cwd.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Canonical slash commands: the exact set the Enter dispatcher executes
/// (name with slash, plus description). Single source of truth for
/// completion — do not invent commands (e.g. there is no `/run`).
pub const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "Show shortcuts & commands"),
    ("/allow", "Grant write/cmd for this session"),
    ("/swarm", "Spawn sub-agent swarm"),
    ("/compact", "Compress chat to memory"),
    ("/compact!", "Compress chat to memory (aggressive)"),
    ("/gc", "Compress chat to memory (alias)"),
    ("/tasks", "List background jobs"),
    ("/cancel-download", "Cancel active model download"),
    ("/cancel_download", "Cancel active model download (alias)"),
    ("/cdl", "Cancel active model download (alias)"),
    ("/download-status", "Show download progress"),
    ("/dlstatus", "Show download progress (alias)"),
    ("/copy", "Copy conversation/chip to clipboard"),
    ("/theme", "Set theme color"),
    ("/save", "Save session"),
    ("/load", "Load session"),
];

/// What kind of token the cursor is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionTrigger {
    /// `/…` slash command.
    Command,
    /// Filesystem path (`@…`, `~…`, `/abs`, `./`, `../`).
    File,
    /// `$CURRENT…` virtual token.
    CurrentPath,
}

/// Candidate kind for display tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    Command,
    File,
    Dir,
}

/// One completion candidate. `insert_text` is the FULL replacement for
/// the token range `[start..end)` (sigils included, e.g. `@src/main.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionCandidate {
    pub insert_text: String,
    pub display_text: String,
    pub kind: CandidateKind,
    pub description: Option<String>,
}

/// The token surrounding the cursor plus its candidates. Char indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionContext {
    pub trigger: CompletionTrigger,
    pub query: String,
    pub start: usize,
    pub end: usize,
    pub candidates: Vec<CompletionCandidate>,
}

const MAX_CANDIDATES: usize = 15;

/// Split `text` into the whitespace-delimited token surrounding the
/// cursor (char index). Returns `(start, end)` char bounds.
fn token_range(text: &str, cursor: usize) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut start = cursor;
    while start > 0 && !chars[start - 1].is_whitespace() {
        start -= 1;
    }
    let mut end = cursor;
    while end < chars.len() && !chars[end].is_whitespace() {
        end += 1;
    }
    (start, end)
}

/// Classify the token + build candidates. Pure: filesystem reads go
/// through `lister`, so tests inject fixtures without touching cwd.
fn classify(
    token: &str,
    cwd: &Path,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Option<(CompletionTrigger, Vec<CompletionCandidate>)> {
    if token.starts_with('/') {
        if token[1..].contains('/') {
            // Absolute path with subdirectories: list directly.
            return Some((CompletionTrigger::File, fs_candidates(token, lister)));
        }
        // `/…` bare: commands win when one matches, otherwise list `/`.
        let cmds: Vec<CompletionCandidate> = SLASH_COMMANDS
            .iter()
            .filter(|(name, _)| name.starts_with(token))
            .map(|(name, desc)| CompletionCandidate {
                insert_text: name.to_string(),
                display_text: name.to_string(),
                kind: CandidateKind::Command,
                description: Some(desc.to_string()),
            })
            .collect();
        if !cmds.is_empty() {
            return Some((CompletionTrigger::Command, cmds));
        }
        return Some((CompletionTrigger::File, fs_candidates(token, lister)));
    }
    if let Some(rest) = token.strip_prefix('@') {
        return Some((CompletionTrigger::File, at_candidates(cwd, rest, lister)));
    }
    if token.starts_with('$') {
        return Some((
            CompletionTrigger::CurrentPath,
            current_candidates(cwd, token, lister),
        ));
    }
    if token.starts_with('~') || token.starts_with("./") || token.starts_with("../") {
        return Some((CompletionTrigger::File, fs_candidates(token, lister)));
    }
    None
}

/// Build the completion context for the token surrounding `cursor`.
/// Returns the token bounds (char indices) plus candidates whose insert
/// replaces exactly `[start..end)`. `lister` supplies single-level
/// directory entries (the App passes its brief dir cache).
pub fn context_for_with(
    cwd: &Path,
    text: &str,
    cursor: usize,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Option<CompletionContext> {
    let chars: Vec<char> = text.chars().collect();
    let cursor = cursor.min(chars.len());
    let (start, end) = token_range(text, cursor);
    if start == end {
        return None;
    }
    let token: String = chars[start..end].iter().collect();
    let frag: String = chars[start..cursor].iter().collect();
    let (trigger, mut candidates) = classify(&token, cwd, lister)?;
    // Narrow to what was actually typed before the cursor
    // (`read @src/ma|in.rs` completes on `@src/ma`).
    candidates.retain(|c| candidate_matches(c, &frag));
    candidates.truncate(MAX_CANDIDATES);
    if candidates.is_empty() {
        return None;
    }
    Some(CompletionContext {
        trigger,
        query: frag,
        start,
        end,
        candidates,
    })
}

/// Convenience wrapper with direct filesystem reads (tests, one-shots).
pub fn context_for(cwd: &Path, text: &str, cursor: usize) -> Option<CompletionContext> {
    context_for_with(cwd, text, cursor, &mut |dir| read_dir_entries(dir))
}

/// Does a candidate survive the typed fragment? Commands match by prefix;
/// path inserts match when the fragment is a case-insensitive prefix of
/// the insert (inserts are built from the fragment, so this is a safety
/// net, not the primary filter).
fn candidate_matches(c: &CompletionCandidate, frag: &str) -> bool {
    c.insert_text
        .to_lowercase()
        .starts_with(&frag.to_lowercase())
}

/// `@…` inserts: `rest` is the token after `@`, already `/`-stripped of a
/// leading slash. Inserts keep the `@` sigil plus any typed directory
/// prefix, so acceptance is a pure range splice.
fn at_candidates(
    cwd: &Path,
    rest: &str,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Vec<CompletionCandidate> {
    let rest = rest.strip_prefix('/').unwrap_or(rest);
    // `@$…` forms resolve against cwd with the token stripped.
    let (sigil_prefix, rest) = if rest == "$CURRENT" || rest == "$CURRENT/" {
        return vec![CompletionCandidate {
            insert_text: "@$CURRENT/".to_string(),
            display_text: "$CURRENT/".to_string(),
            kind: CandidateKind::Dir,
            description: Some("workspace root".to_string()),
        }];
    } else if let Some(sub) = rest.strip_prefix("$CURRENT/") {
        ("@$CURRENT/", sub)
    } else if rest == "$" || ("$CURRENT/".starts_with(rest) && rest.starts_with('$')) {
        return vec![CompletionCandidate {
            insert_text: "@$CURRENT/".to_string(),
            display_text: "$CURRENT/".to_string(),
            kind: CandidateKind::Dir,
            description: Some("workspace root".to_string()),
        }];
    } else {
        ("@", rest)
    };
    let (dir_part, filter) = match rest.rfind('/') {
        Some(i) => (&rest[..=i], rest[i + 1..].to_string()),
        None => ("", rest.to_string()),
    };
    // Filtered, sorted, capped listing of cwd/dir_part.
    at_like_filtered(cwd, dir_part, &filter, sigil_prefix, lister)
}

/// `$…` inserts against cwd.
fn current_candidates(
    cwd: &Path,
    token: &str,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Vec<CompletionCandidate> {
    if "$CURRENT".starts_with(token) {
        // `$`, `$C`, … → the token itself.
        return vec![CompletionCandidate {
            insert_text: "$CURRENT".to_string(),
            display_text: "$CURRENT".to_string(),
            kind: CandidateKind::Dir,
            description: Some("workspace root".to_string()),
        }];
    }
    if token == "$CURRENT/" {
        return at_like(cwd, "", "$CURRENT/", lister);
    }
    if let Some(rest) = token.strip_prefix("$CURRENT/") {
        let (dir_part, filter) = match rest.rfind('/') {
            Some(i) => (&rest[..=i], rest[i + 1..].to_string()),
            None => ("", rest.to_string()),
        };
        return at_like_filtered(cwd, dir_part, &filter, "$CURRENT/", lister);
    }
    Vec::new()
}

/// Shared filtered listing with an arbitrary sigil prefix.
fn at_like(
    cwd: &Path,
    dir_part: &str,
    sigil_prefix: &str,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Vec<CompletionCandidate> {
    at_like_filtered(cwd, dir_part, "", sigil_prefix, lister)
}

fn at_like_filtered(
    cwd: &Path,
    dir_part: &str,
    filter: &str,
    sigil_prefix: &str,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Vec<CompletionCandidate> {
    let base = cwd.join(dir_part);
    let mut out = Vec::new();
    for (name, is_dir) in lister(&base) {
        if !visible(&name, filter) || !name.to_lowercase().starts_with(&filter.to_lowercase()) {
            continue;
        }
        let (insert, display) = if is_dir {
            (
                format!("{sigil_prefix}{dir_part}{name}/"),
                format!("{name}/"),
            )
        } else {
            (format!("{sigil_prefix}{dir_part}{name}"), name.clone())
        };
        out.push(CompletionCandidate {
            insert_text: insert,
            display_text: display,
            kind: if is_dir {
                CandidateKind::Dir
            } else {
                CandidateKind::File
            },
            description: None,
        });
    }
    out.sort_by(|a, b| a.display_text.cmp(&b.display_text));
    out.truncate(MAX_CANDIDATES);
    out
}

/// `~`, `/abs`, `./`, `../` tokens: inserts preserve the typed prefix.
fn fs_candidates(
    token: &str,
    lister: &mut dyn FnMut(&Path) -> Vec<(String, bool)>,
) -> Vec<CompletionCandidate> {
    let (raw_base, filter) = match token.rfind('/') {
        Some(i) => (&token[..=i], token[i + 1..].to_string()),
        None => ("", token.to_string()),
    };
    // Resolve the base the same way the shell/host would.
    let base: PathBuf = if let Some(rest) = raw_base.strip_prefix('~') {
        dirs_home().join(rest.trim_start_matches('/'))
    } else if raw_base.starts_with('/') {
        PathBuf::from(raw_base)
    } else {
        // `./`, `../` — relative to the process cwd, like the legacy
        // path branch they replace.
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(raw_base)
    };
    let mut out = Vec::new();
    for (name, is_dir) in lister(&base) {
        if !visible(&name, &filter) || !name.to_lowercase().starts_with(&filter.to_lowercase()) {
            continue;
        }
        let insert = if is_dir {
            format!("{raw_base}{name}/")
        } else {
            format!("{raw_base}{name}")
        };
        out.push(CompletionCandidate {
            insert_text: insert,
            display_text: if is_dir {
                format!("{name}/")
            } else {
                name.clone()
            },
            kind: if is_dir {
                CandidateKind::Dir
            } else {
                CandidateKind::File
            },
            description: None,
        });
    }
    out.sort_by(|a, b| a.display_text.cmp(&b.display_text));
    out.truncate(MAX_CANDIDATES);
    out
}

fn dirs_home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// Raw directory listing: `(name, is_dir)`, dotfiles included (callers
/// apply the dotfile rule), unsorted — sorting happens per query.
fn read_dir_entries(dir: &Path) -> Vec<(String, bool)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            (name, e.path().is_dir())
        })
        .collect()
}

/// Dotfile rule shared by all path listings.
fn visible(name: &str, filter: &str) -> bool {
    if name.starts_with('.') && !filter.starts_with('.') {
        return false;
    }
    filter.is_empty() || name.to_lowercase().contains(&filter.to_lowercase())
}

/// Briefly-cached directory lister: single-level enumeration only, never
/// recursive; entries refresh when the directory mtime changes or the TTL
/// lapses. The UI holds one and passes it per triage.
pub struct DirCache {
    ttl: Duration,
    map: HashMap<PathBuf, (Instant, u64, Vec<(String, bool)>)>,
}

impl DirCache {
    pub fn new() -> Self {
        Self {
            ttl: Duration::from_millis(750),
            map: HashMap::new(),
        }
    }

    fn mtime(dir: &Path) -> u64 {
        std::fs::metadata(dir)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    pub fn entries(&mut self, dir: &Path) -> Vec<(String, bool)> {
        let now = Instant::now();
        let mt = Self::mtime(dir);
        if let Some((at, old_mt, cached)) = self.map.get(dir) {
            if *old_mt == mt && now.duration_since(*at) < self.ttl {
                return cached.clone();
            }
        }
        let mut entries = read_dir_entries(dir);
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        self.map
            .insert(dir.to_path_buf(), (now, mt, entries.clone()));
        if self.map.len() > 64 {
            // Bounded: drop the oldest entry.
            if let Some(oldest) = self
                .map
                .iter()
                .min_by_key(|(_, (at, _, _))| *at)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&oldest);
            }
        }
        entries
    }
}

impl Default for DirCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Rewrite token-initial `@…` paths to `$CURRENT/…` form for model
/// submission. Only tokens starting with `@` (at buffer start or after
/// whitespace) are touched; everything else passes byte-identical.
pub fn resolve_at_paths(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut first = true;
    for tok in input.split_whitespace() {
        if !first {
            out.push(' ');
        }
        first = false;
        if let Some(rest) = tok.strip_prefix('@') {
            let rest = rest.strip_prefix('/').unwrap_or(rest);
            if rest.is_empty() || rest == "$" {
                // Lone `@`: not a path, leave it for the user to finish.
                out.push_str(tok);
            } else if rest == "$CURRENT" || rest == "$CURRENT/" {
                out.push_str("$CURRENT/");
            } else {
                let rest = rest.strip_prefix("$CURRENT/").unwrap_or(rest);
                out.push_str("$CURRENT/");
                out.push_str(rest);
            }
        } else {
            out.push_str(tok);
        }
    }
    // Preserve a single trailing space (it carried no fragment anyway).
    if input.ends_with(char::is_whitespace) && !input.is_empty() {
        out.push(' ');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn fixture(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hercules-complete2-{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src/agent")).unwrap();
        fs::write(dir.join("src/main.rs"), "fn main(){}").unwrap();
        fs::write(dir.join("Cargo.toml"), "[pkg]").unwrap();
        fs::write(dir.join("README.md"), "hi").unwrap();
        dir
    }

    fn ctx(cwd: &Path, text: &str) -> Option<CompletionContext> {
        context_for(cwd, text, text.chars().count())
    }

    fn inserts(c: &CompletionContext) -> Vec<String> {
        c.candidates.iter().map(|x| x.insert_text.clone()).collect()
    }

    #[test]
    fn command_slash_forms() {
        let cwd = PathBuf::from("/tmp");
        let c = ctx(&cwd, "/").expect("bare slash lists");
        assert_eq!(c.trigger, CompletionTrigger::Command);
        assert!(inserts(&c).contains(&"/help".to_string()));
        assert!(inserts(&c).contains(&"/compact".to_string()));

        let c = ctx(&cwd, "/he").expect("prefix filters");
        assert_eq!(inserts(&c), vec!["/help".to_string()]);

        let c = ctx(&cwd, "/co").expect("prefix filters");
        let got = inserts(&c);
        assert!(got.contains(&"/compact".to_string()));
        assert!(got.contains(&"/copy".to_string()));
        assert!(!got.iter().any(|s| s == "/help"));

        // Every completion is a real dispatchable command.
        for name in SLASH_COMMANDS.iter().map(|(n, _)| n) {
            assert!(name.starts_with('/'));
        }
        // No phantom commands.
        assert!(
            !inserts(&ctx(&cwd, "/").unwrap())
                .iter()
                .any(|s| s == "/run")
        );
    }

    #[test]
    fn file_at_forms() {
        let dir = fixture("file");
        let c = ctx(&dir, "@").expect("@ alone lists");
        assert_eq!(c.trigger, CompletionTrigger::File);
        let got = inserts(&c);
        assert!(got.contains(&"@src/".to_string()));
        assert!(got.contains(&"@Cargo.toml".to_string()));

        let c = ctx(&dir, "@src/").expect("dir lists children");
        assert_eq!(
            inserts(&c),
            vec!["@src/agent/".to_string(), "@src/main.rs".to_string()]
        );

        let c = ctx(&dir, "@src/ma").expect("prefix filters");
        assert_eq!(inserts(&c), vec!["@src/main.rs".to_string()]);

        let c = ctx(&dir, "@Cargo").expect("file prefix");
        assert_eq!(inserts(&c), vec!["@Cargo.toml".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn current_token_forms() {
        let dir = fixture("cur");
        let c = ctx(&dir, "$CUR").expect("partial token");
        assert_eq!(c.trigger, CompletionTrigger::CurrentPath);
        assert_eq!(inserts(&c), vec!["$CURRENT".to_string()]);

        let c = ctx(&dir, "$CURRENT/").expect("token root lists");
        assert!(inserts(&c).contains(&"$CURRENT/src/".to_string()));

        let c = ctx(&dir, "$CURRENT/src/").expect("subpath lists");
        assert_eq!(
            inserts(&c),
            vec![
                "$CURRENT/src/agent/".to_string(),
                "$CURRENT/src/main.rs".to_string()
            ]
        );

        let c = ctx(&dir, "$CURRENT/src/ma").expect("subpath filters");
        assert_eq!(inserts(&c), vec!["$CURRENT/src/main.rs".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn mixed_text_uses_token_around_cursor() {
        let dir = fixture("mixed");
        let c = ctx(&dir, "read @src/ma").expect("mid-sentence token");
        assert_eq!((c.start, c.end), (5, 12));
        assert_eq!(inserts(&c), vec!["@src/main.rs".to_string()]);

        let c = ctx(&dir, "inspect $CURRENT/src/ma").expect("mixed current");
        assert_eq!(inserts(&c), vec!["$CURRENT/src/main.rs".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cursor_mid_token_replaces_whole_token() {
        let dir = fixture("cursor");
        // "read @src/ma|in.rs": cursor after "ma", token spans the rest.
        let text = "read @src/main.rs";
        let cursor = "read @src/ma".chars().count();
        let c = context_for(&dir, text, cursor).expect("mid-token context");
        assert_eq!((c.start, c.end), (5, 17));
        assert_eq!(inserts(&c), vec!["@src/main.rs".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_insert_keeps_trailing_slash() {
        let dir = fixture("dirslash");
        let c = ctx(&dir, "@sr").expect("dir prefix");
        assert_eq!(inserts(&c), vec!["@src/".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn no_match_is_clean_none() {
        let dir = fixture("nomatch");
        assert!(ctx(&dir, "@zzz-no-such").is_none());
        assert!(ctx(&dir, "$CURRENT/zzz-no-such").is_none());
        assert!(ctx(&dir, "/zzz-no-such").is_none());
        assert!(ctx(&dir, "plain words").is_none());
        assert!(ctx(&dir, "").is_none());
        assert!(ctx(&dir, "trailing ").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_rewrites_only_at_tokens() {
        assert_eq!(
            resolve_at_paths("read @src/main.rs please"),
            "read $CURRENT/src/main.rs please"
        );
        assert_eq!(
            resolve_at_paths("@Cargo.toml and @README.md"),
            "$CURRENT/Cargo.toml and $CURRENT/README.md"
        );
        assert_eq!(resolve_at_paths("mail a@b.co here"), "mail a@b.co here");
        assert_eq!(resolve_at_paths("@$CURRENT/y"), "$CURRENT/y");
        assert_eq!(resolve_at_paths("@"), "@");
    }

    #[test]
    fn completion_root_matches_tool_resolver() {
        // One canonical root: the engine lists cwd, expand_path (what
        // tools use) resolves $CURRENT against the process cwd.
        let cwd = std::env::current_dir().unwrap();
        let expanded = crate::agent::AgentEngine::expand_path("$CURRENT/src/main.rs");
        assert_eq!(expanded, cwd.join("src/main.rs"));
    }
}
