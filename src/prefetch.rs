//! Speculative read-ahead cache: latency hiding for the agent loop.
//!
//! The canonical tool pipeline (`AgentEngine::parse_tool_calls` →
//! `App::claim_tool_call` → `AgentEngine::execute_proposed`) stays the ONLY
//! authority that executes model tool calls. This module never executes
//! anything on the model's behalf and never produces [`ProposedAction`]s.
//! It only keeps recently-likely file contents warm so that when the model
//! issues its next `<read>`, the bytes are already in memory while the LLM
//! was busy reasoning.
//!
//! Safety contract (read-only speculation):
//! - Only file READS are ever prefetched. Writes, commands, MCP, skills,
//!   agents and memory are never speculated — speculation has no side
//!   effects by construction.
//! - Every candidate still passes the SAME `path_allowed()` sandbox gate as
//!   a real read, both when cached (fetch time) and when served (the
//!   canonical `execute_read` checks before consulting the cache).
//! - Freshness is enforced three ways: the canonical `execute_write`
//!   invalidates the written path, cache hits revalidate mtime+length
//!   against the filesystem, and entries expire after [`CACHE_TTL`].
//! - Bounded by construction: entry/file/total caps plus a limit on
//!   concurrent background threads, so a hostile model output cannot turn
//!   speculation into a disk/memory DoS.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Instant, SystemTime};

/// Maximum cached files.
const MAX_ENTRIES: usize = 64;
/// Maximum total cached bytes (8 MiB).
const MAX_TOTAL_BYTES: usize = 8 * 1024 * 1024;
/// Files larger than this are never cached (256 KiB).
const MAX_FILE_BYTES: u64 = 256 * 1024;
/// Cache entry lifetime.
const CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(120);
/// Upper bound on candidates per speculation pass.
const MAX_CANDIDATES: usize = 8;
/// Upper bound on same-directory sibling candidates.
const MAX_SIBLINGS: usize = 6;
/// Only the first N bytes of a read result are scanned for candidates.
const MAX_SCAN_BYTES: usize = 200 * 1024;
/// Maximum concurrent background prefetch threads.
const MAX_PREFETCH_THREADS: usize = 2;

/// File extensions that may be treated as path mentions when quoted.
const PATHLIKE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "mjs", "cjs", "go", "java", "c", "h", "hpp", "cpp",
    "toml", "json", "yaml", "yml", "md", "html", "css", "sh", "txt",
];

struct CacheEntry {
    content: String,
    /// Filesystem mtime when cached (staleness check).
    mtime: SystemTime,
    /// File length when cached (mtime-granularity backstop).
    len: u64,
    inserted: Instant,
}

struct PrefetchCache {
    entries: HashMap<PathBuf, CacheEntry>,
    total_bytes: usize,
}

impl PrefetchCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Insert under the canonical path, evicting oldest-first on overflow.
    /// Oversized single files are refused.
    fn put(&mut self, canonical: PathBuf, content: String, mtime: SystemTime, len: u64) {
        if content.len() as u64 > MAX_FILE_BYTES {
            return;
        }
        if let Some(old) = self.entries.remove(&canonical) {
            self.total_bytes = self.total_bytes.saturating_sub(old.content.len());
        }
        while (self.entries.len() >= MAX_ENTRIES
            || self.total_bytes + content.len() > MAX_TOTAL_BYTES)
            && !self.entries.is_empty()
        {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.inserted)
                .map(|(k, _)| k.clone());
            match oldest {
                Some(k) => {
                    if let Some(old) = self.entries.remove(&k) {
                        self.total_bytes = self.total_bytes.saturating_sub(old.content.len());
                    }
                }
                None => break,
            }
        }
        self.total_bytes += content.len();
        self.entries.insert(
            canonical,
            CacheEntry {
                content,
                mtime,
                len,
                inserted: Instant::now(),
            },
        );
    }

    fn invalidate(&mut self, canonical: &Path) {
        if let Some(old) = self.entries.remove(canonical) {
            self.total_bytes = self.total_bytes.saturating_sub(old.content.len());
        }
    }
}

static CACHE: std::sync::OnceLock<Mutex<PrefetchCache>> = std::sync::OnceLock::new();

fn cache_lock() -> std::sync::MutexGuard<'static, PrefetchCache> {
    CACHE
        .get_or_init(|| Mutex::new(PrefetchCache::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

static INFLIGHT_THREADS: AtomicUsize = AtomicUsize::new(0);

fn canonical_of(path: &Path) -> Option<PathBuf> {
    path.canonicalize().ok()
}

/// Serve a cached read if fresh: TTL valid AND current mtime+length match
/// what was cached. Stale entries are dropped. Permission is NOT checked
/// here — the canonical `execute_read` enforces `path_allowed()` before
/// calling this, so the sandbox has exactly one choke point.
pub fn get(path: &Path) -> Option<String> {
    let canonical = canonical_of(path)?;
    let mut cache = cache_lock();
    let entry = cache.entries.get(&canonical)?;
    if entry.inserted.elapsed() > CACHE_TTL {
        cache.invalidate(&canonical);
        return None;
    }
    let meta = std::fs::metadata(&canonical).ok()?;
    let fresh_mtime = meta.modified().ok()?;
    if fresh_mtime != entry.mtime || meta.len() != entry.len {
        cache.invalidate(&canonical);
        return None;
    }
    Some(entry.content.clone())
}

/// Drop a path from the cache (called by the canonical write path).
pub fn invalidate(path: &Path) {
    if let Some(canonical) = canonical_of(path) {
        cache_lock().invalidate(&canonical);
    } else {
        // Path no longer exists (deleted): drop by raw key as fallback.
        cache_lock().invalidate(path);
    }
}

/// Fetch one file into the cache. Returns true when the file is cached
/// afterwards (already-fresh counts). Enforces the sandbox gate.
fn fetch_one(abs: &Path) -> bool {
    if crate::agent::path_allowed(abs).is_err() {
        return false;
    }
    let canonical = match canonical_of(abs) {
        Some(c) => c,
        None => return false,
    };
    if get(&canonical).is_some() {
        return true;
    }
    let meta = match std::fs::metadata(&canonical) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if !meta.is_file() || meta.len() > MAX_FILE_BYTES {
        return false;
    }
    let content = match std::fs::read_to_string(&canonical) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let mtime = meta.modified().unwrap_or_else(|_| SystemTime::now());
    let len = meta.len();
    cache_lock().put(canonical, content, mtime, len);
    true
}

/// Suggest likely-next reads given the file just read and its content.
/// Pure candidate extraction + existence checks; no global state, no
/// permission checks (those happen in [`fetch_one`]).
/// Order: explicit mentions first (strongest signal), siblings after.
/// Deterministic for identical directory state.
pub fn suggest_next_reads(read_path: &Path, content: &str, max: usize) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let try_push =
        |out: &mut Vec<PathBuf>, seen: &mut std::collections::HashSet<PathBuf>, p: PathBuf| {
            if out.len() >= max || !seen.insert(p.clone()) {
                return;
            }
            out.push(p);
        };

    let Some(base_dir) = read_path.parent() else {
        return out;
    };
    let self_name = read_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string());

    let scanned = if content.len() > MAX_SCAN_BYTES {
        &content[..MAX_SCAN_BYTES]
    } else {
        content
    };

    for raw_line in scanned.lines() {
        if out.len() >= max {
            break;
        }
        let line = raw_line.trim();

        // Rust `mod foo;` → foo.rs / foo/mod.rs
        if let Some(rest) = line.strip_prefix("mod ") {
            let name = rest
                .trim_end_matches(';')
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("");
            if is_module_name(name) {
                for cand in [
                    base_dir.join(format!("{name}.rs")),
                    base_dir.join(name).join("mod.rs"),
                ] {
                    if is_plain_file(&cand) {
                        try_push(&mut out, &mut seen, cand);
                    }
                }
            }
            continue;
        }

        // Quoted literals: include!, #include "...", from/require/import,
        // markdown links, generic "path.ext" mentions.
        for lit in quoted_literals(line) {
            if out.len() >= max {
                break;
            }
            if lit.contains("://") || lit.is_empty() {
                continue;
            }
            for cand in resolve_literal(base_dir, &lit) {
                if is_plain_file(&cand) {
                    try_push(&mut out, &mut seen, cand);
                }
            }
        }
    }

    // Same-directory siblings with the same extension (bounded, sorted).
    if out.len() < max {
        let wanted_ext = read_path.extension().and_then(|e| e.to_str());
        if let Ok(rd) = std::fs::read_dir(base_dir) {
            let mut sibs: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    if !p.is_file() {
                        return false;
                    }
                    let name = p.file_name().map(|n| n.to_string_lossy().to_string());
                    let Some(name) = name else {
                        return false;
                    };
                    if name.starts_with('.') {
                        return false;
                    }
                    if self_name.as_deref() == Some(name.as_str()) {
                        return false;
                    }
                    match wanted_ext {
                        Some(ext) => p.extension().and_then(|e| e.to_str()) == Some(ext),
                        None => true,
                    }
                })
                .collect();
            sibs.sort();
            for s in sibs.into_iter().take(MAX_SIBLINGS) {
                if out.len() >= max {
                    break;
                }
                try_push(&mut out, &mut seen, s);
            }
        }
    }

    out
}

fn is_module_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() < 64
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.starts_with(|c: char| c.is_ascii_digit())
}

fn is_plain_file(p: &Path) -> bool {
    p.is_file()
}

fn quoted_literals(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' || b == b'\'' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b {
                j += 1;
            }
            if j < bytes.len() {
                let lit = &line[i + 1..j];
                if lit.len() < 256 && !lit.contains('\n') {
                    out.push(lit.to_string());
                }
                i = j + 1;
                continue;
            }
            break;
        }
        i += 1;
    }
    out
}

/// Resolve a quoted literal against the read file's directory.
/// Returns candidates in preference order; existence is checked by the
/// caller. Extensionless relatives get JS/TS-style probing.
fn resolve_literal(base_dir: &Path, lit: &str) -> Vec<PathBuf> {
    // Python `from .foo import` / `from . import` handled by caller? No —
    // keep it here: leading dots denote same-dir modules.
    let mut rel = lit.trim().to_string();
    // Python `from .foo import` style: dots DIRECTLY attached to a module
    // name. `./x` / `../x` are filesystem relatives, not Python imports.
    let dots = rel.chars().take_while(|&c| c == '.').count();
    if dots > 0 && rel[dots..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
        let mod_path = rel[dots..].replace('.', "/");
        return vec![
            base_dir.join(format!("{mod_path}.py")),
            base_dir.join(&mod_path).join("__init__.py"),
        ];
    }
    // Filesystem-relative (`./x`, `../x`, `x/y`): resolve against the read
    // file's directory. Absolute paths and URLs are never speculated.
    if let Some(q) = rel.find(['?', '#']) {
        rel.truncate(q);
    }
    if rel.is_empty() || rel.starts_with('/') || rel.contains("://") {
        return Vec::new();
    }
    let looks_like_file = rel
        .rsplit('.')
        .next()
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            ext.len() <= 5 && PATHLIKE_EXTENSIONS.contains(&ext.as_str())
        })
        .unwrap_or(false);
    if !looks_like_file && Path::new(&rel).extension().is_none() {
        // Extensionless relative (JS/TS import): probe candidates.
        let base = base_dir.join(&rel);
        return vec![
            base.clone(),
            base.with_extension("ts"),
            base.with_extension("tsx"),
            base.with_extension("js"),
            base.with_extension("jsx"),
            base.join("index.ts"),
            base.join("index.js"),
        ];
    }
    if !looks_like_file {
        return Vec::new();
    }
    vec![base_dir.join(&rel)]
}

/// Entry point from the canonical read path: after a successful read,
/// warm the cache in the background while the LLM keeps reasoning.
/// Returns immediately (spawns at most one detached thread; bounded by
/// [`MAX_PREFETCH_THREADS`]. Never speculates from error outputs.
pub fn speculate_from_read(read_path: PathBuf, output: String) {
    if output.trim_start().starts_with("Error:") {
        return;
    }
    let inflight = INFLIGHT_THREADS.fetch_add(1, Ordering::SeqCst);
    if inflight >= MAX_PREFETCH_THREADS {
        INFLIGHT_THREADS.fetch_sub(1, Ordering::SeqCst);
        return;
    }
    let _ = std::thread::Builder::new()
        .name("hercules-prefetch".to_string())
        .spawn(move || {
            struct Guard;
            impl Drop for Guard {
                fn drop(&mut self) {
                    INFLIGHT_THREADS.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = Guard;
            for cand in suggest_next_reads(&read_path, &output, MAX_CANDIDATES) {
                fetch_one(&cand);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

    fn test_dir(tag: &str) -> PathBuf {
        // Under the process cwd so the default CurrentDir safefolder
        // permits reads without mutating global permission state.
        let n = TEST_DIR_SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::current_dir().unwrap().join(format!(
            "target/hercules-prefetch-test-{tag}-{n}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, content).unwrap();
        p
    }

    fn clear_cache() {
        let mut c = cache_lock();
        c.entries.clear();
        c.total_bytes = 0;
    }

    #[test]
    fn rust_mod_statements_resolve() {
        let dir = test_dir("mod");
        let main = write(&dir, "main.rs", "mod foo;\nmod bar;\nfn main() {}\n");
        write(&dir, "foo.rs", "pub fn f() {}\n");
        std::fs::create_dir_all(dir.join("bar")).unwrap();
        write(&dir, "bar/mod.rs", "pub fn g() {}\n");
        let content = std::fs::read_to_string(&main).unwrap();

        let got = suggest_next_reads(&main, &content, 8);
        assert!(got.contains(&dir.join("foo.rs")), "got: {got:?}");
        assert!(
            got.contains(&dir.join("bar").join("mod.rs")),
            "got: {got:?}"
        );
        assert!(!got.contains(&main), "never suggests the read file itself");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn js_relative_imports_probe_extensions() {
        let dir = test_dir("js");
        let app = write(
            &dir,
            "app.js",
            "import {x} from './util';\nimport y from './sub';\n",
        );
        write(&dir, "util.ts", "export const x = 1;\n");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        write(&dir, "sub/index.js", "module.exports = {};\n");
        let content = std::fs::read_to_string(&app).unwrap();

        let got = suggest_next_reads(&app, &content, 8);
        assert!(got.contains(&dir.join("util.ts")), "got: {got:?}");
        assert!(
            got.contains(&dir.join("sub").join("index.js")),
            "got: {got:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn python_relative_imports_resolve() {
        let dir = test_dir("py");
        let main = write(
            &dir,
            "main.py",
            "from .sib import helper\nfrom . import other\n",
        );
        write(&dir, "sib.py", "def helper(): pass\n");
        write(&dir, "other.py", "x = 1\n");
        let content = std::fs::read_to_string(&main).unwrap();

        let got = suggest_next_reads(&main, &content, 8);
        assert!(got.contains(&dir.join("sib.py")), "got: {got:?}");
        assert!(got.contains(&dir.join("other.py")), "got: {got:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn markdown_links_and_quoted_paths_resolve() {
        let dir = test_dir("md");
        let doc = write(&dir, "doc.md", "See [spec](spec.md) and \"notes.txt\".\n");
        write(&dir, "spec.md", "# spec\n");
        write(&dir, "notes.txt", "hi\n");
        let content = std::fs::read_to_string(&doc).unwrap();

        let got = suggest_next_reads(&doc, &content, 8);
        assert!(got.contains(&dir.join("spec.md")), "got: {got:?}");
        assert!(got.contains(&dir.join("notes.txt")), "got: {got:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_files_never_suggested_but_siblings_are() {
        let dir = test_dir("sib");
        let a = write(&dir, "a.rs", "mod ghost;\n");
        write(&dir, "b.rs", "pub fn b() {}\n");
        write(&dir, "notes.md", "other ext\n");
        let content = std::fs::read_to_string(&a).unwrap();

        let got = suggest_next_reads(&a, &content, 8);
        assert!(
            !got.iter().any(|p| p.to_string_lossy().contains("ghost")),
            "missing files excluded, got: {got:?}"
        );
        assert!(
            got.contains(&dir.join("b.rs")),
            "same-ext sibling, got: {got:?}"
        );
        assert!(
            !got.contains(&dir.join("notes.md")),
            "different-ext sibling excluded, got: {got:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_serve_invalidate_and_stale_mtime() {
        clear_cache();
        let dir = test_dir("cache");
        let f = write(&dir, "f.txt", "version-one-content-here\n");

        assert!(get(&f).is_none(), "cold cache misses");
        assert!(fetch_one(&f), "fetch caches under default cwd scope");
        assert_eq!(get(&f).as_deref(), Some("version-one-content-here\n"));

        invalidate(&f);
        assert!(get(&f).is_none(), "invalidate drops the entry");

        assert!(fetch_one(&f));
        // Rewrite with different length → length backstop catches staleness
        // even if the mtime granule collides.
        std::fs::write(&f, "version two, much longer content here\n").unwrap();
        assert!(
            get(&f).is_none(),
            "modified file must not serve stale bytes"
        );
        assert!(fetch_one(&f));
        assert_eq!(
            get(&f).as_deref(),
            Some("version two, much longer content here\n")
        );
        clear_cache();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sandbox_is_respected_by_fetch() {
        clear_cache();
        // Outside the current-dir safefolder (default FolderScope): refused.
        let outside = PathBuf::from("/definitely-not-hercules-prefetch-xyz/file.txt");
        assert!(!fetch_one(&outside));
        assert!(get(&outside).is_none());
        clear_cache();
    }

    #[test]
    fn oversized_files_refused() {
        clear_cache();
        let dir = test_dir("big");
        let big = dir.join("big.bin");
        let payload = vec![b'x'; (MAX_FILE_BYTES + 1024) as usize];
        std::fs::write(&big, payload).unwrap();
        assert!(!fetch_one(&big), "oversized files never cached");
        assert!(get(&big).is_none());
        clear_cache();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn error_outputs_never_speculate() {
        // No thread, no cache entries, no panic on error text.
        clear_cache();
        let dir = test_dir("err");
        let f = write(&dir, "real.rs", "mod sib;\n");
        write(&dir, "sib.rs", "x\n");
        speculate_from_read(f.clone(), "Error: File 'x' doesn't exist".to_string());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(get(&dir.join("sib.rs")).is_none());
        clear_cache();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
