//! Deterministic read predictor + canonical-path shims.
//!
//! The [`crate::agent_io::AgentIoScheduler`] owns workers, the
//! generation-versioned cache, budgets, metrics and cancellation. This
//! module owns *prediction* (which files will the model read next?) and
//! keeps the original thin shims (`get`, `invalidate`, `fetch_one`,
//! `speculate_from_read`) so the canonical `execute_read` /
//! `execute_write` / `execute_proposed` call sites — and their permission
//! gates — are unchanged.
//!
//! Safety contract (unchanged): only file READS are ever predicted.
//! Every candidate passes the scheduler's `path_allowed()` gate at fetch
//! time, and the canonical `execute_read` re-checks it at serve time.

use std::path::{Path, PathBuf};

use crate::agent_io::{RetrievalCandidate, RetrievalConfig, RetrievalReason};

/// Re-exported for tests; authoritative value lives in `agent_io`.
pub(crate) const MAX_FILE_BYTES: u64 = crate::agent_io::MAX_FILE_BYTES;
/// Upper bound on same-directory sibling candidates.
const MAX_SIBLINGS: usize = 6;
/// Only the first N bytes of a read result are scanned for candidates.
const MAX_SCAN_BYTES: usize = 200 * 1024;

/// File extensions that may be treated as path mentions when quoted.
const PATHLIKE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "mjs", "cjs", "go", "java", "c", "h", "hpp", "cpp",
    "toml", "json", "yaml", "yml", "md", "html", "css", "sh", "txt",
];

/// Ambient run id for static call sites without run context
/// (`execute_proposed` is `&self`-free). Never cancelled by App hooks;
/// App-integrated paths use real run ids.
pub const AMBIENT_RUN_ID: u64 = 0;

/// Serve a cached read if fresh (generation + mtime + length + TTL),
/// through the authorization-aware wrapper (sandbox enforced inside).
/// `execute_read` additionally checks `path_allowed()` first for its
/// user-facing error strings; defense in depth, one gate implementation.
pub fn get(path: &Path) -> Option<String> {
    crate::agent_io::AgentIoScheduler::global()
        .serve_authorized(path)
        .map(|hit| hit.content)
}

/// Drop a path from the cache + bump its generation (canonical write path).
pub fn invalidate(path: &Path) {
    crate::agent_io::AgentIoScheduler::global().notify_write(path);
}

/// Fetch one file into the cache synchronously. Returns true when the
/// file is cached afterwards (already-fresh counts). Goes through the
/// ONE unified stable-read primitive (sandbox → stable identity →
/// snapshot validation), so warm-ups carry the same guarantees as
/// worker fetches. Used by tests and one-shot warm-ups; the background
/// path goes through the scheduler workers instead.
fn fetch_one(abs: &Path) -> bool {
    let sched = crate::agent_io::AgentIoScheduler::global();
    if sched.serve_authorized(abs).is_some() {
        return true;
    }
    sched.read_through(abs).is_ok()
}

/// Suggest likely-next reads given the file just read and its content.
/// Pure candidate extraction + existence checks; no global state, no
/// permission checks (those happen in the scheduler workers).
/// Order: explicit mentions first (strongest signal), siblings after.
/// Deterministic for identical directory state.
pub fn suggest_next_reads(read_path: &Path, content: &str, max: usize) -> Vec<PathBuf> {
    let config = RetrievalConfig::default();
    suggest_candidates(read_path, content, &config)
        .into_iter()
        .take(max)
        .map(|c| c.path)
        .collect()
}

/// Reason-tagged, confidence-scored version of [`suggest_next_reads`]
/// for the scheduler. Confidence comes from [`RetrievalConfig`];
/// candidates are capped per event and carry estimated byte sizes for
/// budget accounting.
pub fn suggest_candidates(
    read_path: &Path,
    content: &str,
    config: &RetrievalConfig,
) -> Vec<RetrievalCandidate> {
    let mut out: Vec<(PathBuf, RetrievalReason)> = Vec::new();
    let mut seen: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    let max = config.max_candidates_per_event;
    let mut try_push = |out: &mut Vec<(PathBuf, RetrievalReason)>,
                        seen: &mut std::collections::HashSet<PathBuf>,
                        p: PathBuf,
                        reason: RetrievalReason| {
        if out.len() >= max || !seen.insert(p.clone()) {
            return;
        }
        out.push((p, reason));
    };

    let Some(base_dir) = read_path.parent() else {
        return Vec::new();
    };
    let self_name = read_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string());
    let read_ext = read_path.extension().and_then(|e| e.to_str());

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
                        try_push(&mut out, &mut seen, cand, RetrievalReason::RustModule);
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
            for (cand, reason) in resolve_literal_reasoned(base_dir, read_ext, line, &lit) {
                if is_plain_file(&cand) {
                    try_push(&mut out, &mut seen, cand, reason);
                }
            }
        }
    }

    // Same-directory siblings with the same extension (bounded, sorted).
    if out.len() < max {
        let wanted_ext = read_ext;
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
                try_push(
                    &mut out,
                    &mut seen,
                    s,
                    RetrievalReason::SameDirectorySibling,
                );
            }
        }
    }

    out.into_iter()
        .map(|(path, reason)| {
            let estimated_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);
            let confidence = config.confidence_for(reason);
            RetrievalCandidate {
                path,
                confidence,
                reason,
                estimated_bytes,
            }
        })
        .collect()
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

pub(crate) fn quoted_literals(line: &str) -> Vec<String> {
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
pub(crate) fn resolve_literal(base_dir: &Path, lit: &str) -> Vec<PathBuf> {
    resolve_literal_reasoned(base_dir, None, "", lit)
        .into_iter()
        .map(|(p, _)| p)
        .collect()
}

fn is_js_family(ext: Option<&str>) -> bool {
    matches!(ext, Some("js" | "ts" | "tsx" | "jsx" | "mjs" | "cjs"))
}

/// Reason-tagged literal resolution: same candidates as
/// [`resolve_literal`], each labeled with *why* it was predicted.
fn resolve_literal_reasoned(
    base_dir: &Path,
    read_ext: Option<&str>,
    line: &str,
    lit: &str,
) -> Vec<(PathBuf, RetrievalReason)> {
    // Python `from .foo import` style: dots DIRECTLY attached to a module
    // name. `./x` / `../x` are filesystem relatives, not Python imports.
    let mut rel = lit.trim().to_string();
    let dots = rel.chars().take_while(|&c| c == '.').count();
    if dots > 0 && rel[dots..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_') {
        let mod_path = rel[dots..].replace('.', "/");
        return vec![
            (
                base_dir.join(format!("{mod_path}.py")),
                RetrievalReason::PythonImport,
            ),
            (
                base_dir.join(&mod_path).join("__init__.py"),
                RetrievalReason::PythonImport,
            ),
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
        let reason = if is_js_family(read_ext) {
            RetrievalReason::JsImport
        } else {
            RetrievalReason::QuotedPath
        };
        return vec![
            base.clone(),
            base.with_extension("ts"),
            base.with_extension("tsx"),
            base.with_extension("js"),
            base.with_extension("jsx"),
            base.join("index.ts"),
            base.join("index.js"),
        ]
        .into_iter()
        .map(|p| (p, reason))
        .collect();
    }
    if !looks_like_file {
        return Vec::new();
    }
    let reason = if line.contains("#include") || line.contains("include!") {
        RetrievalReason::IncludeDirective
    } else if read_ext == Some("md") {
        RetrievalReason::MarkdownLink
    } else if is_js_family(read_ext) {
        RetrievalReason::JsImport
    } else if read_ext == Some("py") {
        RetrievalReason::PythonImport
    } else {
        RetrievalReason::QuotedPath
    };
    vec![(base_dir.join(&rel), reason)]
}

/// Entry point from the canonical read path: after a successful read,
/// submit predictions to the scheduler workers while the LLM keeps
/// reasoning. Returns immediately. Never speculates from error outputs.
/// Static call sites without run context use [`AMBIENT_RUN_ID`].
pub fn speculate_from_read(read_path: PathBuf, output: String) {
    if output.trim_start().starts_with("Error:") {
        return;
    }
    crate::agent_io::AgentIoScheduler::global().notify_explicit_read(
        AMBIENT_RUN_ID,
        &read_path,
        &output,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

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
        crate::agent_io::AgentIoScheduler::global().reset_for_test();
    }

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        crate::agent_io::test_serial_guard()
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
        let _g = serial();
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
        let _g = serial();
        clear_cache();
        // Outside the current-dir safefolder (default FolderScope): refused.
        let outside = PathBuf::from("/definitely-not-hercules-prefetch-xyz/file.txt");
        assert!(!fetch_one(&outside));
        assert!(get(&outside).is_none());
        clear_cache();
    }

    #[test]
    fn oversized_files_refused() {
        let _g = serial();
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
        // No submission, no cache entries, no panic on error text.
        let _g = serial();
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
