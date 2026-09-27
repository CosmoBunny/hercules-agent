//! AI-driven semantic context compaction.
//!
//! The model reads the conversation being retired and writes a compact
//! semantic representation; the old messages then leave the active model
//! context entirely. Deterministic extraction remains only as a labeled
//! fallback — never the primary algorithm.
//!
//! Pipeline (single implementation for manual + automatic):
//!
//! ```text
//! snapshot conversation
//!     → generate compact context (model)
//!     → validate structure + budget (retry once if oversize)
//!     → install compact context (+ recent tail)
//!     → only then discard old active context
//! ```
//!
//! Kept separate from persistent memory on purpose: memory holds
//! long-lived user/project facts, while the compact context holds the
//! compressed state of THIS conversation required to continue it. The
//! compact context never auto-becomes permanent memory.

/// Hard budget for the installed compact context (tokens ≈ chars/4).
/// A 100k-token conversation should compact to roughly this size.
pub const MAX_COMPACT_TOKENS: usize = 12_000;

/// Section headers the summary must use (stable, machine-readable).
pub const COMPACT_SECTIONS: &[&str] = &[
    "## Current Goal",
    "## Project / Repository",
    "## User Requirements",
    "## Current Architecture",
    "## Important Decisions",
    "## Implemented",
    "## Known Bugs / Unresolved",
    "## Files / Components",
    "## Tests / Verification",
    "## Constraints",
    "## Next Required Work",
    "## Important Technical Facts",
];

/// Why compaction runs. Manual and automatic share the whole pipeline;
/// only the trigger (and the recent-tail size) differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactReason {
    Manual,
    Automatic,
}

impl CompactReason {
    pub fn label(self) -> &'static str {
        match self {
            CompactReason::Manual => "manual /compact",
            CompactReason::Automatic => "auto",
        }
    }

    /// Recent conversational turns (You/Agent messages) kept verbatim
    /// after the compact block so immediate wording survives.
    pub fn keep_tail_messages(self) -> usize {
        match self {
            CompactReason::Manual => 2,
            CompactReason::Automatic => 4,
        }
    }
}

/// What can go wrong between snapshot and install. The caller must keep
/// the original conversation intact on every variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactError {
    /// Nothing worth compacting (tiny/empty archive).
    NothingToCompact,
    /// The model call itself failed (backend down, cancelled…).
    GenerationFailed(String),
    /// Produced text failed structural/budget validation, even after the
    /// concise retry.
    InvalidSummary(String),
}

/// Instruction prompt for the summarizer model. The transcript is the
/// complete conversation being retired; the model must describe CURRENT
/// state, not re-narrate history.
pub fn build_summarize_prompt(archive: &str, max_tokens: usize) -> String {
    format!(
        "You are compacting a long agent conversation so work can continue with a small context window.\n\
         Read the FULL transcript below. Write a compact semantic representation of the CURRENT state — \
         not a narration of the chat. Describe what IS true now; mention an old approach ONLY when needed \
         to explain why the current design exists. Never present superseded decisions as current.\n\
         \n\
         Preserve: current goal, project/task, decisions made, architecture, constraints, user requirements, \
         technical facts discovered, fixed bugs, UNRESOLVED bugs, files/modules involved, implementation state, \
         commands/APIs/contracts that matter, test results, regressions, pending TODOs, established terminology, \
         anything required to continue correctly.\n\
         Drop: greetings, filler, repeated explanations, obsolete attempts, redundant responses, stale reasoning.\n\
         Tool interactions: reduce to conclusions (files created/modified, test outcomes, fixes) — never paste \
         thousands of lines of tool output. Reference file paths + symbols + roles, not full file contents; the \
         agent re-reads files with tools when needed. A failed approach is worth ONE line only if it prevents \
         repeating the mistake (\"rejected because X; do not reintroduce\").\n\
         \n\
         Output EXACTLY this structure with these exact headers (omit a section only if truly empty), \
         at most ~{max_tokens} tokens total:\n\
         \n\
         # COMPACT CONTEXT\n\
         \n\
         ## Current Goal\n\
         ## Project / Repository\n\
         ## User Requirements\n\
         ## Current Architecture\n\
         ## Important Decisions\n\
         ## Implemented\n\
         ## Known Bugs / Unresolved\n\
         ## Files / Components\n\
         ## Tests / Verification\n\
         ## Constraints\n\
         ## Next Required Work\n\
         ## Important Technical Facts\n\
         \n\
         <TRANSCRIPT>\n\
         {archive}\n\
         </TRANSCRIPT>"
    )
}

/// Second-chance prompt when the first summary exceeded budget.
pub fn build_retry_prompt(oversize_summary: &str, max_tokens: usize) -> String {
    format!(
        "Compress the following conversation summary to at most ~{max_tokens} tokens. Keep the exact \
         `# COMPACT CONTEXT` structure and headers. Cut filler first, then examples, then older facts — \
         never cut the current goal, unresolved bugs, or next required work.\n\
         \n\
         <SUMMARY>\n\
         {oversize_summary}\n\
         </SUMMARY>"
    )
}

/// Structural + budget validation. Accepts only install-ready summaries.
pub fn validate(summary: &str, max_tokens: usize) -> Result<(), CompactError> {
    let text = summary.trim();
    if text.len() < 100 {
        return Err(CompactError::InvalidSummary(
            "summary too short to carry conversation state".into(),
        ));
    }
    if !text.contains("# COMPACT CONTEXT") {
        return Err(CompactError::InvalidSummary(
            "missing `# COMPACT CONTEXT` header".into(),
        ));
    }
    let sections = COMPACT_SECTIONS
        .iter()
        .filter(|h| text.contains(**h))
        .count();
    if sections < 4 {
        return Err(CompactError::InvalidSummary(format!(
            "only {sections} known sections present, need at least 4"
        )));
    }
    if estimate_tokens(text) > max_tokens {
        return Err(CompactError::InvalidSummary(format!(
            "summary exceeds budget ({} > {} tokens)",
            estimate_tokens(text),
            max_tokens
        )));
    }
    Ok(())
}

/// Token estimate, same convention as settings (chars/4).
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() + 3) / 4
}

/// Keep the newest `keep_n` conversational (You:/Agent:) messages, in
/// chronological order, for the recent tail.
pub fn select_recent_tail(messages: &[String], keep_n: usize) -> Vec<String> {
    messages
        .iter()
        .rev()
        .filter(|m| m.starts_with("You: ") || m.starts_with("Agent: "))
        .take(keep_n)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

/// Render the installed compact block as it appears in the model request.
pub fn compact_block(compact_text: &str) -> String {
    format!(
        "<compact_context>\n{}\n</compact_context>",
        compact_text.trim()
    )
}

/// Deterministic fallback summary, honestly labeled. Used only when the
/// model summary cannot be produced or validated. Wraps the existing
/// deterministic extraction so compaction still relieves context pressure
/// instead of failing destructive-or-nothing.
pub fn fallback_compact(deterministic_summary: &str) -> String {
    format!(
        "# COMPACT CONTEXT\n\
         \n\
         > NOTE: deterministic fallback — the model summary was unavailable, \
         so this block was extracted mechanically, not semantically written. \
         Treat facts as approximate.\n\
         \n\
         ## Current Goal\n\
         (unknown — see raw facts below)\n\
         \n\
         ## Important Technical Facts\n\
         {deterministic_summary}\n\
         \n\
         ## Constraints\n\
         (none recorded)\n\
         \n\
         ## Next Required Work\n\
         (unknown — re-read the recent tail and continue)\n"
    )
}

/// Assemble the model-bound context after compaction: compact block +
/// recent tail. Old messages are NOT included — that is the point.
pub fn assemble_prompt(compact_text: &str, recent_tail: &[String]) -> String {
    let mut parts = vec![compact_block(compact_text)];
    for m in recent_tail {
        parts.push(m.clone());
    }
    parts.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample_archive() -> String {
        "\
        You: build a flowchart app with svelte\n\
        Agent: I will create it. <write src=\"a.txt\">code</write>\n\
        You: use deno backend instead\n\
        Agent: switched to deno. tests pass.\n\
        System: [Task #1 DONE] `cargo test` (3 lines) — result delivered to agent\n\
        [tool]\n[write: a.txt] Wrote 5 lines.\n"
            .to_string()
    }

    pub(crate) fn sample_summary() -> String {
        "# COMPACT CONTEXT\n\
         \n\
         ## Current Goal\n\
         Build a flowchart app.\n\
         \n\
         ## Project / Repository\n\
         flowchart-app, svelte + deno.\n\
         \n\
         ## User Requirements\n\
         - Deno backend (changed from node).\n\
         \n\
         ## Current Architecture\n\
         - Svelte frontend, deno tasks.\n\
         \n\
         ## Important Decisions\n\
         - Deno over node for runtime.\n\
         \n\
         ## Implemented\n\
         - a.txt created.\n\
         \n\
         ## Known Bugs / Unresolved\n\
         - None known.\n\
         \n\
         ## Files / Components\n\
         - a.txt: app entry.\n\
         \n\
         ## Tests / Verification\n\
         - cargo test passes.\n\
         \n\
         ## Constraints\n\
         - None stated.\n\
         \n\
         ## Next Required Work\n\
         - Add magnetic snapping.\n\
         \n\
         ## Important Technical Facts\n\
         - Deno tasks via deno.json.\n"
            .to_string()
    }

    #[test]
    fn summarize_prompt_contains_schema_and_transcript() {
        let p = build_summarize_prompt("HELLO-ARCHIVE", 12000);
        assert!(p.contains("# COMPACT CONTEXT"));
        assert!(p.contains("## Next Required Work"));
        assert!(p.contains("HELLO-ARCHIVE"));
        assert!(p.contains("12000"));
        // Current-state instruction, not narration.
        assert!(p.contains("CURRENT state"));
        // Superseded decisions must not pose as current.
        assert!(p.contains("superseded"));
        // Tool/file economy rules present.
        assert!(p.contains("thousands of lines of tool output") || p.contains("tool output"));
        assert!(p.contains("re-reads files with tools") || p.contains("re-read"));
    }

    #[test]
    fn validate_accepts_wellformed_summary() {
        assert!(validate(&sample_summary(), MAX_COMPACT_TOKENS).is_ok());
    }

    #[test]
    fn validate_rejects_empty_headerless_and_oversize() {
        assert!(matches!(
            validate("", MAX_COMPACT_TOKENS),
            Err(CompactError::InvalidSummary(_))
        ));
        assert!(matches!(
            validate("just some prose without structure", MAX_COMPACT_TOKENS),
            Err(CompactError::InvalidSummary(_))
        ));
        // Budget enforced.
        let big = format!(
            "# COMPACT CONTEXT\n## Current Goal\n## Implemented\n## Next Required Work\n## Constraints\n{}",
            "x".repeat(4 * 100)
        );
        assert!(matches!(
            validate(&big, 10),
            Err(CompactError::InvalidSummary(_))
        ));
    }

    #[test]
    fn recent_tail_keeps_chronological_pairs() {
        let msgs = vec![
            "You: one".to_string(),
            "Agent: uno".to_string(),
            "System: [x] note".to_string(),
            "You: two".to_string(),
            "Agent: dos".to_string(),
        ];
        assert_eq!(
            select_recent_tail(&msgs, 2),
            vec!["You: two".to_string(), "Agent: dos".to_string()]
        );
        assert_eq!(select_recent_tail(&msgs, 4)[0], "You: one".to_string());
    }

    #[test]
    fn assemble_excludes_old_messages() {
        let old = "You: ancient history that must go away";
        let tail = vec!["You: fresh question".to_string()];
        let out = assemble_prompt(&sample_summary(), &tail);
        assert!(out.contains("# COMPACT CONTEXT"));
        assert!(out.contains("You: fresh question"));
        assert!(!out.contains(old));
    }

    #[test]
    fn assemble_smaller_than_original() {
        let old: String = (0..200)
            .map(|i| format!("You: filler line {i}\n"))
            .collect();
        let out = assemble_prompt(&sample_summary(), &[]);
        assert!(out.len() < old.len());
        assert!(estimate_tokens(&out) <= MAX_COMPACT_TOKENS);
    }

    #[test]
    fn fallback_is_labeled_and_schematic() {
        let f = fallback_compact("raw facts");
        assert!(f.contains("# COMPACT CONTEXT"));
        assert!(f.contains("deterministic fallback"));
        assert!(f.contains("raw facts"));
        assert!(
            validate(&f, MAX_COMPACT_TOKENS).is_ok(),
            "fallback installs"
        );
    }

    #[test]
    fn fallback_stays_bounded() {
        // Even a huge archive yields a bounded fallback (no blind dumps).
        let big_archive = "tool output line\n".repeat(50_000);
        let f = fallback_compact(&crate::app::compress_transcript(&big_archive));
        assert!(estimate_tokens(&f) <= MAX_COMPACT_TOKENS);
    }

    #[test]
    fn retry_prompt_targets_budget() {
        let p = build_retry_prompt("BIG", 4000);
        assert!(p.contains("4000"));
        assert!(p.contains("BIG"));
        assert!(p.contains("# COMPACT CONTEXT"));
    }

    #[test]
    fn reasons_share_pipeline_but_differ_in_tail() {
        assert_eq!(CompactReason::Manual.keep_tail_messages(), 2);
        assert_eq!(CompactReason::Automatic.keep_tail_messages(), 4);
        assert_ne!(CompactReason::Manual, CompactReason::Automatic);
    }
}
