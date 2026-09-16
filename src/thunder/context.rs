//! Transport-safe context representation (minimal; T10 expands).
//!
//! Only explicitly selected snippets cross the machine boundary — never
//! whole files, never the repository. Sizes are bounded at encode time.

use super::error::ThunderError;

/// One explicitly selected snippet.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextSnippet {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub content: String,
}

/// What the receiver chose to send with one request.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextEnvelope {
    pub snippets: Vec<ContextSnippet>,
}

impl ContextEnvelope {
    /// Max bytes one snippet may carry (clamp at encode time).
    pub const MAX_SNIPPET_BYTES: usize = 64 * 1024;

    pub fn empty() -> Self {
        Self {
            snippets: Vec::new(),
        }
    }

    /// Builder for one explicitly selected snippet. Content is clamped to
    /// `MAX_SNIPPET_BYTES` — an oversized selection can never blow the
    /// envelope past host limits silently.
    pub fn with_snippet(
        mut self,
        path: &str,
        start_line: u32,
        end_line: u32,
        content: &str,
    ) -> Self {
        if path.trim().is_empty() || content.is_empty() {
            return self;
        }
        if self.snippets.len() >= 64 {
            return self;
        }
        let clamped: String = if content.len() > Self::MAX_SNIPPET_BYTES {
            let mut cut = Self::MAX_SNIPPET_BYTES;
            while !content.is_char_boundary(cut) {
                cut -= 1;
            }
            format!("{}\n… [truncated]", &content[..cut])
        } else {
            content.to_string()
        };
        self.snippets.push(ContextSnippet {
            path: path.to_string(),
            start_line,
            end_line,
            content: clamped,
        });
        self
    }

    /// Render the envelope as fenced context for the host's system prompt.
    /// Empty envelopes render as an empty string.
    pub fn render_as_prompt(&self) -> String {
        if self.snippets.is_empty() {
            return String::new();
        }
        let mut out = String::from("Context selected by the requester (use only if relevant):\n");
        for s in &self.snippets {
            out.push_str(&format!(
                "\n--- {}:{}-{} ---\n{}\n",
                s.path, s.start_line, s.end_line, s.content
            ));
        }
        out
    }

    pub fn total_bytes(&self) -> usize {
        self.snippets.iter().map(|s| s.content.len()).sum()
    }

    pub fn validate(&self, max_bytes: usize) -> Result<(), ThunderError> {
        if self.snippets.len() > 64 {
            return Err(ThunderError::ContextTooLarge);
        }
        if self.total_bytes() > max_bytes {
            return Err(ThunderError::ContextTooLarge);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snippet_builder_and_render() {
        let env = ContextEnvelope::empty()
            .with_snippet("src/main.rs", 1, 4, "fn main() {}\n")
            .with_snippet("", 1, 2, "ignored: empty path")
            .with_snippet("src/x.rs", 1, 1, "");
        assert_eq!(env.snippets.len(), 1);
        assert_eq!(env.snippets[0].path, "src/main.rs");
        let text = env.render_as_prompt();
        assert!(text.contains("src/main.rs:1-4"));
        assert!(text.contains("fn main()"));
        assert!(ContextEnvelope::empty().render_as_prompt().is_empty());
    }

    #[test]
    fn test_snippet_clamped_never_silent_overflow() {
        let big = "x".repeat(ContextEnvelope::MAX_SNIPPET_BYTES + 10);
        let env = ContextEnvelope::empty().with_snippet("big.rs", 1, 2, &big);
        assert_eq!(env.snippets.len(), 1);
        assert!(env.snippets[0].content.len() <= ContextEnvelope::MAX_SNIPPET_BYTES + 16);
        assert!(env.snippets[0].content.contains("[truncated]"));
    }

    #[test]
    fn test_snippet_count_capped() {
        let mut env = ContextEnvelope::empty();
        for i in 0..80 {
            env = env.with_snippet(&format!("f{i}.rs"), 1, 1, "x");
        }
        assert_eq!(env.snippets.len(), 64);
        assert!(env.validate(usize::MAX).is_ok());
    }

    #[test]
    fn test_validate_bounds() {
        let env = ContextEnvelope::empty().with_snippet("a.rs", 1, 1, "hello");
        assert!(env.validate(1024).is_ok());
        assert!(matches!(
            env.validate(2),
            Err(ThunderError::ContextTooLarge)
        ));
    }
}
