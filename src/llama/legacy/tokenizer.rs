//! Tokenizer loaded from GGUF metadata (same source llama.cpp uses).
//!
//! Supports:
//! - `tokenizer.ggml.model = gpt2`  → BPE with merges
//! - `tokenizer.ggml.model = llama` → SentencePiece-style longest-match over vocab
//!
//! Special tokens (BOS/EOS) come from `tokenizer.ggml.*_token_id`.

use crate::llama::gguf::{GgufFile, MetaValue};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_to_id: HashMap<String, u32>,
    pub merges: HashMap<(String, String), u32>,
    pub model: String,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub add_bos: bool,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let model = gguf
            .meta_str("tokenizer.ggml.model")
            .unwrap_or("unknown")
            .to_string();

        let tokens = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| "GGUF missing tokenizer.ggml.tokens".to_string())?;

        let scores = gguf
            .metadata
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_array())
            .unwrap_or_else(|| vec![0.0; tokens.len()]);

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }

        let mut merges = HashMap::new();
        if let Some(merge_list) = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array())
        {
            for (rank, m) in merge_list.iter().enumerate() {
                let parts: Vec<&str> = m.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    merges.insert((parts[0].to_string(), parts[1].to_string()), rank as u32);
                }
            }
        }

        let bos_id = gguf.meta_u64("tokenizer.ggml.bos_token_id").map(|v| v as u32);
        let eos_id = gguf.meta_u64("tokenizer.ggml.eos_token_id").map(|v| v as u32);
        let unk_id = gguf
            .meta_u64("tokenizer.ggml.unknown_token_id")
            .map(|v| v as u32);

        // Many LLaMA GGUFs expect BOS on encode.
        let add_bos = model == "llama" || bos_id.is_some();

        Ok(Self {
            tokens,
            scores,
            token_to_id,
            merges,
            model,
            bos_id,
            eos_id,
            unk_id,
            add_bos,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = if !self.merges.is_empty() {
            self.encode_bpe(text)
        } else {
            self.encode_longest_match(text)
        };

        if add_special && self.add_bos {
            if let Some(bos) = self.bos_id {
                if ids.first().copied() != Some(bos) {
                    ids.insert(0, bos);
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(tok) = self.tokens.get(id as usize) {
                // SentencePiece uses '▁' (U+2581) for space.
                let piece = tok.replace('▁', " ");
                // GPT-2 style byte fallback tokens look like <0x0A>
                if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                    if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) {
                        out.push(b as char);
                        continue;
                    }
                }
                out.push_str(&piece);
            }
        }
        // Collapse leading space artifact from BOS-only decode
        out
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_id == Some(id)
    }

    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        // Simplified GPT-2 style: split on whitespace / punctuation then apply merges.
        let mut words: Vec<String> = Vec::new();
        let mut cur = String::new();
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
                // Keep a leading space marker for GPT-2-like models
                cur.push(ch);
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            words.push(cur);
        }

        let mut ids = Vec::new();
        for word in words {
            ids.extend(self.bpe_word(&word));
        }
        ids
    }

    fn bpe_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }
        // Start as individual characters (as strings).
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_i = None;
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }
            let Some(i) = best_i else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols[i] = merged;
            symbols.remove(i + 1);
        }

        symbols
            .into_iter()
            .map(|s| {
                self.token_to_id
                    .get(&s)
                    .copied()
                    .or(self.unk_id)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// SentencePiece-like greedy longest-match over the vocabulary.
    fn encode_longest_match(&self, text: &str) -> Vec<u32> {
        // Prefer space-as-▁ form used by LLaMA SP.
        let normalized = {
            let mut s = String::new();
            let mut first = true;
            for part in text.split_whitespace() {
                if !first {
                    s.push('▁');
                } else if text.starts_with(|c: char| c.is_whitespace()) {
                    s.push('▁');
                } else {
                    // Leading non-space text often still gets ▁ in LLaMA
                    s.push('▁');
                }
                first = false;
                s.push_str(part);
            }
            if text.ends_with(|c: char| c.is_whitespace()) && !s.ends_with('▁') {
                // trailing space ignored for simplicity
            }
            if text.is_empty() {
                String::new()
            } else if !text.contains(char::is_whitespace) && !text.is_empty() {
                format!("▁{}", text)
            } else {
                s
            }
        };

        let chars: Vec<char> = normalized.chars().collect();
        let mut ids = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let mut matched = false;
            // Cap match length for performance
            let max_len = (chars.len() - i).min(64);
            for len in (1..=max_len).rev() {
                let piece: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&piece) {
                    ids.push(id);
                    i += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // byte fallback
                let ch = chars[i];
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                for b in encoded.as_bytes() {
                    let byte_tok = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.token_to_id.get(&byte_tok) {
                        ids.push(id);
                    } else if let Some(unk) = self.unk_id {
                        ids.push(unk);
                    } else {
                        ids.push(0);
                    }
                }
                i += 1;
            }
        }
        ids
    }
}

/// Build a chat-style prompt from system + user text.
/// Uses `tokenizer.chat_template` metadata when present (very simplified),
/// otherwise a portable ChatML-like envelope many local models accept.
pub fn format_chat_prompt(gguf: Option<&GgufFile>, system: &str, user: &str) -> String {
    if let Some(g) = gguf {
        if let Some(template) = g.meta_str("tokenizer.chat_template") {
            // Extremely small subset of Jinja-like substitution used by many GGUFs.
            if template.contains("{{") {
                let mut out = template.to_string();
                // Not a full Jinja engine — fall through to ChatML if complex.
                if !template.contains("for message") && !template.contains("{%") {
                    out = out.replace("{{ system }}", system);
                    out = out.replace("{{ user }}", user);
                    if !out.contains("{{") {
                        return out;
                    }
                }
            }
        }
        // Architecture-specific common templates
        match g.architecture().unwrap_or("") {
            "llama" => {
                return format!(
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    system, user
                );
            }
            "qwen2" | "qwen2moe" | "qwen3" => {
                return format!(
                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    system, user
                );
            }
            _ => {}
        }
    }

    // Portable ChatML
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        system, user
    )
}

#[allow(dead_code)]
fn _meta_unused(v: &MetaValue) {
    let _ = v;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_prompt_default() {
        let p = format_chat_prompt(None, "sys", "hi");
        assert!(p.contains("sys"));
        assert!(p.contains("hi"));
        assert!(p.contains("assistant"));
    }
}
