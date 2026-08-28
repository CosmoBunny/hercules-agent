//! HTTP client for OpenAI-compatible and llama-server endpoints.
//! Used by llama.rs when no local GGUF is loaded, and by llama.cpp server mode.

use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct HttpInferenceClient {
    pub endpoint: String,
    pub model_name: String,
    client: Client,
}

impl HttpInferenceClient {
    pub fn new(endpoint: String, model_name: String) -> Self {
        Self {
            endpoint,
            model_name,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    fn completion_url(&self) -> String {
        let ep = self.endpoint.trim_end_matches('/');
        if ep.ends_with("/completion")
            || ep.ends_with("/completions")
            || ep.ends_with("/v1/chat/completions")
            || ep.ends_with("/v1/completions")
        {
            ep.to_string()
        } else {
            format!("{}/v1/chat/completions", ep)
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        self.generate_stream(prompt, target.clone(), flag).await
    }

    /// Build OpenAI-style messages: system + You:/Agent:/Tool history.
    ///
    /// llama-server rejects chats that **end with 2+ assistant messages** and
    /// often requires the last turn to be `user`. We normalize for that.
    pub fn chat_messages(system: &str, prompt_or_history: &str) -> Vec<serde_json::Value> {
        let mut turns: Vec<(String, String)> = Vec::new(); // (role, content)

        let push = |turns: &mut Vec<(String, String)>, role: &str, content: &str| {
            let c = content.trim();
            if c.is_empty() {
                return;
            }
            // Merge consecutive same-role turns (prevents 2× assistant at end)
            if let Some(last) = turns.last_mut() {
                if last.0 == role {
                    last.1.push_str("\n\n");
                    last.1.push_str(c);
                    return;
                }
            }
            turns.push((role.to_string(), c.to_string()));
        };

        // Walk lines; multi-line content continues the current role until a new prefix.
        let mut cur_role: Option<String> = None;
        let mut cur_buf = String::new();
        let flush =
            |turns: &mut Vec<(String, String)>, role: &mut Option<String>, buf: &mut String| {
                if let Some(r) = role.take() {
                    push(turns, &r, buf);
                    buf.clear();
                }
            };

        for line in prompt_or_history.lines() {
            let t = line.trim_end();
            if let Some(rest) = t.strip_prefix("You: ") {
                flush(&mut turns, &mut cur_role, &mut cur_buf);
                cur_role = Some("user".into());
                // Nudge small models to emit tools for list/read/run requests
                cur_buf = crate::agent::AgentEngine::with_tool_nudge(rest);
            } else if let Some(rest) = t.strip_prefix("Agent: ") {
                flush(&mut turns, &mut cur_role, &mut cur_buf);
                cur_role = Some("assistant".into());
                cur_buf = rest.to_string();
            } else if let Some(rest) = t.strip_prefix("Tool result:") {
                flush(&mut turns, &mut cur_role, &mut cur_buf);
                // Tool output must be a **user** turn for OpenAI/llama-server/ChatML
                cur_role = Some("user".into());
                cur_buf = format!("[Tool result]\n{}", rest.trim_start());
            } else if let Some(rest) = t.strip_prefix("Result:") {
                flush(&mut turns, &mut cur_role, &mut cur_buf);
                cur_role = Some("user".into());
                cur_buf = format!("[Action result]\n{}", rest.trim_start());
            } else if t.starts_with("<tool_result>") || t.starts_with("<tool_instruction>") {
                flush(&mut turns, &mut cur_role, &mut cur_buf);
                cur_role = Some("user".into());
                cur_buf = t.to_string();
            } else if t.starts_with("[Hercules]") || t.starts_with("[Instruction]") {
                flush(&mut turns, &mut cur_role, &mut cur_buf);
                cur_role = Some("user".into());
                cur_buf = t.to_string();
            } else if cur_role.is_some() {
                if !cur_buf.is_empty() {
                    cur_buf.push('\n');
                }
                cur_buf.push_str(t);
            } else if !t.is_empty() {
                // Bare text before any prefix → user
                cur_role = Some("user".into());
                cur_buf = t.to_string();
            }
        }
        flush(&mut turns, &mut cur_role, &mut cur_buf);

        if turns.is_empty() {
            let user = prompt_or_history.trim();
            if !user.is_empty() {
                turns.push((
                    "user".into(),
                    crate::agent::AgentEngine::with_tool_nudge(user),
                ));
            }
        }

        // Never end with assistant (llama-server: "Cannot have 2 or more assistant…")
        if turns.last().map(|(r, _)| r.as_str()) == Some("assistant") {
            turns.push((
                "user".into(),
                "Continue. If you already have tool results above, answer the user \
                 directly now — do not emit the same tool tag again."
                    .into(),
            ));
        }

        let mut messages = Vec::new();
        if !system.trim().is_empty() {
            // Prefer system role; if the GGUF template is broken, still better than
            // dumping a multi-KB instruction block that 1–3B models recite.
            messages.push(json!({"role": "system", "content": system}));
        }
        for (role, content) in turns {
            messages.push(json!({"role": role, "content": content}));
        }
        messages
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        let url = self.completion_url();
        // Compact system: long system text is recited by small GGUFs (see DeepSeek 1.3B).
        let system = crate::agent::AgentEngine::system_prompt_compact_for_cwd();

        let max_tokens = crate::settings::get_settings().power_mode.max_tokens();
        // Runtime menu temperature (default 0.2 — better tool following on small GGUFs)
        let temperature = crate::settings::temperature();
        // Stop before chat special tokens / role markers (DeepSeek emits <|im_end|>).
        let stop = json!([
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "</s>",
            "\nYou:",
            "\nUser:",
            "\n### Instruction",
            "</write>",
            "\nCRITICAL —",
            "\nCRITICAL -"
        ]);
        let body = if url.contains("/v1/chat/completions") {
            let messages = Self::chat_messages(&system, prompt);
            json!({
                "model": self.model_name,
                "messages": messages,
                "stream": true,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            })
        } else {
            json!({
                "prompt": format!("{}\n\n### User\n{}\n\n### Assistant\n", system, prompt),
                "stream": true,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": stop,
            })
        };

        let res = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| {
                format!(
                    "[HTTP Inference] Cannot connect to '{}': {}. \
                     Start llama-server / vLLM / LM Studio, or load a local GGUF with llama.rs.",
                    url, e
                )
            })?;

        if !res.status().is_success() {
            let status = res.status();
            let body_text = res.text().await.unwrap_or_default();
            return Err(format!(
                "[HTTP Inference] Server returned HTTP {}: {}",
                status, body_text
            ));
        }

        let mut byte_stream = res.bytes_stream();
        let mut full_text = String::new();
        let mut buffer = String::new();
        let mut token_count = 0usize;
        let gen_start_time = std::time::Instant::now();
        let mut first_token_time: Option<std::time::Instant> = None;

        while let Some(chunk_result) = byte_stream.next().await {
            if let Ok(active_gen) = is_generating.lock() {
                if !*active_gen {
                    return Err("[Generation Cancelled by User (CTRL+C)]".to_string());
                }
            }
            let chunk_bytes = chunk_result.map_err(|e| format!("[HTTP Stream Error] {}", e))?;
            let chunk_str = String::from_utf8_lossy(&chunk_bytes);
            buffer.push_str(&chunk_str);

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer.drain(..=newline_pos);

                if line.is_empty() {
                    continue;
                }

                let json_str = if line.starts_with("data:") {
                    let data = line.trim_start_matches("data:").trim();
                    if data == "[DONE]" {
                        return Ok(full_text);
                    }
                    data.to_string()
                } else if line.starts_with('{') {
                    line
                } else {
                    continue;
                };

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json_str) {
                    // llama-server / OpenAI error frames (often the real reason for "no tokens")
                    if let Some(err) = val.get("error") {
                        let msg = err
                            .get("message")
                            .and_then(|m| m.as_str())
                            .or_else(|| err.as_str())
                            .unwrap_or("unknown server error");
                        let code = err
                            .get("code")
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "?".into());
                        return Err(format!(
                            "[HTTP Inference] Server error (code {code}): {msg}. \
                             If this is a Compute error with OpenVINO/Vulkan, restart with \
                             pure CPU: HERCULES_N_GPU_LAYERS=0 (or Runtime Power Saver) and \
                             kill the old llama-server process."
                        ));
                    }

                    if val.get("stop").and_then(|s| s.as_bool()) == Some(true) {
                        return Ok(full_text);
                    }

                    let token = if let Some(c) = val.get("content").and_then(|c| c.as_str()) {
                        Some(c.to_string())
                    } else if let Some(choices) = val.get("choices").and_then(|c| c.as_array()) {
                        let ch0 = choices.first();
                        // Also surface per-choice errors
                        if let Some(msg) = ch0
                            .and_then(|ch| ch.get("error"))
                            .and_then(|e| e.get("message").or(Some(e)))
                            .and_then(|m| m.as_str())
                        {
                            return Err(format!("[HTTP Inference] choice error: {msg}"));
                        }
                        ch0.and_then(|ch| ch.get("delta"))
                            .and_then(|d| {
                                d.get("content")
                                    .and_then(|c| c.as_str())
                                    // Some builds put text in reasoning/tool fields only — ignore empty content
                                    .or_else(|| d.get("text").and_then(|t| t.as_str()))
                            })
                            .map(|s| s.to_string())
                            .or_else(|| {
                                ch0.and_then(|ch| ch.get("text"))
                                    .and_then(|t| t.as_str())
                                    .map(|s| s.to_string())
                            })
                            .or_else(|| {
                                // Non-stream style message in stream chunk
                                ch0.and_then(|ch| ch.get("message"))
                                    .and_then(|m| m.get("content"))
                                    .and_then(|c| c.as_str())
                                    .map(|s| s.to_string())
                            })
                    } else if let Some(tok) = val
                        .get("token")
                        .and_then(|t| t.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        Some(tok.to_string())
                    } else {
                        None
                    };

                    if let Some(token_str) = token {
                        if !token_str.is_empty() {
                            let cleaned =
                                crate::agent::AgentEngine::sanitize_model_output(&token_str);
                            if cleaned.is_empty() {
                                continue;
                            }
                            token_count += 1;
                            if first_token_time.is_none() {
                                let now = std::time::Instant::now();
                                first_token_time = Some(now);
                                let ttft = (now - gen_start_time).as_secs_f64();
                                crate::llama::libinfer::update_inference_telemetry(|t| {
                                    t.ttft_secs = ttft;
                                });
                            }
                            crate::llama::libinfer::update_inference_telemetry(|t| {
                                t.generated_tokens = token_count;
                                let elapsed = gen_start_time.elapsed().as_secs_f64();
                                if elapsed > 0.0 {
                                    t.decode_tok_per_sec = token_count as f64 / elapsed;
                                }
                            });
                            full_text.push_str(&cleaned);
                            if let Ok(mut target) = stream_target.lock() {
                                target.push_str(&cleaned);
                            }
                            // Abort early if model is reciting the system prompt.
                            if full_text.len() > 180
                                && crate::agent::AgentEngine::looks_like_system_echo(&full_text)
                            {
                                let msg = "[model echoed system prompt — stopped. \
                                     Try a stronger instruct model or shorter reply.]";
                                if let Ok(mut target) = stream_target.lock() {
                                    *target = msg.to_string();
                                }
                                return Err(msg.to_string());
                            }
                        }
                    }
                }
            }
        }

        let gen_dur = gen_start_time.elapsed().as_secs_f64().max(0.0001);
        let tok_speed = token_count as f64 / gen_dur;
        let prompt_estimate = (prompt.len() / 4).max(1);
        crate::llama::libinfer::update_inference_telemetry(|t| {
            t.prompt_tokens = prompt_estimate;
            t.generated_tokens = token_count;
            t.decode_duration_secs = gen_dur;
            t.decode_tok_per_sec = tok_speed;
            t.session_total_prompt_tokens += prompt_estimate;
            t.session_total_gen_tokens += token_count;
            t.session_total_inference_secs += gen_dur;
        });

        // Flush trailing buffer line without newline
        if !buffer.trim().is_empty() {
            let line = buffer.trim();
            let data = line.strip_prefix("data:").map(|s| s.trim()).unwrap_or(line);
            if data != "[DONE]" {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(err) = val.get("error") {
                        let msg = err
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("server error");
                        return Err(format!("[HTTP Inference] Server error: {msg}"));
                    }
                }
            }
        }

        let full_text = crate::agent::AgentEngine::sanitize_model_output(&full_text);
        if full_text.is_empty() {
            Err("[HTTP Inference] Stream completed with no tokens. \
                 Check llama-server log (/tmp/hercules/llama-server.last.log) — \
                 often Compute error from GPU/OpenVINO; use HERCULES_N_GPU_LAYERS=0."
                .to_string())
        } else if crate::agent::AgentEngine::looks_like_system_echo(&full_text) {
            Err(
                "[HTTP Inference] Model recited system instructions instead of answering. \
                 Use a stronger instruct GGUF, or say hello again after rebuild."
                    .to_string(),
            )
        } else {
            Ok(full_text)
        }
    }

    pub async fn health_check(&self) -> Result<String, String> {
        let url = format!("{}/health", self.endpoint.trim_end_matches('/'));
        match self.client.get(&url).send().await {
            Ok(res) if res.status().is_success() => {
                Ok(format!("Server at {} is healthy", self.endpoint))
            }
            Ok(res) => Err(format!(
                "Server at {} returned status {}",
                self.endpoint,
                res.status()
            )),
            Err(e) => Err(format!("Cannot reach {}: {}", self.endpoint, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_url_defaults_to_chat() {
        let c = HttpInferenceClient::new("http://localhost:8080".into(), "m".into());
        assert_eq!(
            c.completion_url(),
            "http://localhost:8080/v1/chat/completions"
        );
    }

    #[test]
    fn chat_messages_no_trailing_double_assistant() {
        let hist = "\
You: list the folder\n\n\
Agent: `<ls path=\"$CURRENT\">`\n\n\
Agent: `<ls path=\"$CURRENT\">`\n\n\
Tool result: dir contents here\n\n\
[Hercules] answer now";
        let msgs = HttpInferenceClient::chat_messages("sys", hist);
        // last role must be user
        let last = msgs.last().unwrap();
        assert_eq!(last["role"], "user");
        // no two consecutive assistants
        let roles: Vec<&str> = msgs.iter().filter_map(|m| m["role"].as_str()).collect();
        for w in roles.windows(2) {
            assert!(
                !(w[0] == "assistant" && w[1] == "assistant"),
                "consecutive assistants: {:?}",
                roles
            );
        }
    }

    #[test]
    fn chat_messages_merges_assistant_and_adds_user_tail() {
        let hist = "You: hi\n\nAgent: hello\n\nAgent: more";
        let msgs = HttpInferenceClient::chat_messages("sys", hist);
        let roles: Vec<&str> = msgs.iter().filter_map(|m| m["role"].as_str()).collect();
        assert_eq!(roles[0], "system");
        assert_eq!(*roles.last().unwrap(), "user");
        assert!(
            !roles
                .windows(2)
                .any(|w| w[0] == "assistant" && w[1] == "assistant")
        );
    }
}
