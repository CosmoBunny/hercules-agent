//! Silent clipboard for the TUI.
//!
//! `arboard` / `xclip` often print warnings to **stderr** (e.g. "data was dropped
//! very quickly after writing") which corrupt the alternate-screen UI. We never
//! let child tools write to the terminal.

use std::io::Write;
use std::process::{Command, Stdio};

const CLIP_FILE: &str = "/tmp/hercules_clipboard.txt";

/// Copy `text` without printing anything to the terminal.
/// Returns true if at least the file write or a clip tool succeeded.
pub fn copy_text_silent(text: &str) -> bool {
    let file_ok = std::fs::write(CLIP_FILE, text).is_ok();

    // Prefer Wayland / X tools with fully silenced stdio (never arboard —
    // it re-spawns xclip and leaks "dropped very quickly" onto the TUI).
    let tool_ok = try_pipe_tool("wl-copy", &[], text)
        || try_pipe_tool("xclip", &["-selection", "clipboard", "-in"], text)
        || try_pipe_tool("xsel", &["--clipboard", "--input"], text);

    file_ok || tool_ok
}

fn try_pipe_tool(bin: &str, args: &[&str], text: &str) -> bool {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Detach from TTY so no leakage into alternate screen
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(|| {
            // Don't steal the TUI's controlling terminal messages
            Ok(())
        });
    }
    let Ok(mut child) = cmd.spawn() else {
        return false;
    };
    let mut good = false;
    if let Some(mut stdin) = child.stdin.take() {
        good = stdin.write_all(text.as_bytes()).is_ok();
        let _ = stdin.flush();
        drop(stdin);
    }
    // Don't wait forever; xclip may daemonize holding the selection
    match child.try_wait() {
        Ok(Some(status)) => good && status.success(),
        Ok(None) => {
            // Still running (xclip often does) — treat as success
            true
        }
        Err(_) => good,
    }
}

pub fn read_clipboard_silent() -> Option<String> {
    // Try system clip tools first (wl-paste, xclip, xsel)
    if let Some(txt) = try_read_cmd("wl-paste", &["--no-newline"]) {
        if !txt.is_empty() {
            return Some(txt);
        }
    }
    if let Some(txt) = try_read_cmd("xclip", &["-selection", "clipboard", "-out"]) {
        if !txt.is_empty() {
            return Some(txt);
        }
    }
    if let Some(txt) = try_read_cmd("xsel", &["--clipboard", "--output"]) {
        if !txt.is_empty() {
            return Some(txt);
        }
    }
    // Fallback to local clipboard file
    std::fs::read_to_string(CLIP_FILE).ok().filter(|s| !s.is_empty())
}

fn try_read_cmd(bin: &str, args: &[&str]) -> Option<String> {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let output = cmd.output().ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout).ok()
    } else {
        None
    }
}

pub fn clipboard_file_path() -> &'static str {
    CLIP_FILE
}
