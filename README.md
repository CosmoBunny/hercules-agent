# Hercules Agent

Local coding agent with a terminal UI. Runs models on your machine via:

- **llama.cpp** (`llama-server` HTTP) for practical GGUF inference
- **llama.rs** pure-Rust GGUF path (no C/FFI; still maturing)
- **Ollama** as an alternate backend

Working name / crate: `hercules-agent`. Binary: `hercules`.

## Getting started

Download the latest release here:
[CosmoBunny/hercules-agent — Releases](https://github.com/CosmoBunny/hercules-agent/releases/latest)

### 1. Pick the binary for your machine

| OS      | Integrated graphics | NVIDIA GPU | AMD Radeon GPU | ARM (aarch64) |
| ------- | ------------------- | ---------- | -------------- | ------------- |
| Linux   | `hercules-normal-linux-x86_64.zip` | `hercules-nvidia-linux-x86_64.zip` | `hercules-amd-linux-x86_64.zip` | `hercules-normal-linux-aarch64.zip` |
| Windows | `hercules-normal-windows-x86_64.zip` | `hercules-nvidia-windows-x86_64.zip` | `hercules-amd-windows-x86_64.zip` | — |
| macOS   | `hercules-normal-macos-aarch64.zip` (Metal GPU acceleration) | — | — | ✅ (same file) |

Unzip and run the `hercules` binary.

### 2. Download an LLM model

Press **F2** to open the Registry menu, type a model name to search,
navigate with the **↑/↓** arrow keys, and press **Enter** to download.

> **Caution:** Instruct models may ignore the system tool instructions —
> prefer a tool-capable chat model.

![Registry menu: search and download a model (F2)](https://github.com/user-attachments/assets/ac1925d1-13d6-4b16-9749-906a3967a842)

### 3. Select the model

After the download finishes, press **F3** to open the Model menu and
select the model you just downloaded.

![Model menu: select the downloaded model (F3)](https://github.com/user-attachments/assets/4312585c-28e0-4185-b008-21fdceabf367)

### 4. Configuration (optional)

![Settings: runtime configuration](https://github.com/user-attachments/assets/9af8f4fb-9365-42c5-8f12-4d37e07f61d5)

- **Power Mode** — how much power to spend on this AI model.
- **MTP** — multi-token prediction: extra prediction of the next tokens
  at the cost of RAM usage and precision.
- **Auto Collapse** — when enabled, every Agent response collapses the
  previous label.
- **Target FPS** — increase for smoother-feeling animation.
- **Stall Time** — watchdog against worst cases like the AI getting stuck
  on prefill.
- **Repeat Detector** — detects consecutive repeated text when the AI
  hallucinates/loops, and notifies the AI about the repetition.
- **Context Window** — lowers KV-cache demand. In the worst case the OS
  may kill the process for system safety, so decreasing this can prevent
  crashes.
- **Permission** — allow the AI to act and control directory access.
  Default is always allow.
- **Web Search** — which web provider the AI may use for online search.
  Default is DuckDuckGo.
- **HF Token** — if Hugging Face searches return empty (rate limiting),
  create a Hugging Face token and paste it here to avoid the error.

## Features (current)

- Ratatui TUI chat with tool chips (`write`, `cmd`, `ls`, `read`, `memory`)
- Runtime menu: context size, power mode, temperature, permissions
- Context compact to durable memory (`/compact`)
- Task manager for long-running shell commands
- Optional warm `llama-server` process (load GGUF once)

## Build

```bash
cargo build --release
./target/release/hercules
```

Debug:

```bash
cargo run
```

Requires a recent Rust toolchain (edition 2024).

### llama.cpp track

Install `llama-server` on `PATH` (or under `/opt/llama.cpp`). Prefer a build
matched to your CPU (AVX2-only machines must not use AVX-512 binaries).

Optional:

```bash
export HERCULES_N_GPU_LAYERS=0   # force CPU
export HERCULES_CTX=8192
```

### Ollama track

Run `ollama serve` and pick an Ollama model from the menu.

## Project layout

```
src/
  main.rs          # binary entry
  app.rs           # TUI
  agent.rs         # tools + system prompt
  backend.rs       # Ollama
  llama/           # llama.rs + llama-server client
  settings.rs      # runtime settings
  ...
```

## License

MIT. See [LICENSE](LICENSE).

## Roadmap

See [TODO.md](TODO.md).
