# Hercules Agent

Local coding agent with a terminal UI. Runs models on your machine via:

- **llama.cpp** (`llama-server` HTTP) for practical GGUF inference
- **llama.rs** pure-Rust GGUF path (no C/FFI; still maturing)
- **Ollama** as an alternate backend

Working name / crate: `hercules-agent`. Binary: `hercules`.

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
