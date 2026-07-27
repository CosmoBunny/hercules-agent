# Hercules Agent — TODO

## Tools

- [ ] **Web search** — agent tool to query the web and inject results into context
- [ ] **Skill search** — discover and load project/user skills (SKILL.md style) on demand
- [ ] **MCP** — Model Context Protocol client: connect to MCP servers and expose their tools

## Features

- [ ] **Agent swarm mode** — spawn multiple agents that collaborate on a task using a
      distinct orchestration method (unique routing, shared memory, and role split;
      not a simple parallel fan-out of the same prompt)

## Settings

- [ ] **Benchmark for model recommendation** — run a short local bench (load latency,
      tokens/s, RAM, optional quality smoke) and recommend models/ctx/power for the
      current machine from Runtime menu

## Engines

- [ ] **Improve llama.rs** — reliability, speed, quantization coverage, parity with
      llama-server for day-to-day coding sessions on pure Rust
