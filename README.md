# ScopeFlow

> Visual agent flows with shareable long-term memory scopes. Runs fully locally.

**Status:** early development (Phase 1 of [the plan](docs/PLAN.md)).

ScopeFlow is a local-first demo engine inspired by Flowise AgentFlow. You build flows of
Agent nodes in a web UI, chain them, and attach nodes to **memory scopes**: long-term
memory stores backed by [mem0](https://github.com/mem0ai/mem0) or
[Letta](https://github.com/letta-ai/letta), shareable between nodes and between flows.

- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Plan: [docs/PLAN.md](docs/PLAN.md)
- Phase 0 findings: [docs/SPIKES.md](docs/SPIKES.md)

## Local stack

| Component | Runs | Notes |
|---|---|---|
| Ollama | natively on the host | needs the Metal GPU; Docker on macOS has no GPU access |
| Postgres, Qdrant, Letta, Redis | `docker compose up -d` | memory-capped for a 16 GB machine |

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull nomic-embed-text
ollama create scopeflow-q4 -f ollama/Modelfile.scopeflow-q4
docker compose up -d
.venv/bin/python scripts/verify_letta.py   # smoke check of the Letta setup
```

## Known limitations

- **Letta is pinned to `letta/letta:0.16.8`, the last release of the classic Letta server.**
  The upstream project has since moved to a different product (Letta Code / App Server), and the
  classic server no longer receives updates. ScopeFlow keeps memory providers behind one interface,
  so mem0 is the primary provider and Letta is the second implementation.
- Built for a single local user; no auth.
- Flows are linear chains (no branching or parallel nodes yet).
- Local 4B model: memory extraction is good but not perfect (e.g. small talk is occasionally stored as a fact).
