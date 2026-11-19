# ScopeFlow — Implementation Plan

Companion to [ARCHITECTURE.md](ARCHITECTURE.md). Target: working demo in 6 weeks (hard limit ~8).
Each phase has tasks and an **exit criterion** (how we know it is done). Phases are ordered by risk: unknowns first.

---

## Phase 0 — Environment & spikes (days 1–3)

Status: **done** — see [SPIKES.md](SPIKES.md).

- [x] Homebrew: uv, Ollama, OrbStack installed.
- [x] Project venv `.venv` (Python 3.12) with mem0ai, letta-client, qdrant-client, ollama.
- [x] Ollama running; models `scopeflow-q4` (default), `nomic-embed-text`; `qwen3:8b` rejected (thinking mode, too slow).
- [x] Tool-calling check: 5/5.
- [x] OrbStack + `docker` verified.
- [x] `docker-compose.yml`: Postgres, Qdrant, Letta 0.16.8 (pinned), memory-capped.
- [x] **Spike A — mem0:** extraction + search work; add 3–6 s, search 0.2 s; default `min_score=0.5`.
- [x] **Spike B — Letta:** store 7/8, retrieve 8/8; 5–6 s per call with history reset; three config pitfalls documented.
- [x] Results in `docs/SPIKES.md`.
- [ ] Not yet tested: deleting a scope's data in mem0 (filter delete) and Letta (agent delete) — cover in Phase 3/4 contract tests.

**Exit:** mem0 and Letta both retrieve/store/delete on local models, or a documented fallback for Letta (e.g. free cloud model for Letta only) is chosen.

**Decision gate:** if Letta is unreliable locally, decide here, before any code depends on it.

---

## Phase 1 — Backend skeleton and engine (weeks 1–2)

1. Repo scaffolding: `backend/` (uv project, FastAPI, pytest, mypy, ruff), `frontend/` later, GitHub repo, CI (pytest + mypy).
2. Postgres + SQLAlchemy + Alembic migrations: `flows`, `sessions`, `messages`, `runs`, `node_runs`.
3. `LLMClient` interface + `OllamaClient` (chat, tool calls, embeddings, token counts from Ollama response).
4. Tools: calculator, current time, DuckDuckGo search. Tool registry + JSON schemas.
5. Agent node: prompt assembly, tool loop with iteration cap, error handling.
6. InChat window: last N (input, output) pairs per node per session.
7. Chain runner: validate chain (Start → Agent* → End, 1 in / 1 out), resolve `{{ nodes.<id>.output }}`, emit events.
8. API: flows CRUD, sessions, send message, SSE events, run trace.
9. Tests: engine with a fake `LLMClient` (deterministic), graph validation, variable resolution.

**Exit:** `curl` creates a 2-node flow, sends a message, SSE streams node statuses, trace shows both nodes' inputs and outputs. Tests green in CI.

---

## Phase 2 — Frontend (weeks 2–3)

1. Vite + React + TypeScript + React Flow; generated API client from OpenAPI.
2. Flows list; editor with Start/Agent/End nodes; edge validation (one in / one out).
3. Node settings panel (prompt, model, tools, window size).
4. Chat panel on the Start node; "New chat"; live node status via SSE.
5. Trace panel: click node → input/output; expand → timings, tokens (memory sections empty for now).
6. Notification system (toasts: error / warning / info) wired to API errors.
7. ESLint, `tsc`, CI step.

**Exit:** build a 2–3 node flow in the browser, chat with it, inspect the trace.

---

## Phase 3 — Memory core with mem0 (weeks 3–4)

1. Migrations: `scopes`, `node_scope_links`.
2. `MemoryProvider` interface + `Fact` model; `ScopeManager` (create/delete scope, link rules, default link mode by owner flow, per-scope `asyncio.Lock`).
3. `Mem0Provider`: retrieve (`top_k`, `min_score`), store (extraction), list/add/delete fact, delete scope (provider first, then DB).
4. Engine integration: retrieve before LLM call (inject "Relevant memories"), store after (only if `read_write`). Record `retrieved`/`stored`/timings in `node_runs`.
5. API: scopes CRUD, facts CRUD, scope-link endpoints.
6. UI: memory section in node settings (none / existing scope / new scope → provider → params), link mode toggle (arrow), Scopes page (list facts, add fact, delete fact, delete scope with confirmation), trace panel shows retrieved/stored facts ranked.
7. Warnings: scope unused, read-only link, provider unreachable.
8. Tests: provider contract tests (shared suite run against each provider), ScopeManager rules, lock behaviour.

**Exit (the demo moment):** Node A in flow 1 learns a fact; after "New chat", node B in flow 2 attached to the same scope recalls it; the fact is visible and editable on the Scopes page.

---

## Phase 4 — Letta provider (weeks 4–5)

1. `LettaProvider` implementing the same interface: one memory agent per scope; retrieve = call 1, store = call 2; `score=None`.
2. `list_facts` / `add_fact` / `delete_fact` via archival passages; `delete_scope` deletes the agent.
3. UI: provider-specific params (hide `top_k`/`min_score` for Letta), handle empty score.
4. Run the **same contract test suite** against Letta.
5. Hardening: timeouts, retries, clear errors when Letta or model is down.

**Exit:** the Phase 3 demo scenario works unchanged with a Letta scope.

---

## Phase 5 — Polish and publish (weeks 5–6)

1. Seeded demo: two flows sharing one scope, preloaded facts, "reset demo" button.
2. Error-path pass: Ollama down, model missing, Qdrant/Letta down, empty scope, long input.
3. Docs: README (what/why, architecture diagram, quick start, demo GIF, known limitations, what's next), keep ARCHITECTURE.md in sync.
4. One-command start: `docker compose up` for infra + documented dev commands (Makefile or just-file).
5. Coverage and CI badges; license; project name final.
6. Repo hygiene: clean history, issues for "future work".

**Exit:** a stranger can clone, follow the README, and reproduce the demo in ≤15 minutes on a 16 GB Mac.

---

## Post-MVP backlog

- Token accounting for mem0 internal LLM calls.
- Letta single-call mode (node = Letta agent, shared memory blocks).
- Optional LangGraph executor as an experiment.
- Branching / parallel branches; shared Flow State.
- Token streaming.
- Export/import of flows.

---

## Working rules

- One phase at a time; each ends with a runnable state and a commit.
- Fake `LLMClient` in tests so CI never needs Ollama.
- Provider contract tests shared by mem0 and Letta.
- Decisions that change the architecture get a line in ARCHITECTURE.md (section 10 or a new ADR).
- Scope data deletion always goes through `provider.delete_scope()`.
