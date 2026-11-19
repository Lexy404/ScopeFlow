# ScopeFlow — Architecture & Decisions

> Working name: **ScopeFlow** (placeholder, check GitHub for collisions).
> Tagline: *Visual agent flows with shareable long-term memory scopes.*

A local-first demo engine inspired by Flowise AgentFlow. Users build flows of
Agent nodes in a web UI, chain them, and attach nodes to **memory scopes**
backed by either **mem0** or **Letta**. Scopes are first-class entities that
can be shared between nodes and between flows.

Clean-room implementation: written from scratch using public docs of Flowise
(Apache-2.0), mem0 and Letta. No code from any employer project.

---

## 1. Goals / Non-goals

**Goals**
- Create unlimited flows, each a linear chain of Agent nodes.
- Run a flow from a chat, stream per-node status, inspect a trace per node.
- Long-term memory via scopes, two interchangeable providers (mem0, Letta).
- Everything runs locally and free (Ollama), on a 16 GB MacBook Air M4.
- A repo that looks good: tests, CI, README with diagram and demo GIF.

**Non-goals (v1)**
- Branching, parallel branches, loops, conditions.
- Auth / multi-user. A single constant user.
- Flow versioning, deployment, export/import.
- Token accounting for mem0's internal LLM calls (post-MVP).

---

## 2. Layers

| Layer | Responsibility | Tech |
|---|---|---|
| **LLM** (the "ML" layer) | Provider abstraction for chat + embeddings | Ollama behind a thin `LLMClient` interface |
| **Agent / Engine** | Node execution, tool loop, chain runner, events | Python, asyncio, own executor |
| **InChatMemory** | Per-node, per-session message window | Postgres (keyed by session + node) |
| **LongTermMemory** | `ScopeManager` + `MemoryProvider` (mem0, Letta) | mem0, Qdrant, Letta |
| **API** | REST + SSE | FastAPI |
| **UI** | Graph editor, chat, trace, scope manager, notifications | React, TypeScript, React Flow, Vite |

```
UI (React) ──REST/SSE──> API (FastAPI)
                           │
                  ┌────────┴─────────┐
               Engine            ScopeManager
          (chain runner,        (rules, locks,
           Agent node,           links, lifecycle)
           tools, events)             │
              │              ┌────────┴────────┐
          LLMClient     Mem0Provider     LettaProvider
              │              │                 │
           Ollama     mem0 + Qdrant        Letta server
                      (host, native)       (Docker)
           Postgres: flows, scopes, links, sessions, runs, traces
```

Runtime placement (16 GB constraint):
- **Native on host:** Ollama (needs Metal GPU, which Docker on macOS lacks), backend, frontend dev server.
- **Docker Compose:** Postgres, Qdrant, Letta. Cap Docker Desktop memory (~4 GB).

---

## 3. Core concepts

### 3.1 Flow
A linear chain: `Start → Agent → Agent → … → End`.
- Node types: `Start`, `Agent`, `End`.
- Each node has at most **1 incoming and 1 outgoing** edge. The UI blocks a second edge; the API validates too.
- Stored as a React-Flow-style JSON graph in Postgres (`flows.graph`).
- One chat per flow, entering at the Start node; End node output is shown in chat.

### 3.2 Data passing between nodes
- **(a)** Output of node N becomes the input of node N+1 (no history).
- **(b)** Any node may reference the output of any earlier node via variables, e.g. `{{ nodes.<node_id>.output }}`.
- Shared Flow State is out of scope for v1.

### 3.3 Agent node
Config: name, system prompt, model, tools, InChat window size, optional scope link.
Execution: build prompt → (optional memory retrieve) → LLM + tool loop → (optional memory store).
Tools (v1): calculator, current time, DuckDuckGo web search.

### 3.4 InChatMemory
- Per node, per session: sliding **window of the last N** (input, output) messages.
- A session ends when the user presses **"New chat"**. History is kept in Postgres for trace viewing, but a new session starts with an empty window.

### 3.5 Scope (long-term memory)
A scope is an **independent entity**, not owned by a node.

```
Scope { id, name, provider: "mem0" | "letta", config, provider_ref, owner_flow_id? }
```

Rules:
1. Provider is chosen **when the scope is created** and never changes. Nodes attaching to an existing scope inherit its provider.
2. A node links to **at most one** scope; it retrieves from and stores to that same scope.
3. A link has a mode: `read_write` or `read`.
   - Default: `read_write` if the node's flow is the scope's `owner_flow_id`, else `read`.
   - The user can override the mode on the link (UI: arrow direction on the link).
4. A node may attach to scopes created in other flows.
5. Scopes live across all sessions and survive deletion of their owner flow.
6. **Deleting a scope is explicit only** (button with confirmation). Detaching the last node never deletes data; the UI shows a warning "scope is unused".
7. Concurrent writes to one scope are serialized with a per-scope `asyncio.Lock`.

Scope page in the UI: pick a scope → list facts (most relevant first when searching, otherwise by recency), **add a fact manually**, **delete a fact**, **delete the scope**.

### 3.6 Memory flow for a node run
```
input
  └─ if linked to a scope:
       facts = provider.retrieve(scope, query=input, params)     # always
       inject facts into the system prompt as "Relevant memories"
  └─ LLM + tools → output
  └─ if link.mode == read_write:
       provider.store(scope, text=<input + output>)               # provider extracts facts
```
Only **extracted facts** are stored, never raw transcripts (both providers).

Retrieval params live **on the node link** (different nodes may use different values):
- mem0: `top_k`, `min_score`. Defaults must be calibrated for the chosen embedding model.
- Letta: none; the Letta agent decides what is relevant.

### 3.7 MemoryProvider interface
```
retrieve(scope, query, params) -> list[Fact]
store(scope, text)             -> list[Fact]      # extracted + saved facts
list_facts(scope)              -> list[Fact]
add_fact(scope, text)          -> Fact            # verbatim, no extraction
delete_fact(scope, fact_id)    -> None
create_scope(config)           -> provider_ref
delete_scope(scope)            -> None            # cleans provider storage

Fact { id, text, score: float | None, metadata }
```
`score` is `None` for Letta; the UI must handle that.

| | mem0 | Letta |
|---|---|---|
| Storage | Qdrant, one collection, scope separated by `scope_id` (used as mem0 `user_id`) | Letta server DB, **one memory agent per scope** |
| retrieve | vector search (`top_k`, `min_score`) | call 1 to the Letta agent: "what is relevant to this input?" |
| store | mem0 LLM fact extraction (`infer=True`) | call 2 to the Letta agent: "remember what matters" |
| add_fact | add with `infer=False` | insert archival passage |
| delete_scope | delete by filter in Qdrant | delete the agent |

Letta is used in the **two-call mode** (node model stays independent; the Letta agent is a pure memory manager). A "node = Letta agent" single-call mode is a possible later extension and would need per-node agents plus shared memory blocks.

Validated in Phase 0 ([SPIKES.md](SPIKES.md)):
- Letta server pinned to `letta/letta:0.16.8` (last classic release; `latest` is a different product).
- Memory agent: archival-only tools (`archival_memory_insert`, `archival_memory_search`), explicit `embedding_config` (endpoint `.../v1`), `context_window_limit=12000`, `message_buffer_autoclear=True`.
- Letta client: `max_retries=0` (writes are not idempotent), `max_steps=4`, `messages.cancel()` on timeout. Redis runs alongside Letta (needed for cancel).
- mem0 defaults: `top_k=5`, `min_score=0.5` (nomic-embed-text), telemetry off.
- Default model for nodes, mem0 and Letta: `scopeflow-q4` (Qwen3 4B instruct, 16k context, see `ollama/`).

---

## 4. Data model (Postgres)

```
flows(id, name, graph jsonb, created_at, updated_at)
scopes(id, name, provider, config jsonb, provider_ref jsonb, owner_flow_id null, created_at)
node_scope_links(id, flow_id, node_id, scope_id, mode, retrieval jsonb)   -- unique(flow_id, node_id)
sessions(id, flow_id, created_at)
messages(id, session_id, node_id, role, content, created_at)             -- InChat window source
runs(id, session_id, status, started_at, finished_at, error)
node_runs(id, run_id, node_id, input, output,
          retrieved jsonb, stored jsonb,
          retrieve_ms, store_ms, llm_ms, tokens jsonb, error)
```
Node ids live inside `flows.graph`; `node_scope_links.node_id` is a plain string (no FK). When a node is removed from a graph, its link is removed.

---

## 5. API (sketch)

```
GET/POST/PUT/DELETE  /flows[/{id}]
POST   /flows/{id}/sessions                # "New chat"
POST   /sessions/{id}/messages             # starts a run
GET    /runs/{id}/events                   # SSE: node status, tokens, errors
GET    /runs/{id}                          # full trace (node_runs)

GET/POST/DELETE      /scopes[/{id}]
GET/POST             /scopes/{id}/facts
DELETE               /scopes/{id}/facts/{fact_id}
PUT/DELETE           /flows/{id}/nodes/{node_id}/scope-link
```

SSE events: `run.started`, `node.started`, `node.token` (optional), `node.memory` (retrieved/stored), `node.finished`, `run.finished`, `error`.

---

## 6. UI

- **Flows list** and **editor** (React Flow). Edge validation: one in / one out.
- **Node settings panel**: prompt, model, tools, window size, memory (none / existing scope / new scope → provider → provider params → retrieval params).
- **Chat panel** tied to the Start node, "New chat" button.
- **Trace**: click a node → input/output; expand → retrieved facts (ranked), stored facts, retrieve/store timing, tokens.
- **Scopes page**: facts list, add/delete fact, delete scope.
- **Notifications (toasts)** for errors and warnings, e.g. unused scope, read-only link, provider unreachable, Ollama model missing.

---

## 7. Repository layout (proposed)

```
scopeflow/
  backend/
    app/{api,engine,llm,memory,db,schemas}/
    tests/
  frontend/
    src/{editor,chat,trace,scopes,notifications,api}/
  docker-compose.yml        # postgres, qdrant, letta
  docs/ARCHITECTURE.md
  README.md                 # diagram, demo GIF, quick start
  .github/workflows/ci.yml  # pytest, mypy, eslint, tsc
```

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Letta needs reliable function calling; small local models often fail | Day 1–3 spike on the real machine; fallback: free cloud model for Letta only |
| 16 GB RAM with Ollama + Letta + Qdrant + Postgres | One small chat model (≈4–8B, quantized) shared by node, mem0 extraction and Letta; Ollama keeps one model loaded; Docker memory capped; embeddings via a small model. Exact model names to be verified in the spike |
| mem0 retrieval threshold depends on embedding model | Calibrate default `min_score` during spike; expose in node config |
| Scope data spread over 3 stores (our DB, Qdrant, Letta) | Deletion goes through `provider.delete_scope()`, never raw SQL; consistent order: provider first, then our DB |
| mem0's internal LLM token usage is hard to measure | Post-MVP: wrap the LLM client used by mem0 |
| Docker unfamiliarity | Only 3 containers; learn incrementally |

---

## 9. Milestones (≈6 weeks)

- **Days 1–3 — Spike:** Compose (Postgres, Qdrant, Letta) + native Ollama; mem0 with local models; **Letta tool-calling reliability check**.
- **Weeks 1–2 — Backend:** flows CRUD, chain engine with Ollama, InChat window, SSE.
- **Weeks 2–3 — UI:** React Flow editor, chat, trace panel, toasts.
- **Weeks 3–4 — Memory:** `MemoryProvider`, `ScopeManager`, mem0, links, scopes page.
- **Weeks 4–5 — Letta + rules:** Letta provider, per-scope lock, link modes, cross-flow scopes.
- **Weeks 5–6 — Polish:** tests, CI, seeded demo flow, README + diagram + GIF.

mem0 comes before Letta so nothing else depends on Letta working locally.

---

## 10. Open items (decide later)

- Final project name and license (MIT proposed).
- Token streaming vs. node-status-only streaming (status is mandatory, tokens optional).
- Token accounting for mem0 internals.
- "Node = Letta agent" single-call mode.
- Optional LangGraph-based executor as a post-MVP experiment.
