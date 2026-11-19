# Phase 0 — Spike results

Date: 2026-11-19. Machine: MacBook Air M4, 16 GB. Ollama 0.40.16 (native), OrbStack (Docker 29.8).

## Verdict

**Both providers work fully locally.** No cloud fallback needed. Phase 0 exit criterion met.

## Model choice

| Model | Use | Result |
|---|---|---|
| `qwen3:8b` | — | Works, but thinking mode is on by default and mem0 cannot disable it: ~10x slower (19 s vs 1.9 s on the same prompt; mem0 add ≈ 30 s). Rejected as default. |
| `qwen3:4b-instruct-2507-q4_K_M` | base | Non-thinking. Default context 4096 is too small: mem0's extraction prompt is ~8k tokens → hard error. |
| **`scopeflow-q4`** = the above + `num_ctx 16384` | **default for nodes, mem0 and Letta** | 4.0 GB in RAM, 100% GPU. Tool calling 5/5. mem0 add 3–6 s. |
| `nomic-embed-text` | embeddings (mem0 + Letta) | 768 dims, ~0.3 GB. |

Modelfile: [`ollama/Modelfile.scopeflow-q4`](../ollama/Modelfile.scopeflow-q4).
Create with `ollama create scopeflow-q4 -f ollama/Modelfile.scopeflow-q4`.

## Spike A — mem0 (mem0ai 2.2.1 + Qdrant)

- Fact extraction: correct facts from natural sentences; "ADD" events returned with ids.
- With the 4B model facts are sometimes **merged** into one sentence (8B gave atomic facts). Acceptable for the demo; could be improved with a custom extraction prompt.
- Search latency: ~0.2 s.
- Scores (cosine, nomic-embed-text): relevant hits 0.55–0.76, irrelevant up to ~0.45.
  → **default `min_score = 0.5`**, `top_k = 5`.
- Noise in logs: spaCy / fastembed optional features missing (BM25 disabled). Not needed.
- mem0 sends PostHog telemetry by default → set `MEM0_TELEMETRY=False`.

## Spike B — Letta

### Server version
`letta/letta:latest` (0.31+) is now **Letta Code / App Server** — a different product (TS Agent SDK, coding agents).
The classic REST server is end-of-life. **Pinned `letta/letta:0.16.8`** (last classic release, May 2026).
Python client `letta-client 1.12.1` works with it.

### Required configuration (all three were real failures)
1. **Embedding endpoint:** Letta's auto-discovered Ollama embedding config points at `http://host.docker.internal:11434` and calls `/embeddings` → 404. Archival memory silently fails and the step retries until the client times out.
   Fix: pass `embedding_config` explicitly with endpoint `.../v1`.
2. **Context window:** Letta reports 262k context for the Ollama model; real `num_ctx` is 16k. Fix: `context_window_limit=12000` on agent creation.
3. **New Ollama models** are only visible to Letta after a container restart.

### Memory agent setup
- `include_base_tools=False`, `tools=["archival_memory_insert", "archival_memory_search"]`.
- Persona instructs the STORE / RETRIEVE protocol (see spike script).
- Without restricting tools, the agent writes facts into the core `human` block (in-context), not archival → no fact ids, can't list/delete individually.

### Reliability (8 stores + 8 retrieves)
- STORE: 7/8 correct. Failure: "Thanks, that was helpful!" was stored as a fact.
- RETRIEVE: 8/8 correct, including a negative query ("capital of France" → `none`).
- Stored passages are clean standalone sentences.

### Latency
| | prompt tokens | latency |
|---|---|---|
| without history reset | 3.1k → 4.6k, grows ~450/call | 3–8 s |
| **with `messages.reset` before each call** | constant ~3.3k | **5.6–6.2 s** |

Without reset the history would hit the 12k limit after ~20 calls and trigger Letta's summarization. → **Reset history before every call** (the memory agent needs no conversation history).

An earlier run showed 50–90 s per call: caused by a request abandoned on client timeout that **kept running server-side** and competed for the model.
→ Provider must avoid abandoned requests (generous timeout, cancel/cleanup on timeout).

## Fixes — verified

All Letta pitfalls are handled and checked by [`scripts/verify_letta.py`](../scripts/verify_letta.py)
(3 consecutive runs, all PASS). Rerun it after any change to the stack.

| # | Pitfall | Fix | Where |
|---|---|---|---|
| 1 | `latest` is a different product | pin `letta/letta:0.16.8`; documented in README | `docker-compose.yml`, `README.md` |
| 2 | embeddings sent to `/embeddings` → 404, archival silently broken | explicit `embedding_config` with `.../v1` endpoint (a server-level fix does not work: Letta strips `/v1` from `OLLAMA_BASE_URL`) | agent config |
| 3 | context reported as 262k | `context_window_limit=12000` | agent config |
| 4 | history grows every call | `message_buffer_autoclear=True` (server-side, no extra reset call). In-context messages stay at 5. RETRIEVE prompt size still varies with the number of search hits — bounded by search page size, not by call count | agent config |
| 5 | abandoned requests keep running | **root cause found:** the client auto-retries on timeout (default 2 retries) → one request becomes 3 server runs → duplicated facts + model contention. Fix: `max_retries=0`, `max_steps=4` per call, `messages.cancel()` on timeout; Redis added (required by Letta for cancel) | client config, `docker-compose.yml` |

Note: cancel does not roll back a tool call that already ran — a fact inserted before the cancel stays (once). Acceptable: it is visible and deletable on the Scopes page.

The reference agent/client config lives in `memory_agent_kwargs()` in the verify script and moves into `LettaProvider` in Phase 4.

## Consequences for the architecture

- Letta provider: per-scope agent with explicit `embedding_config`, `context_window_limit`, archival-only tools, history reset per call, per-scope lock (also prevents model contention).
- Letta facts = archival passages (have ids → list/add/delete work).
- Default model `scopeflow-q4` for everything; one chat model loaded at a time.
- Per exchange with a Letta scope: node call + 2 Letta calls ≈ 12 s of memory overhead. With mem0: ≈ 3–6 s (store) + 0.2 s (search).
- Improve STORE prompt to reject small talk (or pre-filter trivially short inputs).

## Leftovers from the spike
Spike agents (Letta) and spike collections (Qdrant) deleted. Spike scripts were throwaway and are not in the repo; `scripts/verify_letta.py` replaces them.
