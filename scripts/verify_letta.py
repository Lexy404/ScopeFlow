"""Smoke check for the local Letta setup (see docs/SPIKES.md).

Verifies the five Letta pitfalls found in Phase 0 are handled:
  1. pinned classic server version
  2. explicit embedding config (archival memory actually stores passages)
  3. context window limited to the real Ollama num_ctx
  4. no history growth between calls (message_buffer_autoclear)
  5. abandoned requests are cancelled server-side and do not block the model.
     The client must NOT auto-retry (max_retries=0): by default it resends on
     timeout, every resend becomes a separate server run, and memory writes get
     duplicated.

Run:  .venv/bin/python scripts/verify_letta.py
Needs: docker compose up -d, Ollama with scopeflow-q4 + nomic-embed-text.

`memory_agent_kwargs()` is the reference config for the Letta provider (Phase 4).
"""

from __future__ import annotations

import sys
import time

import httpx
from letta_client import APITimeoutError, Letta

LETTA_URL = "http://localhost:8283"
EXPECTED_SERVER_VERSION = "0.16.8"
LLM_HANDLE = "ollama/scopeflow-q4:latest"
# Ollama as seen from inside the Letta container; /v1 is required (pitfall 2).
OLLAMA_FROM_LETTA = "http://host.docker.internal:11434/v1"
CONTEXT_WINDOW = 12_000  # below num_ctx 16384 of scopeflow-q4 (pitfall 3)
MAX_STEPS = 4  # bounds server-side work per call even if a client gives up

PERSONA = (
    "You are a memory manager for ONE shared knowledge store. You never chat. "
    "On a message starting with STORE: extract each durable fact about the user or the world "
    "as a short standalone sentence and save EACH fact with a separate "
    "archival_memory_insert call. "
    "Greetings, thanks and small talk are NOT facts: for them call nothing and reply 'none'. "
    "On a message starting with RETRIEVE: call archival_memory_search with a good query, "
    "then reply ONLY with the relevant facts, one per line; "
    "if nothing relevant, reply exactly 'none'. "
    "Never use core memory for facts."
)


def memory_agent_kwargs(name: str) -> dict:
    return {
        "name": name,
        "model": LLM_HANDLE,
        "embedding_config": {
            "embedding_endpoint_type": "openai",
            "embedding_endpoint": OLLAMA_FROM_LETTA,
            "embedding_model": "nomic-embed-text:latest",
            "embedding_dim": 768,
            "embedding_chunk_size": 300,
        },
        "context_window_limit": CONTEXT_WINDOW,
        "message_buffer_autoclear": True,  # pitfall 4
        "include_base_tools": False,
        "tools": ["archival_memory_insert", "archival_memory_search"],
        "memory_blocks": [{"label": "persona", "value": PERSONA}],
    }


results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def send(client: Letta, agent_id: str, text: str, timeout: float | None = None):
    kwargs = {"timeout": timeout} if timeout else {}
    t = time.time()
    r = client.agents.messages.create(
        agent_id=agent_id,
        messages=[{"role": "user", "content": text}],
        max_steps=MAX_STEPS,
        **kwargs,
    )
    return r, time.time() - t


def main() -> int:
    # max_retries=0: memory writes are not idempotent (pitfall 5)
    client = Letta(base_url=LETTA_URL, timeout=180, max_retries=0)

    # 1. version
    version = httpx.get(f"{LETTA_URL}/v1/health/").json().get("version")
    check("1. server version", version == EXPECTED_SERVER_VERSION, f"got {version}")

    agent = client.agents.create(**memory_agent_kwargs("verify-letta"))
    try:
        # 3. context window
        ctx = agent.llm_config.context_window
        check("3. context window limited", ctx == CONTEXT_WINDOW, f"got {ctx}")

        # 2. embeddings: a STORE must produce an archival passage
        send(client, agent.id, "STORE: User said: My name is Alex and I live in Jerusalem.")
        passages = client.agents.passages.list(agent_id=agent.id)
        check(
            "2. archival insert works (embeddings OK)",
            len(passages) >= 1,
            f"{len(passages)} passage(s)",
        )

        r, dt = send(client, agent.id, "RETRIEVE: Where does the user live?")
        answer = " ".join(
            str(m.content) for m in r.messages if m.message_type == "assistant_message"
        )
        check(
            "2b. archival search finds it",
            "jerusalem" in answer.lower(),
            f"{dt:.1f}s -> {answer[:60]!r}",
        )

        # 4. history does not grow: the number of in-context messages stays constant.
        # (RETRIEVE prompt tokens still vary with the size of search results; that is
        # bounded by the search page size, not by the number of calls.)
        in_context_counts = []
        for q in [
            "STORE: User said: My dog is called Rex.",
            "RETRIEVE: What pet?",
            "RETRIEVE: Where?",
        ]:
            send(client, agent.id, q)
            in_context_counts.append(
                len(client.agents.retrieve(agent_id=agent.id).message_ids or [])
            )
        check(
            "4. history does not grow between calls",
            len(set(in_context_counts)) == 1,
            f"in-context messages after each call: {in_context_counts}",
        )

        # 5. abandoned request: exactly one server run, cancelled, no duplicate writes,
        #    and the next call is not blocked
        def run_statuses() -> dict[str, str]:
            return {r.id: r.status for r in client.runs.list(agent_id=agent.id, limit=100)}

        before = run_statuses()
        try:
            send(client, agent.id, "STORE: User said: My sister Dana lives in Berlin.", timeout=1.0)
            check("5. abandoned request cancelled", False, "request finished before timeout; rerun")
        except APITimeoutError:
            client.agents.messages.cancel(agent_id=agent.id)
            deadline = time.time() + 15
            while True:
                new_runs = {k: v for k, v in run_statuses().items() if k not in before}
                if (
                    all(s not in ("created", "running") for s in new_runs.values())
                    or time.time() > deadline
                ):
                    break
                time.sleep(1)
            time.sleep(5)  # give any stray run time to write, so duplicates would show up
            dana = [
                p
                for p in client.agents.passages.list(agent_id=agent.id)
                if "dana" in p.text.lower()
            ]
            _, dt = send(client, agent.id, "RETRIEVE: Where does the user live?")
            check(
                "5. abandoned request cancelled",
                list(new_runs.values()) == ["cancelled"] and len(dana) <= 1 and dt < 30,
                f"server runs for abandoned request: {list(new_runs.values())}, "
                f"'Dana' passages: {len(dana)}, next call {dt:.1f}s",
            )
    finally:
        client.agents.delete(agent_id=agent.id)

    failed = [n for n, ok, _ in results if not ok]
    print("\nALL PASS" if not failed else f"\nFAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
