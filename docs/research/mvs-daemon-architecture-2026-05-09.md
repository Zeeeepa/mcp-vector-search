# MVS Daemon Architecture Design

**Date:** 2026-05-09
**Status:** Design phase — no code written yet
**Goal:** Replace the "spawn subprocess per search" model with a single persistent daemon

---

## Problem Statement

Every `mvs search` invocation today does this cold path:

```
asyncio.run(run_search(...))
  → create_embedding_function()          # load model from disk: ~2-5s
  → LanceVectorDatabase.initialize()     # open LanceDB connection: ~200ms
  → SemanticSearchEngine.search()        # encode + ANN: ~50-100ms
```

That 6,000–11,000 ms wall time is dominated by model load and Python interpreter startup, not the actual search. The MCP server already solves this correctly — it calls `warm_up()` once at startup and amortises the cost across all tool calls. The daemon extends this pattern to the CLI.

---

## Current Architecture (what exists today)

### Key classes and their roles

**`SemanticSearchEngine`** (`core/search.py`)
- Holds a `VectorDatabase` reference and `project_root`
- Lazy-loads `CrossEncoderReranker` on first reranked search
- `warm_up()` pre-loads the embedding model and cross-encoder
- `SearchMode` enum: `VECTOR`, `BM25`, `HYBRID`

**`LanceVectorDatabase`** (`core/lancedb_backend.py`)
- `lancedb.connect(str(persist_directory))` — synchronous, returns immediately
- `initialize()` opens the actual table, expensive only if compaction needed
- `close()` releases the connection
- Has an LRU result cache keyed on query string

**`ComponentFactory`** (`core/factory.py`)
- `create_standard_components(project_root, ...)` — the canonical bootstrap sequence
- Reads config, builds embedding function, database, indexer, optional search engine
- `resolve_index_path()` honours `INDEX_PATH` env var and explicit overrides
- The lance data lives at `<project_root>/.mcp-vector-search/lance/`

**`run_search()`** (`cli/commands/search.py`)
- Synchronous Typer callback → `asyncio.run(run_search(...))`
- Rebuilds everything from scratch on each invocation (model load, DB open, etc.)
- No daemon awareness today

**`MCPVectorSearchServer`** (`mcp/server.py`)
- Warm pattern: `initialize()` → `warm_up()` → keep alive for the session
- Single project root per server instance
- Runs over stdio, not Unix socket

**CLI entry points** (`pyproject.toml`)
- `mvs` → `mcp_vector_search.cli.main:cli_with_suggestions`
- `mcp-vector-search-mcp` → `mcp_vector_search.mcp.__main__:main`

---

## Proposed Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  ~/.mcp-vector-search/daemon.sock  (Unix domain socket, SOCK_STREAM) │
│  ~/.mcp-vector-search/daemon.pid                                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ newline-delimited JSON
          ┌─────────────────────┴─────────────────────┐
          │           MVS Daemon Process               │
          │                                            │
          │  ┌─────────────────────────────────┐       │
          │  │   EmbeddingModel (shared)        │       │
          │  │   CrossEncoderReranker (shared)  │       │
          │  └──────────────┬──────────────────┘       │
          │                 │ one per project path      │
          │  ┌──────────────▼──────────────────┐       │
          │  │   IndexRegistry (LRU, cap=5)     │       │
          │  │   key: canonical project_path    │       │
          │  │   val: SearcherEntry {           │       │
          │  │     engine: SemanticSearchEngine │       │
          │  │     last_used: float             │       │
          │  │     project_path: Path           │       │
          │  │   }                              │       │
          │  └──────────────────────────────────┘       │
          │                                            │
          │  asyncio event loop (single-threaded)       │
          │  Unix socket server (asyncio.start_unix_server) │
          └────────────────────────────────────────────┘

          ┌────────────────────┐     ┌─────────────────────────┐
          │  `mvs search`      │     │  MCP Server (stdio)     │
          │  DaemonClient      │     │  MCPVectorSearchServer  │
          │  (try socket first)│     │  (unchanged — own init) │
          └────────────────────┘     └─────────────────────────┘
```

---

## Section A: Daemon Process Design

### A.1 Entry point

New Typer sub-app registered in `cli/main.py`:

```
mvs daemon start    # fork daemon, write PID file, return
mvs daemon stop     # send SIGTERM via PID file, wait for exit
mvs daemon status   # print daemon PID, uptime, loaded indexes, memory
mvs daemon restart  # stop + start
```

File: `src/mcp_vector_search/cli/commands/daemon.py` (~200 LOC)

Registered in `cli/main.py` as:
```python
app.add_typer(daemon_app, name="daemon", help="Persistent search daemon")
```

### A.2 Socket and PID paths

```python
# src/mcp_vector_search/daemon/paths.py  (~30 LOC)

MVS_HOME = Path(os.environ.get("MVS_HOME", "~/.mcp-vector-search")).expanduser()
DAEMON_SOCK = MVS_HOME / "daemon.sock"
DAEMON_PID  = MVS_HOME / "daemon.pid"
DAEMON_LOG  = MVS_HOME / "daemon.log"
```

`MVS_HOME` can be overridden to support multiple isolated daemon instances (e.g. in tests). The MCP server uses stdio and never touches `daemon.sock`, so no conflict.

### A.3 Protocol: newline-delimited JSON over Unix socket

**Request (client → daemon):**
```json
{
  "id": "uuid4-string",
  "project_path": "/absolute/path/to/project",
  "query": "authentication token refresh",
  "limit": 10,
  "mode": "hybrid",
  "hybrid_alpha": 0.7,
  "filters": {"language": "python"},
  "use_rerank": true,
  "rerank_top_n": 50,
  "use_mmr": true,
  "diversity": 0.5,
  "similarity_threshold": null
}
```

**Response (daemon → client):**
```json
{
  "id": "uuid4-string",
  "project_path": "/absolute/path/to/project",
  "results": [...],
  "latency_ms": 52,
  "model_cached": true,
  "error": null
}
```

**Error response:**
```json
{
  "id": "uuid4-string",
  "error": "IndexNotFoundError",
  "detail": "No index at /path/to/project/.mcp-vector-search — run 'mvs index'",
  "results": null
}
```

Each message is a single JSON object terminated by `\n`. The client reads until `\n`. No length prefix needed — responses are bounded by result count.

### A.4 Daemon startup sequence

```
1. Check if DAEMON_SOCK exists and daemon is alive (send probe ping)
   → if alive: exit with "daemon already running"
2. mkdir -p MVS_HOME
3. Remove stale DAEMON_SOCK if present
4. Write PID file
5. Redirect stdout/stderr to DAEMON_LOG
6. asyncio.run(DaemonServer.run())
   a. load embedding model (warm_up with dummy query)
   b. start asyncio.start_unix_server(handle_client, DAEMON_SOCK)
   c. start idle-shutdown timer
   d. serve_forever()
7. On SIGTERM / SIGINT:
   a. cancel all in-flight requests
   b. close all LanceDB connections
   c. remove DAEMON_SOCK and DAEMON_PID
   d. exit 0
```

### A.5 Process isolation (fork model)

`mvs daemon start` uses `multiprocessing.Process(target=_daemon_main, daemon=False)` or a direct `fork()` to detach from the calling terminal. The child calls `os.setsid()` to become a session leader, then runs the asyncio event loop. The parent waits up to 2 seconds for the PID file to appear before returning.

Alternative: `subprocess.Popen([sys.executable, "-m", "mcp_vector_search.daemon.__main__"])` with `start_new_session=True`. This avoids shared file-descriptor inheritance and is simpler to implement cross-platform. Recommended approach.

### A.6 Auto-shutdown on idle

```python
# In DaemonServer
IDLE_TIMEOUT = int(os.environ.get("MVS_DAEMON_IDLE_SECONDS", "1800"))  # 30 min

async def _idle_watchdog(self):
    while True:
        await asyncio.sleep(60)
        idle_secs = time.monotonic() - self._last_request_time
        if idle_secs > IDLE_TIMEOUT:
            logger.info(f"Idle for {idle_secs:.0f}s — shutting down")
            self._shutdown_event.set()
            break
```

`_last_request_time` is updated on every incoming request completion.

### A.7 Estimated LOC

| File | LOC (est.) |
|------|-----------|
| `daemon/__main__.py` | 30 |
| `daemon/server.py` | 250 |
| `daemon/registry.py` | 120 |
| `daemon/protocol.py` | 60 |
| `daemon/paths.py` | 30 |
| `cli/commands/daemon.py` | 200 |
| **Total daemon** | **~690** |

---

## Section B: Index Registry (multi-project support)

### B.1 IndexRegistry

```
# src/mcp_vector_search/daemon/registry.py

class SearcherEntry:
    engine: SemanticSearchEngine
    database: LanceVectorDatabase
    project_path: Path
    last_used: float          # monotonic time
    embedding_model: str      # stored for health-check

class IndexRegistry:
    max_entries: int          # default 5, env MVS_DAEMON_MAX_INDEXES
    _entries: dict[str, SearcherEntry]   # key = str(canonical path)
    _shared_embedding_function           # one instance, shared across all entries
    _shared_reranker                     # one CrossEncoderReranker instance
```

### B.2 Shared embedding function

The embedding model is the expensive singleton. `create_embedding_function(model_name)` is called once at daemon startup with the default model. Every `LanceVectorDatabase` instance receives the same `embedding_function` object — this is safe because `CodeBERTEmbeddingFunction.__call__` is stateless (encodes text, returns numpy array).

If a project's config specifies a different `embedding_model`, the registry must handle model switching:
- Option A (simple, recommended for v1): Reject queries where `project.config.embedding_model != daemon_model` with a clear error, fall back to subprocess mode.
- Option B (complex): Keep a dict of `{model_name: embedding_function}` and load on demand.

For v1, Option A is sufficient — the vast majority of projects use the same auto-selected model.

### B.3 LRU eviction

```python
async def get_or_create(self, project_path: Path) -> SearcherEntry:
    key = str(project_path.resolve())
    if key in self._entries:
        entry = self._entries[key]
        entry.last_used = time.monotonic()
        return entry

    # Evict LRU if at capacity
    if len(self._entries) >= self.max_entries:
        lru_key = min(self._entries, key=lambda k: self._entries[k].last_used)
        await self._close_entry(self._entries.pop(lru_key))

    entry = await self._open_project(project_path)
    self._entries[key] = entry
    return entry

async def _open_project(self, project_path: Path) -> SearcherEntry:
    project_manager = ProjectManager(project_path)
    if not project_manager.is_initialized():
        raise IndexNotFoundError(project_path)

    config = project_manager.load_config()
    database = LanceVectorDatabase(
        persist_directory=config.index_path / "lance",
        embedding_function=self._shared_embedding_function,
        collection_name="vectors",
    )
    await database.initialize()

    engine = SemanticSearchEngine(
        database=database,
        project_root=project_path,
        similarity_threshold=config.similarity_threshold,
    )
    # Reranker is shared — inject it directly to avoid re-loading
    engine._reranker = self._shared_reranker

    return SearcherEntry(
        engine=engine,
        database=database,
        project_path=project_path,
        last_used=time.monotonic(),
        embedding_model=config.embedding_model or "auto",
    )

async def _close_entry(self, entry: SearcherEntry) -> None:
    try:
        await entry.database.close()
    except Exception:
        pass
```

### B.4 Index health check

On every request for a project, before executing the search:

```python
async def health_check(self, entry: SearcherEntry) -> bool:
    lance_path = entry.project_path / ".mcp-vector-search" / "lance"
    if not lance_path.exists():
        return False
    if hasattr(entry.database, "health_check"):
        return await entry.database.health_check()
    return True
```

If health check fails, the entry is evicted and the error is returned to the client with `"error": "IndexCorruptOrMissing"`. The client falls back to subprocess mode.

### B.5 Concurrency safety

The asyncio event loop is single-threaded. `get_or_create` and search calls are `async def` coroutines — no locks needed. LanceDB's synchronous `connect()` / table operations are called from the event loop via `asyncio.to_thread()` to avoid blocking (LanceDB Python API is currently synchronous at the connection level).

```python
self._db = await asyncio.to_thread(lancedb.connect, str(self.persist_directory))
```

This pattern is already used implicitly — the daemon makes it explicit.

---

## Section C: CLI Client Integration

### C.1 Modified `mvs search` flow

```
mvs search "query"
    │
    ├─ --no-daemon flag? ──YES──► existing asyncio.run(run_search(...))
    │
    ▼ NO
    DaemonClient.try_connect(DAEMON_SOCK)
    │
    ├─ socket missing or connection refused?
    │     │
    │     ├─ MVS_AUTO_START_DAEMON=true (default)?
    │     │     └─► spawn daemon in background, wait 3s for socket
    │     │              └─► retry connection
    │     │
    │     └─► fall back: asyncio.run(run_search(...))
    │
    └─ connected:
          send SearchRequest JSON
          receive SearchResponse JSON
          render results (same print_search_results() call)
```

### C.2 DaemonClient

```
# src/mcp_vector_search/daemon/client.py  (~120 LOC)

class DaemonClient:
    async def search(self, request: SearchRequest) -> SearchResponse: ...
    async def ping(self) -> bool: ...
    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]: ...
```

Connection timeout: 2 seconds. Read timeout: 30 seconds (long enough for cold-open of a new project index). If either times out, raise `DaemonTimeoutError` → fall back to subprocess.

### C.3 New CLI flags

```python
no_daemon: bool = typer.Option(
    False, "--no-daemon", help="Bypass daemon, run search in-process (slower)"
)
```

Environment variable `MVS_NO_DAEMON=1` also bypasses the daemon (useful for scripts and CI).

`MVS_AUTO_START_DAEMON` (default `true`) controls whether `mvs search` automatically starts the daemon if the socket is missing.

### C.4 Result serialisation

`SearchResult` is a Pydantic model — use `.model_dump(mode="json")` for the response payload. The client reconstructs results with `SearchResult.model_validate(item)` before passing to `print_search_results()`. `Path` fields serialise as strings; the client converts back with `Path(item["file_path"])`.

---

## Section D: Compatibility

### D.1 MCP server isolation

The MCP server (`mcp-vector-search-mcp` entry point) is completely unmodified. It:
- Runs over stdio, not a socket
- Has its own `MCPVectorSearchServer.initialize()` / `warm_up()` lifecycle
- Never reads `daemon.sock` or `daemon.pid`

The daemon lives at `~/.mcp-vector-search/daemon.sock`; the MCP server has no concept of `~/.mcp-vector-search/`. Zero conflict.

### D.2 Multiple MCP server instances

Each Claude Desktop project spawns its own `mcp-vector-search-mcp` process. These remain independent. The daemon is CLI-only in v1 — MCP server instances do not share the daemon's model cache (that is a possible future enhancement, not in scope here).

### D.3 Backward compatibility

- `mvs search` with no daemon running: falls back automatically, identical behaviour to today.
- `--no-daemon` flag: always available.
- `MVS_NO_DAEMON=1`: CI/scripting environments can set this to guarantee subprocess mode.
- `mvs daemon` subcommand is additive — no existing commands change signature.
- `ComponentFactory`, `SemanticSearchEngine`, `LanceVectorDatabase` are unchanged — the daemon is a thin orchestration layer on top.

### D.4 Index path resolution

The daemon uses the same `resolve_index_path()` logic from `core/factory.py`. The project path sent in the request becomes the `project_root`, and the standard `ProjectManager(project_root).load_config()` call determines where the lance index lives. No new path conventions introduced.

---

## Section E: Latency Estimates

### E.1 Current cold-start path

| Step | Time |
|------|------|
| Python interpreter startup | ~300ms |
| Import chain (torch, transformers, lancedb) | ~1,500–3,000ms |
| `create_embedding_function()` — load model weights | ~2,000–5,000ms |
| `LanceVectorDatabase.initialize()` | ~100–300ms |
| `SemanticSearchEngine.search()` — encode + ANN | ~50–100ms |
| **Total** | **~4,000–9,000ms** |

### E.2 Daemon warm path (model + index already loaded)

| Step | Time |
|------|------|
| Unix socket IPC (write request + read response) | <1ms |
| Query encoding (GPU: ~5ms, CPU MiniLM: ~25ms) | ~10–25ms |
| LanceDB ANN query | ~15–30ms |
| Cross-encoder reranking (50 candidates) | ~15–30ms |
| **Total** | **~40–85ms** |

### E.3 Daemon cold path (first query to a new project)

| Step | Time |
|------|------|
| Socket IPC | <1ms |
| Query encoding (model already loaded) | ~10–25ms |
| `LanceVectorDatabase.initialize()` (open table) | ~100–300ms |
| LanceDB ANN query | ~15–30ms |
| Cross-encoder reranking | ~15–30ms |
| **Total** | **~140–385ms** |

### E.4 Daemon auto-start path (socket missing, auto-start enabled)

| Step | Time |
|------|------|
| Spawn daemon subprocess | ~100ms |
| Daemon: Python imports + model load | ~3,000–5,000ms |
| Wait for socket to appear | included above |
| First query (cold project) | ~140–385ms |
| **Total first-ever invocation** | **~3,500–6,000ms** |
| **All subsequent invocations** | **~40–85ms** |

The auto-start cost is paid once per machine boot (or after 30-min idle shutdown).

---

## File Layout

```
src/mcp_vector_search/
├── daemon/
│   ├── __init__.py              (10 LOC)
│   ├── __main__.py              (30 LOC — daemon entry point, called by subprocess.Popen)
│   ├── paths.py                 (30 LOC — MVS_HOME, DAEMON_SOCK, DAEMON_PID, DAEMON_LOG)
│   ├── protocol.py              (60 LOC — SearchRequest, SearchResponse Pydantic models)
│   ├── registry.py              (120 LOC — IndexRegistry, SearcherEntry, LRU eviction)
│   ├── server.py                (250 LOC — DaemonServer, handle_client, idle watchdog)
│   └── client.py                (120 LOC — DaemonClient, try_connect, search, ping)
└── cli/
    └── commands/
        └── daemon.py            (200 LOC — Typer subcommand: start/stop/status/restart)
```

**Modifications to existing files:**

| File | Change | LOC delta |
|------|--------|-----------|
| `cli/commands/search.py` | Add `--no-daemon` flag; wrap `asyncio.run(run_search(...))` with daemon client try | +40 |
| `cli/main.py` | Register `daemon_app` Typer sub-app | +5 |
| `core/search.py` | Expose `_reranker` injection point (minor: already accessible) | 0 |

**Total new code:** ~820 LOC
**Total modified code:** ~45 LOC delta

---

## Implementation Order (migration path)

### Phase 1: Protocol and paths (no daemon yet)
1. `daemon/paths.py` — define MVS_HOME, paths
2. `daemon/protocol.py` — SearchRequest / SearchResponse models

### Phase 2: Registry and server
3. `daemon/registry.py` — IndexRegistry with LRU, health check
4. `daemon/server.py` — DaemonServer, asyncio Unix socket handler, idle watchdog
5. `daemon/__main__.py` — entry point called by subprocess

### Phase 3: CLI client and daemon command
6. `daemon/client.py` — DaemonClient with fallback logic
7. `cli/commands/daemon.py` — start/stop/status/restart Typer commands
8. Modify `cli/main.py` to register daemon sub-app

### Phase 4: Integrate into search command
9. Modify `cli/commands/search.py` — add `--no-daemon`, try daemon socket first

Each phase is independently testable and merge-able. Until Phase 4 is merged, existing behaviour is unchanged. Phase 4 is the only change visible to end users.

---

## Key Design Decisions and Rationale

**Unix domain socket over TCP:** Lower overhead, no port conflicts, natural access control via filesystem permissions (`chmod 600 daemon.sock`). Simpler than named pipes.

**asyncio single-threaded event loop:** LanceDB queries are the bottleneck, not CPU. A single event loop avoids locking complexity. `asyncio.to_thread()` is used only for the synchronous LanceDB `connect()` call.

**LRU cap of 5 indexes:** A LanceDB table connection holds the memory-mapped lance files in the OS page cache (~50–500 MB per large index). Capping at 5 prevents excessive RSS on machines with limited RAM. Configurable via `MVS_DAEMON_MAX_INDEXES`.

**Shared embedding function:** `CodeBERTEmbeddingFunction.__call__` is pure — takes text, returns vectors, no mutable state. Sharing one instance across all projects is safe and saves 500 MB+ of model weights.

**`--no-daemon` fallback:** Preserves the existing path for CI, debugging, and environments where the daemon cannot run (e.g. read-only home directories).

**No MCP server changes:** Keeping the MCP server untouched for v1 respects the principle of minimal blast radius. A future v2 could have the MCP server optionally query the daemon's socket to share the warm model.

**Subprocess spawn (not fork) for daemon start:** Using `subprocess.Popen` with `start_new_session=True` avoids inheriting file descriptors from the Typer CLI process (open terminal, signal handlers, etc.). The daemon is a clean process with its own signal handling.

---

## Open Questions (to resolve before implementation)

1. **Permissions on `~/.mcp-vector-search/`** — should `daemon.sock` be `chmod 600` or `660`? Consider multi-user machines where multiple users share a project index path.

2. **Model mismatch handling** — if project A uses `microsoft/codebert-base` and project B uses `sentence-transformers/all-MiniLM-L6-v2`, should the daemon reject project B's queries or load a second model? Recommend: v1 rejects and falls back to subprocess; v2 supports multi-model registry.

3. **Index mutation during search** — if `mvs index` runs while the daemon holds a LanceDB connection open, LanceDB uses file-level locking. The write will succeed; the daemon's open table handle may serve stale data until the table is re-opened. Mitigation: after `mvs index` completes, send `SIGUSR1` to daemon PID → daemon evicts that project's entry from the registry.

4. **`MVS_AUTO_START_DAEMON` default** — `true` is user-friendly but may surprise users in scripts. Consider `false` as safer default, with `mvs daemon start` being explicit. The welcome message from `mvs search` (when daemon is not running) can suggest `mvs daemon start`.

5. **Log rotation** — `daemon.log` will grow unbounded. Recommend `logging.handlers.RotatingFileHandler` with 10 MB max, 3 backups.
