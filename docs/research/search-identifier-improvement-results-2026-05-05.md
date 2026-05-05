# Search improvements for proper-noun / SDK-name queries — results

**Date**: 2026-05-05
**Status**: shipped
**Related research**: `docs/research/search-mode-hybrid-alpha-proper-noun-improvement-2026-03-10.md`

## Problem

Hybrid search uses RRF fusion with `hybrid_alpha=0.7` (70% vector / 30% BM25)
by default. For identifier-style queries — package names, SDK names, dotted
hostnames, npm scoped packages, CamelCase symbols — the bi-encoder
(all-MiniLM-L6-v2) has effectively no signal: those tokens are not natural
language. BM25 was the right fallback, but at 30% weight it could not
overcome a noisy vector head, **and** the BM25 tokenizer was discarding
the very tokens we needed (`getstream.io` → `["getstream", "io"]`,
`@tanstack/query` → `["tanstack", "query"]`).

## What changed

Five coordinated improvements, all already partially in tree from
prior work — this pass completed and aligned them with the spec.

### 1. `is_identifier_query()` in `core/query_processor.py`
Detects identifier-style queries via four regex patterns plus a small
package-keyword set:

- Dotted names: `getstream.io`, `io.getstream.chat`
- camelCase **and** PascalCase-with-internal-cap: `getStream`, `StreamApp`
- npm scoped packages: `@tanstack/query` (drops the `\b` before `@`,
  which never matches against a non-word boundary)
- Multi-segment hyphenated packages: `react-activity-feed`
- Keyword fallback: any of `sdk`, `npm`, `pip`, `pypi`, `crate`, `gem`, …

### 2. Auto-routing in `core/search.py::_search_internal()`
When `search_mode == HYBRID`, `hybrid_alpha == 0.7` (i.e. user is on
defaults), and `is_identifier_query(query)` returns True, alpha is
lowered to `0.2` and a debug log line is emitted. Explicit user
overrides are respected.

### 3. BM25 tokenizer + version stamp in `core/bm25_backend.py`
The tokenizer now runs three passes:

1. Compound tokens via `[\w][\w.\-/]*[\w]` — preserves
   `getstream.io`, `@tanstack/query`, `react-activity-feed` as single
   tokens.
2. Plain `\w+` word tokens for partial matching.
3. snake_case / camelCase sub-word splitting so natural-language
   queries like `find by tag` still match `find_by_tag_docs`.

Because pass 1 changes the surface tokens that get indexed, an existing
on-disk index built with the v1 tokenizer would silently degrade query
results. We added `BM25_TOKENIZER_VERSION = 2` and:

- Stamp the version into the pickled `index_data` on `save()`.
- Validate the version on `load()`. On mismatch we log a clear
  warning, **disable BM25** for this process, and instruct the user
  to run `mcp-vector-search index --force`.
- Surface the version in `get_stats()`.

### 4. Mode controls on `search_context` and `search_similar`
`search_mode` and `hybrid_alpha` are now plumbed end-to-end through:

- `mcp/tool_schemas.py` (both schemas)
- `mcp/search_handlers.py` (both handlers)
- `core/search.py::search_by_context()` and `search_similar()`
- `cli/commands/search.py::run_similar_search()` and
  `run_context_search()`, with forwarding from `search_main`.

### 5. SDK-name tip on empty results
When a search returns zero results in the CLI, and the query looks
like an identifier query, we now print:

```
💡 This query looks like an SDK/package name. Try:
     --search-mode bm25       (exact keyword matching)
     --hybrid-alpha 0.2       (80% BM25, 20% vector)
```

## Verification

`scripts/test_identifier_search.py` classifies a curated set of
16 queries (10 identifier-style, 6 natural-language). Output:

```
16/16 cases passed
```

Targeted unit tests:

- `tests/unit/mcp/test_hybrid_search.py`: 24 passed
- `tests/unit/core/test_search.py`: 29 passed, 3 pre-existing
  failures (stale-result filtering on test fixtures, identical to
  failures on `main` before this work — confirmed via `git stash`).

Pyright on changed files:

- `core/query_processor.py`, `core/bm25_backend.py`,
  `core/search.py`, `mcp/tool_schemas.py`, `mcp/search_handlers.py`:
  0 errors.
- `cli/commands/search.py`: 4 pre-existing errors unrelated to this
  change.

## Before / after behaviour (qualitative)

For a query like `getstream.io`:

| Path                    | Before                                       | After                                           |
| ----------------------- | -------------------------------------------- | ----------------------------------------------- |
| BM25 tokens for chunk   | `getstream`, `io` (split apart)              | `getstream.io`, `getstream`, `io`               |
| Hybrid alpha            | `0.7` (vector dominates)                     | `0.2` (BM25 dominates) when user is on defaults |
| User experience on miss | "no results, try lower threshold"            | also prints SDK/package tip                     |
| MCP `search_context`    | no way to set `search_mode` / `hybrid_alpha` | both exposed                                    |

## Known limitations

- **Reindex required for the tokenizer fix.** Existing indexes built
  before this commit will be refused on load with a warning until
  `mcp-vector-search index --force` is run. This is intentional — using
  v1 tokens with a v2 query tokenizer would silently degrade results.
- The `is_identifier_query` heuristic only kicks in when the user is
  on the default `hybrid_alpha=0.7`. If a user has explicitly set
  `--hybrid-alpha 0.7` with the intent of "70/30 vector-leaning even
  for identifiers", we still demote to 0.2. We considered tracking
  "user-supplied vs default" through the call chain but decided the
  cost of plumbing a sentinel through several public APIs outweighed
  the benefit; the value 0.7 is already the documented default.
- The detector does not yet recognise dotted Java/Kotlin package
  paths with three or more segments without a recognisable TLD/host
  pattern (e.g. `com.example.foo.Bar` matches via the dotted-name
  rule, but `foo.bar.baz` also matches — this is acceptable for now
  because BM25 still does the right thing for those queries).
- The CLI tip uses a heuristic check rather than calling
  `is_identifier_query()` directly so we can also catch non-default
  `hybrid_alpha` values (≥0.5). This means the tip can occasionally
  appear for non-identifier queries that happen to contain a `.` or
  `-`. Acceptable false-positive rate.

## Files touched

- `src/mcp_vector_search/core/query_processor.py` — patterns + helper
- `src/mcp_vector_search/core/bm25_backend.py` — tokenizer version stamp
- `src/mcp_vector_search/core/search.py` — auto-detect, threading params
- `src/mcp_vector_search/mcp/tool_schemas.py` — schema fields
- `src/mcp_vector_search/mcp/search_handlers.py` — handler plumbing
- `src/mcp_vector_search/cli/commands/search.py` — CLI plumbing + tip
- `scripts/test_identifier_search.py` — classification benchmark
- `docs/research/search-identifier-improvement-results-2026-05-05.md` — this doc
