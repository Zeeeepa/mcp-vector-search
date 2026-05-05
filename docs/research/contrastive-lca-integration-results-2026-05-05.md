# Contrastive LCA Integration — Results

**Date:** 2026-05-05
**Author:** Bob Matsuoka
**Status:** Implemented

## Summary

Wired the Contrastive LCA (Lowest Common Ancestor) "geometric resonance"
algorithm from the Nairobi Protocol GDE prototype (`scripts/test_lca_scoring.py`)
into the live `mcp-vector-search` knowledge-graph query path. The signal is
now combined with the existing vector-similarity / Kuzu ordering to break
ties in favor of structurally-related entities.

## What Changed

### New module — `core/lca_scorer.py`

- `ContrastiveLCAScorer`
  - Wraps a `networkx.DiGraph` (parent → child CONTAINS edges)
  - Pre-computes node depths from the root at construction time
  - Caches LCA paths (the hierarchy is static between KG builds)
  - Exposes `score(a, b) -> float`, `score_detailed(a, b) -> dict`,
    and `score_query_vs_results(query, [results]) -> dict`
  - Returns `0.0` (not raise) on unknown nodes — graceful fallback
- `build_lca_scorer_from_kuzu(connection, root_id="root")`
  - Queries `MATCH (a)-[:CONTAINS]->(b) RETURN a.id, b.id`
  - Materializes a `nx.DiGraph` from the result
  - Synthesizes a `"root"` node connecting to every top-level orphan

### `KnowledgeGraph` (core/knowledge_graph.py)

- New `_lca_scorer: ContrastiveLCAScorer | None` instance attribute
- New `async build_lca_scorer()` — lazy builds + caches
- New `async get_lca_scorer()` — returns cached scorer (builds on first call)
- New `invalidate_lca_scorer()` — invalidation hook
- `close()` and `close_sync()` now reset `_lca_scorer = None`
- New `_load_lca_baseline_from_metadata()` — reads `lca_baseline` from
  `kg_metadata.json` and applies it to the scorer

### `KGBuilder` (core/kg_builder.py)

- `_save_metadata()` now also computes and persists `lca_baseline`
  (= `2 * max_depth` of the CONTAINS tree) so query-time scores are
  consistent with the build-time tree

### MCP handler (mcp/kg_handlers.py)

- New module-level constant `LCA_WEIGHT = 0.15`
- `handle_kg_query()`:
  1. Resolves `entity` to a query anchor via `find_entity_by_name`
  2. Calls `await kg.get_lca_scorer()` (builds on first query post-build)
  3. Batch-scores all result IDs against the anchor
  4. Computes `final_score = vector_score + LCA_WEIGHT * lca_score`
  5. Stable-sorts results by `final_score` descending
  6. Adds `lca_score`, `geometric_resonance`, `final_score` per result
  7. Surfaces `lca_baseline` at the top of the response
  - Wraps the whole step in `try/except` so a failure here never breaks
    the query — LCA is a *signal*, not a contract

### Tests (tests/unit/core/test_lca_scorer.py)

14 tests covering:
- Baseline computation (`2 * max_depth`)
- Self-comparison = 1.0
- Score ordering (sibling > same-module > cross-module > 0.0)
- Symmetry: `score(a, b) == score(b, a)`
- Detailed metadata returned (`lca`, `depth_a`, `depth_b`, `penalty`, etc.)
- Batch `score_query_vs_results()`
- Graceful unknown-node handling (returns `0.0`, never raises)
- LCA cache hit verification
- Fresh scorer starts with empty cache (post-build invalidation behavior)
- Explicit baseline override

All 14 pass in 0.11s.

### Dependency

- Added `networkx>=3.0` to `[project].dependencies` in `pyproject.toml`
  (was previously available transitively via torch)

## How Scores Combine

```
final_score = vector_score + 0.15 * lca_score
                              └── ContrastiveLCAScorer ──┘
                                  range [0.0, 1.0]
```

The `LCA_WEIGHT = 0.15` was chosen so that:
- A perfect vector match (1.0) with no structural relation still beats a
  weaker vector match (~0.85) with sibling resonance (0.75 * 0.15 = 0.11)
- Two near-equal vector matches (0.75 vs 0.74) get re-ordered by structure
- The KG-aware ordering remains a *nudge* rather than a takeover

## Example: Before/After Ranking

For query `entity="initialize"` (anchor = `class:KG.initialize`), suppose
Kuzu returns 4 candidates with these vector scores:

| Result                | vector | lca   | final  |
|-----------------------|-------:|------:|-------:|
| `KG.add_entity`       | 0.78   | 0.75  | 0.8925 |
| `KGBuilder.build`     | 0.81   | 0.25  | 0.8475 |
| `PythonParser.parse`  | 0.80   | 0.00  | 0.8000 |
| `BaseParser.validate` | 0.79   | 0.00  | 0.7900 |

**Before:** `KGBuilder.build` (0.81) > `PythonParser.parse` (0.80) >
`BaseParser.validate` (0.79) > `KG.add_entity` (0.78)

**After:** `KG.add_entity` (0.89) > `KGBuilder.build` (0.85) >
`PythonParser.parse` (0.80) > `BaseParser.validate` (0.79)

The sibling method (`KG.add_entity`, same class as the anchor) jumps from
4th to 1st despite its weaker raw vector score, because the structural
signal correctly identifies it as the more relevant result.

## Known Limitations

1. **Anchor resolution required** — When the query has no resolvable entity
   (pure free-text query with no matching node), LCA scoring is skipped
   entirely. The handler degrades silently to the original ordering.
2. **CONTAINS-only signal** — We deliberately use only the CONTAINS
   hierarchy; CALLS / IMPORTS / INHERITS edges are not included. That
   keeps the algorithm tree-shaped (LCA is well-defined) and the signal
   interpretable.
3. **Synthetic root** — When the KG has no single real root entity, we
   inject a `"root"` node connected to every top-level orphan. This
   slightly affects depth values for the orphans but keeps the algorithm
   correct.
4. **Vector-score availability** — The handler treats the per-result
   `score` field as the vector signal and defaults to `1.0` when absent
   (preserving Kuzu's original ordering on ties). This is conservative —
   when Kuzu doesn't return scores, the LCA contribution dominates.
5. **Cache lifetime** — The LCA path cache lives for the lifetime of the
   read-only `KnowledgeGraph` singleton. It is correctly cleared when the
   RO singleton is closed/reopened around a `kg_build`.
6. **Baseline reproducibility** — `lca_baseline` is persisted in
   `kg_metadata.json` and re-loaded on startup; this guarantees the
   normalization constant is identical at query time and build time.

## Files Touched

- `src/mcp_vector_search/core/lca_scorer.py` (new)
- `src/mcp_vector_search/core/knowledge_graph.py`
- `src/mcp_vector_search/core/kg_builder.py`
- `src/mcp_vector_search/mcp/kg_handlers.py`
- `tests/unit/core/test_lca_scorer.py` (new)
- `pyproject.toml` (added `networkx>=3.0`)
- `docs/research/contrastive-lca-integration-results-2026-05-05.md` (this file)

## Verification

```
$ uv run pytest tests/unit/core/test_lca_scorer.py -q --no-cov
..............                                                           [100%]
14 passed in 0.11s
```

End-to-end functional check:

```
$ uv run python -c "from mcp_vector_search.core.lca_scorer import ContrastiveLCAScorer; ..."
baseline: 8.0
siblings: 0.75
cross-module: 0.0
self: 1.0
unknown: 0.0
batch: {'method:add_entity': 0.75, 'method:parse': 0.0}
```
