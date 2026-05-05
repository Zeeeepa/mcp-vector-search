"""Tests for ContrastiveLCAScorer (core/lca_scorer.py)."""

from __future__ import annotations

import networkx as nx
import pytest

from mcp_vector_search.core.lca_scorer import ContrastiveLCAScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_graph() -> nx.DiGraph:
    """Build a minimal 4-level CONTAINS hierarchy.

    root
    ├── module:core
    │   └── file:kg.py
    │       └── class:KG
    │           ├── method:initialize
    │           └── method:add_entity
    └── module:parsers
        └── file:py.py
            └── class:PyParser
                └── method:parse
    """
    g: nx.DiGraph = nx.DiGraph()
    edges = [
        ("root", "module:core"),
        ("module:core", "file:kg.py"),
        ("file:kg.py", "class:KG"),
        ("class:KG", "method:initialize"),
        ("class:KG", "method:add_entity"),
        ("root", "module:parsers"),
        ("module:parsers", "file:py.py"),
        ("file:py.py", "class:PyParser"),
        ("class:PyParser", "method:parse"),
    ]
    for parent, child in edges:
        g.add_edge(parent, child, rel="CONTAINS")
    return g


@pytest.fixture
def scorer(synthetic_graph: nx.DiGraph) -> ContrastiveLCAScorer:
    return ContrastiveLCAScorer(synthetic_graph, root="root")


# ---------------------------------------------------------------------------
# Core scoring behavior
# ---------------------------------------------------------------------------


def test_baseline_is_two_times_max_depth(
    scorer: ContrastiveLCAScorer,
) -> None:
    # max depth = 4 (method nodes), baseline = 2 * 4 = 8
    assert scorer.baseline == pytest.approx(8.0)


def test_self_comparison_returns_one(scorer: ContrastiveLCAScorer) -> None:
    assert scorer.score("method:initialize", "method:initialize") == 1.0


def test_sibling_methods_score_high(scorer: ContrastiveLCAScorer) -> None:
    """Methods sharing a class should score highest among non-self pairs."""
    sibling = scorer.score("method:initialize", "method:add_entity")
    # depths 4 and 4, lca depth 3 -> penalty 1+1+0 = 2 -> 1 - 2/8 = 0.75
    assert sibling == pytest.approx(0.75)


def test_score_ordering_high_med_low(scorer: ContrastiveLCAScorer) -> None:
    """Sibling > same-module > cross-module > 0.0."""
    sibling = scorer.score("method:initialize", "method:add_entity")
    cross_module = scorer.score("method:initialize", "method:parse")
    assert sibling > cross_module
    assert cross_module >= 0.0
    # Methods at same depth across modules: depths 4&4, lca=root(0)
    # penalty = 4+4+0 = 8 -> 1 - 8/8 = 0.0
    assert cross_module == pytest.approx(0.0)


def test_class_vs_method_different_depth(
    scorer: ContrastiveLCAScorer,
) -> None:
    """Pairs at different depths get the contrastive penalty term."""
    # class:KG (depth 3) vs method:parse (depth 4); LCA = root (depth 0)
    # penalty = |3-0| + |4-0| + 0.5 * |3-4| = 3 + 4 + 0.5 = 7.5
    # 1 - 7.5 / 8 = 0.0625
    score = scorer.score("class:KG", "method:parse")
    assert score == pytest.approx(0.0625)


# ---------------------------------------------------------------------------
# score_query_vs_results batch API
# ---------------------------------------------------------------------------


def test_score_query_vs_results_returns_dict(
    scorer: ContrastiveLCAScorer,
) -> None:
    out = scorer.score_query_vs_results(
        "method:initialize",
        ["method:add_entity", "method:parse", "method:initialize"],
    )
    assert set(out.keys()) == {
        "method:add_entity",
        "method:parse",
        "method:initialize",
    }
    assert out["method:initialize"] == 1.0
    assert out["method:add_entity"] > out["method:parse"]


def test_score_query_vs_results_empty_list(
    scorer: ContrastiveLCAScorer,
) -> None:
    assert scorer.score_query_vs_results("method:initialize", []) == {}


# ---------------------------------------------------------------------------
# Graceful handling of unknown nodes
# ---------------------------------------------------------------------------


def test_unknown_node_returns_zero_no_raise(
    scorer: ContrastiveLCAScorer,
) -> None:
    assert scorer.score("method:initialize", "method:does_not_exist") == 0.0
    assert scorer.score("nope_a", "nope_b") == 0.0


def test_unknown_self_comparison_returns_zero(
    scorer: ContrastiveLCAScorer,
) -> None:
    """Self-comparison on an unknown node should still not crash and
    should report 0.0 (no signal) — mirroring the "graceful unknown"
    policy we use everywhere else."""
    assert scorer.score("ghost_node", "ghost_node") == 0.0


def test_score_detailed_includes_metadata(
    scorer: ContrastiveLCAScorer,
) -> None:
    detailed = scorer.score_detailed("method:initialize", "method:add_entity")
    assert detailed["lca"] == "class:KG"
    assert detailed["depth_a"] == 4
    assert detailed["depth_b"] == 4
    assert detailed["lca_depth"] == 3
    assert detailed["penalty"] == pytest.approx(2.0)
    assert detailed["geometric_resonance"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Cache invalidation / construction-time behavior
# ---------------------------------------------------------------------------


def test_lca_cache_is_used(
    scorer: ContrastiveLCAScorer, synthetic_graph: nx.DiGraph
) -> None:
    """Repeated calls hit the LCA cache (we verify via direct attr access)."""
    a, b = "method:initialize", "method:add_entity"
    _ = scorer.score(a, b)
    cache_key = (a, b) if a < b else (b, a)
    assert cache_key in scorer._lca_cache
    assert scorer._lca_cache[cache_key] == "class:KG"


def test_new_scorer_starts_with_empty_cache(
    synthetic_graph: nx.DiGraph,
) -> None:
    """A freshly-built scorer (post kg_build) has no cached LCA paths."""
    new = ContrastiveLCAScorer(synthetic_graph, root="root")
    assert new._lca_cache == {}


def test_explicit_baseline_override(synthetic_graph: nx.DiGraph) -> None:
    """Caller-supplied baseline takes precedence over auto-computed."""
    s = ContrastiveLCAScorer(synthetic_graph, root="root", baseline=20.0)
    assert s.baseline == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_score_is_symmetric(scorer: ContrastiveLCAScorer) -> None:
    a, b = "method:initialize", "method:parse"
    assert scorer.score(a, b) == scorer.score(b, a)
