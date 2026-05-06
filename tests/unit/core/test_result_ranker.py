"""Unit tests for ResultRanker (post-retrieval ranking pipeline).

Covers the public surface of `mcp_vector_search.core.result_ranker.ResultRanker`,
including a critical regression test that ensures `similarity_score` never goes
below 0.0 after penalty application (see commit 50e5490).

The ResultRanker exposes a single public method, `rerank_results`. LCA scoring
and MMR diversity filtering are implemented in separate modules and are tested
elsewhere; this file covers only the boost/penalty/clamp logic that lives in
`result_ranker.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_vector_search.core.models import SearchResult
from mcp_vector_search.core.result_ranker import ResultRanker

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_result(
    *,
    content: str = "def foo():\n    return 1",
    file_path: Path | None = None,
    similarity_score: float = 0.5,
    rank: int = 1,
    chunk_type: str = "code",
    function_name: str | None = None,
    class_name: str | None = None,
    language: str = "python",
    start_line: int = 1,
    end_line: int = 2,
) -> SearchResult:
    """Create a SearchResult with sensible defaults for ranking tests."""
    return SearchResult(
        content=content,
        file_path=file_path if file_path is not None else Path("src/module.py"),
        start_line=start_line,
        end_line=end_line,
        language=language,
        similarity_score=similarity_score,
        rank=rank,
        chunk_type=chunk_type,
        function_name=function_name,
        class_name=class_name,
    )


@pytest.fixture
def ranker() -> ResultRanker:
    return ResultRanker()


# --------------------------------------------------------------------------- #
# Regression test (CRITICAL — issue #160)
# --------------------------------------------------------------------------- #


class TestScoreClampingRegression:
    """Regression tests for the score-clamping fix in commit 50e5490."""

    def test_similarity_score_never_negative(self, ranker: ResultRanker) -> None:
        """Regression: penalties must not drive similarity_score below 0.0.

        Before commit 50e5490 the cumulative penalty of:
          - boilerplate function name (-0.15)
          - test file path (-0.02)
          - deep path (-0.01)
        could push a low starting score (e.g. 0.05) below zero, violating the
        ``0.0 <= similarity_score <= 1.0`` invariant asserted by integration
        tests (test_full_indexing_workflow, test_search_integration_workflow).
        """
        # Three results, each constructed so the total penalty exceeds the
        # initial similarity_score.
        results = [
            make_result(
                similarity_score=0.05,
                function_name="__init__",  # boilerplate, -0.15
                file_path=Path("a/b/c/d/e/f/test_module.py"),  # test + deep
            ),
            make_result(
                similarity_score=0.0,
                function_name="__del__",
                file_path=Path("a/b/c/d/e/f/test_thing.py"),
            ),
            make_result(
                similarity_score=0.10,
                function_name="__repr__",
                file_path=Path("a/b/c/d/e/f/g/test_x.py"),
            ),
        ]

        reranked = ranker.rerank_results(results, query="search classes")

        for r in reranked:
            assert r.similarity_score >= 0.0, (
                f"similarity_score went negative: {r.similarity_score}"
            )
            assert r.similarity_score <= 1.0, (
                f"similarity_score exceeded 1.0: {r.similarity_score}"
            )

    def test_score_clamps_to_exactly_zero(self, ranker: ResultRanker) -> None:
        """A heavy penalty drives the final score to exactly 0.0, not negative."""
        result = make_result(
            similarity_score=0.01,
            function_name="__init__",  # boilerplate -> -0.15 penalty
            file_path=Path("a/b/c/d/e/f/test_x.py"),  # deep + test penalties
            content="x",  # no query word matches
        )

        reranked = ranker.rerank_results([result], query="search")
        # Without clamping the score would be roughly 0.01 - 0.15 - 0.02 - 0.01
        # plus minor +0.02 source-file boost = clearly negative. With clamping
        # it must equal 0.0.
        assert reranked[0].similarity_score == 0.0

    def test_score_clamps_to_one(self, ranker: ResultRanker) -> None:
        """Boosts that would push score above 1.0 must clamp to exactly 1.0."""
        result = make_result(
            similarity_score=0.95,
            function_name="search",  # exact identifier match (+0.15)
            class_name="search",  # exact identifier match (+0.15)
            file_path=Path("search.py"),  # exact filename match
            content="search search search",
            chunk_type="function",
        )

        reranked = ranker.rerank_results([result], query="search")
        assert reranked[0].similarity_score == 1.0


# --------------------------------------------------------------------------- #
# Public API: rerank_results behaviour
# --------------------------------------------------------------------------- #


class TestRerankResultsBasics:
    """Basic correctness of rerank_results."""

    def test_empty_results_returns_empty(self, ranker: ResultRanker) -> None:
        assert ranker.rerank_results([], query="anything") == []

    def test_single_result_returned(self, ranker: ResultRanker) -> None:
        result = make_result(similarity_score=0.5)
        out = ranker.rerank_results([result], query="foo")
        assert len(out) == 1
        assert out[0] is result  # in-place ranking returns same objects
        assert out[0].rank == 1

    def test_results_sorted_descending(self, ranker: ResultRanker) -> None:
        """Output is sorted by final similarity_score descending."""
        a = make_result(similarity_score=0.30, content="alpha")
        b = make_result(similarity_score=0.70, content="beta")
        c = make_result(similarity_score=0.50, content="gamma")

        out = ranker.rerank_results([a, b, c], query="zzz_no_match_zzz")
        scores = [r.similarity_score for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_assigned_in_order(self, ranker: ResultRanker) -> None:
        """`rank` field is assigned 1..N in score-sorted order."""
        results = [
            make_result(similarity_score=0.2),
            make_result(similarity_score=0.8),
            make_result(similarity_score=0.5),
        ]
        out = ranker.rerank_results(results, query="zzz")
        for i, r in enumerate(out, start=1):
            assert r.rank == i


# --------------------------------------------------------------------------- #
# Public API: boost and penalty factors
# --------------------------------------------------------------------------- #


class TestBoosts:
    """Verify each boost factor adjusts scores in the right direction."""

    def test_exact_function_name_match_boosts(self, ranker: ResultRanker) -> None:
        match = make_result(
            similarity_score=0.5,
            function_name="search_engine",
            file_path=Path("src/x.py"),
            content="pass",
        )
        no_match = make_result(
            similarity_score=0.5,
            function_name="unrelated",
            file_path=Path("src/x.py"),
            content="pass",
        )
        ranker.rerank_results([match], query="search")
        ranker.rerank_results([no_match], query="search")
        assert match.similarity_score > no_match.similarity_score

    def test_exact_class_name_match_boosts(self, ranker: ResultRanker) -> None:
        match = make_result(
            similarity_score=0.5,
            class_name="SearchService",
            file_path=Path("src/x.py"),
            content="pass",
        )
        no_match = make_result(
            similarity_score=0.5,
            class_name="OtherThing",
            file_path=Path("src/x.py"),
            content="pass",
        )
        ranker.rerank_results([match], query="search")
        ranker.rerank_results([no_match], query="search")
        assert match.similarity_score > no_match.similarity_score

    def test_filename_match_boosts(self, ranker: ResultRanker) -> None:
        match = make_result(
            similarity_score=0.5,
            file_path=Path("src/search.py"),
            content="pass",
        )
        no_match = make_result(
            similarity_score=0.5,
            file_path=Path("src/other.py"),
            content="pass",
        )
        ranker.rerank_results([match], query="search")
        ranker.rerank_results([no_match], query="search")
        assert match.similarity_score > no_match.similarity_score

    def test_function_chunk_type_boosted_over_class(self, ranker: ResultRanker) -> None:
        """Functions get a larger structural boost than classes."""
        fn = make_result(similarity_score=0.5, chunk_type="function", content="pass")
        cls = make_result(similarity_score=0.5, chunk_type="class", content="pass")
        ranker.rerank_results([fn], query="zzz")
        ranker.rerank_results([cls], query="zzz")
        assert fn.similarity_score > cls.similarity_score

    def test_shallow_path_boosted_over_deep(self, ranker: ResultRanker) -> None:
        shallow = make_result(
            similarity_score=0.5,
            file_path=Path("src/m.py"),  # depth 2
            content="pass",
        )
        deep = make_result(
            similarity_score=0.5,
            file_path=Path("a/b/c/d/e/f/m.py"),  # depth 7
            content="pass",
        )
        ranker.rerank_results([shallow], query="zzz")
        ranker.rerank_results([deep], query="zzz")
        assert shallow.similarity_score > deep.similarity_score

    def test_content_density_boost(self, ranker: ResultRanker) -> None:
        """More query-word matches in content should produce higher score."""
        many = make_result(
            similarity_score=0.5,
            content="alpha beta gamma",
            file_path=Path("src/x.py"),
        )
        none = make_result(
            similarity_score=0.5,
            content="zzz",
            file_path=Path("src/x.py"),
        )
        ranker.rerank_results([many], query="alpha beta gamma")
        ranker.rerank_results([none], query="alpha beta gamma")
        assert many.similarity_score > none.similarity_score


class TestPenalties:
    """Verify each penalty factor reduces scores."""

    def test_test_file_penalty(self, ranker: ResultRanker) -> None:
        test_file = make_result(
            similarity_score=0.5,
            file_path=Path("src/test_module.py"),
            content="pass",
        )
        prod_file = make_result(
            similarity_score=0.5,
            file_path=Path("src/module.py"),
            content="pass",
        )
        ranker.rerank_results([test_file], query="zzz")
        ranker.rerank_results([prod_file], query="zzz")
        assert test_file.similarity_score < prod_file.similarity_score

    def test_boilerplate_penalty_applied(self, ranker: ResultRanker) -> None:
        """`__init__` (Python boilerplate) is penalised when not in query."""
        boilerplate = make_result(
            similarity_score=0.5,
            function_name="__init__",
            file_path=Path("src/x.py"),
            content="pass",
        )
        normal = make_result(
            similarity_score=0.5,
            function_name="custom_method",
            file_path=Path("src/x.py"),
            content="pass",
        )
        ranker.rerank_results([boilerplate], query="search classes")
        ranker.rerank_results([normal], query="search classes")
        assert boilerplate.similarity_score < normal.similarity_score

    def test_boilerplate_penalty_skipped_when_query_explicit(
        self, ranker: ResultRanker
    ) -> None:
        """No penalty when the user explicitly searches for the boilerplate name."""
        with_query = make_result(
            similarity_score=0.5,
            function_name="__init__",
            file_path=Path("src/x.py"),
            content="pass",
        )
        without_query = make_result(
            similarity_score=0.5,
            function_name="__init__",
            file_path=Path("src/x.py"),
            content="pass",
        )
        ranker.rerank_results([with_query], query="find __init__ methods")
        ranker.rerank_results([without_query], query="search classes")
        assert with_query.similarity_score > without_query.similarity_score

    def test_deep_path_penalty(self, ranker: ResultRanker) -> None:
        deep = make_result(
            similarity_score=0.5,
            file_path=Path("a/b/c/d/e/f/m.py"),
            content="pass",
        )
        mid = make_result(
            similarity_score=0.5,
            file_path=Path("a/b/c/d/m.py"),  # depth 5, no boost or penalty
            content="pass",
        )
        ranker.rerank_results([deep], query="zzz")
        ranker.rerank_results([mid], query="zzz")
        assert deep.similarity_score < mid.similarity_score


# --------------------------------------------------------------------------- #
# Public API: NLP boost (via _calculate_nlp_boost as exercised by rerank)
# --------------------------------------------------------------------------- #


class _DuckResult:
    """Minimal duck-typed stand-in for SearchResult used to drive the NLP
    metadata path of `_calculate_nlp_boost`, which inspects ``__dict__`` for
    ``nlp_keywords``/``nlp_code_refs``/``nlp_technical_terms`` attributes.

    Pydantic ``SearchResult`` doesn't store arbitrary attrs in ``__dict__``,
    so we use a simple object to exercise that branch.
    """

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestNLPBoost:
    """Exercises `_calculate_nlp_boost` through both string and list metadata."""

    def test_no_nlp_metadata_returns_zero_boost(self, ranker: ResultRanker) -> None:
        """A result without any NLP fields produces a 0.0 boost."""
        result = _DuckResult(content="anything")
        boost = ranker._calculate_nlp_boost(result, query_words={"foo"})
        assert boost == 0.0

    def test_missing_content_attr_short_circuits(self, ranker: ResultRanker) -> None:
        """Objects without a ``content`` attribute exit immediately."""

        class NoContent:
            pass

        boost = ranker._calculate_nlp_boost(NoContent(), query_words={"foo"})
        assert boost == 0.0

    def test_nlp_keyword_string_metadata_boosts(self, ranker: ResultRanker) -> None:
        """Comma-separated string NLP metadata applies a per-match boost."""
        result = _DuckResult(
            content="x",
            nlp_keywords="alpha,beta,gamma",
        )
        # Two of three keywords match -> 2 * _BOOST_NLP_KEYWORD (0.02)
        boost = ranker._calculate_nlp_boost(result, query_words={"alpha", "beta"})
        assert boost == pytest.approx(2 * ResultRanker._BOOST_NLP_KEYWORD)

    def test_nlp_code_ref_list_metadata_boosts(self, ranker: ResultRanker) -> None:
        """List-form NLP metadata is also accepted (LanceDB native list path)."""
        result = _DuckResult(
            content="x",
            nlp_code_refs=["search_engine", "indexer"],
        )
        boost = ranker._calculate_nlp_boost(result, query_words={"search"})
        # Only "search_engine" matches -> 1 * _BOOST_NLP_CODE_REF
        assert boost == pytest.approx(ResultRanker._BOOST_NLP_CODE_REF)

    def test_nlp_technical_terms_boost(self, ranker: ResultRanker) -> None:
        result = _DuckResult(
            content="x",
            nlp_technical_terms="LanceDB,KuzuDB,SQLite",
        )
        boost = ranker._calculate_nlp_boost(result, query_words={"lancedb"})
        assert boost == pytest.approx(ResultRanker._BOOST_NLP_TECHNICAL_TERM)

    def test_nlp_combined_boosts_sum(self, ranker: ResultRanker) -> None:
        """All three metadata categories contribute additively."""
        result = _DuckResult(
            content="x",
            nlp_keywords="search",
            nlp_code_refs="search_index",
            nlp_technical_terms="SearchService",
        )
        boost = ranker._calculate_nlp_boost(result, query_words={"search"})
        expected = (
            ResultRanker._BOOST_NLP_KEYWORD
            + ResultRanker._BOOST_NLP_CODE_REF
            + ResultRanker._BOOST_NLP_TECHNICAL_TERM
        )
        assert boost == pytest.approx(expected)

    def test_nlp_no_match_no_boost(self, ranker: ResultRanker) -> None:
        result = _DuckResult(
            content="x",
            nlp_keywords="alpha,beta",
            nlp_code_refs="indexer",
            nlp_technical_terms="LanceDB",
        )
        boost = ranker._calculate_nlp_boost(result, query_words={"zzz"})
        assert boost == 0.0

    def test_pydantic_search_result_no_nlp_boost(self, ranker: ResultRanker) -> None:
        """Pydantic SearchResult lacks NLP fields -> ranker still produces
        valid scores without raising."""
        result = make_result(
            similarity_score=0.5,
            file_path=Path("src/x.py"),
            content="pass",
        )
        ranker.rerank_results([result], query="zzz_no_match_zzz")
        assert 0.0 <= result.similarity_score <= 1.0


# --------------------------------------------------------------------------- #
# Combined behaviour
# --------------------------------------------------------------------------- #


class TestIntegratedRanking:
    """End-to-end: many factors interacting, output well-formed."""

    def test_multi_result_invariants(self, ranker: ResultRanker) -> None:
        results = [
            make_result(
                similarity_score=0.7,
                function_name="search_engine",
                file_path=Path("src/search.py"),
                chunk_type="function",
                content="search results",
            ),
            make_result(
                similarity_score=0.6,
                function_name="__init__",
                file_path=Path("a/b/c/d/e/f/test_x.py"),
                content="pass",
            ),
            make_result(
                similarity_score=0.4,
                class_name="OtherThing",
                file_path=Path("src/other.py"),
                chunk_type="class",
                content="zzz",
            ),
        ]
        out = ranker.rerank_results(results, query="search")

        # Invariants: all scores in [0, 1], strictly sorted desc, ranks 1..N
        for r in out:
            assert 0.0 <= r.similarity_score <= 1.0
        scores = [r.similarity_score for r in out]
        assert scores == sorted(scores, reverse=True)
        assert [r.rank for r in out] == [1, 2, 3]

        # The query-relevant result with the highest base score should be #1
        assert out[0].function_name == "search_engine"
