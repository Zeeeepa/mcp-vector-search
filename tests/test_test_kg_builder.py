"""Tests for TestKGBuilder (issue #156).

These tests run against a real KuzuDB instance backed by ``tmp_path``.
They are intentionally narrow: they verify that the builder creates
TestSuite/TestCase nodes, BELONGS_TO_SUITE edges, and TESTS edges via
name-convention matching, and that the KG stats include the new counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_vector_search.core.knowledge_graph import CodeEntity, KnowledgeGraph
from mcp_vector_search.core.test_kg_builder import TestKGBuilder


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    """Initialize a fresh KuzuDB-backed knowledge graph for testing."""
    db_path = tmp_path / "kg"
    graph = KnowledgeGraph(db_path)
    graph.initialize_sync()
    return graph


def _prod_chunk(name: str, file_path: str, kind: str = "function") -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "function_name": name if kind != "class" else None,
        "class_name": name if kind == "class" else None,
        "chunk_type": kind,
        "content": f"def {name}():\n    return 1\n",
        "start_line": 1,
        "end_line": 5,
        "chunk_id": f"prod::{file_path}::{name}",
    }


def _test_chunk(
    name: str,
    file_path: str,
    content: str | None = None,
    kind: str = "function",
) -> dict:
    return {
        "file_path": file_path,
        "name": name,
        "function_name": name if kind != "class" else None,
        "class_name": name if kind == "class" else None,
        "chunk_type": kind,
        "content": content
        or f"import pytest\n\ndef {name}():\n    assert foo() == 1\n",
        "start_line": 10,
        "end_line": 20,
        "chunk_id": f"test::{file_path}::{name}",
    }


def _seed_code_entity(kg: KnowledgeGraph, name: str, file_path: str) -> str:
    entity = CodeEntity(
        id=f"prod::{file_path}::{name}",
        name=name,
        entity_type="function",
        file_path=file_path,
    )
    kg.add_entities_batch_sync([entity])
    return entity.id


def test_build_creates_test_suite_and_case_nodes(kg: KnowledgeGraph, tmp_path: Path):
    """A test file with one test function should produce 1 suite + 1 case."""
    builder = TestKGBuilder(kg, tmp_path)

    # Seed production entity that the test convention will resolve to
    _seed_code_entity(kg, "foo", "src/mod.py")

    chunks = [
        _prod_chunk("foo", "src/mod.py"),
        _test_chunk("test_foo", "tests/test_mod.py"),
    ]
    stats = builder.build(chunks)

    assert stats["test_suites"] == 1
    assert stats["test_cases"] == 1
    assert stats["belongs_to_edges"] == 1
    # TESTS edge resolves test_foo -> CodeEntity foo
    assert stats["tests_edges"] >= 1


def test_test_suite_present_in_kg(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    builder.build([_test_chunk("test_a", "tests/test_x.py")])

    result = kg._execute_query("MATCH (ts:TestSuite) RETURN ts.name AS name")
    names = []
    while result.has_next():
        names.append(result.get_next()[0])
    # The TestSuite name is the file stem
    assert "test_x" in names


def test_test_case_present_in_kg(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    builder.build(
        [
            _test_chunk("test_alpha", "tests/test_x.py"),
            _test_chunk("test_beta", "tests/test_x.py"),
        ]
    )

    result = kg._execute_query("MATCH (tc:TestCase) RETURN tc.name AS name")
    names = []
    while result.has_next():
        names.append(result.get_next()[0])
    assert "test_alpha" in names
    assert "test_beta" in names


def test_belongs_to_suite_edge(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    builder.build([_test_chunk("test_a", "tests/test_x.py")])

    result = kg._execute_query(
        """
        MATCH (tc:TestCase)-[:BELONGS_TO_SUITE]->(ts:TestSuite)
        RETURN tc.name AS tc, ts.name AS ts
        """
    )
    rows = []
    while result.has_next():
        rows.append(tuple(result.get_next()))
    assert ("test_a", "test_x") in rows


def test_tests_edge_via_name_convention(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    _seed_code_entity(kg, "calculate", "src/calc.py")
    builder.build([_test_chunk("test_calculate", "tests/test_calc.py")])

    result = kg._execute_query(
        """
        MATCH (tc:TestCase {name: 'test_calculate'})-[:TESTS]->(e:CodeEntity)
        RETURN e.name AS name
        """
    )
    names = []
    while result.has_next():
        names.append(result.get_next()[0])
    assert "calculate" in names


def test_parametrized_detection_from_name(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    builder.build(
        [
            _test_chunk("test_thing[case1]", "tests/test_x.py"),
            _test_chunk("test_thing", "tests/test_x.py"),
        ]
    )
    result = kg._execute_query(
        "MATCH (tc:TestCase) RETURN tc.name AS name, tc.is_parametrized AS p"
    )
    flags = {}
    while result.has_next():
        row = result.get_next()
        flags[row[0]] = row[1]
    assert flags.get("test_thing[case1]") is True


def test_stats_include_test_node_counts(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    builder.build(
        [
            _test_chunk("test_a", "tests/test_x.py"),
            _test_chunk("test_b", "tests/test_y.py"),
        ]
    )

    stats = kg.get_stats_sync()
    assert stats.get("test_suites") == 2
    assert stats.get("test_cases") == 2


def test_empty_chunks_returns_zero_stats(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    stats = builder.build([])
    assert stats == {
        "test_suites": 0,
        "test_cases": 0,
        "tests_edges": 0,
        "belongs_to_edges": 0,
        "uses_fixture_edges": 0,
    }


def test_non_test_chunks_ignored(kg: KnowledgeGraph, tmp_path: Path):
    builder = TestKGBuilder(kg, tmp_path)
    stats = builder.build(
        [
            _prod_chunk("foo", "src/mod.py"),
            _prod_chunk("Bar", "src/other.py", kind="class"),
        ]
    )
    assert stats["test_suites"] == 0
    assert stats["test_cases"] == 0
