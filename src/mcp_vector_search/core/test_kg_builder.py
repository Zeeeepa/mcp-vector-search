"""Build TestSuite/TestCase nodes and test relationships in the knowledge graph.

Issue #156: Adds a dedicated builder that:

1. Identifies test chunks via :mod:`test_detection` heuristics.
2. Creates one ``TestSuite`` node per test file (with detected framework).
3. Creates one ``TestCase`` node per test function or test class chunk.
4. Inserts ``BELONGS_TO_SUITE`` edges (TestCase -> TestSuite).
5. Inserts ``TESTS`` edges by name-convention matching against ``CodeEntity``
   nodes already present in the graph (e.g. ``test_foo`` -> ``foo``).
6. Inserts ``USES_FIXTURE`` edges for fixture parameters that match
   existing function entities (best-effort).

Failures during insertion are logged but never propagate, so test KG
construction cannot break the main build pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from .test_detection import is_test_chunk, is_test_path

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph


# Special pytest parameter names that are NOT fixtures
_PYTEST_RESERVED_PARAMS = frozenset(
    {
        "self",
        "cls",
        "request",
        "monkeypatch",
        "tmp_path",
        "tmp_path_factory",
        "tmpdir",
        "tmpdir_factory",
        "capsys",
        "capfd",
        "caplog",
        "recwarn",
        "pytestconfig",
        "record_property",
        "record_xml_attribute",
        "record_testsuite_property",
        "doctest_namespace",
    }
)

# Regex helpers
_PARAM_NAMES_RE = re.compile(r"\bdef\s+\w+\s*\(([^)]*)\)", re.MULTILINE)
_PARAMETRIZE_RE = re.compile(r"@pytest\.mark\.parametrize", re.MULTILINE)
_PYTEST_IMPORT_RE = re.compile(r"^\s*(?:import\s+pytest|from\s+pytest\b)", re.MULTILINE)
_UNITTEST_RE = re.compile(
    r"(?:^\s*import\s+unittest)|(?:from\s+unittest\b)|(?:unittest\.TestCase)",
    re.MULTILINE,
)
_JEST_RE = re.compile(r"\b(?:describe|it|test|expect)\s*\(", re.MULTILINE)
_MOCHA_RE = re.compile(r"^\s*(?:require|import).*\bmocha\b", re.MULTILINE)
_RSPEC_RE = re.compile(r"\bRSpec\.describe\b|\bdescribe\s+['\"]", re.MULTILINE)


def _detect_framework(content: str, file_path: str) -> str:
    """Best-effort framework detection from a file's text content."""
    if not content:
        # Fallback by extension
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            return "pytest"
        if suffix in {".js", ".jsx", ".ts", ".tsx"}:
            return "jest"
        if suffix == ".rb":
            return "rspec"
        return "unknown"

    if _PYTEST_IMPORT_RE.search(content) or "pytest" in content:
        return "pytest"
    if _UNITTEST_RE.search(content):
        return "unittest"
    if _MOCHA_RE.search(content):
        return "mocha"
    if _JEST_RE.search(content):
        return "jest"
    if _RSPEC_RE.search(content):
        return "rspec"

    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return "pytest"
    return "unknown"


def _extract_param_names(content: str) -> list[str]:
    """Extract parameter names from the first ``def`` in the chunk content.

    Returns a list of parameter names with default values stripped.
    """
    if not content:
        return []
    m = _PARAM_NAMES_RE.search(content)
    if not m:
        return []
    raw = m.group(1)
    params: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        # Strip type annotations & defaults
        piece = piece.split(":", 1)[0].strip()
        piece = piece.split("=", 1)[0].strip()
        # Skip *args / **kwargs
        if piece.startswith("*"):
            continue
        if piece:
            params.append(piece)
    return params


def _is_parametrized(name: str, content: str) -> bool:
    """Return True if a test is parametrized (pytest)."""
    if "[" in name and "]" in name:
        return True
    if content and _PARAMETRIZE_RE.search(content):
        return True
    return False


def _strip_test_prefix(name: str) -> str | None:
    """Map a test name to the production symbol it likely tests.

    Examples:
        test_foo        -> foo
        TestFoo         -> Foo
        test_FooBar     -> FooBar
        it_does_thing   -> does_thing
    """
    if not name:
        return None
    if name.startswith("test_"):
        return name[len("test_") :] or None
    if name.startswith("Test") and len(name) > 4 and name[4].isupper():
        return name[4:]
    if name.startswith("it_"):
        return name[len("it_") :] or None
    return None


def _chunk_get(chunk: Any, key: str, default: Any = None) -> Any:
    """Read a value from a chunk that may be a dict or a dataclass."""
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _chunk_name(chunk: Any) -> str:
    """Best-effort extraction of a chunk's display name."""
    return (
        _chunk_get(chunk, "name")
        or _chunk_get(chunk, "function_name")
        or _chunk_get(chunk, "class_name")
        or ""
    )


def _file_path_str(chunk: Any) -> str:
    fp = _chunk_get(chunk, "file_path", "")
    return str(fp) if fp is not None else ""


class TestKGBuilder:
    """Builds TestSuite/TestCase nodes and test edges in the knowledge graph."""

    # Tell pytest this is not a test class despite the "Test" prefix
    __test__ = False

    def __init__(self, kg: KnowledgeGraph, project_root: Path):
        self.kg = kg
        self.project_root = Path(project_root)

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def build(self, chunks: list[Any]) -> dict[str, int]:
        """Process chunks, extract test nodes and edges, insert into the KG.

        Args:
            chunks: List of CodeChunk dataclasses or chunk dicts.

        Returns:
            Stats dict with keys: ``test_suites``, ``test_cases``,
            ``tests_edges``, ``belongs_to_edges``, ``uses_fixture_edges``.
        """
        stats = {
            "test_suites": 0,
            "test_cases": 0,
            "tests_edges": 0,
            "belongs_to_edges": 0,
            "uses_fixture_edges": 0,
        }

        try:
            test_chunks, test_files = self._partition_test_chunks(chunks)
        except Exception as e:
            logger.warning(f"TestKGBuilder: failed to partition chunks: {e}")
            return stats

        if not test_chunks and not test_files:
            return stats

        # Build TestSuite nodes (one per test file)
        suites = self._build_test_suites(test_files, test_chunks)
        # Build TestCase nodes (one per test function/class chunk)
        cases = self._build_test_cases(test_chunks)

        # Insert nodes
        stats["test_suites"] = self._insert_test_suites(suites)
        stats["test_cases"] = self._insert_test_cases(cases)

        # Insert BELONGS_TO_SUITE edges
        stats["belongs_to_edges"] = self._insert_belongs_to_suite(cases)

        # Insert TESTS edges via name-convention matching
        stats["tests_edges"] = self._insert_tests_edges(cases)

        # Insert USES_FIXTURE edges (best-effort)
        stats["uses_fixture_edges"] = self._insert_uses_fixture(cases)

        return stats

    # ------------------------------------------------------------------ #
    # Partition / classification                                         #
    # ------------------------------------------------------------------ #
    def _partition_test_chunks(
        self, chunks: list[Any]
    ) -> tuple[list[Any], dict[str, list[Any]]]:
        """Split chunks into (test_chunks, files_with_any_test_chunk).

        ``files_with_any_test_chunk`` maps file_path -> list of all chunks
        belonging to that file (used to build the per-suite framework hint).
        """
        test_chunks: list[Any] = []
        per_file: dict[str, list[Any]] = {}

        for c in chunks:
            fp = _file_path_str(c)
            if not fp:
                continue
            chunk_dict = (
                c
                if isinstance(c, dict)
                else {
                    "file_path": fp,
                    "name": _chunk_name(c),
                    "function_name": _chunk_get(c, "function_name"),
                    "class_name": _chunk_get(c, "class_name"),
                    "chunk_type": _chunk_get(c, "chunk_type"),
                    "content": _chunk_get(c, "content", ""),
                    "start_line": _chunk_get(c, "start_line"),
                    "end_line": _chunk_get(c, "end_line"),
                    "chunk_id": _chunk_get(c, "chunk_id") or _chunk_get(c, "id"),
                }
            )
            is_test_file = is_test_path(fp)
            if is_test_file:
                per_file.setdefault(fp, []).append(chunk_dict)
            if is_test_chunk(chunk_dict):
                test_chunks.append(chunk_dict)

        return test_chunks, per_file

    # ------------------------------------------------------------------ #
    # Build node payloads                                                #
    # ------------------------------------------------------------------ #
    def _build_test_suites(
        self,
        files: dict[str, list[dict[str, Any]]],
        test_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """One TestSuite per test file."""
        # Count test cases per file
        cases_per_file: dict[str, int] = {}
        for tc in test_chunks:
            cases_per_file[tc["file_path"]] = cases_per_file.get(tc["file_path"], 0) + 1

        suites: list[dict[str, Any]] = []
        for fp, chunks_in_file in files.items():
            # Aggregate content snippets for framework detection
            content_blob = "\n".join(
                str(c.get("content") or "")[:2000] for c in chunks_in_file[:10]
            )
            framework = _detect_framework(content_blob, fp)
            suites.append(
                {
                    "id": f"testsuite::{fp}",
                    "name": Path(fp).stem,
                    "file_path": fp,
                    "framework": framework,
                    "test_count": cases_per_file.get(fp, 0),
                }
            )
        return suites

    def _build_test_cases(
        self, test_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        for c in test_chunks:
            name = c.get("name") or c.get("function_name") or c.get("class_name") or ""
            if not name:
                continue
            fp = c["file_path"]
            content = str(c.get("content") or "")
            param_names = _extract_param_names(content)
            fixture_deps = [p for p in param_names if p not in _PYTEST_RESERVED_PARAMS]
            cases.append(
                {
                    "id": f"testcase::{fp}::{name}",
                    "name": name,
                    "file_path": fp,
                    "line_start": int(c.get("start_line") or 0),
                    "line_end": int(c.get("end_line") or 0),
                    "is_parametrized": _is_parametrized(name, content),
                    "fixture_deps": fixture_deps,
                    # Internal-only: not a column, drop before insert
                    "_suite_id": f"testsuite::{fp}",
                }
            )
        return cases

    # ------------------------------------------------------------------ #
    # Inserts                                                            #
    # ------------------------------------------------------------------ #
    def _insert_test_suites(self, suites: list[dict[str, Any]]) -> int:
        if not suites:
            return 0
        inserted = 0
        for s in suites:
            try:
                self.kg._execute_query(
                    """
                    MERGE (ts:TestSuite {id: $id})
                    ON CREATE SET ts.name = $name,
                                  ts.file_path = $file_path,
                                  ts.framework = $framework,
                                  ts.test_count = $test_count
                    ON MATCH  SET ts.name = $name,
                                  ts.file_path = $file_path,
                                  ts.framework = $framework,
                                  ts.test_count = $test_count
                    """,
                    s,
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert TestSuite {s.get('id')}: {e}")
        return inserted

    def _insert_test_cases(self, cases: list[dict[str, Any]]) -> int:
        if not cases:
            return 0
        inserted = 0
        for c in cases:
            params = {
                "id": c["id"],
                "name": c["name"],
                "file_path": c["file_path"],
                "line_start": c["line_start"],
                "line_end": c["line_end"],
                "is_parametrized": c["is_parametrized"],
                "fixture_deps": c["fixture_deps"],
            }
            try:
                self.kg._execute_query(
                    """
                    MERGE (tc:TestCase {id: $id})
                    ON CREATE SET tc.name = $name,
                                  tc.file_path = $file_path,
                                  tc.line_start = $line_start,
                                  tc.line_end = $line_end,
                                  tc.is_parametrized = $is_parametrized,
                                  tc.fixture_deps = $fixture_deps
                    ON MATCH  SET tc.name = $name,
                                  tc.file_path = $file_path,
                                  tc.line_start = $line_start,
                                  tc.line_end = $line_end,
                                  tc.is_parametrized = $is_parametrized,
                                  tc.fixture_deps = $fixture_deps
                    """,
                    params,
                )
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert TestCase {c.get('id')}: {e}")
        return inserted

    def _insert_belongs_to_suite(self, cases: list[dict[str, Any]]) -> int:
        if not cases:
            return 0
        inserted = 0
        for c in cases:
            try:
                self.kg._execute_query(
                    """
                    MATCH (tc:TestCase {id: $tc_id}),
                          (ts:TestSuite {id: $ts_id})
                    MERGE (tc)-[:BELONGS_TO_SUITE]->(ts)
                    """,
                    {"tc_id": c["id"], "ts_id": c["_suite_id"]},
                )
                inserted += 1
            except Exception as e:
                logger.warning(
                    f"Failed to insert BELONGS_TO_SUITE for {c.get('id')}: {e}"
                )
        return inserted

    def _insert_tests_edges(self, cases: list[dict[str, Any]]) -> int:
        """Match test cases to production code by naming convention.

        For each TestCase, derive a candidate production symbol name and
        attempt to find any CodeEntity with that name. If found, create
        a TESTS edge.
        """
        if not cases:
            return 0
        inserted = 0
        for c in cases:
            target_name = _strip_test_prefix(c["name"])
            if not target_name:
                continue
            try:
                # Find matching CodeEntity (function/class/method) by name
                result = self.kg._execute_query(
                    """
                    MATCH (e:CodeEntity {name: $name})
                    RETURN e.id AS id
                    LIMIT 5
                    """,
                    {"name": target_name},
                )
                target_ids: list[str] = []
                while result.has_next():
                    target_ids.append(result.get_next()[0])

                for tid in target_ids:
                    try:
                        self.kg._execute_query(
                            """
                            MATCH (tc:TestCase {id: $tc_id}),
                                  (e:CodeEntity {id: $e_id})
                            MERGE (tc)-[:TESTS]->(e)
                            """,
                            {"tc_id": c["id"], "e_id": tid},
                        )
                        inserted += 1
                    except Exception as e:
                        logger.debug(
                            f"TESTS edge insert failed for {c['id']}->{tid}: {e}"
                        )
            except Exception as e:
                logger.debug(f"TESTS lookup failed for {c.get('id')}: {e}")
        return inserted

    def _insert_uses_fixture(self, cases: list[dict[str, Any]]) -> int:
        """Best-effort USES_FIXTURE edges.

        For each fixture dep name, look for a CodeEntity with matching
        name and ``entity_type IN ('function', 'method')``.
        """
        if not cases:
            return 0
        inserted = 0
        for c in cases:
            for fixture_name in c.get("fixture_deps", []) or []:
                try:
                    result = self.kg._execute_query(
                        """
                        MATCH (e:CodeEntity)
                        WHERE e.name = $name
                          AND (e.entity_type = 'function' OR e.entity_type = 'method')
                        RETURN e.id AS id
                        LIMIT 1
                        """,
                        {"name": fixture_name},
                    )
                    if not result.has_next():
                        continue
                    fixture_id = result.get_next()[0]
                    self.kg._execute_query(
                        """
                        MATCH (tc:TestCase {id: $tc_id}),
                              (e:CodeEntity {id: $e_id})
                        MERGE (tc)-[:USES_FIXTURE]->(e)
                        """,
                        {"tc_id": c["id"], "e_id": fixture_id},
                    )
                    inserted += 1
                except Exception as e:
                    logger.debug(
                        f"USES_FIXTURE insert failed for "
                        f"{c.get('id')} fixture={fixture_name}: {e}"
                    )
        return inserted
