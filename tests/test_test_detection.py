"""Tests for test code detection utilities (issue #154)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_vector_search.core.test_detection import (
    build_test_where_clause,
    is_test_chunk,
    is_test_path,
)


class TestIsTestPath:
    """Tests for is_test_path()."""

    @pytest.mark.parametrize(
        "path",
        [
            "tests/foo.py",
            "src/tests/foo.py",
            "test/foo.py",
            "src/test/foo.py",
            "src/__tests__/Component.tsx",
            "spec/example_spec.rb",
            "test_foo.py",
            "src/test_foo.py",
            "foo_test.py",
            "src/foo_test.go",
            "src/foo_test.java",
            "src/Component.spec.ts",
            "src/Component.spec.tsx",
            "src/Component.spec.js",
            "src/Component.spec.jsx",
            "spec/example_spec.rb",
            "conftest.py",
            "src/conftest.py",
        ],
    )
    def test_test_paths_detected(self, path: str) -> None:
        assert is_test_path(path) is True, f"Expected {path!r} to be a test path"

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "src/utils/helpers.py",
            "lib/database.js",
            "app/components/Button.tsx",
            "README.md",
            "package.json",
            "src/manifest.py",  # 'test' substring not at boundary
            "src/contestant.py",  # 'test' substring not at boundary
        ],
    )
    def test_non_test_paths_not_detected(self, path: str) -> None:
        assert is_test_path(path) is False, f"Expected {path!r} to NOT be a test path"

    def test_accepts_path_object(self) -> None:
        assert is_test_path(Path("tests/foo.py")) is True
        assert is_test_path(Path("src/main.py")) is False

    def test_windows_style_separators(self) -> None:
        assert is_test_path("src\\tests\\foo.py") is True
        assert is_test_path("src\\main.py") is False


class TestIsTestChunk:
    """Tests for is_test_chunk()."""

    def test_test_path_chunk(self) -> None:
        assert is_test_chunk({"file_path": "tests/foo.py", "name": "do_thing"}) is True

    def test_test_name_in_non_test_path(self) -> None:
        # Test function in a non-test file (rare but valid)
        assert (
            is_test_chunk({"file_path": "src/main.py", "name": "test_something"})
            is True
        )

    def test_capitalized_test_class_name(self) -> None:
        assert (
            is_test_chunk({"file_path": "src/main.py", "name": "TestUserService"})
            is True
        )

    def test_non_test_chunk(self) -> None:
        assert (
            is_test_chunk({"file_path": "src/main.py", "name": "process_data"}) is False
        )

    def test_function_name_field_fallback(self) -> None:
        assert (
            is_test_chunk({"file_path": "src/main.py", "function_name": "test_login"})
            is True
        )

    def test_empty_chunk(self) -> None:
        assert is_test_chunk({}) is False


class TestBuildTestWhereClause:
    """Tests for build_test_where_clause()."""

    def test_returns_non_empty_string(self) -> None:
        clause = build_test_where_clause()
        assert isinstance(clause, str)
        assert len(clause) > 0

    def test_uses_or_disjunction(self) -> None:
        clause = build_test_where_clause()
        assert " OR " in clause

    def test_references_file_path(self) -> None:
        clause = build_test_where_clause()
        assert "file_path" in clause

    def test_wrapped_in_parens(self) -> None:
        clause = build_test_where_clause()
        assert clause.startswith("(") and clause.endswith(")")

    def test_includes_common_patterns(self) -> None:
        clause = build_test_where_clause()
        # Spot-check that common test conventions are represented.
        for needle in [
            "/tests/",
            "/__tests__/",
            "/spec/",
            "_test.py",
            ".spec.ts",
            "conftest.py",
        ]:
            assert needle in clause, f"Expected pattern containing {needle!r}"
