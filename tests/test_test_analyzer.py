"""Tests for the TestAnalyzer (issue #155)."""

from __future__ import annotations

from pathlib import Path

from mcp_vector_search.core.test_analyzer import TestAnalyzer
from mcp_vector_search.core.test_detection import is_test_path


# ---------------------------------------------------------------------------
# is_test_path partitioning
# ---------------------------------------------------------------------------
def test_is_test_path_recognizes_common_layouts() -> None:
    assert is_test_path("tests/test_foo.py")
    assert is_test_path("src/pkg/test_bar.py")
    assert is_test_path("pkg/foo_test.py")
    assert is_test_path("ui/component.spec.ts")
    assert not is_test_path("src/pkg/foo.py")
    assert not is_test_path("src/pkg/__init__.py")


# ---------------------------------------------------------------------------
# Coverage gap detection
# ---------------------------------------------------------------------------
def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_coverage_gap_empty_when_symbol_referenced(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "main.py",
        "def foo():\n    return 1\n",
    )
    _write(
        tmp_path / "tests" / "test_main.py",
        "from src.main import foo\n\ndef test_foo():\n    assert foo() == 1\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    gap_symbols = {g.symbol for g in result.coverage_gaps}
    assert "foo" not in gap_symbols


def test_coverage_gap_detects_untested_symbol(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "main.py",
        "def foo():\n    return 1\n\ndef bar():\n    return 2\n",
    )
    # Test only references foo by name; bar is never imported nor named.
    _write(
        tmp_path / "tests" / "test_main.py",
        "def test_foo():\n    assert True\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    gap_symbols = {g.symbol for g in result.coverage_gaps}
    assert "bar" in gap_symbols


def test_coverage_gap_class_detection(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "models.py",
        "class Untested:\n    pass\n",
    )
    _write(
        tmp_path / "tests" / "test_other.py",
        "def test_nothing():\n    assert 1 == 1\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    syms = {(g.symbol, g.kind) for g in result.coverage_gaps}
    assert ("Untested", "class") in syms


# ---------------------------------------------------------------------------
# Anti-pattern detection
# ---------------------------------------------------------------------------
def test_anti_pattern_no_assertion(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_a.py",
        "def test_no_assert():\n    x = 1 + 1\n    print(x)\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    types = {p.type for p in result.anti_patterns}
    assert "no_assertion" in types


def test_anti_pattern_empty_body(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_empty.py",
        "def test_pass_only():\n"
        "    pass\n\n"
        "def test_doc_only():\n"
        '    """Just a docstring."""\n',
    )

    result = TestAnalyzer().analyze(tmp_path)
    empty = [p for p in result.anti_patterns if p.type == "empty_body"]
    assert len(empty) == 2
    # Empty bodies should NOT also be reported as no_assertion (deduped).
    no_assert = [
        p
        for p in result.anti_patterns
        if p.type == "no_assertion" and p.test in {"test_pass_only", "test_doc_only"}
    ]
    assert no_assert == []


def test_anti_pattern_assertion_via_pytest_raises(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_b.py",
        "import pytest\n\n"
        "def test_raises():\n"
        "    with pytest.raises(ValueError):\n"
        "        raise ValueError('boom')\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    no_assert = [p for p in result.anti_patterns if p.type == "no_assertion"]
    assert no_assert == []


def test_anti_pattern_excessive_mocking(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_mock.py",
        "from unittest.mock import patch, Mock, MagicMock\n\n"
        "def test_too_many_mocks():\n"
        "    a = Mock()\n"
        "    b = Mock()\n"
        "    c = MagicMock()\n"
        "    d = MagicMock()\n"
        "    e = Mock()\n"
        "    f = Mock()\n"
        "    assert a and b and c and d and e and f\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    excess = [p for p in result.anti_patterns if p.type == "excessive_mocking"]
    assert len(excess) == 1


def test_anti_pattern_test_calls_test(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "test_c.py",
        "def test_helper():\n"
        "    assert True\n\n"
        "def test_caller():\n"
        "    test_helper()\n"
        "    assert True\n",
    )

    result = TestAnalyzer().analyze(tmp_path)
    calls = [p for p in result.anti_patterns if p.type == "test_calls_test"]
    assert any(p.test == "test_caller" for p in calls)


# ---------------------------------------------------------------------------
# Fixture map
# ---------------------------------------------------------------------------
def test_fixture_map_detection(tmp_path: Path) -> None:
    _write(
        tmp_path / "tests" / "conftest.py",
        "import pytest\n\n@pytest.fixture\ndef sample_data():\n    return {'k': 'v'}\n",
    )
    _write(
        tmp_path / "tests" / "test_uses_fixture.py",
        "def test_with_fixture(sample_data):\n"
        "    assert sample_data['k'] == 'v'\n\n"
        "def test_without_fixture():\n"
        "    assert True\n",
    )

    result = TestAnalyzer().analyze(tmp_path, include_fixture_map=True)
    assert "sample_data" in result.fixture_map
    assert "test_with_fixture" in result.fixture_map["sample_data"]
    assert "test_without_fixture" not in result.fixture_map["sample_data"]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_analyzer_skips_syntax_errors(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "broken.py", "def broken(:\n    pass\n")
    _write(
        tmp_path / "tests" / "test_ok.py",
        "def test_ok():\n    assert True\n",
    )
    # Should not raise.
    result = TestAnalyzer().analyze(tmp_path)
    assert result.summary["test_files"] == 1


def test_analyzer_summary_counts(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "m.py", "def public():\n    return 1\n")
    _write(
        tmp_path / "tests" / "test_m.py",
        "def test_one():\n    assert True\n\ndef test_two():\n    assert True\n",
    )
    result = TestAnalyzer().analyze(tmp_path)
    assert result.summary["test_functions"] == 2
    assert result.summary["test_files"] == 1
    assert result.summary["production_files"] == 1
