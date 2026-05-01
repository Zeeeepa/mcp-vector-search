"""Test code analysis: coverage gaps, anti-patterns, fixture mapping.

Provides static analysis of test code (pytest-oriented) to surface common
quality issues: missing tests for production symbols, anti-patterns inside
tests (no assertion, empty body, excessive mocking, test calls test), and a
fixture-to-consumer map.

Production-grade analyzer powering the ``analyze_tests`` MCP tool and CLI
command.  All file traversal is bounded by ``path``; failures parsing any
single file are logged and skipped rather than aborting the run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from .test_detection import is_test_path

# File extensions we statically analyze.  Test detection covers more
# languages, but the AST-based analyzer here is Python-only.
_PY_EXT = ".py"

# Names of mocking constructs we count for the excessive_mocking heuristic.
_MOCK_NAMES = {"patch", "Mock", "MagicMock", "AsyncMock", "PropertyMock"}

# Default per-file mock-call ceiling before we flag the test.
_DEFAULT_MOCK_THRESHOLD = 5

# Default directories to skip during file walks.
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    ".mcp-vector-search",
}


@dataclass
class CoverageGap:
    """A public production symbol that no test references."""

    symbol: str
    file: str
    line: int
    kind: str  # "function" | "class" | "method"


@dataclass
class AntiPattern:
    """An anti-pattern detected inside a test function."""

    type: str  # "no_assertion" | "test_calls_test" | "empty_body" | "excessive_mocking"
    test: str
    file: str
    line: int
    detail: str = ""


@dataclass
class FixtureUsage:
    """A pytest fixture and the test functions that consume it."""

    name: str
    consumers: list[str] = field(default_factory=list)


@dataclass
class TestAnalysisResult:
    """Container returned by :class:`TestAnalyzer.analyze`."""

    summary: dict[str, Any]
    coverage_gaps: list[CoverageGap]
    anti_patterns: list[AntiPattern]
    fixture_map: dict[str, list[str]]


class TestAnalyzer:
    """Static analyzer for test quality and coverage gaps.

    The analyzer is intentionally side-effect free: callers pass a path,
    receive a :class:`TestAnalysisResult`, and can serialize as needed.
    """

    def __init__(self, mock_threshold: int = _DEFAULT_MOCK_THRESHOLD) -> None:
        self.mock_threshold = mock_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        path: Path,
        include_coverage_gaps: bool = True,
        include_pattern_analysis: bool = True,
        include_fixture_map: bool = False,
    ) -> TestAnalysisResult:
        """Analyze ``path`` for test quality concerns.

        Args:
            path: File or directory to analyze.
            include_coverage_gaps: Compute coverage gaps for production
                symbols (public top-level functions / classes).
            include_pattern_analysis: Detect anti-patterns inside tests.
            include_fixture_map: Build a fixture-name → consumers mapping.

        Returns:
            Populated :class:`TestAnalysisResult`.
        """
        test_files, prod_files = self._partition_files(path)

        # Parse all relevant files once.  Each parsed file is keyed by its
        # absolute path; values are AST modules or ``None`` on parse error.
        parsed: dict[Path, ast.Module | None] = {}
        for fp in test_files + prod_files:
            parsed[fp] = self._safe_parse(fp)

        # Collect public production symbols and per-file imported names.
        prod_symbols: list[CoverageGap] = []
        for fp in prod_files:
            tree = parsed.get(fp)
            if tree is None:
                continue
            prod_symbols.extend(self._collect_public_symbols(fp, tree))

        # Collect names referenced anywhere across test files (imports,
        # attribute access, calls, plain Name nodes, and test_<name>
        # heuristics).
        referenced_names: set[str] = set()
        imported_modules: set[str] = set()
        anti_patterns: list[AntiPattern] = []
        fixtures: dict[str, list[str]] = {}
        test_function_count = 0

        for fp in test_files:
            tree = parsed.get(fp)
            if tree is None:
                continue
            file_refs, file_imports = self._collect_test_references(tree)
            referenced_names |= file_refs
            imported_modules |= file_imports

            test_funcs = self._find_test_functions(tree)
            test_function_count += len(test_funcs)

            if include_pattern_analysis:
                anti_patterns.extend(self._detect_anti_patterns(fp, tree, test_funcs))

            if include_fixture_map:
                self._update_fixture_map(tree, test_funcs, fixtures)

        # Compute coverage gaps: a public symbol is "covered" when its name
        # is referenced in a test file OR its module is imported by a test.
        coverage_gaps: list[CoverageGap] = []
        if include_coverage_gaps:
            covered_modules = self._covered_module_paths(
                imported_modules, prod_files, path
            )
            for sym in prod_symbols:
                if sym.symbol in referenced_names:
                    continue
                if Path(sym.file) in covered_modules:
                    # The whole module is imported; assume coverage for now.
                    # (Heuristic — false negatives acceptable here.)
                    continue
                coverage_gaps.append(sym)

        summary = {
            "test_files": len(test_files),
            "production_files": len(prod_files),
            "test_functions": test_function_count,
            "public_symbols": len(prod_symbols),
            "coverage_gaps": len(coverage_gaps),
            "anti_patterns": len(anti_patterns),
            "fixtures": len(fixtures),
        }

        return TestAnalysisResult(
            summary=summary,
            coverage_gaps=coverage_gaps,
            anti_patterns=anti_patterns,
            fixture_map=fixtures,
        )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def _partition_files(self, path: Path) -> tuple[list[Path], list[Path]]:
        """Split discovered Python files into (tests, production)."""
        if path.is_file():
            files = [path] if path.suffix == _PY_EXT else []
        else:
            files = []
            for fp in path.rglob(f"*{_PY_EXT}"):
                if any(part in _SKIP_DIRS for part in fp.parts):
                    continue
                files.append(fp)

        tests: list[Path] = []
        prod: list[Path] = []
        for fp in files:
            if is_test_path(fp):
                tests.append(fp)
            else:
                prod.append(fp)
        return tests, prod

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_parse(fp: Path) -> ast.Module | None:
        try:
            source = fp.read_text(encoding="utf-8", errors="replace")
            return ast.parse(source, filename=str(fp))
        except SyntaxError as e:
            logger.warning(f"Skipping {fp} (SyntaxError: {e})")
            return None
        except OSError as e:
            logger.warning(f"Skipping {fp} (read error: {e})")
            return None

    # ------------------------------------------------------------------
    # Production-side collection
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_public_symbols(fp: Path, tree: ast.Module) -> list[CoverageGap]:
        """Collect top-level public functions and classes from a module."""
        out: list[CoverageGap] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                out.append(
                    CoverageGap(
                        symbol=node.name,
                        file=str(fp),
                        line=node.lineno,
                        kind="function",
                    )
                )
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                out.append(
                    CoverageGap(
                        symbol=node.name,
                        file=str(fp),
                        line=node.lineno,
                        kind="class",
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Test-side collection
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_test_references(
        tree: ast.Module,
    ) -> tuple[set[str], set[str]]:
        """Collect names and imported modules referenced anywhere in tree."""
        names: set[str] = set()
        modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # Strip ``test_`` / ``Test`` prefix for fuzzy matching.
                names.add(node.id)
                if node.id.startswith("test_"):
                    names.add(node.id[len("test_") :])
                if node.id.startswith("Test") and len(node.id) > 4:
                    names.add(node.id[len("Test") :])
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    names.add(node.name[len("test_") :])
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("Test") and len(node.name) > 4:
                    names.add(node.name[len("Test") :])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name)
                    names.add(alias.asname or alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module)
                for alias in node.names:
                    names.add(alias.asname or alias.name)

        return names, modules

    @staticmethod
    def _find_test_functions(
        tree: ast.Module,
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Return every test_* function (top-level or inside a Test class)."""
        out: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    out.append(node)
        return out

    # ------------------------------------------------------------------
    # Anti-pattern detection
    # ------------------------------------------------------------------
    def _detect_anti_patterns(
        self,
        fp: Path,
        tree: ast.Module,
        test_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> list[AntiPattern]:
        out: list[AntiPattern] = []

        # Names of all test functions in this file (for "test calls test").
        test_names = {fn.name for fn in test_funcs}

        for fn in test_funcs:
            body = fn.body

            # empty_body: only a docstring or a single ``pass``.
            if self._is_empty_body(body):
                out.append(
                    AntiPattern(
                        type="empty_body",
                        test=fn.name,
                        file=str(fp),
                        line=fn.lineno,
                        detail="Test body is empty (pass / docstring only)",
                    )
                )
                # Empty bodies will trivially have no assertion, so skip
                # the no_assertion check to avoid double-reporting.
                continue

            if not self._has_assertion(fn):
                out.append(
                    AntiPattern(
                        type="no_assertion",
                        test=fn.name,
                        file=str(fp),
                        line=fn.lineno,
                        detail="Test contains no assert / pytest.raises / self.assert*",
                    )
                )

            mock_count = self._count_mocks(fn)
            if mock_count > self.mock_threshold:
                out.append(
                    AntiPattern(
                        type="excessive_mocking",
                        test=fn.name,
                        file=str(fp),
                        line=fn.lineno,
                        detail=f"{mock_count} mock-related calls (threshold: {self.mock_threshold})",
                    )
                )

            for callee in self._direct_test_callees(fn, test_names):
                out.append(
                    AntiPattern(
                        type="test_calls_test",
                        test=fn.name,
                        file=str(fp),
                        line=fn.lineno,
                        detail=f"Calls another test function: {callee}",
                    )
                )

        return out

    @staticmethod
    def _is_empty_body(body: list[ast.stmt]) -> bool:
        if not body:
            return True
        # Strip a single leading docstring if present.
        stripped = body[:]
        if (
            isinstance(stripped[0], ast.Expr)
            and isinstance(stripped[0].value, ast.Constant)
            and isinstance(stripped[0].value.value, str)
        ):
            stripped = stripped[1:]
        if not stripped:
            return True
        return len(stripped) == 1 and isinstance(stripped[0], ast.Pass)

    @staticmethod
    def _has_assertion(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                return True
            if isinstance(node, ast.Call):
                func = node.func
                # self.assert*, cls.assert*, anything.assert*
                if isinstance(func, ast.Attribute) and func.attr.startswith("assert"):
                    return True
                # pytest.raises(...) / pytest.warns(...) / pytest.fail(...)
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "pytest" and func.attr in {
                        "raises",
                        "warns",
                        "fail",
                        "deprecated_call",
                    }:
                        return True
            if isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
                for item in node.items:
                    expr = item.context_expr
                    if (
                        isinstance(expr, ast.Call)
                        and isinstance(expr.func, ast.Attribute)
                        and isinstance(expr.func.value, ast.Name)
                        and expr.func.value.id == "pytest"
                        and expr.func.attr in {"raises", "warns", "deprecated_call"}
                    ):
                        return True
        return False

    @staticmethod
    def _count_mocks(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        count = 0
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _MOCK_NAMES:
                    count += 1
                elif isinstance(func, ast.Attribute) and func.attr in _MOCK_NAMES:
                    count += 1
        # Also count @patch decorators applied to the test itself.
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call):
                func = dec.func
            else:
                func = dec
            if isinstance(func, ast.Name) and func.id in _MOCK_NAMES:
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr in _MOCK_NAMES:
                count += 1
        return count

    @staticmethod
    def _direct_test_callees(
        fn: ast.FunctionDef | ast.AsyncFunctionDef, test_names: set[str]
    ) -> list[str]:
        """Return names of test_* functions called directly from ``fn``."""
        out: list[str] = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Name)
                    and func.id in test_names
                    and func.id != fn.name
                ):
                    out.append(func.id)
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr in test_names
                    and func.attr != fn.name
                ):
                    out.append(func.attr)
        return out

    # ------------------------------------------------------------------
    # Fixture map
    # ------------------------------------------------------------------
    @staticmethod
    def _is_pytest_fixture_decorator(dec: ast.expr) -> bool:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute):
            return target.attr == "fixture"
        if isinstance(target, ast.Name):
            return target.id == "fixture"
        return False

    def _update_fixture_map(
        self,
        tree: ast.Module,
        test_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef],
        fixtures: dict[str, list[str]],
    ) -> None:
        # Discover fixture definitions in this file.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if self._is_pytest_fixture_decorator(dec):
                        fixtures.setdefault(node.name, [])
                        break

        # Map each known fixture to test functions that consume it.
        if not fixtures:
            return
        for fn in test_funcs:
            for arg in fn.args.args:
                if arg.arg in fixtures and fn.name not in fixtures[arg.arg]:
                    fixtures[arg.arg].append(fn.name)

    # ------------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _covered_module_paths(
        imported_modules: set[str],
        prod_files: list[Path],
        root: Path,
    ) -> set[Path]:
        """Resolve imported dotted module names back to production file paths."""
        covered: set[Path] = set()
        if not imported_modules:
            return covered

        # Build a quick index of suffix paths (a/b/c.py) → absolute file.
        for fp in prod_files:
            try:
                rel = fp.relative_to(root) if root.is_dir() else fp
            except ValueError:
                rel = fp
            module_dotted = ".".join(rel.with_suffix("").parts)
            # Match if any imported module endswith the module's dotted name
            # OR the module's tail matches an import.
            for imp in imported_modules:
                if module_dotted == imp or module_dotted.endswith("." + imp):
                    covered.add(fp)
                    break
                # Tail match: e.g., import x.y.foo where production is foo.py
                if rel.stem == imp.split(".")[-1]:
                    covered.add(fp)
                    break
        return covered
