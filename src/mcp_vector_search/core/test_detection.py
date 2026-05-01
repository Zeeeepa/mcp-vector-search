"""Utilities for detecting test code chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Path patterns that indicate test code
TEST_PATH_PATTERNS = [
    r"(^|/)tests?/",
    r"(^|/)__tests__/",
    r"(^|/)spec/",
    r"(^|/)test_[^/]+\.(py|js|ts|rb|go|java|cs|cpp|rs)$",
    r"(^|/)[^/]+_test\.(py|js|ts|rb|go|java|cs|cpp|rs)$",
    r"(^|/)[^/]+\.spec\.(js|ts|jsx|tsx)$",
    r"(^|/)[^/]+_spec\.(rb|py)$",
    r"(^|/)conftest\.py$",
]

_TEST_PATH_RE = re.compile("|".join(TEST_PATH_PATTERNS), re.IGNORECASE)

# Name patterns for test functions/classes
TEST_NAME_PATTERNS = re.compile(
    r"^(test_|Test[A-Z]|it_|describe_|should_)", re.IGNORECASE
)


def is_test_path(file_path: str | Path) -> bool:
    """Return True if the file path looks like a test file."""
    return bool(_TEST_PATH_RE.search(str(file_path).replace("\\", "/")))


def is_test_chunk(chunk: dict[str, Any]) -> bool:
    """Return True if a chunk dict represents test code."""
    file_path = chunk.get("file_path", "")
    if is_test_path(file_path):
        return True
    name = chunk.get("name", "") or chunk.get("function_name", "") or ""
    return bool(TEST_NAME_PATTERNS.match(name))


def build_test_where_clause() -> str:
    """Build a LanceDB SQL WHERE fragment that matches test file paths.

    LanceDB supports a subset of DuckDB SQL; ``regexp_match`` is not always
    available, so we use a disjunction of ``LIKE`` patterns covering the
    common test-file conventions across multiple languages.
    """
    patterns = [
        "file_path LIKE '%/tests/%'",
        "file_path LIKE '%/test/%'",
        "file_path LIKE '%/__tests__/%'",
        "file_path LIKE '%/spec/%'",
        "file_path LIKE 'test_%'",
        "file_path LIKE '%/test_%'",
        "file_path LIKE '%_test.py'",
        "file_path LIKE '%_test.js'",
        "file_path LIKE '%_test.ts'",
        "file_path LIKE '%_test.rb'",
        "file_path LIKE '%_test.go'",
        "file_path LIKE '%_test.java'",
        "file_path LIKE '%.spec.js'",
        "file_path LIKE '%.spec.ts'",
        "file_path LIKE '%.spec.jsx'",
        "file_path LIKE '%.spec.tsx'",
        "file_path LIKE '%_spec.rb'",
        "file_path LIKE '%/conftest.py'",
    ]
    return "(" + " OR ".join(patterns) + ")"
