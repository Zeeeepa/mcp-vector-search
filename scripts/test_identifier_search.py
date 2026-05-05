#!/usr/bin/env python3
"""Sanity check for is_identifier_query().

Runs a small classification benchmark over a hand-curated set of queries
to demonstrate that the identifier auto-detector correctly distinguishes
SDK / package / code-identifier queries (which should fall back to BM25)
from natural-language queries (which should keep the default 70/30
hybrid weighting).

Usage:
    uv run python scripts/test_identifier_search.py

Exits 0 on full success, 1 if any query is misclassified.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_vector_search.core.query_processor import (  # noqa: E402
    is_identifier_query,
)

# (query, expected_is_identifier)
CASES: list[tuple[str, bool]] = [
    # --- Identifier-style queries: should be detected (BM25-friendly) ---
    ("getstream.io", True),
    ("StreamApp", True),
    ("@tanstack/query", True),
    ("react-activity-feed", True),
    ("io.getstream.chat", True),
    ("react-query.dev", True),
    ("getStream", True),
    ("install the stream-chat sdk", True),  # 'sdk' keyword
    ("how to use npm install", True),  # 'npm' keyword
    ("fetch from pypi", True),  # 'pypi' keyword
    # --- Natural language: should NOT be detected ---
    ("how do I authenticate users", False),
    ("websocket reconnection logic", False),
    ("parse the configuration file", False),
    ("error handling in async functions", False),
    ("login flow", False),
    ("write tests for the search engine", False),
]


def main() -> int:
    failures: list[tuple[str, bool, bool]] = []
    print(f"{'Query':<45} {'Expected':<10} {'Got':<10} {'OK?'}")
    print("-" * 75)
    for query, expected in CASES:
        got = is_identifier_query(query)
        ok = got == expected
        marker = "PASS" if ok else "FAIL"
        print(f"{query:<45} {str(expected):<10} {str(got):<10} {marker}")
        if not ok:
            failures.append((query, expected, got))

    print("-" * 75)
    total = len(CASES)
    passed = total - len(failures)
    print(f"{passed}/{total} cases passed")

    if failures:
        print("\nFailures:")
        for query, expected, got in failures:
            print(f"  - {query!r}: expected {expected}, got {got}")
        return 1

    print("\nAll cases passed - identifier auto-detection is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
