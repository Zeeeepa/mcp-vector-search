"""Verify BM25 index path is computed at the .mcp-vector-search/ parent dir,
not inside the lance/ subdirectory.

Regression guard for the bug where _check_bm25_backend() did:
    bm25_path = index_path / "bm25_index.pkl"
when index_path was the lance/ subdir, resulting in BM25 never loading.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main() -> int:
    # Import the engine module under test
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from mcp_vector_search.core.search import SemanticSearchEngine

    # Build a mock database whose persist_directory points at the lance/ subdir
    mcp_dir = Path("/tmp/test/.mcp-vector-search")  # nosec B108 - mock path for unit test
    lance_dir = mcp_dir / "lance"
    expected = mcp_dir / "bm25_index.pkl"
    wrong = lance_dir / "bm25_index.pkl"

    mock_db = SimpleNamespace(persist_directory=lance_dir)

    # Construct minimal engine instance without running __init__ side effects.
    engine = SemanticSearchEngine.__new__(SemanticSearchEngine)
    engine.database = mock_db  # type: ignore[attr-defined]
    engine._bm25_backend = None  # type: ignore[attr-defined]

    captured: dict[str, Path] = {}

    # Patch the lazy-build helper to capture the bm25_path argument it would
    # have been called with — the file doesn't exist so this path triggers.
    def _capture(self, bm25_path: Path) -> bool:  # type: ignore[no-untyped-def]
        captured["bm25_path"] = bm25_path
        return False

    with patch.object(SemanticSearchEngine, "_try_lazy_build_bm25", _capture):
        engine._check_bm25_backend()  # type: ignore[attr-defined]

    actual = captured.get("bm25_path")
    if actual == expected:
        print(f"PASS — bm25_path resolved to {actual}")
        return 0

    print("FAIL — bm25_path mis-resolved")
    print(f"  expected: {expected}")
    print(f"  actual:   {actual}")
    print(f"  wrong-old-path would have been: {wrong}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
