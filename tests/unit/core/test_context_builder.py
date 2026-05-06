"""Unit tests for context_builder.build_contextual_text() and build_embed_text()."""

import json
from pathlib import Path

from mcp_vector_search.core.context_builder import (
    build_contextual_text,
    build_embed_text,
)
from mcp_vector_search.core.models import CodeChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(**kwargs) -> CodeChunk:
    """Create a minimal CodeChunk with sensible defaults."""
    defaults = {
        "content": "def foo(): pass",
        "file_path": Path("src/utils.py"),
        "start_line": 1,
        "end_line": 1,
        "language": "python",
    }
    defaults.update(kwargs)
    return CodeChunk(**defaults)


# ---------------------------------------------------------------------------
# Tests: CodeChunk dataclass input
# ---------------------------------------------------------------------------


class TestBuildContextualTextCodeChunk:
    """Tests with CodeChunk dataclass input."""

    def test_returns_original_content_unchanged_with_no_metadata(self):
        """Content-only chunk (no metadata) returns content unchanged."""
        chunk = CodeChunk(
            content="x = 1",
            file_path=Path(""),  # empty path
            start_line=1,
            end_line=1,
            language="",  # blank language
        )
        result = build_contextual_text(chunk)
        # No metadata parts should be produced — result equals content
        assert result == "x = 1"

    def test_prepends_file_lang_fn_header(self):
        """Function in a file gets a compact header prepended."""
        chunk = _make_chunk(
            content="def add(a, b):\n    return a + b",
            file_path=Path("src/math_utils.py"),
            language="python",
            function_name="add",
        )
        result = build_contextual_text(chunk)
        assert result.startswith("File:")
        assert "Lang: python" in result
        assert "Fn: add" in result
        assert "---" in result
        # Original content appears after separator
        assert "def add(a, b):" in result

    def test_class_context_included(self):
        """Method chunk includes Class: field in header."""
        chunk = _make_chunk(
            content="def compute(self): pass",
            class_name="Calculator",
            function_name="compute",
        )
        result = build_contextual_text(chunk)
        assert "Class: Calculator" in result
        assert "Fn: compute" in result

    def test_docstring_truncated_at_200_chars(self):
        """Long docstrings are truncated to 200 chars."""
        long_doc = "A" * 300
        chunk = _make_chunk(docstring=long_doc)
        result = build_contextual_text(chunk)
        # The Desc field should appear but be truncated
        assert "Desc:" in result
        assert "A" * 201 not in result  # Truncated before 201

    def test_short_docstring_included_verbatim(self):
        """Short docstrings appear verbatim in the header."""
        chunk = _make_chunk(docstring="Compute the sum of two numbers.")
        result = build_contextual_text(chunk)
        assert "Desc: Compute the sum of two numbers." in result

    def test_imports_list_of_dicts_extracts_sources(self):
        """Imports as list[dict] with 'source' key are summarized."""
        chunk = _make_chunk(
            imports=[
                {"source": "os", "statement": "import os"},
                {"source": "re", "statement": "import re"},
            ]
        )
        result = build_contextual_text(chunk)
        assert "Uses: os, re" in result

    def test_imports_capped_at_10_sources(self):
        """More than 10 import sources are capped at 10."""
        imports = [
            {"source": f"mod{i}", "statement": f"import mod{i}"} for i in range(15)
        ]
        chunk = _make_chunk(imports=imports)
        result = build_contextual_text(chunk)
        # mod10 through mod14 should not appear
        assert "mod14" not in result
        assert "mod0" in result

    def test_text_language_skipped(self):
        """Language 'text' is not included in the header."""
        chunk = _make_chunk(language="text")
        result = build_contextual_text(chunk)
        assert "Lang:" not in result

    def test_file_path_shortened_to_two_segments(self):
        """Deep paths are shortened to last two segments."""
        chunk = _make_chunk(
            file_path=Path("a/b/c/d/e/src/utils.py"),
        )
        result = build_contextual_text(chunk)
        assert "File: src/utils.py" in result
        # Should not contain the long prefix
        assert "a/b/c" not in result

    def test_original_content_not_modified(self):
        """The original chunk.content attribute is not mutated."""
        chunk = _make_chunk(
            content="original code",
            function_name="foo",
        )
        _ = build_contextual_text(chunk)
        assert chunk.content == "original code"

    def test_separator_newline_between_header_and_code(self):
        """Header and code are separated by '---\\n'."""
        chunk = _make_chunk(function_name="bar")
        result = build_contextual_text(chunk)
        assert "\n---\n" in result


# ---------------------------------------------------------------------------
# Tests: dict input (two-phase pipeline format)
# ---------------------------------------------------------------------------


class TestBuildContextualTextDict:
    """Tests with plain dict input (as produced by indexer.py pipeline)."""

    def test_basic_dict_with_all_fields(self):
        """Dict with all fields produces enriched text."""
        chunk_dict = {
            "chunk_id": "abc123",
            "file_path": "src/core/engine.py",
            "language": "python",
            "class_name": "Engine",
            "function_name": "run",
            "docstring": "Run the engine.",
            "imports": [
                json.dumps({"source": "asyncio", "statement": "import asyncio"})
            ],
            "content": "def run(self): ...",
        }
        result = build_contextual_text(chunk_dict)
        assert "File:" in result
        assert "Lang: python" in result
        assert "Class: Engine" in result
        assert "Fn: run" in result
        assert "Uses: asyncio" in result
        assert "Desc: Run the engine." in result
        assert "def run(self): ..." in result

    def test_dict_imports_as_json_encoded_strings(self):
        """Imports stored as JSON-encoded strings (indexer.py format) are parsed."""
        chunk_dict = {
            "file_path": "app.py",
            "language": "python",
            "content": "x = 1",
            "imports": [
                json.dumps({"source": "os", "statement": "import os"}),
                json.dumps({"source": "sys", "statement": "import sys"}),
            ],
        }
        result = build_contextual_text(chunk_dict)
        assert "Uses: os, sys" in result

    def test_dict_imports_as_plain_strings(self):
        """Imports as plain module-name strings are included as-is."""
        chunk_dict = {
            "file_path": "app.py",
            "language": "python",
            "content": "pass",
            "imports": ["os", "re", "sys"],
        }
        result = build_contextual_text(chunk_dict)
        assert "Uses: os, re, sys" in result

    def test_dict_missing_optional_fields_graceful(self):
        """Dict missing optional fields (class_name, etc.) does not raise."""
        chunk_dict = {
            "file_path": "minimal.py",
            "content": "pass",
        }
        result = build_contextual_text(chunk_dict)
        # Should at least include the file
        assert "File: minimal.py" in result
        assert "pass" in result

    def test_dict_empty_content_returns_empty_with_header(self):
        """Empty content still produces a header when metadata is present."""
        chunk_dict = {
            "file_path": "empty.py",
            "language": "python",
            "content": "",
        }
        result = build_contextual_text(chunk_dict)
        assert "File: empty.py" in result
        assert "---" in result

    def test_dict_no_metadata_returns_content_only(self):
        """Dict with no meaningful metadata returns content unchanged."""
        chunk_dict = {"content": "raw code"}
        result = build_contextual_text(chunk_dict)
        assert result == "raw code"

    def test_dict_imports_as_single_json_list_string(self):
        """Imports stored as a single JSON-encoded list string (lancedb legacy) are parsed."""
        imports_json = json.dumps(
            [
                {"source": "os", "statement": "import os"},
                {"source": "re", "statement": "import re"},
            ]
        )
        chunk_dict = {
            "file_path": "foo.py",
            "language": "python",
            "content": "pass",
            "imports": imports_json,  # Single JSON string (legacy format)
        }
        result = build_contextual_text(chunk_dict)
        assert "Uses: os, re" in result


# ---------------------------------------------------------------------------
# Tests: Edge cases and token-budget constraints
# ---------------------------------------------------------------------------


class TestContextBuilderEdgeCases:
    """Edge cases and token budget constraints."""

    def test_none_class_name_not_included(self):
        """None class_name does not produce 'Class: None' in header."""
        chunk = _make_chunk(class_name=None)
        result = build_contextual_text(chunk)
        assert "Class:" not in result

    def test_none_function_name_not_included(self):
        """None function_name does not produce 'Fn: None' in header."""
        chunk = _make_chunk(function_name=None)
        result = build_contextual_text(chunk)
        assert "Fn:" not in result

    def test_empty_imports_list_no_uses_field(self):
        """Empty imports list does not produce 'Uses:' in header."""
        chunk = _make_chunk(imports=[])
        result = build_contextual_text(chunk)
        assert "Uses:" not in result

    def test_header_uses_pipe_separator(self):
        """Header parts are separated by ' | ' (compact, not newlines)."""
        chunk = _make_chunk(
            language="python",
            class_name="Foo",
            function_name="bar",
        )
        result = build_contextual_text(chunk)
        header_line = result.split("\n---\n")[0]
        assert " | " in header_line
        # Header should be a single line (no internal newlines)
        assert "\n" not in header_line

    def test_full_contextual_text_format(self):
        """Comprehensive test of the exact output format."""
        chunk = _make_chunk(
            content="return self.value",
            file_path=Path("src/models/user.py"),
            language="python",
            class_name="User",
            function_name="get_value",
            docstring="Return the user value.",
            imports=[{"source": "uuid", "statement": "import uuid"}],
        )
        result = build_contextual_text(chunk)
        expected_header = (
            "File: models/user.py | Lang: python | Class: User | Fn: get_value | "
            "Uses: uuid | Desc: Return the user value."
        )
        assert result == f"{expected_header}\n---\nreturn self.value"


# ---------------------------------------------------------------------------
# Tests: build_embed_text() — bracket-tagged contextual chunking format
# ---------------------------------------------------------------------------


class TestBuildEmbedTextFullContext:
    """Tests for build_embed_text with rich, fully-populated chunks."""

    def test_full_context_all_parts_present(self):
        """Chunk with class, file, imports, docstring, calls, content emits all sections."""
        chunk = _make_chunk(
            content="return self.value + 1",
            file_path=Path("src/mcp_vector_search/core/search.py"),
            class_name="Engine",
            docstring="Compute the next value.",
            imports=[
                {"source": "os"},
                {"source": "sys"},
                {"source": "re"},
                {"source": "json"},
                {"source": "asyncio"},
                {"source": "logging"},
                {"source": "pathlib"},
            ],
            calls=["foo", "bar", "baz", "qux", "quux", "extra1", "extra2", "extra3"],
        )
        result = build_embed_text(chunk)

        assert "[class Engine]" in result
        assert "[module" in result
        assert "[imports:" in result
        assert "Compute the next value." in result
        assert "[calls:" in result
        assert "return self.value + 1" in result

    def test_imports_capped_at_5(self):
        """When >5 imports provided, only top 5 appear in [imports: ...] line."""
        chunk = _make_chunk(
            imports=[
                {"source": "os"},
                {"source": "sys"},
                {"source": "re"},
                {"source": "json"},
                {"source": "asyncio"},
                {"source": "logging"},
                {"source": "pathlib"},
            ],
        )
        result = build_embed_text(chunk)
        # Find the imports line
        imports_line = next(
            line for line in result.split("\n") if line.startswith("[imports:")
        )
        # Should have exactly 5 sources, comma separated
        # Format: "[imports: os, sys, re, json, asyncio]"
        assert "os" in imports_line
        assert "asyncio" in imports_line
        # 6th and 7th must NOT appear in the imports header line
        assert "logging" not in imports_line
        assert "pathlib" not in imports_line
        # Count commas — exactly 4 commas means exactly 5 entries
        assert imports_line.count(",") == 4

    def test_calls_capped_at_5(self):
        """When >5 calls provided, only top 5 appear in [calls: ...] line."""
        chunk = _make_chunk(
            calls=["a", "b", "c", "d", "e", "f", "g", "h"],
        )
        result = build_embed_text(chunk)
        calls_line = next(
            line for line in result.split("\n") if line.startswith("[calls:")
        )
        assert "a" in calls_line
        assert "e" in calls_line
        # 6th, 7th, 8th must not be in the calls line
        assert "f" not in calls_line
        assert "g" not in calls_line
        assert "h" not in calls_line
        assert calls_line.count(",") == 4


class TestBuildEmbedTextOptionalFields:
    """Tests verifying each optional field is omitted when absent."""

    def test_no_class_name_none(self):
        """class_name=None — no '[class ...]' line appears."""
        chunk = _make_chunk(class_name=None)
        result = build_embed_text(chunk)
        assert "[class" not in result

    def test_no_class_name_empty_string(self):
        """class_name='' — no '[class ...]' line appears."""
        chunk = _make_chunk(class_name="")
        result = build_embed_text(chunk)
        assert "[class" not in result

    def test_no_docstring_none(self):
        """docstring=None — docstring missing, content still present."""
        chunk = _make_chunk(content="x = 42", docstring=None)
        result = build_embed_text(chunk)
        # Content still emitted
        assert "x = 42" in result

    def test_no_docstring_empty_string(self):
        """docstring='' — docstring not in output."""
        chunk = _make_chunk(content="x = 42", docstring="")
        result = build_embed_text(chunk)
        assert "x = 42" in result

    def test_no_imports_empty_list(self):
        """imports=[] — no '[imports: ...]' line appears."""
        chunk = _make_chunk(imports=[])
        result = build_embed_text(chunk)
        assert "[imports:" not in result

    def test_no_imports_none(self):
        """imports=None — no '[imports: ...]' line appears."""
        chunk = _make_chunk(imports=None)
        result = build_embed_text(chunk)
        assert "[imports:" not in result

    def test_no_calls_empty_list(self):
        """calls=[] — no '[calls: ...]' line appears."""
        chunk = _make_chunk(calls=[])
        result = build_embed_text(chunk)
        assert "[calls:" not in result

    def test_no_calls_none(self):
        """calls=None — no '[calls: ...]' line appears."""
        chunk = _make_chunk(calls=None)
        result = build_embed_text(chunk)
        assert "[calls:" not in result


class TestBuildEmbedTextModulePath:
    """Tests for module path formatting from file_path."""

    def test_python_module_path_dotted(self):
        """src/mcp_vector_search/core/search.py → dotted module name without .py."""
        chunk = _make_chunk(
            file_path=Path("src/mcp_vector_search/core/search.py"),
        )
        result = build_embed_text(chunk)
        assert "[module src.mcp_vector_search.core.search]" in result

    def test_module_path_strips_py_suffix(self):
        """The .py suffix is stripped from the module label."""
        chunk = _make_chunk(file_path=Path("foo/bar.py"))
        result = build_embed_text(chunk)
        assert "[module foo.bar]" in result
        # No trailing .py inside the bracket
        assert "[module foo.bar.py]" not in result

    def test_non_python_path_passes_through(self):
        """Non-.py paths get slashes converted to dots, no suffix stripping."""
        chunk = _make_chunk(file_path=Path("docs/readme.md"))
        result = build_embed_text(chunk)
        assert "[module docs.readme.md]" in result

    def test_degenerate_paths_skipped(self):
        """Paths like '.', '/', '' do not produce a [module ...] line."""
        # Empty path — handled at chunk level by passing empty string via dict
        chunk_dict = {"file_path": "", "content": "pass"}
        result = build_embed_text(chunk_dict)
        assert "[module" not in result

        chunk_dict = {"file_path": ".", "content": "pass"}
        result = build_embed_text(chunk_dict)
        assert "[module" not in result

        chunk_dict = {"file_path": "/", "content": "pass"}
        result = build_embed_text(chunk_dict)
        assert "[module" not in result

    def test_windows_path_separator_normalized(self):
        """Backslashes in file paths are normalized to dots (Windows compat)."""
        chunk_dict = {
            "file_path": "src\\foo\\bar.py",
            "content": "pass",
        }
        result = build_embed_text(chunk_dict)
        assert "[module src.foo.bar]" in result


class TestBuildEmbedTextOutputStructure:
    """Tests for output structure: ordering, joining, content placement."""

    def test_empty_chunk_returns_only_content(self):
        """Chunk with only content set returns content unchanged."""
        chunk_dict = {"content": "raw code"}
        result = build_embed_text(chunk_dict)
        assert result == "raw code"

    def test_truly_empty_chunk_returns_empty_string(self):
        """Chunk with no fields at all returns empty string."""
        chunk_dict = {}
        result = build_embed_text(chunk_dict)
        # Only content is appended (which is empty), so result is ""
        assert result == ""

    def test_content_always_last(self):
        """Regardless of other fields, chunk.content must be the final line."""
        chunk = _make_chunk(
            content="THE_CONTENT_LINE",
            class_name="MyClass",
            file_path=Path("a/b.py"),
            imports=[{"source": "os"}],
            docstring="A docstring.",
            calls=["foo"],
        )
        result = build_embed_text(chunk)
        lines = result.split("\n")
        # The content might itself contain newlines normally, but here it's single-line
        assert lines[-1] == "THE_CONTENT_LINE"

    def test_multiline_content_preserved_at_end(self):
        """Multi-line content is preserved verbatim at the end."""
        content = "def foo():\n    return 1\n    # comment"
        chunk = _make_chunk(
            content=content,
            class_name="MyClass",
        )
        result = build_embed_text(chunk)
        assert result.endswith(content)

    def test_parts_joined_with_single_newline(self):
        """Output parts are joined with '\\n' — no double newlines between sections."""
        chunk = _make_chunk(
            content="x",
            class_name="C",
            file_path=Path("a.py"),
            imports=[{"source": "os"}],
            docstring="doc",
            calls=["f"],
        )
        result = build_embed_text(chunk)
        # No double newlines between any of the bracket-tagged sections
        assert "\n\n" not in result

    def test_section_ordering(self):
        """Sections appear in the documented order: class, module, imports, docstring, calls, content."""
        chunk = _make_chunk(
            content="CONTENT",
            class_name="ClassName",
            file_path=Path("mod.py"),
            imports=[{"source": "imp_src"}],
            docstring="DOCSTRING",
            calls=["call_name"],
        )
        result = build_embed_text(chunk)
        # Verify ordering by checking indices
        idx_class = result.index("[class ClassName]")
        idx_module = result.index("[module mod]")
        idx_imports = result.index("[imports:")
        idx_doc = result.index("DOCSTRING")
        idx_calls = result.index("[calls:")
        idx_content = result.index("CONTENT")

        assert idx_class < idx_module < idx_imports < idx_doc < idx_calls < idx_content


class TestBuildEmbedTextImportFormats:
    """Tests for the various accepted import shapes."""

    def test_imports_list_of_dicts(self):
        """imports as list[dict] with 'source' key."""
        chunk = _make_chunk(
            imports=[
                {"source": "os"},
                {"source": "sys"},
            ],
        )
        result = build_embed_text(chunk)
        assert "[imports: os, sys]" in result

    def test_imports_list_of_plain_strings(self):
        """imports as list[str] with plain module names."""
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": ["os", "sys", "re"],
        }
        result = build_embed_text(chunk_dict)
        assert "[imports: os, sys, re]" in result

    def test_imports_list_of_json_encoded_strings(self):
        """imports as list[str] where each string is a JSON-encoded dict."""
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": [
                json.dumps({"source": "os", "statement": "import os"}),
                json.dumps({"source": "sys", "statement": "import sys"}),
            ],
        }
        result = build_embed_text(chunk_dict)
        assert "[imports: os, sys]" in result

    def test_imports_as_single_json_list_string(self):
        """imports as a single JSON-encoded list string (legacy lancedb format)."""
        imports_json = json.dumps([{"source": "os"}, {"source": "re"}])
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": imports_json,
        }
        result = build_embed_text(chunk_dict)
        assert "[imports: os, re]" in result

    def test_imports_invalid_json_string_falls_back_to_empty(self):
        """A single string that fails JSON parsing yields no imports section."""
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": "not valid json {",
        }
        result = build_embed_text(chunk_dict)
        assert "[imports:" not in result

    def test_imports_json_string_decodes_to_non_list(self):
        """JSON-encoded string that decodes to a non-list yields no imports."""
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": json.dumps({"not": "a list"}),
        }
        result = build_embed_text(chunk_dict)
        assert "[imports:" not in result

    def test_imports_dict_without_source_key_skipped(self):
        """Import dicts without a 'source' key are silently skipped."""
        chunk = _make_chunk(
            imports=[
                {"statement": "import os"},  # missing 'source'
                {"source": "sys"},
            ],
        )
        result = build_embed_text(chunk)
        # Only sys should appear
        assert "[imports: sys]" in result

    def test_imports_invalid_inner_json_treated_as_plain_string(self):
        """A string starting with '{' but not valid JSON is treated as plain name."""
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": ["{not valid json"],
        }
        # Should not raise; the broken JSON falls through to plain-string branch
        result = build_embed_text(chunk_dict)
        # The literal stripped string ends up being included
        assert "[imports:" in result
        assert "{not valid json" in result

    def test_imports_empty_strings_filtered(self):
        """Empty/whitespace-only strings in imports are filtered out."""
        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": ["", "   ", "os"],
        }
        result = build_embed_text(chunk_dict)
        assert "[imports: os]" in result


class TestBuildEmbedTextDocstring:
    """Tests for docstring handling."""

    def test_docstring_truncated_at_200_chars(self):
        """Docstrings longer than 200 chars are truncated with '...' suffix."""
        long_doc = "A" * 300
        chunk = _make_chunk(docstring=long_doc)
        result = build_embed_text(chunk)
        # 200 'A's + '...'
        assert "A" * 200 + "..." in result
        # Should not contain 201 consecutive 'A's
        assert "A" * 201 not in result

    def test_docstring_short_included_verbatim(self):
        """Short docstrings appear verbatim in the output."""
        chunk = _make_chunk(docstring="Brief description.")
        result = build_embed_text(chunk)
        assert "Brief description." in result

    def test_docstring_whitespace_stripped(self):
        """Leading/trailing whitespace on docstrings is stripped."""
        chunk = _make_chunk(docstring="   trimmed doc   ")
        result = build_embed_text(chunk)
        lines = result.split("\n")
        # Find the docstring line — should be exactly "trimmed doc"
        assert "trimmed doc" in lines

    def test_docstring_whitespace_only_skipped(self):
        """Whitespace-only docstrings produce no output line."""
        chunk = _make_chunk(content="x", docstring="   ")
        result = build_embed_text(chunk)
        # Output should just be the content
        assert result.endswith("x")
        # No empty docstring line appearing as standalone whitespace
        for line in result.split("\n"):
            # Every non-content line should be either bracket-tagged or absent
            if line and line != "x":
                assert line.startswith("[") or line.strip() != ""


class TestBuildEmbedTextDictVsObject:
    """Tests confirming dict and dataclass inputs produce identical output."""

    def test_dict_and_object_produce_same_output(self):
        """build_embed_text accepts both dict and CodeChunk identically."""
        chunk_obj = _make_chunk(
            content="pass",
            file_path=Path("a/b.py"),
            class_name="X",
            docstring="hello",
            imports=[{"source": "os"}],
            calls=["foo"],
        )
        chunk_dict = {
            "content": "pass",
            "file_path": "a/b.py",
            "class_name": "X",
            "docstring": "hello",
            "imports": [{"source": "os"}],
            "calls": ["foo"],
        }
        assert build_embed_text(chunk_obj) == build_embed_text(chunk_dict)


class TestBuildEmbedTextEdgeCases:
    """Robustness / edge-case tests."""

    def test_calls_with_falsy_entries_filtered(self):
        """Falsy entries in calls (None, '') are filtered out."""
        chunk = _make_chunk(calls=["foo", None, "", "bar"])
        result = build_embed_text(chunk)
        calls_line = next(
            line for line in result.split("\n") if line.startswith("[calls:")
        )
        assert "foo" in calls_line
        assert "bar" in calls_line
        # No literal "None" appearing in the calls list
        assert "None" not in calls_line

    def test_calls_non_string_coerced(self):
        """Non-string call entries are coerced via str()."""
        chunk = _make_chunk(calls=[123, "named_call"])
        result = build_embed_text(chunk)
        assert "123" in result
        assert "named_call" in result

    def test_no_metadata_at_all_returns_content(self):
        """No metadata, only content — output is the content itself."""
        chunk_dict = {"content": "just code"}
        result = build_embed_text(chunk_dict)
        assert result == "just code"

    def test_imports_iterable_typeerror_handled(self):
        """A non-iterable imports value is gracefully handled."""

        class WeirdImports:
            """Object that raises TypeError on iteration."""

            def __iter__(self):  # noqa: D401
                raise TypeError("not iterable")

        chunk_dict = {
            "file_path": "a.py",
            "content": "pass",
            "imports": WeirdImports(),
        }
        # Should not raise — the conversion is wrapped in try/except
        result = build_embed_text(chunk_dict)
        assert "[imports:" not in result
