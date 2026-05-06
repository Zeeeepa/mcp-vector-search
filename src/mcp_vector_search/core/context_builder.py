"""Context builder for enriched embeddings.

Prepends metadata header (file path, language, class/function context, imports,
docstring) to chunk content before embedding.  The stored ``content`` field in
the database is left completely untouched — only the text sent to the embedding
model is enriched.

Research shows that contextual metadata prepending improves semantic retrieval
quality by 35–49% by giving the model more signal about where and what a chunk
is, rather than treating every snippet as context-free code.

Token budget note
-----------------
MiniLM has a hard 256-token limit.  A compact pipe-separated header like::

    File: src/foo.py | Lang: python | Class: MyClass | Fn: my_method | Uses: os, re | Desc: Does X

is typically 15–25 tokens, leaving 230–240 tokens for the original code — well
within the budget.  We therefore use ``|`` separators (not newlines) for the
header and insert a single ``---\\n`` divider before the code.
"""

from __future__ import annotations

import json as _json
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers (shared by build_contextual_text and build_embed_text)
# ---------------------------------------------------------------------------

# Field names consulted on chunks (both dict-key and attribute access).
_CONTEXTUAL_FIELDS = (
    "file_path",
    "language",
    "class_name",
    "function_name",
    "imports",
    "docstring",
    "content",
)

_EMBED_FIELDS = (
    "class_name",
    "file_path",
    "imports",
    "docstring",
    "calls",
    "content",
)


def _extract_fields(chunk: Any, field_names: tuple[str, ...]) -> dict[str, Any]:
    """Extract requested fields from a chunk (dict or dataclass-like object).

    Missing or falsy fields are normalised to "" or [] depending on the
    field name (``imports``/``calls`` default to ``[]``; everything else
    defaults to ``""``).  ``file_path`` is coerced to ``str`` for object
    inputs to mirror the original behaviour.
    """
    list_fields = {"imports", "calls"}
    out: dict[str, Any] = {}
    is_dict = isinstance(chunk, dict)
    for name in field_names:
        if is_dict:
            value = chunk.get(name)
        else:
            value = getattr(chunk, name, None)
        if name in list_fields:
            out[name] = value or []
        elif name == "file_path":
            # Object inputs historically str()-coerced file_path; preserve that.
            if is_dict:
                out[name] = value or ""
            else:
                out[name] = str(value or "")
        else:
            out[name] = value or ""
    return out


def _normalise_imports_raw(imports_raw: Any) -> list[Any]:
    """Decode imports if stored as a single JSON-encoded list string.

    Returns the raw imports as a list (possibly the same object).  Falls back
    to ``[]`` when decoding fails or the result isn't a list.  Non-string
    iterables are returned as-is for the caller to iterate.
    """
    if isinstance(imports_raw, str):
        try:
            decoded = _json.loads(imports_raw)
        except (ValueError, TypeError):
            return []
        return decoded if isinstance(decoded, list) else []
    return imports_raw


def _extract_import_source(imp: Any) -> str | None:
    """Return the source name from a single import entry, or None to skip.

    Accepts:
      * dict with a ``source`` key
      * JSON-encoded dict string (legacy format)
      * plain module-name string
    """
    if isinstance(imp, dict):
        src = imp.get("source", "")
        return src or None
    if isinstance(imp, str):
        stripped = imp.strip()
        if stripped.startswith("{"):
            try:
                decoded_imp = _json.loads(stripped)
                src = decoded_imp.get("source", "")
                return src or None
            except (ValueError, TypeError):
                # Fall through to plain-string handling
                pass
        return stripped or None
    return None


def _collect_import_sources(imports_raw: Any) -> list[str]:
    """Walk an imports collection and return a flat list of source names."""
    imports_raw = _normalise_imports_raw(imports_raw)
    try:
        imports_iter = list(imports_raw)
    except TypeError:
        return []

    sources: list[str] = []
    for imp in imports_iter:
        src = _extract_import_source(imp)
        if src:
            sources.append(src)
    return sources


def _truncate_docstring(docstring: str, limit: int = 200) -> str:
    """Strip and truncate a docstring to ``limit`` chars with ``...`` suffix."""
    if not docstring:
        return ""
    doc_summary = docstring.strip()
    if len(doc_summary) > limit:
        doc_summary = doc_summary[:limit].rstrip() + "..."
    return doc_summary


# ---------------------------------------------------------------------------
# build_contextual_text helpers
# ---------------------------------------------------------------------------


def _format_short_file_path(file_path: Any) -> str | None:
    """Return the compact `File: ...` segment, or None to omit."""
    if not file_path:
        return None
    fp_str = str(file_path)
    if fp_str in (".", "/", ""):
        return None
    segments = fp_str.replace("\\", "/").split("/")
    short_path = "/".join(segments[-2:]) if len(segments) > 2 else fp_str
    return f"File: {short_path}"


def _format_language(language: str) -> str | None:
    """Return the `Lang: ...` segment, or None for generic/unknown values."""
    if language and language not in ("text", "unknown", ""):
        return f"Lang: {language}"
    return None


def build_contextual_text(chunk: Any) -> str:
    """Return context-enriched text for embedding.

    The function accepts both ``CodeChunk`` dataclass instances and plain
    ``dict`` objects (as produced by the two-phase pipeline in ``indexer.py``).

    Args:
        chunk: Either a :class:`~mcp_vector_search.core.models.CodeChunk`
            dataclass or a plain ``dict`` produced during the indexing pipeline.
            In either case the following fields are consulted (missing/falsy
            fields are silently skipped):
            ``file_path``, ``language``, ``class_name``, ``function_name``,
            ``imports``, ``docstring``, ``content``.

    Returns:
        A string of the form::

            File: ... | Lang: ... | Class: ... | Fn: ... | Uses: ... | Desc: ...
            ---
            <original content>

        If no metadata is available the original content is returned unchanged.
    """
    fields = _extract_fields(chunk, _CONTEXTUAL_FIELDS)
    content = fields["content"]

    parts: list[str] = []

    file_segment = _format_short_file_path(fields["file_path"])
    if file_segment:
        parts.append(file_segment)

    lang_segment = _format_language(fields["language"])
    if lang_segment:
        parts.append(lang_segment)

    if fields["class_name"]:
        parts.append(f"Class: {fields['class_name']}")

    if fields["function_name"]:
        parts.append(f"Fn: {fields['function_name']}")

    sources = _collect_import_sources(fields["imports"])
    if sources:
        # Cap at 10 sources to keep the header compact
        parts.append(f"Uses: {', '.join(sources[:10])}")

    doc_summary = _truncate_docstring(fields["docstring"])
    if doc_summary:
        parts.append(f"Desc: {doc_summary}")

    if parts:
        header = " | ".join(parts)
        return f"{header}\n---\n{content}"

    return content


# ---------------------------------------------------------------------------
# build_embed_text helpers
# ---------------------------------------------------------------------------


def _format_module_path(file_path: Any) -> str | None:
    """Return the `[module ...]` segment, or None for degenerate paths."""
    if not file_path:
        return None
    fp_str = str(file_path)
    if fp_str in (".", "/", ""):
        return None
    module_norm = fp_str.replace("\\", "/").replace("/", ".")
    if module_norm.endswith(".py"):
        module_norm = module_norm[:-3]
    return f"[module {module_norm}]"


def _collect_calls(calls_raw: Any) -> list[str]:
    """Coerce a calls iterable into a list[str], filtering falsy entries."""
    if not calls_raw:
        return []
    try:
        return [str(c) for c in calls_raw if c]
    except TypeError:
        return []


def build_embed_text(chunk: Any) -> str:
    """Build context-enriched embedding text using bracket-style tags.

    This is the "Step 3a Immediate" contextual chunking implementation from the
    chunking-embedding research doc (2026-02-24).  Compared with
    :func:`build_contextual_text` (which uses a compact pipe-separated header
    targeted at MiniLM's 256-token budget), :func:`build_embed_text` produces a
    richer, multi-line bracket-tagged header that is well-suited to long-context
    code embedding models such as ``nomic-ai/CodeRankEmbed`` (8192-token
    context).

    Format::

        [class ClassName]
        [module path.to.module]
        [imports: a, b, c, d, e]
        <docstring>
        [calls: foo, bar, baz]
        <chunk content>

    Anthropic's contextual retrieval research reports a 35–49% reduction in
    top-20 retrieval failures from prepending class/file/import context to the
    text passed to the embedder.  Adding call-graph context further improves
    cross-reference queries by giving the model a dependency-aware signal.

    Notes on the MiniLM token budget
    --------------------------------
    With ``all-MiniLM-L6-v2`` (256-token limit), the bracket-style header still
    works correctly — longer chunks will simply be truncated.  When the active
    model is upgraded to CodeRankEmbed (or any other 8K+ context model) the
    full context will fit and contribute fully to the embedding.

    Args:
        chunk: Either a :class:`~mcp_vector_search.core.models.CodeChunk`
            dataclass or a plain ``dict``.  The following fields are consulted
            (missing/falsy fields are skipped):
            ``class_name``, ``file_path``, ``imports``, ``docstring``,
            ``calls``, ``content``.

    Returns:
        A multi-line string with bracket-tagged context lines followed by the
        chunk's raw content.  If no metadata is available the original
        ``content`` is returned unchanged.
    """
    fields = _extract_fields(chunk, _EMBED_FIELDS)

    parts: list[str] = []

    # Class context — names the enclosing class for method chunks
    if fields["class_name"]:
        parts.append(f"[class {fields['class_name']}]")

    # Module context — derived from the file path
    module_segment = _format_module_path(fields["file_path"])
    if module_segment:
        parts.append(module_segment)

    # Imports — extract source/module names and cap at top 5 to keep header
    # compact even for files with very long import lists.
    sources = _collect_import_sources(fields["imports"])
    if sources:
        parts.append(f"[imports: {', '.join(sources[:5])}]")

    # Docstring — included verbatim (truncated to 200 chars to bound header
    # size).  The docstring is the highest-signal natural-language description
    # of what the chunk does.
    doc_summary = _truncate_docstring(fields["docstring"])
    if doc_summary:
        parts.append(doc_summary)

    # Call-graph context — names of functions/methods invoked by this chunk.
    # Capped at 5 to keep the header compact.
    calls_iter = _collect_calls(fields["calls"])
    if calls_iter:
        parts.append(f"[calls: {', '.join(calls_iter[:5])}]")

    parts.append(fields["content"])
    return "\n".join(parts)
