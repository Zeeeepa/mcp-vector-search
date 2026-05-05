"""MCP handlers for knowledge graph functionality."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.types import CallToolResult, TextContent

from ..core.embeddings import create_embedding_function
from ..core.factory import create_database
from ..core.knowledge_graph import KnowledgeGraph
from ..core.project import ProjectManager

# Weight for the structural (LCA) signal when combining with vector
# similarity. The LCA score is in [0, 1]; this weight controls how much
# nudge it provides. 0.15 was chosen empirically: large enough to break
# vector-similarity ties between structurally-related entities, small
# enough not to overwhelm a strong vector signal.
LCA_WEIGHT = 0.15


class KGHandlers:
    """MCP handlers for knowledge graph operations.

    Locking strategy (see issue: Kuzu exclusive write lock collisions):

    1. ``self._kg_ro`` is a *single* persistent ``KnowledgeGraph`` instance
       opened in ``read_only=True`` mode. All query handlers share it. Multiple
       in-process readers do not collide because Kuzu's exclusive lock only
       applies to writers.

    2. ``self._kg_build_lock`` is an :class:`asyncio.Lock` that serializes
       writers. The build handler:

       * acquires the lock,
       * closes the read-only singleton (so the writer can take the
         exclusive lock in a subprocess),
       * runs the build in a subprocess (matching the CLI pattern), and
       * reopens the read-only singleton in a ``finally`` block.

    This eliminates the three lock-collision causes diagnosed in the
    investigation: in-process write races, leaked write connections on
    error paths, and long-running write locks blocking queries.
    """

    def __init__(self, project_root: Path):
        """Initialize KG handlers.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self._kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"

        # Lazily-initialized read-only singleton shared by all query handlers.
        # We don't open it eagerly here because the handlers are constructed
        # before the KG necessarily exists on disk — opening a missing DB
        # in read-only mode raises. Use ``_get_ro_kg()`` to obtain it.
        self._kg_ro: KnowledgeGraph | None = None

        # Serializes write operations (kg_build) and the matching
        # close/reopen of the read-only singleton.
        self._kg_build_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Read-only singleton helpers
    # ------------------------------------------------------------------

    def _kg_db_dir_exists(self) -> bool:
        """Return True if the on-disk Kuzu DB directory has been created."""
        return (self._kg_path / "code_kg").exists()

    async def _get_ro_kg(self) -> KnowledgeGraph | None:
        """Return the shared read-only ``KnowledgeGraph`` singleton.

        Returns ``None`` when the underlying Kuzu DB has not yet been
        created on disk (e.g. ``kg_build`` has never run). Callers must
        handle this case and surface a "build the graph first" error.
        """
        if self._kg_ro is not None and self._kg_ro._initialized:
            return self._kg_ro

        if not self._kg_db_dir_exists():
            return None

        kg = KnowledgeGraph(self._kg_path, read_only=True)
        await kg.initialize()
        self._kg_ro = kg
        return self._kg_ro

    async def _close_ro_kg(self) -> None:
        """Close the read-only singleton, if any.

        Used by the build handler to release the read handle so that the
        writing subprocess can acquire Kuzu's exclusive write lock.
        """
        if self._kg_ro is not None:
            try:
                await self._kg_ro.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to close RO KG cleanly: {exc}")
            self._kg_ro = None

    async def aclose(self) -> None:
        """Public cleanup hook for server shutdown."""
        await self._close_ro_kg()

    # ------------------------------------------------------------------
    # kg_build (write path: subprocess + asyncio.Lock + finally)
    # ------------------------------------------------------------------

    async def handle_kg_build(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_build tool call.

        Implementation notes:
            * Serialized by ``self._kg_build_lock`` so concurrent calls
              cannot race for Kuzu's exclusive write lock.
            * The read-only singleton is closed before launching the
              subprocess and reopened in a ``finally`` block, even when
              the build fails.
            * Build runs in an isolated subprocess (mirrors the CLI
              ``mvs kg build`` path) — never opens Kuzu in-process here.

        Args:
            args: Tool arguments containing:
                - force (bool): Force rebuild even if graph exists
                - skip_documents (bool): Skip DOCUMENTS extraction (faster)
                - limit (int | None): Limit chunks to process (for testing)

        Returns:
            CallToolResult with build statistics
        """
        force = bool(args.get("force", False))
        skip_documents = bool(args.get("skip_documents", False))
        limit = args.get("limit")

        async with self._kg_build_lock:
            # Release the read-only handle so the subprocess can take the
            # exclusive write lock. ALWAYS reopen in the finally block.
            await self._close_ro_kg()
            try:
                return await self._run_build_subprocess(
                    force=force,
                    skip_documents=skip_documents,
                    limit=limit,
                )
            finally:
                # Best-effort reopen; if the DB still doesn't exist
                # (e.g. very first build failed before creating it),
                # _get_ro_kg returns None and that's fine.
                try:
                    await self._get_ro_kg()
                except Exception as reopen_exc:  # pragma: no cover - defensive
                    logger.warning(
                        f"Failed to reopen read-only KG after build: {reopen_exc}"
                    )

    async def _run_build_subprocess(
        self, *, force: bool, skip_documents: bool, limit: int | None
    ) -> CallToolResult:
        """Run the KG build in an isolated subprocess (Kuzu-safe).

        Mirrors the pattern used by the CLI ``mvs kg build`` command in
        ``cli/commands/kg.py`` — chunks are dumped to a temp file in this
        process, then ``_kg_subprocess.py`` is invoked as a separate
        Python process which owns the Kuzu write lock for the build's
        full duration.
        """
        import json as _json
        import os as _os
        import shutil
        import subprocess
        import sys
        import tempfile
        from dataclasses import asdict

        # ------------------------------------------------------------------
        # 1. Load chunks via the LanceDB backend (parent process owns LanceDB)
        # ------------------------------------------------------------------
        project_manager = ProjectManager(self.project_root)
        config = project_manager.load_config()

        embedding_function, _ = create_embedding_function(
            model_name=config.embedding_model
        )
        database = create_database(
            persist_directory=config.index_path / "lance",
            embedding_function=embedding_function,
        )
        await database.__aenter__()

        chunks_file_path: str | None = None
        try:
            chunk_count = database.get_chunk_count()
            if chunk_count == 0:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="No index found. Run 'index_project' first.",
                        )
                    ],
                    isError=True,
                )

            # Stream chunks to a temp JSON file (matches CLI _kg_subprocess
            # contract — list of dataclass-style dicts with stringified paths).
            chunks: list[Any] = []
            for batch in database.iter_chunks_batched(batch_size=5000):
                chunks.extend(batch)
                if limit and len(chunks) >= int(limit):
                    chunks = chunks[: int(limit)]
                    break

            chunk_dicts = [asdict(chunk) for chunk in chunks]
            for chunk_dict in chunk_dicts:
                if "file_path" in chunk_dict:
                    chunk_dict["file_path"] = str(chunk_dict["file_path"])

            temp_fd, chunks_file_path = tempfile.mkstemp(
                suffix=".json", prefix="kg_chunks_"
            )
            try:
                with open(chunks_file_path, "w", encoding="utf-8") as f:
                    _json.dump(chunk_dicts, f, default=str)
            finally:
                _os.close(temp_fd)
        finally:
            await database.__aexit__(None, None, None)

        # ------------------------------------------------------------------
        # 2. Pick the same Python interpreter as the CLI uses
        # ------------------------------------------------------------------
        python_executable = sys.executable
        mcp_cmd = shutil.which("mcp-vector-search")
        if mcp_cmd:
            try:
                with open(mcp_cmd) as f:
                    shebang = f.readline().strip()
                if shebang.startswith("#!"):
                    python_executable = shebang[2:].strip()
            except OSError:
                pass  # fall back to sys.executable

        subprocess_script = (
            Path(__file__).parent.parent / "cli" / "commands" / "_kg_subprocess.py"
        )

        cmd = [
            python_executable,
            str(subprocess_script),
            str(self.project_root.absolute()),
            chunks_file_path,
        ]
        if force:
            cmd.append("--force")
        if skip_documents:
            cmd.append("--skip-documents")

        # ------------------------------------------------------------------
        # 3. Run subprocess in an executor so we don't block the event loop
        # ------------------------------------------------------------------
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: subprocess.run(  # nosec B603 - args are fully controlled
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                ),
            )
        except Exception as exc:
            logger.error(f"KG build subprocess invocation failed: {exc}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Knowledge graph build failed: {exc}",
                    )
                ],
                isError=True,
            )
        finally:
            # _kg_subprocess.py also tries to remove this; tolerate races.
            if chunks_file_path:
                try:
                    Path(chunks_file_path).unlink(missing_ok=True)
                except Exception:  # pragma: no cover - defensive
                    pass

        if result.returncode != 0:
            stderr_tail = (result.stderr or "")[-2000:]
            stdout_tail = (result.stdout or "")[-1000:]
            logger.error(
                "KG build subprocess failed (rc=%d): %s",
                result.returncode,
                stderr_tail,
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Knowledge graph build failed (exit code "
                            f"{result.returncode}).\n"
                            f"stderr:\n{stderr_tail}\n"
                            f"stdout:\n{stdout_tail}"
                        ),
                    )
                ],
                isError=True,
            )

        # The subprocess prints a Rich table; surface its tail to the caller.
        output_tail = (result.stdout or "")[-2000:]
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        "Knowledge graph build completed successfully.\n\n"
                        + output_tail
                    ),
                )
            ],
            isError=False,
        )

    # ------------------------------------------------------------------
    # Read-only handlers (use the shared singleton — no per-call open/close)
    # ------------------------------------------------------------------

    async def _kg_required_or_error(
        self,
    ) -> tuple[KnowledgeGraph | None, CallToolResult | None]:
        """Helper: fetch the RO singleton or return a "build first" error."""
        kg = await self._get_ro_kg()
        if kg is None:
            err = CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Knowledge graph has not been built yet. Run 'kg_build' first.",
                    )
                ],
                isError=True,
            )
            return None, err
        return kg, None

    async def handle_kg_stats(self, _args: dict[str, Any]) -> CallToolResult:
        """Handle kg_stats tool call (uses RO singleton)."""
        try:
            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101  # for type checker

            stats = await kg.get_stats()

            result = {
                "status": "success",
                "statistics": {
                    "total_entities": stats["total_entities"],
                    "database_path": stats["database_path"],
                    "relationships": stats.get("relationships", {}),
                },
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG stats failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to get knowledge graph statistics: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_ontology(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_ontology tool call (uses RO singleton)."""
        try:
            category = args.get("category")

            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101

            ontology = await kg.get_document_ontology(category=category)

            result = {
                "status": "success",
                "ontology": ontology,
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG ontology failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to get document ontology: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_ia(self, _args: dict[str, Any]) -> CallToolResult:
        """Handle kg_ia tool call — return IA tree (uses RO singleton)."""
        try:
            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101

            result = await kg.get_ia_tree()

            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2, default=str),
                    )
                ],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG IA tree failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Failed to get Information Architecture tree: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_kg_query(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_query tool call (uses RO singleton).

        Args:
            args: Tool arguments containing:
                - entity (str): Entity name to query, OR a tag query in the
                  form "tag:<name>" / "tags:<name>" for tag-based doc lookup.
                - query_type (str | None): "tag" forces tag-query mode.
                - relationship (str | None): Relationship type filter.
                - limit (int): Max results (default: 20)
        """
        try:
            entity = args.get("entity")
            relationship = args.get("relationship")
            limit = args.get("limit", 20)
            query_type = args.get("query_type", "")

            if not entity:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Missing required parameter: entity",
                        )
                    ],
                    isError=True,
                )

            # --- Detect tag-query mode ---
            tag_names: list[str] = []
            is_tag_query = query_type == "tag"
            if not is_tag_query:
                lower_entity = entity.lower()
                if lower_entity.startswith("tag:") or lower_entity.startswith("tags:"):
                    is_tag_query = True
                    prefix_end = entity.index(":") + 1
                    raw_tags = entity[prefix_end:]
                    tag_names = [t.strip() for t in raw_tags.split(",") if t.strip()]
            if is_tag_query and not tag_names:
                tag_names = [t.strip() for t in entity.split(",") if t.strip()]

            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101

            stats = await kg.get_stats()
            if stats["total_entities"] == 0:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Knowledge graph is empty. Run 'kg_build' first.",
                        )
                    ],
                    isError=True,
                )

            # --- Tag query path ---
            if is_tag_query:
                tag_results = await kg.find_by_tag_docs(tag_names, limit=limit)

                if not tag_results:
                    result = {
                        "status": "success",
                        "query": {"type": "tag", "tags": tag_names},
                        "results": [],
                        "message": f"No documents found with tag(s): {tag_names}",
                    }
                else:
                    result = {
                        "status": "success",
                        "query": {"type": "tag", "tags": tag_names},
                        "results": tag_results,
                        "count": len(tag_results),
                    }

                return CallToolResult(
                    content=[
                        TextContent(type="text", text=json.dumps(result, indent=2))
                    ],
                    isError=False,
                )

            # --- Normal entity query path ---
            results: list[Any] = []

            if relationship in ["calls", "called_by"]:
                calls = await kg.get_call_graph(entity)
                if relationship == "calls":
                    results = [c for c in calls if c["direction"] == "calls"]
                else:
                    results = [c for c in calls if c["direction"] == "called_by"]

            elif relationship in ["inherits", "inherited_by"]:
                hierarchy = await kg.get_inheritance_tree(entity)
                if relationship == "inherits":
                    results = [h for h in hierarchy if h["relation"] == "parent"]
                else:
                    results = [h for h in hierarchy if h["relation"] == "child"]

            elif relationship in ["imports", "imported_by", "contains", "contained_by"]:
                related = await kg.find_related(entity, max_hops=1)
                results = related

            else:
                results = await kg.find_related(entity, max_hops=2)

            # ------------------------------------------------------------
            # Contrastive LCA re-ranking
            # ------------------------------------------------------------
            # We blend the original Kuzu/vector ordering with a structural
            # signal: how close each result is to the query anchor in the
            # CONTAINS hierarchy. This breaks ties between equally-relevant
            # results in favor of the more structurally-related one.
            #
            # If we can't resolve a query anchor (e.g. no entity matches
            # the text query), we skip LCA scoring gracefully — never fail.
            lca_baseline_used: float | None = None
            try:
                anchor_id = await kg.find_entity_by_name(entity)
                if anchor_id and results:
                    scorer = await kg.get_lca_scorer()
                    lca_baseline_used = scorer.baseline
                    result_ids = [
                        r.get("id")
                        for r in results
                        if isinstance(r, dict) and r.get("id")
                    ]
                    lca_scores = scorer.score_query_vs_results(anchor_id, result_ids)

                    # Treat existing per-result "score" (if any) as the
                    # vector-similarity component. Default to 1.0 so the
                    # original ordering is preserved on ties.
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        rid = r.get("id")
                        lca_score = lca_scores.get(rid, 0.0) if rid else 0.0
                        vector_score = float(r.get("score", 1.0) or 0.0)
                        r["lca_score"] = lca_score
                        r["geometric_resonance"] = lca_score
                        r["final_score"] = vector_score + LCA_WEIGHT * lca_score

                    # Stable re-sort: ties preserve Kuzu's original order.
                    results.sort(
                        key=lambda r: (
                            r.get("final_score", 0.0) if isinstance(r, dict) else 0.0
                        ),
                        reverse=True,
                    )
            except Exception as exc:  # pragma: no cover - defensive
                # Never fail the query because of LCA scoring.
                logger.debug(f"LCA re-ranking skipped: {exc}")

            results = results[:limit]

            if not results:
                result = {
                    "status": "success",
                    "query": {"entity": entity, "relationship": relationship},
                    "results": [],
                    "message": f"No related entities found for '{entity}'",
                }
            else:
                result = {
                    "status": "success",
                    "query": {"entity": entity, "relationship": relationship},
                    "results": results,
                    "count": len(results),
                }
                if lca_baseline_used is not None:
                    result["lca_baseline"] = lca_baseline_used

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"KG query failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Knowledge graph query failed: {str(e)}",
                    )
                ],
                isError=True,
            )

    async def handle_trace_execution_flow(self, args: dict[str, Any]) -> CallToolResult:
        """Handle trace_execution_flow tool call (uses RO singleton)."""
        entry_point = args.get("entry_point", "")
        if not entry_point:
            return CallToolResult(
                content=[
                    TextContent(type="text", text="entry_point parameter is required")
                ],
                isError=True,
            )

        depth = int(args.get("depth", 3))
        direction = args.get("direction", "outgoing")
        if direction not in ("outgoing", "incoming", "both"):
            direction = "outgoing"

        try:
            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101

            result = await kg.trace_execution_flow(
                entry_point=entry_point,
                depth=depth,
                direction=direction,
            )
        except Exception as e:
            logger.error(f"trace_execution_flow failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Trace failed: {e}")],
                isError=True,
            )

        if result["entry"] is None:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"No entity found matching '{entry_point}'. Try a more specific name.",
                    )
                ],
                isError=True,
            )

        entry = result["entry"]
        nodes = result["nodes"]
        edges = result["edges"]

        lines = [
            f"## Execution Flow: {entry['name']}",
            f"Entry: {entry['name']} ({entry.get('entity_type', 'function')}) "
            f"[{entry.get('file_path', '').split('/src/')[-1] if entry.get('file_path') else '?'}]",
            f"Direction: {direction} | Depth: {result['depth_reached']}/{depth} | "
            f"Nodes found: {result['total_nodes']}"
            + (" (truncated)" if result["truncated"] else ""),
            "",
        ]

        if not nodes:
            lines.append(
                f"No {'callees' if direction == 'outgoing' else 'callers'} found."
            )
        else:
            lines.append(f"### Reachable nodes ({len(nodes)}):")
            for node in sorted(nodes, key=lambda n: n["depth"]):
                indent = "  " * node["depth"]
                short_file = (node.get("file_path") or "?").split("/src/")[-1]
                lines.append(
                    f"{indent}[depth {node['depth']}] {node['name']} "
                    f"({node.get('entity_type', '?')}) [{short_file}]"
                )

            lines.append("")
            lines.append(f"### Call edges ({len(edges)}):")
            for edge in edges[:30]:
                lines.append(
                    f"  {edge['from_name']} → {edge['to_name']} (depth {edge['depth']})"
                )
            if len(edges) > 30:
                lines.append(f"  ... and {len(edges) - 30} more edges")

        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))],
        )

    async def handle_kg_history(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_history tool call (uses RO singleton)."""
        entity_name = args.get("entity_name", "")
        if not entity_name:
            return CallToolResult(
                content=[TextContent(type="text", text="entity_name is required")],
                isError=True,
            )

        try:
            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101

            history = await kg.get_entity_history(entity_name)

            result = {
                "status": "success",
                "entity_name": entity_name,
                "history": history,
                "note": (
                    "V1: reflects the most recent commit per file at kg_build time, "
                    "not the full git log."
                ),
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"kg_history failed: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"kg_history failed: {e}")],
                isError=True,
            )

    async def handle_kg_callers_at_commit(self, args: dict[str, Any]) -> CallToolResult:
        """Handle kg_callers_at_commit tool call (uses RO singleton)."""
        entity_name = args.get("entity_name", "")
        commit_sha = args.get("commit_sha", "")

        if not entity_name or not commit_sha:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="entity_name and commit_sha are required",
                    )
                ],
                isError=True,
            )

        try:
            kg, err = await self._kg_required_or_error()
            if err is not None:
                return err
            assert kg is not None  # noqa: S101  # nosec B101

            callers = await kg.get_callers_at_commit(
                entity_name, commit_sha, self.project_root
            )

            result = {
                "status": "success",
                "entity_name": entity_name,
                "commit_sha": commit_sha,
                "callers": callers,
                "note": (
                    "V1: reflects the most recent commit per file at kg_build time, "
                    "not the full git log."
                ),
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"kg_callers_at_commit failed: {e}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"kg_callers_at_commit failed: {e}")
                ],
                isError=True,
            )
